"""Self-attention (VLM prefix) and cross-attention (action expert) for SmolVLA."""

import torch
from hbdk4.compiler import leap

from leap_llm.models.pi0.blocks.attention import GemmaAttention, RotaryPosEmb
from leap_llm.models.smolvla.blocks.configuration_smolvlm import SmolLM2Config
from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul
from leap_llm.nn.utils import Module

__all__ = [
    "SmolLM2Attention",
    "CrossExpertAttention",
    "SmolVLAExpertSelfAttention",
    "RotaryPosEmb",
]


class SmolLM2Attention(GemmaAttention):
    """Same graph as Gemma attention; SmolLM2 uses compatible QKV layout."""

    pass


# ---------------------------------------------------------------------------
# SmolVLAExpertSelfAttention
# Pure self-attention for expert even-index layers (action → action, no prefix).
# Weight shapes: q_proj[nq*d, hidden], k_proj[nkv*d, hidden], v_proj[nkv*d, hidden]
# ---------------------------------------------------------------------------

class SmolVLAExpertSelfAttention(Module):
    """
    LeRobot-style pure self-attention for expert self-attn layers.

    Q, K, V all come from the action token hidden states.  The VLM prefix KV
    cache is NOT used.  The caller is responsible for passing only the
    suffix part of the attention mask (last action_len columns).
    """

    def __init__(self, config: SmolLM2Config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.hidden_size = config.hidden_size

        nq_dim = config.num_attention_heads * self.head_dim
        nkv_dim = config.num_key_value_heads * self.head_dim

        self.q_proj = DynamicQuantLinear(config.hidden_size, nq_dim, bias=config.attention_bias)
        self.k_proj = DynamicQuantLinear(config.hidden_size, nkv_dim, bias=config.attention_bias)
        self.v_proj = DynamicQuantLinear(config.hidden_size, nkv_dim, bias=config.attention_bias)
        self.o_proj = DynamicQuantLinear(nq_dim, config.hidden_size, bias=config.attention_bias)
        self.apply_rotary_pos_emb = RotaryPosEmb()
        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()

    # ------------------------------------------------------------------
    # LEAP (BPU compile) path
    # ------------------------------------------------------------------
    def build(self, hidden_states, attention_mask, cos, sin, cache_k=None, cache_v=None):
        seqlen = hidden_states.type.shape[1]

        query_states = self.q_proj(hidden_states)  # [1, L, nq*d]
        key_states   = self.k_proj(hidden_states)  # [1, L, nkv*d]
        value_states = self.v_proj(hidden_states)  # [1, L, nkv*d]

        query_states = leap.reshape(query_states, [seqlen, self.num_attention_heads, self.head_dim])
        query_states = leap.transpose(query_states, [1, 0, 2])   # [nq, L, d]
        key_states   = leap.reshape(key_states,   [seqlen, self.num_key_value_heads, self.head_dim])
        key_states   = leap.transpose(key_states,   [1, 0, 2])   # [nkv, L, d]
        value_states = leap.reshape(value_states, [seqlen, self.num_key_value_heads, self.head_dim])
        value_states = leap.transpose(value_states, [1, 0, 2])   # [nkv, L, d]

        query_states, key_states = self.apply_rotary_pos_emb.build(
            query_states, key_states, cos, sin
        )

        if cache_k is not None:
            prefix_len = cache_k.type.shape[1]
            key_states = leap.concat([cache_k, key_states], dim=1)
            value_states = leap.concat([cache_v, value_states], dim=1)
            total_len = prefix_len + seqlen
        else:
            total_len = seqlen

        H, W, _ = query_states.type.shape  # [nq, L, d]
        query_states = leap.reshape(
            query_states,
            [self.num_key_value_heads, self.num_key_value_groups * W, self.head_dim],
        )
        attn_weights = self.qk(query_states, key_states)          # [nkv, groups*L, total_len]
        attn_weights = leap.reshape(attn_weights, [H, seqlen, total_len])
        attn_weights = leap.mul(attn_weights, self.scaling)

        if attention_mask is not None:
            attn_weights = leap.add(attn_weights, attention_mask)

        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, total_len],
        )
        value_states = leap.transpose(value_states, [0, 2, 1])    # [nkv, d, total_len]
        attn_output  = self.sv(attn_weights, value_states)         # [nkv, groups*L, d]
        attn_output  = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output  = leap.transpose(attn_output, [1, 0, 2])     # [L, nq, d]
        attn_output  = leap.reshape(attn_output, [seqlen, self.num_attention_heads * self.head_dim])
        return self.o_proj(attn_output), attn_weights

    def forward(self, hidden_states, attention_mask, cos, sin, cache_k=None, cache_v=None):
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)  # [B, L, nq*d]
        key_states   = self.k_proj(hidden_states)  # [B, L, nkv*d]
        value_states = self.v_proj(hidden_states)  # [B, L, nkv*d]

        query_states = query_states.reshape(
            batch_size, seqlen, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # [B, nq, L, d]
        key_states = key_states.reshape(
            batch_size, seqlen, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [B, nkv, L, d]
        value_states = value_states.reshape(
            batch_size, seqlen, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [B, nkv, L, d]

        cos_b = cos.unsqueeze(1) if cos.ndim == 3 else cos
        sin_b = sin.unsqueeze(1) if sin.ndim == 3 else sin
        query_states, key_states = self.apply_rotary_pos_emb.forward(
            query_states, key_states, cos_b, sin_b
        )

        if cache_k is not None:
            pk = cache_k.unsqueeze(0)  # [1, nkv, P, d]
            pv = cache_v.unsqueeze(0)
            key_states = torch.cat([pk, key_states], dim=2)
            value_states = torch.cat([pv, value_states], dim=2)

        key_states   = key_states.repeat_interleave(self.num_key_value_groups, dim=1)    # [B, nq, T, d]
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, nq, T, d]

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling  # [B, nq, L, T]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output  = torch.matmul(attn_weights, value_states)  # [B, nq, L, d]
        nq_out_dim   = self.num_attention_heads * self.head_dim
        attn_output  = attn_output.transpose(1, 2).reshape(batch_size, seqlen, nq_out_dim)
        return self.o_proj(attn_output), attn_weights


# ---------------------------------------------------------------------------
# CrossExpertAttention
# Expert queries attend to VLM prefix KV cache.
# Weight shapes: k_proj[nkv*d, vlm_kv_flat], v_proj[nkv*d, vlm_kv_flat]
# ---------------------------------------------------------------------------

class CrossExpertAttention(Module):
    """Expert queries attend to prefix KV (with k/v projection from VLM cache)."""

    def __init__(self, config: SmolLM2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.vlm_kv_flat = config.vlm_kv_in_dim
        expert_kv_dim = config.num_key_value_heads * config.head_dim
        self.k_proj = DynamicQuantLinear(
            self.vlm_kv_flat, expert_kv_dim, bias=config.attention_bias
        )
        self.v_proj = DynamicQuantLinear(
            self.vlm_kv_flat, expert_kv_dim, bias=config.attention_bias
        )
        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()
        self.apply_rotary_pos_emb = RotaryPosEmb()

    # ------------------------------------------------------------------
    # LEAP (BPU compile) path
    # ------------------------------------------------------------------
    def build(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        seqlen    = hidden_states.type.shape[1]
        prefix_len = cache_k.type.shape[1]

        query_states = self.q_proj(hidden_states)
        query_states = leap.reshape(
            query_states, [seqlen, self.num_attention_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        # Only apply RoPE to query (no RoPE on KV coming from frozen VLM cache)
        query_states, _ = self.apply_rotary_pos_emb.build(query_states, query_states, cos, sin)

        # cache_k/v from VLM: [nkv, prefix_len, head_dim] → transpose → [prefix_len, nkv, head_dim] → reshape
        cache_k = leap.transpose(cache_k, [1, 0, 2])
        cache_v = leap.transpose(cache_v, [1, 0, 2])
        cache_k_flat = leap.reshape(cache_k, [prefix_len, self.vlm_kv_flat])
        cache_v_flat = leap.reshape(cache_v, [prefix_len, self.vlm_kv_flat])
        key_states   = self.k_proj(cache_k_flat)
        value_states = self.v_proj(cache_v_flat)
        key_states   = leap.reshape(key_states,   [prefix_len, self.num_key_value_heads, self.head_dim])
        key_states   = leap.transpose(key_states,   [1, 0, 2])  # [nkv, P, d]
        value_states = leap.reshape(value_states, [prefix_len, self.num_key_value_heads, self.head_dim])
        value_states = leap.transpose(value_states, [1, 0, 2])  # [nkv, P, d]

        H, W, _ = query_states.type.shape   # [nq, L, d]
        query_states = leap.reshape(
            query_states,
            [self.num_key_value_heads, self.num_key_value_groups * W, self.head_dim],
        )
        attn_weights = self.qk(query_states, key_states)         # [nkv, groups*L, P]
        attn_weights = leap.reshape(attn_weights, [H, seqlen, prefix_len])
        attn_weights = leap.mul(attn_weights, self.scaling)

        if attention_mask is not None:
            # Take first prefix_len columns from the full [1,1,L,prefix+L] mask
            if attention_mask.type.shape[-1] > prefix_len:
                attention_mask = leap.slice(
                    attention_mask,
                    [0, 0, 0, 0],
                    [1, 1, seqlen, prefix_len],
                    [1, 1, 1, 1],
                )
            attn_weights = leap.add(attn_weights, attention_mask)

        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, prefix_len],
        )
        value_states = leap.transpose(value_states, [0, 2, 1])   # [nkv, d, P]
        attn_output  = self.sv(attn_weights, value_states)        # [nkv, groups*L, d]
        attn_output  = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output  = leap.transpose(attn_output, [1, 0, 2])
        attn_output  = leap.reshape(attn_output, [seqlen, self.num_attention_heads * self.head_dim])
        return self.o_proj(attn_output), attn_weights

    # ------------------------------------------------------------------
    # PyTorch (CPU calibration / forward) path
    # ------------------------------------------------------------------
    def forward(self, hidden_states, attention_mask, cache_k, cache_v, cos, sin):
        batch_size, seqlen, _ = hidden_states.shape
        # cache_k: [nkv, prefix_len, head_dim] (from VLM prefix output)
        prefix_len = cache_k.shape[1]

        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(
            batch_size, seqlen, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # [B, nq, L, d]

        # RoPE on query only
        cos_b = cos.unsqueeze(1) if cos.ndim == 3 else cos
        sin_b = sin.unsqueeze(1) if sin.ndim == 3 else sin
        query_states, _ = self.apply_rotary_pos_emb.forward(
            query_states, query_states, cos_b, sin_b
        )

        # Project VLM prefix KV: cache_k [nkv_vlm, P, d] (3D, no batch from GemmaAttention.forward)
        cache_k_flat = cache_k.permute(1, 0, 2).reshape(prefix_len, -1)  # [P, nkv_vlm*d]
        cache_v_flat = cache_v.permute(1, 0, 2).reshape(prefix_len, -1)  # [P, nkv_vlm*d]
        key_states   = self.k_proj(cache_k_flat)   # [P, nkv_expert*d]
        value_states = self.v_proj(cache_v_flat)

        # add batch dim → [1, nkv_expert, P, d]
        key_states = key_states.reshape(
            prefix_len, self.num_key_value_heads, self.head_dim
        ).unsqueeze(0).transpose(1, 2)
        value_states = value_states.reshape(
            prefix_len, self.num_key_value_heads, self.head_dim
        ).unsqueeze(0).transpose(1, 2)

        # GQA: expand nkv → nq heads
        key_states   = key_states.repeat_interleave(self.num_key_value_groups, dim=1)    # [B, nq, P, d]
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)  # [B, nq, P, d]

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling  # [B, nq, L, P]
        if attention_mask is not None:
            # Only first prefix_len columns of the combined mask
            mask_prefix = attention_mask[..., :prefix_len]  # [B, 1, L, P]
            attn_weights = attn_weights + mask_prefix

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output  = torch.matmul(attn_weights, value_states)  # [B, nq, L, d]
        nq_out_dim   = self.num_attention_heads * self.head_dim
        attn_output  = attn_output.transpose(1, 2).reshape(batch_size, seqlen, nq_out_dim)
        return self.o_proj(attn_output), attn_weights
