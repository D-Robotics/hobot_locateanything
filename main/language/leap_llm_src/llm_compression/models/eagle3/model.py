import json
import math
import os

import torch
from horizon_plugin_pytorch.nn import RMSNorm
from torch import nn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(query_states, key_states, cos, sin):
    q_embed = query_states * cos + rotate_half(query_states) * sin
    k_embed = key_states * cos + rotate_half(key_states) * sin
    return q_embed, k_embed


class Eagle3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_state):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Eagle3Attention(nn.Module):
    """Eagle3 draft model attention. Input dim = hidden_size * 2 (concat of input_emb + hidden_states)."""

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        batch_size, seqlen, _ = hidden_states.shape

        query_states = (
            self.q_proj(hidden_states).view(batch_size, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states).view(batch_size, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(batch_size, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache_k = torch.cat([cache_k[..., seqlen:, :], key_states], dim=-2)
        cache_v = torch.cat([cache_v[..., seqlen:, :], value_states], dim=-2)

        B, H, W, C = query_states.shape
        query_states = query_states.reshape(B, self.num_key_value_heads, self.num_key_value_groups * W, self.head_dim)

        c_len = cache_k.shape[-2]
        attn_weights = torch.matmul(query_states, cache_k.transpose(2, 3))
        attn_weights = attn_weights.reshape(B, H, seqlen, c_len) * self.scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.reshape(B, self.num_key_value_heads, self.num_key_value_groups * seqlen, c_len)
        attn_output = torch.matmul(attn_weights, cache_v)

        attn_output = attn_output.reshape(B, H, seqlen, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, seqlen, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, key_states, value_states


class Eagle3DecoderLayer(nn.Module):
    """Eagle3 draft model decoder layer.

    Concatenates input_emb and hidden_states along the last dimension
    before passing to attention (input dim = hidden_size * 2).
    """

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, intermediate_size, rms_norm_eps):
        super().__init__()
        self.self_attn = Eagle3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        self.mlp = Eagle3MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, input_emb, hidden_states, cos, sin, cache_k, cache_v, mask):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        hidden_states, new_key, new_value = self.self_attn(hidden_states, cos, sin, cache_k, cache_v, mask)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value


class Eagle3LlmModel(nn.Module):
    """Eagle3 draft model: 1-layer decoder that predicts next tokens from
    base model hidden states and token embeddings.

    Architecture:
        fc: Linear(hidden_size * 3, hidden_size) - fuse multi-layer hidden states
        midlayer: single Eagle3DecoderLayer
        norm: RMSNorm
        lm_head: Linear(hidden_size, draft_vocab_size)
        d2t/t2d: vocabulary mapping buffers
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config["hidden_size"] * 3, config["hidden_size"], bias=False)

        self.midlayer = Eagle3DecoderLayer(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            intermediate_size=config["intermediate_size"],
            rms_norm_eps=config["rms_norm_eps"],
        )

        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["draft_vocab_size"], bias=False)

        self.register_buffer("d2t", torch.zeros(config["draft_vocab_size"], dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(config["vocab_size"], dtype=torch.bool))

        head_dim = config["hidden_size"] // config["num_attention_heads"]
        rope_theta = config.get("rope_theta", 1000000.0)
        max_seq_len = config.get("max_position_embeddings", 32768)
        cos, sin = self._compute_rope(max_seq_len, head_dim, rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _compute_rope(max_seq_len, head_dim, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.int64).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, input_embeddings, in_hidden_states, position_ids, attention_mask, caches):
        cache_k, cache_v = caches[0], caches[1]

        cos = self.cos.to(device=position_ids.device, dtype=in_hidden_states.dtype)[position_ids[0]]
        sin = self.sin.to(device=position_ids.device, dtype=in_hidden_states.dtype)[position_ids[0]]
        cos = cos[None]
        sin = sin[None]

        if in_hidden_states.shape[-1] != input_embeddings.shape[-1]:
            in_hidden_states = self.fc(in_hidden_states)

        cache_k = cache_k.transpose(2, 1)
        cache_v = cache_v.transpose(2, 1)

        hidden_states, new_k, new_v = self.midlayer(
            input_embeddings, in_hidden_states, cos, sin, cache_k, cache_v, attention_mask
        )

        new_k = new_k.transpose(2, 1)
        new_v = new_v.transpose(2, 1)

        top_k = self.config.get("top_k", 10)
        out_hidden = hidden_states[:, -top_k:]
        logits = self.lm_head(self.norm(out_hidden))

        return logits, new_k, new_v, out_hidden

    @classmethod
    def from_pretrained(cls, eagle3_model_path, device="cpu", dtype=torch.float32):
        config_path = os.path.join(eagle3_model_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        model = cls(config)

        ckpt_files = [f for f in os.listdir(eagle3_model_path) if f.endswith((".pt", ".bin", ".pth"))]
        if ckpt_files:
            ckpt_path = os.path.join(eagle3_model_path, ckpt_files[0])
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model.to(device=device, dtype=dtype)
        return model
