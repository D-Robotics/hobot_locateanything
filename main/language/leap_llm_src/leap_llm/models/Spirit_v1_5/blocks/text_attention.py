import warnings
from typing import List, Tuple

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.const_fake_quant import ConstFakeQuant
from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.modules.matmul import DynamicQuantMatmul, FakeQuantMatmul
from leap_llm.nn.modules.rms_norm import RMSNorm
from leap_llm.nn.utils import Module


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_leap(x):
    n_local_head, seq_len, head_dim = x.type.shape
    x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
    x2 = leap.slice(
        x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1]
    )
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], 2)
    return rotate_x


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_leap(query_states, key_states, cos, sin):
    q_embed = leap.mul(query_states, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half_leap(query_states), sin))
    k_embed = leap.mul(key_states, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half_leap(key_states), sin))
    return q_embed, k_embed


class Qwen3VLTextAttention(Module):
    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        if config.hidden_size // config.num_attention_heads != config.head_dim:
            warnings.warn(
                f"TextAttention head_dim ({config.head_dim}) is not "
                f"config.hidden_size // config.num_attention_heads "
                f"({config.hidden_size // config.num_attention_heads})"
            )
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.dynamic_matmul = False

        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.k_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.v_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, use_plugin=self.use_plugin
        )
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, use_plugin=self.use_plugin
        )
        if self.dynamic_matmul:
            self.qk_matmul = DynamicQuantMatmul()
            self.sv_matmul = DynamicQuantMatmul()
        else:
            self.q_bit, self.k_bit = 8, 16
            self.s_bit, self.v_bit = 16, 8
            self.qk_matmul = FakeQuantMatmul(self.q_bit, self.k_bit, None)
            self.sv_matmul = FakeQuantMatmul(self.s_bit, self.v_bit, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        """pytorch forward
        arguments:
            hidden_states:  fp16            [batch_size, seq_len, hidden_size]
            cos:            fp16            [batch_size, seq_len, head_dim]
            sin:            fp16            [batch_size, seq_len, head_dim]
            mask:           fp16            [batch_size, seq_len, ctx_len]
        return:
            attn_output:    fp16            [batch_size, seq_len, hidden_size]
        """

        bs, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hs_shape = (*input_shape, -1, self.head_dim)

        # (bs, seq_len, hidden_size) -> (bs, seq_len, num_heads_hat, head_dim)
        # -> (bs, num_heads, seq_len, head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hs_shape)).transpose(
            1, 2
        )

        # (bs, seq_len, hidden_size) -> (bs, seq_len, num_kv_heads, head_dim)
        # -> (bs, num_kv_heads, seq_len, head_dim)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hs_shape)).transpose(
            1, 2
        )

        # (bs, seq_len, hidden_size) -> (bs, seq_len, num_kv_heads, head_dim)
        # -> (bs, num_kv_heads, seq_len, head_dim)
        value_states = self.v_proj(hidden_states).view(hs_shape).transpose(1, 2)

        # (bs, seq_len, head_dim)
        cos, sin = position_embeddings

        query_states = query_states.reshape(-1, seq_len, self.head_dim)
        key_states = key_states.reshape(-1, seq_len, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # (bs*num_heads, seq_len, head_dim)
        # -> (bs, num_kv_head, num_kv_grp*seq_len, head_dim)
        query_states = query_states.reshape(
            bs, self.num_key_value_heads, -1, self.head_dim
        )

        # (bs, num_kv_head, num_kv_grp*seq_len, ctx_len)
        # -> (bs, num_head, seq_len, ctx_len)
        attn_wt = self.qk_matmul(query_states, key_states.transpose(2, 3))
        attn_wt = attn_wt.reshape(bs, self.num_heads, seq_len, -1)
        attn_wt = attn_wt * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=1)
            attn_wt = torch.add(attn_wt, attention_mask)

        attn_wt = torch.nn.functional.softmax(attn_wt, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_wt = attn_wt.reshape(
            bs, self.num_key_value_heads, self.num_key_value_groups * seq_len, -1
        )

        # (bs, num_kv_heads, num_kv_grp*seq_len, ctx_len)
        # * (bs, num_kv_heads, ctx_len, head_dim)
        # -> (bs, num_kv_heads, num_kv_grp*seq_len, head_dim)
        attn_output = self.sv_matmul(attn_wt, value_states)
        attn_output = torch.reshape(attn_output, [bs, -1, seq_len, self.head_dim])
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
    ):
        """Qwen3VLTextAttention leap forward() function

        Args:
            hidden_states (float16) [bs, seq_len, hidden_size]: hidden states
            attention_mask (float16)[bs, seq_len, ctx_len]: causal attention mask
            position_embeddings (_type_): _description_

        Returns:
            _type_: _description_
        """
        bsz, q_len, _ = hidden_states.type.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # q_norm
        # [bs, seq_len, hidden_size] -> [bs, num_head, seq_len, head_dim]
        query_states = leap.reshape(query_states, [bsz, -1, self.head_dim])
        query_states = self.q_norm(query_states)
        query_states = leap.reshape(query_states, [bsz, q_len, -1, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])
        # k_norm
        # [bs, seq_len, hidden_size] -> [bs, num_kv_head, seq_len, head_dim]
        key_states = leap.reshape(key_states, [bsz, -1, self.head_dim])
        key_states = self.k_norm(key_states)
        key_states = leap.reshape(key_states, [bsz, q_len, -1, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 1, 3])
        # value: [bs, num_kv_head, seq_len, head_dim]
        value_states = leap.reshape(value_states, [bsz, q_len, -1, self.head_dim])
        value_states = leap.transpose(value_states, [0, 2, 1, 3])
        cos, sin = position_embeddings
        query_states = leap.reshape(query_states, [-1, q_len, self.head_dim])
        key_states = leap.reshape(key_states, [-1, q_len, self.head_dim])
        # xk, xv
        query_states, key_states = apply_rotary_pos_emb_leap(
            query_states, key_states, cos, sin
        )
        key_states = leap.reshape(key_states, [bsz, -1, q_len, self.head_dim])
        # NOTE: k,v cast to unquant cache dtype
        key_states = leap.cast_type(key_states, output_type=leap.float32)
        value_states = leap.cast_type(value_states, output_type=leap.float32)
        cur_len = key_states.type.shape[2]
        kv_len = key_states.type.shape[2]
        query_states = leap.reshape(
            query_states, [bsz, self.num_key_value_heads, -1, self.head_dim]
        )
        if not self.dynamic_matmul:
            # matmul rsh shall transpose if matmul is fake-quant
            key_states = leap.transpose(key_states, [0, 1, 3, 2])
        query_states = leap.cast_type(query_states, output_type=leap.float32)
        # [bs, num_kv_head, num_kv_grp*seqlen, head_dim]
        # * [bs, num_kv_head, head_dim, ctx_len]
        # -> [bs, num_kv_head, num_kv_grp*seq_len, ctx_len]
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.cast_type(
            attn_weights, output_type=hidden_states.type.element_type
        )
        # [bs, num_head, seq_len, ctx_len]
        attn_weights = leap.reshape(attn_weights, [bsz, self.num_heads, q_len, -1])
        attn_weights = leap.mul(attn_weights, self.scaling)
        if attention_mask is not None:
            attention_mask = leap.reshape(attention_mask, [bsz, 1, q_len, kv_len])
            attn_weights = leap.add(attn_weights, attention_mask)
        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)
        attn_weights = leap.reshape(
            attn_weights,
            [bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1],
        )
        if self.dynamic_matmul:
            value_states = leap.transpose(value_states, [0, 1, 3, 2])
        # [bs, num_kv_head, num_kv_grp*seq_len, ctx_len]
        # * [bs, num_kv_head, ctx_len, head_dim]
        attn_output = self.sv_matmul(attn_weights, value_states)
        attn_output = leap.cast_type(
            attn_output, output_type=hidden_states.type.element_type
        )
        # [bs, num_kv_head, num_kv_grp*seq_len, head_dim]
        # -> [bs, num_head, seq_len, head_dim]
        # -> [bs, seq_len, hidden_size]
        attn_output = leap.reshape(attn_output, [bsz, -1, q_len, self.head_dim])
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output
