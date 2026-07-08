# flake8: noqa: E501


import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    FakeQuantMatmul,
    RMSNorm,
)
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


class Attention(Module):
    def __init__(self, config, layer_idx, use_plugin=False):
        super().__init__()
        self.use_plugin = use_plugin
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_bias = config.attention_bias
        self.w_bits = config.w_bits
        self.has_scale = config.has_scale

        # NOTE: qwen3 bias = False, qwen2 bias = True
        self.q_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.attention_bias,
            w_bits=self.w_bits,
            has_scale=self.has_scale,
        )
        self.k_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
            w_bits=self.w_bits,
            has_scale=self.has_scale,
        )
        # v_proj out is quantized to 8 bits
        self.v_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
            w_bits=self.w_bits,
            has_scale=self.has_scale,
        )

        self.o_proj = DynamicQuantLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            w_bits=self.w_bits,
            has_scale=self.has_scale,
        )
        # unlike olmo, only on the head dim!
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, use_plugin=use_plugin
        )
        # # thus post q_norm does not need reshape
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, use_plugin=use_plugin
        )
        self.k_q_bit = 16
        self.qk_matmul = FakeQuantMatmul(8, self.k_q_bit, None)
        self.wv_matmul = FakeQuantMatmul(16, 8, None)
        self.cache_k_fq = ConstFakeQuant(self.k_q_bit)
        self.cache_v_fq = ConstFakeQuant(8)

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.type.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(query_states, [bsz, -1, self.head_dim])
        query_states = self.q_norm(query_states)
        query_states = leap.reshape(query_states, [bsz, q_len, -1, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])

        key_states = leap.reshape(key_states, [bsz, -1, self.head_dim])
        key_states = self.k_norm(key_states)
        key_states = leap.reshape(key_states, [bsz, q_len, -1, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 1, 3])

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
        key_states = leap.cast_type(key_states, output_type=leap.float32)
        value_states = leap.cast_type(value_states, output_type=leap.float32)

        new_key = key_states
        new_value = value_states

        cache_keys = self.cache_k_fq(cache_keys)
        _, c_len, num_heads, embed_dim = cache_keys.type.shape
        cur_len = key_states.type.shape[2]
        cache_keys = leap.slice(
            cache_keys,
            [0, cur_len, 0, 0],
            [bsz, c_len, num_heads, embed_dim],
            [1, 1, 1, 1],
        )
        cache_keys = leap.transpose(cache_keys, [0, 2, 1, 3])
        key_states = leap.concat([cache_keys, key_states], 2)

        kv_len = key_states.type.shape[2]
        query_states = leap.reshape(
            query_states, [bsz, self.num_key_value_heads, -1, self.head_dim]
        )

        key_states = leap.transpose(key_states, [0, 1, 3, 2])
        query_states = leap.cast_type(query_states, output_type=leap.float32)
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.cast_type(
            attn_weights, output_type=hidden_states.type.element_type
        )
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

        cache_values = self.cache_v_fq(cache_values)
        cache_values = leap.slice(
            cache_values,
            [0, cur_len, 0, 0],
            [bsz, c_len, num_heads, embed_dim],
            [1, 1, 1, 1],
        )
        cache_values = leap.transpose(cache_values, [0, 2, 1, 3])
        value_states = leap.concat([cache_values, value_states], 2)

        attn_output = self.wv_matmul(attn_weights, value_states)
        attn_output = leap.cast_type(
            attn_output, output_type=hidden_states.type.element_type
        )

        attn_output = leap.reshape(attn_output, [bsz, -1, q_len, self.head_dim])
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)

        bs, num_heads, seq_len, _ = new_key.type.shape
        new_key = leap.transpose(new_key, [0, 2, 1, 3])
        new_key = self.cache_k_fq(new_key)
        new_value = leap.transpose(new_value, [0, 2, 1, 3])
        new_value = self.cache_v_fq(new_value)
        return attn_output, new_key, new_value

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.shape

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states = query_states.reshape(-1, q_len, self.head_dim)
        key_states = key_states.reshape(-1, q_len, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        new_key = key_states
        new_value = value_states

        if cache_keys is not None and cache_values is not None:
            cache_keys = self.cache_k_fq(cache_keys)
            cache_values = self.cache_v_fq(cache_values)
            cur_len = key_states.shape[2]
            cache_keys = cache_keys[:, cur_len:].transpose(1, 2)
            key_states = torch.cat([cache_keys, key_states], dim=2)
            cache_values = cache_values[:, cur_len:].transpose(1, 2)
            value_states = torch.cat([cache_values, value_states], dim=2)

        query_states = query_states.reshape(
            bsz, self.num_key_value_heads, -1, self.head_dim
        )

        attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, -1)

        attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attn_weights = torch.add(attn_weights, attention_mask)

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = attn_weights.reshape(
            bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1
        )

        attn_output = self.wv_matmul(attn_weights, value_states)

        attn_output = attn_output.view(bsz, -1, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        new_key = new_key.transpose(1, 2)
        new_value = new_value.transpose(1, 2)
        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)

        return attn_output, new_key, new_value
