import math

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    DynamicQuantMatmul,
    FakeQuantMatmul,
    RMSNorm,
)
from leap_llm.nn.utils import Module


class InternAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = DynamicQuantLinear(
            self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias
        )
        self.proj = DynamicQuantLinear(self.embed_dim, self.embed_dim)

        self.qk_matmul = DynamicQuantMatmul()
        self.wv_matmul = DynamicQuantMatmul()

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.type.shape
        qkv = self.qkv(hidden_states)
        qkv = leap.reshape(qkv, [B, N, 3, self.num_heads, -1])
        qkv = leap.transpose(qkv, [2, 0, 3, 1, 4])

        query_states = leap.select(qkv, 0, 0)
        key_states = leap.select(qkv, 0, 1)
        value_states = leap.select(qkv, 0, 2)

        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        attn_weights = leap.softmax(attn_weights, -1)

        value_states = leap.transpose(value_states, [0, 1, 3, 2])
        attn_output = self.wv_matmul(attn_weights, value_states)
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, (B, N, C))
        attn_output = self.proj(attn_output)
        return attn_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(B, N, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        key_states = key_states.transpose(2, 3)
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        attn_weights = torch.softmax(attn_weights, -1)

        attn_output = self.wv_matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        return attn_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half_leap(x):
    n_local_head, seq_len, head_dim = x.type.shape
    x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
    x2 = leap.slice(
        x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1]
    )
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], 2)
    return rotate_x


def apply_multimodal_rotary_pos_emb_leap(query_states, key_states, cos, sin):
    q_embed = leap.mul(query_states, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half_leap(query_states), sin))
    k_embed = leap.mul(key_states, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half_leap(key_states), sin))
    return q_embed, k_embed


class Qwen3Attention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, num_layer):
        super().__init__()
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            has_scale=config.has_scale,
        )
        self.k_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            has_scale=config.has_scale,
        )
        self.v_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            has_scale=config.has_scale,
        )
        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            has_scale=config.has_scale,
        )
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        if num_layer in [0]:
            print("num layer: ", num_layer, ", use int16 k")
            self.qk_matmul = FakeQuantMatmul(8, 16, None)
            self.cache_k_fq = ConstFakeQuant(16)
        else:
            self.qk_matmul = FakeQuantMatmul(8, 8, None)
            self.cache_k_fq = ConstFakeQuant(8)
        self.wv_matmul = FakeQuantMatmul(8, 8, None)
        self.wv_matmul_int16 = FakeQuantMatmul(16, 8, None)

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
        query_states = leap.reshape(query_states, [bsz, q_len, -1, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])
        query_states = self.q_norm(query_states)

        key_states = self.k_proj(hidden_states)
        key_states = leap.reshape(key_states, [bsz, q_len, -1, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 1, 3])
        key_states = self.k_norm(key_states)

        value_states = self.v_proj(hidden_states)
        value_states = leap.reshape(value_states, (bsz, q_len, -1, self.head_dim))
        value_states = leap.transpose(value_states, [0, 2, 1, 3])

        cos, sin = position_embeddings

        query_states = leap.reshape(query_states, [-1, q_len, self.head_dim])
        key_states = leap.reshape(key_states, [-1, q_len, self.head_dim])

        query_states, key_states = apply_multimodal_rotary_pos_emb_leap(
            query_states, key_states, cos, sin
        )

        key_states = leap.reshape(key_states, [bsz, -1, q_len, self.head_dim])

        key_states = leap.cast_type(key_states, output_type=leap.float32)
        value_states = leap.cast_type(value_states, output_type=leap.float32)

        new_key = key_states
        new_value = value_states

        if cache_keys is not None and cache_values is not None:
            cache_keys = self.cache_k_fq(cache_keys)
            cache_values = self.cache_v_fq(cache_values)
            cur_len = key_states.type.shape[2]
            _, c_len, num_heads, embed_dim = cache_keys.type.shape

            cache_keys = leap.slice(
                cache_keys,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_keys = leap.transpose(cache_keys, [0, 2, 1, 3])
            key_states = leap.concat([cache_keys, key_states], dim=2)

            cache_values = leap.slice(
                cache_values,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_values = leap.transpose(cache_values, [0, 2, 1, 3])
            value_states = leap.concat([cache_values, value_states], dim=2)

        query_states = leap.reshape(
            query_states, (bsz, self.num_key_value_heads, -1, self.head_dim)
        )
        key_states = leap.transpose(key_states, [0, 1, 3, 2])

        query_states = leap.cast_type(query_states, output_type=leap.float32)
        # key_states = leap.cast_type(key_states, output_type=leap.float32)
        # value_states = leap.cast_type(value_states, output_type=leap.float32)

        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.cast_type(
            attn_weights, output_type=hidden_states.type.element_type
        )

        attn_weights = leap.reshape(attn_weights, (bsz, self.num_heads, q_len, -1))
        attn_weights = leap.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = leap.add(attn_weights, attention_mask)
        attn_weights = leap.softmax(attn_weights, -1)

        attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)
        attn_weights = leap.reshape(
            attn_weights,
            (bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1),
        )
        if q_len > 1:
            attn_output = self.wv_matmul(attn_weights, value_states)
        else:
            attn_output = self.wv_matmul_int16(attn_weights, value_states)
        attn_output = leap.cast_type(
            attn_output, output_type=hidden_states.type.element_type
        )

        attn_output = leap.reshape(attn_output, (bsz, -1, q_len, self.head_dim))
        attn_output = leap.transpose(attn_output, (0, 2, 1, 3))
        attn_output = leap.reshape(attn_output, (bsz, q_len, -1))
        attn_output = self.o_proj(attn_output)

        new_key = leap.transpose(new_key, (0, 2, 1, 3))
        new_value = leap.transpose(new_value, (0, 2, 1, 3))
        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)
        return attn_output, new_key, new_value

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
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
        attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)
        attn_weights = torch.softmax(attn_weights, -1).to(query_states.dtype)
        attn_weights_out = attn_weights
        attn_weights = attn_weights.reshape(
            bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1
        )
        attn_output = self.wv_matmul_int16(attn_weights, value_states)
        tmp_output = self.wv_matmul(attn_weights, value_states)
        print("tmp_output shape ", tmp_output.shape)

        attn_output = attn_output.view(bsz, -1, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        new_key = new_key.transpose(1, 2)
        new_value = new_value.transpose(1, 2)
        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)

        return attn_output, attn_weights_out, new_key, new_value
