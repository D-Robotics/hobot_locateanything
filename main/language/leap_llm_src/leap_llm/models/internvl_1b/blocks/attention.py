import math
from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul
from leap_llm.nn.modules.const_fake_quant import ConstFakeQuant
from leap_llm.nn.modules.matmul import FakeQuantMatmul
from leap_llm.nn.utils import Module


def rotate_half(x):
    # [n_local_head, seqlen, head_dim]
    n_local_head, seq_len, head_dim = x.type.shape
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
    x2 = leap.slice(
        x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1]
    )
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], 2)
    return rotate_x


def apply_rotary_pos_emb(query_states, key_states, cos, sin):
    """
    # query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    # key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    """
    q_embed = leap.mul(query_states, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half(query_states), sin))
    k_embed = leap.mul(key_states, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half(key_states), sin))
    return q_embed, k_embed


def rotate_half_torch(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]  # noqa: E203
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torch(query_states, key_states, cos, sin):
    """
    # query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    # key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    """
    q_embed = torch.mul(query_states, cos)
    q_embed = torch.add(q_embed, torch.mul(rotate_half_torch(query_states), sin))
    k_embed = torch.mul(key_states, cos)
    k_embed = torch.add(k_embed, torch.mul(rotate_half_torch(key_states), sin))
    return q_embed, k_embed


class Attention(Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: Optional[int],
        num_key_value_heads: int,
        max_position_embeddings: int,
        w_bits: int,
        has_scale: bool,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.k_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.v_proj = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.o_proj = DynamicQuantLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale,
        )

        self.qk = FakeQuantMatmul(8, 8, None)
        self.sv_int16 = FakeQuantMatmul(16, 8, None)
        self.sv = FakeQuantMatmul(8, 8, None)
        self.cache_k_fq = ConstFakeQuant(8)
        self.cache_v_fq = ConstFakeQuant(8)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        cache_k = self.cache_k_fq(cache_k)
        cache_v = self.cache_v_fq(cache_v)

        seqlen = hidden_states.type.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [seqlen, self.num_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.reshape(
            key_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )
        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.reshape(
            value_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [1, 0, 2])

        # xk, xv
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        _, c_len, _ = cache_k.type.shape

        cache_k = leap.slice(
            cache_k,
            [0, seqlen, 0],
            [self.num_key_value_heads, c_len, self.head_dim],
            [1, 1, 1],
        )
        key_states = leap.cast_type(key_states, output_type=leap.float32)
        cache_k = leap.concat([cache_k, key_states], 1)
        cache_v = leap.slice(
            cache_v,
            [0, seqlen, 0],
            [self.num_key_value_heads, c_len, self.head_dim],
            [1, 1, 1],
        )
        value_states = leap.cast_type(value_states, output_type=leap.float32)
        cache_v = leap.concat([cache_v, value_states], 1)

        if seqlen > 1:
            km = leap.reduce_mean(cache_k, [1])
            cache_k = leap.sub(cache_k, km)
        key_states_t = leap.transpose(cache_k, [0, 2, 1])

        H, W, C = query_states.type.shape

        query_states = leap.reshape(
            query_states,
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ],
        )
        query_states = leap.cast_type(query_states, output_type=leap.float32)
        attn_weights = self.qk(query_states, key_states_t)
        attn_weights = leap.cast_type(attn_weights,
                                      output_type=hidden_states.type.element_type)
        # attn_weights = self.qk(query_states, cache_k)

        attn_weights = leap.reshape(attn_weights, [H, seqlen, c_len])
        attn_weights = leap.mul(attn_weights, self.q_mul_value)

        if mask is not None:
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = leap.add(
                attn_weights, mask
            )  # (n_local_heads, seqlen, cache_len)

        attn_weights = leap.softmax(attn_weights, -1)
        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )

        attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)
        if seqlen > 1:
            attn_output = self.sv(attn_weights, cache_v)
        else:
            attn_output = self.sv_int16(attn_weights, cache_v)
        attn_output = leap.cast_type(attn_output,
                                     output_type=hidden_states.type.element_type)
        # attn_output = self.sv(attn_weights, leap.transpose(cache_v, [0, 2, 1]))

        attn_output = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [seqlen, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        key_states = self.cache_k_fq(key_states)
        value_states = self.cache_v_fq(value_states)
        return attn_output, key_states, value_states  # For update cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        mask: torch.Tensor,
    ):
        cache_k = self.cache_k_fq(cache_k)
        cache_v = self.cache_v_fq(cache_v)

        assert hidden_states.ndim == 3
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([seqlen, self.num_heads, self.head_dim])
        query_states = query_states.transpose(1, 0)

        key_states = key_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )
        key_states = key_states.transpose(1, 0)
        value_states = value_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = value_states.transpose(1, 0)

        # xk, xv
        query_states, key_states = apply_rotary_pos_emb_torch(
            query_states, key_states, cos, sin
        )

        _, c_len, _ = cache_k.shape

        cache_k = cache_k[:, seqlen:, :]
        cache_k = torch.cat([cache_k, key_states], -2)

        cache_v = cache_v[:, seqlen:, :]
        cache_v = torch.cat([cache_v, value_states], -2)

        key_states_t = cache_k.transpose(2, 1)

        H, W, C = query_states.shape
        # === qk before reshape query_states: torch.Size([12, 256, 128])
        query_states = query_states.reshape(
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ]
        )

        attn_weights = self.qk(query_states, key_states_t)

        attn_weights = attn_weights.reshape([H, seqlen, c_len])

        # attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        # NOTE: 会有误差 0.001953125
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if mask is not None:
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = torch.add(
                attn_weights, mask
            )  # (n_local_heads, seqlen, cache_len)

        attn_weights = torch.softmax(attn_weights, -1)

        attn_weights = torch.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )

        attn_output = self.sv(attn_weights, cache_v)
        tmp_output = self.sv_int16(attn_weights, cache_v)
        print("tmp_output shape ", tmp_output.shape)
        attn_output = torch.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = torch.transpose(attn_output, 1, 0)
        attn_output = torch.reshape(attn_output, [batch_size, seqlen, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        key_states = self.cache_k_fq(key_states)
        value_states = self.cache_v_fq(value_states)

        return attn_output, key_states, value_states  # For update cache


class Encoder(Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: Optional[int],
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_attention_heads
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)

        self.wq = DynamicQuantLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.wk = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.wv = DynamicQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.proj = DynamicQuantLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        self.qk = DynamicQuantMatmul()
        self.sv = DynamicQuantMatmul()

    def build(self, hidden_states):
        batch, seqlen, _ = hidden_states.type.shape
        query_states = self.wq(hidden_states)
        key_states = self.wk(hidden_states)
        value_states = self.wv(hidden_states)

        query_states = leap.reshape(
            query_states, [seqlen, self.num_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [1, 0, 2])

        key_states = leap.reshape(
            key_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )

        value_states = leap.reshape(
            value_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )

        # for matmul quantize
        key_states = leap.transpose(key_states, [1, 0, 2])
        # key_states = leap.transpose(key_states, [1, 2, 0])
        attn_weights = self.qk(leap.mul(query_states, self.q_mul_value), key_states)
        att_score = leap.softmax(attn_weights, -1)
        # value_states = leap.transpose(value_states, [1, 0, 2])
        value_states = leap.transpose(value_states, [1, 2, 0])
        attn_output = self.sv(att_score, value_states)
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [batch, seqlen, self.hidden_size])
        return self.proj(attn_output)

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.ndim == 3
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.wq(hidden_states)
        key_states = self.wk(hidden_states)
        value_states = self.wv(hidden_states)

        query_states = query_states.reshape(
            [batch_size, seqlen, self.num_heads, self.head_dim]
        )
        query_states = query_states.transpose(2, 1)

        key_states = key_states.reshape(
            [batch_size, seqlen, self.num_key_value_heads, self.head_dim]
        )
        key_states = key_states.transpose(2, 1)

        value_states = value_states.reshape(
            [batch_size, seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = value_states.transpose(2, 1)

        key_states = key_states.transpose(3, 2)
        attn_weights = self.qk(query_states * self.q_mul_value, key_states)
        attn_weights = torch.softmax(attn_weights, -1)
        attn_output = self.sv(attn_weights, value_states)

        attn_output = torch.transpose(attn_output, 2, 1)
        attn_output = torch.reshape(attn_output, [batch_size, seqlen, self.hidden_size])
        return self.proj(attn_output)
