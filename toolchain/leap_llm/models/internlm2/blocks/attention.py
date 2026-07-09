# flake8: noqa: E501

import math
from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    FakeQuantAdd,
    ConstFakeQuant,
    FakeQuantLinear,
    FakeQuantMatmul,
    FakeQuantMul,
    FakeQuantSoftmax,
)
from leap_llm.nn.utils import Module


class RotateHalf(Module):
    def __init__(self, preserve_precision: bool = False):
        super().__init__()

        quantized = False if preserve_precision else True
        self.mul = FakeQuantMul(quantized=quantized)

        self.in_fq = ConstFakeQuant(16, quantized=quantized)
        self.out_fq = ConstFakeQuant(16, quantized=quantized)

    def build(self, x):
        x = self.in_fq(x)
        if len(x.type.shape) == 4:
            batch_size, n_local_head, seq_len, head_dim = x.type.shape
            x1 = leap.slice(
                x,
                [0, 0, 0, 0],
                [batch_size, n_local_head, seq_len, head_dim // 2],
                [1, 1, 1, 1],
            )
            x2 = leap.slice(
                x,
                [0, 0, 0, head_dim // 2],
                [batch_size, n_local_head, seq_len, head_dim],
                [1, 1, 1, 1],
            )
            x2 = self.mul(-1, x2)
            rotate_x = leap.concat([x2, x1], -1)
        else:
            n_local_head, seq_len, head_dim = x.type.shape
            x1 = leap.slice(
                x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1]
            )
            x2 = leap.slice(
                x,
                [0, 0, head_dim // 2],
                [n_local_head, seq_len, head_dim],
                [1, 1, 1],
            )
            x2 = self.mul(-1, x2)
            rotate_x = leap.concat([x2, x1], 2)
        rotate_x = self.out_fq(rotate_x)
        return rotate_x

    def forward(self, x):
        """Rotates half the hidden dims of the input."""
        x = self.in_fq(x)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]  # noqa: E203
        x2 = self.mul(-1, x2)
        rotate_x = torch.cat((x2, x1), dim=-1)
        rotate_x = self.out_fq(rotate_x)
        return rotate_x


class RotaryPosEmb(Module):
    def __init__(self, preserve_precision: bool = False):
        super().__init__()

        quantized = False if preserve_precision else True

        self.rotate_half_query = RotateHalf(preserve_precision=preserve_precision)
        self.rotate_half_key = RotateHalf(preserve_precision=preserve_precision)

        self.mul_query_cos = FakeQuantMul(quantized=quantized)
        self.mul_rotate_query = FakeQuantMul(quantized=quantized)
        self.add_query = FakeQuantAdd(quantized=quantized)

        self.mul_key_cos = FakeQuantMul(quantized=quantized)
        self.mul_rotate_key = FakeQuantMul(quantized=quantized)

        self.add_key = FakeQuantAdd(quantized=True)

        self.query_states_fq = ConstFakeQuant(16, quantized=quantized)
        self.key_states_fq = ConstFakeQuant(16, quantized=quantized)

    def build(self, query_states, key_states, cos, sin):
        query_states = self.query_states_fq(query_states)
        key_states = self.key_states_fq(key_states)

        q_embed = self.mul_query_cos(query_states, cos)
        rotate_q = self.mul_rotate_query(self.rotate_half_query(query_states), sin)
        q_embed = self.add_query(q_embed, rotate_q)
        k_embed = self.mul_key_cos(key_states, cos)

        rotate_k = self.mul_rotate_key(self.rotate_half_key(key_states), sin)
        k_embed = self.add_key(k_embed, rotate_k)
        return q_embed, k_embed

    def forward(self, query_states, key_states, cos, sin):
        query_states = self.query_states_fq(query_states)
        key_states = self.key_states_fq(key_states)

        q_embed = self.mul_query_cos(query_states, cos)
        rotate_q = self.mul_rotate_query(self.rotate_half_query(query_states), sin)
        q_embed = self.add_query(q_embed, rotate_q)
        k_embed = self.mul_key_cos(key_states, cos)

        rotate_k = self.mul_rotate_key(self.rotate_half_key(key_states), sin)
        k_embed = self.add_key(k_embed, rotate_k)
        return q_embed, k_embed


class Attention(Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: Optional[int],
        num_key_value_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        preserve_precision: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        self.wq = FakeQuantLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.wk = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        v_quant_bits = 8
        self.wv = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            quant_bits=v_quant_bits,
        )

        self.wo = FakeQuantLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        q_quant_bits = 8
        k_quant_bits = 8
        self.qk = FakeQuantMatmul(q_quant_bits, k_quant_bits)
        self.sv = FakeQuantMatmul(None, v_quant_bits)

        self.mul_attn_weight = FakeQuantMul(quantized=False)

        quantized = True
        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=quantized)

        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )

        self.apply_rotary_pos_emb = RotaryPosEmb(preserve_precision=preserve_precision)

        self.key_out_fq = ConstFakeQuant(k_quant_bits)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        seqlen = hidden_states.type.shape[0]
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

        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.reshape(
            value_states, [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [1, 0, 2])

        # xk, xv
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = self.key_out_fq(key_states)

        _, c_len, _ = cache_k.type.shape

        cache_k = leap.slice(
            cache_k,
            [0, seqlen, 0],
            [self.num_key_value_heads, c_len, self.head_dim],
            [1, 1, 1],
        )
        cache_k = leap.concat([cache_k, key_states], 1)
        cache_v = leap.slice(
            cache_v,
            [0, seqlen, 0],
            [self.num_key_value_heads, c_len, self.head_dim],
            [1, 1, 1],
        )
        cache_v = leap.concat([cache_v, value_states], 1)
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

        attn_weights = self.qk(query_states, key_states_t)
        attn_weights = leap.reshape(attn_weights, [H, seqlen, c_len])
        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        if mask is not None:
            attn_weights = self.add_mask(attn_weights, mask)

        attn_weights = self.softmax(attn_weights)

        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )
        attn_output = self.sv(attn_weights, cache_v)

        attn_output = leap.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [seqlen, self.hidden_size])
        attn_output = self.wo(attn_output)
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
        assert hidden_states.ndim == 3
        batch_size, seqlen, _ = hidden_states.shape

        query_states = self.wq(hidden_states)
        key_states = self.wk(hidden_states)
        value_states = self.wv(hidden_states)

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

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = self.key_out_fq(key_states)

        _, c_len, _ = cache_k.shape

        cache_k = cache_k[:, seqlen:, :]
        cache_k = torch.cat([cache_k, key_states], -2)

        cache_v = cache_v[:, seqlen:, :]
        cache_v = torch.cat([cache_v, value_states], -2)

        key_states_t = cache_k.transpose(2, 1)

        H, W, C = query_states.shape
        query_states = query_states.reshape(
            [
                self.num_key_value_heads,
                self.num_key_value_groups * W,
                self.head_dim,
            ]
        )

        attn_weights = self.qk(query_states, key_states_t)
        attn_weights = attn_weights.reshape([H, seqlen, c_len])

        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        if mask is not None:
            attn_weights = self.add_mask(attn_weights, mask)

        attn_weights = self.softmax(attn_weights)

        attn_weights = torch.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
        )

        attn_output = self.sv(attn_weights, cache_v)

        attn_output = torch.reshape(attn_output, [H, seqlen, self.head_dim])
        attn_output = torch.transpose(attn_output, 1, 0)
        attn_output = torch.reshape(attn_output, [batch_size, seqlen, self.hidden_size])
        attn_output = self.wo(attn_output)

        return (
            attn_output,
            key_states,
            value_states,
        )  # For update cache
