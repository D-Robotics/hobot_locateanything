# flake8: noqa: E501

import math
from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    FakeQuantAdd,
    FakeQuantLinear,
    FakeQuantMatmul,
    FakeQuantMul,
    FakeQuantSoftmax,
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
    x2 = leap.slice(x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1])
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


class RotateHalf(Module):
    def __init__(self, preserve_precision: bool = False):
        super().__init__()

        quantized = not preserve_precision
        self.mul = FakeQuantMul(quantized=quantized)

    def build(self, x):
        # [n_local_head, seqlen, head_dim]
        n_local_head, seq_len, head_dim = x.type.shape
        x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
        x2 = leap.slice(
            x,
            [0, 0, head_dim // 2],
            [n_local_head, seq_len, head_dim],
            [1, 1, 1],
        )
        x2 = self.mul(-1, x2)
        rotate_x = leap.concat([x2, x1], 2)
        return rotate_x

    def forward(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]  # noqa: E203
        x2 = self.mul(-1, x2)
        return torch.cat((x2, x1), dim=-1)


class RotaryPosEmb(Module):
    def __init__(self, preserve_precision: bool = False):
        super().__init__()

        quantized = not preserve_precision

        self.rotate_half_query = RotateHalf(preserve_precision=preserve_precision)
        self.rotate_half_key = RotateHalf(preserve_precision=preserve_precision)

        self.mul_query_cos = FakeQuantMul(quantized=quantized)
        self.mul_rotate_query = FakeQuantMul(quantized=quantized)
        self.add_query = FakeQuantAdd(quantized=quantized)

        self.mul_key_cos = FakeQuantMul(quantized=quantized)
        self.mul_rotate_key = FakeQuantMul(quantized=quantized)
        # Key need INT out
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
        w_bits: int = 8,
        has_scale: bool = False,
        march: str = "nash-e",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.scaling = self.head_dim**-0.5
        self.march = march
        if "nash-p" in self.march:
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
        else:
            self.q_proj = FakeQuantLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=True,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.k_proj = FakeQuantLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            # v_proj out is quantized to 8 bits
            self.v_proj = FakeQuantLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=True,
                quant_bits=8,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.o_proj = FakeQuantLinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )

        self.q_bit, self.k_bit, self.w_bit, self.v_bit = 8, 16, 16, 8
        self.qk_matmul = FakeQuantMatmul(self.q_bit, self.k_bit, None)
        self.wv_matmul = FakeQuantMatmul(self.w_bit, self.v_bit, None)
        self.cache_k_fq = ConstFakeQuant(self.k_bit)
        self.cache_v_fq = ConstFakeQuant(self.v_bit)

        q_quant_bits = 8
        self.qk = FakeQuantMatmul(q_quant_bits, 16)

        v_quant_bits = 8
        self.sv = FakeQuantMatmul(None, v_quant_bits)

        self.mul_attn_weight = FakeQuantMul(quantized=False)

        quantized = True
        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=quantized)

        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(quant_bits=softmax_out_quant_bits, quantized=True)

        self.apply_rotary_pos_emb = RotaryPosEmb(preserve_precision=preserve_precision)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
        if "nash-p" in self.march:
            bsz, seq_len, _ = hidden_states.type.shape
            ctx_len = mask.type.shape[-1]

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # query reshape
            query_states = leap.reshape(query_states, [bsz, seq_len, self.num_heads, self.head_dim])
            query_states = leap.transpose(query_states, [0, 2, 1, 3])

            key_states = leap.reshape(key_states, [bsz, seq_len, self.num_key_value_heads, self.head_dim])
            key_states = leap.transpose(key_states, [0, 2, 1, 3])

            value_states = leap.reshape(value_states, [bsz, seq_len, self.num_key_value_heads, self.head_dim])
            value_states = leap.transpose(value_states, [0, 2, 1, 3])

            # make q, k broadcastable for RoPE
            query_states = leap.reshape(query_states, [-1, seq_len, self.head_dim])
            key_states = leap.reshape(key_states, [-1, seq_len, self.head_dim])

            query_states, key_states = apply_rotary_pos_emb_leap(query_states, key_states, cos, sin)

            key_states = leap.reshape(key_states, [bsz, self.num_key_value_heads, seq_len, self.head_dim])

            key_states = leap.cast_type(key_states, output_type=leap.float32)
            value_states = leap.cast_type(value_states, output_type=leap.float32)

            new_key = key_states
            new_value = value_states

            # cache key dealing
            cache_keys = self.cache_k_fq(cache_k)
            cache_values = self.cache_v_fq(cache_v)
            _, c_len, num_k_heads, embed_dim = cache_keys.type.shape
            cur_len = key_states.type.shape[2]
            cache_keys = leap.slice(
                cache_keys,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_k_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_keys = leap.transpose(cache_keys, [0, 2, 1, 3])
            key_states = leap.concat([cache_keys, key_states], 2)

            # kv_len = key_states.type.shape[2]

            query_states = leap.reshape(query_states, [bsz, self.num_key_value_heads, -1, self.head_dim])

            # tranpose last two dimensions for qk_matmaul
            key_states = leap.transpose(key_states, [0, 1, 3, 2])

            # cast to fp32 for fakequantMatMul
            query_states = leap.cast_type(query_states, output_type=leap.float32)

            attn_weights = self.qk_matmul(query_states, key_states)

            attn_weights = leap.reshape(attn_weights, [bsz, self.num_heads, seq_len, -1])
            attn_weights = leap.mul(attn_weights, self.scaling)

            attn_weights = leap.cast_type(attn_weights, output_type=hidden_states.type.element_type)

            if mask is not None:
                attention_mask = leap.reshape(mask, [bsz, 1, seq_len, ctx_len])
                attn_weights = leap.add(attn_weights, attention_mask)

            attn_weights = leap.cast_type(attn_weights, output_type=leap.float16)
            attn_weights = leap.softmax(attn_weights, -1)
            attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)
            attn_weights = leap.reshape(
                attn_weights,
                [
                    bsz,
                    self.num_key_value_heads,
                    self.num_key_value_groups * seq_len,
                    -1,
                ],
            )

            cache_values = self.cache_v_fq(cache_values)
            cache_values = leap.slice(
                cache_values,
                [0, cur_len, 0, 0],
                [bsz, c_len, num_k_heads, embed_dim],
                [1, 1, 1, 1],
            )
            cache_values = leap.transpose(cache_values, [0, 2, 1, 3])
            value_states = leap.concat([cache_values, value_states], 2)

            attn_output = self.wv_matmul(attn_weights, value_states)
            # for DQMatmul, cast to hidden_states dtype
            attn_output = leap.cast_type(attn_output, output_type=hidden_states.type.element_type)
            attn_output = leap.reshape(attn_output, [bsz, -1, seq_len, self.head_dim])
            attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
            attn_output = leap.reshape(attn_output, [bsz, seq_len, -1])
            attn_output = self.o_proj(attn_output)

            # output KV cache, take care the fq here.
            new_key = leap.transpose(new_key, [0, 2, 1, 3])
            new_key = self.cache_k_fq(new_key)
            new_value = leap.transpose(new_value, [0, 2, 1, 3])
            new_value = self.cache_v_fq(new_value)
            return attn_output, new_key, new_value
        else:
            seqlen = hidden_states.type.shape[0]
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = leap.reshape(query_states, [seqlen, self.num_heads, self.head_dim])
            query_states = leap.transpose(query_states, [1, 0, 2])
            key_states = leap.reshape(key_states, [seqlen, self.num_key_value_heads, self.head_dim])

            key_states = leap.transpose(key_states, [1, 0, 2])
            value_states = leap.reshape(value_states, [seqlen, self.num_key_value_heads, self.head_dim])
            value_states = leap.transpose(value_states, [1, 0, 2])

            # xk, xv
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
            attn_weights = self.mul_attn_weight(attn_weights, 1.0 / math.sqrt(self.head_dim))

            if mask is not None:
                # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = self.add_mask(attn_weights, mask)

            # NOTE: test
            attn_weights = self.softmax(attn_weights)

            attn_weights = leap.reshape(
                attn_weights,
                [self.num_key_value_heads, self.num_key_value_groups * W, c_len],
            )
            attn_output = self.sv(attn_weights, cache_v)
            attn_output = leap.reshape(attn_output, [H, seqlen, self.head_dim])
            attn_output = leap.transpose(attn_output, [1, 0, 2])
            attn_output = leap.reshape(attn_output, [seqlen, self.hidden_size])
            attn_output = self.o_proj(attn_output)
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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if "nash-p" in self.march:
            # nash-p cache with [bs, ctx_len, num_kv_head, head_dim]
            query_states = query_states.reshape([batch_size, seqlen, self.num_heads, self.head_dim])
            query_states = query_states.transpose(1, 2)

            key_states = key_states.reshape([batch_size, seqlen, self.num_key_value_heads, self.head_dim])
            key_states = key_states.transpose(1, 2)

            value_states = value_states.reshape([batch_size, seqlen, self.num_key_value_heads, self.head_dim])
            value_states = value_states.transpose(1, 2)

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            new_key = key_states
            new_value = value_states

            if True:
                cache_keys = self.cache_k_fq(cache_k)
                cache_values = self.cache_v_fq(cache_v)
                cur_len = key_states.shape[2]

                cache_keys = cache_keys[:, cur_len:].transpose(1, 2)
                # (bs, num_kv_heads, ctx_len, head_dim)
                key_states = torch.cat([cache_keys, key_states], dim=2)
                cache_values = cache_values[:, cur_len:].transpose(1, 2)
                # (bs, num_kv_heads, ctx_len, head_dim)
                value_states = torch.cat([cache_values, value_states], dim=2)

            query_states = query_states.reshape(batch_size, self.num_key_value_heads, -1, self.head_dim)
            # (bs, num_kv_head, num_kv_grp*seq_len, head_dim)
            # * (bs, num_kv_heads, ctx_len, head_dim).tsp(2,3)
            # -> (bs, num_kv_heads, num_kv_grp*seq_len, ctx_len)
            attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3))
            attn_weights = attn_weights.reshape(batch_size, self.num_heads, seqlen, -1)
            attn_weights = attn_weights * self.scaling

            if mask is not None:
                mask = mask.unsqueeze(1)
                attn_weights = torch.add(attn_weights, mask)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            attn_weights = attn_weights.reshape(
                batch_size,
                self.num_key_value_heads,
                self.num_key_value_groups * seqlen,
                -1,
            )  # (bs, num_kv_heads, num_kv_grp*seq, ctx_len)

            # (bs, num_kv_heads, num_kv_grp*seq, ctx_len)
            # * (bs, num_kv_heads, ctx_len, head_dim)
            # -> (bs, num_kv_head, num_kv_grp*seq_len, head_dim)
            attn_output = self.wv_matmul(attn_weights, value_states)
            attn_output = torch.reshape(attn_output, [batch_size, self.num_heads, seqlen, self.head_dim])
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seqlen, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            new_key = new_key.transpose(1, 2)
            new_key = self.cache_k_fq(new_key)
            new_value = new_value.transpose(1, 2)
            new_value = self.cache_v_fq(new_value)

            return attn_output, new_key, new_value
        else:
            query_states = query_states.reshape([seqlen, self.num_heads, self.head_dim])
            query_states = query_states.transpose(1, 0)

            key_states = key_states.reshape([seqlen, self.num_key_value_heads, self.head_dim])

            key_states = key_states.transpose(1, 0)

            value_states = value_states.reshape([seqlen, self.num_key_value_heads, self.head_dim])
            value_states = value_states.transpose(1, 0)

            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

            attn_weights = self.mul_attn_weight(attn_weights, 1.0 / math.sqrt(self.head_dim))

            if mask is not None:
                # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
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
            attn_output = self.o_proj(attn_output)

            return (
                attn_output,
                key_states,
                value_states,
            )  # For update cache
