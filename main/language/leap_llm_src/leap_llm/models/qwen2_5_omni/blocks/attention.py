# flake8: noqa: E501

import math
from typing import Optional, Tuple

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
        self.add_key = FakeQuantAdd(quantized=False)

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

        self.q_proj = FakeQuantLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        # v_proj out is quantized to 8 bits
        self.v_proj = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            quant_bits=8,
        )

        self.o_proj = FakeQuantLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        q_quant_bits = 8
        self.qk = FakeQuantMatmul(q_quant_bits, 16)

        v_quant_bits = 8
        self.sv = FakeQuantMatmul(None, v_quant_bits)

        self.mul_attn_weight = FakeQuantMul(quantized=False)

        quantized = True
        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=quantized)

        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )

        self.apply_rotary_pos_emb = RotaryPosEmb(preserve_precision=preserve_precision)

    def build(self, hidden_states, cos, sin, cache_k, cache_v, mask):
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
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

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
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
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

        query_states = query_states.reshape([seqlen, self.num_heads, self.head_dim])
        query_states = query_states.transpose(1, 0)

        key_states = key_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )
        key_states_out = key_states
        key_states = key_states.transpose(1, 0)

        value_states = value_states.reshape(
            [seqlen, self.num_key_value_heads, self.head_dim]
        )
        value_states_out = value_states
        value_states = value_states.transpose(1, 0)

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
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
        attn_output = self.o_proj(attn_output)

        return (
            attn_output,
            key_states_out,
            value_states_out,
        )  # For update cache


class Qwen2_5OmniAudioAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)

        self.qk = FakeQuantMatmul(8, 8, 16)
        self.sv = FakeQuantMatmul(16, 8, 16)

        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=True)
        self.mul_attn_weight = FakeQuantMul(quantized=True)
        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )

    def build(self, hidden_states, attention_mask):
        """Input shape: Batch x Time x Channel"""
        # hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
        seq_length, _ = hidden_states.type.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [seq_length, self.num_heads, self.head_dim]
        )
        key_states = leap.reshape(
            key_states, [seq_length, self.num_heads, self.head_dim]
        )
        value_states = leap.reshape(
            value_states, [seq_length, self.num_heads, self.head_dim]
        )

        query_states = leap.transpose(query_states, [1, 0, 2])
        key_states = leap.transpose(key_states, [1, 0, 2])
        value_states = leap.transpose(value_states, [1, 0, 2])
        key_states = leap.transpose(key_states, [0, 2, 1])

        attn_weights = self.qk(query_states, key_states)

        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        attn_weights = self.add_mask(attn_weights, attention_mask)
        attn_weights = self.softmax(attn_weights)

        attn_output = self.sv(attn_weights, value_states)

        attn_output = leap.transpose(attn_output, [1, 0, 2])

        attn_output = leap.reshape(attn_output, [seq_length, self.embed_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(
            seq_length, self.num_heads, -1
        )

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        key_states = key_states.transpose(1, 2)

        attn_weights = self.qk(query_states, key_states)

        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        attn_weights = self.add_mask(attn_weights, attention_mask)

        attn_weights = self.softmax(attn_weights).to(query_states.dtype)

        attn_output = self.sv(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2_5OmniVisionAttention(Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = FakeQuantLinear(dim, dim, bias=True, quant_bits=8)
        self.k = FakeQuantLinear(dim, dim, bias=True, quant_bits=16)
        self.v = FakeQuantLinear(dim, dim, bias=True, quant_bits=8)
        self.proj = FakeQuantLinear(dim, dim)

        self.apply_rotary_pos_emb = RotaryPosEmb(preserve_precision=False)

        self.qk = FakeQuantMatmul(8, 16, 16)
        self.sv = FakeQuantMatmul(16, 8, 16)

        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=True)

        self.mul_attn_weight = FakeQuantMul(quantized=True)
        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )

    def build(
        self,
        hidden_states,
        window_attention,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        seq_length = hidden_states.type.shape[0]

        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = leap.reshape(q, [seq_length, self.num_heads, -1])
        k = leap.reshape(k, [seq_length, self.num_heads, -1])
        v = leap.reshape(v, [seq_length, self.num_heads, -1])

        q, k = self.apply_rotary_pos_emb(q, k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        q = leap.transpose(q, [1, 0, 2])
        k = leap.transpose(k, [1, 0, 2])
        v = leap.transpose(v, [1, 0, 2])

        if not window_attention:
            k = leap.transpose(k, [0, 2, 1])

        if window_attention:
            # # --- 窗口化自注意力开始 ---
            window_size = 64
            num_windows = seq_length // window_size  # =16
            # 把每个 head 的序列分成 num_windows 个窗口
            # 形状 [num_heads, num_windows, window_size, head_dim]

            q = leap.reshape(
                q, [self.num_heads, num_windows, window_size, self.head_dim]
            )
            k = leap.reshape(
                k, [self.num_heads, num_windows, window_size, self.head_dim]
            )
            v = leap.reshape(
                v, [self.num_heads, num_windows, window_size, self.head_dim]
            )
            k = leap.transpose(k, [0, 1, 3, 2])

        attn_weights = self.qk(q, k)
        attn_weights = self.mul_attn_weight(attn_weights, self.scale)
        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, v)
        if window_attention:
            # 恢复序列维度 [heads, seq, head_dim]
            attn_output = leap.reshape(
                attn_output, [self.num_heads, seq_length, self.head_dim]
            )

        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        window_attention: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor = None,
        rotary_pos_emb_sin: torch.Tensor = None,
    ) -> torch.Tensor:

        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)

        q, k = self.apply_rotary_pos_emb(q, k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        if not window_attention:
            k = k.transpose(1, 2)

        if window_attention:
            # --- 窗口化自注意力开始 ---
            window_size = 64
            num_windows = seq_length // window_size  # =16
            # 把每个 head 的序列分成 num_windows 个窗口
            # 形状 [num_heads, num_windows, window_size, head_dim]
            q = q.view(self.num_heads, num_windows, window_size, self.head_dim)
            k = k.view(self.num_heads, num_windows, window_size, self.head_dim)
            v = v.view(self.num_heads, num_windows, window_size, self.head_dim)
            k = k.transpose(-2, -1)

        attn_weights = self.qk(q, k)
        attn_weights = self.mul_attn_weight(attn_weights, self.scale)

        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, v)

        if window_attention:
            # 恢复序列维度 [heads, seq, head_dim]
            attn_output = attn_output.view(self.num_heads, seq_length, self.head_dim)

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


def apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section=None, unsqueeze_dim=1
):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # mrope_section = mrope_section * 2
    # cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )
    # sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5OmniAttention(Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.rope_scaling = config.rope_scaling

        q_quant_bits = 8
        k_quant_bits = 8

        self.q_proj = FakeQuantLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
            quant_bits=q_quant_bits,
        )
        self.k_proj = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            quant_bits=k_quant_bits,
        )

        v_quant_bits = 8

        self.v_proj = FakeQuantLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            quant_bits=v_quant_bits,
        )
        self.o_proj = FakeQuantLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.apply_rotary_pos_emb = RotaryPosEmb(preserve_precision=False)

        self.qk = FakeQuantMatmul(q_quant_bits, k_quant_bits)
        self.sv = FakeQuantMatmul(None, v_quant_bits)
        self.mul_attn_weight = FakeQuantMul(quantized=False)

        quantized = True
        self.add_mask = FakeQuantAdd(quant_bits=16, quantized=quantized)

        softmax_out_quant_bits = 16
        self.softmax = FakeQuantSoftmax(
            quant_bits=softmax_out_quant_bits, quantized=True
        )
        self.key_out_fq = ConstFakeQuant(k_quant_bits)

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings: Optional[Tuple] = None,
        cache_k=None,
        cache_v=None,
    ):

        bsz, q_len, _ = hidden_states.type.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(query_states, [bsz, q_len, -1, self.head_dim])
        query_states = leap.transpose(query_states, [0, 2, 1, 3])

        key_states = leap.reshape(key_states, [bsz, q_len, -1, self.head_dim])
        key_states = leap.transpose(key_states, [0, 2, 1, 3])

        value_states = leap.reshape(value_states, [bsz, q_len, -1, self.head_dim])
        value_states = leap.transpose(value_states, [0, 2, 1, 3])

        cos, sin = position_embeddings

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        _, _, c_len, _ = cache_k.type.shape

        if cache_k is not None:
            cache_k = leap.slice(
                cache_k,
                [0, 0, q_len, 0],
                [bsz, self.num_key_value_heads, c_len, self.head_dim],
                [1, 1, 1, 1],
            )
            cache_k = leap.concat([cache_k, key_states], -2)
        else:
            cache_k = key_states

        if cache_v is not None:
            cache_v = leap.slice(
                cache_v,
                [0, 0, q_len, 0],
                [bsz, self.num_key_value_heads, c_len, self.head_dim],
                [1, 1, 1, 1],
            )
            cache_v = leap.concat([cache_v, value_states], -2)
        else:
            cache_v = value_states

        key_states = self.key_out_fq(key_states)
        key_states_out = key_states

        key_states = leap.transpose(cache_k, [0, 1, 3, 2])

        query_states = leap.reshape(
            query_states,
            [
                bsz,
                self.num_key_value_heads,
                self.num_key_value_groups * q_len,
                self.head_dim,
            ],
        )
        attn_weights = self.qk(query_states, key_states)

        attn_weights = leap.reshape(attn_weights, [bsz, -1, q_len, c_len])

        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = self.add_mask(attn_weights, attention_mask)

        attn_weights = self.softmax(attn_weights)

        attn_weights = leap.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * q_len, -1],
        )

        attn_output = self.sv(attn_weights, cache_v)
        attn_output = leap.reshape(attn_output, [bsz, -1, q_len, self.head_dim])

        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bsz, q_len, -1])

        attn_output = self.o_proj(attn_output)
        return attn_output, key_states_out, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_k: torch.Tensor = None,
        cache_v: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if cache_k is not None:
            cache_k = cache_k[..., q_len:, :]
            cache_k = torch.cat([cache_k, key_states], -2)
        else:
            cache_k = key_states

        if cache_v is not None:
            cache_v = cache_v[..., q_len:, :]
            cache_v = torch.cat([cache_v, value_states], -2)
        else:
            cache_v = value_states

        key_states = self.key_out_fq(key_states)
        key_states_out = key_states

        key_states = cache_k.transpose(2, 3)

        query_states = query_states.reshape(
            [
                bsz,
                self.num_key_value_heads,
                self.num_key_value_groups * q_len,
                self.head_dim,
            ]
        )

        _, _, _, c_len = attention_mask.shape

        attn_weights = self.qk(query_states, key_states)
        attn_weights = attn_weights.reshape([bsz, -1, q_len, c_len])

        attn_weights = self.mul_attn_weight(
            attn_weights, 1.0 / math.sqrt(self.head_dim)
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = self.add_mask(attn_weights, attention_mask)

        # upcast attention to fp32
        attn_weights = self.softmax(attn_weights)

        attn_weights = torch.reshape(
            attn_weights,
            [self.num_key_value_heads, self.num_key_value_groups * q_len, -1],
        )

        attn_output = self.sv(attn_weights, cache_v)

        attn_output = torch.reshape(attn_output, [bsz, -1, q_len, self.head_dim])

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, key_states_out, value_states
