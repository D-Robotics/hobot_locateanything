"""MoonViT encoder block implemented with the Leap DSL.

Structure: LN → attention → residual → LN → MLP2 (GELU-tanh) → residual.

Design note: attention is **inlined into the block** rather than delegating
to a separate LocateAnythingVisionAttention sub-module. This keeps the
state_dict layout flat (norm0 / norm1 / wqkv / wo / mlp.*) matching upstream
MoonVitEncoderLayer exactly. The standalone attention class is kept for
unit testing but is not composed here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn

from model.layers import DynamicQuantLinear, DynamicQuantMatmul, LayerNorm
from model.base import Module

from .vision_attention_leap import (
    apply_rope_leap_2d,
    apply_rope_torch_2d,
)


class LocateAnythingVisionMLP2(Module):
    """Two-linear MLP with GELU-tanh. Matches upstream MoonViT MLP2
    (modeling_vit.py:390). State-dict keys `mlp.fc0.*`, `mlp.fc1.*`
    match the checkpoint 1:1.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int, bias: bool = True,
                 use_plugin: bool = False, w_bits: int = 8) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        if use_plugin:
            self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
            self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)
        else:
            self.fc0 = DynamicQuantLinear(
                hidden_dim, mlp_dim, bias=bias, w_bits=w_bits
            )
            self.fc1 = DynamicQuantLinear(
                mlp_dim, hidden_dim, bias=bias, w_bits=w_bits
            )

    def build(self, x):
        x = self.fc0(x)
        try:
            x = leap.gelu(x, approximate="tanh")
        except TypeError:
            x = leap.gelu(x)
        return self.fc1(x)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class LocateAnythingDynamicQuantMatmul(DynamicQuantMatmul):
    """Dynamic matmul with the transposed-RHS contract required by HBDK."""

    transpose_rhs = True

    def forward(self, x, transposed_rhs):
        return torch.matmul(x, transposed_rhs.transpose(-1, -2))


class LocateAnythingCenteredValueWVMatmul(LocateAnythingDynamicQuantMatmul):
    """WV matmul with token-mean-centered V and unshifted attention.

    Centering V keeps its dynamic S8 range compact. Attention remains on the
    compiler's direct dynamic S8 path, so the graph needs no quantized-sum
    compensation term that can amplify target rounding differences.
    """

    def build(self, attention, transposed_rhs):
        value_sum_raw = leap.reduce_sum(
            transposed_rhs,
            dims=[3],
            keepDim=True,
            output_type=leap.float16,
        )
        value_mean = leap.mul(
            value_sum_raw,
            1.0 / transposed_rhs.type.shape[-1],
        )
        centered_value = leap.sub(transposed_rhs, value_mean)

        attention_q, attention_scale = leap.dynamic_quantize(
            attention,
            blockSize=-1,
        )
        value_q, value_scale = leap.dynamic_quantize(
            centered_value,
            blockSize=-1,
        )
        main = leap.block_quantized_matmul(
            attention_q,
            value_q,
            attention_scale,
            value_scale,
            mmaAlpha=1024.0,
        )

        attention_sum = leap.reduce_sum(
            attention,
            dims=[3],
            keepDim=True,
            output_type=leap.float16,
        )
        mean_term = leap.mul(
            attention_sum,
            leap.transpose(value_mean, [0, 1, 3, 2]),
        )
        return leap.add(main, mean_term)


class LocateAnythingVisionBlock(Module):
    """Single MoonViT encoder layer — attention inlined for state_dict parity.

    State-dict keys mirror upstream MoonVitEncoderLayer:
      norm0.{weight,bias}, norm1.{weight,bias}
      wqkv.{weight,bias}, wo.{weight,bias}
      mlp.fc0.{weight,bias}, mlp.fc1.{weight,bias}
    """

    def __init__(self, config, use_plugin: bool = False, w_bits: int = 8) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)

        self.norm0 = LayerNorm(config.hidden_size)
        self.norm1 = LayerNorm(config.hidden_size)

        # Packed QKV + output projection — matches upstream `wqkv` / `wo`.
        if use_plugin:
            self.wqkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
            self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        else:
            self.wqkv = DynamicQuantLinear(
                config.hidden_size, config.hidden_size * 3, bias=True, w_bits=w_bits,
            )
            self.wo = DynamicQuantLinear(
                config.hidden_size, config.hidden_size, bias=True, w_bits=w_bits,
            )
            self.qk_matmul = LocateAnythingDynamicQuantMatmul()
            self.wv_matmul = LocateAnythingCenteredValueWVMatmul()

        self.mlp = LocateAnythingVisionMLP2(
            config.hidden_size, config.intermediate_size,
            bias=True, use_plugin=use_plugin, w_bits=w_bits,
        )

    # ------------------------------------------------------------------
    # Inlined attention — leap DSL
    # ------------------------------------------------------------------
    def _attention_leap(self, hidden_states, rope_cos, rope_sin):
        seq_length = hidden_states.type.shape[1]

        qkv = self.wqkv(hidden_states)                              # (1, seq, 3*dim)
        qkv = leap.reshape(qkv, [seq_length, 3, self.num_heads, -1])
        qkv = leap.transpose(qkv, [1, 0, 2, 3])                     # (3, seq, H, hd)
        q = leap.select(qkv, 0, 0)
        k = leap.select(qkv, 0, 1)
        v = leap.select(qkv, 0, 2)

        # rope apply in (H, seq, hd) layout
        q = leap.transpose(q, [1, 0, 2])
        k = leap.transpose(k, [1, 0, 2])
        q, k = apply_rope_leap_2d(q, k, rope_cos, rope_sin)

        v = leap.transpose(v, [1, 0, 2])
        q = leap.reshape(q, [1, self.num_heads, seq_length, -1])
        k = leap.reshape(k, [1, self.num_heads, seq_length, -1])
        v = leap.reshape(v, [1, self.num_heads, seq_length, -1])

        attn_weights = self.qk_matmul(q, k)
        attn_weights = leap.mul(attn_weights, self.q_mul_value)
        attn_weights = leap.softmax(attn_weights, -1)
        v_transposed = leap.transpose(v, [0, 1, 3, 2])
        attn_output = self.wv_matmul(attn_weights, v_transposed)    # (1, H, seq, hd)

        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [1, seq_length, -1])
        return self.wo(attn_output)

    def _attention_torch(self, hidden_states, rope_cos, rope_sin):
        seq_length = hidden_states.shape[1]
        qkv = self.wqkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)                                     # (seq, H, hd)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        q, k = apply_rope_torch_2d(q, k, rope_cos, rope_sin)

        v = v.transpose(0, 1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        if self.use_plugin:
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
        else:
            attn_weights = self.qk_matmul(q, k)
        attn_weights = attn_weights * self.q_mul_value
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if self.use_plugin:
            attn_output = torch.matmul(attn_weights, v)
        else:
            attn_output = self.wv_matmul(attn_weights, v.transpose(-1, -2))
        attn_output = attn_output.transpose(1, 2).reshape(1, seq_length, -1)
        return self.wo(attn_output)

    def build(self, hidden_states, rope_cos, rope_sin):
        residual = hidden_states
        h = self.norm0(hidden_states)
        h = self._attention_leap(h, rope_cos, rope_sin)
        hidden_states = leap.add(residual, h)

        residual = hidden_states
        h = self.norm1(hidden_states)
        h = self.mlp(h)
        hidden_states = leap.add(residual, h)
        return hidden_states

    def forward(self, hidden_states, rope_cos, rope_sin):
        residual = hidden_states
        h = self.norm0(hidden_states)
        h = self._attention_torch(h, rope_cos, rope_sin)
        hidden_states = residual + h

        residual = hidden_states
        h = self.norm1(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h
        return hidden_states
