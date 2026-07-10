"""MoonViT encoder block — leap DSL (M3-α).

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

from leap_llm.nn.modules import DynamicQuantLinear, DynamicQuantMatmul, FakeQuantMatmul, LayerNorm
from leap_llm.nn.utils import Module

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
                 use_plugin: bool = False) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        if use_plugin:
            self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
            self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)
        else:
            self.fc0 = DynamicQuantLinear(hidden_dim, mlp_dim, bias=bias, w_bits=8)
            self.fc1 = DynamicQuantLinear(mlp_dim, hidden_dim, bias=bias, w_bits=8)

    def build(self, x):
        x = self.fc0(x)
        try:
            x = leap.gelu(x, approximate="tanh")
        except TypeError:
            x = leap.gelu(x)
        return self.fc1(x)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class LocateAnythingVisionBlock(Module):
    """Single MoonViT encoder layer — attention inlined for state_dict parity.

    State-dict keys mirror upstream MoonVitEncoderLayer:
      norm0.{weight,bias}, norm1.{weight,bias}
      wqkv.{weight,bias}, wo.{weight,bias}
      mlp.fc0.{weight,bias}, mlp.fc1.{weight,bias}
    """

    def __init__(self, config, use_plugin: bool = False) -> None:
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
                config.hidden_size, config.hidden_size * 3, bias=True, w_bits=8,
            )
            self.wo = DynamicQuantLinear(
                config.hidden_size, config.hidden_size, bias=True, w_bits=8,
            )
            self.qk_matmul = FakeQuantMatmul(8, 8, None)
            self.wv_matmul = FakeQuantMatmul(8, 8, None)

        self.mlp = LocateAnythingVisionMLP2(
            config.hidden_size, config.intermediate_size,
            bias=True, use_plugin=use_plugin,
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

        k = leap.transpose(k, [0, 1, 3, 2])                         # (1, H, hd, seq)
        attn_weights = self.qk_matmul(q, k)
        attn_weights = leap.mul(attn_weights, self.q_mul_value)
        attn_weights = leap.softmax(attn_weights, -1)
        attn_output = self.wv_matmul(attn_weights, v)               # (1, H, seq, hd)

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

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.q_mul_value
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
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
