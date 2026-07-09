"""MoonViT attention — leap DSL version (M3-α).

Vendored & adapted from qwen2_5_vl/blocks/attention.py::Qwen2_5_VLVisionAttention
with these MoonViT-specific simplifications:

  1. Full global attention only — no window/full alternation, no lengths
     splitting, no per-window qk_matmul / wv_matmul lists (Qwen2.5-VL has 16
     of each; we need just one).
  2. Packed wqkv name (not qkv/proj) to match LocateAnything checkpoint keys:
     `vision_model.encoder.blocks.i.wqkv.{weight,bias}`,
     `vision_model.encoder.blocks.i.wo.{weight,bias}`.
  3. head_dim = 1152 / 16 = 72 (Qwen2.5-VL vision is 80).
  4. 2D rope apply — leap DSL version using rotate_half_leap pattern from
     qwen2_5_vl. The (cos, sin) tables are pre-computed 2D-aware
     (interleaves x/y channels) — see utils/rope_2d.py::precompute_freqs_cos_sin.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn
from torch.quantization import DeQuantStub

from leap_llm.nn.modules import (
    DynamicQuantLinear,
    DynamicQuantMatmul,
    FakeQuantMatmul,
)
from leap_llm.nn.utils import Module

try:
    from horizon_plugin_pytorch.quantization import QuantStub
except ImportError:
    QuantStub = None


# ---------------------------------------------------------------------------
# rotate_half — used by rope apply. Same as qwen2_5_vl/blocks/attention.py.
# ---------------------------------------------------------------------------
def rotate_half_leap(x):
    shape = x.type.shape
    if len(shape) == 3:
        n_local_head, seq_len, head_dim = shape
        x1 = leap.slice(x, [0, 0, 0], [n_local_head, seq_len, head_dim // 2], [1, 1, 1])
        x2 = leap.slice(x, [0, 0, head_dim // 2], [n_local_head, seq_len, head_dim], [1, 1, 1])
        x2 = leap.mul(-1, x2)
        return leap.concat([x2, x1], 2)

    bs, n_heads, seq_len, head_dim = shape
    x1 = leap.slice(x, [0, 0, 0, 0], [bs, n_heads, seq_len, head_dim // 2], [1, 1, 1, 1])
    x2 = leap.slice(x, [0, 0, 0, head_dim // 2], [bs, n_heads, seq_len, head_dim], [1, 1, 1, 1])
    x2 = leap.mul(-1, x2)
    return leap.concat([x2, x1], 3)


def rotate_half_torch(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_leap_2d(q, k, cos, sin):
    """Same shape convention as qwen2_5_vl's apply_multimodal_rotary_pos_emb_leap.

    cos/sin are pre-computed 2D-aware tables (interleaved x/y channels in the
    head_dim/2 axis; see utils/rope_2d.py::precompute_freqs_cos_sin).
    """
    q_embed = leap.mul(q, cos)
    q_embed = leap.add(q_embed, leap.mul(rotate_half_leap(q), sin))
    k_embed = leap.mul(k, cos)
    k_embed = leap.add(k_embed, leap.mul(rotate_half_leap(k), sin))
    return q_embed, k_embed


def apply_rope_torch_2d(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half_torch(q) * sin)
    k_embed = (k * cos) + (rotate_half_torch(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# MoonViT attention — global, no window split.
# ---------------------------------------------------------------------------
class LocateAnythingVisionAttention(Module):
    """Global-attention leap DSL vision attention with packed wqkv/wo.

    __init__ args mirror the checkpoint layout:
      dim = 1152, num_heads = 16 (so head_dim = 72)
    """

    def __init__(self, dim: int, num_heads: int = 16, use_plugin: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.use_plugin = use_plugin
        # bias=True — MoonViT checkpoint has wqkv/wo biases (verified during M2 P4).
        if self.use_plugin:
            self.wqkv = nn.Linear(dim, dim * 3, bias=True)
            self.wo = nn.Linear(dim, dim, bias=True)
        else:
            self.wqkv = DynamicQuantLinear(dim, dim * 3, bias=True, w_bits=8)
            self.wo = DynamicQuantLinear(dim, dim, bias=True, w_bits=8)
            self.qk_matmul = DynamicQuantMatmul()
            self.wv_matmul = DynamicQuantMatmul()
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)

    def build(self, hidden_states, rope_cos, rope_sin):
        """
        Args:
          hidden_states : (1, seq_len, dim) — packed image tokens
          rope_cos, rope_sin : (seq_len, head_dim) — pre-computed 2D rope
        """
        seq_length = hidden_states.type.shape[1]

        qkv = self.wqkv(hidden_states)                      # (1, seq, 3*dim)
        qkv = leap.reshape(qkv, [seq_length, 3, self.num_heads, -1])
        qkv = leap.transpose(qkv, [1, 0, 2, 3])             # (3, seq, H, hd)
        q = leap.select(qkv, 0, 0)                          # (seq, H, hd)
        k = leap.select(qkv, 0, 1)
        v = leap.select(qkv, 0, 2)

        # rope apply — head-major layout (H, seq, hd) to match rotate_half_leap 3D branch
        q = leap.transpose(q, [1, 0, 2])                    # (H, seq, hd)
        k = leap.transpose(k, [1, 0, 2])
        q, k = apply_rope_leap_2d(q, k, rope_cos, rope_sin)

        v = leap.transpose(v, [1, 0, 2])                    # (H, seq, hd)

        # Add batch dim for SDPA-style matmul path
        q = leap.reshape(q, [1, self.num_heads, seq_length, -1])
        k = leap.reshape(k, [1, self.num_heads, seq_length, -1])
        v = leap.reshape(v, [1, self.num_heads, seq_length, -1])

        # Full global attention (single block, no windowing).
        k = leap.transpose(k, [0, 1, 3, 2])                 # (1, H, hd, seq)
        attn_weights = self.qk_matmul(q, k)                 # (1, H, seq, seq)
        attn_weights = leap.mul(attn_weights, self.q_mul_value)
        attn_weights = leap.softmax(attn_weights, -1)
        attn_output = self.wv_matmul(attn_weights, v)       # (1, H, seq, hd)

        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])   # (1, seq, H, hd)
        attn_output = leap.reshape(attn_output, [1, seq_length, -1])
        attn_output = self.wo(attn_output)
        return attn_output

    def forward(self, hidden_states, rope_cos, rope_sin):
        """PyTorch reference matching build() semantics — used for calibration."""
        seq_length = hidden_states.shape[1]
        qkv = self.wqkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)                             # (seq, H, hd)

        q = q.transpose(0, 1)                               # (H, seq, hd)
        k = k.transpose(0, 1)
        q, k = apply_rope_torch_2d(q, k, rope_cos, rope_sin)

        v = v.transpose(0, 1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # Global attention (no mask needed — bidirectional)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.q_mul_value
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(1, seq_length, -1)
        return self.wo(attn_output)
