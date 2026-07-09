"""Qwen2 Attention (GQA + SDPA) — PyTorch reference for calibration and sanity.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_qwen2.py:219 (Qwen2Attention base)
    LocateAnything_hyphen_3B/modeling_qwen2.py:647 (Qwen2SdpaAttention)
    LocateAnything_hyphen_3B/modeling_qwen2.py:733 (Qwen2SdpaAttentionGqa)

Design decisions (report pits #3, #4):

  - SDPA-only for the compile path. flash_attn / magi_attention are never
    touched here because hbdk4 cannot lower those custom kernels.
  - attention_mask is *always* passed in from the caller (either a causal
    triangular mask for prefill, or a PBD block-diagonal mask for the
    decode branch). We never build the mask inside the module.
  - KV cache is passed in as separate `cache_keys` / `cache_values`
    tensors and returned as `new_keys` / `new_values`, matching the
    Qwen2_5_VLDecoderLayer contract used by leap decode HBMs.

State-dict keys mirror upstream:
  q_proj.{weight,bias}       (bias=True — Qwen2 convention)
  k_proj.{weight,bias}
  v_proj.{weight,bias}
  o_proj.weight              (bias=False)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.rope_1d import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA broadcast: (bs, num_kv, seq, hd) -> (bs, num_kv*n_rep, seq, hd).

    Matches upstream `repeat_kv` (modeling_qwen2.py:207).
    """
    if n_rep == 1:
        return hidden_states
    bs, num_kv, seq, hd = hidden_states.shape
    return (
        hidden_states.unsqueeze(2)
        .expand(bs, num_kv, n_rep, seq, hd)
        .reshape(bs, num_kv * n_rep, seq, hd)
    )


class Qwen2GQAAttentionStatic(nn.Module):
    """Qwen2 GQA attention with SDPA — compile-time PyTorch reference.

    Args:
      hidden_size          : e.g. 2048
      num_heads            : e.g. 16
      num_kv_heads         : e.g. 2   (num_heads must be a multiple of num_kv_heads)

    Forward:
      hidden_states  : (bs, q_len, hidden_size)
      attention_mask : (bs, 1, q_len, kv_len) additive mask; None means no mask
      position_ids   : (bs, q_len) — for 1D rope lookup
      cos, sin       : (max_seq, head_dim) — pre-computed rope tables
      cache_keys     : (bs, past_len, num_kv_heads, head_dim) or None
      cache_values   : (bs, past_len, num_kv_heads, head_dim) or None

    Returns:
      out            : (bs, q_len, hidden_size)
      new_keys       : (bs, kv_len, num_kv_heads, head_dim)   — cache to write back
      new_values     : (bs, kv_len, num_kv_heads, head_dim)

    Convention: cache is stored in (bs, seq, num_kv_heads, head_dim) layout
    to match the leap decode HBM signature used by qwen2_5_vl. Internally we
    transpose to (bs, num_kv_heads, seq, head_dim) for SDPA.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0, (num_heads, num_kv_heads)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, q_len, _ = hidden_states.shape

        # Project Q/K/V.
        q = self.q_proj(hidden_states)                                # (bs, q, num_heads*hd)
        k = self.k_proj(hidden_states)                                # (bs, q, num_kv_heads*hd)
        v = self.v_proj(hidden_states)                                # (bs, q, num_kv_heads*hd)

        # Reshape → (bs, num_heads, q_len, hd)
        q = q.view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rope to the newly-produced q/k (before appending to cache).
        # unsqueeze_dim=1 because q/k here are (bs, heads, seq, hd).
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)

        # Prepend historical KV cache along the seq axis.
        if cache_keys is not None and cache_values is not None:
            # cache layout on disk: (bs, past_len, num_kv_heads, head_dim)
            past_k = cache_keys.transpose(1, 2)                       # (bs, num_kv, past, hd)
            past_v = cache_values.transpose(1, 2)
            full_k = torch.cat([past_k, k], dim=2)                    # (bs, num_kv, past+q, hd)
            full_v = torch.cat([past_v, v], dim=2)
        else:
            full_k = k
            full_v = v

        # Save the *pre-repeat* cache for return (compact layout).
        new_keys = full_k.transpose(1, 2).contiguous()                # (bs, kv_len, num_kv, hd)
        new_values = full_v.transpose(1, 2).contiguous()

        # GQA broadcast for SDPA.
        full_k = repeat_kv(full_k, self.num_kv_groups)
        full_v = repeat_kv(full_v, self.num_kv_groups)

        # SDPA. attention_mask is expected to be (bs, 1, q_len, kv_len) additive.
        attn = F.scaled_dot_product_attention(
            q, full_k, full_v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )                                                             # (bs, num_heads, q, hd)

        attn = attn.transpose(1, 2).contiguous().reshape(bs, q_len, self.hidden_size)
        out = self.o_proj(attn)
        return out, new_keys, new_values
