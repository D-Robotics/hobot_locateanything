"""MoonViT attention block — packed wqkv + 2D RoPE + global SDPA.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_vit.py:414 (MoonVitEncoderLayer)
    LocateAnything_hyphen_3B/modeling_vit.py:123 (sdpa_attention)

Compile-friendly restrictions applied (report pit #4):
  - Only the SDPA path is exported; flash_attn_varlen is never touched
    during compile because it requires a magi-adjacent CUDA kernel that
    hbdk4 does not know how to lower.
  - Single-image case: cu_seqlens = [0, N] so the block-diagonal mask
    collapses to an all-True mask over the full sequence — no need to
    build the multi-image varlen mask inside the leap DSL.
  - `attention_qkvpacked` becomes the module `forward` (no need for the
    dispatch dict `VL_VISION_ATTENTION_FUNCTIONS`).

State-dict compatibility:
  wqkv.{weight,bias}, wo.{weight,bias} — identical names to upstream
  MoonVitEncoderLayer, so the LocateAnything checkpoint loads without
  a remap step at this layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.rope_2d import apply_rope_real


class MoonViTAttentionStatic(nn.Module):
    """Compile-time attention: packed wqkv + 2D RoPE + SDPA.

    Args to __init__ mirror MoonVitEncoderLayer:
      num_heads:  16 for MoonViT-SO-400M
      hidden_dim: 1152 for MoonViT-SO-400M
      attn_bias:  False by default (matches upstream default)

    Forward:
      x:                 (B, N, hidden_dim)   or (N, hidden_dim)
      freqs_cos_sin:     (N, head_dim/2, 2)   pre-computed 2D RoPE table
                          (broadcasts across B, num_heads via apply_rope_real)
    Returns:
      out: same leading shape as x, hidden_dim channels
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        attn_bias: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (hidden_dim, num_heads)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Same names as upstream so the LocateAnything checkpoint keys
        # (vision_model.encoder.blocks.i.wqkv/wo.{weight,bias}) map here directly.
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Track whether the caller passed a leading batch dim.
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)                             # (1, N, D)

        B, N, D = x.shape
        assert D == self.hidden_dim, (D, self.hidden_dim)

        # Packed qkv: one Linear then split.
        xqkv = self.wqkv(x)                                # (B, N, 3*D)
        xqkv = xqkv.view(B, N, 3, self.num_heads, self.head_dim)
        xq, xk, xv = xqkv.unbind(dim=-3)                   # each (B, N, H, hd)

        # Apply 2D RoPE. freqs_cos_sin is (N, head_dim/2, 2) — apply_rope_real
        # broadcasts over B and num_heads. Expand to (B, N, head_dim/2, 2) so the
        # leading dims match xq/xk minus the num_heads axis.
        if freqs_cos_sin.dim() == 3:
            freqs_bc = freqs_cos_sin.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            freqs_bc = freqs_cos_sin
        xq, xk = apply_rope_real(xq, xk, freqs_bc)         # unchanged shape

        # SDPA expects (B, H, N, hd). We have (B, N, H, hd) — permute.
        xq = xq.transpose(1, 2)                            # (B, H, N, hd)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # MoonViT vision attention is bidirectional (no causal mask). For the
        # single-image compile path, the block-diagonal mask collapses to all-True
        # over the full sequence, which is exactly what SDPA does when
        # attention_mask is None + is_causal=False.
        attn = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, is_causal=False, dropout_p=0.0,
        )                                                  # (B, H, N, hd)

        attn = attn.transpose(1, 2).reshape(B, N, D)       # (B, N, D)
        out = self.wo(attn)                                # (B, N, D)

        if not has_batch:
            out = out.squeeze(0)                           # (N, D)
        return out
