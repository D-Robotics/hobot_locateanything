"""MoonViT single encoder block — pre-norm + attention + MLP2 with residuals.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_vit.py:414 (MoonVitEncoderLayer)
    LocateAnything_hyphen_3B/modeling_vit.py:390 (MLP2)

Structure identical to upstream:
  x = x + attention(LN0(x), freqs)
  x = x + mlp(LN1(x))

MLP2 is fc0 -> GELU -> fc1 (no gating; simpler than Qwen2 SwiGLU MLP).

Design note: we do *not* delegate to `MoonViTAttentionStatic` here. Even
though a nested sub-module keeps concerns separated, PyTorch's Module
system re-registers Linear params under the sub-module's prefix, which
creates duplicate state_dict keys and breaks strict `load_state_dict`
against the upstream checkpoint. Instead we inline the packed wqkv +
2D RoPE + SDPA sequence directly here so state_dict layout matches
upstream 1:1 (norm0/norm1/wqkv/wo/mlp.*).

`MoonViTAttentionStatic` is kept as a standalone module for unit testing
the attention core in isolation; not used as a sub-component of Block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.rope_2d import apply_rope_real


class MoonViTMLP2(nn.Module):
    """Two-layer MLP with GELU. Matches upstream MLP2 (modeling_vit.py:390)."""

    def __init__(self, hidden_dim: int, mlp_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MoonViT uses GELU with tanh approximation (upstream feeds
        # `PytorchGELUTanh()` into MLP2). Default `F.gelu` is the erf
        # variant which differs by ~1e-2 on typical activations, so we
        # must match the tanh flavour explicitly.
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class MoonViTBlockStatic(nn.Module):
    """Single MoonViT encoder layer for the compile pipeline.

    State-dict keys (matches upstream MoonVitEncoderLayer exactly):
      norm0.{weight,bias}
      norm1.{weight,bias}
      wqkv.{weight,bias}   (bias only if attn_bias=True)
      wo.{weight,bias}
      mlp.fc0.{weight,bias}
      mlp.fc1.{weight,bias}
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        attn_bias: bool = False,
        mlp_bias: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (hidden_dim, num_heads)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.mlp = MoonViTMLP2(hidden_dim, mlp_dim, bias=mlp_bias)

    def _attention_packed_static(
        self,
        x: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Inlined packed-wqkv + 2D RoPE + SDPA. See vision_attention.py for
        the standalone version (kept for unit testing the attention core)."""
        has_batch = x.dim() == 3
        if not has_batch:
            x = x.unsqueeze(0)

        B, N, D = x.shape
        assert D == self.hidden_dim, (D, self.hidden_dim)

        xqkv = self.wqkv(x)                                    # (B, N, 3*D)
        xqkv = xqkv.view(B, N, 3, self.num_heads, self.head_dim)
        xq, xk, xv = xqkv.unbind(dim=-3)                       # (B, N, H, hd)

        if freqs_cos_sin.dim() == 3:
            freqs_bc = freqs_cos_sin.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            freqs_bc = freqs_cos_sin
        xq, xk = apply_rope_real(xq, xk, freqs_bc)

        # SDPA layout: (B, H, N, hd)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, is_causal=False, dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, N, D)
        out = self.wo(attn)

        if not has_batch:
            out = out.squeeze(0)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.norm0(hidden_states)
        h = self._attention_packed_static(h, freqs_cos_sin)
        hidden_states = residual + h

        residual = hidden_states
        h = self.norm1(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h
        return hidden_states
