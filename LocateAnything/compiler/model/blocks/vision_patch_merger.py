"""MoonViT patch merger + mlp1 projector to LLM hidden.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_vit.py:538 (patch_merger function)
    LocateAnything_hyphen_3B/modeling_locateanything.py:136 (self.mlp1)

Report pit #10: the LayerNorm inside mlp1 sits at 4×hidden (4608), not
hidden (1152). We must compile the merger and mlp1 as one unit — splitting
them across two hbm files would need a separate 4608-dim intermediate and
buy nothing.

Structure:
  x (N_patches, hidden)
     -> reshape/permute (2×2 merge, no params)
     -> (N_patches/4, 4*hidden = 4608)
     -> LN(4608)
     -> Linear(4608 → 2048)
     -> GELU
     -> Linear(2048 → 2048)
     -> (N_patches/4, llm_hidden = 2048)

State-dict layout mirrors upstream `nn.Sequential`:
  mlp1.0.{weight,bias}    (LayerNorm)
  mlp1.1.{weight,bias}    (Linear 4608 -> llm_hidden)
  mlp1.3.{weight,bias}    (Linear llm_hidden -> llm_hidden)
  # index 2 is nn.GELU (no params)
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn


def patch_merger_2x2_static(
    x: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Single-image variant of upstream `patch_merger` (modeling_vit.py:538).

    Args:
      x: (grid_h * grid_w, d_model) — the flat sequence output by the encoder
      grid_h, grid_w: post-encoder grid (before merger)
      merge_kernel_size: (kh, kw), defaults to (2, 2) for MoonViT-SO-400M

    Returns:
      (grid_h/kh * grid_w/kw, kh*kw * d_model)
      For MoonViT + 448×448 image: (32*32, 1152) -> (16*16, 4608)
    """
    kh, kw = merge_kernel_size
    assert grid_h % kh == 0 and grid_w % kw == 0, (grid_h, grid_w, kh, kw)
    assert x.dim() == 2, x.dim()
    assert x.size(0) == grid_h * grid_w, (x.size(0), grid_h, grid_w)

    d_model = x.size(-1)
    new_h, new_w = grid_h // kh, grid_w // kw

    x = x.view(new_h, kh, new_w, kw, d_model)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(new_h * new_w, kh * kw * d_model)
    return x


class MoonViTPatchMergerAndProjectorStatic(nn.Module):
    """Wraps 2×2 patch merger + mlp1 projector as a single compile-time module.

    The merger itself is parameter-free (pure reshape/permute). Only mlp1
    carries weights, whose state_dict keys map to the LocateAnything checkpoint:
      mlp1.0 -> LayerNorm(4608)
      mlp1.1 -> Linear(4608, llm_hidden)
      mlp1.2 -> GELU  (no params)
      mlp1.3 -> Linear(llm_hidden, llm_hidden)

    Note that the checkpoint stores these under
      state_dict["mlp1.0.weight"], "mlp1.1.weight", "mlp1.3.weight", ...
    at the top-level LocateAnythingForConditionalGeneration scope. Loading
    happens in the wrapper (model.py) that remaps `mlp1.*` -> this module's
    `mlp1.*`.
    """

    def __init__(
        self,
        vit_hidden: int,
        llm_hidden: int,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        grid_h: int = 32,
        grid_w: int = 32,
    ) -> None:
        super().__init__()
        self.vit_hidden = vit_hidden
        self.llm_hidden = llm_hidden
        self.merge_kernel_size = merge_kernel_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        merged_dim = vit_hidden * merge_kernel_size[0] * merge_kernel_size[1]

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(merged_dim),
            nn.Linear(merged_dim, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (grid_h * grid_w, vit_hidden) — flat encoder output

        Returns:
          (grid_h/2 * grid_w/2, llm_hidden) — merged + projected embeddings
        """
        x = patch_merger_2x2_static(
            x, grid_h=self.grid_h, grid_w=self.grid_w,
            merge_kernel_size=self.merge_kernel_size,
        )
        x = self.mlp1(x)
        return x
