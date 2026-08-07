"""MoonViT patch embedding — Conv2d + 2D interpolatable pos-emb.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_vit.py:224 (Learnable2DInterpPosEmb)
    LocateAnything_hyphen_3B/modeling_vit.py:258 (MoonVisionPatchEmbed)

Design notes for the compile-friendly version:

1. Upstream `Learnable2DInterpPosEmb.forward` runs `F.interpolate(mode="bicubic")`
   on every image at inference time. bicubic with dynamic target size is not
   trace-friendly for the leap DSL (report pit #9). We therefore split the
   module into a *training-time* module (identical to upstream, so state_dict
   loads cleanly) and a *compile-time* module that consumes a pre-interpolated
   pos-emb table.

2. Upstream forward takes "packed L" input `(L, C, patch_h, patch_w)` where L
   is a concatenation of multiple images. Our compile path always processes
   a single fixed-resolution image at a time, so the compile-time version
   drops the grid_hws parameter and works on `(N_patches, C, patch_h, patch_w)`.

3. State-dict keys mirror upstream so we can load the LocateAnything checkpoint
   without a remap step. Only difference: pos_emb.weight is (H, W, dim) in
   upstream — we keep the same shape but pre-interpolate on load.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Training-time Learnable2DInterpPosEmb — byte-for-byte compatible with
# upstream. Used only for loading the LocateAnything checkpoint into memory.
# ---------------------------------------------------------------------------
class Learnable2DInterpPosEmb(nn.Module):
    """Upstream-compatible learnable 2D pos-emb with bicubic interpolation.

    Kept identical to modeling_vit.py:224 so that the .weight parameter can
    be loaded from the LocateAnything checkpoint without renaming.
    """

    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == list(self.weight.shape[:-1]):
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute(2, 0, 1).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )
        return x + torch.cat(pos_embs)

    @torch.no_grad()
    def precompute_for_grid(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Return the pre-interpolated (grid_h*grid_w, dim) pos-emb tensor.

        Called once at compile time to freeze the bicubic interpolation into
        a constant tensor that the compile-time module can consume directly.
        """
        if [grid_h, grid_w] == [self.height, self.width]:
            return self.weight.flatten(end_dim=1).detach().clone()
        interp = F.interpolate(
            self.weight.permute(2, 0, 1).unsqueeze(0),
            size=(grid_h, grid_w),
            mode=self.interpolation_mode,
        )
        return interp.squeeze(0).permute(1, 2, 0).flatten(end_dim=1).detach().clone()


# ---------------------------------------------------------------------------
# Training-time MoonVisionPatchEmbed — upstream-compatible. Also the class
# used when instantiating the reference model for calibration forward passes.
# ---------------------------------------------------------------------------
class MoonVisionPatchEmbed(nn.Module):
    """Upstream MoonVisionPatchEmbed (modeling_vit.py:258)."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Learnable2DInterpPosEmb(
            height=pos_emb_height, width=pos_emb_width, dim=out_dim,
        )

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_hws)
        return x


# ---------------------------------------------------------------------------
# Compile-time MoonVisionPatchEmbedStatic
#
# Single fixed image resolution → no grid_hws, no dynamic interpolate.
# The pos-emb is stored as a *buffer* (constant) pre-interpolated on load.
# ---------------------------------------------------------------------------
class MoonVisionPatchEmbedStatic(nn.Module):
    """Compile-friendly patch embed for a fixed (image_h, image_w).

    Semantics equivalent to upstream `MoonVisionPatchEmbed` when its
    input is a single image with `grid_hws == [[image_h//patch, image_w//patch]]`,
    but with all dynamic-shape ops replaced by constants:

      - grid_hws → static (grid_h, grid_w) known at build() time
      - bicubic interpolation → executed once during load_pretrained(), the
        result stored as a non-persistent buffer `pos_emb_static`
      - packed L input → simple (n_patches, ...) input; batch is handled by
        the caller because the leap DSL expects a fixed leading dim
    """

    def __init__(
        self,
        out_dim: int,
        image_h: int,
        image_w: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert image_h % patch_size[0] == 0, (image_h, patch_size)
        assert image_w % patch_size[1] == 0, (image_w, patch_size)
        self.patch_size = patch_size
        self.image_h = image_h
        self.image_w = image_w
        self.grid_h = image_h // patch_size[0]
        self.grid_w = image_w // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w
        self.out_dim = out_dim

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        # Pre-interpolated pos-emb — populated by load_from_learnable().
        # Shape: (num_patches, out_dim). Registered as a buffer so torch.save
        # picks it up but state_dict of the reference model does not overwrite.
        self.register_buffer(
            "pos_emb_static",
            torch.zeros(self.num_patches, out_dim),
            persistent=False,
        )

    @torch.no_grad()
    def load_from_learnable(self, learnable: Learnable2DInterpPosEmb) -> None:
        """Copy Conv2d weights are done via load_state_dict at the wrapper
        level. Here we only bake pos_emb from the Learnable2DInterpPosEmb
        source at the target (grid_h, grid_w)."""
        table = learnable.precompute_for_grid(self.grid_h, self.grid_w)
        assert table.shape == self.pos_emb_static.shape, (
            table.shape, self.pos_emb_static.shape,
        )
        self.pos_emb_static.copy_(table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (N_patches, C, patch_h, patch_w) — single image already broken
             into patches by the caller
        Returns:
          (N_patches, out_dim) with pos-emb added.

        NB. If x comes as a full image (C, H, W) or batched (B, C, H, W),
        the caller must reshape appropriately before calling.
        """
        x = self.proj(x).view(x.size(0), -1)
        return x + self.pos_emb_static
