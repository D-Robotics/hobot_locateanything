"""MoonViT patch embed — leap DSL (M3-α).

Compile-time design: input is already patchified into
    (1, N_patches, patch_size² * in_channels)  = (1, 1024, 588)  for 448x448
so the "Conv2d + reshape" in upstream MoonVisionPatchEmbed is folded into a
single Linear projection (588 -> 1152). This matches Qwen2_5_VisionPatchEmbed's
`use_conv2d=False` path which is essentially a Linear on flattened patches.

Static pos_emb is baked in as a buffer (from `pos_emb.precompute_for_grid`
during checkpoint load; see model.py wrapper — the buffer is called
`pos_emb_static`).

State-dict remap (done in vision_model_leap.py at load time):
  vision_model.patch_embed.proj.{weight,bias}  -> proj.weight  (reshaped)
                                                  proj.bias    (same)
  vision_model.patch_embed.pos_emb.weight       -> pos_emb_static (interpolated)
"""

from __future__ import annotations

import torch
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class LocateAnythingVisionPatchEmbed(Module):
    """Patch embed as a Linear over flat (patch²*C) input tokens.

    __init__:
      hidden_size = 1152, patch_size = 14, in_channels = 3
    Input:
      hidden_states: (1, N_patches, patch²*C) = (1, 1024, 588)
    Output:
      (1, N_patches, hidden_size)  with pos_emb added
    """

    def __init__(self, hidden_size: int, patch_size: int = 14, in_channels: int = 3,
                 num_patches: int = 1024, use_plugin: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.use_plugin = use_plugin

        flat_dim = patch_size * patch_size * in_channels
        if use_plugin:
            self.proj = nn.Linear(flat_dim, hidden_size, bias=True)
        else:
            self.proj = DynamicQuantLinear(flat_dim, hidden_size, bias=True, w_bits=8)

        # Pre-interpolated pos_emb, baked in as buffer at model load time.
        self.register_buffer(
            "pos_emb_static",
            torch.zeros(num_patches, hidden_size),
            persistent=False,
        )

    def build(self, hidden_states):
        # hidden_states: (1, N, flat_dim)
        x = self.proj(hidden_states)                        # (1, N, hidden)
        # Broadcast pos_emb (N, hidden) → (1, N, hidden)
        # leap.add supports broadcast via reshape.
        pos = leap.reshape(self.pos_emb_static, [1, self.num_patches, self.hidden_size])
        return leap.add(x, pos)

    def forward(self, hidden_states):
        x = self.proj(hidden_states)
        return x + self.pos_emb_static.unsqueeze(0)
