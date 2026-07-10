"""MoonViT patch merger + mlp1 projector — leap DSL (M3-α).

Structure:
  Input : (1, N=1024, hidden=1152)   post-encoder tokens
  Step 1: 2x2 spatial merge via reshape + permute       → (N/4, 4*hidden = 4608)
  Step 2: LayerNorm(4608) → Linear(4608, 2048) → GELU → Linear(2048, 2048)
  Output: (N/4=256, llm_hidden=2048)

State-dict keys (matches LocateAnything checkpoint after remap):
  mlp1.0.{weight,bias}  → merger.mlp1.0.*  (LayerNorm(4608))
  mlp1.1.{weight,bias}  → merger.mlp1.1.*  (Linear 4608 → 2048)
  mlp1.3.{weight,bias}  → merger.mlp1.3.*  (Linear 2048 → 2048)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.modules import DynamicQuantLinear, LayerNorm
from leap_llm.nn.utils import Module


class LocateAnythingVisionPatchMerger(Module):
    """2x2 patch merger + mlp1 projection to LLM hidden."""

    def __init__(self, vit_hidden: int, llm_hidden: int,
                 grid_h: int = 32, grid_w: int = 32,
                 merge_kernel: tuple[int, int] = (2, 2),
                 use_plugin: bool = False) -> None:
        super().__init__()
        self.use_plugin = use_plugin
        self.vit_hidden = vit_hidden
        self.llm_hidden = llm_hidden
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.merge_kh, self.merge_kw = merge_kernel
        assert grid_h % self.merge_kh == 0 and grid_w % self.merge_kw == 0

        self.new_h = grid_h // self.merge_kh
        self.new_w = grid_w // self.merge_kw
        self.merged_seq = self.new_h * self.new_w                # 256
        self.merged_dim = vit_hidden * self.merge_kh * self.merge_kw  # 4608

        if use_plugin:
            self.mlp1 = nn.Sequential(
                LayerNorm(self.merged_dim),
                nn.Linear(self.merged_dim, llm_hidden),
                nn.GELU(),
                nn.Linear(llm_hidden, llm_hidden),
            )
        else:
            # LayerNorm + two DynamicQuantLinears + GELU. State-dict keys
            # mlp1.0.weight (LN), mlp1.1.weight (Linear1), mlp1.3.weight (Linear2).
            self.mlp1 = nn.Sequential(
                LayerNorm(self.merged_dim),
                DynamicQuantLinear(self.merged_dim, llm_hidden, bias=True, w_bits=8),
                nn.GELU(),
                DynamicQuantLinear(llm_hidden, llm_hidden, bias=True, w_bits=8),
            )

    def _merge_2x2_leap(self, x):
        """(1, grid_h*grid_w, vit_hidden) -> (1, new_h*new_w, merged_dim)."""
        # Reshape then transpose to interleave 2x2 patches into channels.
        x = leap.reshape(x, [1, self.new_h, self.merge_kh, self.new_w, self.merge_kw, self.vit_hidden])
        x = leap.transpose(x, [0, 1, 3, 2, 4, 5])
        x = leap.reshape(x, [1, self.merged_seq, self.merged_dim])
        return x

    def _merge_2x2_torch(self, x):
        x = x.view(1, self.new_h, self.merge_kh, self.new_w, self.merge_kw, self.vit_hidden)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(1, self.merged_seq, self.merged_dim)
        return x

    def build(self, hidden_states):
        # (1, N, hidden) -> (1, N/4, 4*hidden)
        x = self._merge_2x2_leap(hidden_states)
        # mlp1 with LN → Linear → GELU → Linear
        ln, lin1, _gelu, lin2 = self.mlp1[0], self.mlp1[1], self.mlp1[2], self.mlp1[3]
        x = ln(x)                                       # LayerNorm on last dim
        x = lin1(x)
        try:
            x = leap.gelu(x, approximate="tanh")
        except TypeError:
            x = leap.gelu(x)
        x = lin2(x)
        return x

    def forward(self, hidden_states):
        x = self._merge_2x2_torch(hidden_states)
        return self.mlp1(x)
