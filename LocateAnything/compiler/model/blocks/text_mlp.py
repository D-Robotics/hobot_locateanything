"""Qwen2 SwiGLU MLP — compile-friendly port.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_qwen2.py:191 (Qwen2MLP)

Structure:
  gate_proj(x)               (hidden -> intermediate)
  up_proj(x)                 (hidden -> intermediate)
  SiLU(gate) * up            (intermediate)
  down_proj(...)             (intermediate -> hidden)

State-dict layout matches upstream:
  gate_proj.weight    (intermediate, hidden)
  up_proj.weight      (intermediate, hidden)
  down_proj.weight    (hidden, intermediate)

All Linears have bias=False (matches Qwen2 canonical config).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen2MLPStatic(nn.Module):
    """SwiGLU MLP for the compile pipeline."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
