import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class SmolLM2MLP(Module):
    """SmolLM2 FFN (SiLU gate, no bias on projections)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = DynamicQuantLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = DynamicQuantLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = DynamicQuantLinear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def build(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = leap.swish(x)
        up_proj_h = self.up_proj(hidden_state)
        x = leap.mul(x, up_proj_h)
        return self.down_proj(x)

    def forward(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = torch.nn.functional.silu(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)
