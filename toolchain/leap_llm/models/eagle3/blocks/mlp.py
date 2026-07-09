import torch
from torch import nn
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class MLP(Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.up_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.down_proj = DynamicQuantLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.act_fn = torch.nn.functional.silu

    def build(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = leap.swish(x)
        up_proj_h = self.up_proj(hidden_state)
        x = leap.mul(x, up_proj_h)
        return self.down_proj(x)

    def forward(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)
