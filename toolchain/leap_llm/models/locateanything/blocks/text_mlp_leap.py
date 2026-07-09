import torch
from torch import nn
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class LocateAnythingTextMLP(Module):
    def __init__(self, config, bias: bool = False, use_plugin: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.use_plugin = use_plugin
        if self.use_plugin:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=bias
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=bias
            )
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias=bias
            )
            self.act_fn = nn.SiLU()
        else:
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
        x = self.down_proj(x)

        return x

    def forward(self, hidden_state: torch.Tensor):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)


class Qwen2_5_VLPatchMergerMLP(Module):
    def __init__(self, hidden_size, dim: bool = False, use_plugin: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = dim
        self.use_plugin = use_plugin
        if self.use_plugin:
            self.proj0 = nn.Linear(self.hidden_size, self.hidden_size)
            self.act_fn = nn.GELU()
            self.proj1 = nn.Linear(self.hidden_size, self.dim)
        else:
            self.proj0 = DynamicQuantLinear(
                self.hidden_size, self.hidden_size, w_bits=8
            )
            self.act_fn = torch.nn.functional.gelu
            self.proj1 = DynamicQuantLinear(self.hidden_size, self.dim, w_bits=8)

    def build(self, hidden_state):
        hidden_state = self.proj0(hidden_state)
        hidden_state = leap.gelu(hidden_state)
        out = self.proj1(hidden_state)
        return out

    def forward(self, hidden_state):
        out = self.proj1(self.act_fn(self.proj0(hidden_state)))
        return out
