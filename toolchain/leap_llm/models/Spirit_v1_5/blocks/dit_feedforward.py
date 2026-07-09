import torch
import torch.nn.functional as F
from torch import nn
from leap_llm.nn.utils import Module
from leap_llm.nn.modules.activation import FakeQuantGELU
from leap_llm.nn.modules.linear import DynamicQuantLinear

class GELU(Module):

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = DynamicQuantLinear(dim_in, dim_out, bias=bias)
        self.approximate = approximate
        self.gelu = FakeQuantGELU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states

    def build(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states

class Identity(Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states
    
    def build(self, hidden_states):
        return hidden_states

class FeedForward(Module):

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(Identity())
        self.net.append(DynamicQuantLinear(inner_dim, dim_out, bias=bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

    def build(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states