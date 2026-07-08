import torch
from leap_llm.nn.modules import (
    FakeQuantLinear,
    FakeQuantMul,
    FakeQuantSwish,
    FakeQuantGELU,
)
from leap_llm.nn.utils import Module


class MLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = FakeQuantLinear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = FakeQuantSwish(True, 16)
        self.mul = FakeQuantMul(quantized=False)

    def build(self, hidden_state):

        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.down_proj(x)

    def forward(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.down_proj(x)


class Qwen2_5OmniMLP(Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, bias=bias
        )
        self.up_proj = FakeQuantLinear(
            self.hidden_size, self.intermediate_size, bias=bias
        )

        self.down_proj = FakeQuantLinear(
            self.intermediate_size, self.hidden_size, bias=bias
        )
        self.act_fn = FakeQuantSwish(True, 16)

        self.mul = FakeQuantMul(quantized=False)

    def build(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)

        return self.down_proj(x)

    def forward(self, hidden_state: torch.Tensor):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.down_proj(x)


class Qwen2_5OmniPatchMergerMLP(Module):
    def __init__(self, hidden_size, dim: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = dim

        self.proj0 = FakeQuantLinear(self.hidden_size, self.hidden_size)
        self.act_fn = FakeQuantGELU()
        self.proj1 = FakeQuantLinear(self.hidden_size, self.dim)

    def build(self, hidden_state):
        out = self.proj1(self.act_fn(self.proj0(hidden_state)))
        return out

    def forward(self, hidden_state):
        out = self.proj1(self.act_fn(self.proj0(hidden_state)))
        return out
