import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    DynamicQuantLinear,
    FakeQuantLinear,
    FakeQuantMul,
    FakeQuantSwish,
)
from leap_llm.nn.utils import Module


class MLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        w_bits: int,
        has_scale: bool,
        march: str = "nash-e",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.march = march

        if "nash-p" in self.march:
            self.gate_proj = DynamicQuantLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.up_proj = DynamicQuantLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.down_proj = DynamicQuantLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.act_fn = torch.nn.functional.silu
        else:
            self.gate_proj = FakeQuantLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.up_proj = FakeQuantLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            self.down_proj = FakeQuantLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                w_bits=w_bits,
                has_scale=has_scale,
            )
            # self.act_fn = ACT2FN[config.hidden_act]
            self.act_fn = FakeQuantSwish(True, 16)

        self.mul = FakeQuantMul(quantized=False)

    def build(self, hidden_state):
        if "nash-p" in self.march:
            x = self.gate_proj(hidden_state)
            x = leap.swish(x)
            up_proj_h = self.up_proj(hidden_state)
            x = leap.mul(x, up_proj_h)
            return self.down_proj(x)
        else:
            x = self.gate_proj(hidden_state)
            x = self.act_fn(x)
            up_proj_h = self.up_proj(hidden_state)
            x = self.mul(x, up_proj_h)
            return self.down_proj(x)

    def forward(self, hidden_state):
        if "nash-p" in self.march:
            x = self.gate_proj(hidden_state)
            x = self.act_fn(x)
            up_proj_h = self.up_proj(hidden_state)
            x = torch.mul(x, up_proj_h)
            return self.down_proj(x)
        else:
            x = self.gate_proj(hidden_state)
            x = self.act_fn(x)
            up_proj_h = self.up_proj(hidden_state)
            x = self.mul(x, up_proj_h)
            return self.down_proj(x)
