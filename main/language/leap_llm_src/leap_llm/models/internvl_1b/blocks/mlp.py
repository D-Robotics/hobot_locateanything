import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class FeedForward(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        w_bits: int,
        has_scale: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale
        )
        self.up_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale
        )
        self.down_proj = DynamicQuantLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale
        )
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = torch.nn.functional.silu

    def build(self, hidden_state):
        return self.down_proj(
            leap.mul(
                leap.swish(self.gate_proj(hidden_state)),
                self.up_proj(hidden_state),
            )
        )

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        )


class MLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.fc1 = DynamicQuantLinear(
            self.hidden_size, self.intermediate_size, bias=True
        )
        self.fc2 = DynamicQuantLinear(
            self.intermediate_size, self.hidden_size, bias=True
        )
        self.act_fn = torch.nn.functional.gelu

    def build(self, hidden_state):
        return self.fc2(leap.gelu(self.fc1(hidden_state), approximate="tanh"))

    def forward(self, hidden_state):
        return self.fc2(self.act_fn(self.fc1(hidden_state)))
