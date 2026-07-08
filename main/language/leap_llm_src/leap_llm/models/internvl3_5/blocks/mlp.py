import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear, LayerNorm
from leap_llm.nn.utils import Module


class InternMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = torch.nn.functional.gelu
        self.fc1 = DynamicQuantLinear(config.hidden_size, config.intermediate_size)
        self.fc2 = DynamicQuantLinear(config.intermediate_size, config.hidden_size)

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = leap.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternProjcetMLP(Module):
    def __init__(self, vit_hidden_size, llm_hidden_size):
        super().__init__()
        self.act = torch.nn.functional.gelu
        self.fc1 = DynamicQuantLinear(vit_hidden_size, llm_hidden_size)
        self.fc2 = DynamicQuantLinear(llm_hidden_size, llm_hidden_size)
        self.norm = LayerNorm(vit_hidden_size)

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = leap.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class Qwen3MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            has_scale=config.has_scale,
        )
        self.up_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            has_scale=config.has_scale,
        )
        self.down_proj = DynamicQuantLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            has_scale=config.has_scale,
        )
        self.act_fn = torch.nn.functional.silu

    def build(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.gate_proj(hidden_states)
        x = leap.swish(x)
        up_proj_h = self.up_proj(hidden_states)
        x = leap.mul(x, up_proj_h)
        x = self.down_proj(x)

        return x

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
