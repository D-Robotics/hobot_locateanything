
import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.nn.modules.linear import DynamicQuantLinear
from leap_llm.nn.utils import Module


class Qwen3VLTextMLP(Module):
    """
    (mlp): Qwen3VLTextMLP(
        (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
        (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
        (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
        (act_fn): SiLUActivation()
    )
    """

    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.up_proj = DynamicQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.down_proj = DynamicQuantLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.act_fn = F.silu

    def forward(self, hidden_state: torch.Tensor):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)

    def build(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = leap.swish(x)
        hs_up = self.up_proj(hidden_state)
        x = leap.mul(x, hs_up)
        return self.down_proj(x)
