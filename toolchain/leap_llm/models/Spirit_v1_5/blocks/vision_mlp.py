import torch
import torch.nn.functional as F
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear
from leap_llm.nn.utils import Module


class Qwen3VLVisionMLP(Module):
    """
    (mlp): Qwen3VLVisionMLP(
    (linear_fc1): Linear(in_features=1024, out_features=4096, bias=True)
    (linear_fc2): Linear(in_features=4096, out_features=1024, bias=True)
    (act_fn): GELUTanh()
    )
    """

    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = DynamicQuantLinear(
            self.hidden_size, self.intermediate_size, bias=True
        )
        self.linear_fc2 = DynamicQuantLinear(
            self.intermediate_size, self.hidden_size, bias=True
        )
        self.act_fn = "gelu_pytorch_tanh"

    def build(self, hidden_state):
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = leap.gelu(hidden_state, approximate="tanh")
        hidden_state = self.linear_fc2(hidden_state)
        return hidden_state

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = F.gelu(hidden_state, approximate="tanh")
        hidden_state = self.linear_fc2(hidden_state)
        return hidden_state
