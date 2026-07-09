import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import DynamicQuantLinear, LayerNorm
from leap_llm.nn.utils import Module


class Projector(Module):
    def __init__(
        self, vision_hidden_size: int, llm_hidden_size: int, downsample_ratio: float
    ):
        super().__init__()
        intermediate_size = vision_hidden_size * int(1 / downsample_ratio) ** 2
        self.norm = LayerNorm(intermediate_size, 1e-5, True)
        self.linear1 = DynamicQuantLinear(intermediate_size, llm_hidden_size, True)
        self.act_fn = torch.nn.GELU()
        self.linear3 = DynamicQuantLinear(llm_hidden_size, llm_hidden_size, True)

    def build(self, vit_embeds):
        hidden_states = self.norm(vit_embeds)
        hidden_states = self.linear1(hidden_states)
        hidden_states = leap.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear3(hidden_states)
        return hidden_states

    def forward(self, vit_embeds: torch.Tensor):
        hidden_states = self.norm(vit_embeds)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear3(hidden_states)
        return hidden_states
