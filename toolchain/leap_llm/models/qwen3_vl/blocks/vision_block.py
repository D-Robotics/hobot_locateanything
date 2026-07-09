from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.layer_norm import LayerNorm
from leap_llm.nn.utils import Module

from .vision_attention import Qwen3VLVisionAttention
from .vision_mlp import Qwen3VLVisionMLP


class Qwen3VLVisionBlock(Module):
    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config, use_plugin=self.use_plugin)
        self.mlp = Qwen3VLVisionMLP(config=config, use_plugin=self.use_plugin)

    def build(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states), position_embeddings=position_embeddings
        )
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
