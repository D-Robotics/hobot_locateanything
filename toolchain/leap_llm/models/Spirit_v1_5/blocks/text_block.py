from typing import Optional

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.rms_norm import RMSNorm
from leap_llm.nn.utils import Module

from .text_attention import Qwen3VLTextAttention
from .text_mlp import Qwen3VLTextMLP


class Qwen3VLTextDecoderLayer(Module):
    def __init__(self, config, use_plugin: bool = False):
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLTextAttention(
            config=config, use_plugin=use_plugin
        )

        self.mlp = Qwen3VLTextMLP(config=config, use_plugin=use_plugin)

        self.input_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps, use_plugin=use_plugin
        )

        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps, use_plugin=use_plugin
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_embeddings
        )
        hidden_states = torch.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)
        return hidden_states

    def build(
        self, hidden_states, attention_mask, position_embeddings
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            position_embeddings,
        )
        hidden_states = leap.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states
