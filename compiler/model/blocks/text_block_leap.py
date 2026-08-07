from typing import Optional, Tuple

import torch
from hbdk4.compiler import leap

from model.layers import RMSNorm
from model.base import Module

from .text_attention_leap import LocateAnythingTextAttention
from .text_mlp_leap import LocateAnythingTextMLP


class LocateAnythingDecoderLayer(Module):
    def __init__(self, config, layer_idx: int, use_plugin=False):
        super().__init__()
        self.use_plugin = use_plugin
        self.hidden_size = config.hidden_size
        self.self_attn = LocateAnythingTextAttention(config, layer_idx, self.use_plugin)

        self.mlp = LocateAnythingTextMLP(config, use_plugin=self.use_plugin)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )

    def build(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: torch.Tensor = None,
        cache_values: torch.Tensor = None,
    ):
        residual = hidden_states
        _, seq_len, hidden_size = hidden_states.type.shape
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = leap.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states, new_key, new_value

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        residual = hidden_states
        _, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = torch.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        return hidden_states, new_key, new_value
