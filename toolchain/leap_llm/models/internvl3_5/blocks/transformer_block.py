from typing import Optional, Tuple

import torch
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.modules import LayerNorm, RMSNorm
from leap_llm.nn.utils import Module

from .attention import InternAttention, Qwen3Attention
from .mlp import InternMLP, Qwen3MLP


class InternVisionEncoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def build(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states)
        ls1 = self.ls1.data
        hidden_states = leap.mul(hidden_states, ls1)
        hidden_states = leap.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        ls2 = self.ls2.data
        hidden_states = leap.mul(hidden_states, ls2)
        hidden_states = leap.add(residual, hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor]],
    ]:
        hidden_states = (
            hidden_states
            + self.attn(self.norm1(hidden_states).to(hidden_states.dtype)) * self.ls1
        )
        hidden_states = (
            hidden_states
            + self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        )

        return hidden_states


class Qwen3DecoderLayer(Module):
    def __init__(self, config, num_layer):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, num_layer=num_layer)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def build(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
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
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attn_weights, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, new_key, new_value
