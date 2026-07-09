# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) Horizon Robotics. All rights reserved.

import torch
from horizon_plugin_pytorch.nn import RMSNorm
from torch import nn

from .attention import InternAttention, InternLM2Attention
from .mlp import InternLM2MLP, InternMLP


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states).to(hidden_states.dtype)
        hidden_states = self.attn(hidden_states)
        hidden_states = torch.mul(hidden_states, self.ls1)
        hidden_states = torch.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states).to(hidden_states.dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.mul(hidden_states, self.ls2)
        hidden_states = torch.add(residual, hidden_states)
        return hidden_states


class InternLM2DecoderLayer(nn.Module):
    """Decoder layer for InternVL 2B (InternLM2 backbone).

    - q/k/v bias=False (InternLM2 uses fused wqkv)
    - head_dim=128
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = InternLM2Attention(config=config, layer_idx=layer_idx)
        self.mlp = InternLM2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
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
        hidden_states = torch.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        return hidden_states, new_key, new_value
