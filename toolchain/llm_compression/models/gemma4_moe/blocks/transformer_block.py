# Copyright 2026 the HuggingFace Team. All rights reserved.
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

"""Gemma4 DecoderLayer - aligned with transformers Gemma4TextDecoderLayer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm

from .attention import Gemma4Attention
from .mlp import Gemma4MLP
from .moe import Gemma4Experts, Gemma4Router


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # layer_scalar: initialized to 1.0, loaded from checkpoint
        self.register_buffer("layer_scalar", torch.ones(1))

        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.router = Gemma4Router(config)
            self.experts = Gemma4Experts(config)
            self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, new_key, new_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache_keys=cache_keys,
            cache_values=cache_values,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = torch.add(residual, hidden_states)
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, new_key, new_value
