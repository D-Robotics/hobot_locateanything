# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from .attention import Qwen3_5Attention, Qwen3_5RMSNorm
from .linear_attention import Qwen3_5GatedDeltaNet
from .mlp import Qwen3_5MLP


class Qwen3_5DecoderLayer(torch.nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        linear_attention_mask=None,
        cache_key=None,
        cache_value=None,
        conv_state=None,
        recurrent_state=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states, _, _, new_conv_state, new_recurrent_state = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=linear_attention_mask,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
            new_key = cache_key
            new_value = cache_value
        else:
            hidden_states, new_key, new_value, _, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_key,
                cache_values=cache_value,
            )
            new_conv_state = conv_state
            new_recurrent_state = recurrent_state

        hidden_states = torch.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        return (
            hidden_states,
            new_key,
            new_value,
            new_conv_state,
            new_recurrent_state,
        )
