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

"""Qwen3MoeDecoderLayer - attention + MLP or MoE per layer.

Uses decoder_sparse_step and mlp_only_layers to decide MLP vs MoE, aligned with transformers.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm

from .attention import Qwen3MoeAttention
from .mlp import Qwen3MoeMLP
from .moe import Qwen3MoeSparseMoeBlock


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        num_experts = getattr(config, "num_experts", 0)

        use_moe = layer_idx not in mlp_only_layers and num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0
        if use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = torch.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = torch.add(residual, hidden_states)

        return hidden_states, new_key, new_value
