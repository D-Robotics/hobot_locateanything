# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

# from llm.nn.modules.rms_norm import RMSNorm
from .attention import Qwen2_5_VLAttention, Qwen2_5_VLVisionAttention
from .mlp import Qwen2_5_VLMLP, Qwen2_5_VLPatchMergerMLP


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = Qwen2_5_VLMLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

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


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            config.hidden_size,
            eps=1e-6,
        )
        self.norm2 = RMSNorm(
            config.hidden_size,
            eps=1e-6,
        )
        self.attn = Qwen2_5_VLVisionAttention(config.hidden_size, num_heads=config.num_heads)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states,
        lengths,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states),
            lengths=lengths,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        hidden_states = torch.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = torch.add(residual, hidden_states)
        return hidden_states


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = Qwen2_5_VLPatchMergerMLP(
            self.hidden_size,
            dim,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = self.mlp(hidden_states)
        return hidden_states
