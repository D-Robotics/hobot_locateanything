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
import torch.nn as nn
import torch.nn.functional as F

from llm_compression.utils import AttentionManager


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        if AttentionManager.is_flash_attn():
            from llm_compression.models.horizon_modules.flash_attention import HzFlashAttention

            self.attention = HzFlashAttention(block_size=AttentionManager.get_flash_block_size())
        else:
            self.attention = self.local_atten

    def local_atten(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return torch.matmul(attn_weights, value_states)

    def forward(self, hidden_states: torch.Tensor, position_embeddings):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)
