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

import math

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )

        key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = torch.mul(attn_weights, self.scale)
        attn_weights = torch.softmax(attn_weights, -1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.cache_k_fq = QuantStub()
        self.cache_v_fq = QuantStub()
        self.dequant = DeQuantStub()

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        cache_keys,
        cache_values,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply QK normalization (Qwen3-specific)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        new_key = key_states
        new_value = value_states

        if cache_keys is not None and cache_values is not None:
            cache_keys = self.cache_k_fq(cache_keys)
            cache_values = self.cache_v_fq(cache_values)
            cur_len = key_states.shape[2]
            cache_keys = cache_keys[:, cur_len:].transpose(1, 2)

            key_states = self.dequant(key_states)
            cache_keys = self.dequant(cache_keys)
            key_states = torch.cat([cache_keys, key_states], dim=2)

            cache_values = cache_values[:, cur_len:].transpose(1, 2)
            value_states = self.dequant(value_states)
            cache_values = self.dequant(cache_values)
            value_states = torch.cat([cache_values, value_states], dim=2)

            key_states = self.cache_k_fq(key_states)
            value_states = self.cache_v_fq(value_states)

        # GQA attention
        query_states = query_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, -1)
        attn_weights = torch.mul(attn_weights, 1.0 / math.sqrt(self.head_dim))
        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)
        attn_weights = torch.softmax(attn_weights, -1).to(query_states.dtype)
        attn_weights = attn_weights.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.view(bsz, -1, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        new_key = new_key.transpose(1, 2)
        new_value = new_value.transpose(1, 2)
        new_key = self.cache_k_fq(new_key)
        new_value = self.cache_v_fq(new_value)
        new_key = self.dequant(new_key)
        new_value = self.dequant(new_value)

        return attn_output, new_key, new_value
