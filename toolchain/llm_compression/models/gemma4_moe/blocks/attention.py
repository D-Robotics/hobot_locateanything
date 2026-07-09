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

"""Gemma4 Attention - aligned with transformers Gemma4TextAttention."""

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from llm_compression.utils import AttentionManager


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_4d(x, cos, sin):
    """Apply rotary pos emb to 4D tensor [bs, seq, num_heads, head_dim]."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma4Attention(nn.Module):
    """Hybrid sliding/full attention with per-layer head_dim and num_kv_heads."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        if self.is_sliding:
            self.head_dim = config.head_dim
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.head_dim = getattr(config, "global_head_dim", config.head_dim)
            self.num_key_value_heads = getattr(config, "num_global_key_value_heads", config.num_key_value_heads)

        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = 1.0
        self.attention_bias = getattr(config, "attention_bias", False)

        self.use_k_eq_v = getattr(config, "attention_k_eq_v", False) and not self.is_sliding

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.attention_bias,
            )
        else:
            self.v_proj = None

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_bias,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

        self.cache_k_fq = QuantStub()
        self.cache_v_fq = QuantStub()
        self.dequant = DeQuantStub()
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
        bsz, _, q_len, _ = query_states.shape
        query_states = query_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, -1)
        attn_weights = attn_weights * scale

        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1)
        return torch.matmul(attn_weights, value_states)

    def forward(self, hidden_states, attention_mask, position_embeddings, cache_keys, cache_values):
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))

        if self.v_proj is not None:
            value_states = self.v_proj(hidden_states).view(hidden_shape)
        else:
            value_states = self.k_proj(hidden_states).view(hidden_shape)

        value_states = self.v_norm(value_states)
        cos, sin = position_embeddings

        query_states = apply_rotary_pos_emb_4d(query_states, cos, sin)
        key_states = apply_rotary_pos_emb_4d(key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

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

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scale=self.scaling,
        )

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
