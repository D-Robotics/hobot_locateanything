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

import math

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from llm_compression.utils import AttentionManager


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads"
                f"(got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
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
        if q_len >= 1024:
            key_states = key_states.reshape([bsz, self.num_key_value_heads, 1, -1, self.head_dim])
            key_states = key_states.repeat([1, 1, self.num_key_value_groups, 1, 1])
            key_states = key_states.reshape(
                [bsz, self.num_key_value_heads * self.num_key_value_groups, -1, self.head_dim]
            )
            value_states = value_states.reshape([bsz, self.num_key_value_heads, 1, -1, self.head_dim])
            value_states = value_states.repeat([1, 1, self.num_key_value_groups, 1, 1])
            value_states = value_states.reshape(
                [bsz, self.num_key_value_heads * self.num_key_value_groups, -1, self.head_dim]
            )
        else:
            query_states = query_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if q_len < 1024:
            attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, -1)

        attn_weights = torch.mul(attn_weights, scale)
        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)
        attn_weights = torch.softmax(attn_weights, -1).to(query_states.dtype)

        if q_len < 1024:
            attn_weights = attn_weights.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups * q_len, -1)
        return torch.matmul(attn_weights, value_states)

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

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states = query_states.reshape(-1, q_len, self.head_dim)
        key_states = key_states.reshape(-1, q_len, self.head_dim)
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin)
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

        if AttentionManager.is_flash_attn():
            query_states = query_states.reshape(bsz, self.num_heads, q_len, self.head_dim)
            key_states = key_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
            value_states = value_states.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)

        attn_output = self.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scale=self.q_mul_value,
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


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.q_mul_value = 1.0 / math.sqrt(self.head_dim)
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
        attn_weights = torch.mul(attn_weights, scale)
        attn_weights = torch.softmax(attn_weights, -1)
        return torch.matmul(attn_weights, value_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor = None,
        rotary_pos_emb_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[1]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, rotary_pos_emb_cos, rotary_pos_emb_sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = lengths[1:] - lengths[:-1]

        num_splits = torch.unique_consecutive(lengths)

        diffs = torch.cat((torch.tensor([1], device=lengths.device), torch.diff(lengths).abs()))
        group_ids = (diffs != 0).cumsum(dim=0) - 1
        result = torch.zeros_like(lengths)
        result.scatter_add_(0, group_ids, lengths)

        unique_groups = torch.unique_consecutive(group_ids)
        lengths = result[unique_groups]

        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = []
        for q, k, v, num_split in zip(*splits, num_splits):
            bs, num_heads, _, num_embeds = q.shape
            if num_split != 1:
                q = q.view(bs * self.num_heads, -1, num_split, num_embeds)
                k = k.view(bs * self.num_heads, -1, num_split, num_embeds)
                v = v.view(bs * self.num_heads, -1, num_split, num_embeds)

            attn_output_tmp = self.attention(
                q,
                k,
                v,
                attention_mask=None,
                scale=self.q_mul_value,
            )
            attn_output_tmp = attn_output_tmp.view(bs, num_heads, -1, num_embeds)
            attn_output_tmp = attn_output_tmp.transpose(1, 2)
            attn_outputs.append(attn_output_tmp)
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(1, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output
