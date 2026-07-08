"""
Gemma4 Attention - hybrid sliding/full with QK/V norms and KV sharing.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from hbdk4.compiler import leap

from leap_llm.models.gemma4_e2b.blocks.rmsnorm import Gemma4RMSNorm
from leap_llm.models.gemma4_e2b.config.configuration_gemma4 import Gemma4TextConfig
from leap_llm.nn.modules import (
    ConstFakeQuant,
    DynamicQuantLinear,
    FakeQuantMatmul,
)
from leap_llm.nn.utils import Module

DUMP_DIR = "/tmp/gemma4_ws"


def dump_tensor(name, tensor, step=None):
    """Dump a tensor to a .npy file and print stats."""
    if tensor is None:
        return
    arr = tensor.detach().cpu().float().numpy()
    prefix = f"layer{step}_" if step is not None else ""
    filepath = os.path.join(DUMP_DIR, f"{prefix}{name}.npy")
    os.makedirs(DUMP_DIR, exist_ok=True)
    np.save(filepath, arr)
    print(
        f"  [DUMP] {prefix}{name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}"
    )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_leap(x):
    bs, dim1, dim2, head_dim = x.type.shape
    x1 = leap.slice(x, [0, 0, 0, 0], [bs, dim1, dim2, head_dim // 2], [1, 1, 1, 1])
    x2 = leap.slice(x, [0, 0, 0, head_dim // 2], [bs, dim1, dim2, head_dim], [1, 1, 1, 1])
    x2 = leap.mul(-1, x2)
    rotate_x = leap.concat([x2, x1], -1)
    return rotate_x


def apply_rotary_pos_emb_torch(x, cos, sin, unsqueeze_dim=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def apply_rotary_pos_emb_leap(x, cos, sin):
    """
    states: (bs, seqlen, #head, head_dim)
    pe: (1, seqlen, 1, head_dim)
    """
    x_embed = leap.mul(x, cos)
    x_embed = leap.add(x_embed, leap.mul(rotate_half_leap(x), sin))
    return x_embed


class Gemma4TextAttention(Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if (not self.is_sliding) and config.global_head_dim else config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and (not self.is_sliding)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        num_kv_shared_layers = getattr(self.config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = self.config.num_hidden_layers - num_kv_shared_layers
        if layer_idx >= first_kv_shared_layer_idx:
            self.is_kv_shared_layer = True
        else:
            self.is_kv_shared_layer = False

        prev_layers = config.layer_types[:first_kv_shared_layer_idx]

        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(
                config.layer_types[layer_idx]
            )

        self.q_proj = DynamicQuantLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

            self.k_proj = DynamicQuantLinear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                w_bits=config.w_bits,
                has_scale=config.has_scale,
            )
            self.v_proj = (
                DynamicQuantLinear(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                    w_bits=config.w_bits,
                    has_scale=config.has_scale,
                )
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = DynamicQuantLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            w_bits=config.w_bits,
            has_scale=config.has_scale,
        )
        self.q_bit, self.k_bit, self.a_bit, self.v_bit = 8, 8, 16, 8
        self.qk_matmul = FakeQuantMatmul(self.q_bit, self.k_bit, None)
        self.wv_matmul = FakeQuantMatmul(self.a_bit, self.v_bit, None)
        self.cache_k_fq = ConstFakeQuant(self.k_bit)
        self.cache_v_fq = ConstFakeQuant(self.v_bit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings,
        past_key,
        past_value,
        shared_kv_states,
    ):
        bs, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        new_key, new_value = None, None

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb_torch(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb_torch(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            if self.v_proj is not None:
                value_states = self.v_proj(hidden_states).view(hidden_shape)
            else:
                value_states = key_states.transpose(1, 2).view(hidden_shape).clone()

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

            new_key = key_states
            new_value = value_states

            if (past_key is not None) and (past_value is not None):
                past_key = self.cache_k_fq(past_key)
                past_value = self.cache_v_fq(past_value)
                past_key = past_key.to(key_states.device)
                past_value = past_value.to(value_states.device)
                cur_len = key_states.shape[2]
                past_key = past_key[:, cur_len:].transpose(1, 2)
                key_states = torch.cat([past_key, key_states], dim=2)
                past_value = past_value[:, cur_len:].transpose(1, 2)
                value_states = torch.cat([past_value, value_states], dim=2)

            if self.store_full_length_kv:
                # (bs, #heads, kv_len, head_dim)
                # Store full-length KV (past + current) for KV sharing
                shared_kv_states[self.layer_idx] = key_states, value_states

        query_states = query_states.reshape(bs, self.num_attention_heads, -1, self.head_dim)

        attn_weights = self.qk_matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights.reshape(bs, self.num_attention_heads, q_len, -1)

        if self.scaling is not None and self.scaling != 1:
            attn_weights = attn_weights * self.scaling

        if attention_mask is not None:
            attn_weights = torch.add(attn_weights, attention_mask)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = attn_weights.reshape(bs, self.num_key_value_heads, self.num_key_value_groups * q_len, -1)

        attn_output = self.wv_matmul(attn_weights, value_states)

        attn_output = attn_output.view(bs, -1, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if new_key is not None and new_value is not None:
            new_key = new_key.transpose(1, 2)
            new_value = new_value.transpose(1, 2)
            new_key = self.cache_k_fq(new_key)
            new_value = self.cache_v_fq(new_value)

        return attn_output, new_key, new_value

    def build(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        shared_keys,
        shared_values,
        past_keys,
        past_values,
    ):
        """Build method using leap operations for NPU compilation."""
        bs, q_len, _ = hidden_states.type.shape
        input_shape = (bs, q_len)
        hidden_shape = (*input_shape, -1, self.head_dim)

        new_key, new_value, store_key, store_value = None, None, None, None

        cos, sin = position_embeddings

        # Q projection and reshape
        query_states = self.q_proj(hidden_states)
        query_states = leap.reshape(query_states, list(hidden_shape))
        query_states = self.q_norm(query_states)

        # Apply RoPE using leap operations
        query_states = apply_rotary_pos_emb_leap(query_states, cos, sin)
        query_states = leap.transpose(query_states, [0, 2, 1, 3])  # [bs, heads, seq, hd]

        if self.is_kv_shared_layer:
            key_states = shared_keys
            value_states = shared_values
        else:
            key_states = self.k_proj(hidden_states)
            key_states = leap.reshape(key_states, list(hidden_shape))
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb_leap(key_states, cos, sin)

            key_states = leap.transpose(key_states, [0, 2, 1, 3])

            value_states = self.v_proj(hidden_states) if self.v_proj is not None else key_states

            # print(f"[0] value_states.shape={value_states.type.shape}")
            value_states = leap.reshape(value_states, list(hidden_shape))
            # print(f"[1] value_states.shape={value_states.type.shape}")
            value_states = self.v_norm(value_states)
            # print(f"[2] value_states.shape={value_states.type.shape}")
            value_states = leap.transpose(value_states, [0, 2, 1, 3])
            # print(f"[3] value_states.shape={value_states.type.shape}")

            key_states = leap.cast_type(key_states, output_type=leap.float32)
            value_states = leap.cast_type(value_states, output_type=leap.float32)
            new_key = key_states
            new_value = value_states

            # Handle past keys/values
            if past_keys is not None:  # FIXME: postpone value process later.
                # print(f"past_keys.shape={past_keys.type.shape}")
                cache_keys = self.cache_k_fq(past_keys)
                _, c_len, nkvh, hd = cache_keys.type.shape
                cur_len = key_states.type.shape[2]
                # cropping according to left-padding
                cache_keys = leap.slice(
                    cache_keys,
                    [0, cur_len, 0, 0],
                    [bs, c_len, nkvh, hd],
                    [1, 1, 1, 1],
                )
                cache_keys = leap.transpose(cache_keys, [0, 2, 1, 3])
                key_states = leap.concat([cache_keys, key_states], 2)
                store_key = key_states

        # Reshape Q for matmul
        query_states = leap.reshape(query_states, [bs, self.num_attention_heads, -1, self.head_dim])

        query_states = leap.cast_type(query_states, output_type=leap.float32)

        # FAKEQUANT
        key_states = leap.transpose(key_states, [0, 1, 3, 2])
        # QK matmul with fake quantization
        attn_weights = self.qk_matmul(query_states, key_states)
        attn_weights = leap.cast_type(attn_weights, output_type=hidden_states.type.element_type)
        attn_weights = leap.reshape(attn_weights, [bs, self.num_attention_heads, q_len, -1])

        # Add attention mask
        if attention_mask is not None:
            attn_weights = leap.add(attn_weights, attention_mask)

        # Softmax
        attn_weights = leap.softmax(attn_weights, -1)
        # Reshape for WV matmul (GQA)

        attn_weights = leap.reshape(attn_weights, [bs, self.num_key_value_heads, self.num_key_value_groups * q_len, -1])

        attn_weights = leap.cast_type(attn_weights, output_type=leap.float32)

        # post process value states
        if not self.is_kv_shared_layer and past_values is not None:
            # print(f"past_values.shape={past_values.type.shape}")
            cache_values = self.cache_v_fq(past_values)
            _, c_len, nkvh, hd = cache_values.type.shape
            cur_len = value_states.type.shape[2]
            # cropping according to left-padding
            cache_values = leap.slice(
                cache_values,
                [0, cur_len, 0, 0],
                [bs, c_len, nkvh, hd],
                [1, 1, 1, 1],
            )
            cache_values = leap.transpose(cache_values, [0, 2, 1, 3])
            value_states = leap.concat([cache_values, value_states], 2)
            store_value = value_states

        # WV matmul with fake quantization
        attn_output = self.wv_matmul(attn_weights, value_states)
        attn_output = leap.cast_type(attn_output, output_type=hidden_states.type.element_type)

        # Reshape and project
        attn_output = leap.reshape(attn_output, [bs, -1, q_len, self.head_dim])
        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [bs, q_len, -1])
        attn_output = self.o_proj(attn_output)

        # Prepare output keys/values for cache
        # All layers return their computed K/V; model handles KV sharing
        if not self.is_kv_shared_layer:
            new_key = leap.transpose(new_key, [0, 2, 1, 3])
            new_key = self.cache_k_fq(new_key)
            new_value = leap.transpose(new_value, [0, 2, 1, 3])
            new_value = self.cache_v_fq(new_value)
            if self.store_full_length_kv:
                # Store full-length KV (past + current) for KV sharing
                return attn_output, new_key, new_value, store_key, store_value
            else:
                return attn_output, new_key, new_value, store_key, store_value
        else:
            return attn_output, new_key, new_value, store_key, store_value
