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
from horizon_plugin_pytorch.nn.rms_norm import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from llm_compression.utils.trace_utils import is_tracing_all_branches


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g))

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    lower_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(lower_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    chunk_outputs = []
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(upper_mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        chunk_outputs.append(attn_inter + attn @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = torch.stack(chunk_outputs, dim=2).reshape(
        batch_size,
        num_heads,
        -1,
        v_head_dim,
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g))

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.rmsnorm = RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states, gate=None):
        self.rmsnorm.weight = self.weight
        hidden_states = self.rmsnorm(hidden_states)
        hidden_states = hidden_states * F.silu(gate)
        return hidden_states


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d_update = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.quant_conv_state = QuantStub()
        self.quant_recurrent_state = QuantStub()
        self.dequant = DeQuantStub()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        conv_weight_key = prefix + "conv1d.weight"
        conv_weight_update_key = prefix + "conv1d_update.weight"
        if conv_weight_key in state_dict and conv_weight_update_key not in state_dict:
            state_dict[conv_weight_update_key] = state_dict[conv_weight_key].clone()

        conv_bias_key = prefix + "conv1d.bias"
        conv_bias_update_key = prefix + "conv1d_update.bias"
        if conv_bias_key in state_dict and conv_bias_update_key not in state_dict:
            state_dict[conv_bias_update_key] = state_dict[conv_bias_key].clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _empty_state(self, hidden_states: torch.Tensor):
        return hidden_states.new_zeros((hidden_states.shape[0], 0))

    def _decode_conv(self, conv_state, mixed_qkv, seq_len):
        """Decode path: concat cached conv_state with new token, apply conv1d_update."""
        hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1)
        result = F.silu(self.conv1d_update(hidden_states_new)[:, :, -seq_len:])
        new_conv_state = hidden_states_new[:, :, -conv_state.shape[-1] :]
        return result, new_conv_state

    def _prefill_conv(self, mixed_qkv, seq_len):
        """Prefill path: pad and apply full conv1d."""
        pad_size = max(0, self.conv_kernel_size - mixed_qkv.shape[-1])
        new_conv_state = F.pad(mixed_qkv, (pad_size, 0))[..., -self.conv_kernel_size :]
        result = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        return result, new_conv_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ):
        conv_state = self.quant_conv_state(conv_state)
        recurrent_state = self.quant_recurrent_state(recurrent_state)
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            conv_state is not None
            and recurrent_state is not None
            and conv_state.numel() > 0
            and recurrent_state.numel() > 0
            and seq_len == 1
        )

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if is_tracing_all_branches():
            # JIT trace mode: execute BOTH branches so all operators
            # (torch.cat, conv1d_update, conv1d) are recorded in the trace
            # graph. Without this, "lm" mode traces with seq_len>1 and only
            # covers the prefill path, leaving decode ops (torch.cat) untraced.
            # Untraced ops fail when inputs become QTensors during calibration.
            self._decode_conv(conv_state, mixed_qkv, seq_len)
            mixed_qkv, new_conv_state = self._prefill_conv(mixed_qkv, seq_len)
        elif use_precomputed_states:
            mixed_qkv, new_conv_state = self._decode_conv(conv_state, mixed_qkv, seq_len)
        else:
            mixed_qkv, new_conv_state = self._prefill_conv(mixed_qkv, seq_len)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if is_tracing_all_branches():
            # Trace mode: execute both attention branches.
            # Same reason as conv branches above — torch_recurrent_gated_delta_rule
            # uses ops (mul, add, etc.) that must be traced for QTensor support.
            torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out, new_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        elif use_precomputed_states:
            core_attn_out, new_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, new_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)

        new_conv_state = self.quant_conv_state(new_conv_state)
        new_recurrent_state = self.quant_recurrent_state(new_recurrent_state)
        new_conv_state = self.dequant(new_conv_state)
        new_recurrent_state = self.dequant(new_recurrent_state)

        return output, None, None, new_conv_state, new_recurrent_state
