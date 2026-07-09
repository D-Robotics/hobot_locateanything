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

"""Gemma4 MoE blocks - aligned with transformers Gemma4TextRouter + Gemma4TextExperts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from horizon_plugin_pytorch.nn import RMSNorm
from transformers.activations import ACT2FN


class Gemma4SingleExpert(nn.Module):
    """Single expert for Gemma4 - gate_up + down structure."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma4Experts(nn.Module):
    """Collection of experts. Loads from HF packed 3D format and unpacks to ModuleList."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.experts = nn.ModuleList([Gemma4SingleExpert(config) for _ in range(self.num_experts)])

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
        packed_gate_up = state_dict.pop(prefix + "gate_up_proj", None)
        packed_down = state_dict.pop(prefix + "down_proj", None)

        if packed_gate_up is not None:
            expected_shape = (
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            )
            if packed_gate_up.shape != expected_shape:
                error_msgs.append(
                    f"{prefix}gate_up_proj has shape {tuple(packed_gate_up.shape)}, " f"expected {expected_shape}"
                )
            else:
                for idx, _expert in enumerate(self.experts):
                    state_dict[prefix + f"experts.{idx}.gate_up_proj.weight"] = packed_gate_up[idx]

        if packed_down is not None:
            expected_shape = (
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            )
            if packed_down.shape != expected_shape:
                error_msgs.append(
                    f"{prefix}down_proj has shape {tuple(packed_down.shape)}, " f"expected {expected_shape}"
                )
            else:
                for idx, _expert in enumerate(self.experts):
                    state_dict[prefix + f"experts.{idx}.down_proj.weight"] = packed_down[idx]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx, expert in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            traced_state = torch.cat(
                [current_state.new_zeros(1, hidden_states.shape[-1]), current_state],
                dim=0,
            )
            current_hidden_states = expert(traced_state)[1:]
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Gemma4Router(nn.Module):
    """Router with RMSNorm + learned scale + per_expert_scale."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps, elementwise_affine=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probabilities = F.softmax(expert_scores, dim=-1)
        top_k_weights, top_k_index = torch.topk(router_probabilities, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        top_k_weights = top_k_weights.to(router_probabilities.dtype)
        return router_probabilities, top_k_weights, top_k_index


class Gemma4SparseMoeBlock(nn.Module):
    """Gemma4 MoE block: router + experts."""

    def __init__(self, config):
        super().__init__()
        self.router = Gemma4Router(config)
        self.experts = Gemma4Experts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.router(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
