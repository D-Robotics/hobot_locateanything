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

"""Qwen3Moe MoE blocks - aligned with transformers Qwen3MoeSparseMoeBlock.

No shared_expert (unlike Qwen3_5Moe). Router uses nn.Linear(hidden_dim -> num_experts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class Qwen3MoeSingleExpert(nn.Module):
    """Single expert for Qwen3Moe - gate_up + down, no shared expert."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3MoeExperts(nn.Module):
    """Collection of experts, loads from HF packed format: gate_up_proj, down_proj."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.experts = nn.ModuleList([Qwen3MoeSingleExpert(config) for _ in range(self.num_experts)])

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

        # Do not skip experts when token_idx is empty: skipping changes how many
        # times ops like mul run per forward, which breaks Horizon trace/calib
        # (fixed graph vs variable routing). Same pattern as Qwen3_5MoeExperts.
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


class Qwen3MoeTopKRouter(nn.Module):
    """Router linear: hidden_dim -> num_experts (same as HF gate weight layout)."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        self.hidden_dim = config.hidden_size
        self.linear = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.linear(hidden_states)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block - gate + experts, no shared_expert."""

    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
