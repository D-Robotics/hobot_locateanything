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
from transformers.activations import ACT2FN


class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5MoeSingleExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_size
        intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Token routing via indexing can turn activations back into plain fp tensors.
        # Align with the linear weight dtype before entering QAT Linear.
        # hidden_states = hidden_states.to(self.gate_up_proj.weight.dtype)
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.experts = nn.ModuleList([Qwen3_5MoeSingleExpert(config) for _ in range(self.num_experts)])

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
        # Keep compatibility with the original packed HF checkpoint layout:
        # `gate_up_proj[num_experts, 2*intermediate_dim, hidden_dim]`
        # `down_proj[num_experts, hidden_dim, intermediate_dim]`
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
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx, expert in enumerate(self.experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # Feed each expert at least one dummy token so all expert modules are
            # present in the traced graph, then discard the dummy output.
            traced_state = torch.cat(
                [current_state.new_zeros(1, hidden_states.shape[-1]), current_state],
                dim=0,
            )
            current_hidden_states = expert(traced_state)[1:]
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        self.linear.weight = self.weight
        router_logits = self.linear(hidden_states)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)
