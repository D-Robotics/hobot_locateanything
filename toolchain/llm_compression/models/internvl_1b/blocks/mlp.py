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

import torch
from torch import nn


class InternMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternProjcetMLP(nn.Module):
    def __init__(self, vit_hidden_size, llm_hidden_size):
        super().__init__()
        self.act = nn.GELU()
        self.fc1 = nn.Linear(vit_hidden_size, llm_hidden_size)
        self.fc2 = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.norm = nn.LayerNorm(vit_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternLM2MLP(nn.Module):
    """MLP for InternLM2 backbone (SwiGLU, same structure as Qwen3MLP)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
