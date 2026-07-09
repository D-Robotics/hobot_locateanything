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

import torch
from torch import nn


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state: torch.Tensor):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)


class Qwen2_5_VLPatchMergerMLP(nn.Module):
    def __init__(self, hidden_size, dim: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dim = dim
        self.proj0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.proj1 = nn.Linear(self.hidden_size, self.dim)

    def forward(self, hidden_state):
        out = self.proj1(self.act_fn(self.proj0(hidden_state)))
        return out
