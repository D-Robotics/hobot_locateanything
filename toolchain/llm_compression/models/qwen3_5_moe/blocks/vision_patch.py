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


class Qwen3_5MoeVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        flatten_size = self.temporal_patch_size * self.patch_size * self.patch_size * self.in_channels
        self.proj = nn.Linear(flatten_size, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        return self.proj(hidden_states.to(dtype=target_dtype))


class Qwen3_5MoeVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False):
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)

        if self.use_postshuffle_norm:
            self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size)) if self.use_postshuffle_norm else self.norm(x)
        x = x.view(-1, self.hidden_size)
        x = self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="none"))
        return x
