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


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        use_conv2d=False,
        quant_output=True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_conv2d = use_conv2d

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        if self.use_conv2d is True:
            self.proj_2d = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=[patch_size, patch_size],
                stride=[patch_size, patch_size],
                bias=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        if self.use_conv2d is False:
            hidden_states = hidden_states.view(
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        else:
            hidden_states = hidden_states.view(
                -1,
                self.in_channels,
                self.patch_size,
                self.patch_size,
            )
            weight_2d = self.proj.weight.data.sum(2)
            self.proj_2d.weight.data = weight_2d

            hidden_states = self.proj_2d(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
