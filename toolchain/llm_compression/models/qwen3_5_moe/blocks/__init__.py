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

from .attention import Qwen3_5MoeAttention, Qwen3_5MoeRMSNorm
from .linear_attention import Qwen3_5MoeGatedDeltaNet
from .moe import Qwen3_5MoeSparseMoeBlock
from .transformer_block import Qwen3_5MoeDecoderLayer
from .vision_block import Qwen3_5MoeVisionBlock
from .vision_patch import Qwen3_5MoeVisionPatchEmbed, Qwen3_5MoeVisionPatchMerger

__all__ = [
    "Qwen3_5MoeAttention",
    "Qwen3_5MoeRMSNorm",
    "Qwen3_5MoeGatedDeltaNet",
    "Qwen3_5MoeSparseMoeBlock",
    "Qwen3_5MoeDecoderLayer",
    "Qwen3_5MoeVisionBlock",
    "Qwen3_5MoeVisionPatchEmbed",
    "Qwen3_5MoeVisionPatchMerger",
]
