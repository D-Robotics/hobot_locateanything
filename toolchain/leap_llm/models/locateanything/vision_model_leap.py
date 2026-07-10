"""LocateAnything MoonViT vision model — leap DSL wrapper (M3-α).

Vendored & simplified from qwen2_5_vl/model.py::Qwen2_5_VLVisionModel with
these MoonViT-specific changes:

  1. No window attention, no window_index, no fullatt_block_indexes,
     no lengths splitting. All 27 layers run global attention.
  2. 2D rope table (pre-computed by utils/rope_2d.precompute_freqs_cos_sin,
     max_H=max_W=512, gathered at target grid_h/grid_w).
  3. Patch embed is a single Linear over flattened (patch²*C) input tokens.
  4. Merger + mlp1 projector fold into vision output (256, 2048) directly
     — no separate compilation stage.
"""

from __future__ import annotations

from typing import List

import torch
from hbdk4.compiler import leap
from torch import nn
from torch.quantization import DeQuantStub

from leap_llm.nn.modules import LayerNorm
from leap_llm.nn.utils import Model

try:
    from horizon_plugin_pytorch.quantization import QuantStub
except ImportError:
    QuantStub = None

from .blocks.vision_patch_leap import LocateAnythingVisionPatchEmbed
from .blocks.vision_block_leap import LocateAnythingVisionBlock
from .blocks.vision_patch_merger_leap import LocateAnythingVisionPatchMerger
from .utils.rope_2d import precompute_freqs_cos_sin, gather_freqs_by_grid


class LocateAnythingVisionModel(Model):
    """27-layer MoonViT + patch merger + mlp1 projector.

    __init__ args:
      vision_config: MoonViTConfig-like (hidden_size, num_hidden_layers,
                      num_attention_heads, patch_size, intermediate_size,
                      merge_kernel_size, image_height, image_width)
      llm_hidden:    text_config.hidden_size (target of mlp1 projection)

    build() input:
      hidden_states: (1, N_patches, patch²*in_channels)   = (1, 1024, 588)
    build() output:
      visual_embeds: (1, N_patches/4, llm_hidden)          = (1, 256, 2048)
    """

    def __init__(self, vision_config, llm_hidden: int, use_plugin: bool = False) -> None:
        super().__init__()
        self.config = vision_config
        self.llm_hidden = llm_hidden
        self.use_plugin = use_plugin

        self.patch_size = vision_config.patch_size
        self.grid_h = vision_config.image_height // self.patch_size
        self.grid_w = vision_config.image_width // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.head_dim = vision_config.hidden_size // vision_config.num_attention_heads

        # Patch embedding (flat Linear).
        self.patch_embed = LocateAnythingVisionPatchEmbed(
            hidden_size=vision_config.hidden_size,
            patch_size=self.patch_size,
            in_channels=3,
            num_patches=self.num_patches,
            use_plugin=use_plugin,
        )

        # 27 encoder blocks.
        self.blocks = nn.ModuleList([
            LocateAnythingVisionBlock(vision_config, use_plugin=use_plugin)
            for _ in range(vision_config.num_hidden_layers)
        ])
        self.final_layernorm = LayerNorm(vision_config.hidden_size)

        # Merger + mlp1 projector.
        self.merger = LocateAnythingVisionPatchMerger(
            vit_hidden=vision_config.hidden_size,
            llm_hidden=llm_hidden,
            grid_h=self.grid_h, grid_w=self.grid_w,
            merge_kernel=tuple(vision_config.merge_kernel_size),
            use_plugin=use_plugin,
        )

        # Pre-compute 2D rope table and stash as buffers (fp32).
        # max_H/W = 512 mirrors upstream Rope2DPosEmb default in
        # LocateAnything MoonVitEncoder.
        rope_table = precompute_freqs_cos_sin(
            max_height=512, max_width=512, dim=self.head_dim,
        )
        freqs = gather_freqs_by_grid(rope_table, self.grid_h, self.grid_w)  # (N, dim/2, 2)
        # Split cos/sin; expand cos/sin to (N, dim) via duplication so that
        # rotate_half apply matches (last axis of q/k is dim).
        cos_half = freqs[..., 0]                                            # (N, dim/2)
        sin_half = freqs[..., 1]
        cos = torch.cat([cos_half, cos_half], dim=-1)                       # (N, dim)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        self.register_buffer("rope_cos", cos.to(torch.float32), persistent=True)
        self.register_buffer("rope_sin", sin.to(torch.float32), persistent=True)

        if self.use_plugin:
            self.quant_hiddenstates = QuantStub()
            self.quant_cos = QuantStub()
            self.quant_sin = QuantStub()
        self.dequant = DeQuantStub()

    # ------------------------------------------------------------------
    # leap DSL build
    # ------------------------------------------------------------------
    def build(self, hidden_states):
        # hidden_states: (1, N, flat_dim)
        if self.use_plugin:
            hidden_states = self.quant_hiddenstates(hidden_states)

        # Rope tables materialized as leap tensors.
        rope_cos = self.rope_cos
        rope_sin = self.rope_sin
        if self.use_plugin:
            rope_cos = self.quant_cos(rope_cos)
            rope_sin = self.quant_sin(rope_sin)

        hidden_states = self.patch_embed(hidden_states)                     # (1, N, 1152)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, rope_cos, rope_sin)

        hidden_states = self.final_layernorm(hidden_states)
        # Clip to fp16 safe range before merger (mirrors qwen2_5_vl pattern
        # at model.py:419: leap.clip before self.merger).
        hidden_states = leap.clip(hidden_states, -65504.0, 65504.0)
        visual_embeds = self.merger(hidden_states)                           # (1, 256, 2048)
        visual_embeds = self.dequant(visual_embeds)
        return visual_embeds

    # ------------------------------------------------------------------
    # PyTorch forward — for calibration passes
    # ------------------------------------------------------------------
    def forward(self, hidden_states):
        hidden_states = self.patch_embed(hidden_states)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, self.rope_cos, self.rope_sin)
        hidden_states = self.final_layernorm(hidden_states)
        return self.merger(hidden_states)

    # ------------------------------------------------------------------
    # leap input types — matches Qwen2_5_VLVisionModel signature
    # ------------------------------------------------------------------
    def get_leap_input_types(self) -> List[leap.TensorType]:
        flat_dim = self.patch_size * self.patch_size * 3
        return [
            leap.TensorType([1, self.num_patches, flat_dim], leap.float16),
        ]
