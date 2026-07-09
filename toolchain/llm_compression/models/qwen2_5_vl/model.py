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

import logging

import torch
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.nn import functional as F
from torch.quantization import DeQuantStub
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from .blocks import (
    Qwen2_5_VisionPatchEmbed,
)
from .blocks.transformer_block import (
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionBlock,
)

logger = logging.getLogger(__name__)


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # transformers version compatibility:
        # - 4.x: ROPE_INIT_FUNCTIONS contains 'default', can be used directly
        # - 5.x: ROPE_INIT_FUNCTIONS removed 'default', need compute_default_rope_parameters
        try:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        except KeyError:
            # transformers 5.x does not support this rope_type, use default method
            self.rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        """Default RoPE initialization method for transformers 5.x compatibility.

        transformers 5.x removed ROPE_INIT_FUNCTIONS['default'],
        need to manually implement default RoPE parameter computation.
        """
        # Get rope_theta, prefer rope_parameters (5.x), then rope_scaling (4.x)
        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            base = config.rope_parameters.get("rope_theta", 10000.0)
        elif hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            base = config.rope_scaling.get("rope_theta", 10000.0)
        elif hasattr(config, "rope_theta"):
            base = config.rope_theta
        else:
            base = 10000.0

        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(1, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float().to(device=inv_freq_expanded.device)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        cos = cos.squeeze()
        sin = sin.squeeze()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VLVisionModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.dtype = torch.float32
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.merge_size = config.spatial_merge_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            use_conv2d=True,
            quant_output=False,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # Align image size to factor = patch_size * spatial_merge_size,
        # matching the smart_resize logic in qwen_vl_utils / transformers ImageProcessor.
        factor = config.patch_size * config.spatial_merge_size
        config.image_height = round(config.image_height / factor) * factor
        config.image_width = round(config.image_width / factor) * factor
        logger.info(
            f"Vision grid aligned: image_size={config.image_height}x{config.image_width}, "
            f"grid_h={config.image_height // config.patch_size}, "
            f"grid_w={config.image_width // config.patch_size}"
        )
        grid_thw = [
            1,
            config.image_height // config.patch_size,
            config.image_width // config.patch_size,
        ]
        self.grid_thw = torch.tensor([grid_thw])
        self.seq_len = grid_thw[1] * grid_thw[2]

        rotary_pos_emb_cos_sin = self.get_rotary_pos_emb_cos_sin(seq_len=self.seq_len)
        self.rotary_pos_emb_cos = rotary_pos_emb_cos_sin[0]
        self.rotary_pos_emb_sin = rotary_pos_emb_cos_sin[1]

        window_index, cu_window_seqlens = self.get_window_index(self.grid_thw)
        self.window_index = window_index

        fullatt_lengths = self.vision_mask_lengths(
            self.grid_thw,
        )
        normal_lengths = self.vision_mask_lengths(
            self.grid_thw,
            cu_window_seqlens,
        )
        self.lengths = [fullatt_lengths, normal_lengths]
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])

        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.quant_hiddenstates = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.dequant = DeQuantStub()

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )

            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def vision_position_ids(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            llm_h, llm_w = h // self.merge_size, w // self.merge_size
            # compute pos_ids
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(llm_h, self.merge_size, llm_w, self.merge_size)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        position_ids = torch.cat(pos_ids, dim=0)
        return position_ids

    def vision_mask_lengths(self, grid_thw, cu_window_seqlens=None, min_value=-512):
        seq_len = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        if cu_window_seqlens is None:
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        else:
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens,
                device=seq_len.device,
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = torch.unique_consecutive(cu_window_seqlens)
        return cu_seqlens

    def get_rotary_pos_emb_cos_sin(self, seq_len):
        grid_thw = self.grid_thw
        position_ids = self.vision_position_ids(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        pos_ids = position_ids
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).float()
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.ndim == 3:
            _, seq_len, _ = hidden_states.size()
        elif hidden_states.ndim == 2:
            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.unsqueeze(0)

        hidden_states = self.quant_hiddenstates(hidden_states)
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb_cos = self.rotary_pos_emb_cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
        rotary_pos_emb_sin = self.rotary_pos_emb_sin.to(device=hidden_states.device, dtype=hidden_states.dtype)

        rotary_pos_emb_cos = self.quant_cos(rotary_pos_emb_cos)
        rotary_pos_emb_sin = self.quant_sin(rotary_pos_emb_sin)

        window_index = self.window_index

        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(1, seq_len, -1)

        lengths = self.lengths
        fullattn_lengths = lengths[0]
        normal_lengths = lengths[1]
        for layer_num, blk in enumerate(self.blocks):
            lengths_now = fullattn_lengths if layer_num in self.fullatt_block_indexes else normal_lengths

            hidden_states = blk(
                hidden_states,
                lengths=lengths_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )
        hidden_states = self.merger(hidden_states)
        hidden_states = self.dequant(hidden_states)
        return hidden_states


class Qwen2_5_VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.config = config
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config)
        if hasattr(config, "layer_types") is False:
            config.layer_types = ["full_attention" for i in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        pseudo_input_ids = torch.arange(self.config.max_kvcache_len).view(1, 1, -1).float()
        cache_cos, cache_sin = self.rotary_emb(pseudo_input_ids, pseudo_input_ids)
        mrope_section = self.config.rope_scaling["mrope_section"]
        self.mrope_section = mrope_section * 2
        self.quant_input_embeds = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()
        self.register_buffer("cache_cos", cache_cos, persistent=True)
        self.register_buffer("cache_sin", cache_sin, persistent=True)

    def get_rotary_emb(self):
        return self.rotary_emb

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_embeddings,
        position_ids,
        attention_mask,
        caches=None,
        return_all_logits: bool = False,
    ):
        if caches is None:
            caches = []
        new_keys = []
        new_values = []
        dim = position_ids.shape[1]
        if dim > 1:
            split_cache_cos = self.cache_cos.split(self.mrope_section, dim=-1)
            split_cache_sin = self.cache_sin.split(self.mrope_section, dim=-1)

            split_position_ids = position_ids.to(self.cache_cos.device).split([1, 1, 1], dim=1)

            used_cos = []
            used_sin = []
            for (i, cos), sin in zip(enumerate(split_cache_cos), split_cache_sin):
                cur_position_ids = split_position_ids[i % 3]
                cos = cos.contiguous()
                used_cos.append(cos[cur_position_ids])
                sin = sin.contiguous()
                used_sin.append(sin[cur_position_ids])

            cos = torch.cat(used_cos, dim=-1)

            sin = torch.cat(used_sin, dim=-1)
        else:
            position_ids = position_ids.to(self.cache_cos.device)
            cos = self.cache_cos[position_ids]
            sin = self.cache_sin[position_ids]

        cos = self.quant_cos(cos)
        sin = self.quant_sin(sin)
        input_embeddings = self.quant_input_embeds(input_embeddings)
        attention_mask = self.quant_attention_mask(attention_mask)
        hidden_states = input_embeddings
        position_embeddings = (cos, sin)
        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        if return_all_logits:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = self.dequant(logits)
            return logits, new_keys, new_values
        else:
            _, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states[:, -1]

            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)
            token_logits = self.dequant(token_logits)
            return token_logits, new_keys, new_values
