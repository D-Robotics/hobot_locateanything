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
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .blocks.text_transformer_block import Qwen3VLDecoderLayer
from .blocks.vision_block import Qwen3VLVisionBlock
from .blocks.vision_patch import Qwen3VLVisionPatchEmbed, Qwen3VLVisionPatchMerger

logger = logging.getLogger(__name__)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """
    RoPE frequency table implementation aligned with HF `Qwen3VLVisionRotaryEmbedding`.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)

        # position encoding: use interpolatable pos_embed as in HF
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_heads)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = getattr(config, "deepstack_visual_indexes", [])
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

        image_h = getattr(config, "image_height", 448)
        image_w = getattr(config, "image_width", 448)
        factor = self.patch_size * self.spatial_merge_size
        config.image_height = round(image_h / factor) * factor
        config.image_width = round(image_w / factor) * factor
        logger.info(
            f"Vision grid aligned: image_size={config.image_height}x{config.image_width}, "
            f"grid_h={config.image_height // self.patch_size}, "
            f"grid_w={config.image_width // self.patch_size}"
        )
        self.grid_thw = torch.tensor(
            [[1, config.image_height // self.patch_size, config.image_width // self.patch_size]],
            dtype=torch.long,
        )
        rotary_emb = self.rot_pos_emb(self.grid_thw)
        seq_len = int(self.grid_thw[:, 1].item() * self.grid_thw[:, 2].item())
        emb = torch.cat((rotary_emb.reshape(seq_len, -1), rotary_emb.reshape(seq_len, -1)), dim=-1)
        self.rotary_cos = emb.cos().unsqueeze(0).unsqueeze(2)
        self.rotary_sin = emb.sin().unsqueeze(0).unsqueeze(2)

        self.quant_hidden_states = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.dequant = DeQuantStub()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(int(num_frames.item()), 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # (total_tokens, 2, dim/2)
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _t, h, w in zip(grid_ts, grid_hs, grid_ws, strict=True):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()), device=self.pos_embed.weight.device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()), device=self.pos_embed.weight.device)

            h_idxs_floor = h_idxs.long()
            w_idxs_floor = w_idxs.long()
            h_idxs_ceil = (h_idxs_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs_floor + 1).clamp(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int(h.item() * w.item()) for h, w in zip(grid_hs, grid_ws, strict=True)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=True):
            pos_embed = pos_embed.repeat(int(t.item()), 1)
            pos_embed = (
                pos_embed.view(
                    int(t.item()), int(h.item()) // merge_size, merge_size, int(w.item()) // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: (1, seq_len, flatten_size), batch-first.
        Unlike transformers which squeezes the batch dim and works on (seq_len, dim),
        we keep batch-first throughout to align with the framework convention.
        seq_len must be consistent with the number of patches in grid_thw at init.

        Returns:
            image_embeds: (1, num_patches_merged, text_hidden_size)
            deepstack_feature_lists: list of (1, num_patches_merged, text_hidden_size)
        """
        grid_thw = self.grid_thw.to(hidden_states.device)
        assert hidden_states.size(0) == 1, "Vision model batch size must be 1"

        hidden_states = self.quant_hidden_states(hidden_states)
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states + pos_embeds

        cos = self.rotary_cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
        sin = self.rotary_sin.to(device=hidden_states.device, dtype=hidden_states.dtype)
        cos = self.quant_cos(cos)
        sin = self.quant_sin(sin)
        position_embeddings = (cos, sin)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                ds_feat = self.deepstack_merger_list[ds_idx](hidden_states)
                ds_feat = self.dequant(ds_feat)
                deepstack_feature_lists.append(ds_feat)

        hidden_states = self.merger(hidden_states)
        hidden_states = self.dequant(hidden_states)
        return hidden_states, deepstack_feature_lists


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Multi-modal RoPE (MRoPE) for Qwen3-VL text model.

    Uses interleaved MRoPE: position_ids shape (3, bsz, seq_len) — T, H, W.
    mrope_section from rope_scaling config specifies how many freq slots belong to T/H/W.
    """

    def __init__(self, config, device=None):
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        if isinstance(rope_scaling, dict):
            self.mrope_section = rope_scaling.get("mrope_section", [16, 24, 24])
        else:
            self.mrope_section = getattr(config, "mrope_section", [16, 24, 24])

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        base = getattr(config, "rope_theta", 5_000_000)
        max_pos = getattr(config, "max_position_embeddings", 10240)

        freq_dim = head_dim // 2

        inv_freq_full = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
        )
        t = torch.arange(max_pos, dtype=torch.int64).float()
        freqs = torch.outer(t, inv_freq_full)

        self.register_buffer("cache_freq", freqs, persistent=False)
        self.freq_dim = freq_dim

        _, h_slots, w_slots = self.mrope_section
        h_indices = torch.arange(1, h_slots * 3, 3)
        w_indices = torch.arange(2, w_slots * 3, 3)
        h_mask = torch.zeros(freq_dim, dtype=torch.bool)
        w_mask = torch.zeros(freq_dim, dtype=torch.bool)
        h_idx_valid = h_indices[h_indices < freq_dim]
        w_idx_valid = w_indices[w_indices < freq_dim]
        if h_idx_valid.numel() > 0:
            h_mask[h_idx_valid] = True
        if w_idx_valid.numel() > 0:
            w_mask[w_idx_valid] = True
        self.register_buffer("h_mask", h_mask, persistent=False)
        self.register_buffer("w_mask", w_mask, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        x: (bsz, seq_len, hidden_size) — used only for device/dtype
        position_ids: (3, bsz, seq_len) — T, H, W indices
        Returns:
            cos, sin: each (bsz, seq_len, head_dim)
        """
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        cache = self.cache_freq.to(device=x.device, dtype=x.dtype)

        pos_t = position_ids[0]
        pos_h = position_ids[1]
        pos_w = position_ids[2]

        freq_t = cache[pos_t]
        freq_h = cache[pos_h]
        freq_w = cache[pos_w]

        freq = freq_t.clone()
        h_mask = self.h_mask.to(x.device)
        w_mask = self.w_mask.to(x.device)
        freq = torch.where(h_mask.unsqueeze(0).unsqueeze(0), freq_h, freq)
        freq = torch.where(w_mask.unsqueeze(0).unsqueeze(0), freq_w, freq)

        emb = torch.cat([freq, freq], dim=-1)
        return emb.cos(), emb.sin()


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        self.layers = nn.ModuleList(
            [Qwen3VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        # Pre-compute cos/sin for all positions at init time, so that
        # _get_rotary_cos_sin() only needs index lookup instead of
        # recalculating cos()/sin() on every forward call.
        # Transformers computes cos/sin on the fly via rotary_emb.forward();
        # we trade init-time memory for inference-time speed.
        max_kvcache_len = getattr(config, "max_kvcache_len", config.max_position_embeddings)
        pseudo_hidden = torch.zeros(1, max_kvcache_len, config.hidden_size, dtype=torch.float32)
        pseudo_position_ids = torch.arange(max_kvcache_len, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
        cache_cos, cache_sin = self.rotary_emb(pseudo_hidden, pseudo_position_ids)
        cache_cos = cache_cos.squeeze(0)  # (max_kvcache_len, head_dim)
        cache_sin = cache_sin.squeeze(0)

        # Reorder cache columns from interleaved [T,H,W,T,H,W,...] to chunked
        # [T_all | H_all | W_all] layout. This allows runtime to use cheap
        # split-index-cat instead of torch.where, then a fixed permutation
        # restores the interleaved order expected by attention weights.
        freq_dim = self.rotary_emb.freq_dim
        h_mask = self.rotary_emb.h_mask
        w_mask = self.rotary_emb.w_mask
        t_mask = ~h_mask & ~w_mask

        t_idx = torch.cat([t_mask.nonzero().squeeze(-1), t_mask.nonzero().squeeze(-1) + freq_dim])
        h_idx = torch.cat([h_mask.nonzero().squeeze(-1), h_mask.nonzero().squeeze(-1) + freq_dim])
        w_idx = torch.cat([w_mask.nonzero().squeeze(-1), w_mask.nonzero().squeeze(-1) + freq_dim])
        chunked_order = torch.cat([t_idx, h_idx, w_idx])
        unperm = torch.argsort(chunked_order)

        self.mrope_section = [t_idx.numel(), h_idx.numel(), w_idx.numel()]
        self.register_buffer("cache_cos", cache_cos[:, chunked_order].contiguous(), persistent=True)
        self.register_buffer("cache_sin", cache_sin[:, chunked_order].contiguous(), persistent=True)
        self.register_buffer("mrope_unperm", unperm, persistent=False)

        self.quant_input_embeds = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_visual_embed = QuantStub()

    def _get_rotary_cos_sin(self, position_ids: torch.Tensor, device, dtype):
        """Look up pre-computed cos/sin by position_ids (no cos/sin math here).

        For decode (dim=1): position_ids is (1, bsz, 1), T/H/W share the same
        position, so directly index into cache.
        For prefill MRoPE (dim=3): position_ids is (3, bsz, seq_len), split
        chunked cache by [T|H|W] sections, index each axis independently,
        cat back, then apply a fixed permutation to restore interleaved order.
        """
        position_ids = position_ids.long()
        cache_cos = self.cache_cos.to(device=device, dtype=dtype)
        cache_sin = self.cache_sin.to(device=device, dtype=dtype)

        dim = position_ids.shape[0]
        if dim == 1:
            position_ids = position_ids.squeeze(0)
            cos = cache_cos[position_ids][:, :, self.mrope_unperm]
            sin = cache_sin[position_ids][:, :, self.mrope_unperm]
            return cos, sin

        # Prefill MRoPE: cache is stored as [T_cols | H_cols | W_cols].
        # Split, index each axis by its own positions, cat, then unperm.
        split_cos = cache_cos.split(self.mrope_section, dim=-1)
        split_sin = cache_sin.split(self.mrope_section, dim=-1)

        cos = torch.cat([split_cos[0][position_ids[0]],
                         split_cos[1][position_ids[1]],
                         split_cos[2][position_ids[2]]], dim=-1)
        sin = torch.cat([split_sin[0][position_ids[0]],
                         split_sin[1][position_ids[1]],
                         split_sin[2][position_ids[2]]], dim=-1)

        cos = cos[:, :, self.mrope_unperm]
        sin = sin[:, :, self.mrope_unperm]
        return cos, sin

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor] = None,
        caches: list[torch.Tensor] = None,
        return_all_logits: bool = False,
    ):
        """
        input_embeddings:      (bsz, seq_len, hidden_size)
        position_ids:          (3, bsz, seq_len) for prefill  OR  (3, bsz, 1) for decode
        attention_mask:        (bsz, seq_len, max_kvcache_len)  OR  (max_kvcache_len,) for decode
        caches:                list of 2*num_layers KV tensors (bsz, max_kvcache_len, num_kv_heads, head_dim)
        deepstack_visual_embeds: list of (1, num_patches_merged, hidden_size), applied to layers 0..N-1
        """
        if caches is None:
            caches = []
        new_keys = []
        new_values = []

        cos, sin = self._get_rotary_cos_sin(position_ids, input_embeddings.device, input_embeddings.dtype)
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
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                visual_embed = deepstack_visual_embeds[idx].to(hidden_states.device, hidden_states.dtype)
                visual_embed = self.quant_visual_embed(visual_embed)
                hidden_states = hidden_states + visual_embed

        if return_all_logits:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = self.dequant(logits)
            return logits, new_keys, new_values
        else:
            hidden_states = hidden_states[:, -1]
            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)
            token_logits = self.dequant(token_logits)
            return token_logits, new_keys, new_values
