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
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .blocks.attention import Qwen3_5MoeRMSNorm
from .blocks.transformer_block import Qwen3_5MoeDecoderLayer
from .blocks.vision_block import Qwen3_5MoeVisionBlock
from .blocks.vision_patch import (
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
)


class Qwen3_5MoeVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class Qwen3_5MoeVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3_5MoeVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5MoeVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3_5MoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5MoeVisionPatchMerger(config, use_postshuffle_norm=False)

        image_h = getattr(config, "image_height", 448)
        image_w = getattr(config, "image_width", 448)
        self.default_grid_thw = torch.tensor(
            [[1, image_h // self.patch_size, image_w // self.patch_size]],
            dtype=torch.long,
        )
        rotary_emb = self.rot_pos_emb(self.default_grid_thw)
        seq_len = int(
            self.default_grid_thw[:, 0].item() * self.default_grid_thw[:, 1].item() * self.default_grid_thw[:, 2].item()
        )
        emb = torch.cat(
            (rotary_emb.reshape(seq_len, -1), rotary_emb.reshape(seq_len, -1)),
            dim=-1,
        )
        self.rotary_cos = emb.cos().unsqueeze(0).unsqueeze(2)
        self.rotary_sin = emb.sin().unsqueeze(0).unsqueeze(2)

        self.quant_hidden_states = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.dequant = DeQuantStub()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
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
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
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

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, hidden_states: torch.Tensor):
        grid_thw = self.default_grid_thw.to(hidden_states.device)

        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)
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

        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        hidden_states = self.dequant(hidden_states)
        return hidden_states.unsqueeze(0)


class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.base = rope_parameters.get("rope_theta", 1000000.0)
        self.partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 0.25)
        self.mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_dim = int(head_dim * self.partial_rotary_factor)
        self.attention_scaling = 1.0
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).float() / self.rotary_dim)
        )

    def apply_interleaved_mrope(self, freqs):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3_5MoeTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)

        self.quant_input_embeds = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.quant_position_ids = QuantStub()
        self.dequant = DeQuantStub()

    def get_input_embeddings(self):
        return self.embed_tokens

    def _split_caches(self, caches):
        n_layers = len(self.layers)
        cache_keys = caches[:n_layers]
        cache_values = caches[n_layers : 2 * n_layers]
        conv_states = caches[2 * n_layers : 3 * n_layers]
        recurrent_states = caches[3 * n_layers :]
        return cache_keys, cache_values, conv_states, recurrent_states

    def forward(
        self,
        input_embeddings,
        position_ids,
        attention_mask,
        linear_attention_mask,
        caches=None,
        return_all_logits: bool = False,
    ):
        if caches is None:
            caches = []
        new_keys = []
        new_values = []
        new_conv_states = []
        new_recurrent_states = []

        input_embeddings = self.quant_input_embeds(input_embeddings)
        if attention_mask is not None:
            attention_mask = self.quant_attention_mask(attention_mask)

        hidden_states = input_embeddings
        position_ids = position_ids.to(hidden_states.device)
        position_ids = self.quant_position_ids(position_ids).to(hidden_states.dtype)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cache_keys, cache_values, conv_states, recurrent_states = self._split_caches(caches)

        for idx, decoder_layer in enumerate(self.layers):
            (
                hidden_states,
                new_key,
                new_value,
                new_conv_state,
                new_recurrent_state,
            ) = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                linear_attention_mask=linear_attention_mask,
                cache_key=cache_keys[idx] if len(cache_keys) else None,
                cache_value=cache_values[idx] if len(cache_values) else None,
                conv_state=conv_states[idx] if len(conv_states) else None,
                recurrent_state=recurrent_states[idx] if len(recurrent_states) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
            new_conv_states.append(new_conv_state)
            new_recurrent_states.append(new_recurrent_state)

        if return_all_logits:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            logits = self.dequant(logits)
            return (
                logits,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            )
        else:
            hidden_states = hidden_states[:, -1]
            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)
            token_logits = self.dequant(token_logits)
            return (
                token_logits,
                new_keys,
                new_values,
                new_conv_states,
                new_recurrent_states,
            )
