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
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from .blocks.mlp import InternProjcetMLP
from .blocks.transformer_block import InternVisionEncoderLayer, Qwen3DecoderLayer


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)

        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding.to(target_dtype)
        return embeddings


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        cos = cos.squeeze()
        sin = sin.squeeze()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InternVisionModel(nn.Module):
    def __init__(self, vision_config, downsample_ratio, llm_hidden_size):
        super().__init__()
        self.config = vision_config
        self.downsample_ratio = downsample_ratio

        self.embeddings = InternVisionEmbeddings(vision_config)
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )

        vit_hidden_size = vision_config.hidden_size * int(1 / downsample_ratio) ** 2
        self.mlp1 = InternProjcetMLP(vit_hidden_size, llm_hidden_size)

        self.quant_pixel_values = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = self.quant_pixel_values(pixel_values)
        hidden_states = self.embeddings(pixel_values)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        # Remove CLS token
        vit_embeds = hidden_states[:, 1:, :]

        # Pixel shuffle downsampling
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        n, w, h, c = vit_embeds.size()
        scale_factor = self.downsample_ratio
        vit_embeds = vit_embeds.view(
            n,
            int(w * scale_factor),
            -1,
            int(h * scale_factor),
            int(c / scale_factor),
        )
        vit_embeds = vit_embeds.permute(0, 1, 3, 2, 4).contiguous()
        vit_embeds = vit_embeds.reshape(n, -1, int(c / (scale_factor * scale_factor)))

        # Project to LLM hidden size
        vit_embeds = self.mlp1(vit_embeds)
        vit_embeds = self.dequant(vit_embeds)
        return vit_embeds


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # Pre-compute and cache RoPE embeddings
        max_seq_len = getattr(config, "max_kvcache_len", config.max_position_embeddings)
        pseudo_input_ids = torch.arange(max_seq_len).view(1, -1).float()
        cache_cos, cache_sin = self.rotary_emb(pseudo_input_ids, pseudo_input_ids.long())

        # Quantization stubs
        self.quant_input_embeds = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()

        self.register_buffer("cache_cos", cache_cos, persistent=True)
        self.register_buffer("cache_sin", cache_sin, persistent=True)

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

        # Get position embeddings from cache
        cos = self.cache_cos[position_ids]
        sin = self.cache_sin[position_ids]

        # Apply quantization stubs
        cos = self.quant_cos(cos)
        sin = self.quant_sin(sin)
        input_embeddings = self.quant_input_embeds(input_embeddings)
        attention_mask = self.quant_attention_mask(attention_mask)

        hidden_states = input_embeddings
        position_embeddings = (cos, sin)

        # Split caches into keys and values
        cache_keys = caches[: len(caches) // 2] if caches else []
        cache_values = caches[len(caches) // 2 :] if caches else []

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if cache_keys else None,
                cache_values=cache_values[idx] if cache_values else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        if return_all_logits:
            hidden_states = self.norm(hidden_states)
            if self.lm_head.weight.device != hidden_states.device:
                self.lm_head.to(hidden_states.device)
            logits = self.lm_head(hidden_states)
            logits = self.dequant(logits)
            return logits, new_keys, new_values
        else:
            # Take only the last token for generation
            _, seq_len, _ = hidden_states.shape
            hidden_states = hidden_states[:, -1]

            # Final normalization and LM head
            hidden_states = self.norm(hidden_states)
            if self.lm_head.weight.device != hidden_states.device:
                self.lm_head.to(hidden_states.device)
            token_logits = self.lm_head(hidden_states)
            token_logits = self.dequant(token_logits)

            return token_logits, new_keys, new_values
