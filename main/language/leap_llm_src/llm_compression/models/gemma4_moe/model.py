# Copyright 2026 the HuggingFace Team. All rights reserved.
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

"""Gemma4 TextModel - pure LLM, aligned with transformers Gemma4TextModel."""

import copy

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from llm_compression.utils.logger import get_logger

from .blocks.transformer_block import Gemma4DecoderLayer

logger = get_logger(__name__)


class Gemma4ScaledEmbedding(nn.Embedding):
    """Embedding with sqrt(hidden_size) scaling, fused into weight after loading."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        self.scalar_embed_scale = embed_scale

    def fuse_embed_scale_into_weight(self):
        if self.scalar_embed_scale == 1.0:
            return
        self.weight.data.mul_(self.scalar_embed_scale)
        self.embed_scale.fill_(1.0)
        self.scalar_embed_scale = 1.0

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma4TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        padding_idx = getattr(config, "pad_token_id", None)

        self.embed_tokens = Gemma4ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

        # Precompute cos/sin via HF's Gemma4TextRotaryEmbedding for bit-identical RoPE
        self.layer_types = config.layer_types
        context_len = config.max_kvcache_len

        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextRotaryEmbedding,
        )

        rotary_config = copy.deepcopy(config)
        rotary_config.layer_types = ["sliding_attention", "full_attention"]
        hf_rotary = Gemma4TextRotaryEmbedding(rotary_config)

        dummy_x = torch.zeros(1, dtype=torch.float32)
        all_position_ids = torch.arange(context_len).unsqueeze(0)

        with torch.no_grad():
            sliding_cos, sliding_sin = hf_rotary(dummy_x, all_position_ids, layer_type="sliding_attention")
            full_cos, full_sin = hf_rotary(dummy_x, all_position_ids, layer_type="full_attention")

        self.sliding_cos = sliding_cos.float()
        self.sliding_sin = sliding_sin.float()
        self.full_cos = full_cos.float()
        self.full_sin = full_sin.float()

        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.quant_input_embeds = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.quant_slide_attention_mask = QuantStub()
        self.dequant = DeQuantStub()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_embeddings,
        position_ids,
        attention_mask,
        slide_attention_mask=None,
        caches=None,
        return_all_logits: bool = False,
    ):
        if caches is None:
            caches = []
        new_keys = []
        new_values = []

        self.sliding_cos = self.sliding_cos.to(device=position_ids.device, dtype=input_embeddings.dtype)
        self.sliding_sin = self.sliding_sin.to(device=position_ids.device, dtype=input_embeddings.dtype)
        self.full_cos = self.full_cos.to(device=position_ids.device, dtype=input_embeddings.dtype)
        self.full_sin = self.full_sin.to(device=position_ids.device, dtype=input_embeddings.dtype)

        sliding_cos = self.quant_cos(self.sliding_cos.squeeze(0)[position_ids])
        sliding_sin = self.quant_sin(self.sliding_sin.squeeze(0)[position_ids])
        full_cos = self.quant_cos(self.full_cos.squeeze(0)[position_ids])
        full_sin = self.quant_sin(self.full_sin.squeeze(0)[position_ids])

        input_embeddings = self.quant_input_embeds(input_embeddings)
        attention_mask = self.quant_attention_mask(attention_mask)
        if slide_attention_mask is None:
            slide_attention_mask = attention_mask
        slide_attention_mask = self.quant_slide_attention_mask(slide_attention_mask)

        hidden_states = input_embeddings
        sliding_position_embeddings = (sliding_cos, sliding_sin)
        full_position_embeddings = (full_cos, full_sin)

        n_layers = len(self.layers)
        cache_keys = caches[:n_layers] if len(caches) >= n_layers else [None] * n_layers
        cache_values = caches[n_layers : 2 * n_layers] if len(caches) >= 2 * n_layers else [None] * n_layers

        for idx, decoder_layer in enumerate(self.layers):
            if self.layer_types[idx] == "sliding_attention":
                position_embeddings = sliding_position_embeddings
                layer_mask = slide_attention_mask
            else:
                position_embeddings = full_position_embeddings
                layer_mask = attention_mask

            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if cache_keys[idx] is not None else None,
                cache_values=(cache_values[idx] if cache_values[idx] is not None else None),
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        if return_all_logits:
            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)
        else:
            hidden_states = hidden_states[:, -1]
            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)

        if self.final_logit_softcapping is not None:
            token_logits = token_logits / self.final_logit_softcapping
            token_logits = torch.tanh(token_logits)
            token_logits = token_logits * self.final_logit_softcapping

        token_logits = self.dequant(token_logits)
        return token_logits, new_keys, new_values
