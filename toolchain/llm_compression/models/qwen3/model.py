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
from horizon_plugin_pytorch.nn import RMSNorm
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .blocks.transformer_block import Qwen3DecoderLayer


class Qwen3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        context_len = config.max_kvcache_len
        # Compatibility for Qwen3: rope_theta is in rope_scaling dictionary
        if hasattr(config, "rope_theta"):
            rope_theta = config.rope_theta
        elif hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_theta = config.rope_scaling.get("rope_theta", 1000000.0)
        else:
            rope_theta = 1000000.0
        cos, sin = self._set_cos_sin_cache(
            config.max_position_embeddings,
            head_dim,
            base=rope_theta,
        )
        self.cos = cos[:, :context_len, :]
        self.sin = sin[:, :context_len, :]

        self.quant_input_embeds = QuantStub()
        self.quant_cos = QuantStub()
        self.quant_sin = QuantStub()
        self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()

    def _set_cos_sin_cache(self, max_seq_len, head_dim, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.float32).unsqueeze(0)  # [1, max_seq_len, head_dim]
        sin_cached = emb.sin().to(torch.float32).unsqueeze(0)
        return cos_cached, sin_cached

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

        self.cos = self.cos.to(device=position_ids.device, dtype=input_embeddings.dtype)
        self.sin = self.sin.to(device=position_ids.device, dtype=input_embeddings.dtype)

        cos = self.quant_cos(self.cos)
        sin = self.quant_sin(self.sin)

        position_ids_expanded = position_ids.unsqueeze(-1).expand(-1, -1, cos.size(-1)).to(torch.int64)
        cos = torch.gather(cos, 1, position_ids_expanded)
        sin = torch.gather(sin, 1, position_ids_expanded)
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
            hidden_states = hidden_states[:, -1]
            hidden_states = self.norm(hidden_states)
            token_logits = self.lm_head(hidden_states)
            token_logits = self.dequant(token_logits)
            return token_logits, new_keys, new_values
