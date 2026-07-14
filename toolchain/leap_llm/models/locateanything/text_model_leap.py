"""LocateAnything language model — leap DSL version.

Source: heavily inspired by (i.e. copy-and-adapt from)
    toolchain/leap_llm/models/qwen2_5_vl/model.py::Qwen2_5_VLTextModel
Copyright of the original file: The Qwen Team, HuggingFace, D-Robotics.
This file is our LocateAnything-specific adaptation and lives entirely
in leap_llm/models/locateanything/.

Key differences from Qwen2_5_VLTextModel:
  1. Vanilla 1D rope only (no mrope 3-way split). position_ids shape is
     always (bs, 1, seq); the mrope branch is deleted rather than gated.
  2. tie_word_embeddings=True is respected — lm_head is computed as
     matmul against embed_tokens.weight, no independent lm_head allocation
     (avoids the 92MB duplicate weight; see design_notes.md pit #1).
  3. attention_mask is the PBD-style additive mask constructed by the
     host (see blocks/pbd_mask.py). We do not build a causal mask here.
  4. decode_seq_len defaults to 6 (PBD block_size).

Classes:
  LocateAnythingRotaryEmbedding      — precomputes 1D cos/sin tables
  LocateAnythingTextModel             — full 36-layer text stack with
                                        leap DSL `build()` for compile
                                        and PyTorch `forward()` for
                                        calibration.
"""

from __future__ import annotations

from typing import List

import torch
from hbdk4.compiler import leap
from torch import nn
from torch.quantization import DeQuantStub

from leap_llm.nn.modules import (
    DynamicQuantLinear,
    Embedding,
    RMSNorm,
)
from leap_llm.nn.utils import Model

try:
    from horizon_plugin_pytorch.quantization import QuantStub
except ImportError:
    QuantStub = None

from .blocks.text_block_leap import LocateAnythingDecoderLayer


# ---------------------------------------------------------------------------
# Vanilla 1D rope table — precompute cos/sin caches once at build time.
# ---------------------------------------------------------------------------
class LocateAnythingRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = float(config.rope_theta)

        # Same schedule as Qwen2's Qwen2RotaryEmbedding (modeling_qwen2.py:117)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def build_cos_sin(self, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) with shape (max_len, dim).

        Duplicated freq layout (`emb = cat([freqs, freqs], dim=-1)`)
        matches Qwen2's rope apply (which uses rotate_half).
        """
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)                # (max_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)              # (max_len, dim)
        return emb.cos(), emb.sin()


# ---------------------------------------------------------------------------
# The main text stack.
# ---------------------------------------------------------------------------
class LocateAnythingTextModel(Model):
    """36-layer Qwen2 decoder stack with vanilla 1D rope + PBD-aware
    attention_mask input (mask itself is constructed by the host).

    leap DSL `build(inputs_embeds, position_ids, attention_mask, *caches)`
    signature matches Qwen2_5_VLTextModel exactly so the driver code in
    compile() / apis/model/*.py can be reused with minimal changes.
    """

    def __init__(self, config, use_plugin: bool = False) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.use_plugin = use_plugin
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        self.config = config

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_plugin=self.use_plugin,
        )

        # layer_types is used by Qwen2_5_VLDecoderLayer's sliding-window
        # branch; we do not use sliding window, but the field is expected.
        if not hasattr(config, "layer_types"):
            config.layer_types = [
                "full_attention" for _ in range(config.num_hidden_layers)
            ]
        self.layers = nn.ModuleList([
            LocateAnythingDecoderLayer(config, i, self.use_plugin)
            for i in range(config.num_hidden_layers)
        ])

        # lm_head. Always use nn.Linear (plain float matmul).
        # DynamicQuantLinear's block_quantized_matmul with mmaAlpha=1024
        # divides int32 output by 1024 => fp16 collapses to 0.
        # Baseline Qwen2.5-VL uses nn.Linear and it works.
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False,
        )

        # Rotary precompute — cache full max_lm_tokens table.
        rope = LocateAnythingRotaryEmbedding(config)
        max_len = getattr(config, "max_lm_tokens", config.cache_len)
        cache_cos, cache_sin = rope.build_cos_sin(max_len)
        self.register_buffer("cache_cos", cache_cos, persistent=True)
        self.register_buffer("cache_sin", cache_sin, persistent=True)

        if self.use_plugin:
            self.quant_input_embeds = None  # QuantStub removed: text embed too small (mean 0.02) gets quantized to zero in decode
            self.quant_cos = QuantStub()
            self.quant_sin = QuantStub()
            self.quant_attention_mask = QuantStub()
        self.dequant = DeQuantStub()

    def get_input_embeddings(self):
        return self.embed_tokens

    def tie_lm_head_to_embeddings(self) -> None:
        """Copy embed_tokens.weight into lm_head.weight for tied case.

        Must be called AFTER load_state_dict so the checkpoint's embed
        weights are what get replicated.
        """
        if not self.tie_word_embeddings:
            return
        with torch.no_grad():
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)

    # ------------------------------------------------------------------
    # leap DSL build() — 1D rope only.
    # ------------------------------------------------------------------
    def build(self, inputs_embeds, position_ids, attention_mask, *caches):
        bs, _, num_tokens = position_ids.type.shape
        # position_ids: (bs, 1, seq) — gather cos/sin at those positions.
        if bs > 1:
            position_ids = leap.reshape(position_ids, (bs, num_tokens, 1))
            cos = leap.gather_nd(self.cache_cos, position_ids, 0)
            cos = leap.reshape(cos, (bs, 1, num_tokens, -1))
            sin = leap.gather_nd(self.cache_sin, position_ids, 0)
            sin = leap.reshape(sin, (bs, 1, num_tokens, -1))
        else:
            position_ids = leap.reshape(position_ids, (bs, -1))
            position_ids = leap.transpose(position_ids, (1, 0))
            cos = leap.gather_nd(self.cache_cos, position_ids, 0)
            cos = leap.transpose(cos, (1, 0))
            cos = leap.reshape(cos, (bs, 1, num_tokens, -1))
            sin = leap.gather_nd(self.cache_sin, position_ids, 0)
            sin = leap.transpose(sin, (1, 0))
            sin = leap.reshape(sin, (bs, 1, num_tokens, -1))

        if self.use_plugin:
            cos = self.quant_cos(cos)
            sin = self.quant_sin(sin)
            if self.quant_input_embeds is not None: inputs_embeds = self.quant_input_embeds(inputs_embeds)  # bypassed for LA
            attention_mask = self.quant_attention_mask(attention_mask)

        hidden_states = inputs_embeds
        position_embeddings = (cos, sin)

        n = len(caches) // 2
        cache_keys = caches[:n]
        cache_values = caches[n:]
        new_keys = []
        new_values = []
        for idx, layer in enumerate(self.layers):
            hidden_states, nk, nv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(nk)
            new_values.append(nv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = self.dequant(logits)
        return (logits, *new_keys, *new_values)

    # ------------------------------------------------------------------
    # PyTorch forward — for calibration passes.
    # Same signature as build() but uses eager torch ops.
    # ------------------------------------------------------------------
    def forward(self, inputs_embeds, position_ids, attention_mask, *caches):
        bs, _, num_tokens = position_ids.shape
        # 1D rope gather
        flat_pos = position_ids.view(bs, num_tokens)               # (bs, seq)
        cos = self.cache_cos[flat_pos].unsqueeze(1)                # (bs, 1, seq, dim)
        sin = self.cache_sin[flat_pos].unsqueeze(1)

        hidden_states = inputs_embeds
        n = len(caches) // 2
        cache_keys = caches[:n]
        cache_values = caches[n:]
        new_keys = []
        new_values = []
        position_embeddings = (cos, sin)
        for idx, layer in enumerate(self.layers):
            hidden_states, nk, nv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_keys=cache_keys[idx] if len(cache_keys) else None,
                cache_values=cache_values[idx] if len(cache_values) else None,
            )
            new_keys.append(nk)
            new_values.append(nv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_keys, new_values

    # ------------------------------------------------------------------
    # leap input types — identical to Qwen2_5_VLTextModel except that
    # position_ids is (bs, 1, seq) instead of (bs, 3, seq).
    # ------------------------------------------------------------------
    def get_leap_input_types_text_model(
        self, num_layers: int, seq_len: int, cache_len: int, batch_size: int = 1,
    ) -> List[leap.TensorType]:
        bs = max(batch_size, 1)
        types: List[leap.TensorType] = []
        types.append(leap.TensorType([bs, seq_len, self.hidden_size], leap.float16))
        types.append(leap.TensorType([bs, 1, seq_len], leap.int32))                # 1D rope
        types.append(leap.TensorType([bs, seq_len, cache_len], leap.float16))
        head_dim = self.hidden_size // self.config.num_attention_heads
        num_kv = self.config.num_key_value_heads
        cache_ks, cache_vs = [], []
        for _ in range(num_layers):
            cache_ks.append(leap.TensorType([bs, cache_len, num_kv, head_dim], leap.float32))
            cache_vs.append(leap.TensorType([bs, cache_len, num_kv, head_dim], leap.float32))
        types.append(cache_ks + cache_vs)
        return types

    def get_leap_input_types_decode_model(
        self, num_layers: int, seq_len: int, cache_len: int, batch_size: int = 1,
    ) -> List[leap.TensorType]:
        # Same as text_model but seq_len is decode_seq_len (default 6 for PBD).
        return self.get_leap_input_types_text_model(
            num_layers, seq_len, cache_len, batch_size,
        )
