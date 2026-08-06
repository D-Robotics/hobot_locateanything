"""LocateAnything Language model implemented with the Leap DSL.

Model-specific behavior:
  1. Vanilla 1D rope only (no mrope 3-way split). position_ids shape is
     always (bs, 1, seq); the mrope branch is deleted rather than gated.
  2. tie_word_embeddings=True is respected — lm_head is computed as
     matmul against embed_tokens.weight, no independent lm_head allocation
     to avoid a duplicate output projection.
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
from .bpu_sampler import PBD_ROWS, VOCAB_SIZE, build_pbd_sampler


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
        # The export driver sets this for one graph at a time.  Keeping the
        # contract here avoids duplicating stage-specific slicing in wrappers.
        self._export_stage: str | None = None

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

        # The Leap export path requires a Module with build() support. The
        # hidden-domain rotation keeps DynamicQuantLinear's activation range
        # stable; nn.Linear cannot consume an HBDK OpResult during BC export.
        if self.use_plugin:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )
        else:
            self.lm_head = DynamicQuantLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                w_bits=config.lm_head_w_bits,
                has_scale=config.has_scale,
            )

        # Rotary precompute — cache full max_lm_tokens table.
        rope = LocateAnythingRotaryEmbedding(config)
        max_len = getattr(config, "max_lm_tokens", config.cache_len)
        cache_cos, cache_sin = rope.build_cos_sin(max_len)
        self.register_buffer("cache_cos", cache_cos, persistent=False)
        self.register_buffer("cache_sin", cache_sin, persistent=False)

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

    def set_export_stage(self, stage: str | None) -> None:
        """Select the static output contract used by the next BC export."""
        self._export_stage = stage

    def _export_output_range(self, num_tokens: int) -> tuple[int, int]:
        """Return the hidden-state rows that require a vocabulary projection."""
        stage = self._export_stage or ""
        if stage == "prefill":
            return num_tokens - 1, num_tokens
        if stage.startswith("decode_pbd_q"):
            return num_tokens - self.config.block_size, num_tokens
        if stage.startswith("decode_ar_q"):
            return num_tokens - 1, num_tokens
        return 0, num_tokens

    def _export_cache_rows(self, cache: list, num_tokens: int) -> list:
        """Drop PBD MASK rows that are never committed to the KV cache."""
        stage = self._export_stage or ""
        if not stage.startswith("decode_pbd_q"):
            return cache
        prefix_len = int(stage.rsplit("q", 1)[1]) - self.config.block_size
        if prefix_len <= 0:
            return cache
        head_dim = self.hidden_size // self.config.num_attention_heads
        num_kv = self.config.num_key_value_heads
        end = [1, prefix_len, num_kv, head_dim]
        return [
            leap.slice(value, [0, 0, 0, 0], end, [1, 1, 1, 1])
            for value in cache
        ]

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
            cos = leap.reshape(cos, (bs, 1, num_tokens, -1))
            sin = leap.gather_nd(self.cache_sin, position_ids, 0)
            sin = leap.reshape(sin, (bs, 1, num_tokens, -1))

        if self.use_plugin:
            cos = self.quant_cos(cos)
            sin = self.quant_sin(sin)
            if self.quant_input_embeds is not None: inputs_embeds = self.quant_input_embeds(inputs_embeds)  # bypassed for LA
            attention_mask = self.quant_attention_mask(attention_mask)

        hidden_states = inputs_embeds
        position_embeddings = (cos, sin)

        n = self.config.num_hidden_layers
        cache_keys = caches[:n]
        cache_values = caches[n:2 * n]
        sampling_inputs = caches[2 * n:]
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
        start, end = self._export_output_range(num_tokens)
        if start < 0 or end > num_tokens or start >= end:
            raise ValueError(
                f"invalid {self._export_stage!r} output range "
                f"[{start}, {end}) for q_len={num_tokens}"
            )
        if start != 0 or end != num_tokens:
            hidden_states = leap.slice(
                hidden_states,
                [0, start, 0],
                [bs, end, self.hidden_size],
                [1, 1, 1],
            )
        logits = self.lm_head(hidden_states)
        logits = self.dequant(logits)
        new_keys = self._export_cache_rows(new_keys, num_tokens)
        new_values = self._export_cache_rows(new_values, num_tokens)
        stage = self._export_stage or ""
        use_bpu_sampling = (
            getattr(self.config, "sampling_backend", "bpu") == "bpu"
            and (stage == "decode" or stage.startswith("decode_pbd_q"))
        )
        if not use_bpu_sampling:
            return (logits, *new_keys, *new_values)
        if len(sampling_inputs) != 2:
            raise ValueError(f"{stage} requires history mask and random values")
        compact = build_pbd_sampler(
            logits, sampling_inputs[0], sampling_inputs[1],
            temperature=self.config.sampling_temperature,
            top_p=self.config.sampling_top_p,
            repetition_penalty=self.config.sampling_repetition_penalty,
        )
        return (*compact, *new_keys, *new_values)

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
        *, pbd: bool = False,
    ) -> List[leap.TensorType]:
        # Same as text_model but seq_len is decode_seq_len (default 6 for PBD).
        inputs = self.get_leap_input_types_text_model(
            num_layers, seq_len, cache_len, batch_size,
        )
        if pbd and getattr(self.config, "sampling_backend", "bpu") == "bpu":
            inputs.extend([
                leap.TensorType([1, PBD_ROWS, VOCAB_SIZE], leap.uint8),
                leap.TensorType([1, PBD_ROWS, 1], leap.float16),
            ])
        return inputs
