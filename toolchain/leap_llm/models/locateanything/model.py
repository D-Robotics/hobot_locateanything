"""LocateAnything top-level wrapper — MoonViT + mlp1 + Qwen2 decoder + final norm.

Ground truth (upstream):
  /home/kangjie.xu/.cache/huggingface/modules/transformers_modules/
    LocateAnything_hyphen_3B/modeling_locateanything.py
    LocateAnything_hyphen_3B/modeling_vit.py:568 (MoonVitPretrainedModel)
    LocateAnything_hyphen_3B/modeling_qwen2.py:1144 (Qwen2Model)

Composition:
  vision_model
    ├── patch_embed              (Conv2d + Learnable2DInterpPosEmb)
    ├── encoder
    │    ├── blocks (27 x MoonViTBlockStatic)
    │    └── final_layernorm     (nn.LayerNorm)
    └── merger                   (patch_merger_2x2 + mlp1)
  mlp1  (aliased to vision_model.merger.mlp1 so state_dict keys match upstream)
  language_model
    └── model
         ├── embed_tokens        (Embedding, 152681 x 2048)
         ├── layers (36 x Qwen2DecoderLayerStatic)
         └── norm                (Qwen2RMSNormStatic)
    └── lm_head                  (tied to embed_tokens.weight in forward
                                  when config.tie_word_embeddings=True)

State-dict key layout mirrors the LocateAnything checkpoint exactly so
that `torch.load` + `load_state_dict(strict=False)` works with only a
minimal remap.

This file is P4 of M2 — it does *not* yet emit leap DSL. That happens
in the LocateAnythingApi.compile() path (P5), which walks these
sub-modules and calls their PyTorch forwards inside the calibration
pass, then calls leap.export_module on the traced graphs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from .config.locateanything_3b import LocateAnythingConfig, load_config_from_json
from .blocks.vision_patch import MoonVisionPatchEmbedStatic, Learnable2DInterpPosEmb
from .blocks.vision_block import MoonViTBlockStatic
from .blocks.vision_patch_merger import MoonViTPatchMergerAndProjectorStatic
from .blocks.text_block import Qwen2DecoderLayerStatic, Qwen2RMSNormStatic


# ---------------------------------------------------------------------------
# Vision — MoonViT encoder + patch merger + mlp1 projector to LLM hidden.
# ---------------------------------------------------------------------------
class LocateAnythingVisionEncoder(nn.Module):
    """MoonViT encoder — 27 blocks + final_layernorm."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MoonViTBlockStatic(
                num_heads=cfg.num_attention_heads,
                hidden_dim=cfg.hidden_size,
                mlp_dim=cfg.intermediate_size,
                attn_bias=True,
                mlp_bias=True,
            )
            for _ in range(cfg.num_hidden_layers)
        ])
        self.final_layernorm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, hidden_states: torch.Tensor, freqs_cos_sin: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states, freqs_cos_sin)
        return self.final_layernorm(hidden_states)


class LocateAnythingVisionModel(nn.Module):
    """Upstream: MoonVitPretrainedModel + top-level `mlp1` projector.

    We keep `patch_embed`, `encoder` at this level so state_dict keys
    starting with `vision_model.` from the checkpoint load 1:1.

    The `merger` (patch_merger + mlp1) is our compile-time addition;
    it re-uses the same mlp1 weights as the upstream top-level `mlp1`.
    """

    def __init__(
        self,
        vision_cfg,
        llm_hidden: int,
        image_h: int,
        image_w: int,
    ) -> None:
        super().__init__()
        # For upstream state_dict compatibility, we mirror the
        # MoonVitPretrainedModel structure at vision_model.*.
        self.patch_embed = MoonVisionPatchEmbedStatic(
            out_dim=vision_cfg.hidden_size,
            image_h=image_h, image_w=image_w,
            in_dim=3, patch_size=vision_cfg.patch_size,
        )
        self.encoder = LocateAnythingVisionEncoder(vision_cfg)

        # Merger + mlp1. Note: upstream stores mlp1 at the *top-level* of
        # LocateAnythingForConditionalGeneration (not under vision_model).
        # We keep it here as vision_model.merger so the compile pipeline can
        # emit the full vision graph as one sub-module. State-dict remap
        # (see remap_state_dict) rewrites `mlp1.*` -> `vision_model.merger.mlp1.*`.
        self.merger = MoonViTPatchMergerAndProjectorStatic(
            vit_hidden=vision_cfg.hidden_size,
            llm_hidden=llm_hidden,
            merge_kernel_size=tuple(vision_cfg.merge_kernel_size),
            grid_h=image_h // vision_cfg.patch_size,
            grid_w=image_w // vision_cfg.patch_size,
        )

    def forward(
        self,
        pixel_patches: torch.Tensor,
        freqs_cos_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          pixel_patches : (N_patches, C=3, patch_h, patch_w)
          freqs_cos_sin : (N_patches, head_dim/2, 2)   — pre-computed 2D RoPE

        Returns:
          (N_patches/4, llm_hidden)   — merged + projected visual embeddings
        """
        x = self.patch_embed(pixel_patches)                         # (N, vit_hidden)
        x = self.encoder(x, freqs_cos_sin)                          # (N, vit_hidden)
        x = self.merger(x)                                          # (N/4, llm_hidden)
        return x


# ---------------------------------------------------------------------------
# Language — Qwen2 decoder + optional tied lm_head.
# ---------------------------------------------------------------------------
class LocateAnythingLanguageModel(nn.Module):
    """Upstream: Qwen2Model + Qwen2ForCausalLM.

    State-dict layout after remap:
      model.embed_tokens.weight    (152681, 2048)
      model.layers.{i}.*
      model.norm.weight
      lm_head.weight               (152681, 2048; tied to embed_tokens if
                                     tie_word_embeddings=True)
    """

    def __init__(self, text_cfg) -> None:
        super().__init__()
        self.tie_word_embeddings = text_cfg.tie_word_embeddings
        self.vocab_size = text_cfg.vocab_size
        self.hidden_size = text_cfg.hidden_size

        self.embed_tokens = nn.Embedding(
            text_cfg.vocab_size, text_cfg.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayerStatic(
                hidden_size=text_cfg.hidden_size,
                intermediate_size=text_cfg.intermediate_size,
                num_heads=text_cfg.num_attention_heads,
                num_kv_heads=text_cfg.num_key_value_heads,
                rms_norm_eps=text_cfg.rms_norm_eps,
                qkv_bias=True,
                mlp_bias=False,
            )
            for _ in range(text_cfg.num_hidden_layers)
        ])
        self.norm = Qwen2RMSNormStatic(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)

        # lm_head: only allocate independent weights when NOT tied.
        # When tied, forward() computes logits = hidden @ embed_tokens.weight.T
        # directly (see report pit #1: avoids 92MB duplicate weight tensor).
        if not text_cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(
                text_cfg.hidden_size, text_cfg.vocab_size, bias=False,
            )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tied lm_head via matmul against embed_tokens.weight."""
        if self.tie_word_embeddings:
            # (bs, seq, hidden) @ (vocab, hidden).T  ->  (bs, seq, vocab)
            return hidden_states @ self.embed_tokens.weight.T
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Top-level wrapper — build() loads checkpoint + remaps state_dict.
# ---------------------------------------------------------------------------
def remap_state_dict(raw_sd: dict) -> dict:
    """Rewrite upstream key names to our internal structure.

    Rules:
      vision_model.<x>            -> vision_model.<x>                 (no change)
      mlp1.<x>                    -> vision_model.merger.mlp1.<x>     (moved)
      language_model.model.<x>    -> language_model.<x>               (drop `.model.` mid-level)
      language_model.lm_head.<x>  -> language_model.lm_head.<x>       (kept only when NOT tied;
                                                                       forward computes tied version
                                                                       from embed_tokens anyway)
    """
    remapped = {}
    for k, v in raw_sd.items():
        if k.startswith("vision_model."):
            remapped[k] = v
        elif k.startswith("mlp1."):
            remapped["vision_model.merger.mlp1." + k[len("mlp1."):]] = v
        elif k.startswith("language_model.model."):
            remapped["language_model." + k[len("language_model.model."):]] = v
        elif k.startswith("language_model.lm_head."):
            # Keep this key: LocateAnythingLanguageModel.compute_logits handles
            # the tied case at runtime; it never reads lm_head.weight when tied
            # so even if we keep it here PyTorch will just ignore it.
            # (We drop it to avoid a 92MB duplicate — see report pit #1.)
            continue
        else:
            # e.g. `image_token_index` etc. are HF config properties, not
            # state_dict entries — should not appear in raw_sd from safetensors.
            remapped[k] = v
    return remapped


class LocateAnything(nn.Module):
    """Top-level compile-time model.

    Not the same as LocateAnythingForConditionalGeneration in upstream —
    that class also holds generate() logic + tokenizer glue. We only need
    the forward-composable module tree for the leap compile pipeline.
    """

    def __init__(self, cfg: LocateAnythingConfig) -> None:
        super().__init__()
        self.config = cfg
        self.vision_model = LocateAnythingVisionModel(
            vision_cfg=cfg.vision_config,
            llm_hidden=cfg.text_config.hidden_size,
            image_h=cfg.vision_config.image_height,
            image_w=cfg.vision_config.image_width,
        )
        self.language_model = LocateAnythingLanguageModel(cfg.text_config)

    def get_vision_model(self) -> LocateAnythingVisionModel:
        return self.vision_model

    def get_text_model(self) -> LocateAnythingLanguageModel:
        return self.language_model

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def get_config(self) -> LocateAnythingConfig:
        return self.config

    # ---- Loading ----------------------------------------------------------
    @staticmethod
    def build(
        model_dir: str,
        image_height: int = 448,
        image_width: int = 448,
        decode_seq_len: int = 6,
        chunk_size: int = 256,
        cache_len: int = 1024,
        batch_size: int = 1,
        w_bits: int = 4,
    ) -> "LocateAnything":
        """Load LocateAnything checkpoint at `model_dir` and instantiate the
        wrapper. `model_dir` must contain `config.json` and safetensors shards.
        """
        cfg_path = os.path.join(model_dir, "config.json")
        assert os.path.exists(cfg_path), f"config.json missing at {model_dir}"
        cfg = load_config_from_json(cfg_path)

        # Overwrite compile-time knobs.
        cfg.vision_config.image_height = image_height
        cfg.vision_config.image_width = image_width
        cfg.text_config.decode_seq_len = decode_seq_len
        cfg.text_config.prefill_seq_len = chunk_size
        cfg.text_config.cache_len = cache_len
        cfg.text_config.batch_size = batch_size
        cfg.text_config.w_bits = w_bits

        model = LocateAnything(cfg)

        # Load state_dict. Prefer safetensors; fall back to .pth if the wrapper
        # already produced one (baseline dry-run flow used torch.save).
        from safetensors import safe_open
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            file_map = index["weight_map"]
            files = sorted(set(file_map.values()))
            raw_sd = {}
            for fname in files:
                fpath = os.path.join(model_dir, fname)
                with safe_open(fpath, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        raw_sd[k] = f.get_tensor(k)
        else:
            # Fallback: single shard
            single = os.path.join(model_dir, "model.safetensors")
            assert os.path.exists(single), f"no safetensors at {model_dir}"
            with safe_open(single, framework="pt", device="cpu") as f:
                raw_sd = {k: f.get_tensor(k) for k in f.keys()}

        remapped = remap_state_dict(raw_sd)

        # Bake pos_emb from the Learnable2DInterpPosEmb weight (a checkpoint
        # entry) into the static buffer. First find that weight in remapped.
        pos_emb_key = "vision_model.patch_embed.pos_emb.weight"
        if pos_emb_key in remapped:
            pos_emb_weight = remapped.pop(pos_emb_key)              # (64, 64, 1152)
            tmp = Learnable2DInterpPosEmb(
                height=pos_emb_weight.shape[0],
                width=pos_emb_weight.shape[1],
                dim=pos_emb_weight.shape[2],
            )
            with torch.no_grad():
                tmp.weight.copy_(pos_emb_weight)
            model.vision_model.patch_embed.load_from_learnable(tmp)

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # Reporting: missing usually contains just the static buffer +
        # (when tied) lm_head; unexpected should be empty.
        return model, missing, unexpected
