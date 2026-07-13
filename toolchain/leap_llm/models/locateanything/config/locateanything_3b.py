"""Dataclass-based config for LocateAnything-3B.

Loaded from the checkpoint `config.json` via `dataclass_from_dict`.
Kept intentionally flat and explicit — no nested-schema hacks like the
Qwen2.5-VL branch does. Fields mirror the on-disk config exactly so that
`json.load(config.json)` → `dataclass_from_dict(LocateAnythingConfig, ...)`
round-trips without loss.

Reference: /home/kangjie.xu/oe_locateanything/eagle/Embodied/LocateAnything-3B/config.json
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Helper — same shape as `qwen2_5_vl/model.py:dataclass_from_dict` but broken
# out so all locateanything files can import it.
# ---------------------------------------------------------------------------
def dataclass_from_dict(cls, dikt):
    """Recursively instantiate a @dataclass `cls` from a plain dict `dikt`.

    Unknown keys are ignored (config.json may carry HF-only metadata like
    `_attn_implementation_autoset`, `transformers_version`).
    """
    if not is_dataclass(cls):
        return dikt

    init_kwargs = {}
    known = {f.name for f in fields(cls)}
    for f in fields(cls):
        raw = dikt.get(f.name, None)
        if raw is None:
            continue
        if is_dataclass(f.type) and isinstance(raw, dict):
            init_kwargs[f.name] = dataclass_from_dict(f.type, raw)
        else:
            init_kwargs[f.name] = raw
    return cls(**init_kwargs)


# ---------------------------------------------------------------------------
# Vision — MoonViT-SO-400M
# ---------------------------------------------------------------------------
@dataclass
class MoonViTConfig:
    # Mirror of config.json → vision_config
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    patch_size: int = 14
    merge_kernel_size: List[int] = field(default_factory=lambda: [2, 2])

    # 2D learnable pos-emb base grid (Learnable2DInterpPosEmb in modeling_vit.py:224)
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64

    # Runtime-shape metadata (set by LocateAnythingApi from CLI before build())
    image_height: int = 448
    image_width: int = 448

    # Attention mask sentinel used inside compiled ViT (matches Qwen2.5-VL branch)
    mask_min_value: float = -32768.0

    # Optional (present in some HF exports); we do not need them for compile.
    torch_dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Text — Qwen2 decoder with PBD scaffolding
# ---------------------------------------------------------------------------
@dataclass
class Qwen2PBDTextConfig:
    # Mirror of config.json → text_config
    vocab_size: int = 152681
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"

    # PBD-specific — the reason we cannot reuse qwen2_5-vl-3b builder
    block_size: int = 6              # PBD unit = 6 tokens (box block)
    causal_attn: bool = False        # False -> diagonal block bidirectional mask

    # Sentinel tokens duplicated inside text_config in the checkpoint
    text_mask_token_id: int = 151676
    null_token_id: int = 152678
    switch_token_id: int = 152679

    # Optional / rarely-touched
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    max_window_layers: int = 70
    sliding_window: int = 32768
    use_sliding_window: bool = False

    # Compile-time knobs (overwritten by LocateAnythingApi from CLI flags)
    prefill_seq_len: int = 256       # from --chunk_size
    cache_len: int = 4096            # from --cache_len
    decode_seq_len: int = 6          # default matches block_size — the PBD path
    batch_size: int = 1
    w_bits: int = 4
    has_scale: bool = False


# ---------------------------------------------------------------------------
# Top-level model config
# ---------------------------------------------------------------------------
@dataclass
class LocateAnythingConfig:
    vision_config: MoonViTConfig = field(default_factory=MoonViTConfig)
    text_config: Qwen2PBDTextConfig = field(default_factory=Qwen2PBDTextConfig)

    # Cross-modal wiring
    image_token_index: int = 151665
    mlp_connector_layers: int = 2

    # Special tokens exposed at top level (see config/special_tokens.py for the
    # canonical id table)
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    none_token_id: int = 4064

    # HF hints — kept for reference, unused by the compile pipeline
    architectures: List[str] = field(default_factory=lambda: [
        "LocateAnythingForConditionalGeneration"
    ])
    model_type: str = "locateanything"
    torch_dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Public loader — thin wrapper that reads config.json and returns the dataclass.
# ---------------------------------------------------------------------------
def load_config_from_json(config_json_path: str) -> LocateAnythingConfig:
    """Read `config.json` at the given path and hydrate LocateAnythingConfig.

    Unlike the qwen2_5_vl branch which flattens vision/text into a single
    dataclass with `config.get("vision_config", config)` fallback trickery,
    LocateAnything's config.json has a clean nested schema so we consume it
    literally.
    """
    import json
    with open(config_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    return dataclass_from_dict(LocateAnythingConfig, raw)
