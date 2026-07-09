"""Map LeRobot SmolVLA safetensors keys to leap_llm submodule state dicts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from safetensors.torch import load_file

# Canonical prefixes inside model.safetensors (LeRobot policy checkpoint).
VLM_ROOT = "vlm_with_expert.vlm"
VLM_MODEL = f"{VLM_ROOT}.model"
VISION_PREFIX = f"{VLM_MODEL}.vision_model."
CONNECTOR_PREFIX = f"{VLM_MODEL}.connector."
TEXT_PREFIX = f"{VLM_MODEL}.text_model."
EXPERT_PREFIX = "vlm_with_expert.lm_expert."

# state_proj lives at model top-level in the checkpoint, loaded into the VLM prefix graph
PREFIX_PROJS = ("state_proj",)
# action projections live at model top-level, loaded into the expert graph
EXPERT_PROJS = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


def _strip_model_prefix(key: str) -> str:
    """Remove leading 'model.' from LeRobot policy checkpoint keys."""
    if key.startswith("model."):
        return key[len("model."):]
    return key


def _normalize_text_rel_key(rel: str) -> str:
    """Strip optional 'model.' wrapper from text_model relative keys."""
    if rel.startswith("model."):
        return rel[len("model."):]
    return rel


def _text_layer_index(key: str) -> int | None:
    if not key.startswith(TEXT_PREFIX):
        return None
    rel = _normalize_text_rel_key(key[len(TEXT_PREFIX):])
    if not rel.startswith("layers."):
        return None
    idx = rel.split("layers.")[1].split(".")[0]
    return int(idx) if idx.isdigit() else None


def load_full_state_dict(model_path: str | Path) -> dict[str, object]:
    path = Path(model_path)
    if path.is_dir():
        path = path / "model.safetensors"
    state = load_file(str(path))
    return {_strip_model_prefix(k): v for k, v in state.items()}


def filter_by_prefix(
    state_dict: dict[str, object], prefix: str, new_prefix: str = ""
) -> dict[str, object]:
    return {
        new_prefix + k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


def vision_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    return filter_by_prefix(state_dict, VISION_PREFIX)


def connector_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    """Map connector.* keys to SmolVLMConnector state dict.

    Checkpoint key after CONNECTOR_PREFIX: modality_projection.proj.{weight,bias}
    Our SmolVLMConnector: self.modality_projection.proj.{weight,bias}  (same path)
    """
    return filter_by_prefix(state_dict, CONNECTOR_PREFIX)


def text_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    """SmolLM2 text stack + state_proj for SmolLM2PrefixModel.

    Checkpoint keys (after stripping model.):
      vlm_with_expert.vlm.model.text_model.embed_tokens.weight
      vlm_with_expert.vlm.model.text_model.layers.N.*
      vlm_with_expert.vlm.model.text_model.norm.weight
    Our model (SmolLM2PrefixModel) keys:
      embed_tokens.weight, layers.N.*, norm.weight, state_proj.{weight,bias}
    """
    out: dict[str, object] = {}
    for k, v in state_dict.items():
        if not k.startswith(TEXT_PREFIX):
            continue
        rel = _normalize_text_rel_key(k[len(TEXT_PREFIX):])
        out[rel] = v

    # Tie embed_tokens to lm_head when embed_tokens absent
    lm_head_key = f"{VLM_ROOT}.lm_head.weight"
    embed_key = "embed_tokens.weight"
    if embed_key not in out and lm_head_key in state_dict:
        out[embed_key] = state_dict[lm_head_key]

    # state_proj is stored at top-level in the checkpoint
    for name in PREFIX_PROJS:
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state_dict:
                out[key] = state_dict[key]
    return out


def expert_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    """SmolVLAExpert state dict.

    Checkpoint keys (after stripping model.):
      vlm_with_expert.lm_expert.layers.N.*
      vlm_with_expert.lm_expert.norm.weight
    Our SmolVLAExpert wraps SmolVLAExpertCore as self.model, so keys become:
      model.layers.N.*, model.norm.weight
    Top-level projections (action_in/out, time_mlp) are at checkpoint top-level.
    """
    out: dict[str, object] = {}
    for k, v in state_dict.items():
        if not k.startswith(EXPERT_PREFIX):
            continue
        rel = k[len(EXPERT_PREFIX):]
        # Expert core is stored under SmolVLAExpert.model
        if rel.startswith("layers.") or rel.startswith("norm."):
            rel = f"model.{rel}"
        out[rel] = v

    for name in EXPERT_PROJS:
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key in state_dict:
                out[key] = state_dict[key]
    return out


def infer_text_hidden_size(state_dict: dict[str, object]) -> int | None:
    for suffix in (
        "layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ):
        w = state_dict.get(f"{TEXT_PREFIX}{suffix}")
        if w is not None:
            return int(w.shape[1])
    return None


def infer_expert_hidden_size(state_dict: dict[str, object]) -> int | None:
    w = state_dict.get(f"{EXPERT_PREFIX}layers.0.self_attn.q_proj.weight")
    if w is not None:
        return int(w.shape[1])
    return None


def infer_vision_tokens_per_image(
    state_dict: dict[str, object],
    vision_hidden_size: int = 768,
    image_height: int = 512,
    vision_patch_size: int = 16,
) -> int | None:
    """Infer connector output tokens per image from connector weight shape."""
    connector_key = f"{CONNECTOR_PREFIX}modality_projection.proj.weight"
    w = state_dict.get(connector_key)
    if w is None:
        return None
    pool_factor = w.shape[1] // vision_hidden_size
    raw_patches = (image_height // vision_patch_size) ** 2
    return raw_patches // pool_factor


def count_text_layers(state_dict: dict[str, object]) -> int:
    layers = set()
    for k in state_dict:
        idx = _text_layer_index(k)
        if idx is not None:
            layers.add(idx)
    return len(layers) if layers else 0


def list_key_prefixes(state_dict: dict[str, object], max_lines: int = 40) -> Iterable[str]:
    seen = set()
    for k in sorted(state_dict.keys()):
        parts = k.split(".")
        prefix = ".".join(parts[:4]) if len(parts) >= 4 else k
        if prefix not in seen:
            seen.add(prefix)
            yield prefix
            if len(seen) >= max_lines:
                break
