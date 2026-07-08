"""Masks, position ids, and config helpers for SmolVLA quantization."""

from __future__ import annotations

import json
import math
from pathlib import Path

from leap_llm.models.smolvla.blocks.configuration_smolvlm import SmolVLAPolicyConfig
from leap_llm.models.smolvla.weight_mapper import (
    count_text_layers,
    infer_expert_hidden_size,
    infer_text_hidden_size,
    infer_vision_tokens_per_image,
    load_full_state_dict,
)


def load_policy_config(
    input_model_path: str | Path,
    config_path: str | Path | None = None,
) -> SmolVLAPolicyConfig:
    root = Path(input_model_path)
    if config_path:
        cfg_path = Path(config_path)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
    else:
        cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"SmolVLA policy config not found: {cfg_path}")
    cfg = SmolVLAPolicyConfig.from_json_path(cfg_path)

    weights = root / "model.safetensors"
    if weights.is_file():
        sd = load_full_state_dict(root)
        th = infer_text_hidden_size(sd)
        eh = infer_expert_hidden_size(sd)
        nl = count_text_layers(sd)
        vt = infer_vision_tokens_per_image(
            sd,
            vision_hidden_size=cfg.vision_hidden_size,
            image_height=cfg.image_height,
            vision_patch_size=cfg.vision_patch_size,
        )
        if th:
            cfg.text_hidden_size = th
        if eh:
            cfg.expert_width_multiplier = eh / cfg.text_hidden_size
        if nl:
            cfg.text_num_hidden_layers = nl
            if cfg.num_vlm_layers <= 0:
                cfg.num_vlm_layers = min(16, nl)
        if vt:
            cfg.vision_tokens_num = vt
    return cfg


def make_att_2d_masks(pad_masks, att_masks):
    """LeRobot SmolVLA block attention (big_vision style). True = can attend."""
    import torch

    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}")

    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def att_2d_to_float(att_2d, neg_value: float = -32767.0, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float16
    if att_2d.ndim == 3:
        att_2d = att_2d.unsqueeze(1)
    zeros = torch.zeros((), dtype=dtype, device=att_2d.device)
    neg = torch.full((), neg_value, dtype=dtype, device=att_2d.device)
    return torch.where(att_2d, zeros, neg)


def generate_prefix_position_ids(prefix_pad_masks):
    """RoPE position ids matching LeRobot embed_prefix / HF forward."""
    import torch

    return (torch.cumsum(prefix_pad_masks, dim=1) - 1).to(torch.int32)


def build_prefix_pad_att_masks(
    vision_len: int,
    valid_lang_len: int,
    total_lang_len: int = 48,
    state_len: int = 1,
    device: str = "cpu",
):
    """Prefix pad/att masks matching LeRobot embed_prefix (image + lang + state)."""
    import torch

    vision_pad = torch.ones(1, vision_len, dtype=torch.bool, device=device)
    lang_pad = torch.zeros(1, total_lang_len, dtype=torch.bool, device=device)
    if valid_lang_len > 0:
        lang_pad[:, :valid_lang_len] = True
    state_pad = torch.ones(1, state_len, dtype=torch.bool, device=device)
    pad_masks = torch.cat([vision_pad, lang_pad, state_pad], dim=1)

    att_list = [0] * vision_len + [0] * total_lang_len + [1] * state_len
    att_masks = torch.tensor(att_list, dtype=torch.int32, device=device).unsqueeze(0)
    return pad_masks, att_masks


def build_vlm_prefix_mask(
    vision_len: int,
    valid_lang_len: int,
    neg_value: float = -32767.0,
    total_lang_len: int = 48,
    device: str = "cpu",
    dtype=None,
):
    """Float attention mask for VLM prefix subgraph compile/calib."""
    import torch

    if dtype is None:
        dtype = torch.float16
    pad_masks, att_masks = build_prefix_pad_att_masks(
        vision_len, valid_lang_len, total_lang_len=total_lang_len, device=device
    )
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    return att_2d_to_float(att_2d, neg_value=neg_value, dtype=dtype)


def build_suffix_pad_att_masks(action_len: int, device: str = "cpu"):
    """Suffix masks for action-only embed_suffix."""
    import torch

    pad_masks = torch.ones(1, action_len, dtype=torch.bool, device=device)
    att_masks = torch.ones(1, action_len, dtype=torch.int32, device=device)
    return pad_masks, att_masks


def build_denoise_attention_mask(
    prefix_pad_masks,
    action_len: int,
    neg_value: float = -32767.0,
    device: str = "cpu",
    dtype=None,
):
    """Expert denoise mask: suffix rows attend to prefix KV + suffix block-attn."""
    import torch

    if dtype is None:
        dtype = torch.float16
    bsize, prefix_len = prefix_pad_masks.shape
    suffix_pad, suffix_att = build_suffix_pad_att_masks(action_len, device=device)
    suffix_len = action_len

    prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
    suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
    full_bool = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
    return att_2d_to_float(full_bool.unsqueeze(1), neg_value=neg_value, dtype=dtype)


def build_action_expert_mask(
    prefix_pad_masks,
    action_len: int = 50,
    neg_value: float = -32767.0,
    device: str = "cpu",
    dtype=None,
):
    """Alias for denoise-step attention mask (LeRobot denoise_step layout)."""
    return build_denoise_attention_mask(
        prefix_pad_masks,
        action_len=action_len,
        neg_value=neg_value,
        device=device,
        dtype=dtype,
    )


def generate_denoise_position_ids(prefix_pad_masks, action_len: int, device: str = "cpu"):
    """Position ids for suffix tokens during denoise (LeRobot denoise_step)."""
    import torch

    prefix_offsets = prefix_pad_masks.sum(dim=-1, keepdim=True)
    suffix_pad = torch.ones(1, action_len, dtype=torch.int32, device=device)
    position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
    return position_ids.to(torch.int32)


def generate_action_position_ids(
    prefix_token_num: int,
    valid_prompt_token: int,
    suffix_rows: int = 50,
    device: str = "cuda",
    prefix_pad_masks=None,
):
    """Backward-compatible wrapper; prefer prefix_pad_masks when available."""
    import torch

    if prefix_pad_masks is not None:
        return generate_denoise_position_ids(prefix_pad_masks, suffix_rows, device=device)

    start = prefix_token_num + valid_prompt_token + 1
    pos_ids = torch.arange(start, start + suffix_rows, dtype=torch.int32, device=device)
    return pos_ids.view(1, suffix_rows)


def create_sinusoidal_pos_embedding(
    time,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
    dtype=None,
):
    """LeRobot flow-matching timestep embedding."""
    import torch

    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time tensor must be shape (batch_size,)")

    if dtype is None:
        dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time.to(dtype)[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def resize_with_pad(img, width, height, pad_value=-1.0):
    """LeRobot resize_with_pad for BCHW float tensors."""
    import torch.nn.functional as F

    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    _, _, cur_height, cur_width = img.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def prefix_sequence_len(policy_cfg: SmolVLAPolicyConfig) -> int:
    """Total prefix sequence length: vision(after connector) + lang + state."""
    vision_len = policy_cfg.vision_tokens_num * policy_cfg.num_images
    return vision_len + policy_cfg.tokenizer_max_length + 1


def dump_config_template(path: str | Path) -> None:
    """Write a minimal config.json template for custom checkpoints."""
    template = {
        "type": "smolvla",
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "num_vlm_layers": 16,
        "attention_mode": "cross_attn",
        "expert_width_multiplier": 0.75,
        "chunk_size": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "tokenizer_max_length": 48,
        "prefix_length": 48,
        "num_steps": 10,
        "min_period": 4e-3,
        "max_period": 4.0,
        "resize_imgs_with_padding": [512, 512],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
