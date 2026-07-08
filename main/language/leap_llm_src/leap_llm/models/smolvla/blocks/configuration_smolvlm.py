"""SmolVLA policy / SmolVLM configuration loaded from LeRobot checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SmolVLAPolicyConfig:
    """Runtime config for SmolVLA quantization (from checkpoint config.json)."""

    # Policy I/O
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48
    prefix_length: int = 48
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    # Architecture
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    num_vlm_layers: int = 16
    num_expert_layers: int = 0
    attention_mode: str = "cross_attn"
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    num_images: int = 3

    # SmolVLM text (defaults for SmolVLM2-500M; overridden from weight shapes when possible)
    text_hidden_size: int = 960
    text_intermediate_size: int = 2560
    text_num_hidden_layers: int = 24
    text_num_attention_heads: int = 15
    text_num_key_value_heads: int = 5
    text_head_dim: int = 64
    text_vocab_size: int = 49280
    text_rms_norm_eps: float = 1e-5
    text_rope_theta: float = 100000.0
    # LeRobot smolvlm_with_expert.apply_rope uses max_wavelength=10000 (not text_config.rope_theta).
    smolvla_rope_theta: float = 10000.0

    # SmolVLM vision (SigLIP-B/16 class)
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12
    vision_image_size: int = 512
    vision_patch_size: int = 16

    # Derived
    vision_tokens_num: int = 64
    image_height: int = 512
    image_width: int = 512

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def expert_hidden_size(self) -> int:
        return int(self.text_hidden_size * self.expert_width_multiplier)

    @property
    def expert_intermediate_size(self) -> int:
        """Match get_intermediate_size from smolvlm_with_expert.py (multiple_of=256)."""
        hidden = self.expert_hidden_size
        hidden_dim = int(2 * hidden / 3)
        hidden_dim = int(4 * hidden_dim)
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        return hidden_dim

    @property
    def active_vlm_layers(self) -> int:
        return self.num_vlm_layers if self.num_vlm_layers > 0 else self.text_num_hidden_layers

    @property
    def active_expert_layers(self) -> int:
        if self.num_expert_layers > 0:
            return self.num_expert_layers
        return self.active_vlm_layers

    @classmethod
    def from_json_path(cls, path: str | Path) -> SmolVLAPolicyConfig:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SmolVLAPolicyConfig:
        h, w = raw.get("resize_imgs_with_padding", [512, 512])
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            img_h, img_w = int(h[0]), int(h[1])
        else:
            img_h = img_w = int(raw.get("image_height", 512))

        num_images = 3
        for key in ("input_features", "features"):
            feats = raw.get(key, {})
            if isinstance(feats, dict):
                cam_keys = [k for k in feats if "image" in k.lower()]
                if cam_keys:
                    num_images = len(cam_keys)
                    break

        cfg = cls(
            chunk_size=int(raw.get("chunk_size", 50)),
            n_action_steps=int(raw.get("n_action_steps", 50)),
            max_state_dim=int(raw.get("max_state_dim", 32)),
            max_action_dim=int(raw.get("max_action_dim", 32)),
            resize_imgs_with_padding=(img_h, img_w),
            tokenizer_max_length=int(raw.get("tokenizer_max_length", 48)),
            prefix_length=int(raw.get("prefix_length", 48)),
            num_steps=int(raw.get("num_steps", 10)),
            min_period=float(raw.get("min_period", 4e-3)),
            max_period=float(raw.get("max_period", 4.0)),
            vlm_model_name=str(
                raw.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            ),
            num_vlm_layers=int(raw.get("num_vlm_layers", 16)),
            num_expert_layers=int(raw.get("num_expert_layers", 0)),
            attention_mode=str(raw.get("attention_mode", "cross_attn")),
            self_attn_every_n_layers=int(raw.get("self_attn_every_n_layers", 2)),
            expert_width_multiplier=float(raw.get("expert_width_multiplier", 0.75)),
            num_images=num_images,
            image_height=img_h,
            image_width=img_w,
            vision_tokens_num=int(
                raw.get("vision_tokens_num", 64)
            ),
            extra={k: v for k, v in raw.items() if k not in cls.__dataclass_fields__},
        )
        return cfg


@dataclass
class SmolVLMVisionConfig:
    """SigLIP vision tower config for SmolVLM2 (SmolVLA vision subgraph)."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 512
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    projection_dim: int = 960
    visual_token_num: int = 64

    @classmethod
    def from_policy_config(cls, policy_cfg: SmolVLAPolicyConfig) -> SmolVLMVisionConfig:
        return cls(
            hidden_size=policy_cfg.vision_hidden_size,
            intermediate_size=policy_cfg.vision_intermediate_size,
            num_hidden_layers=policy_cfg.vision_num_hidden_layers,
            num_attention_heads=policy_cfg.vision_num_attention_heads,
            image_size=policy_cfg.image_height,
            patch_size=policy_cfg.vision_patch_size,
            projection_dim=policy_cfg.text_hidden_size,
            visual_token_num=policy_cfg.vision_tokens_num,
        )


class SmolLM2Config:
    """SmolLM2 text decoder / action expert config for leap graph modules."""

    model_type = "smollm2"

    def __init__(
        self,
        vocab_size: int = 49280,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        head_dim: int = 64,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 100000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int = 0,
        vision_tokens_num: int = 192,
        is_cross_attn_expert: bool = False,
        vlm_kv_dim: int | None = None,
        vlm_kv_in_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.vision_tokens_num = vision_tokens_num
        self.is_cross_attn_expert = is_cross_attn_expert
        self.vlm_kv_dim = vlm_kv_dim or hidden_size
        self.vlm_kv_in_dim = vlm_kv_in_dim or (num_key_value_heads * head_dim)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_policy(
        cls, policy_cfg: SmolVLAPolicyConfig, *, for_expert: bool = False
    ) -> SmolLM2Config:
        if for_expert:
            return cls(
                vocab_size=policy_cfg.text_vocab_size,
                hidden_size=policy_cfg.expert_hidden_size,
                intermediate_size=policy_cfg.expert_intermediate_size,
                num_hidden_layers=policy_cfg.active_expert_layers,
                num_attention_heads=policy_cfg.text_num_attention_heads,
                num_key_value_heads=policy_cfg.text_num_key_value_heads,
                head_dim=policy_cfg.text_head_dim,
                max_position_embeddings=policy_cfg.text_num_hidden_layers * 512,
                rms_norm_eps=policy_cfg.text_rms_norm_eps,
                rope_theta=policy_cfg.smolvla_rope_theta,
                vision_tokens_num=policy_cfg.vision_tokens_num,
                is_cross_attn_expert=True,
                vlm_kv_dim=policy_cfg.text_hidden_size,
                vlm_kv_in_dim=(
                    policy_cfg.text_num_key_value_heads * policy_cfg.text_head_dim
                ),
            )
        return cls(
            vocab_size=policy_cfg.text_vocab_size,
            hidden_size=policy_cfg.text_hidden_size,
            intermediate_size=policy_cfg.text_intermediate_size,
            num_hidden_layers=policy_cfg.active_vlm_layers,
            num_attention_heads=policy_cfg.text_num_attention_heads,
            num_key_value_heads=policy_cfg.text_num_key_value_heads,
            head_dim=policy_cfg.text_head_dim,
            max_position_embeddings=policy_cfg.text_num_hidden_layers * 512,
            rms_norm_eps=policy_cfg.text_rms_norm_eps,
            rope_theta=policy_cfg.smolvla_rope_theta,
            vision_tokens_num=policy_cfg.vision_tokens_num,
            is_cross_attn_expert=False,
        )
