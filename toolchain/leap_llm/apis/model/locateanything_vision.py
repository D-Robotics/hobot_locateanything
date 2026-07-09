"""LocateAnything vision-only compile Api (M3-α).

Mirrors locateanything_language.py but for the MoonViT vision tower.

Produces:
  LocateAnything-3B_vision_448x448_w8_nash-p_corenum_*.hbm
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Optional

import torch
from hbdk4.compiler import leap
from safetensors import safe_open

from leap_llm.nn.utils import standard_vit_name

from leap_llm.models.locateanything.config.locateanything_3b import (
    load_config_from_json,
)
from leap_llm.models.locateanything.vision_model_leap import LocateAnythingVisionModel


def remap_vision_state_dict(raw_sd: dict, num_patches: int, patch_size: int,
                            in_channels: int, hidden_size: int) -> dict:
    """Extract vision_model.* + mlp1.* and re-key to our layout.

    Transformations:
      vision_model.encoder.blocks.i.{norm0,norm1,wqkv,wo,mlp.*} -> blocks.i.{...}
      vision_model.encoder.final_layernorm.*                    -> final_layernorm.*
      vision_model.patch_embed.proj.weight (Conv2d)             -> patch_embed.proj.weight (Linear, reshaped)
      vision_model.patch_embed.proj.bias                        -> patch_embed.proj.bias
      vision_model.patch_embed.pos_emb.weight (H,W,dim)         -> patch_embed.pos_emb_static (interpolated to grid)
      mlp1.{0,1,3}.{weight,bias}                                -> merger.mlp1.{0,1,3}.*
    """
    out = {}
    for k, v in raw_sd.items():
        if k.startswith("vision_model.encoder.blocks."):
            new_k = k[len("vision_model.encoder."):]     # drop "vision_model.encoder."
            out[new_k] = v
        elif k.startswith("vision_model.encoder.final_layernorm."):
            new_k = "final_layernorm." + k.rsplit(".", 1)[-1]
            out[new_k] = v
        elif k == "vision_model.patch_embed.proj.weight":
            # Conv2d weight: (hidden, in_channels, patch_h, patch_w)
            # Linear weight expected: (hidden, patch²*in_channels)
            assert v.dim() == 4, f"expected 4D conv weight, got shape {v.shape}"
            hidden = v.shape[0]
            out["patch_embed.proj.weight"] = v.reshape(hidden, -1)   # (hidden, patch²*C)
        elif k == "vision_model.patch_embed.proj.bias":
            out["patch_embed.proj.bias"] = v
        elif k == "vision_model.patch_embed.pos_emb.weight":
            # Interpolate to target grid_h × grid_w — handled in Api.__init__
            # by baking into pos_emb_static buffer. Store raw here as sentinel.
            out["__raw_pos_emb"] = v
        elif k.startswith("mlp1."):
            new_k = "merger.mlp1." + k[len("mlp1."):]
            out[new_k] = v
    return out


def load_vision_state_dict(model_dir: str, num_patches: int, patch_size: int,
                            in_channels: int, hidden_size: int) -> dict:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        files = sorted(set(idx["weight_map"].values()))
    else:
        files = ["model.safetensors"]

    raw = {}
    for fname in files:
        with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("vision_model.") or k.startswith("mlp1."):
                    raw[k] = f.get_tensor(k)
    return remap_vision_state_dict(raw, num_patches, patch_size, in_channels, hidden_size)


class LocateAnythingVisionApi:
    """Compile-only API for the MoonViT vision HBM."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        image_width: int = 448,
        image_height: int = 448,
        device: str = "cpu",
        w_bits: int = 8,
        vit_core_num: Optional[list[int]] = None,
        march: str = "nash-p",
    ) -> None:
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.w_bits = w_bits
        self.vit_core_num = vit_core_num or [1]
        self.march = march

        os.makedirs(output_model_path, exist_ok=True)
        self.output_vit_model_path = standard_vit_name(
            input_model_path, output_model_path, march,
            self.vit_core_num, image_width, image_height,
        )

        cfg_path = os.path.join(input_model_path, "config.json")
        la_cfg = load_config_from_json(cfg_path)
        vc = la_cfg.vision_config
        vc.image_height = image_height
        vc.image_width = image_width

        print("[LocateAnythingVisionApi] adapted vision_config:")
        print(f"  hidden_size         = {vc.hidden_size}")
        print(f"  num_hidden_layers   = {vc.num_hidden_layers}")
        print(f"  num_attention_heads = {vc.num_attention_heads}")
        print(f"  patch_size          = {vc.patch_size}")
        print(f"  image               = {image_height} x {image_width}")

        self.model = LocateAnythingVisionModel(
            vc, la_cfg.text_config.hidden_size, use_plugin=False,
        )
        self.vision_cfg = vc

        # Load + remap state dict.
        num_patches = self.model.num_patches
        sd = load_vision_state_dict(
            input_model_path,
            num_patches=num_patches, patch_size=vc.patch_size,
            in_channels=3, hidden_size=vc.hidden_size,
        )

        # Extract raw pos_emb (H, W, dim) and bake into pos_emb_static via bicubic.
        raw_pos_emb = sd.pop("__raw_pos_emb", None)
        assert raw_pos_emb is not None, "vision_model.patch_embed.pos_emb.weight missing"

        import torch.nn.functional as F
        # raw_pos_emb: (init_H=64, init_W=64, dim=1152)
        # Interpolate to (grid_h, grid_w) with bicubic.
        pe = raw_pos_emb.permute(2, 0, 1).unsqueeze(0).float()  # (1, dim, 64, 64)
        pe = F.interpolate(pe, size=(self.model.grid_h, self.model.grid_w),
                            mode="bicubic")                     # (1, dim, gH, gW)
        pe = pe.squeeze(0).permute(1, 2, 0).reshape(num_patches, -1)  # (N, dim)
        sd["patch_embed.pos_emb_static"] = pe.to(torch.float32)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # `rope_cos`, `rope_sin` are computed in __init__ and marked persistent
        # → they will be reported as missing since they're not in the checkpoint.
        missing = [k for k in missing if k not in {"rope_cos", "rope_sin"}]
        # `patch_embed.pos_emb_static` is a non-persistent buffer so it can end
        # up in unexpected — filter it.
        unexpected = [k for k in unexpected if k not in {"patch_embed.pos_emb_static"}]

        if missing or unexpected:
            print(f"  WARN missing: {missing[:5]}")
            print(f"  WARN unexpected: {unexpected[:5]}")
        else:
            print("  load_state_dict: clean")

    def compile(self, vit_kwargs: Optional[dict] = None,
                llm_kwargs: Optional[dict] = None) -> None:
        """Compile the vision HBM.

        `llm_kwargs` accepted but ignored (this API produces only the vision HBM).
        """
        vit_kwargs = vit_kwargs or {}
        self.model.compile_mode(True)
        self.model = self.model.to("cpu", dtype=torch.float16)
        gc.collect()

        print("[LocateAnythingVisionApi] export visual...")
        inputs = self.model.get_leap_input_types()
        bc_path = str(Path(self.output_vit_model_path).with_suffix(".visual.bc"))
        bc = self.model.export_module(inputs, "visual", bc_path, high_precision_qpp=True)

        print("[LocateAnythingVisionApi] convert_mlir visual...")
        convert_bc_path = str(Path(self.output_vit_model_path).with_suffix(".visual_convert.bc"))
        mlir = self.model.convert_mlir(
            bc, convert_bc_path,
            enable_vpu=True, march=self.march, dynamic_quant=True,
        )
        func = mlir.functions[0]
        func.remove_io_op(["Dequantize", "Quantize"])

        hbo_path = str(Path(self.output_vit_model_path).with_suffix(".visual.hbo"))
        kwargs = {
            "march": self.march,
            "jobs": vit_kwargs.get("jobs", 16),
            "progress_bar": True,
            "max_time_per_fc": 0.0,
            "opt": 2,
            "debug": False,
            "advice": 0.0,
            "balance": 100,
            "input_no_padding": True,
            "output_no_padding": True,
            "core_num": self.vit_core_num[0],
        }
        if kwargs["core_num"] > 1:
            kwargs["max_l2m_size"] = 25165824

        print(f"[LocateAnythingVisionApi] compile_hbo visual (core_num={kwargs['core_num']})...")
        hbo = self.model.compile_hbo(mlir, save_path=hbo_path, **kwargs)

        print(f"[LocateAnythingVisionApi] link_models -> {self.output_vit_model_path}")
        self.model.link_models([hbo], save_path=self.output_vit_model_path)
        print(f"[LocateAnythingVisionApi] DONE — {self.output_vit_model_path}")

    def get_hbm_path(self) -> str:
        return self.output_vit_model_path
