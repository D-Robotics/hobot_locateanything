"""LocateAnything language-only compile Api — self-contained, no
Qwen2_5_VLTextModel dependency at runtime.

Uses `leap_llm.models.locateanything.text_model_leap.LocateAnythingTextModel`,
whose classes were copied from qwen2_5_vl at authoring time and now live
entirely in the locateanything/ tree (rename + minor adaptations: 1D rope,
tied lm_head, PBD-aware attention_mask input).

Produces:
  LocateAnything-3B_language_chunk_{chunk}_cache_{cache}_w{w}_nash-p_corenum_*_*.hbm
  LocateAnything-3B_embed_tokens.bin
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

from leap_llm.nn.utils import (
    standard_lm_name,
    standard_token_embeddings_name,
)

from leap_llm.models.locateanything.config.locateanything_3b import (
    load_config_from_json,
)
from leap_llm.models.locateanything.text_model_leap import LocateAnythingTextModel
from leap_llm.models.locateanything.hidden_rotation import (
    load_hidden_rotation,
    rotate_language_to_hidden_domain,
)


def remap_language_state_dict(raw_sd: dict) -> dict:
    """Extract the language_model.* portion and re-key to our layout."""
    remapped = {}
    for k, v in raw_sd.items():
        if not k.startswith("language_model."):
            continue
        sub = k[len("language_model."):]
        if sub.startswith("model."):
            remapped[sub[len("model."):]] = v
        elif sub.startswith("lm_head."):
            remapped[sub] = v
    return remapped


def load_language_state_dict(model_dir: str) -> dict:
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
                if k.startswith("language_model."):
                    raw[k] = f.get_tensor(k)
    return remap_language_state_dict(raw)


class LocateAnythingLanguageApi:
    """Compile-only API — language HBM (prefill + decode)."""

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        chunk_size: int = 256,
        batch_size: int = 1,
        cache_len: int = 1024,
        decode_seq_len: int = 6,
        device: str = "cpu",
        w_bits: int = 4,
        mask_value: int = -32768,
        prefill_core_num: Optional[list[int]] = None,
        decode_core_num: Optional[list[int]] = None,
        march: str = "nash-p",
        hidden_rotation_path: Optional[str] = None,
        apply_hidden_rotation: bool = True,
        export_only: bool = False,
    ) -> None:
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cache_len = cache_len
        self.decode_seq_len = decode_seq_len
        self.device = device
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.prefill_core_num = prefill_core_num or [1]
        self.decode_core_num = decode_core_num or [1]
        self.march = march
        self.hidden_rotation_path = hidden_rotation_path
        self.apply_hidden_rotation = apply_hidden_rotation
        self.export_only = export_only

        os.makedirs(output_model_path, exist_ok=True)
        self.output_lm_model_path = standard_lm_name(
            input_model_path, output_model_path, chunk_size, cache_len,
            w_bits, march, self.prefill_core_num, self.decode_core_num,
            batch_size=batch_size,
        )
        self.token_embeddings_file_name = standard_token_embeddings_name(
            input_model_path, output_model_path,
        )

        cfg_path = os.path.join(input_model_path, "config.json")
        la_cfg = load_config_from_json(cfg_path)
        tc = la_cfg.text_config
        # Populate compile-time fields on the dataclass we already have.
        tc.prefill_seq_len = chunk_size
        tc.decode_seq_len = decode_seq_len
        tc.cache_len = cache_len
        tc.batch_size = batch_size
        tc.w_bits = w_bits
        tc.has_scale = False

        print("[LocateAnythingLanguageApi] adapted text_config:")
        print(f"  vocab_size          = {tc.vocab_size}")
        print(f"  hidden_size         = {tc.hidden_size}")
        print(f"  num_hidden_layers   = {tc.num_hidden_layers}")
        print(f"  num_kv_heads        = {tc.num_key_value_heads}")
        print(f"  chunk_size          = {chunk_size}")
        print(f"  cache_len           = {cache_len}")
        print(f"  decode_seq_len      = {decode_seq_len}  (PBD block_size)")
        print(f"  tie_word_embeddings = {tc.tie_word_embeddings}")

        # Build model with our own class (no Qwen2_5_VLTextModel here).
        self.text_model = LocateAnythingTextModel(tc, use_plugin=False)
        self.text_cfg = tc

        sd = load_language_state_dict(input_model_path)
        # If tied, drop lm_head from remap (we'll copy embed weights instead).
        if tc.tie_word_embeddings:
            sd.pop("lm_head.weight", None)
        missing, unexpected = self.text_model.load_state_dict(sd, strict=False)
        # cache_cos/cache_sin are computed inside __init__ and marked
        # persistent (present in state_dict): if the load misses/adds these
        # we just filter them.
        missing = [k for k in missing if k not in {"cache_cos", "cache_sin"}]
        unexpected = [k for k in unexpected if k not in {"cache_cos", "cache_sin"}]
        # Depending on the layer type of lm_head (Dynamic vs plain), the
        # weight key can also appear here — filter and tie afterwards.
        missing = [k for k in missing if not k.startswith("lm_head.")]

        if unexpected:
            print(f"  WARN unexpected: {unexpected[:3]}")
        if missing:
            print(f"  WARN missing (post-tie): {missing[:3]}")
        else:
            print("  load_state_dict: clean")

        if tc.tie_word_embeddings:
            self.text_model.tie_lm_head_to_embeddings()
            print("  lm_head tied to embed_tokens.weight")

        if self.apply_hidden_rotation:
            rotation, source = load_hidden_rotation(
                self.hidden_rotation_path,
                tc.hidden_size,
            )
            rotation_device = (
                self.device
                if self.device.startswith("cuda") and torch.cuda.is_available()
                else "cpu"
            )
            orthogonal_error = rotate_language_to_hidden_domain(
                self.text_model,
                rotation,
                device=rotation_device,
            )
            print(f"  hidden rotation     = {source}")
            print(f"  orthogonal max err  = {orthogonal_error:.9g}")

        # Save embed_tokens.bin
        self._save_embed_tokens()

    def _save_embed_tokens(self) -> None:
        emb = self.text_model.embed_tokens.weight.detach().to(
            dtype=torch.float16,
        ).cpu().numpy()
        out = self.token_embeddings_file_name
        temporary = f"{out}.tmp"
        emb.tofile(temporary)
        os.replace(temporary, out)
        print(f"[save_embed_tokens] wrote {emb.nbytes/1e6:.1f} MB -> {out}")

    def compile(self, vit_kwargs: Optional[dict] = None,
                llm_kwargs: Optional[dict] = None) -> None:
        """Compile pipeline: prefill + decode -> convert -> compile_hbo -> link.

        `vit_kwargs` is accepted (oellm_build.py uniform signature) but
        ignored — this Api does not emit a vision HBM.
        """
        llm_kwargs = llm_kwargs or {}
        self.text_model.compile_mode(True)
        self.text_model = self.text_model.to("cpu", dtype=torch.float16)
        gc.collect()

        num_layers = self.text_cfg.num_hidden_layers
        chunk_size = self.text_cfg.prefill_seq_len
        cache_len = self.text_cfg.cache_len
        batch_size = self.text_cfg.batch_size

        stage_core_map = {
            "prefill": self.prefill_core_num[0],
            "decode": self.decode_core_num[0],
        }

        # ---- Stage 1: export .bc ----
        stage_inputs = {
            "prefill": self.text_model.get_leap_input_types_text_model(
                num_layers, chunk_size, cache_len, batch_size,
            ),
            "decode": self.text_model.get_leap_input_types_decode_model(
                num_layers, self.decode_seq_len, cache_len, batch_size,
            ),
            "decode_ar": self.text_model.get_leap_input_types_decode_model(
                num_layers, 1, cache_len, batch_size,
            ),
        }
        stage_core_map["decode_ar"] = self.decode_core_num[0]
        bc_modules = []
        for stage_name, inputs in stage_inputs.items():
            print(f"[LocateAnythingLanguageApi] export {stage_name}...")
            bc_path = str(Path(self.output_lm_model_path).with_suffix(f".{stage_name}.bc"))
            bc = self.text_model.export_module(
                inputs, stage_name, bc_path, high_precision_qpp=True,
            )
            bc_modules.append(bc)

        if self.export_only:
            print("[LocateAnythingLanguageApi] export-only validation passed")
            return

        # ---- Stage 2/3/4: convert -> compile_hbo -> link ----
        hbos = []
        for bc in bc_modules:
            func_name = bc.functions[0].name
            convert_bc = str(Path(self.output_lm_model_path).with_suffix(f".{func_name}_convert.bc"))
            print(f"[LocateAnythingLanguageApi] convert_mlir {func_name}...")
            mlir = self.text_model.convert_mlir(
                bc, convert_bc,
                enable_vpu=True, march=self.march, dynamic_quant=True,
            )
            func = mlir.functions[0]
            func.remove_io_op(["Dequantize", "Quantize"])

            hbo_path = str(Path(self.output_lm_model_path).with_suffix(f".{func_name}.hbo"))
            kwargs = {
                "march": self.march,
                "jobs": llm_kwargs.get("jobs", 16),
                "progress_bar": True,
                "max_time_per_fc": 0.0,
                "opt": 2,
                "debug": False,
                "advice": 0.0,
                "balance": 100,
                "enable_hpc": True,
                "input_no_padding": True,
                "output_no_padding": True,
                "core_num": stage_core_map[func_name],
            }
            if kwargs["core_num"] > 1:
                kwargs["max_l2m_size"] = 25165824
            print(f"[LocateAnythingLanguageApi] compile_hbo {func_name} "
                  f"(core_num={kwargs['core_num']})...")
            hbo = self.text_model.compile_hbo(mlir, save_path=hbo_path, **kwargs)
            hbos.append(hbo)

        print(f"[LocateAnythingLanguageApi] link_models -> {self.output_lm_model_path}")
        self.text_model.link_models(hbos, save_path=self.output_lm_model_path)
        print(f"[LocateAnythingLanguageApi] DONE — {self.output_lm_model_path}")

    def get_hbm_path(self) -> str:
        return self.output_lm_model_path
