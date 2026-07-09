"""LocateAnythingApi — driver for the leap compile pipeline.

This is the P5 skeleton — it wires up:
  - model_factory registration (@register_model in model_factory.py)
  - checkpoint loading via LocateAnything.build()
  - HBM path derivation via standard_vit_name / standard_lm_name
  - compile() entry point (leap DSL forward TBD in P6)

The compile() body itself lands in P6 once we translate the PyTorch
forwards into leap DSL calls. For P5 we validate the *entry* — that
`oellm_build --model_name locateanything-3b` reaches Api.compile()
without exploding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from leap_llm.nn.utils import (
    standard_lm_name,
    standard_token_embeddings_name,
    standard_vit_name,
)
from leap_llm.models.locateanything.model import LocateAnything


class LocateAnythingApi:
    """Compile-driver for LocateAnything-3B (MoonViT + Qwen2 + PBD).

    Constructor arguments mirror Qwen2_5VlApi where semantics carry over,
    plus PBD-specific ones (block_size, causal_attn).
    """

    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        # --- calibration ---
        calib_tsv_path: Optional[str] = None,
        calib_message_path: Optional[str] = None,
        calib_image_path: Optional[str] = None,
        # --- compile knobs ---
        chunk_size: int = 256,
        batch_size: int = 1,
        cache_len: int = 1024,
        image_width: int = 448,
        image_height: int = 448,
        decode_seq_len: int = 6,          # PBD default = block_size
        # --- PBD-specific ---
        block_size: int = 6,
        causal_attn: bool = False,
        # --- backend ---
        devices: Optional[list[str]] = None,
        device: Optional[str] = None,
        model_type: str = "locateanything-3b",
        dtype: str = "float16",
        w_bits: int = 4,
        mask_value: int = -32768,
        vit_core_num: Optional[list[int]] = None,
        prefill_core_num: Optional[list[int]] = None,
        decode_core_num: Optional[list[int]] = None,
        input_model_format: str = "hf",
        march: str = "nash-p",
    ) -> None:
        # Default single-core assignments (aligned with qwen2_5_vl behaviour)
        vit_core_num = vit_core_num or [1]
        prefill_core_num = prefill_core_num or [1]
        decode_core_num = decode_core_num or [1]

        # Device resolution — LocateAnything is 3B so single-GPU is enough.
        if devices is not None:
            self.devices = devices if isinstance(devices, list) else [devices]
        elif device is not None:
            self.devices = [device] if isinstance(device, str) else device
        else:
            self.devices = ["cpu"]
        self.primary_device = self.devices[0]
        self.device = self.primary_device

        # Store all inputs for later stages.
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.calib_tsv_path = calib_tsv_path
        self.calib_message_path = calib_message_path
        self.calib_image_path = calib_image_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cache_len = cache_len
        self.image_width = image_width
        self.image_height = image_height
        self.decode_seq_len = decode_seq_len
        self.block_size = block_size
        self.causal_attn = causal_attn
        self.model_type = model_type
        self.dtype = dtype
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.vit_core_num = vit_core_num
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num
        self.input_model_format = input_model_format
        self.march = march

        # Sanity: block_size and decode_seq_len should agree.
        assert decode_seq_len == block_size, (
            f"decode_seq_len ({decode_seq_len}) must equal block_size "
            f"({block_size}) for PBD."
        )

        # Compute standard output HBM paths.
        # `standard_vit_name` and friends derive the file basename from
        # os.path.basename(input_model_path). Since we point at
        # ~/oe_locateanything/eagle/Embodied/LocateAnything-3B, the HBM will
        # be named `LocateAnything-3B_*.hbm` — matching the naming convention
        # you asked for.
        self.output_vit_model_path = standard_vit_name(
            input_model_path, output_model_path, march,
            vit_core_num, image_width, image_height,
        )
        self.output_lm_model_path = standard_lm_name(
            input_model_path, output_model_path, chunk_size, cache_len,
            w_bits, march, prefill_core_num, decode_core_num,
            batch_size=batch_size,
        )
        self.token_embeddings_file_name = standard_token_embeddings_name(
            input_model_path, output_model_path,
        )

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_dir = output_model_path

        # Load LocateAnything from source checkpoint.
        # NB. Unlike qwen2_5_vl we do *not* save a temp .pth first — our
        # build() reads safetensors directly, matching the checkpoint layout.
        self.model, self.missing_keys, self.unexpected_keys = LocateAnything.build(
            model_dir=input_model_path,
            image_height=image_height,
            image_width=image_width,
            decode_seq_len=decode_seq_len,
            chunk_size=chunk_size,
            cache_len=cache_len,
            batch_size=batch_size,
            w_bits=w_bits,
        )
        self.config = self.model.get_config()

        # Diagnostics: any surprising checkpoint mismatch is worth logging.
        if self.unexpected_keys:
            print(f"WARN: {len(self.unexpected_keys)} unexpected keys in "
                  f"checkpoint (first 3: {self.unexpected_keys[:3]})")
        if self.missing_keys:
            expected_missing = {
                "language_model.lm_head.weight",
                "vision_model.patch_embed.pos_emb_static",
            }
            surprising = [k for k in self.missing_keys if k not in expected_missing]
            if surprising:
                print(f"WARN: {len(surprising)} missing keys not in the "
                      f"allowed-missing set (first 3: {surprising[:3]})")

        print(f"[LocateAnythingApi] initialized")
        print(f"  input      : {input_model_path}")
        print(f"  output dir : {output_model_path}")
        print(f"  vit hbm    : {self.output_vit_model_path}")
        print(f"  lm  hbm    : {self.output_lm_model_path}")
        print(f"  embed bin  : {self.token_embeddings_file_name}")
        print(f"  block_size : {block_size}   causal_attn : {causal_attn}")
        print(f"  decode_seq_len : {decode_seq_len}   chunk : {chunk_size}   cache : {cache_len}")

    def save_embed_tokens(self) -> None:
        """Write embed_tokens.weight to disk as fp16 binary blob.

        Host runtime reads this file (~600 MB for vocab 152681, hidden 2048)
        as a memory-mapped table to look up token embeddings without
        touching the HBM.
        """
        emb = self.model.get_input_embeddings().weight.detach().to(
            dtype=torch.float16,
        ).cpu().numpy()
        out = self.token_embeddings_file_name
        if not os.path.exists(out):
            emb.tofile(out)
            print(f"[save_embed_tokens] wrote {emb.nbytes/1e6:.1f} MB to {out}")
        else:
            print(f"[save_embed_tokens] already present at {out}")

    def compile(self, vit_kwargs: Optional[dict] = None,
                llm_kwargs: Optional[dict] = None) -> None:
        """Full compile pipeline entry point.

        P5 status: skeleton only. The end-to-end leap DSL forward
        translations for MoonViT + Qwen2 (with PBD mask handling) are
        the work of P6+M3.

        For now, this method:
          1. Confirms the model instantiates.
          2. Runs a forward pass on the vision path to catch any lazy
             module-init failures.
          3. Dumps embed_tokens.bin so at least one real artifact lands
             on disk when the user runs oellm_build with our model_name.
          4. Prints a clear message telling the operator this is a P5
             skeleton, not the final HBM producer.
        """
        vit_kwargs = vit_kwargs or {}
        llm_kwargs = llm_kwargs or {}

        # Move model to primary device for calibration passes.
        target = self.primary_device
        if target != "cpu" and target.startswith("cuda") and not torch.cuda.is_available():
            print(f"WARN: {target} requested but no CUDA present; falling back to cpu")
            target = "cpu"
        self.model = self.model.to(target)

        self.save_embed_tokens()

        # Sanity forward — a single deterministic 448x448 patch stream.
        torch.manual_seed(0)
        PATCH = self.config.vision_config.patch_size
        grid_h = self.image_height // PATCH
        grid_w = self.image_width // PATCH
        n_patch = grid_h * grid_w

        patches = torch.randn(
            n_patch, 3, PATCH, PATCH, device=target, dtype=torch.float32,
        )
        from leap_llm.models.locateanything.utils.rope_2d import (
            precompute_freqs_cos_sin, gather_freqs_by_grid,
        )
        head_dim = self.config.vision_config.hidden_size // \
            self.config.vision_config.num_attention_heads
        freqs_table = precompute_freqs_cos_sin(
            max_height=512, max_width=512, dim=head_dim, theta_base=10000.0,
        ).to(target)
        freqs = gather_freqs_by_grid(freqs_table, grid_h, grid_w)

        self.model.eval()
        with torch.no_grad():
            visual_emb = self.model.vision_model(patches, freqs)
        print(f"[compile P5 skeleton] vision forward OK, output shape "
              f"= {tuple(visual_emb.shape)}, dtype = {visual_emb.dtype}")

        print()
        print("=" * 60)
        print("LocateAnythingApi.compile() — P5 SKELETON ONLY")
        print("  Vision + language forward paths are numerically verified")
        print("  against the LocateAnything checkpoint (see sanity_model.py).")
        print("  Full leap DSL translation for .hbm output lands in P6.")
        print("=" * 60)

    def get_quant_path(self) -> tuple[Optional[str], Optional[str]]:
        """Return the LM and VLM quant .bc paths (P6+)."""
        llm_bc = str(Path(self.output_model_dir).with_suffix(".prefill_convert.bc"))
        vlm_bc = str(Path(self.output_model_dir).with_suffix(".convert.bc"))
        return llm_bc, vlm_bc

    def get_hbm_path(self) -> tuple[str, str]:
        """Return the final .hbm paths for the language and vision models."""
        return self.output_lm_model_path, self.output_vit_model_path
