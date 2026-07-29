"""Export one calibration Language payload for the S600 C++ runner.

The runner intentionally does not depend on PyTorch. This exporter converts a
single calibration ``.pt`` bundle into two small, stable binary inputs:

* prompt token IDs: little-endian int32, unpadded;
* fixed-profile Vision input: contiguous FP16 ``[1, 2304, 588]``;
* projected visual features after the same 2048-dim hidden rotation as the
  compiled embedding table: contiguous FP16, one row per image placeholder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from leap_llm.models.locateanything.hidden_rotation import load_hidden_rotation

    payload = torch.load(args.input_file, map_location="cpu", weights_only=False)
    prompt_ids = payload["prompt_input_ids"].reshape(-1).to(torch.int32).contiguous()
    vision_input = payload["vision_input"].to(torch.float16).contiguous()
    if tuple(vision_input.shape) != (1, 2304, 588):
        raise ValueError(f"unexpected vision_input shape: {tuple(vision_input.shape)}")
    projected = payload["projected_visual_features"].reshape(-1, 2048).float()
    image_count = int((prompt_ids == 151665).sum().item())
    if image_count != projected.shape[0]:
        raise ValueError(
            f"image placeholder count {image_count} != visual rows {projected.shape[0]}"
        )
    rotation, source = load_hidden_rotation(None, 2048)
    rotated = (projected @ rotation.float()).to(torch.float16).contiguous()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = args.output_dir / "prompt_tokens.i32.bin"
    visual_path = args.output_dir / "visual_features.f16.bin"
    vision_input_path = args.output_dir / "vision_input.f16.bin"
    tokens_path.write_bytes(prompt_ids.numpy().tobytes(order="C"))
    vision_input_path.write_bytes(vision_input.numpy().tobytes(order="C"))
    visual_path.write_bytes(rotated.numpy().tobytes(order="C"))
    manifest = {
        "schema_version": 1,
        "source_pt": str(args.input_file.resolve()),
        "prompt_tokens": int(prompt_ids.numel()),
        "image_tokens": image_count,
        "hidden_size": 2048,
        "rotation_source": source,
        "tokens_file": tokens_path.name,
        "tokens_sha256": sha256(tokens_path),
        "vision_input_file": vision_input_path.name,
        "vision_input_shape": list(vision_input.shape),
        "vision_input_sha256": sha256(vision_input_path),
        "visual_file": visual_path.name,
        "visual_sha256": sha256(visual_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
