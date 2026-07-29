#!/usr/bin/env python3
"""Export LocateAnything vision inputs from PyTorch bundles to portable NPY files."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

VISION_INPUT_SHAPE = (1, 2304, 588)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def export_inputs(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    source_dir = input_dir.resolve()
    destination = output_dir.resolve()
    sources = sorted(source_dir.glob("*.pt"))
    if not sources:
        raise ValueError(f"no .pt files found in {source_dir}")

    source_stems = {path.stem for path in sources}
    stale = sorted(path.name for path in destination.glob("*.npy") if path.stem not in source_stems)
    if stale:
        raise ValueError(f"{destination} contains {len(stale)} unrelated NPY files")

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    progress = tqdm(sources, desc="Export calibration", unit="file", dynamic_ncols=True)
    for source in progress:
        output = destination / f"{source.stem}.npy"
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "vision_input" not in payload:
                raise ValueError("bundle lacks vision_input")
            tensor = payload["vision_input"]
            if not hasattr(tensor, "detach"):
                raise TypeError("vision_input is not a tensor")
            if tensor.dtype != torch.float16:
                raise TypeError(f"vision_input dtype is {tensor.dtype}, expected torch.float16")
            value = tensor.detach().cpu().numpy()
            if value.shape != VISION_INPUT_SHAPE:
                raise ValueError(f"vision_input shape is {value.shape}, expected {VISION_INPUT_SHAPE}")
            if not np.isfinite(value).all():
                raise ValueError("vision_input contains NaN or Inf")
            atomic_npy(output, value)
            records.append(
                {
                    "id": source.stem,
                    "source": str(source),
                    "output": str(output),
                    "output_sha256": sha256(output),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": output.stat().st_size,
                }
            )
        except Exception as error:
            raise RuntimeError(f"failed to export {source}: {error}") from error
        finally:
            if "payload" in locals():
                del payload
            if "tensor" in locals():
                del tensor
            if "value" in locals():
                del value
            gc.collect()
        progress.set_postfix(file=source.name, refresh=False)

    outputs = sorted(destination.glob("*.npy"))
    if len(outputs) != len(sources):
        raise RuntimeError(f"exported {len(outputs)} NPY files from {len(sources)} sources")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "input_dir": str(source_dir),
        "output_dir": str(destination),
        "source_count": len(sources),
        "output_count": len(outputs),
        "shape": list(VISION_INPUT_SHAPE),
        "dtype": "float16",
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    atomic_json(destination / "export_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--input_dir", type=Path, required=True, help="directory containing calibration .pt files")
    root.add_argument("--output_dir", type=Path, required=True, help="directory for portable .npy files")
    return root


def main() -> int:
    args = parser().parse_args()
    result = export_inputs(args.input_dir, args.output_dir)
    print(f"Exported {result['output_count']} calibration inputs to {result['output_dir']}")
    print(f"Manifest: {Path(result['output_dir']) / 'export_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"ERROR: {error}") from error
