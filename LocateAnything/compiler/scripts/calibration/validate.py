#!/usr/bin/env python3
"""Validate every tensor in a generated LocateAnything calibration bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    import torch

    started = time.monotonic()
    generated_path = args.generated_jsonl.resolve()
    root = generated_path.parent
    generated = read_jsonl(generated_path)
    selected = read_jsonl(args.selected_jsonl.resolve())
    selected_by_id = {row.get("bundle_id"): row for row in selected}
    errors: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    seen: set[str] = set()

    for index, record in enumerate(generated, 1):
        bundle_id = str(record.get("bundle_id") or "")
        reasons = []
        if not bundle_id or bundle_id in seen:
            reasons.append("missing or duplicate bundle_id")
        seen.add(bundle_id)
        source = selected_by_id.get(bundle_id)
        if source is None:
            reasons.append("bundle_id absent from selected manifest")
        else:
            for field in ("task", "prompt", "target_response", "image_sha256"):
                if record.get(field) != source.get(field):
                    reasons.append(f"selected/generated mismatch: {field}")

        tensor_path = root / str(record.get("tensor_file") or "")
        if not tensor_path.is_file():
            reasons.append("tensor file missing")
        elif sha256_file(tensor_path) != record.get("tensor_sha256"):
            reasons.append("tensor SHA256 mismatch")
        else:
            try:
                tensor_format = record.get("tensor_format") or tensor_path.suffix.lstrip(".")
                if tensor_format == "npy":
                    value = np.load(tensor_path, allow_pickle=False)
                    if value.shape != (1, 2304, 588):
                        reasons.append(f"unexpected vision_input shape: {value.shape}")
                    if value.dtype != np.float16:
                        reasons.append(f"unexpected vision_input dtype: {value.dtype}")
                    if not np.isfinite(value).all():
                        reasons.append("vision_input contains NaN or Inf")
                    for mode in (record.get("prediction") or {}):
                        mode_counts[mode] += 1
                elif tensor_format == "pt":
                    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    if tuple(payload["vision_input"].shape) != (1, 2304, 588):
                        reasons.append(f"unexpected vision_input shape: {tuple(payload['vision_input'].shape)}")
                    if tuple(payload["projected_visual_features"].shape) != (1, 576, 2048):
                        reasons.append(
                            "unexpected projected_visual_features shape: "
                            f"{tuple(payload['projected_visual_features'].shape)}"
                        )
                    profile = payload.get("fixed_profile") or {}
                    if profile.get("image_width") != 672 or profile.get("image_height") != 672:
                        reasons.append("profile is not 672x672")
                    if payload.get("source_target_response") != record.get("target_response"):
                        reasons.append("source target response mismatch")
                    predictions = payload.get("prediction_token_ids") or {}
                    if "hybrid" not in predictions:
                        reasons.append("missing hybrid/PBD prediction tokens")
                    for mode in predictions:
                        mode_counts[mode] += 1
                else:
                    reasons.append(f"unsupported tensor format: {tensor_format}")
            except Exception as exc:
                reasons.append(f"tensor load/field validation failed: {exc!r}")

        task_counts[str(record.get("task"))] += 1
        if reasons:
            errors.append({"bundle_id": bundle_id, "reasons": reasons})
        if index % args.progress_every == 0 or index == len(generated):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[calibration validation] {index}/{len(generated)} "
                f"({100 * index / max(len(generated), 1):.1f}%) "
                f"rate={index / elapsed:.2f}/s errors={len(errors)}",
                flush=True,
            )

    if len(generated) != len(selected):
        errors.append({"bundle_id": None, "reasons": [f"record count {len(generated)} != selected {len(selected)}"]})
    missing_tasks = sorted(set(TASKS) - set(task_counts))
    if missing_tasks:
        errors.append({"bundle_id": None, "reasons": [f"missing tasks: {missing_tasks}"]})

    result = {
        "schema_version": 1,
        "passed": not errors,
        "selected_manifest": str(args.selected_jsonl.resolve()),
        "selected_manifest_sha256": sha256_file(args.selected_jsonl.resolve()),
        "generated_manifest": str(generated_path),
        "generated_manifest_sha256": sha256_file(generated_path),
        "selected_count": len(selected),
        "generated_count": len(generated),
        "unique_bundle_ids": len(seen),
        "task_counts": dict(task_counts),
        "prediction_mode_counts": dict(mode_counts),
        "error_count": len(errors),
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "passed", "selected_count", "generated_count", "task_counts",
        "prediction_mode_counts", "error_count", "elapsed_seconds",
    )}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
