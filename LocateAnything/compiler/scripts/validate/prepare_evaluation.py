#!/usr/bin/env python3
"""Create profile-adjusted grounding references without running the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from compiler.scripts.calibration.prepare import transform_target_response


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def letterbox_transform(width: int, height: int, target_width: int, target_height: int) -> dict[str, Any]:
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale = min(target_width / width, target_height / height)
    resized_width = min(target_width, max(1, int(round(width * scale))))
    resized_height = min(target_height, max(1, int(round(height * scale))))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return {
        "mode": "letterbox",
        "source_size": [width, height],
        "target_size": [target_width, target_height],
        "resized_size": [resized_width, resized_height],
        "scale_xy": [resized_width / width, resized_height / height],
        "padding_ltrb": [left, top, target_width - resized_width - left, target_height - resized_height - top],
        "letterbox_fill": 128,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=672)
    parser.add_argument("--image-height", type=int, default=672)
    args = parser.parse_args()
    selected = args.selected_jsonl.resolve()
    rows = read_jsonl(selected)
    output = []
    for row in rows:
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = selected.parent / image_path
        with Image.open(image_path) as image:
            transform = letterbox_transform(image.width, image.height, args.image_width, args.image_height)
        item = dict(row)
        item["spatial_transform"] = transform
        item["profile_target_response"] = transform_target_response(
            str(row.get("target_response") or ""), transform
        )
        item["evaluation_profile"] = {
            "image_width": args.image_width,
            "image_height": args.image_height,
            "resize_mode": "letterbox",
        }
        output.append(item)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(args.output_jsonl.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    os.replace(temporary, args.output_jsonl)
    print(json.dumps({
        "records": len(output),
        "selected_manifest_sha256": sha256(selected),
        "reference_manifest_sha256": sha256(args.output_jsonl),
        "profile": "672x672 letterbox" if (args.image_width, args.image_height) == (672, 672) else f"{args.image_width}x{args.image_height} letterbox",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
