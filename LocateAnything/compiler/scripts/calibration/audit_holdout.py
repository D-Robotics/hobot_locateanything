#!/usr/bin/env python3
"""Audit exact and perceptual image leakage between calibration and held-out sets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def resolve_image(manifest: Path, row: dict[str, Any]) -> Path:
    image = Path(str(row.get("image") or ""))
    if not image.is_absolute():
        image = manifest.parent / image
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"missing image for {row.get('bundle_id')}: {image}")
    return image


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8))
        getter = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(getter())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument(
        "--leaked-heldout-jsonl",
        type=Path,
        help="write complete held-out rows implicated by exact, ID, or dHash leakage",
    )
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--fail-on-near", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.dhash_threshold <= 64:
        raise ValueError("dHash threshold must be in [0,64]")

    calibration = read_jsonl(args.calibration_jsonl.resolve())
    heldout = read_jsonl(args.heldout_jsonl.resolve())
    calibration_sha = {str(row.get("image_sha256") or "") for row in calibration}
    heldout_sha = {str(row.get("image_sha256") or "") for row in heldout}
    exact_sha_overlap = sorted(calibration_sha & heldout_sha)
    calibration_ids = {str(row.get("sample_id") or "") for row in calibration if row.get("sample_id")}
    heldout_ids = {str(row.get("sample_id") or "") for row in heldout if row.get("sample_id")}
    sample_id_overlap = sorted(calibration_ids & heldout_ids)

    calibration_hashes = [
        (row, difference_hash(resolve_image(args.calibration_jsonl.resolve(), row)))
        for row in calibration
    ]
    heldout_hashes = [
        (row, difference_hash(resolve_image(args.heldout_jsonl.resolve(), row)))
        for row in heldout
    ]
    near_pairs = []
    for heldout_row, heldout_hash in heldout_hashes:
        for calibration_row, calibration_hash in calibration_hashes:
            distance = hamming_distance(heldout_hash, calibration_hash)
            if distance <= args.dhash_threshold:
                near_pairs.append({
                    "calibration_bundle_id": calibration_row.get("bundle_id"),
                    "calibration_image_sha256": calibration_row.get("image_sha256"),
                    "heldout_bundle_id": heldout_row.get("bundle_id"),
                    "heldout_image_sha256": heldout_row.get("image_sha256"),
                    "dhash_distance": distance,
                })
    near_pairs.sort(key=lambda item: (item["dhash_distance"], str(item["heldout_bundle_id"])))
    args.pairs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_jsonl.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in near_pairs),
        encoding="utf-8",
    )
    leaked_bundle_ids = {
        str(item["heldout_bundle_id"])
        for item in near_pairs
    }
    leaked_bundle_ids.update(
        str(row.get("bundle_id"))
        for row in heldout
        if row.get("image_sha256") in exact_sha_overlap
        or row.get("sample_id") in sample_id_overlap
    )
    if args.leaked_heldout_jsonl is not None:
        args.leaked_heldout_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.leaked_heldout_jsonl.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in heldout
                if str(row.get("bundle_id")) in leaked_bundle_ids
            ),
            encoding="utf-8",
        )
    status = "fail" if exact_sha_overlap or sample_id_overlap or (args.fail_on_near and near_pairs) else "pass"
    report = {
        "schema_version": 1,
        "status": status,
        "calibration_count": len(calibration),
        "heldout_count": len(heldout),
        "exact_image_sha256_overlap_count": len(exact_sha_overlap),
        "sample_id_overlap_count": len(sample_id_overlap),
        "dhash_threshold": args.dhash_threshold,
        "near_pair_count": len(near_pairs),
        "leaked_heldout_record_count": len(leaked_bundle_ids),
        "fail_on_near": args.fail_on_near,
    }
    write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
