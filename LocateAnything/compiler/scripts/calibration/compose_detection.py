#!/usr/bin/env python3
"""Compose the frozen 1,200-record detection-primary calibration sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 20260729
DEFAULT_COCO_SINGLE = 200
DEFAULT_COCO_DOUBLE = 220
DEFAULT_COCO_MULTI = 80
DEFAULT_RETAIL = 120
TABLE9_POLICY = "locateanything_paper_table9_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def normalized_categories(row: dict[str, Any]) -> list[str]:
    categories = [" ".join(str(value).strip().split()) for value in row.get("categories", [])]
    categories = [value for value in categories if value]
    if not categories or any("</c>" in value for value in categories):
        raise ValueError(f"invalid categories for sample {row.get('sample_id')!r}")
    return categories


def table9_detection_prompt(categories: list[str]) -> str:
    return (
        "Locate all the instances that matches the following description: "
        + "</c>".join(categories)
        + "."
    )


def table9_layout_prompt(categories: list[str]) -> str:
    return (
        "Detect all the objects in the image that belong to the category set: "
        + "</c>".join(categories)
        + "."
    )


def with_source_root(row: dict[str, Any], manifest: Path) -> dict[str, Any]:
    copied = dict(row)
    copied["image_root"] = str(manifest.parent.resolve())
    return copied


def selected_coco_rows(
    rows: list[dict[str, Any]], quotas: dict[str, int], seed: int, manifest: Path
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in quotas}
    for row in rows:
        if row.get("task") != "detection":
            raise ValueError(f"COCO manifest contains non-Detection sample: {row.get('sample_id')}")
        stratum = str((row.get("metadata") or {}).get("calibration_stratum") or "")
        if stratum not in grouped:
            raise ValueError(f"unknown COCO stratum {stratum!r}: {row.get('sample_id')}")
        grouped[stratum].append(row)

    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        ranked = sorted(
            grouped[stratum],
            key=lambda row: stable_digest(seed, stratum, row.get("sample_id"), row.get("image_sha256")),
        )
        if len(ranked) < quota:
            raise ValueError(f"COCO {stratum} has {len(ranked)} rows, requested {quota}")
        for source in ranked[:quota]:
            row = with_source_root(source, manifest)
            categories = normalized_categories(row)
            row["prompt"] = table9_detection_prompt(categories)
            row["prompt_policy"] = TABLE9_POLICY
            metadata = dict(row.get("metadata") or {})
            metadata["calibration_source_role"] = "coco_multicategory_detection"
            row["metadata"] = metadata
            selected.append(row)
    return selected


def selected_retail_rows(
    rows: list[dict[str, Any]], count: int, seed: int, manifest: Path
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("task") == "detection"
        and (row.get("source_dataset") == "SKU110K_fixed" or row.get("source") == "SKU110K")
    ]
    candidates.sort(
        key=lambda row: stable_digest(seed, "retail", row.get("sample_id"), row.get("image_sha256"))
    )
    if len(candidates) < count:
        raise ValueError(f"retail source has {len(candidates)} rows, requested {count}")

    selected = []
    for source in candidates[:count]:
        row = with_source_root(source, manifest)
        categories = normalized_categories(row)
        row["legacy_prompt"] = row.get("prompt")
        row["prompt"] = table9_detection_prompt(categories)
        row["prompt_policy"] = TABLE9_POLICY
        metadata = dict(row.get("metadata") or {})
        metadata["calibration_source_role"] = "dense_retail_detection"
        row["metadata"] = metadata
        selected.append(row)
    return selected


def non_detection_rows(rows: list[dict[str, Any]], manifest: Path) -> list[dict[str, Any]]:
    selected = []
    for source in rows:
        if source.get("task") == "detection":
            continue
        row = with_source_root(source, manifest)
        if row.get("task") == "layout":
            categories = normalized_categories(row)
            row["legacy_prompt"] = row.get("prompt")
            row["prompt"] = table9_layout_prompt(categories)
            row["prompt_policy"] = TABLE9_POLICY
        selected.append(row)
    return selected


def compose(
    coco_manifest: Path,
    baseline_manifest: Path,
    output_dir: Path,
    coco_quotas: dict[str, int],
    retail_count: int,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "coco": output_dir / "detection_coco.jsonl",
        "retail": output_dir / "detection_retail.jsonl",
        "other": output_dir / "other_tasks.jsonl",
        "summary": output_dir / "composition_summary.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(f"composition output exists: {existing[0]}; pass --force to replace")

    coco_rows = selected_coco_rows(
        read_jsonl(coco_manifest), coco_quotas, seed, coco_manifest
    )
    baseline_rows = read_jsonl(baseline_manifest)
    retail_rows = selected_retail_rows(
        baseline_rows, retail_count, seed, baseline_manifest
    )
    other_rows = non_detection_rows(baseline_rows, baseline_manifest)

    all_rows = coco_rows + retail_rows + other_rows
    hashes = [str(row.get("image_sha256") or "") for row in all_rows]
    if any(not value for value in hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("composed sources contain a missing or duplicate image SHA256")

    task_counts = Counter(str(row.get("task")) for row in all_rows)
    source_roles = Counter(
        str((row.get("metadata") or {}).get("calibration_source_role") or "existing_non_detection")
        for row in all_rows
    )
    atomic_write_jsonl(output_paths["coco"], coco_rows)
    atomic_write_jsonl(output_paths["retail"], retail_rows)
    atomic_write_jsonl(output_paths["other"], other_rows)
    summary = {
        "seed": seed,
        "input_manifest_sha256": {
            str(coco_manifest.resolve()): sha256_file(coco_manifest),
            str(baseline_manifest.resolve()): sha256_file(baseline_manifest),
        },
        "coco_quotas": coco_quotas,
        "retail_count": retail_count,
        "task_counts": dict(task_counts),
        "source_role_counts": dict(source_roles),
        "unique_images": len(set(hashes)),
        "output_manifest_sha256": {
            key: sha256_file(path)
            for key, path in output_paths.items()
            if key != "summary"
        },
    }
    atomic_write_json(output_paths["summary"], summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-selected-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coco-single", type=int, default=DEFAULT_COCO_SINGLE)
    parser.add_argument("--coco-double", type=int, default=DEFAULT_COCO_DOUBLE)
    parser.add_argument("--coco-multi", type=int, default=DEFAULT_COCO_MULTI)
    parser.add_argument("--retail-detection", type=int, default=DEFAULT_RETAIL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = {
        "single": args.coco_single,
        "double": args.coco_double,
        "multi": args.coco_multi,
    }
    if any(value < 0 for value in (*quotas.values(), args.retail_detection)):
        raise ValueError("source quotas must be non-negative")
    summary = compose(
        args.coco_jsonl.resolve(),
        args.baseline_selected_jsonl.resolve(),
        args.output_dir.resolve(),
        quotas,
        args.retail_detection,
        args.seed,
        args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
