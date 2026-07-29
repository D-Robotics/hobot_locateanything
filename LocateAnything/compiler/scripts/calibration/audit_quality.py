#!/usr/bin/env python3
"""Deep quality audit for a selected LocateAnything calibration bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compiler.scripts.calibration.qa import (
    BOX_RE,
    COORD_RE,
    draw_target,
    letterbox,
    read_jsonl,
    sha256_file,
    write_json,
)


TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p05", "p25", "p50", "p75", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def image_metrics(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    gray = np.asarray(rgb.convert("L").resize((256, 256)), dtype=np.float32)
    histogram = np.bincount(gray.astype(np.uint8).reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    gradient_x = np.abs(np.diff(gray, axis=1)).mean()
    gradient_y = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": max(width / height, height / width),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "entropy": entropy,
        "gradient_mean": float((gradient_x + gradient_y) / 2.0),
    }


def dhash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def geometry_metrics(response: str) -> dict[str, Any]:
    points = 0
    boxes = 0
    degenerate = 0
    tiny = 0
    huge = 0
    edge_touching = 0
    areas: list[float] = []
    for match in BOX_RE.finditer(response):
        coords = [int(value) for value in COORD_RE.findall(match.group(1))]
        if len(coords) == 2:
            points += 1
            edge_touching += int(any(value in {0, 1000} for value in coords))
            continue
        boxes += 1
        x1, y1, x2, y2 = coords
        edge_touching += int(any(value in {0, 1000} for value in coords))
        if x2 <= x1 or y2 <= y1:
            degenerate += 1
            continue
        area = (x2 - x1) * (y2 - y1) / 1_000_000.0
        areas.append(area)
        tiny += int(area < 0.0001)
        huge += int(area > 0.9)
    return {
        "points": points,
        "boxes": boxes,
        "degenerate_boxes": degenerate,
        "tiny_boxes": tiny,
        "huge_boxes": huge,
        "edge_touching_geometries": edge_touching,
        "box_areas": areas,
    }


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def stable_order(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{row['image_sha256']}".encode()).hexdigest()


def build_review_sheet(
    selected: list[dict[str, Any]],
    bundle_dir: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    tile_width, tile_height = 360, 420
    columns = 4
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    font = ImageFont.load_default()
    index: list[dict[str, Any]] = []
    for offset, row in enumerate(selected):
        image_path = bundle_dir / row["image"]
        with Image.open(image_path) as source:
            overlay, transform = letterbox(source)
        geometry_count = draw_target(overlay, str(row.get("target_response") or ""), transform)
        overlay.thumbnail((tile_width, tile_height - 72))
        tile = Image.new("RGB", (tile_width, tile_height), (238, 238, 238))
        tile.paste(overlay, ((tile_width - overlay.width) // 2, 72))
        label = " | ".join([
            str(row.get("bundle_id")),
            str(row.get("source")),
            str(row.get("prompt") or row.get("phrase") or ""),
        ])
        draw = ImageDraw.Draw(tile)
        for line_no, line in enumerate(textwrap.wrap(label, width=52)[:4]):
            draw.text((5, 4 + line_no * 15), line, fill="black", font=font)
        x = (offset % columns) * tile_width
        y = (offset // columns) * tile_height
        sheet.paste(tile, (x, y))
        index.append({
            "bundle_id": row["bundle_id"],
            "task": row["task"],
            "source": row.get("source"),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "image": row["image"],
            "geometry_count": geometry_count,
        })
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    return index


def export_individual_reviews(
    selected: list[dict[str, Any]],
    bundle_dir: Path,
    destination: Path,
) -> list[dict[str, Any]]:
    """Write one annotated image per reviewed sample for unambiguous inspection."""
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    for row in selected:
        image_path = bundle_dir / row["image"]
        with Image.open(image_path) as source:
            overlay, transform = letterbox(source)
            geometry_count = draw_target(overlay, str(row.get("target_response") or ""), transform)
            output_path = destination / f"{row['bundle_id']}.jpg"
            overlay.save(output_path, quality=94)
        index.append({
            "bundle_id": row["bundle_id"],
            "task": row["task"],
            "source": row.get("source"),
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "image": row["image"],
            "review_image": output_path.relative_to(destination.parent.parent).as_posix(),
            "geometry_count": geometry_count,
        })
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-per-domain", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.selected_jsonl)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics_by_id: dict[str, dict[str, Any]] = {}
    hashes: dict[str, int] = {}
    task_counts = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts = Counter()
    license_counts = Counter()
    dimensions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    geometry: dict[str, Counter[str]] = defaultdict(Counter)
    box_areas: dict[str, list[float]] = defaultdict(list)
    prompt_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        bundle_id = str(row.get("bundle_id"))
        task = str(row.get("task"))
        task_counts[task] += 1
        source_counts[task][str(row.get("source_dataset") or row.get("source") or "missing")] += 1
        split_counts[str(row.get("split"))] += 1
        license_counts[str(row.get("license") or "missing")] += 1
        prompt = str(row.get("prompt") or "")
        prompt_counts[task][prompt] += 1
        reasons = []
        required = ["bundle_id", "task", "source", "sample_id", "split", "image", "image_sha256", "prompt", "target_response"]
        missing = [field for field in required if not row.get(field)]
        if missing:
            reasons.append(f"missing fields: {missing}")
        image_path = args.bundle_dir / str(row.get("image"))
        if not image_path.is_file():
            reasons.append("missing image")
            hard_errors.append({"bundle_id": bundle_id, "task": task, "reasons": reasons})
            continue
        if sha256_file(image_path) != row.get("image_sha256"):
            reasons.append("image SHA256 mismatch")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_info = image_metrics(image)
                perceptual = dhash(image)
                mode = image.mode
        except Exception as exc:
            reasons.append(f"image decode failed: {exc}")
            hard_errors.append({"bundle_id": bundle_id, "task": task, "reasons": reasons})
            continue
        declared = (row.get("source_width"), row.get("source_height"))
        if declared != (image_info["width"], image_info["height"]):
            reasons.append(f"declared dimensions {declared} != actual dimensions")
        response = str(row.get("target_response") or "")
        geom = geometry_metrics(response)
        if not geom["points"] and not geom["boxes"]:
            reasons.append("no valid geometry")
        if geom["degenerate_boxes"]:
            reasons.append(f"degenerate boxes: {geom['degenerate_boxes']}")
        if response.count("<ref>") != response.count("</ref>"):
            reasons.append("unbalanced ref tokens")
        if response.count("<box>") != response.count("</box>"):
            reasons.append("unbalanced box tokens")
        if reasons:
            hard_errors.append({"bundle_id": bundle_id, "task": task, "reasons": reasons})

        row_warnings = []
        if image_info["aspect_ratio"] > 4:
            row_warnings.append("extreme aspect ratio > 4")
        if min(image_info["width"], image_info["height"]) < 224:
            row_warnings.append("short side < 224 pixels")
        if image_info["gray_std"] < 12:
            row_warnings.append("very low contrast")
        if image_info["gradient_mean"] < 2:
            row_warnings.append("very low gradient/sharpness proxy")
        if geom["tiny_boxes"]:
            row_warnings.append(f"tiny boxes: {geom['tiny_boxes']}")
        if geom["huge_boxes"]:
            row_warnings.append(f"huge boxes: {geom['huge_boxes']}")
        if row_warnings:
            warnings.append({"bundle_id": bundle_id, "task": task, "reasons": row_warnings})

        hashes[bundle_id] = perceptual
        metrics_by_id[bundle_id] = {
            **image_info,
            "mode": mode,
            "geometry_count": geom["points"] + geom["boxes"],
            "tiny_boxes": geom["tiny_boxes"],
            "risk_score": (
                4 * int(image_info["aspect_ratio"] > 4)
                + 4 * int(min(image_info["width"], image_info["height"]) < 224)
                + 3 * int(image_info["gray_std"] < 12)
                + 3 * int(image_info["gradient_mean"] < 2)
                + min(4, geom["tiny_boxes"])
                + min(4, (geom["points"] + geom["boxes"]) // 12)
            ),
        }
        for key in ("width", "height", "area", "aspect_ratio", "gray_mean", "gray_std", "entropy", "gradient_mean"):
            dimensions[task][key].append(float(image_info[key]))
        for key in ("points", "boxes", "degenerate_boxes", "tiny_boxes", "huge_boxes", "edge_touching_geometries"):
            geometry[task][key] += int(geom[key])
        box_areas[task].extend(geom["box_areas"])

    near_duplicates = []
    ids = sorted(hashes)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            distance = hamming(hashes[left], hashes[right])
            if distance <= 4:
                near_duplicates.append({"left": left, "right": right, "dhash_distance": distance})

    review_index = []
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task and str(row.get("bundle_id")) in metrics_by_id]
        random_count = max(1, args.review_per_domain // 2)
        random_rows = sorted(task_rows, key=lambda row: stable_order(row, "quality-review"))[:random_count]
        chosen = {row["bundle_id"] for row in random_rows}
        risk_rows = sorted(
            (row for row in task_rows if row["bundle_id"] not in chosen),
            key=lambda row: (
                -metrics_by_id[row["bundle_id"]]["risk_score"],
                stable_order(row, "risk-review"),
            ),
        )[: args.review_per_domain - len(random_rows)]
        selected = random_rows + risk_rows
        review_index.extend(export_individual_reviews(
            selected,
            args.bundle_dir,
            output_dir / "review_samples" / task,
        ))
        review_index.extend(build_review_sheet(
            selected,
            args.bundle_dir,
            output_dir / "review_sheets" / f"{task}_review.jpg",
        ))

    summary = {
        "schema_version": 1,
        "selected_manifest": str(args.selected_jsonl.resolve()),
        "selected_manifest_sha256": sha256_file(args.selected_jsonl),
        "sample_count": len(rows),
        "task_counts": dict(task_counts),
        "source_counts_by_task": {task: dict(counts) for task, counts in source_counts.items()},
        "split_counts": dict(split_counts),
        "license_counts": dict(license_counts),
        "hard_error_count": len(hard_errors),
        "warning_record_count": len(warnings),
        "near_duplicate_pair_count_dhash_le_4": len(near_duplicates),
        "image_metrics_by_task": {
            task: {key: quantiles(values) for key, values in task_values.items()}
            for task, task_values in dimensions.items()
        },
        "geometry_by_task": {task: dict(counts) for task, counts in geometry.items()},
        "box_area_quantiles_by_task": {task: quantiles(values) for task, values in box_areas.items()},
        "unique_prompts_by_task": {task: len(counts) for task, counts in prompt_counts.items()},
        "top_prompts_by_task": {
            task: [{"prompt": prompt, "count": count} for prompt, count in counts.most_common(5)]
            for task, counts in prompt_counts.items()
        },
        "semantic_review_samples_per_task": args.review_per_domain,
    }
    write_json(output_dir / "quality_audit_summary.json", summary)
    (output_dir / "hard_errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hard_errors), encoding="utf-8"
    )
    (output_dir / "quality_warnings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in warnings), encoding="utf-8"
    )
    (output_dir / "near_duplicates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in near_duplicates), encoding="utf-8"
    )
    (output_dir / "semantic_review_index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_index), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not hard_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
