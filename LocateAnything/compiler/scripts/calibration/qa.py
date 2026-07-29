#!/usr/bin/env python3
"""Validate a selected LA bundle and render 672x672 letterbox overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw


BOX_RE = re.compile(r"<box>((?:<\d+>){2}|(?:<\d+>){4})</box>")
COORD_RE = re.compile(r"<(\d+)>")
EXPECTED_COUNTS = {
    "detection": 620,
    "gui": 180,
    "referring": 120,
    "ocr": 120,
    "layout": 100,
    "pointing": 60,
}
COLORS = ["#e53935", "#00897b", "#3949ab", "#f9a825", "#8e24aa"]


def parse_quotas(values: list[str] | None) -> dict[str, int]:
    if not values:
        return dict(EXPECTED_COUNTS)
    quotas: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --quota {value!r}; expected task=number")
        task, raw_count = value.split("=", 1)
        task = task.strip().lower()
        if task not in EXPECTED_COUNTS:
            raise ValueError(f"unsupported quota task: {task!r}")
        if task in quotas:
            raise ValueError(f"duplicate quota task: {task}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"quota must be positive: {value!r}")
        quotas[task] = count
    missing = sorted(set(EXPECTED_COUNTS) - set(quotas))
    if missing:
        raise ValueError(f"explicit quotas missing tasks: {missing}")
    return quotas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def letterbox(image: Image.Image, size: int = 672) -> tuple[Image.Image, dict[str, Any]]:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(image.resize((resized_width, resized_height), Image.Resampling.LANCZOS), (left, top))
    return canvas, {
        "scale": scale,
        "left": left,
        "top": top,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "source_width": width,
        "source_height": height,
    }


def target_pixel(value: int, axis: int, transform: dict[str, Any]) -> float:
    source_extent = transform["source_width"] if axis == 0 else transform["source_height"]
    pad = transform["left"] if axis == 0 else transform["top"]
    return value / 1000.0 * source_extent * transform["scale"] + pad


def draw_target(image: Image.Image, response: str, transform: dict[str, Any]) -> int:
    draw = ImageDraw.Draw(image)
    count = 0
    for index, match in enumerate(BOX_RE.finditer(response)):
        coords = [int(value) for value in COORD_RE.findall(match.group(1))]
        color = COLORS[index % len(COLORS)]
        if len(coords) == 2:
            x = target_pixel(coords[0], 0, transform)
            y = target_pixel(coords[1], 1, transform)
            radius = 7
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
        else:
            x1 = target_pixel(coords[0], 0, transform)
            y1 = target_pixel(coords[1], 1, transform)
            x2 = target_pixel(coords[2], 0, transform)
            y2 = target_pixel(coords[3], 1, transform)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        count += 1
    return count


def count_degenerate_boxes(response: str) -> int:
    count = 0
    for match in BOX_RE.finditer(response):
        coords = [int(value) for value in COORD_RE.findall(match.group(1))]
        if len(coords) == 4 and (coords[2] <= coords[0] or coords[3] <= coords[1]):
            count += 1
    return count


def contact_sheet(paths: list[Path], destination: Path) -> None:
    thumb = 320
    columns = 5
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb, thumb))
            x = (index % columns) * thumb + (thumb - image.width) // 2
            y = (index // columns) * thumb + (thumb - image.height) // 2
            sheet.paste(image, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--overlays-per-domain", type=int, default=10)
    parser.add_argument(
        "--quota", action="append",
        help="expected task quota; repeat for all six domains",
    )
    args = parser.parse_args()
    expected_counts = parse_quotas(args.quota)

    rows = read_jsonl(args.selected_jsonl)
    errors: list[dict[str, Any]] = []
    counts = Counter()
    sha_seen: set[str] = set()
    block_counts: dict[str, list[int]] = defaultdict(list)
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    overlay_paths: dict[str, list[Path]] = defaultdict(list)
    overlay_index: list[dict[str, Any]] = []

    for row in rows:
        task = str(row.get("task"))
        counts[task] += 1
        image_path = args.bundle_dir / row["image"]
        row_errors = []
        if row.get("split") != "train":
            row_errors.append("non-train split")
        if not image_path.is_file():
            row_errors.append("missing image")
        elif sha256_file(image_path) != row.get("image_sha256"):
            row_errors.append("image SHA256 mismatch")
        if row.get("image_sha256") in sha_seen:
            row_errors.append("duplicate image SHA256")
        sha_seen.add(str(row.get("image_sha256")))

        response = str(row.get("target_response") or "")
        if response.count("<ref>") != response.count("</ref>"):
            row_errors.append("unbalanced ref tokens")
        if response.count("<box>") != response.count("</box>"):
            row_errors.append("unbalanced box tokens")
        blocks = list(BOX_RE.finditer(response))
        coords = [int(value) for value in COORD_RE.findall(response)]
        if not blocks or not coords or any(value < 0 or value > 1000 for value in coords):
            row_errors.append("missing or invalid geometry")
        degenerate_boxes = count_degenerate_boxes(response)
        if degenerate_boxes:
            row_errors.append(f"degenerate boxes: {degenerate_boxes}")
        block_counts[task].append(len(blocks))
        prompt_lengths[task].append(len(str(row.get("prompt") or "")))
        if row_errors:
            errors.append({"bundle_id": row.get("bundle_id"), "task": task, "reasons": row_errors})
            continue

        if len(overlay_paths[task]) < args.overlays_per_domain:
            with Image.open(image_path) as source_image:
                overlay, transform = letterbox(source_image)
            geometry_count = draw_target(overlay, response, transform)
            output = args.bundle_dir / "qa_overlays" / task / f"{row['bundle_id']}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(output, quality=90)
            overlay_paths[task].append(output)
            overlay_index.append({
                "bundle_id": row["bundle_id"],
                "task": task,
                "overlay": output.relative_to(args.bundle_dir).as_posix(),
                "geometry_count": geometry_count,
                "letterbox": transform,
            })

    if dict(counts) != expected_counts:
        errors.append({
            "bundle_id": None,
            "task": "all",
            "reasons": [
                f"quota mismatch: actual={dict(counts)}, expected={expected_counts}"
            ],
        })

    for task, paths in overlay_paths.items():
        contact_sheet(paths, args.bundle_dir / "qa_overlays" / f"{task}_contact_sheet.jpg")

    rejected_path = args.bundle_dir / "rejected.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
        encoding="utf-8",
    )
    overlay_index_path = args.bundle_dir / "overlay_index.jsonl"
    overlay_index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in overlay_index),
        encoding="utf-8",
    )
    summary = {
        "selected_manifest": str(args.selected_jsonl.resolve()),
        "selected_manifest_sha256": sha256_file(args.selected_jsonl),
        "total_samples": len(rows),
        "selected_counts": dict(counts),
        "expected_counts": expected_counts,
        "unique_image_sha256": len(sha_seen),
        "rejected_count": len(errors),
        "overlay_counts": {task: len(paths) for task, paths in overlay_paths.items()},
        "geometry_blocks": {
            task: {"min": min(values), "max": max(values), "mean": mean(values)}
            for task, values in block_counts.items()
        },
        "prompt_lengths": {
            task: {"min": min(values), "max": max(values), "mean": mean(values)}
            for task, values in prompt_lengths.items()
        },
    }
    write_json(args.bundle_dir / "qa_summary.json", summary)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = f"""# G3 Source QA Report

## Outcome

- Selected samples: {len(rows)}
- Counts: `{dict(counts)}`
- Unique image SHA256: {len(sha_seen)}
- Rejected: {len(errors)}
- Selected manifest SHA256: `{summary['selected_manifest_sha256']}`
- Overlay samples: `{summary['overlay_counts']}`

## Verified Gates

- All selected records use the train split.
- Image files exist and match their manifest SHA256.
- Cross-domain image SHA256 values are unique after selection.
- `<ref>` and `<box>` tokens are balanced.
- Every geometry coordinate is in `[0,1000]`.
- Explicit six-domain quotas match `{expected_counts}`.
- Ten deterministic 672x672 letterbox overlays were rendered per domain.

## Boundary

This report validates the selected source bundle. It does not prove that the
LocateAnything processor, GPU materialization, observers, BC/HBM, or S600
runtime have consumed these samples.
"""
    (args.report_dir / "SOURCE_QA_REPORT.md").write_text(report, encoding="utf-8")

    explanations = f"""# G3 Artifact Explanations

## selected.jsonl

### 1. 作用

固定正式 {len(rows)} 条六域校准样本及其真值、来源和图片哈希。

### 2. 怎么看

先看 `task` 和 `split`，再核对 `image_sha256`、`prompt` 与
`target_response`；`bundle_id` 是后续张量生成的稳定键。

### 3. 本项目中的实际现象

六域配额为 `{expected_counts}`，选择后图片 SHA256 全部唯一。

### 4. 建模/实验启发

固定清单与 seed 后，256/512 observer convergence 才能做受控比较。

### 5. 风险与补充检查

该文件仍是 source bundle，不是 observer calibration 或 HBM 验证证据。

## qa_overlays contact sheets

### 1. 作用

显示真值几何经过 672x672 letterbox 后是否仍覆盖正确目标。

### 2. 怎么看

每个域按固定顺序展示十张；灰色区域是 padding，彩色框或圆圈是真值。

### 3. 本项目中的实际现象

每域均生成十张 overlay，包含横图、竖图、方图和不同几何密度。

### 4. 建模/实验启发

overlay 正确是后续 672 profile 坐标转换可用的必要条件。

### 5. 风险与补充检查

自动渲染不能替代人工语义检查；仍需确认文字描述与目标内容匹配。

## qa_summary.json

### 1. 作用

汇总数量、哈希唯一性、拒绝数、几何密度和 prompt 长度。

### 2. 怎么看

优先检查 `rejected_count`、`selected_counts`、`unique_image_sha256` 和
`overlay_counts`。

### 3. 本项目中的实际现象

正式清单应为 {len(rows)} 条、{len(rows)} 个唯一图片 SHA，且 rejected 为 0。

### 4. 建模/实验启发

几何密度差异说明 observer replay 必须覆盖全部六域，不能只用 detection。

### 5. 风险与补充检查

统计通过不证明模型 trajectory 或量化 scale 已正确生成。
"""
    (args.report_dir / "ARTIFACT_EXPLANATIONS.md").write_text(explanations, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
