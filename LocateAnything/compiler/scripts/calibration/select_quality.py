#!/usr/bin/env python3
"""Select a high-quality, diverse, prompt-aligned LA calibration bundle.

The selector is intentionally conservative: it rejects geometry/schema issues,
low-information images, prompt/target mismatches, OCR label loss, and layout
truncation before applying deterministic diversity-aware sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def normalize_phrase(value: Any) -> str:
    """Normalize source prose before inserting it into a canonical prompt."""
    phrase = " ".join(str(value or "").strip().split())
    return phrase.rstrip(" \t\r\n.,!?;:")


def normalize_response_refs(response: str) -> str:
    return REF_RE.sub(
        lambda match: f"<ref>{normalize_phrase(match.group(1))}</ref>", response
    )


class Progress:
    """Dependency-free progress reporter for local and CI runs."""

    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, total)
        self.started = time.monotonic()
        self.last = -1

    def update(self, current: int) -> None:
        percent = min(100, current * 100 // self.total)
        if percent == self.last or (percent % 5 and current != self.total):
            return
        self.last = percent
        elapsed = time.monotonic() - self.started
        print(
            f"[progress] {self.label}: {current}/{self.total} ({percent:3d}%) "
            f"elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )


TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")
BOX_RE = re.compile(r"<box>((?:<\d+>){2}|(?:<\d+>){4})</box>")
COORD_RE = re.compile(r"<(\d+)>")
REF_RE = re.compile(r"<ref>(.*?)</ref>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def excluded_image_sha256(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in read_jsonl(path.resolve()):
            value = str(row.get("image_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"exclude manifest has invalid image_sha256: {path}")
            excluded.add(value)
    return excluded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(
        f"{salt}|{row.get('task')}|{row.get('sample_id')}|{row.get('image_sha256')}".encode()
    ).hexdigest()


def dhash(image: Image.Image) -> int:
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(pixels.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(values[y * 9 + x + 1] > values[y * 9 + x])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def image_quality(path: Path) -> dict[str, float]:
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        small = image.resize((128, 128), Image.Resampling.BILINEAR)
        values = list(small.getdata())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    histogram = Counter(values)
    entropy = -sum(
        (count / len(values)) * math.log2(count / len(values))
        for count in histogram.values()
        if count
    )
    gradient = sum(
        abs(values[y * 128 + x + 1] - values[y * 128 + x])
        for y in range(128)
        for x in range(127)
    ) / (128 * 127)
    return {
        "width": float(width),
        "height": float(height),
        "short_side": float(min(width, height)),
        "aspect_ratio": float(max(width / height, height / width)),
        "gray_std": math.sqrt(variance),
        "entropy": entropy,
        "gradient_mean": gradient,
        "dhash": float(dhash(image)),
    }


def geometries(response: str) -> list[tuple[int, ...]]:
    result = []
    for match in BOX_RE.finditer(response):
        values = tuple(int(value) for value in COORD_RE.findall(match.group(1)))
        result.append(values)
    return result


def refs(response: str) -> list[str]:
    return [value.strip() for value in REF_RE.findall(response) if value.strip()]


def response_valid(row: dict[str, Any]) -> tuple[bool, str]:
    task = row["task"]
    response = str(row.get("target_response") or "")
    labels = refs(response)
    geo = geometries(response)
    if not labels or not geo:
        return False, "missing labels or geometry"
    if response.count("<ref>") != response.count("</ref>"):
        return False, "unbalanced ref tokens"
    if response.count("<box>") != response.count("</box>"):
        return False, "unbalanced box tokens"
    for item in geo:
        if any(value < 0 or value > 1000 for value in item):
            return False, "coordinate outside [0,1000]"
        if len(item) == 4 and (item[2] <= item[0] or item[3] <= item[1]):
            return False, "degenerate box"

    target_count = (row.get("metadata") or {}).get("target_count")
    if not isinstance(target_count, int) or target_count != len(geo):
        return False, "metadata target_count disagrees with geometry count"

    phrase = normalize_phrase(row.get("phrase"))
    normalized_labels = {normalize_phrase(label) for label in labels}
    if task == "detection":
        categories = [normalize_phrase(value) for value in row.get("categories") or []]
        if not categories or normalized_labels != set(categories):
            return False, "detection categories and target refs disagree"
    if task in {"gui", "referring"} and (not phrase or normalized_labels != {phrase}):
        return False, "prompt phrase and target ref disagree"
    if task == "pointing" and (not phrase or normalized_labels != {phrase}):
        return False, "pointing prompt and target ref disagree"
    if task == "ocr":
        stats = (row.get("metadata") or {}).get("hiertext_filter") or {}
        if stats.get("dropped_non_positive_extent") or stats.get(
            "dropped_degenerate_after_normalization"
        ):
            return False, "OCR label loss"
        if stats.get("parsed_word_boxes", 0) > 48:
            return False, "OCR target truncation"
        if stats.get("parsed_word_boxes") != target_count:
            return False, "OCR source label count disagrees with target count"
    if task == "layout":
        stats = (row.get("metadata") or {}).get("layout_filter") or {}
        if stats.get("invalid_source_boxes") or stats.get(
            "degenerate_after_normalization"
        ):
            return False, "layout invalid geometry"
        if stats.get("unique_valid_boxes", 0) > 48:
            return False, "layout target truncation"
        if stats.get("unique_valid_boxes") != target_count:
            return False, "layout source label count disagrees with target count"
        categories = set(row.get("categories") or [])
        if not categories or not set(labels).issubset(categories):
            return False, "layout category and target ref disagree"
    return True, ""


def canonical_prompt(row: dict[str, Any]) -> str:
    task = row["task"]
    labels = list(dict.fromkeys(refs(str(row["target_response"]))))
    phrase = normalize_phrase(row.get("phrase"))
    if task == "detection":
        categories = [normalize_phrase(value) for value in row.get("categories") or labels]
        return (
            "Locate all the instances that matches the following description: "
            + "</c>".join(categories)
            + "."
        )
    if task == "gui":
        if str(row.get("output_type")) == "point":
            return f"Point to: {phrase}."
        return f"Locate the region that matches the following description: {phrase}."
    if task == "referring":
        return f"Locate a single instance that matches the following description: {phrase}."
    if task == "ocr":
        return "Detect all the text in box format."
    if task == "layout":
        categories = [normalize_phrase(value) for value in row.get("categories") or labels]
        return (
            "Detect all the objects in the image that belong to the category set: "
            + "</c>".join(categories)
            + "."
        )
    if task == "pointing":
        return f"Point to: {phrase}."
    raise ValueError(f"unsupported task: {task}")


def diversity_key(row: dict[str, Any], quality: dict[str, float]) -> tuple[Any, ...]:
    task = row["task"]
    aspect = quality["aspect_ratio"]
    aspect_bin = "square" if aspect <= 1.2 else "standard" if aspect <= 1.6 else "wide"
    geo_bin = min(4, max(1, len(geometries(str(row["target_response"]))) // 8 + 1))
    metadata = row.get("metadata") or {}
    if task == "gui":
        return (str(metadata.get("software") or "unknown"), aspect_bin, geo_bin)
    if task == "layout":
        return (str(metadata.get("document_category") or "unknown"), aspect_bin, geo_bin)
    if task == "pointing":
        return (str(metadata.get("collection_method") or "unknown"), aspect_bin, geo_bin)
    if task == "ocr":
        return (aspect_bin, geo_bin)
    if task == "referring":
        phrase_len = len(str(row.get("phrase") or ""))
        return (aspect_bin, "short" if phrase_len < 25 else "long", geo_bin)
    if task == "detection":
        category_count = len(row.get("categories") or [])
        category_bin = "1" if category_count <= 1 else "2" if category_count == 2 else "3+"
        return (category_bin, aspect_bin, geo_bin)
    return (aspect_bin, geo_bin)


def select_diverse(
    rows: list[dict[str, Any]],
    qualities: dict[str, dict[str, float]],
    quota: int,
    salt: str,
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[diversity_key(row, qualities[row["_candidate_id"]])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: stable_key(row, salt))
    selected: list[dict[str, Any]] = []
    used_hashes: list[int] = []
    keys = sorted(groups, key=lambda key: stable_key({"task": str(key), "sample_id": ""}, salt))
    cursor = 0
    near_duplicate_drops = 0
    while len(selected) < quota and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        if not groups[key]:
            keys.remove(key)
            continue
        row = groups[key].pop(0)
        perceptual = int(qualities[row["_candidate_id"]]["dhash"])
        if any(hamming(perceptual, previous) <= 4 for previous in used_hashes):
            near_duplicate_drops += 1
            continue
        selected.append(row)
        used_hashes.append(perceptual)
    return selected, near_duplicate_drops


def resolve_image(row: dict[str, Any], manifest_path: Path) -> Path:
    image = Path(str(row["image"]))
    if not image.is_absolute():
        image = manifest_path.parent / image
    return image.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="TASK=JSONL")
    parser.add_argument("--quota", action="append", required=True, metavar="TASK=N")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="exclude every image SHA256 present in this JSONL; repeatable",
    )
    args = parser.parse_args()

    input_paths: dict[str, Path] = {}
    for value in args.input:
        task, raw = value.split("=", 1)
        if task not in TASKS or task in input_paths:
            raise ValueError(f"invalid or duplicate input task: {task}")
        input_paths[task] = Path(raw).resolve()
    quotas = {}
    for value in args.quota:
        task, raw = value.split("=", 1)
        quotas[task] = int(raw)
    if set(quotas) != set(TASKS):
        raise ValueError("quotas must cover all six tasks")

    excluded_sha256 = excluded_image_sha256(args.exclude_jsonl)

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualities: dict[str, dict[str, float]] = {}
    rejected: list[dict[str, Any]] = []
    candidate_counts = {}
    excluded_counts: Counter[str] = Counter()
    for task in TASKS:
        manifest = input_paths[task]
        rows = read_jsonl(manifest)
        candidate_counts[task] = len(rows)
        progress = Progress(f"quality-gate:{task}", len(rows))
        for index, row in enumerate(rows):
            row = dict(row)
            row["task"] = task
            candidate_id = f"{task}:{index}:{row.get('sample_id')}"
            row["_candidate_id"] = candidate_id
            if row.get("image_sha256") in excluded_sha256:
                excluded_counts[task] += 1
                progress.update(index + 1)
                continue
            try:
                image_path = resolve_image(row, manifest)
                if not image_path.is_file():
                    raise ValueError("missing image")
                quality = image_quality(image_path)
                if quality["short_side"] < 224:
                    raise ValueError("short side < 224")
                if quality["entropy"] < 1.0 or quality["gray_std"] < 5.0:
                    raise ValueError("low-information image")
                valid, reason = response_valid(row)
                if not valid:
                    raise ValueError(reason)
                row["_image_path"] = str(image_path)
                row["source_prompt"] = row.get("prompt")
                row["source_target_response"] = row.get("target_response")
                if task in {"gui", "referring", "pointing"}:
                    row["phrase"] = normalize_phrase(row.get("phrase"))
                    row["target_response"] = normalize_response_refs(
                        str(row["target_response"])
                    )
                row["prompt"] = canonical_prompt(row)
                qualities[candidate_id] = quality
                candidates[task].append(row)
            except Exception as exc:
                rejected.append({
                    "task": task,
                    "sample_id": row.get("sample_id"),
                    "reason": str(exc),
                })
            finally:
                progress.update(index + 1)

    selected: list[dict[str, Any]] = []
    selection_stats = {}
    for task in TASKS:
        print(
            f"[select] {task}: quota={quotas[task]} "
            f"candidates={len(candidates[task])}"
        )
        chosen, near_drops = select_diverse(
            candidates[task], qualities, quotas[task], f"{args.seed}|{task}"
        )
        if len(chosen) < quotas[task]:
            raise RuntimeError(
                f"{task} has only {len(chosen)} diverse candidates after quality gates; "
                f"requested {quotas[task]}"
            )
        selection_stats[task] = {
            "quality_candidates": len(candidates[task]),
            "selected": len(chosen),
            "near_duplicate_drops": near_drops,
            "strata_selected": dict(Counter(str(diversity_key(r, qualities[r["_candidate_id"]])) for r in chosen)),
        }
        selected.extend(chosen)

    # Enforce global image uniqueness after domain-level diversity selection.
    selected.sort(key=lambda row: (TASKS.index(row["task"]), stable_key(row, str(args.seed))))
    seen_sha: set[str] = set()
    materialized: list[dict[str, Any]] = []
    for row in selected:
        if row["image_sha256"] in seen_sha:
            raise RuntimeError(f"cross-domain duplicate survived selection: {row['sample_id']}")
        seen_sha.add(row["image_sha256"])
        source_image = Path(row["_image_path"])
        destination = image_dir / f"{row['image_sha256']}{source_image.suffix.lower() or '.jpg'}"
        if not destination.exists():
            shutil.copy2(source_image, destination)
        output = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        output["schema_version"] = 2
        output["bundle_id"] = f"{len(materialized):04d}-{row['task']}-{row['image_sha256'][:12]}"
        output["image"] = destination.relative_to(output_dir).as_posix()
        output["prompt_policy"] = "upstream_locateanything_task_prompt_v2"
        materialized.append(output)

    selected_path = output_dir / "selected.jsonl"
    selected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "selection_seed": args.seed,
        "candidate_counts": candidate_counts,
        "excluded_image_sha256_count": len(excluded_sha256),
        "excluded_counts_by_task": dict(excluded_counts),
        "quality_candidate_counts": {task: len(candidates[task]) for task in TASKS},
        "selected_counts": dict(Counter(row["task"] for row in materialized)),
        "quotas": quotas,
        "rejected_count": len(rejected),
        "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
        "selection_stats": selection_stats,
        "unique_images": len(seen_sha),
        "selected_manifest_sha256": sha256_file(selected_path),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "rejected.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejected),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
