#!/usr/bin/env python3
"""Materialize a compact COCO-2017 training subset from Hugging Face.

The output is a self-contained LocateAnything Detection source manifest.  It
keeps genuine COCO category names and uses the model's ``</c>`` category
separator, while downloading only the selected images into the output folder.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


DATASET_REPOSITORY = "detection-datasets/coco"
DATASET_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SEED = 20260728
DEFAULT_SINGLE = 200
DEFAULT_DOUBLE = 220
DEFAULT_MULTI = 80
DEFAULT_SHUFFLE_BUFFER = 2_048
MAX_CATEGORIES = 5
MAX_BOXES = 48


def stable_digest(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_coordinate(value: float, extent: int) -> int:
    return max(0, min(1000, int(round(float(value) / float(extent) * 1000))))


def normalized_box_xywh(
    bbox: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    values = list(bbox)
    if len(values) != 4:
        return None
    x, y, box_width, box_height = map(float, values)
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    x1 = normalized_coordinate(x, width)
    y1 = normalized_coordinate(y, height)
    x2 = normalized_coordinate(x + box_width, width)
    y2 = normalized_coordinate(y + box_height, height)
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def box_token(box: tuple[int, int, int, int]) -> str:
    return "<box>" + "".join(f"<{value}>" for value in box) + "</box>"


def choose_category_ids(
    available: Iterable[int], count: int, image_id: int, seed: int
) -> list[int]:
    return sorted(
        set(available), key=lambda category_id: stable_digest(seed, image_id, category_id)
    )[:count]


def category_count_for_stratum(
    stratum: str, available: int, image_id: int, seed: int
) -> int:
    if stratum == "single":
        return 1
    if stratum == "double":
        return 2
    upper = min(MAX_CATEGORIES, available)
    return 3 + int(stable_digest(seed, image_id, "count")[:8], 16) % (upper - 2)


def selection_stratum(category_count: int, remaining: Mapping[str, int]) -> str | None:
    # Reserve category-rich images first so later one-category selection does
    # not consume images needed by the multi-category quotas.
    if category_count >= 3 and remaining.get("multi", 0) > 0:
        return "multi"
    if category_count >= 2 and remaining.get("double", 0) > 0:
        return "double"
    if category_count >= 1 and remaining.get("single", 0) > 0:
        return "single"
    return None


@dataclass(frozen=True)
class Candidate:
    image_id: int
    width: int
    height: int
    boxes_by_category: dict[int, list[tuple[int, int, int, int]]]


def candidate_from_row(
    row: Mapping[str, Any], category_names: list[str], min_short_side: int
) -> Candidate | None:
    image_id = int(row["image_id"])
    width = int(row["width"])
    height = int(row["height"])
    if width <= 0 or height <= 0 or min(width, height) < min_short_side:
        return None

    objects = row.get("objects")
    if not isinstance(objects, Mapping):
        return None
    categories = objects.get("category")
    boxes = objects.get("bbox")
    if not isinstance(categories, list) or not isinstance(boxes, list):
        return None

    boxes_by_category: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for raw_category, raw_box in zip(categories, boxes):
        category_id = int(raw_category)
        if category_id < 0 or category_id >= len(category_names):
            continue
        if not isinstance(raw_box, (list, tuple)):
            continue
        box = normalized_box_xywh(raw_box, width, height)
        if box is not None:
            boxes_by_category[category_id].append(box)
    if not boxes_by_category:
        return None

    ordered = {
        category_id: sorted(values, key=lambda box: stable_digest(image_id, category_id, *box))
        for category_id, values in boxes_by_category.items()
    }
    return Candidate(image_id, width, height, ordered)


def select_boxes(
    candidate: Candidate, category_ids: list[int]
) -> dict[int, list[tuple[int, int, int, int]]]:
    selected = {category_id: [] for category_id in category_ids}
    while sum(map(len, selected.values())) < MAX_BOXES:
        changed = False
        for category_id in category_ids:
            index = len(selected[category_id])
            values = candidate.boxes_by_category[category_id]
            if index < len(values):
                selected[category_id].append(values[index])
                changed = True
                if sum(map(len, selected.values())) == MAX_BOXES:
                    break
        if not changed:
            break
    return selected


def image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy().convert("RGB")
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            with Image.open(io.BytesIO(raw)) as opened:
                return opened.convert("RGB")
    raise TypeError("Hugging Face image column is neither a decoded image nor byte payload")


def write_image(value: Any, path: Path, expected_size: tuple[int, int]) -> None:
    image = image_from_value(value)
    try:
        if image.size != expected_size:
            raise ValueError(
                f"image size mismatch for {path.name}: expected {expected_size}, got {image.size}"
            )
        temporary = path.with_suffix(".tmp")
        image.save(temporary, format="JPEG", quality=95, optimize=True)
        os.replace(temporary, path)
    finally:
        image.close()


def build_record(
    candidate: Candidate,
    category_names: list[str],
    stratum: str,
    image_path: Path,
    seed: int,
    repository: str,
    revision: str,
) -> dict[str, Any]:
    category_count = category_count_for_stratum(
        stratum, len(candidate.boxes_by_category), candidate.image_id, seed
    )
    category_ids = choose_category_ids(
        candidate.boxes_by_category, category_count, candidate.image_id, seed
    )
    selected = select_boxes(candidate, category_ids)
    names = [category_names[category_id] for category_id in category_ids]
    response = "".join(
        f"<ref>{category_names[category_id]}</ref>"
        + "".join(box_token(box) for box in selected[category_id])
        for category_id in category_ids
    )
    return {
        "sample_id": f"coco2017-train-{candidate.image_id:012d}-{stratum}",
        "task": "detection",
        "source": "COCO 2017 via Hugging Face",
        "split": "train",
        "license": "COCO 2017 terms",
        "license_url": "https://cocodataset.org/#termsofuse",
        "image": str(Path("images") / image_path.name),
        "image_sha256": sha256_file(image_path),
        "categories": names,
        "prompt": (
            "Locate all the instances that matches the following description: "
            + "</c>".join(names)
            + "."
        ),
        "target_response": response,
        "metadata": {
            "target_count": sum(len(boxes) for boxes in selected.values()),
            "category_count": len(category_ids),
            "category_ids": category_ids,
            "category_instance_counts": {
                category_names[category_id]: len(selected[category_id])
                for category_id in category_ids
            },
            "calibration_stratum": stratum,
            "coco_image_id": candidate.image_id,
            "hf_repository": repository,
            "hf_revision": revision,
        },
        "source_width": candidate.width,
        "source_height": candidate.height,
    }


def category_names_from_features(features: Any) -> list[str]:
    objects = features["objects"]
    object_feature = getattr(objects, "feature", None)
    category = object_feature["category"] if object_feature is not None else objects["category"]
    names = getattr(category, "names", None)
    if not isinstance(names, list) or not names:
        raise ValueError("COCO dataset does not expose named object categories")
    return [str(name) for name in names]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--repository", default=DATASET_REPOSITORY)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--single-category", type=int, default=DEFAULT_SINGLE)
    parser.add_argument("--two-category", type=int, default=DEFAULT_DOUBLE)
    parser.add_argument("--multi-category", type=int, default=DEFAULT_MULTI)
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = {
        "single": args.single_category,
        "double": args.two_category,
        "multi": args.multi_category,
    }
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) <= 0:
        raise ValueError("category quotas must be non-negative and cannot all be zero")
    if args.min_short_side <= 0 or args.shuffle_buffer <= 0:
        raise ValueError("--min-short-side and --shuffle-buffer must be positive")

    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    manifest = output_dir / "coco_detection.jsonl"
    if manifest.exists() and not args.force:
        raise FileExistsError(f"manifest already exists: {manifest}; pass --force to replace it")
    images_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    # Set this before importing datasets/huggingface_hub; their endpoint value
    # is read at import time.
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    try:
        from datasets import load_dataset
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required; install the compiler environment dependencies first"
        ) from exc

    dataset = load_dataset(
        args.repository,
        split="train",
        streaming=True,
        revision=args.revision,
        cache_dir=str(args.cache_dir.resolve()),
    )
    category_names = category_names_from_features(dataset.features)
    stream = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    remaining = dict(quotas)
    records: list[dict[str, Any]] = []
    scanned = 0
    print(
        f"streaming={args.repository}@{args.revision} "
        f"shuffle_buffer={args.shuffle_buffer}",
        flush=True,
    )
    progress = tqdm(
        stream,
        desc="COCO train stream",
        unit="row",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    progress.set_postfix(selected=0, single=0, double=0, multi=0)
    for row in progress:
        scanned += 1
        candidate = candidate_from_row(row, category_names, args.min_short_side)
        if candidate is None:
            continue
        stratum = selection_stratum(len(candidate.boxes_by_category), remaining)
        if stratum is None:
            continue

        image_path = images_dir / f"{candidate.image_id:012d}.jpg"
        write_image(row["image"], image_path, (candidate.width, candidate.height))
        records.append(
            build_record(
                candidate,
                category_names,
                stratum,
                image_path,
                args.seed,
                args.repository,
                args.revision,
            )
        )
        remaining[stratum] -= 1
        completed = {key: quotas[key] - remaining[key] for key in quotas}
        progress.set_postfix(selected=len(records), **completed)
        if not any(remaining.values()):
            break
    progress.close()

    if any(remaining.values()):
        raise RuntimeError(
            f"stream ended after {scanned} rows; unfilled quotas: {remaining}"
        )
    records.sort(key=lambda row: stable_digest(args.seed, row["sample_id"]))
    write_jsonl(manifest, records)
    print(f"repository={args.repository}@{args.revision}")
    print(f"scanned={scanned} selected={len(records)}")
    print(" ".join(f"{key}={quotas[key]}" for key in ("single", "double", "multi")))
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
