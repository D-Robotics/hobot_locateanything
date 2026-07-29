#!/usr/bin/env python3
"""Build a deterministic multi-category Detection source manifest from COCO."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_SINGLE = 200
DEFAULT_DOUBLE = 220
DEFAULT_MULTI = 80
DEFAULT_SEED = 20260728
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


def normalized_box(annotation: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, box_width, box_height = map(float, bbox)
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    x1 = normalized_coordinate(x, width)
    y1 = normalized_coordinate(y, height)
    x2 = normalized_coordinate(x + box_width, width)
    y2 = normalized_coordinate(y + box_height, height)
    return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


def box_token(box: tuple[int, int, int, int]) -> str:
    return "<box>" + "".join(f"<{value}>" for value in box) + "</box>"


def normalized_category_name(value: object) -> str:
    name = " ".join(str(value or "").strip().split())
    if not name or "</c>" in name or "<ref>" in name or "</ref>" in name:
        raise ValueError(f"invalid COCO category name: {value!r}")
    return name


def choose_category_ids(
    available: Iterable[int], count: int, image_id: int, seed: int
) -> list[int]:
    ranked = sorted(
        set(available), key=lambda category_id: stable_digest(seed, image_id, category_id)
    )
    return ranked[:count]


def select_boxes(
    annotations: dict[int, list[dict[str, Any]]],
    category_ids: list[int],
    width: int,
    height: int,
    image_id: int,
    seed: int,
) -> dict[int, list[tuple[int, int, int, int]]]:
    queues: dict[int, list[tuple[int, int, int, int]]] = {}
    for category_id in category_ids:
        values = []
        for annotation in annotations.get(category_id, []):
            box = normalized_box(annotation, width, height)
            if box is not None:
                values.append(box)
        values.sort(key=lambda box: stable_digest(seed, image_id, category_id, *box))
        if not values:
            raise ValueError(f"image {image_id} category {category_id} has no valid boxes")
        queues[category_id] = values

    selected: dict[int, list[tuple[int, int, int, int]]] = {
        category_id: [] for category_id in category_ids
    }
    while sum(map(len, selected.values())) < MAX_BOXES:
        changed = False
        for category_id in category_ids:
            index = len(selected[category_id])
            if index < len(queues[category_id]):
                selected[category_id].append(queues[category_id][index])
                changed = True
                if sum(map(len, selected.values())) == MAX_BOXES:
                    break
        if not changed:
            break
    return selected


def category_count_for_stratum(stratum: str, available: int, image_id: int, seed: int) -> int:
    if stratum == "single":
        return 1
    if stratum == "double":
        return 2
    upper = min(MAX_CATEGORIES, available)
    return 3 + int(stable_digest(seed, image_id, "count")[:8], 16) % (upper - 2)


def build_records(
    annotations_path: Path,
    image_dir: Path,
    quotas: dict[str, int],
    seed: int = DEFAULT_SEED,
    min_short_side: int = 224,
) -> list[dict[str, Any]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): normalized_category_name(item["name"])
        for item in payload.get("categories", [])
    }
    images = {int(item["id"]): item for item in payload.get("images", [])}
    licenses = {int(item["id"]): item for item in payload.get("licenses", [])}
    by_image: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for annotation in payload.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        if image_id in images and category_id in categories:
            by_image[image_id][category_id].append(annotation)

    eligible: dict[str, list[int]] = {"single": [], "double": [], "multi": []}
    for image_id, image in images.items():
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        path = image_dir / str(image.get("file_name") or "")
        category_count = len(by_image.get(image_id, {}))
        if width <= 0 or height <= 0 or min(width, height) < min_short_side or not path.is_file():
            continue
        if category_count >= 1:
            eligible["single"].append(image_id)
        if category_count >= 2:
            eligible["double"].append(image_id)
        if category_count >= 3:
            eligible["multi"].append(image_id)

    used_images: set[int] = set()
    records: list[dict[str, Any]] = []
    # Reserve category-rich images before filling the less restrictive strata.
    for stratum in ("multi", "double", "single"):
        ranked = sorted(
            eligible[stratum], key=lambda image_id: stable_digest(seed, stratum, image_id)
        )
        chosen = [image_id for image_id in ranked if image_id not in used_images][
            : quotas[stratum]
        ]
        if len(chosen) != quotas[stratum]:
            raise RuntimeError(
                f"COCO has only {len(chosen)} unused {stratum} candidates; "
                f"requested {quotas[stratum]}"
            )
        for image_id in chosen:
            image = images[image_id]
            width = int(image["width"])
            height = int(image["height"])
            category_count = category_count_for_stratum(
                stratum, len(by_image[image_id]), image_id, seed
            )
            category_ids = choose_category_ids(
                by_image[image_id], category_count, image_id, seed
            )
            selected = select_boxes(
                by_image[image_id], category_ids, width, height, image_id, seed
            )
            names = [categories[category_id] for category_id in category_ids]
            response = "".join(
                f"<ref>{categories[category_id]}</ref>"
                + "".join(box_token(box) for box in selected[category_id])
                for category_id in category_ids
            )
            image_path = (image_dir / str(image["file_name"])).resolve()
            with Image.open(image_path) as opened:
                if opened.size != (width, height):
                    raise ValueError(
                        f"COCO image size mismatch for {image_path}: "
                        f"JSON={width}x{height}, file={opened.size}"
                    )
            license_info = licenses.get(int(image.get("license", -1)), {})
            records.append(
                {
                    "sample_id": f"coco2017-train-{image_id:012d}-{stratum}",
                    "task": "detection",
                    "source": "COCO 2017",
                    "split": "train",
                    "license": str(license_info.get("name") or "see COCO image license"),
                    "license_url": str(license_info.get("url") or ""),
                    "image": str(image_path),
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
                            categories[category_id]: len(selected[category_id])
                            for category_id in category_ids
                        },
                        "calibration_stratum": stratum,
                        "coco_image_id": image_id,
                    },
                    "source_width": width,
                    "source_height": height,
                }
            )
            used_images.add(image_id)
    records.sort(key=lambda row: stable_digest(seed, row["sample_id"]))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--single-category", type=int, default=DEFAULT_SINGLE)
    parser.add_argument("--two-category", type=int, default=DEFAULT_DOUBLE)
    parser.add_argument("--multi-category", type=int, default=DEFAULT_MULTI)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-short-side", type=int, default=224)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = args.annotations.resolve()
    image_dir = args.image_dir.resolve()
    output = args.output_jsonl.resolve()
    if not annotations.is_file():
        raise FileNotFoundError(annotations)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to replace it: {output}")
    quotas = {
        "single": args.single_category,
        "double": args.two_category,
        "multi": args.multi_category,
    }
    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) <= 0:
        raise ValueError("category-stratum quotas must be non-negative and non-zero")
    records = build_records(
        annotations, image_dir, quotas, args.seed, args.min_short_side
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "source": "COCO 2017 train",
        "annotations": str(annotations),
        "annotations_sha256": sha256_file(annotations),
        "image_dir": str(image_dir),
        "seed": args.seed,
        "stratum_quotas": quotas,
        "records": len(records),
        "unique_images": len({row["metadata"]["coco_image_id"] for row in records}),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
