from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).parents[1]
    / "compiler"
    / "scripts"
    / "calibration/build_coco.py"
)
SPEC = importlib.util.spec_from_file_location("build_coco_multiclass_detection", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "train2017"
    image_dir.mkdir()
    images = []
    annotations = []
    annotation_id = 1
    category_sets = {
        1: [1, 2, 3],
        2: [1, 2],
        3: [3],
    }
    for image_id, category_ids in category_sets.items():
        name = f"{image_id:012d}.jpg"
        Image.new("RGB", (640, 480), (image_id * 30, 80, 120)).save(image_dir / name)
        images.append({
            "id": image_id,
            "file_name": name,
            "width": 640,
            "height": 480,
            "license": 1,
        })
        for offset, category_id in enumerate(category_ids):
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [10 + offset * 100, 20, 80, 100],
                "iscrowd": 0,
            })
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "motorcycle"},
            {"id": 3, "name": "dog"},
        ],
        "licenses": [{"id": 1, "name": "fixture", "url": "https://example.test"}],
    }
    annotations_path = tmp_path / "instances_train2017.json"
    annotations_path.write_text(json.dumps(payload), encoding="utf-8")
    return annotations_path, image_dir


def test_builds_unique_single_double_and_multi_category_records(tmp_path):
    annotations, image_dir = fixture(tmp_path)
    rows = builder.build_records(
        annotations,
        image_dir,
        {"single": 1, "double": 1, "multi": 1},
        seed=7,
    )

    assert len(rows) == 3
    assert len({row["image"] for row in rows}) == 3
    by_stratum = {row["metadata"]["calibration_stratum"]: row for row in rows}
    assert len(by_stratum["single"]["categories"]) == 1
    assert len(by_stratum["double"]["categories"]) == 2
    assert len(by_stratum["multi"]["categories"]) == 3
    for row in rows:
        assert row["prompt"].endswith("</c>".join(row["categories"]) + ".")
        assert row["target_response"].count("<ref>") == len(row["categories"])
        assert row["target_response"].count("<box>") == row["metadata"]["target_count"]


def test_ignores_crowd_and_rejects_invalid_category_names(tmp_path):
    assert builder.normalized_category_name("traffic light") == "traffic light"
    try:
        builder.normalized_category_name("cat</c>dog")
    except ValueError as error:
        assert "invalid COCO category" in str(error)
    else:
        raise AssertionError("separator injection was accepted")
