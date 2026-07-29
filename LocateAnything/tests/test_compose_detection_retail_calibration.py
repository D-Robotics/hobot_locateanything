from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "compiler"
    / "scripts"
    / "calibration/compose_detection.py"
)
SPEC = importlib.util.spec_from_file_location("compose_detection_retail", SCRIPT)
composer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(composer)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def detection(sample_id: str, stratum: str, source: str = "COCO") -> dict:
    return {
        "sample_id": sample_id,
        "task": "detection",
        "source": source,
        "source_dataset": "SKU110K_fixed" if source == "SKU110K" else "COCO",
        "split": "train",
        "license": "fixture",
        "image": f"images/{sample_id}.jpg",
        "image_sha256": sample_id.zfill(64),
        "categories": ["object"] if source == "SKU110K" else ["person"],
        "prompt": "legacy",
        "target_response": "<ref>object</ref><box><1><2><3><4></box>",
        "metadata": {"target_count": 1, "calibration_stratum": stratum},
    }


def test_composes_coco_retail_and_non_detection_with_table9_prompts(tmp_path):
    coco_path = tmp_path / "coco.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    output = tmp_path / "sources"
    write_jsonl(
        coco_path,
        [
            detection("1", "single"),
            detection("2", "double"),
            detection("3", "multi"),
        ],
    )
    write_jsonl(
        baseline_path,
        [
            detection("4", "single", "SKU110K"),
            {
                "sample_id": "5",
                "task": "layout",
                "source": "DocLayNet",
                "split": "train",
                "license": "fixture",
                "image": "images/5.jpg",
                "image_sha256": "5".zfill(64),
                "categories": ["title", "table"],
                "prompt": "legacy layout",
                "target_response": "<ref>title</ref><box><1><2><3><4></box>",
                "metadata": {"target_count": 1},
            },
        ],
    )

    summary = composer.compose(
        coco_path,
        baseline_path,
        output,
        {"single": 1, "double": 1, "multi": 1},
        retail_count=1,
        seed=7,
    )

    retail = composer.read_jsonl(output / "detection_retail.jsonl")
    other = composer.read_jsonl(output / "other_tasks.jsonl")
    assert summary["task_counts"] == {"detection": 4, "layout": 1}
    assert retail[0]["prompt"] == (
        "Locate all the instances that matches the following description: object."
    )
    assert other[0]["prompt"] == (
        "Detect all the objects in the image that belong to the category set: "
        "title</c>table."
    )
