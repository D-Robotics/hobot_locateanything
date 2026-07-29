from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).parents[1]
    / "compiler"
    / "scripts"
    / "calibration/materialize_coco.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_hf_coco", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


NAMES = ["person", "motorcycle", "cat", "dog"]


def test_candidate_uses_real_categories_and_normalized_boxes():
    candidate = module.candidate_from_row(
        {
            "image_id": 17,
            "width": 640,
            "height": 480,
            "objects": {
                "category": [1, 2, 2],
                "bbox": [[64, 48, 320, 240], [1, 2, 0, 3], [320, 120, 160, 120]],
            },
        },
        NAMES,
        min_short_side=224,
    )

    assert candidate is not None
    assert sorted(candidate.boxes_by_category) == [1, 2]
    assert candidate.boxes_by_category[1] == [(100, 100, 600, 600)]
    assert candidate.boxes_by_category[2] == [(500, 250, 750, 500)]


def test_selection_reserves_multi_category_capacity():
    assert module.DEFAULT_SHUFFLE_BUFFER == 2_048
    remaining = {"single": 200, "double": 220, "multi": 80}
    assert module.selection_stratum(3, remaining) == "multi"
    remaining["multi"] = 0
    assert module.selection_stratum(3, remaining) == "double"
    remaining["double"] = 0
    assert module.selection_stratum(3, remaining) == "single"


def test_record_uses_category_separator_and_grouped_references(tmp_path):
    image_path = tmp_path / "000000000017.jpg"
    Image.new("RGB", (640, 480), "white").save(image_path)
    candidate = module.Candidate(
        image_id=17,
        width=640,
        height=480,
        boxes_by_category={
            0: [(100, 100, 200, 200)],
            1: [(300, 200, 500, 600)],
            2: [(500, 300, 750, 700)],
        },
    )

    record = module.build_record(
        candidate,
        NAMES,
        "multi",
        image_path,
        seed=3,
        repository="fixture/coco",
        revision="fixture",
    )

    assert "</c>" in record["prompt"]
    assert record["target_response"].count("<ref>") == 3
    assert record["metadata"]["target_count"] == 3
    assert record["split"] == "train"
