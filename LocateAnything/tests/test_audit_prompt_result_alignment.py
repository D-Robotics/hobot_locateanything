from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "compiler"
    / "scripts"
    / "calibration/audit_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("audit_prompt_result_alignment", SCRIPT)
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_module)


def test_detection_prompt_preserves_real_multi_category_labels():
    row = {
        "task": "detection",
        "categories": ["person", "motorcycle"],
        "target_response": (
            "<ref>person</ref><box><10><20><30><40></box>"
            "<ref>motorcycle</ref><box><50><60><70><80></box>"
        ),
    }
    row["prompt"] = audit_module.expected_prompt(row)
    row["metadata"] = {"target_count": 2}

    errors, counts = audit_module.audit([row])

    assert errors == []
    assert counts["detection"] == 1
    assert "person</c>motorcycle" in row["prompt"]


def test_layout_prompt_uses_table9_template_and_category_separator():
    row = {
        "task": "layout",
        "categories": ["title", "table"],
        "target_response": (
            "<ref>title</ref><box><10><20><30><40></box>"
            "<ref>table</ref><box><50><60><70><80></box>"
        ),
        "metadata": {
            "target_count": 2,
            "layout_filter": {"unique_valid_boxes": 2},
        },
    }
    row["prompt"] = audit_module.expected_prompt(row)

    errors, counts = audit_module.audit([row])

    assert errors == []
    assert counts["layout"] == 1
    assert row["prompt"] == (
        "Detect all the objects in the image that belong to the category set: "
        "title</c>table."
    )
