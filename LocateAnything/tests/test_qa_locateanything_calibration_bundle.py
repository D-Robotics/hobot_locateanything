from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "compiler/scripts/calibration/qa.py"
SPEC = importlib.util.spec_from_file_location("qa_locateanything_bundle", SCRIPT)
qa = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qa)


def test_default_quotas_match_the_1200_record_release_contract():
    assert qa.parse_quotas(None) == {
        "detection": 620,
        "gui": 180,
        "referring": 120,
        "ocr": 120,
        "layout": 100,
        "pointing": 60,
    }


def test_explicit_v3_quotas():
    values = [
        "detection=208", "gui=130", "referring=83",
        "ocr=65", "layout=65", "pointing=50",
    ]
    assert qa.parse_quotas(values) == {
        "detection": 208, "gui": 130, "referring": 83,
        "ocr": 65, "layout": 65, "pointing": 50,
    }


def test_explicit_quotas_require_all_domains():
    with pytest.raises(ValueError, match="missing tasks"):
        qa.parse_quotas(["detection=1"])


def test_count_degenerate_boxes_ignores_points_and_valid_boxes():
    response = (
        "<ref>a</ref><box><10><20></box>"
        "<ref>b</ref><box><10><20><30><40></box>"
    )
    assert qa.count_degenerate_boxes(response) == 0


def test_count_degenerate_boxes_rejects_zero_or_reversed_extent():
    response = (
        "<ref>a</ref><box><10><20><10><40></box>"
        "<ref>b</ref><box><50><30><20><40></box>"
        "<ref>c</ref><box><10><20><30><20></box>"
    )
    assert qa.count_degenerate_boxes(response) == 3
