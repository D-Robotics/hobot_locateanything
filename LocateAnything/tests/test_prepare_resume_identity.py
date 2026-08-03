"""Tests for safe reuse of completed Prepare records."""

from compiler.scripts.calibration.prepare import (
    RESUME_IDENTITY_FIELDS,
    resume_identity_mismatches,
)
from compiler.scripts.common.identity import (
    identity_mismatches,
    prepare_source_is_compatible,
)


def identity_record():
    return {
        "bundle_id": "0001-gui-deadbeef0000",
        "task": "gui",
        "sample_id": "sample-1",
        "source": "GroundCUA",
        "split": "train",
        "image": "images/deadbeef.jpg",
        "image_sha256": "deadbeef",
        "prompt": "Point to: Undo button.",
        "target_response": "<ref>Undo button</ref><point><10><20></point>",
    }


def test_resume_identity_accepts_exact_selected_record():
    selected = identity_record()
    completed = {**selected, "tensor_sha256": "abc", "prediction": {}}
    assert resume_identity_mismatches(selected, completed) == []


def test_resume_identity_rejects_every_bound_field():
    selected = identity_record()
    for field in RESUME_IDENTITY_FIELDS:
        completed = dict(selected)
        completed[field] = "changed"
        assert resume_identity_mismatches(selected, completed) == [field]


def test_resume_identity_rejects_v5_prompt_for_v6_record():
    selected = identity_record()
    completed = dict(selected)
    completed["prompt"] = "Point to: Undo button.."
    completed["target_response"] = (
        "<ref>Undo button.</ref><point><10><20></point>"
    )
    assert resume_identity_mismatches(selected, completed) == [
        "prompt",
        "target_response",
    ]


def test_run_identity_reports_nested_checkpoint_and_generation_changes():
    expected = {
        "checkpoint": {"index": {"sha256": "a"}},
        "dtype": "bfloat16",
        "generation_config": {"temperature": 0.7},
    }
    actual = {
        "checkpoint": {"index": {"sha256": "b"}},
        "dtype": "float16",
        "generation_config": {"temperature": 0.9},
    }
    assert identity_mismatches(expected, actual) == [
        "checkpoint.index.sha256",
        "dtype",
        "generation_config.temperature",
    ]


def test_prepare_progress_only_source_update_is_compatible():
    previous = {
        "bytes": 45168,
        "sha256": "30e036d39ce00b1cd3c510d91385bf5c2df10e88ba42f6c091684090c01fd271",
    }
    current = {
        "bytes": 46335,
        "sha256": "57cf747532c6d3453291c9fd76bf853a03e17bed0edfa8629636771c2765123d",
    }
    unknown = {"bytes": 1, "sha256": "0" * 64}

    assert prepare_source_is_compatible(previous, current) is True
    assert prepare_source_is_compatible(previous, unknown) is False
