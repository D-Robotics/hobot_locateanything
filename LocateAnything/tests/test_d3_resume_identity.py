from compiler.scripts.calibration.prepare import (
    RESUME_IDENTITY_FIELDS,
    resume_identity_mismatches,
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
