import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "validate/evaluate_grounding.py"
spec = importlib.util.spec_from_file_location("evaluate_locateanything_grounding", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


def response(label="item", geometry="<box><0><0><100><100></box>"):
    return f"<ref>{label}</ref>{geometry}"


def test_parse_response_preserves_ref_geometry_association():
    parsed = evaluator.parse_response(
        "<ref>Button</ref><box><10><20></box><ref>Panel</ref><box><0><0><40><50></box>"
    )
    assert parsed.valid
    assert parsed.refs == ("Button", "Panel")
    assert [(value.label, value.kind) for value in parsed.geometries] == [
        ("Button", "point"),
        ("Panel", "box"),
    ]


def test_parse_response_allows_ocr_comparison_sign_in_label():
    parsed = evaluator.parse_response("<ref><30</ref><box><10><20><30><40></box>")
    assert parsed.valid
    assert parsed.refs == ("<30",)


def test_parse_response_accepts_only_locateanything_terminal_eos():
    expected = response()
    parsed = evaluator.parse_response(expected + "\n<|im_end|>\n")
    assert parsed.valid
    assert parsed.terminal_token == "<|im_end|>"
    assert parsed.geometries == evaluator.parse_response(expected).geometries


@pytest.mark.parametrize("suffix", ["</s>", "<|endoftext|>", "explanation"])
def test_parse_response_rejects_non_eos_trailing_content(suffix):
    parsed = evaluator.parse_response(response() + suffix)
    assert not parsed.valid
    assert "expected <ref>" in parsed.error


def test_parse_response_rejects_repeated_im_end():
    parsed = evaluator.parse_response(response() + "<|im_end|><|im_end|>")
    assert not parsed.valid


@pytest.mark.parametrize(
    "value,error",
    [
        ("free text", "expected <ref>"),
        ("<ref>x</ref>", "no following geometry"),
        (response(geometry="<box><0><0><0><10></box>"), "degenerate"),
        (response(geometry="<box><0><1001></box>"), "outside"),
        (response(geometry="<box><0><0><10></box>"), "no following geometry"),
    ],
)
def test_parse_response_rejects_malformed_output(value, error):
    parsed = evaluator.parse_response(value)
    assert not parsed.valid
    assert error in parsed.error


def test_box_metrics_are_label_aware_and_include_invalid_predictions():
    rows = [
        {
            "bundle_id": "a",
            "task": "referring",
            "profile_target_response": response("item"),
            "prediction": {"hybrid": {"answer": response("item", "<box><0><0><50><100></box>")}},
        },
        {
            "bundle_id": "b",
            "task": "referring",
            "profile_target_response": response("item"),
            # Same coordinates but the wrong ref must not match.
            "prediction": {"hybrid": {"answer": response("other")}},
        },
        {
            "bundle_id": "c",
            "task": "referring",
            "profile_target_response": response("item"),
            "prediction": {"hybrid": {"answer": "not grounding syntax"}},
        },
    ]
    result, details = evaluator.evaluate(rows, iou_threshold=0.5)
    metrics = result["modes"]["hybrid"]["overall"]
    assert metrics["format"]["valid_rate"] == pytest.approx(2 / 3)
    assert metrics["label_ref"]["true_positive"] == 1
    assert metrics["label_ref"]["precision"] == pytest.approx(0.5)
    assert metrics["label_ref"]["recall"] == pytest.approx(1 / 3)
    assert metrics["box_iou"]["true_positive"] == 1
    assert metrics["box_iou"]["precision"] == pytest.approx(0.5)
    assert metrics["box_iou"]["recall"] == pytest.approx(1 / 3)
    assert metrics["single_box_iou"]["mean_iou"] == pytest.approx(1 / 6)
    assert len(details) == 3


def test_label_matching_is_case_sensitive_for_ocr_fidelity():
    rows = [{
        "bundle_id": "ocr",
        "task": "ocr",
        "target_response": response("CDC"),
        "answer": response("cdc"),
    }]
    result, _ = evaluator.evaluate(rows)
    metrics = result["modes"]["s600"]["overall"]
    assert metrics["label_ref"]["true_positive"] == 0
    assert metrics["box_iou"]["true_positive"] == 0


def test_point_pck_and_distance_use_grid_diagonal_normalization():
    rows = [{
        "bundle_id": "point",
        "task": "pointing",
        "profile_target_response": response("button", "<box><100><100></box>"),
        "slow_answer": response("button", "<box><200><100></box>"),
    }]
    result, _ = evaluator.evaluate(rows)
    point = result["modes"]["slow"]["overall"]["point"]
    assert point["target_mean_distance_grid"] == 100
    assert point["target_mean_distance_normalized_diagonal"] == pytest.approx(
        100 / (1000 * 2 ** 0.5)
    )
    assert point["pck"]["0.05"]["recall"] == 0
    assert point["pck"]["0.1"]["recall"] == 1


def test_flat_s600_rows_join_to_reference_and_score_all_six_domains():
    references = []
    predictions = []
    for index, task in enumerate(evaluator.TASKS):
        target = response("thing")
        references.append({
            "bundle_id": str(index),
            "task": task,
            "profile_target_response": target,
        })
        predictions.append({
            "bundle_id": str(index),
            "mode": "s600_hybrid",
            "answer": target,
        })
    result, _ = evaluator.evaluate(predictions, references)
    mode = result["modes"]["s600_hybrid"]
    assert mode["overall"]["structured_exact_match_rate"] == 1
    assert set(mode["by_task"]) == set(evaluator.TASKS)
    assert all(mode["by_task"][task]["records"] == 1 for task in evaluator.TASKS)


def test_reference_manifest_keeps_entirely_missing_s600_row_in_denominator():
    references = [
        {"bundle_id": "present", "task": "gui", "target_response": response()},
        {"bundle_id": "missing", "task": "gui", "target_response": response()},
    ]
    predictions = [{"bundle_id": "present", "mode": "s600", "answer": response()}]
    result, details = evaluator.evaluate(predictions, references)
    mode = result["modes"]["s600"]
    assert mode["prediction_coverage"] == {"present": 1, "total": 2, "rate": 0.5}
    assert mode["overall"]["format"]["valid_rate"] == 0.5
    assert mode["overall"]["box_iou"]["recall"] == 0.5
    assert mode["available_predictions_only"]["overall"]["records"] == 1
    assert mode["available_predictions_only"]["overall"]["format"]["valid_rate"] == 1
    assert mode["available_predictions_only"]["overall"]["box_iou"]["recall"] == 1
    missing = next(value for value in details if value["bundle_id"] == "missing")
    assert not missing["prediction_present"]


def test_profile_target_on_prediction_beats_unprofiled_reference_target():
    original = response(geometry="<box><0><0><100><100></box>")
    profiled = response(geometry="<box><10><20><110><120></box>")
    references = [{"bundle_id": "a", "task": "gui", "target_response": original}]
    predictions = [{
        "bundle_id": "a",
        "mode": "s600",
        "answer": profiled,
        "profile_target_response": profiled,
    }]
    result, details = evaluator.evaluate(predictions, references)
    assert result["modes"]["s600"]["overall"]["structured_exact_match_rate"] == 1
    assert details[0]["target_field"] == "prediction.profile_target_response"


def test_missing_requested_mode_is_scored_as_invalid():
    rows = [{
        "bundle_id": "a",
        "task": "gui",
        "profile_target_response": response(),
        "prediction": {"hybrid": {"answer": response()}},
    }]
    result, _ = evaluator.evaluate(rows, modes=["hybrid", "slow"])
    assert result["modes"]["hybrid"]["overall"]["format"]["valid_rate"] == 1
    assert result["modes"]["slow"]["overall"]["format"]["valid_rate"] == 0
    assert result["modes"]["slow"]["overall"]["box_iou"]["recall"] == 0


def test_invalid_reference_fails_instead_of_silently_corrupting_accuracy():
    rows = [{
        "bundle_id": "bad-target",
        "task": "layout",
        "target_response": "bad",
        "answer": response(),
    }]
    with pytest.raises(ValueError, match="invalid reference"):
        evaluator.evaluate(rows)


def test_cli_writes_summary_and_per_record_details(tmp_path, monkeypatch):
    source = tmp_path / "generated.jsonl"
    source.write_text(json.dumps({
        "bundle_id": "a",
        "task": "detection",
        "profile_target_response": response(),
        "prediction": {"hybrid": {"answer": response()}},
    }) + "\n", encoding="utf-8")
    output = tmp_path / "metrics.json"
    details = tmp_path / "details.jsonl"

    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--generated-jsonl", str(source), "--output-json", str(output),
        "--details-jsonl", str(details),
    ])
    assert evaluator.main() == 0

    assert json.loads(output.read_text(encoding="utf-8"))["record_count"] == 1
    assert len(details.read_text(encoding="utf-8").splitlines()) == 1
