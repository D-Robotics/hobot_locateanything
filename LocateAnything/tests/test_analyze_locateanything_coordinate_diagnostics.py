from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "compiler" / "scripts" / "validate/analyze_coordinates.py"
SPEC = importlib.util.spec_from_file_location("la_coordinate_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def decoded(values):
    return {"type": "box", "coordinate_values": values, "fallback": False}


def comparison(iou, pixel, exact=0.0):
    return {
        "structure_agreement": 1.0,
        "coordinate_token_exact": exact,
        "pixel_mae": pixel / 2,
        "pixel_max_abs": pixel,
        "box_iou": iou,
    }


def decision(index, iou, pixel, *, top4_hit, exact=0.0):
    float_output = decoded([100, 200, 300, 400])
    quantized = decoded([102, 200, 298, 400])
    metric = comparison(iou, pixel, exact)
    return {
        "index": index,
        "source": {
            "kind": "box",
            "coordinate_values": [100, 200, 300, 400],
        },
        "float": float_output,
        "quantized_eager": quantized,
        "comparison": metric,
        "resolved": {"float": float_output, "quantized_eager": quantized},
        "resolved_comparison": metric,
        "ar_q1": {"float": None, "quantized_eager": None},
        "position_diagnostics": [
            {
                "position": 1,
                "comparison": {
                    "float_token_top4_hit": top4_hit,
                    "float_token_rank_in_quantized": 1.0 if top4_hit else 7.0,
                },
            }
        ],
    }


def test_coordinate_analysis_counts_tail_failures_and_pairwise_changes(tmp_path):
    report = {
        "status": "completed",
        "suite": "coordinate-u8",
        "experiments": [
            {
                "name": "baseline_s8",
                "samples": [
                    {
                        "id": "sample-a",
                        "coordinate_audit": {
                            "decisions": [decision(0, 0.40, 50.0, top4_hit=0.0)]
                        },
                    },
                    {
                        "id": "sample-b",
                        "coordinate_audit": {
                            "decisions": [
                                decision(0, 1.0, 0.0, top4_hit=1.0, exact=1.0)
                            ]
                        },
                    },
                ],
            },
            {
                "name": "u8",
                "samples": [
                    {
                        "id": "sample-a",
                        "coordinate_audit": {
                            "decisions": [decision(0, 0.95, 2.0, top4_hit=1.0)]
                        },
                    },
                    {
                        "id": "sample-b",
                        "coordinate_audit": {
                            "decisions": [
                                decision(0, 0.80, 8.0, top4_hit=1.0, exact=0.0)
                            ]
                        },
                    },
                ],
            },
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    analysis = MODULE.analyze(report_path)
    baseline = analysis["experiments"]["baseline_s8"]["summary"]
    delta = analysis["comparisons_to_baseline"]["u8"]

    assert baseline["box_iou_lt_0_5_count"] == 1
    assert baseline["pixel_max_gt_10_0_count"] == 1
    assert baseline["pbd_top4_miss_count"] == 1
    assert baseline["decode_paths"]["no_ar"]["box_count"] == 2
    assert baseline["decode_paths"]["both_ar"]["box_count"] == 0
    assert baseline["pbd_top4_groups"]["has_miss"]["box_iou"]["mean"] == 0.40
    assert delta["box_iou_improved_gt_0_01_count"] == 1
    assert delta["box_iou_regressed_gt_0_01_count"] == 1
    assert delta["box_iou_crossed_up_0_90_count"] == 1
    assert delta["coordinate_exact_loss_count"] == 1
    assert delta["pbd_top4_miss_delta"] == -1
    assert analysis["experiments"]["baseline_s8"]["worst_cases"][0][
        "sample_id"
    ] == "sample-a"

    assert MODULE.main([str(report_path)]) == 0
    output = json.loads((tmp_path / "coordinate_analysis.json").read_text())
    assert output["source_report_sha256"] == MODULE.sha256(report_path)
    assert (tmp_path / "coordinate_analysis.csv").is_file()
    assert (tmp_path / "coordinate_worst_cases.csv").is_file()


def test_statistics_preserves_counts_and_percentiles():
    values = [0.0, 0.5, 1.0]
    result = MODULE.statistics(values)

    assert result["count"] == 3
    assert result["mean"] == pytest.approx(0.5)
    assert result["median"] == pytest.approx(0.5)
    assert result["p05"] == pytest.approx(0.05)
    assert result["p95"] == pytest.approx(0.95)
