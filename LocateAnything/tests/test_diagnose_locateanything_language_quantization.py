from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "compiler" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "validate/diagnose_language.py"
SPEC = importlib.util.spec_from_file_location("la_language_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_focused_suite_is_single_variable_and_includes_known_hotspots():
    experiments = dict(MODULE.focused_experiments())

    assert list(experiments)[:4] == [
        "baseline",
        "lm_head_float_weight",
        "lm_head_float_activation",
        "lm_head_float",
    ]
    assert experiments["baseline"] == []
    assert experiments["block2_value_cache"][0].module_pattern == (
        r"layers\.2\.self_attn\.cache_v_fq"
    )
    assert experiments["gui_ar_block7_attention"][0].stages == ("ar_q1",)
    assert all(
        rule.stages == ("pbd_q6",)
        for rule in experiments["ocr_pbd_blocks8_33_attention"]
    )


def test_suffix_suite_expands_from_last_block_to_full_decoder():
    experiments = MODULE.suffix_experiments(num_layers=4)

    assert [name for name, _rules in experiments] == [
        "baseline",
        "decode_suffix_3_3",
        "decode_suffix_2_3",
        "decode_suffix_1_3",
        "decode_suffix_0_3",
        "decode_all_layers_and_lm_head_float",
    ]
    assert experiments[1][1][0].module_pattern == r"layers\.(?:3)\..*"
    assert experiments[-2][1][0].module_pattern == r"layers\.(?:0|1|2|3)\..*"
    assert experiments[-1][1][1].module_pattern == r"lm_head"


def test_refine_suite_adds_back_each_block_and_splits_block9_paths():
    experiments = dict(MODULE.refine_experiments(num_layers=12))

    assert "decode_suffix_9_11" in experiments
    assert "decode_suffix_9_11_addback_block_9" in experiments
    assert "decode_suffix_9_11_addback_block_11" in experiments
    assert "decode_suffix_10_11_plus_block9_attention" in experiments
    assert "decode_suffix_10_11_plus_block9_value_path" in experiments
    assert experiments["decode_suffix_9_11"][0].module_pattern == (
        r"layers\.(?:9|10|11)\..*"
    )
    assert experiments["decode_suffix_9_11_addback_block_10"][0].module_pattern == (
        r"layers\.(?:9|11)\..*"
    )


def test_path_suite_separates_q_weight_activation_and_wv_operands():
    experiments = dict(MODULE.path_experiments())

    q_weight = experiments["all_q_projection_float_weight"][0]
    q_activation = experiments["all_q_projection_float_activation"][0]
    wv_attention = experiments["all_wv_attention_operand_float"][0]
    wv_value = experiments["all_wv_value_operand_float"][0]
    assert q_weight.mode == "float_weight"
    assert q_activation.mode == "float_activation"
    assert wv_attention.module_pattern.endswith(r"wv_matmul\.x_fake_quant")
    assert wv_value.module_pattern.endswith(r"wv_matmul\.y_fake_quant")
    assert len(experiments["sensitive_q_activation_and_wv_float"]) == 2


def test_dynamic_suite_separates_attention_and_cache_quantizers():
    names = [name for name, _rules in MODULE.dynamic_experiments()]

    assert names[0] == "baseline_static"
    assert MODULE.dynamic_patterns("baseline_static") == []
    assert len(MODULE.dynamic_patterns("dynamic_wv_operands")) == 2
    assert len(MODULE.dynamic_patterns("dynamic_attention_and_kv_cache")) == 5
    assert "cache_v_fq" in MODULE.dynamic_patterns("dynamic_value_cache")[0]


def test_coordinate_suite_separates_prefill_and_coordinate_pbd_causes():
    experiments = dict(MODULE.coordinate_experiments())

    assert list(experiments)[:4] == [
        "baseline",
        "prefill_all_float",
        "pbd_all_float",
        "prefill_and_pbd_all_float",
    ]
    assert experiments["prefill_all_float"][0].stages == ("prefill",)
    assert experiments["pbd_all_float"][0].stages == ("pbd_q6",)
    assert experiments["pbd_all_linear_float_activation"][0].mode == (
        "float_activation"
    )
    assert len(experiments["pbd_attention_state_float"]) == 2
    assert experiments["both_all_linear_float_weight"][0].stages == (
        "prefill",
        "pbd_q6",
    )
    assert experiments["both_all_linear_float_weight"][0].mode == "float_weight"
    assert len(experiments["both_attention_state_float"]) == 2
    assert experiments["both_blocks_12_17_float"][0].module_pattern == (
        r"layers\.(?:12|13|14|15|16|17)\..*"
    )
    assert experiments["both_blocks_30_35_wv_float"][0].kinds == ("static",)
    assert experiments["both_sensitive_blocks_float"][0].module_pattern == (
        r"layers\.(?:28|29|32|33|34|35)\..*"
    )
    assert experiments["both_sensitive_linear_float_activation"][0].mode == (
        "float_activation"
    )


def test_coordinate_summary_keeps_only_comparable_geometry():
    audit = {
        "decisions": [
            {
                "comparison": {
                    "structure_agreement": 1.0,
                    "float_valid": 1.0,
                    "candidate_valid": 1.0,
                    "coordinate_token_exact": 0.0,
                    "coordinate_mae": 2.0,
                    "pixel_mae": 1.344,
                    "pixel_max_abs": 2.688,
                    "box_iou": 0.9,
                }
            },
            {
                "comparison": {
                    "structure_agreement": 0.0,
                    "float_valid": 1.0,
                    "candidate_valid": 0.0,
                }
            },
        ]
    }

    summary = MODULE.coordinate_summary(audit)

    assert summary["decision_count"] == 2
    assert summary["structure_agreement"] == pytest.approx(0.5)
    assert summary["comparable_coordinate_count"] == 1
    assert summary["pixel_max_abs"] == pytest.approx(2.688)
    assert summary["box_iou_min"] == pytest.approx(0.9)


def test_coordinate_batch_summary_keeps_resolved_source_and_position_metrics(
    tmp_path,
):
    def decoded(kind, values, fallback=False):
        return {
            "type": kind,
            "coordinate_values": values,
            "fallback": fallback,
        }

    source = decoded("box", [100, 200, 300, 400])
    float_output = decoded("box", [100, 200, 300, 400])
    quantized_output = decoded("box", [102, 200, 298, 400])
    float_quantized = {
        "structure_agreement": 1.0,
        "float_valid": 1.0,
        "candidate_valid": 1.0,
        "coordinate_token_exact": 0.0,
        "pixel_mae": 0.672,
        "pixel_max_abs": 1.344,
        "box_iou": 0.95,
    }
    source_float = {
        "structure_agreement": 1.0,
        "coordinate_token_exact": 1.0,
        "pixel_mae": 0.0,
        "box_iou": 1.0,
    }
    decision = {
        "source": {"kind": "box"},
        "float": float_output,
        "quantized_eager": quantized_output,
        "comparison": float_quantized,
        "resolved": {
            "float": float_output,
            "quantized_eager": quantized_output,
        },
        "resolved_comparison": float_quantized,
        "source_to_float": source_float,
        "source_to_quantized_eager": float_quantized,
        "source_to_resolved_float": source_float,
        "source_to_resolved_quantized_eager": float_quantized,
        "position_diagnostics": [
            {
                "position": 1,
                "comparison": {
                    "comparable": 1.0,
                    "token_exact": 0.0,
                    "pixel_abs_delta": 1.344,
                    "float_token_top4_hit": 0.0,
                    "float_token_rank_in_quantized": 5.0,
                },
            }
        ],
    }
    samples = [{"coordinate_audit": {"decisions": [decision]}}]

    summary = MODULE.aggregate_coordinate_samples(samples)
    group = summary["groups"]["box"]

    assert summary["sample_count"] == 1
    assert summary["decision_count"] == 1
    assert group["resolved_outputs"]["quantized_coordinate_valid"]["mean"] == 1.0
    assert group["comparisons"]["resolved_float_vs_quantized"]["box_iou"][
        "mean"
    ] == pytest.approx(0.95)
    assert group["comparisons"]["upstream_prediction_vs_resolved_float"][
        "box_iou"
    ]["mean"] == 1.0
    position = group["pbd_positions"]["position_1"]
    assert position["float_token_top4_hit"]["mean"] == 0.0
    assert position["float_token_rank_in_quantized"]["max"] == 5.0

    report = [{"name": "baseline_s8", "coordinate_summary": summary}]
    path = tmp_path / "report.csv"
    MODULE.write_coordinate_csv(path, report)
    csv_text = path.read_text(encoding="utf-8")
    assert "upstream_prediction_vs_resolved_float" in csv_text
    assert "pbd_position_float_vs_quantized,position_1" in csv_text
    assert "float_token_top4_hit" in csv_text


def test_operator_suite_is_one_coordinate_baseline():
    assert MODULE.coordinate_operator_experiments() == [("baseline", [])]
    assert MODULE.coordinate_baseline_experiments() == [("baseline", [])]
    suite = MODULE.parser()._option_string_actions["--suite"]
    assert "operator" in suite.choices
    assert "coordinate-baseline" in suite.choices


def test_coordinate_refine_suite_covers_attention_components_and_combination():
    experiments = dict(MODULE.coordinate_refine_experiments())

    assert len(experiments) == 15
    assert len(experiments["prefill_qk_float"]) == 1
    assert len(experiments["both_wv_float"]) == 2
    combined = experiments["prefill_qk_kv_plus_both_wv_float"]
    assert len(combined) == 4
    assert {rule.stages for rule in combined} == {
        ("prefill",),
        MODULE.COORDINATE_STAGES,
    }
    suite = MODULE.parser()._option_string_actions["--suite"]
    assert "coordinate-refine" in suite.choices


def test_coordinate_u8_suite_changes_only_wv_attention_operand():
    experiments = dict(MODULE.coordinate_u8_experiments())

    assert list(experiments) == [
        "baseline_s8",
        "prefill_wv_attention_u8",
        "pbd_wv_attention_u8",
        "prefill_and_pbd_wv_attention_u8",
    ]
    rule = experiments["prefill_and_pbd_wv_attention_u8"][0]
    assert rule.mode == "nonnegative_u8"
    assert rule.stages == MODULE.COORDINATE_STAGES
    assert rule.module_pattern.endswith(r"wv_matmul\.x_fake_quant")
    suite = MODULE.parser()._option_string_actions["--suite"]
    assert "coordinate-u8" in suite.choices


def test_coordinate_ar_suite_isolated_q1_rescues_and_nonnegative_wv_u8():
    experiments = dict(MODULE.coordinate_ar_experiments())

    assert list(experiments) == [
        "baseline",
        "ar_qk_float",
        "ar_wv_float",
        "ar_kv_cache_float",
        "ar_attention_state_float",
        "ar_attention_float",
        "ar_linear_float",
        "ar_lm_head_float",
        "ar_wv_attention_u8",
    ]
    assert all(
        rule.stages == ("ar_q1",)
        for name, rules in experiments.items()
        if name != "baseline"
        for rule in rules
    )
    u8 = experiments["ar_wv_attention_u8"][0]
    assert u8.mode == "nonnegative_u8"
    assert u8.module_pattern.endswith(r"wv_matmul\.x_fake_quant")
    suite = MODULE.parser()._option_string_actions["--suite"]
    assert "coordinate-ar" in suite.choices


def test_coordinate_centered_wv_suite_targets_parent_matmul():
    experiments = dict(MODULE.coordinate_centered_wv_experiments())

    assert list(experiments) == [
        "baseline",
        "prefill_centered_value_wv_s8",
        "pbd_centered_value_wv_s8",
        "prefill_and_pbd_centered_value_wv_s8",
    ]
    rule = experiments["prefill_and_pbd_centered_value_wv_s8"][0]
    assert rule.mode == "centered_value_s8"
    assert rule.kinds == ("fake_matmul",)
    assert rule.stages == MODULE.COORDINATE_STAGES
    suite = MODULE.parser()._option_string_actions["--suite"]
    assert "coordinate-centered-wv" in suite.choices


def test_rescue_suites_keep_the_formal_dynamic_a8_candidate():
    assert MODULE.experiment_dynamic_patterns("focused", "baseline") == list(
        MODULE.DYNAMIC_A8_PATTERNS
    )
    assert MODULE.experiment_dynamic_patterns(
        "dynamic", "baseline_static"
    ) == []


def test_diagnostic_cli_exposes_only_paths_and_suite():
    actions = MODULE.parser()._option_string_actions

    assert set(actions) == {
        "-h",
        "--help",
        "--input_dir",
        "--output_dir",
        "--model_path",
        "--suite",
    }


def test_decision_recovery_uses_the_original_float_and_quantized_tokens():
    baseline_decision = {
        "position": [0],
        "top1_flip": True,
        "reference_topk": [
            {"token_id": 1, "logit": 4.0},
            {"token_id": 2, "logit": 3.0},
        ],
        "candidate_topk": [
            {"token_id": 2, "logit": 3.7},
            {"token_id": 1, "logit": 3.5},
        ],
    }
    rescue_decision = {
        "position": [0],
        "top1_flip": False,
        "reference_topk": baseline_decision["reference_topk"],
        "candidate_topk": [
            {"token_id": 1, "logit": 3.9},
            {"token_id": 2, "logit": 3.2},
        ],
    }

    def sample(decision):
        return {
            "id": "sample",
            "comparisons": [{
                "stage": "ar_q1",
                "module": "logits",
                "comparison": {"decisions": [decision]},
            }],
        }

    report = {
        "experiments": [
            {"samples": [sample(baseline_decision)]},
            {"samples": [sample(rescue_decision)]},
        ]
    }
    MODULE.update_decision_recovery(report)

    baseline = report["experiments"][0]["samples"][0]["decision_recovery"][0]
    rescue = report["experiments"][1]["samples"][0]["decision_recovery"][0]
    assert baseline["recovery"] == pytest.approx(0.0)
    assert rescue["current_gap"] == pytest.approx(0.7)
    assert rescue["recovery"] == pytest.approx(0.75)
    assert rescue["current_token_id"] == 1
