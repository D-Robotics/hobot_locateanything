#!/usr/bin/env python3
"""Run causal Float-rescue experiments for LocateAnything Language QDQ."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compiler"))

from compiler.scripts.validate.compare_pipeline import (  # noqa: E402
    atomic_json,
    detect_float_device,
    resolve_scale_manifest,
    restore_calibration_scales,
    sha256,
    utc_now,
)
from compiler.scripts.common.language import (  # noqa: E402
    DYNAMIC_A8_PATTERNS,
    LanguageEagerRunner,
    create_language_model,
    language_quantization_policy,
    load_payload,
)
from compiler.scripts.common.quantization import (  # noqa: E402
    FloatRescuePolicy,
    FloatRescueRule,
)


DECODE_STAGES = ("pbd_q6", "ar_q1")
COORDINATE_STAGES = ("prefill", "pbd_q6")
COORDINATE_SUITES = {
    "coordinate", "coordinate-baseline", "coordinate-centered-wv", "coordinate-refine",
    "coordinate-u8", "coordinate-ar", "operator",
}


def _rule(
    module_pattern: str,
    *,
    stages: tuple[str, ...] = DECODE_STAGES,
    kinds: tuple[str, ...] = ("linear", "matmul", "static"),
    mode: str = "float",
) -> FloatRescueRule:
    return FloatRescueRule(
        module_pattern,
        stages=stages,
        kinds=kinds,
        mode=mode,
    )


def focused_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    attention = lambda layer: rf"layers\.{layer}\.self_attn\..*"
    return [
        ("baseline", []),
        ("lm_head_float_weight", [
            _rule(r"lm_head", kinds=("linear",), mode="float_weight"),
        ]),
        ("lm_head_float_activation", [
            _rule(r"lm_head", kinds=("linear",), mode="float_activation"),
        ]),
        ("lm_head_float", [
            _rule(r"lm_head", kinds=("linear",)),
        ]),
        ("block2_value_cache", [
            _rule(r"layers\.2\.self_attn\.cache_v_fq", kinds=("static",)),
        ]),
        ("block2_v_projection", [
            _rule(r"layers\.2\.self_attn\.v_proj", kinds=("linear",)),
        ]),
        ("block2_wv_operands", [
            _rule(
                r"layers\.2\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
                kinds=("static",),
            ),
        ]),
        ("block2_attention", [_rule(attention(2))]),
        ("block2_full", [_rule(r"layers\.2\..*")]),
        ("gui_ar_block7_attention", [
            _rule(attention(7), stages=("ar_q1",)),
        ]),
        ("ocr_pbd_blocks8_33_attention", [
            _rule(attention(8), stages=("pbd_q6",)),
            _rule(attention(33), stages=("pbd_q6",)),
        ]),
        ("decode_hotspots_attention", [
            _rule(attention(2)),
            _rule(attention(7), stages=("ar_q1",)),
            _rule(attention(8), stages=("pbd_q6",)),
            _rule(attention(33), stages=("pbd_q6",)),
        ]),
    ]


def suffix_experiments(num_layers: int = 36) -> list[tuple[str, list[FloatRescueRule]]]:
    experiments: list[tuple[str, list[FloatRescueRule]]] = [("baseline", [])]
    for first in range(num_layers - 1, -1, -1):
        layers = "|".join(str(layer) for layer in range(first, num_layers))
        experiments.append((
            f"decode_suffix_{first}_{num_layers - 1}",
            [_rule(rf"layers\.(?:{layers})\..*")],
        ))
    experiments.append((
        "decode_all_layers_and_lm_head_float",
        [
            _rule(r"layers\..*"),
            _rule(r"lm_head", kinds=("linear",)),
        ],
    ))
    return experiments


def _layers_rule(layers: list[int]) -> FloatRescueRule:
    encoded = "|".join(str(layer) for layer in layers)
    return _rule(rf"layers\.(?:{encoded})\..*")


def refine_experiments(num_layers: int = 36) -> list[tuple[str, list[FloatRescueRule]]]:
    last = num_layers - 1
    suffix = list(range(9, num_layers))
    later = list(range(10, num_layers))
    experiments: list[tuple[str, list[FloatRescueRule]]] = [
        ("baseline", []),
        (f"decode_suffix_9_{last}", [_layers_rule(suffix)]),
    ]
    experiments.extend(
        (
            f"decode_suffix_9_{last}_addback_block_{block}",
            [_layers_rule([layer for layer in suffix if layer != block])],
        )
        for block in suffix
    )
    prefix = [_layers_rule(later)]
    components = [
        ("attention", r"layers\.9\.self_attn\..*", ("linear", "matmul", "static")),
        ("mlp", r"layers\.9\.mlp\..*", ("linear",)),
        ("q_projection", r"layers\.9\.self_attn\.q_proj", ("linear",)),
        ("k_projection", r"layers\.9\.self_attn\.k_proj", ("linear",)),
        ("v_projection", r"layers\.9\.self_attn\.v_proj", ("linear",)),
        ("output_projection", r"layers\.9\.self_attn\.o_proj", ("linear",)),
        (
            "qk_operands",
            r"layers\.9\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
            ("static",),
        ),
        (
            "wv_operands",
            r"layers\.9\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
            ("static",),
        ),
        ("key_cache", r"layers\.9\.self_attn\.cache_k_fq", ("static",)),
        ("value_cache", r"layers\.9\.self_attn\.cache_v_fq", ("static",)),
    ]
    experiments.extend(
        (
            f"decode_suffix_10_{last}_plus_block9_{name}",
            [*prefix, _rule(pattern, kinds=kinds)],
        )
        for name, pattern, kinds in components
    )
    experiments.extend([
        (
            f"decode_suffix_10_{last}_plus_block9_qk_path",
            [
                *prefix,
                _rule(
                    r"layers\.9\.self_attn\.(?:q_proj|k_proj|cache_k_fq|"
                    r"qk_matmul\.(?:x|y)_fake_quant)"
                ),
            ],
        ),
        (
            f"decode_suffix_10_{last}_plus_block9_value_path",
            [
                *prefix,
                _rule(
                    r"layers\.9\.self_attn\.(?:v_proj|cache_v_fq|"
                    r"wv_matmul\.(?:x|y)_fake_quant)"
                ),
            ],
        ),
    ])
    return experiments


def _attention_path_rule(
    layers: list[int],
    path: str,
    *,
    kinds: tuple[str, ...],
    mode: str = "float",
) -> FloatRescueRule:
    encoded = "|".join(str(layer) for layer in layers)
    return _rule(
        rf"layers\.(?:{encoded})\.self_attn\.{path}",
        kinds=kinds,
        mode=mode,
    )


def path_experiments(num_layers: int = 36) -> list[tuple[str, list[FloatRescueRule]]]:
    all_layers = list(range(num_layers))
    suffix = list(range(9, num_layers))
    sensitive = [9, 10, 13, 15, 18, 19, 20, 22, 25, 27, 28, 31]

    def q_activation(layers: list[int]) -> FloatRescueRule:
        return _attention_path_rule(
            layers, "q_proj", kinds=("linear",), mode="float_activation"
        )

    def q_weight(layers: list[int]) -> FloatRescueRule:
        return _attention_path_rule(
            layers, "q_proj", kinds=("linear",), mode="float_weight"
        )

    def wv_x(layers: list[int]) -> FloatRescueRule:
        return _attention_path_rule(
            layers, r"wv_matmul\.x_fake_quant", kinds=("static",)
        )

    def wv_y(layers: list[int]) -> FloatRescueRule:
        return _attention_path_rule(
            layers, r"wv_matmul\.y_fake_quant", kinds=("static",)
        )

    def wv(layers: list[int]) -> FloatRescueRule:
        return _attention_path_rule(
            layers, r"wv_matmul\.(?:x|y)_fake_quant", kinds=("static",)
        )

    return [
        ("baseline", []),
        ("all_q_projection_float_weight", [q_weight(all_layers)]),
        ("all_q_projection_float_activation", [q_activation(all_layers)]),
        ("all_wv_attention_operand_float", [wv_x(all_layers)]),
        ("all_wv_value_operand_float", [wv_y(all_layers)]),
        ("all_wv_operands_float", [wv(all_layers)]),
        ("all_q_activation_and_wv_float", [q_activation(all_layers), wv(all_layers)]),
        ("suffix9_q_projection_float_activation", [q_activation(suffix)]),
        ("suffix9_wv_operands_float", [wv(suffix)]),
        ("suffix9_q_activation_and_wv_float", [q_activation(suffix), wv(suffix)]),
        ("sensitive_q_projection_float_activation", [q_activation(sensitive)]),
        ("sensitive_wv_operands_float", [wv(sensitive)]),
        ("sensitive_q_activation_and_wv_float", [q_activation(sensitive), wv(sensitive)]),
        ("block9_q_projection_float_activation", [q_activation([9])]),
        ("block9_wv_attention_operand_float", [wv_x([9])]),
        ("block9_wv_value_operand_float", [wv_y([9])]),
        ("block9_q_activation_and_wv_float", [q_activation([9]), wv([9])]),
    ]


def dynamic_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    return [
        ("baseline_static", []),
        ("dynamic_qk_operands", []),
        ("dynamic_wv_attention_operand", []),
        ("dynamic_wv_value_operand", []),
        ("dynamic_wv_operands", []),
        ("dynamic_qk_and_wv_operands", []),
        ("dynamic_key_cache", []),
        ("dynamic_value_cache", []),
        ("dynamic_kv_cache", []),
        ("dynamic_attention_and_kv_cache", []),
    ]


def coordinate_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    def all_float(stages: tuple[str, ...]) -> list[FloatRescueRule]:
        return [
            _rule(r"layers\..*", stages=stages),
            _rule(r"lm_head", stages=stages, kinds=("linear",)),
        ]

    def attention_state(stages: tuple[str, ...]) -> list[FloatRescueRule]:
        return [
            _rule(
                r"layers\.\d+\.self_attn\.(?:qk|wv)_matmul\.(?:x|y)_fake_quant",
                stages=stages,
                kinds=("static",),
            ),
            _rule(
                r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq",
                stages=stages,
                kinds=("static",),
            ),
        ]

    experiments = [
        ("baseline", []),
        ("prefill_all_float", all_float(("prefill",))),
        ("pbd_all_float", all_float(("pbd_q6",))),
        ("prefill_and_pbd_all_float", all_float(COORDINATE_STAGES)),
        ("prefill_all_linear_float", [
            _rule(r"layers\..*", stages=("prefill",), kinds=("linear",)),
        ]),
        ("prefill_attention_state_float", attention_state(("prefill",))),
        ("pbd_lm_head_float", [
            _rule(r"lm_head", stages=("pbd_q6",), kinds=("linear",)),
        ]),
        ("pbd_all_linear_float_activation", [
            _rule(
                r"layers\..*|lm_head",
                stages=("pbd_q6",),
                kinds=("linear",),
                mode="float_activation",
            ),
        ]),
        ("pbd_all_linear_float_weight", [
            _rule(
                r"layers\..*|lm_head",
                stages=("pbd_q6",),
                kinds=("linear",),
                mode="float_weight",
            ),
        ]),
        ("pbd_all_linear_float", [
            _rule(r"layers\..*|lm_head", stages=("pbd_q6",), kinds=("linear",)),
        ]),
        ("pbd_qk_operands_float", [
            _rule(
                r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
                stages=("pbd_q6",),
                kinds=("static",),
            ),
        ]),
        ("pbd_wv_operands_float", [
            _rule(
                r"layers\.\d+\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
                stages=("pbd_q6",),
                kinds=("static",),
            ),
        ]),
        ("pbd_kv_cache_float", [
            _rule(
                r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq",
                stages=("pbd_q6",),
                kinds=("static",),
            ),
        ]),
        ("pbd_attention_state_float", attention_state(("pbd_q6",))),
        ("both_all_linear_float_activation", [
            _rule(
                r"layers\..*|lm_head",
                stages=COORDINATE_STAGES,
                kinds=("linear",),
                mode="float_activation",
            ),
        ]),
        ("both_all_linear_float_weight", [
            _rule(
                r"layers\..*|lm_head",
                stages=COORDINATE_STAGES,
                kinds=("linear",),
                mode="float_weight",
            ),
        ]),
        ("both_all_linear_float", [
            _rule(
                r"layers\..*|lm_head",
                stages=COORDINATE_STAGES,
                kinds=("linear",),
            ),
        ]),
        ("both_qk_operands_float", [
            _rule(
                r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
        ("both_wv_operands_float", [
            _rule(
                r"layers\.\d+\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
        ("both_kv_cache_float", [
            _rule(
                r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
        ("both_attention_state_float", attention_state(COORDINATE_STAGES)),
    ]
    for first in range(0, 36, 6):
        last = first + 5
        encoded = "|".join(str(layer) for layer in range(first, last + 1))
        experiments.append((
            f"both_blocks_{first}_{last}_float",
            [_rule(rf"layers\.(?:{encoded})\..*", stages=COORDINATE_STAGES)],
        ))
        experiments.append((
            f"both_blocks_{first}_{last}_wv_float",
            [_rule(
                rf"layers\.(?:{encoded})\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            )],
        ))
    sensitive = "28|29|32|33|34|35"
    experiments.extend([
        ("both_sensitive_blocks_float", [
            _rule(rf"layers\.(?:{sensitive})\..*", stages=COORDINATE_STAGES),
        ]),
        ("both_blocks_28_35_float", [
            _rule(
                r"layers\.(?:28|29|30|31|32|33|34|35)\..*",
                stages=COORDINATE_STAGES,
            ),
        ]),
        ("both_blocks_28_29_float", [
            _rule(r"layers\.(?:28|29)\..*", stages=COORDINATE_STAGES),
        ]),
        ("both_blocks_32_35_float", [
            _rule(r"layers\.(?:32|33|34|35)\..*", stages=COORDINATE_STAGES),
        ]),
        ("both_blocks_33_35_float", [
            _rule(r"layers\.(?:33|34|35)\..*", stages=COORDINATE_STAGES),
        ]),
        ("both_sensitive_linear_float_activation", [
            _rule(
                rf"layers\.(?:{sensitive})\..*",
                stages=COORDINATE_STAGES,
                kinds=("linear",),
                mode="float_activation",
            ),
        ]),
        ("both_sensitive_linear_float_weight", [
            _rule(
                rf"layers\.(?:{sensitive})\..*",
                stages=COORDINATE_STAGES,
                kinds=("linear",),
                mode="float_weight",
            ),
        ]),
        ("both_sensitive_wv_float", [
            _rule(
                rf"layers\.(?:{sensitive})\.self_attn\.wv_matmul\.(?:x|y)_fake_quant",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
        ("both_sensitive_qk_float", [
            _rule(
                rf"layers\.(?:{sensitive})\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
        ("both_sensitive_kv_cache_float", [
            _rule(
                rf"layers\.(?:{sensitive})\.self_attn\.cache_(?:k|v)_fq",
                stages=COORDINATE_STAGES,
                kinds=("static",),
            ),
        ]),
    ])
    return experiments


def coordinate_baseline_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    return [("baseline", [])]


def coordinate_operator_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    return coordinate_baseline_experiments()


def coordinate_refine_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    def component(pattern: str, stages: tuple[str, ...]) -> FloatRescueRule:
        return _rule(pattern, stages=stages, kinds=("static",))

    qk_pattern = r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant"
    wv_x_pattern = r"layers\.\d+\.self_attn\.wv_matmul\.x_fake_quant"
    wv_y_pattern = r"layers\.\d+\.self_attn\.wv_matmul\.y_fake_quant"
    kv_pattern = r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq"

    prefill_qk = component(qk_pattern, ("prefill",))
    prefill_wv_x = component(wv_x_pattern, ("prefill",))
    prefill_wv_y = component(wv_y_pattern, ("prefill",))
    prefill_kv = component(kv_pattern, ("prefill",))
    both_wv_x = component(wv_x_pattern, COORDINATE_STAGES)
    both_wv_y = component(wv_y_pattern, COORDINATE_STAGES)
    return [
        ("baseline", []),
        ("prefill_qk_float", [prefill_qk]),
        ("prefill_wv_x_float", [prefill_wv_x]),
        ("prefill_wv_y_float", [prefill_wv_y]),
        ("prefill_kv_float", [prefill_kv]),
        ("prefill_qk_wv_float", [prefill_qk, prefill_wv_x, prefill_wv_y]),
        ("prefill_qk_kv_float", [prefill_qk, prefill_kv]),
        ("prefill_wv_kv_float", [prefill_wv_x, prefill_wv_y, prefill_kv]),
        (
            "prefill_attention_state_float",
            [prefill_qk, prefill_wv_x, prefill_wv_y, prefill_kv],
        ),
        ("both_wv_x_float", [both_wv_x]),
        ("both_wv_y_float", [both_wv_y]),
        ("both_wv_float", [both_wv_x, both_wv_y]),
        ("prefill_qk_plus_both_wv_float", [prefill_qk, both_wv_x, both_wv_y]),
        ("prefill_kv_plus_both_wv_float", [prefill_kv, both_wv_x, both_wv_y]),
        (
            "prefill_qk_kv_plus_both_wv_float",
            [prefill_qk, prefill_kv, both_wv_x, both_wv_y],
        ),
    ]


def coordinate_u8_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    pattern = r"layers\.\d+\.self_attn\.wv_matmul\.x_fake_quant"

    def u8(stages: tuple[str, ...]) -> FloatRescueRule:
        return _rule(
            pattern,
            stages=stages,
            kinds=("static",),
            mode="nonnegative_u8",
        )

    return [
        ("baseline_s8", []),
        ("prefill_wv_attention_u8", [u8(("prefill",))]),
        ("pbd_wv_attention_u8", [u8(("pbd_q6",))]),
        ("prefill_and_pbd_wv_attention_u8", [u8(COORDINATE_STAGES)]),
    ]


def coordinate_ar_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    stage = ("ar_q1",)
    qk = r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant"
    wv = r"layers\.\d+\.self_attn\.wv_matmul\.(?:x|y)_fake_quant"
    wv_attention = r"layers\.\d+\.self_attn\.wv_matmul\.x_fake_quant"
    cache = r"layers\.\d+\.self_attn\.cache_(?:k|v)_fq"
    return [
        ("baseline", []),
        ("ar_qk_float", [_rule(qk, stages=stage, kinds=("static",))]),
        ("ar_wv_float", [_rule(wv, stages=stage, kinds=("static",))]),
        ("ar_kv_cache_float", [_rule(cache, stages=stage, kinds=("static",))]),
        ("ar_attention_state_float", [
            _rule(qk, stages=stage, kinds=("static",)),
            _rule(wv, stages=stage, kinds=("static",)),
            _rule(cache, stages=stage, kinds=("static",)),
        ]),
        ("ar_attention_float", [
            _rule(r"layers\.\d+\.self_attn\..*", stages=stage),
        ]),
        ("ar_linear_float", [
            _rule(r"layers\..*|lm_head", stages=stage, kinds=("linear",)),
        ]),
        ("ar_lm_head_float", [
            _rule(r"lm_head", stages=stage, kinds=("linear",)),
        ]),
        ("ar_wv_attention_u8", [
            _rule(
                wv_attention,
                stages=stage,
                kinds=("static",),
                mode="nonnegative_u8",
            ),
        ]),
    ]


def coordinate_centered_wv_experiments() -> list[tuple[str, list[FloatRescueRule]]]:
    pattern = r"layers\.\d+\.self_attn\.wv_matmul"

    def centered(stages: tuple[str, ...]) -> FloatRescueRule:
        return _rule(
            pattern,
            stages=stages,
            kinds=("fake_matmul",),
            mode="centered_value_s8",
        )

    return [
        ("baseline", []),
        ("prefill_centered_value_wv_s8", [centered(("prefill",))]),
        ("pbd_centered_value_wv_s8", [centered(("pbd_q6",))]),
        ("prefill_and_pbd_centered_value_wv_s8", [centered(COORDINATE_STAGES)]),
    ]


def dynamic_patterns(experiment_name: str) -> list[str]:
    patterns = {
        "qk": r"layers\.\d+\.self_attn\.qk_matmul\.(?:x|y)_fake_quant",
        "wv_x": r"layers\.\d+\.self_attn\.wv_matmul\.x_fake_quant",
        "wv_y": r"layers\.\d+\.self_attn\.wv_matmul\.y_fake_quant",
        "cache_k": r"layers\.\d+\.self_attn\.cache_k_fq",
        "cache_v": r"layers\.\d+\.self_attn\.cache_v_fq",
    }
    selected = {
        "baseline_static": (),
        "dynamic_qk_operands": ("qk",),
        "dynamic_wv_attention_operand": ("wv_x",),
        "dynamic_wv_value_operand": ("wv_y",),
        "dynamic_wv_operands": ("wv_x", "wv_y"),
        "dynamic_qk_and_wv_operands": ("qk", "wv_x", "wv_y"),
        "dynamic_key_cache": ("cache_k",),
        "dynamic_value_cache": ("cache_v",),
        "dynamic_kv_cache": ("cache_k", "cache_v"),
        "dynamic_attention_and_kv_cache": (
            "qk", "wv_x", "wv_y", "cache_k", "cache_v",
        ),
    }[experiment_name]
    return [patterns[name] for name in selected]


def experiment_dynamic_patterns(suite: str, experiment_name: str) -> list[str]:
    if suite == "dynamic":
        return dynamic_patterns(experiment_name)
    return list(DYNAMIC_A8_PATTERNS)


def discover_payloads(input_dir: Path) -> list[Path]:
    paths = sorted(path.resolve() for path in input_dir.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"no .pt Language payloads under {input_dir}")
    return paths


def result_record(result: Any) -> dict[str, Any]:
    return {
        "comparisons": result.comparisons,
        "boundaries": result.boundaries,
        "operators": result.operators,
    }


def coordinate_summary(audit: dict[str, Any]) -> dict[str, Any]:
    comparisons = [
        decision.get("resolved_comparison", decision["comparison"])
        for decision in audit["decisions"]
    ]

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in comparisons if name in row]

    def mean(name: str) -> float | None:
        items = values(name)
        return sum(items) / len(items) if items else None

    pixel_maxima = values("pixel_max_abs")
    box_ious = values("box_iou")
    point_distances = values("point_distance_pixels")
    return {
        "decision_count": len(comparisons),
        "structure_agreement": mean("structure_agreement"),
        "float_valid_rate": mean("float_valid"),
        "candidate_valid_rate": mean("candidate_valid"),
        "comparable_coordinate_count": len(values("coordinate_mae")),
        "coordinate_token_exact_rate": mean("coordinate_token_exact"),
        "pixel_mae": mean("pixel_mae"),
        "pixel_max_abs": max(pixel_maxima) if pixel_maxima else None,
        "box_iou_mean": mean("box_iou"),
        "box_iou_min": min(box_ious) if box_ious else None,
        "point_distance_pixels_mean": mean("point_distance_pixels"),
        "point_distance_pixels_max": max(point_distances) if point_distances else None,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_statistics(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, float | int] = {
        "count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
    }
    if not finite:
        return result
    result.update(
        mean=sum(finite) / len(finite),
        min=min(finite),
        p05=_percentile(finite, 0.05),
        median=_percentile(finite, 0.50),
        p95=_percentile(finite, 0.95),
        max=max(finite),
    )
    return result


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    names = sorted({name for row in rows for name in row})
    summary: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
        ]
        if values:
            summary[name] = _metric_statistics(values)
    return summary


def _coordinate_output_valid(decoded: dict[str, Any]) -> float:
    expected = {"box": 4, "point": 2}.get(str(decoded.get("type")))
    return float(
        expected is not None
        and len(decoded.get("coordinate_values", [])) == expected
    )


def _summarize_coordinate_decisions(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    output_rows: list[dict[str, float]] = []
    positions: dict[int, list[dict[str, Any]]] = {}
    for decision in decisions:
        resolved = decision.get("resolved", {})
        reference = resolved.get("float", decision["float"])
        candidate = resolved.get("quantized_eager", decision["quantized_eager"])
        source_kind = str(decision["source"]["kind"])
        output_rows.append(
            {
                "float_coordinate_valid": _coordinate_output_valid(reference),
                "quantized_coordinate_valid": _coordinate_output_valid(candidate),
                "float_source_type_match": float(reference.get("type") == source_kind),
                "quantized_source_type_match": float(candidate.get("type") == source_kind),
                "float_fallback": float(reference.get("fallback", False)),
                "quantized_fallback": float(candidate.get("fallback", False)),
            }
        )
        for item in decision.get("position_diagnostics", []):
            positions.setdefault(int(item["position"]), []).append(item["comparison"])

    scopes = {
        "resolved_float_vs_quantized": [
            decision.get("resolved_comparison", decision["comparison"])
            for decision in decisions
        ],
        "upstream_prediction_vs_resolved_float": [
            decision.get("source_to_resolved_float", decision["source_to_float"])
            for decision in decisions
        ],
        "upstream_prediction_vs_resolved_quantized": [
            decision.get(
                "source_to_resolved_quantized_eager",
                decision["source_to_quantized_eager"],
            )
            for decision in decisions
        ],
    }
    return {
        "decision_count": len(decisions),
        "resolved_outputs": _numeric_summary(output_rows),
        "comparisons": {
            name: _numeric_summary(rows) for name, rows in scopes.items()
        },
        "pbd_positions": {
            f"position_{position}": _numeric_summary(rows)
            for position, rows in sorted(positions.items())
        },
    }


def aggregate_coordinate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [
        decision
        for sample in samples
        for decision in sample["coordinate_audit"].get("decisions", [])
    ]
    kinds = sorted({str(decision["source"]["kind"]) for decision in decisions})
    groups = {"all": decisions}
    groups.update(
        {
            kind: [
                decision
                for decision in decisions
                if str(decision["source"]["kind"]) == kind
            ]
            for kind in kinds
        }
    )
    return {
        "sample_count": len(samples),
        "samples_with_decisions": sum(
            bool(sample["coordinate_audit"].get("decisions")) for sample in samples
        ),
        "decision_count": len(decisions),
        "reference_contract": {
            "upstream_prediction": (
                "saved prediction_token_ids.hybrid output; not dataset ground truth"
            ),
            "float": "adapted LocateAnything Float replay",
            "quantized": "W8 plus dynamic-A8 Quantized Eager replay",
        },
        "groups": {
            name: _summarize_coordinate_decisions(group)
            for name, group in groups.items()
        },
    }


def write_coordinate_csv(path: Path, experiments: list[dict[str, Any]]) -> None:
    columns = (
        "experiment", "source_kind", "scope", "position", "metric",
        "count", "finite_count", "nonfinite_count", "mean", "min", "p05",
        "median", "p95", "max",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for experiment in experiments:
            summary = experiment.get("coordinate_summary")
            if not summary:
                continue
            for source_kind, group in summary["groups"].items():
                sections = {
                    "resolved_outputs": group["resolved_outputs"],
                    **group["comparisons"],
                }
                for scope, metrics in sections.items():
                    for metric, statistics in metrics.items():
                        writer.writerow(
                            {
                                "experiment": experiment["name"],
                                "source_kind": source_kind,
                                "scope": scope,
                                "position": "",
                                "metric": metric,
                                **statistics,
                            }
                        )
                for position, metrics in group["pbd_positions"].items():
                    for metric, statistics in metrics.items():
                        writer.writerow(
                            {
                                "experiment": experiment["name"],
                                "source_kind": source_kind,
                                "scope": "pbd_position_float_vs_quantized",
                                "position": position,
                                "metric": metric,
                                **statistics,
                            }
                        )
    temporary.replace(path)


def _topk_score(decision: dict[str, Any], side: str, token_id: int) -> float | None:
    for item in decision[f"{side}_topk"]:
        if int(item["token_id"]) == token_id:
            return float(item["logit"])
    return None


def _decision_rows(sample: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["stage"]), str(row["module"])): row["comparison"]
        for row in sample["comparisons"]
        if "decisions" in row["comparison"]
    }


def update_decision_recovery(report: dict[str, Any]) -> None:
    """Measure every rescue against the original Float/quantized winner pair."""

    if not report["experiments"]:
        return
    baseline_samples = {
        sample["id"]: sample for sample in report["experiments"][0]["samples"]
    }
    for experiment in report["experiments"]:
        for sample in experiment["samples"]:
            baseline_rows = _decision_rows(baseline_samples[sample["id"]])
            current_rows = _decision_rows(sample)
            recoveries: list[dict[str, Any]] = []
            for key, baseline_metric in baseline_rows.items():
                current_metric = current_rows[key]
                current_by_position = {
                    tuple(decision["position"]): decision
                    for decision in current_metric["decisions"]
                }
                for baseline_decision in baseline_metric["decisions"]:
                    if not baseline_decision["top1_flip"]:
                        continue
                    position = tuple(baseline_decision["position"])
                    current_decision = current_by_position[position]
                    reference_id = int(
                        baseline_decision["reference_topk"][0]["token_id"]
                    )
                    baseline_id = int(
                        baseline_decision["candidate_topk"][0]["token_id"]
                    )
                    reference_scores = [
                        _topk_score(baseline_decision, "reference", token_id)
                        for token_id in (reference_id, baseline_id)
                    ]
                    baseline_scores = [
                        _topk_score(baseline_decision, "candidate", token_id)
                        for token_id in (reference_id, baseline_id)
                    ]
                    current_scores = [
                        _topk_score(current_decision, "candidate", token_id)
                        for token_id in (reference_id, baseline_id)
                    ]
                    values = [*reference_scores, *baseline_scores, *current_scores]
                    if any(value is None for value in values):
                        recovery = None
                        reference_gap = baseline_gap = current_gap = None
                    else:
                        reference_gap = reference_scores[0] - reference_scores[1]
                        baseline_gap = baseline_scores[0] - baseline_scores[1]
                        current_gap = current_scores[0] - current_scores[1]
                        denominator = reference_gap - baseline_gap
                        recovery = (
                            (current_gap - baseline_gap) / denominator
                            if denominator != 0 else None
                        )
                    recoveries.append({
                        "stage": key[0],
                        "module": key[1],
                        "position": list(position),
                        "reference_token_id": reference_id,
                        "baseline_token_id": baseline_id,
                        "current_token_id": int(
                            current_decision["candidate_topk"][0]["token_id"]
                        ),
                        "reference_gap": reference_gap,
                        "baseline_gap": baseline_gap,
                        "current_gap": current_gap,
                        "recovery": recovery,
                    })
            sample["decision_recovery"] = recoveries


def run(args: argparse.Namespace) -> int:
    import torch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    csv_path = output_dir / "report.csv"
    payload_paths = discover_payloads(args.input_dir.resolve())
    experiments = {
        "focused": focused_experiments,
        "suffix": suffix_experiments,
        "refine": refine_experiments,
        "paths": path_experiments,
        "dynamic": dynamic_experiments,
        "coordinate": coordinate_experiments,
        "coordinate-baseline": coordinate_baseline_experiments,
        "coordinate-centered-wv": coordinate_centered_wv_experiments,
        "coordinate-refine": coordinate_refine_experiments,
        "coordinate-u8": coordinate_u8_experiments,
        "coordinate-ar": coordinate_ar_experiments,
        "operator": coordinate_operator_experiments,
    }[args.suite]()
    device = detect_float_device()
    if not device.startswith("cuda"):
        raise RuntimeError("Language Float rescue requires an NVIDIA CUDA device")

    scale_manifest = resolve_scale_manifest("language")
    api, model, rotation = create_language_model(
        args.model_path.resolve(), output_dir / "work" / "model", device
    )
    calibration = restore_calibration_scales(model, scale_manifest, "language")
    runner = LanguageEagerRunner(
        model,
        rotation,
        device,
        quantized=True,
        capture_boundaries=True,
        capture_operators=args.suite == "operator",
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "suite": args.suite,
        "model_path": str(args.model_path.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "scale_manifest": str(scale_manifest),
        "scale_manifest_sha256": sha256(scale_manifest),
        "calibration": calibration,
        "weight_policy": language_quantization_policy(),
        "cache_contract": {
            "prefill": "Float and quantized runs use independent zero caches",
            "pbd_q6": (
                "each run uses the cache produced by its corresponding "
                "teacher-forced Prefill"
            ),
            "ar_q1": (
                "each run replays the accepted PBD prefix into its corresponding "
                "Prefill cache"
            ),
            "pbd_acceptance": (
                "native Top-4 coordinate pattern check with deterministic AR fallback"
                if args.suite in COORDINATE_SUITES
                else "not evaluated by this arithmetic probe"
            ),
        },
        "summary_csv": str(csv_path) if args.suite in COORDINATE_SUITES else None,
        "inputs": [str(path) for path in payload_paths],
        "experiments": [],
        "started_at": utc_now(),
    }
    atomic_json(report_path, report)

    try:
        for experiment_index, (name, rules) in enumerate(experiments, 1):
            print(f"\n================== [{experiment_index}/{len(experiments)}] {name} ==================")
            policy = FloatRescuePolicy(rules, name=name) if rules else None
            runner.set_rescue_policy(policy)
            dynamic_quantizers = runner.set_dynamic_quantizer_patterns(
                experiment_dynamic_patterns(args.suite, name)
            )
            experiment = {
                "name": name,
                "rules": [rule.as_dict() for rule in rules],
                "dynamic_quantizers": dynamic_quantizers,
                "samples": [],
            }
            report["experiments"].append(experiment)
            atomic_json(report_path, report)
            started = time.monotonic()
            for sample_index, path in enumerate(payload_paths, 1):
                sample_started = time.monotonic()
                payload = load_payload(path)
                if args.suite in COORDINATE_SUITES:
                    audit = runner.audit_coordinates(payload)
                    sample = {
                        "id": path.stem,
                        "path": str(path),
                        "sha256": sha256(path),
                        "elapsed_seconds": time.monotonic() - sample_started,
                        "coordinate_summary": coordinate_summary(audit),
                        "coordinate_audit": audit,
                    }
                    result = None
                else:
                    result = runner.run(payload)
                    sample = {
                        "id": path.stem,
                        "path": str(path),
                        "sha256": sha256(path),
                        "elapsed_seconds": time.monotonic() - sample_started,
                        **result_record(result),
                    }
                experiment["samples"].append(sample)
                print(
                    f"[{sample_index}/{len(payload_paths)}] {path.stem} "
                    f"{sample['elapsed_seconds']:.2f}s"
                )
                del payload, result
                gc.collect()
                torch.cuda.empty_cache()
                if args.suite in COORDINATE_SUITES and (
                    sample_index % 10 == 0 or sample_index == len(payload_paths)
                ):
                    experiment["coordinate_summary"] = aggregate_coordinate_samples(
                        experiment["samples"]
                    )
                    atomic_json(report_path, report)
                    write_coordinate_csv(csv_path, report["experiments"])
            experiment["elapsed_seconds"] = time.monotonic() - started
            experiment["rescue_policy"] = policy.describe() if policy else None
            if args.suite in COORDINATE_SUITES:
                experiment["coordinate_summary"] = aggregate_coordinate_samples(
                    experiment["samples"]
                )
                write_coordinate_csv(csv_path, report["experiments"])
            if args.suite not in COORDINATE_SUITES:
                update_decision_recovery(report)
            atomic_json(report_path, report)
    except BaseException as error:
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        atomic_json(report_path, report)
        if args.suite in COORDINATE_SUITES:
            write_coordinate_csv(csv_path, report["experiments"])
        raise
    finally:
        runner.close()
        del model, api
        gc.collect()
        torch.cuda.empty_cache()

    report["status"] = "completed"
    report["finished_at"] = utc_now()
    atomic_json(report_path, report)
    print(f"\nREPORT: {report_path}")
    if args.suite in COORDINATE_SUITES:
        print(f"CSV: {csv_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Locate the Language quantization operation that changes Top-1 decisions."
    )
    root.add_argument("--input_dir", type=Path, required=True)
    root.add_argument("--output_dir", type=Path, required=True)
    root.add_argument("--model_path", type=Path, required=True)
    root.add_argument(
        "--suite",
        choices=(
            "focused", "suffix", "refine", "paths", "dynamic", "coordinate",
            "coordinate-baseline",
            "coordinate-centered-wv", "coordinate-refine", "coordinate-u8",
            "coordinate-ar", "operator",
        ),
        default="focused",
        help="focused probes hotspots; suffix finds a stable cutoff; refine performs add-back",
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    if args.output_dir.exists():
        raise FileExistsError(
            f"diagnostic output already exists; choose a new --output_dir: {args.output_dir}"
        )
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
