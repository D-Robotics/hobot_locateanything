#!/usr/bin/env python3
"""Fail-closed preflight tying selected data, D3, D4, and compile profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")
PBD_STAGES = tuple(f"pbd_q{q_len}" for q_len in range(6, 13))
AR_STAGES = tuple(f"ar_q{q_len}" for q_len in range(1, 6))
LANGUAGE_STAGES = ("prefill", *PBD_STAGES, *AR_STAGES)
GRAPH_STAGES = ("vision", *LANGUAGE_STAGES)
COMPONENT_GROUPS = {
    "full": ("vision", "language"),
    "vision": ("vision",),
    "language": ("language",),
}
COMPONENT_STAGES = {
    "full": GRAPH_STAGES,
    "vision": ("vision",),
    "language": LANGUAGE_STAGES,
}
DEFAULT_ROTATION_NAME = "signed normalized Sylvester Hadamard rotation (2048x2048)"
LEGACY_ROTATION_NAMES = {"built-in qwen2.5-vl S600 reference Hadamard"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{label} missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def read_jsonl(path: Path, errors: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{label} missing: {path}")
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid {label} JSONL at line {line_no}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{label} line {line_no} is not an object")
            continue
        records.append(value)
    return records


def record_ids(records: list[dict[str, Any]], label: str, errors: list[str]) -> list[str]:
    values = [str(row.get("bundle_id") or "") for row in records]
    if any(not value for value in values):
        errors.append(f"{label} record missing bundle_id")
    if len(set(values)) != len(values):
        errors.append(f"{label} has duplicate bundle_id")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--scale-manifest", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument(
        "--component", choices=tuple(COMPONENT_GROUPS), default="full"
    )
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--cache-len", type=int, required=True)
    parser.add_argument("--decode-seq-len", type=int, required=True)
    parser.add_argument("--lm-head-w-bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--hidden-rotation-path", type=Path)
    parser.add_argument("--disable-hidden-rotation", action="store_true")
    args = parser.parse_args()
    required_groups = COMPONENT_GROUPS[args.component]
    required_stages = COMPONENT_STAGES[args.component]

    errors: list[str] = []
    selected = read_jsonl(args.selected_jsonl, errors, "selected manifest")
    generated = read_jsonl(args.generated_jsonl, errors, "generated manifest")
    scale = read_json(args.scale_manifest, errors, "D4 scale manifest")
    coverage = read_json(args.coverage_json, errors, "D4 graph coverage")

    selected_ids = record_ids(selected, "selected manifest", errors)
    generated_ids = record_ids(generated, "generated manifest", errors)
    if selected_ids != generated_ids:
        errors.append("selected/generated bundle_id order differs")
    if len(selected) < 256 or len(generated) != len(selected):
        errors.append(f"invalid record counts: selected={len(selected)} generated={len(generated)}")
    if args.expected_samples is not None and len(selected) != args.expected_samples:
        errors.append(
            f"release sample count mismatch: selected={len(selected)} "
            f"expected={args.expected_samples}"
        )

    selected_by_id = {row.get("bundle_id"): row for row in selected}
    for row in generated:
        source = selected_by_id.get(row.get("bundle_id"))
        if source is None:
            continue
        for field in ("task", "prompt", "target_response", "image_sha256"):
            if row.get(field) != source.get(field):
                errors.append(f"selected/generated {field} mismatch: {row.get('bundle_id')}")
                break

    selected_tasks = dict(Counter(str(row.get("task")) for row in selected))
    if set(selected_tasks) != set(TASKS):
        errors.append(f"selected manifest does not cover all six tasks: {selected_tasks}")

    expected_profile = {
        "image_width": args.image_width,
        "image_height": args.image_height,
        "resize_mode": "letterbox",
        "patch_count": 2304,
        "visual_token_count": 576,
        "prefill_limit": args.chunk_size,
        "pbd_block_size": args.decode_seq_len,
    }
    generated_mode_counts: Counter[str] = Counter()
    for row in generated:
        profile = row.get("fixed_profile")
        if not isinstance(profile, dict):
            errors.append(f"generated record lacks fixed_profile: {row.get('bundle_id')}")
        else:
            mismatches = {
                key: profile.get(key)
                for key, expected in expected_profile.items()
                if profile.get(key) != expected
            }
            if mismatches:
                errors.append(
                    f"generated fixed_profile mismatch: {row.get('bundle_id')} {mismatches}"
                )
        prediction = row.get("prediction")
        if not isinstance(prediction, dict) or "hybrid" not in prediction:
            errors.append(f"generated record lacks hybrid/PBD trajectory: {row.get('bundle_id')}")
        else:
            generated_mode_counts.update(str(mode) for mode in prediction)

    generated_sha = sha256(args.generated_jsonl) if args.generated_jsonl.is_file() else None
    if scale:
        if scale.get("generated_manifest_sha256") != generated_sha:
            errors.append("D4 scale manifest generated SHA256 mismatch")
        scale_count = scale.get("sample_count")
        if scale_count != len(generated):
            errors.append("D4 scale manifest sample_count mismatch")
        if scale.get("task_counts") != selected_tasks:
            errors.append("D4 scale manifest task_counts mismatch")
        checkpoint = scale.get("checkpoint_samples")
        if type(checkpoint) is not int or not 0 < checkpoint < len(generated):
            errors.append("D4 scale manifest checkpoint_samples is invalid")
        for group in required_groups:
            snapshots = scale.get(group)
            if not isinstance(snapshots, dict):
                errors.append(f"D4 scale manifest lacks {group} snapshots")
                continue
            full_snapshot = snapshots.get(str(len(generated)))
            if not isinstance(full_snapshot, dict) or not full_snapshot:
                errors.append(f"D4 scale manifest lacks full {group}/{len(generated)} snapshot")
            else:
                for name, observer in full_snapshot.items():
                    if not isinstance(observer, dict):
                        errors.append(f"D4 {group} observer is invalid: {name}")
                    elif observer.get("kind") == "ConstFakeQuant":
                        absmax = observer.get("absmax")
                        if (
                            type(absmax) not in (int, float)
                            or not math.isfinite(absmax)
                            or absmax <= 0
                        ):
                            errors.append(f"D4 {group} observer has invalid absmax: {name}")
                    else:
                        norm_scale = observer.get("scale")
                        summax = observer.get("summax_hidden")
                        if (
                            type(norm_scale) not in (int, float)
                            or not math.isfinite(norm_scale)
                            or norm_scale <= 0
                            or type(summax) not in (int, float)
                            or not math.isfinite(summax)
                            or summax <= 0
                        ):
                            errors.append(f"D4 {group} observer has invalid norm scale: {name}")
            if type(checkpoint) is int and (
                not isinstance(snapshots.get(str(checkpoint)), dict)
                or not snapshots.get(str(checkpoint))
            ):
                errors.append(f"D4 scale manifest lacks checkpoint {group}/{checkpoint} snapshot")

        rotation_source = scale.get("rotation_source")
        rotation_file_sha = scale.get("rotation_file_sha256")
        if args.disable_hidden_rotation:
            errors.append("release compile cannot disable the hidden rotation used by D4")
        elif args.hidden_rotation_path is not None:
            if not args.hidden_rotation_path.is_file():
                errors.append(f"hidden rotation file missing: {args.hidden_rotation_path}")
            else:
                if rotation_file_sha != sha256(args.hidden_rotation_path):
                    errors.append("D4 scale manifest hidden rotation SHA256 mismatch")
                if rotation_source != str(args.hidden_rotation_path.resolve()):
                    errors.append("D4 scale manifest hidden rotation path mismatch")
        elif rotation_source not in {DEFAULT_ROTATION_NAME, *LEGACY_ROTATION_NAMES}:
            errors.append("D4 scale manifest was not collected with the built-in release rotation")
        elif rotation_file_sha not in (None, ""):
            errors.append("built-in D4 rotation must not declare an external rotation file SHA256")

    if coverage:
        if coverage.get("generated_manifest_sha256") != generated_sha:
            errors.append("D4 coverage generated SHA256 mismatch")
        if coverage.get("sample_count") != len(generated):
            errors.append("D4 coverage sample_count mismatch")
        if coverage.get("checkpoint_samples") != scale.get("checkpoint_samples"):
            errors.append("D4 coverage/scale checkpoint_samples mismatch")
        if coverage.get("task_counts") != selected_tasks:
            errors.append("D4 coverage task_counts mismatch")
        if "language" in required_groups:
            expected_language_profile = {
                "language_decoder_weight_bits": 8,
                "language_lm_head_weight_bits": args.lm_head_w_bits,
                "text_mask_token_id": 151676,
                "pbd_block_size": 6,
                "pbd_total_query_lengths": list(range(6, 13)),
                "ar_total_query_lengths": list(range(1, 6)),
                "pbd_q6_role": "post_prefill_bootstrap_only",
                "pbd_input_protocol": "accepted_prefix_plus_duplicated_anchor_plus_5_text_masks",
            }
            for source_name, profile in (
                ("scale", scale.get("profile")),
                ("coverage", coverage.get("profile")),
            ):
                if not isinstance(profile, dict):
                    errors.append(f"D4 {source_name} lacks Language profile")
                    continue
                mismatches = {
                    key: profile.get(key)
                    for key, expected in expected_language_profile.items()
                    if profile.get(key) != expected
                }
                if mismatches:
                    errors.append(
                        f"D4 {source_name} Language profile mismatch: {mismatches}"
                    )
        if coverage.get("expected_stages") != list(required_stages):
            errors.append("D4 coverage expected_stages mismatch")
        stage_counts = coverage.get("stage_sample_counts")
        if not isinstance(stage_counts, dict):
            errors.append("D4 coverage lacks stage_sample_counts")
            stage_counts = {}
        for stage in required_stages:
            if stage_counts.get(stage) != len(generated):
                errors.append(
                    f"D4 coverage {stage} count={stage_counts.get(stage)} "
                    f"expected={len(generated)}"
                )
        if coverage.get("all_stages_executed") is not True:
            errors.append("D4 did not execute all graph stages")
        audit_passed = coverage.get(
            "activation_statistics_audit_passed",
            coverage.get("observer_audit_passed"),
        )
        if audit_passed is not True:
            errors.append("D4 activation statistics audit did not pass")
        audits = coverage.get(
            "activation_statistics_audit",
            coverage.get("observer_audit"),
        )
        if not isinstance(audits, dict):
            errors.append("D4 coverage lacks activation_statistics_audit details")
            audits = {}
        for group in required_groups:
            audit = audits.get(group)
            if not isinstance(audit, dict) or audit.get("passed") is not True:
                errors.append(f"D4 {group} activation statistics audit did not pass")
                continue
            for issue_key in ("unexecuted", "zero_absmax", "invalid_norm"):
                if audit.get(issue_key) != []:
                    errors.append(
                        f"D4 {group} activation statistics audit has non-empty {issue_key}"
                    )
            snapshot = scale.get(group, {}).get(str(len(generated)), {})
            if audit.get("observer_count") != len(snapshot):
                errors.append(f"D4 {group} observer count does not match full snapshot")

    if (args.image_width, args.image_height) != (672, 672):
        errors.append("LA release profile requires 672x672 letterbox")
    if (args.chunk_size, args.cache_len) != (1024, 4096):
        errors.append("LA release profile requires chunk=1024 and cache=4096")
    if args.decode_seq_len != 6:
        errors.append("LA release profile requires PBD decode_seq_len=6")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 2
    print(f"[PASS] selected_records={len(selected)} sha256={sha256(args.selected_jsonl)}")
    print(f"[PASS] generated_records={len(generated)} sha256={generated_sha}")
    print(f"[PASS] task_counts={selected_tasks}")
    print(f"[PASS] generation_mode_counts={dict(generated_mode_counts)}")
    print(f"[PASS] scale_manifest={args.scale_manifest}")
    print(f"[PASS] coverage={args.coverage_json}")
    print(
        f"[PASS] component={args.component} "
        "profile=672x672 chunk=1024 cache=4096 pbd=6"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
