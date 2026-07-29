#!/usr/bin/env python3
"""Fail-closed preflight tying source, prepare, calibrate, and build contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compiler.scripts.common.identity import (  # noqa: E402
    artifact_identities,
    checkpoint_identity,
    file_identity,
    identity_mismatches,
    prepared_bundle_identity_errors,
    read_json as read_identity_json,
    release_checkpoint_errors,
    sha256_json,
    source_tree_identity,
    tokenizer_identity,
)

TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")
RELEASE_SAMPLE_COUNT = 1200
RELEASE_CHECKPOINT_SAMPLES = 512
RELEASE_TASK_COUNTS = {
    "detection": 620,
    "gui": 180,
    "referring": 120,
    "ocr": 120,
    "layout": 100,
    "pointing": 60,
}
RELEASE_DETECTION_SOURCE_COUNTS = {
    "coco_multicategory_detection": 500,
    "dense_retail_detection": 120,
}
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


def release_distribution_errors(
    expected_samples: int | None,
    task_counts: dict[str, int],
    detection_source_counts: dict[str, int],
) -> list[str]:
    if expected_samples != RELEASE_SAMPLE_COUNT:
        return []
    errors = []
    if task_counts != RELEASE_TASK_COUNTS:
        errors.append(
            "release task counts mismatch: "
            f"selected={task_counts} expected={RELEASE_TASK_COUNTS}"
        )
    if detection_source_counts != RELEASE_DETECTION_SOURCE_COUNTS:
        errors.append(
            "release Detection source counts mismatch: "
            f"selected={detection_source_counts} "
            f"expected={RELEASE_DETECTION_SOURCE_COUNTS}"
        )
    return errors


def release_checkpoint_errors(
    expected_samples: int | None,
    checkpoint_samples: Any,
) -> list[str]:
    if (
        expected_samples == RELEASE_SAMPLE_COUNT
        and checkpoint_samples != RELEASE_CHECKPOINT_SAMPLES
    ):
        return [
            "release checkpoint_samples mismatch: "
            f"selected={checkpoint_samples} expected={RELEASE_CHECKPOINT_SAMPLES}"
        ]
    return []


def selected_manifest_sha_errors(
    expected_samples: int | None,
    expected_sha256: str | None,
    actual_sha256: str | None,
) -> list[str]:
    """Validate the frozen release manifest identity, not only its row count."""
    if expected_samples == RELEASE_SAMPLE_COUNT and not expected_sha256:
        return ["release selected manifest SHA256 is required"]
    if not expected_sha256:
        return []
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        return ["expected selected manifest SHA256 is invalid"]
    if actual_sha256 != normalized:
        return [
            "selected manifest SHA256 mismatch: "
            f"actual={actual_sha256} expected={normalized}"
        ]
    return []


def release_identity_errors(
    *,
    expected_samples: int | None,
    selected_jsonl: Path,
    generated_jsonl: Path,
    scale_manifest_path: Path,
    scale: dict[str, Any],
    coverage: dict[str, Any],
    generated_sha: str | None,
    model_path: Path | None,
    compiler_source_root: Path | None = None,
    prepare_source_path: Path | None = None,
    enforce_frozen_checkpoint: bool = True,
) -> list[str]:
    """Verify that release calibration artifacts still match every input."""
    if expected_samples != RELEASE_SAMPLE_COUNT:
        return []

    errors: list[str] = []
    if model_path is None:
        errors.append("release validation requires --model-path")
    else:
        if enforce_frozen_checkpoint:
            errors.extend(release_checkpoint_errors(model_path))
        source_path = prepare_source_path or (
            PROJECT_ROOT / "compiler" / "scripts" / "calibration" / "prepare.py"
        )
        errors.extend(
            prepared_bundle_identity_errors(
                selected_jsonl=selected_jsonl,
                generated_jsonl=generated_jsonl,
                model_path=model_path,
                prepare_source_path=source_path,
                expected_sample_count=RELEASE_SAMPLE_COUNT,
            )
        )
    identity_path = scale_manifest_path.parent / "calibration_run_identity.json"
    if not identity_path.is_file():
        return ["release calibration_run_identity.json is missing"]

    try:
        calibration_identity = read_identity_json(identity_path)
        run_identity = calibration_identity.get("identity")
        if calibration_identity.get("status") != "complete" or not isinstance(
            run_identity, dict
        ):
            return ["release calibration identity is not complete"]

        run_identity_sha = sha256_json(run_identity)
        if scale.get("calibration_run_identity_sha256") != run_identity_sha:
            errors.append("scale manifest calibration identity SHA256 mismatch")
        if coverage.get("calibration_run_identity_sha256") != run_identity_sha:
            errors.append("coverage calibration identity SHA256 mismatch")
        generated_identity = run_identity.get("generated_manifest", {})
        if generated_identity.get("sha256") != generated_sha:
            errors.append("calibration identity generated manifest mismatch")
        if identity_mismatches(
            run_identity.get("selected_manifest"), file_identity(selected_jsonl)
        ):
            errors.append("calibration identity selected manifest mismatch")

        expected_artifacts = calibration_identity.get("artifacts")
        durable = [
            scale_manifest_path,
            scale_manifest_path.parent / "calibration_graph_coverage.json",
            scale_manifest_path.parent / "scale_convergence.json",
            scale_manifest_path.parent
            / f"scale_convergence_{RELEASE_CHECKPOINT_SAMPLES}_vs_{RELEASE_SAMPLE_COUNT}.json",
        ]
        if not isinstance(expected_artifacts, dict):
            errors.append("release calibration identity lacks artifact catalog")
        elif all(path.is_file() for path in durable):
            mismatches = identity_mismatches(
                expected_artifacts, artifact_identities(durable)
            )
            if mismatches:
                errors.append(
                    "release calibration artifact identity mismatch: "
                    + ", ".join(mismatches[:8])
                )
        else:
            errors.append("release calibration convergence artifacts are incomplete")

        prepare_identity = generated_jsonl.parent / "prepare_run_identity.json"
        generation_summary = generated_jsonl.parent / "generation_summary.json"
        if not prepare_identity.is_file() or not generation_summary.is_file():
            errors.append("release Prepare identity/summary is missing")
        else:
            if identity_mismatches(
                run_identity.get("prepare_run_identity"),
                file_identity(prepare_identity),
            ):
                errors.append("calibration identity Prepare input mismatch")
            if identity_mismatches(
                run_identity.get("generation_summary"),
                file_identity(generation_summary),
            ):
                errors.append("calibration identity generation summary mismatch")

        if model_path is not None:
            resolved_model = model_path.resolve()
            if identity_mismatches(
                run_identity.get("checkpoint"), checkpoint_identity(resolved_model)
            ):
                errors.append("calibration identity checkpoint mismatch")
            if identity_mismatches(
                run_identity.get("tokenizer"), tokenizer_identity(resolved_model)
            ):
                errors.append("calibration identity tokenizer mismatch")

        source_root = compiler_source_root or (PROJECT_ROOT / "compiler")
        if identity_mismatches(
            run_identity.get("compiler_source"), source_tree_identity(source_root)
        ):
            errors.append("calibration identity compiler source mismatch")
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(f"cannot validate release calibration identity: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--scale-manifest", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
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
    parser.add_argument("--expected-selected-sha256")
    parser.add_argument("--hidden-rotation-path", type=Path)
    parser.add_argument("--disable-hidden-rotation", action="store_true")
    args = parser.parse_args()
    required_groups = COMPONENT_GROUPS[args.component]
    required_stages = COMPONENT_STAGES[args.component]

    errors: list[str] = []
    selected = read_jsonl(args.selected_jsonl, errors, "selected manifest")
    generated = read_jsonl(args.generated_jsonl, errors, "generated manifest")
    scale = read_json(args.scale_manifest, errors, "calibration scale manifest")
    coverage = read_json(args.coverage_json, errors, "calibration graph coverage")
    selected_sha = sha256(args.selected_jsonl) if args.selected_jsonl.is_file() else None
    errors.extend(selected_manifest_sha_errors(
        args.expected_samples,
        args.expected_selected_sha256,
        selected_sha,
    ))

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
    selected_detection_sources = dict(Counter(
        str((row.get("metadata") or {}).get("calibration_source_role") or "missing")
        for row in selected
        if row.get("task") == "detection"
    ))
    release_gate = args.expected_samples == RELEASE_SAMPLE_COUNT
    errors.extend(release_distribution_errors(
        args.expected_samples,
        selected_tasks,
        selected_detection_sources,
    ))

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
    errors.extend(release_identity_errors(
        expected_samples=args.expected_samples,
        selected_jsonl=args.selected_jsonl,
        generated_jsonl=args.generated_jsonl,
        scale_manifest_path=args.scale_manifest,
        scale=scale,
        coverage=coverage,
        generated_sha=generated_sha,
        model_path=args.model_path,
    ))
    if scale:
        if scale.get("generated_manifest_sha256") != generated_sha:
            errors.append("calibration scale manifest generated SHA256 mismatch")
        scale_count = scale.get("sample_count")
        if scale_count != len(generated):
            errors.append("calibration scale manifest sample_count mismatch")
        if scale.get("task_counts") != selected_tasks:
            errors.append("calibration scale manifest task_counts mismatch")
        checkpoint = scale.get("checkpoint_samples")
        if type(checkpoint) is not int or not 0 < checkpoint < len(generated):
            errors.append("calibration scale manifest checkpoint_samples is invalid")
        errors.extend(release_checkpoint_errors(args.expected_samples, checkpoint))
        for group in required_groups:
            snapshots = scale.get(group)
            if not isinstance(snapshots, dict):
                errors.append(f"calibration scale manifest lacks {group} snapshots")
                continue
            full_snapshot = snapshots.get(str(len(generated)))
            if not isinstance(full_snapshot, dict) or not full_snapshot:
                errors.append(
                    f"calibration scale manifest lacks full {group}/{len(generated)} snapshot"
                )
            else:
                for name, observer in full_snapshot.items():
                    if not isinstance(observer, dict):
                        errors.append(
                            f"calibration {group} activation-statistics entry is invalid: {name}"
                        )
                    elif observer.get("kind") == "ConstFakeQuant":
                        absmax = observer.get("absmax")
                        if (
                            type(absmax) not in (int, float)
                            or not math.isfinite(absmax)
                            or absmax <= 0
                        ):
                            errors.append(
                                f"calibration {group} activation-statistics entry "
                                f"has invalid absmax: {name}"
                            )
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
                            errors.append(
                                f"calibration {group} activation-statistics entry "
                                f"has invalid norm scale: {name}"
                            )
            if type(checkpoint) is int and (
                not isinstance(snapshots.get(str(checkpoint)), dict)
                or not snapshots.get(str(checkpoint))
            ):
                errors.append(
                    f"calibration scale manifest lacks checkpoint "
                    f"{group}/{checkpoint} snapshot"
                )

        rotation_source = scale.get("rotation_source")
        rotation_file_sha = scale.get("rotation_file_sha256")
        if args.disable_hidden_rotation:
            errors.append(
                "release build cannot disable the hidden rotation used during calibration"
            )
        elif args.hidden_rotation_path is not None:
            if not args.hidden_rotation_path.is_file():
                errors.append(f"hidden rotation file missing: {args.hidden_rotation_path}")
            else:
                if rotation_file_sha != sha256(args.hidden_rotation_path):
                    errors.append(
                        "calibration scale manifest hidden rotation SHA256 mismatch"
                    )
                if rotation_source != str(args.hidden_rotation_path.resolve()):
                    errors.append(
                        "calibration scale manifest hidden rotation path mismatch"
                    )
        elif rotation_source not in {DEFAULT_ROTATION_NAME, *LEGACY_ROTATION_NAMES}:
            errors.append(
                "calibration scale manifest was not collected with the built-in "
                "release rotation"
            )
        elif rotation_file_sha not in (None, ""):
            errors.append(
                "built-in calibration rotation must not declare an external "
                "rotation file SHA256"
            )

    if coverage:
        if coverage.get("generated_manifest_sha256") != generated_sha:
            errors.append("calibration graph coverage generated SHA256 mismatch")
        if coverage.get("sample_count") != len(generated):
            errors.append("calibration graph coverage sample_count mismatch")
        if coverage.get("checkpoint_samples") != scale.get("checkpoint_samples"):
            errors.append("calibration coverage/scale checkpoint_samples mismatch")
        if coverage.get("task_counts") != selected_tasks:
            errors.append("calibration graph coverage task_counts mismatch")
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
                    errors.append(f"calibration {source_name} lacks Language profile")
                    continue
                mismatches = {
                    key: profile.get(key)
                    for key, expected in expected_language_profile.items()
                    if profile.get(key) != expected
                }
                if mismatches:
                    errors.append(
                        f"calibration {source_name} Language profile mismatch: {mismatches}"
                    )
        if coverage.get("expected_stages") != list(required_stages):
            errors.append("calibration graph coverage expected_stages mismatch")
        stage_counts = coverage.get("stage_sample_counts")
        if not isinstance(stage_counts, dict):
            errors.append("calibration graph coverage lacks stage_sample_counts")
            stage_counts = {}
        for stage in required_stages:
            if stage_counts.get(stage) != len(generated):
                errors.append(
                    f"calibration graph coverage {stage} count={stage_counts.get(stage)} "
                    f"expected={len(generated)}"
                )
        if coverage.get("all_stages_executed") is not True:
            errors.append("calibration did not execute all required graph paths")
        audit_passed = coverage.get(
            "activation_statistics_audit_passed",
            coverage.get("observer_audit_passed"),
        )
        if audit_passed is not True:
            errors.append("calibration activation-statistics audit did not pass")
        audits = coverage.get(
            "activation_statistics_audit",
            coverage.get("observer_audit"),
        )
        if not isinstance(audits, dict):
            errors.append(
                "calibration graph coverage lacks activation_statistics_audit details"
            )
            audits = {}
        for group in required_groups:
            audit = audits.get(group)
            if not isinstance(audit, dict) or audit.get("passed") is not True:
                errors.append(
                    f"calibration {group} activation-statistics audit did not pass"
                )
                continue
            for issue_key in ("unexecuted", "zero_absmax", "invalid_norm"):
                if audit.get(issue_key) != []:
                    errors.append(
                        f"calibration {group} activation-statistics audit "
                        f"has non-empty {issue_key}"
                    )
            snapshot = scale.get(group, {}).get(str(len(generated)), {})
            if audit.get("observer_count") != len(snapshot):
                errors.append(
                    f"calibration {group} activation-statistics count "
                    "does not match full snapshot"
                )

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
    print(f"[PASS] selected_records={len(selected)} sha256={selected_sha}")
    print(f"[PASS] generated_records={len(generated)} sha256={generated_sha}")
    print(f"[PASS] task_counts={selected_tasks}")
    if release_gate:
        print(f"[PASS] detection_source_counts={selected_detection_sources}")
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
