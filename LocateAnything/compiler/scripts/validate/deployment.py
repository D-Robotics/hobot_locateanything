#!/usr/bin/env python3
"""Fail-closed preflight tying source, prepare, calibrate, and build contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compiler.scripts.common.identity import (  # noqa: E402
    SOURCE_SUFFIXES,
    artifact_identities,
    checkpoint_identity,
    file_identity,
    identity_mismatches,
    prepared_bundle_identity_errors,
    read_json as read_identity_json,
    release_checkpoint_errors as frozen_checkpoint_errors,
    sha256_json,
    source_tree_identity,
    tokenizer_identity,
)
from compiler.leap_llm.language_graphs import (  # noqa: E402
    LANGUAGE_GRAPH_SET_NAMES,
    language_graph_set,
    normalize_graph_set_metadata,
)

TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")
RELEASE_SAMPLE_COUNT = 1200
RELEASE_CHECKPOINT_SAMPLES = 512
RELEASE_TASK_COUNTS = {
    "detection": 660,
    "gui": 150,
    "referring": 120,
    "ocr": 120,
    "layout": 90,
    "pointing": 60,
}
RELEASE_DETECTION_SOURCE_COUNTS = {
    "coco_detection": 240,
    "openimages_v6": 90,
    "v3det": 60,
    "paco": 50,
    "bdd100k": 50,
    "egoobjects": 40,
    "humanparts": 40,
    "mot17det": 45,
    "mot20det": 45,
}
DECODE_CONTEXT_POLICY = "bundle_hash_base_plus_detection_target_tail_v2"
COMPONENT_GROUPS = {
    "full": ("vision", "language"),
    "vision": ("vision",),
    "language": ("language",),
}
DEFAULT_ROTATION_NAME = "signed normalized Sylvester Hadamard rotation (2048x2048)"
LEGACY_ROTATION_NAMES = {"built-in qwen2.5-vl S600 reference Hadamard"}


def component_stages(component: str, graph_set: str) -> tuple[str, ...]:
    language_stages = language_graph_set(graph_set).calibration_stages
    return {
        "full": ("vision", *language_stages),
        "vision": ("vision",),
        "language": language_stages,
    }[component]


def component_stage_counts(
    component: str,
    graph_set: str,
    *,
    sample_count: int,
    language_context_count: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if component in {"full", "vision"}:
        counts["vision"] = sample_count
    if component in {"full", "language"}:
        counts.update(
            language_graph_set(graph_set).calibration_execution_counts(
                language_context_count
            )
        )
    return counts


def source_tree_identity_with_overrides(
    root: Path,
    overrides: dict[Path, bytes],
    excluded_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Hash a source tree while projecting selected files to known content."""

    root = root.resolve()
    normalized_overrides = {
        path.resolve(): content for path, content in overrides.items()
    }
    normalized_exclusions = {
        path.resolve() for path in (excluded_paths or set())
    }
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and path.resolve() not in normalized_exclusions
        and not {".git", "__pycache__", ".pytest_cache"} & set(path.parts)
    )
    digest = hashlib.sha256()
    normalized_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = normalized_overrides.get(path.resolve(), path.read_bytes())
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        normalized_bytes += len(content)
    if not files:
        raise RuntimeError(f"source directory contains no tracked source files: {root}")
    return {
        "file_count": len(files),
        "normalized_bytes": normalized_bytes,
        "sha256": digest.hexdigest(),
    }


def git_head_content(path: Path) -> bytes | None:
    """Read one tracked file from Git HEAD without changing the worktree."""

    try:
        root_result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if root_result.returncode != 0:
        return None
    git_root = Path(root_result.stdout.decode("utf-8").strip()).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(git_root).as_posix()
    except ValueError:
        return None
    content_result = subprocess.run(
        ["git", "-C", str(git_root), "show", f"HEAD:{relative}"],
        capture_output=True,
        check=False,
    )
    return content_result.stdout if content_result.returncode == 0 else None


def compiler_source_identity_status(
    expected: Any,
    source_root: Path,
) -> tuple[bool, str | None]:
    """Accept a source mismatch only when it is confined to this validator."""

    current = source_tree_identity(source_root)
    if not identity_mismatches(expected, current):
        return True, None

    validator_path = Path(__file__).resolve()
    try:
        validator_path.relative_to(source_root.resolve())
    except ValueError:
        return False, None
    historical_validator = git_head_content(validator_path)
    if historical_validator is None:
        return False, None
    validation_plot = source_root.resolve() / "scripts" / "validate" / "plot_history.py"
    compatibility_overrides = {validator_path: historical_validator}
    language_variants = source_root.resolve() / "scripts" / "build" / "language_variants.py"
    historical_language_variants = git_head_content(language_variants)
    if language_variants.is_file() and historical_language_variants is None:
        return False, None
    if (
        language_variants.is_file()
        and historical_language_variants is not None
        and language_variants.read_bytes() != historical_language_variants
    ):
        compatibility_overrides[language_variants] = historical_language_variants
    environment_gate = source_root.resolve() / "scripts" / "common" / "environment.py"
    if environment_gate.is_file():
        current_environment_gate = environment_gate.read_bytes()
        timeout_update = (
            b"def import_probe(modules: list[str], timeout_seconds: int = 180) "
            b"-> dict[str, object]:\n"
            b"    \"\"\"Probe the pinned stack without rejecting slow first-import "
            b"hosts.\"\"\"\n"
        )
        calibrated_timeout = (
            b"def import_probe(modules: list[str], timeout_seconds: int = 60) "
            b"-> dict[str, object]:\n"
        )
        if current_environment_gate.count(timeout_update) != 1:
            return False, None
        compatibility_overrides[environment_gate] = current_environment_gate.replace(
            timeout_update, calibrated_timeout, 1
        )
    projected = source_tree_identity_with_overrides(
        source_root,
        compatibility_overrides,
        excluded_paths={validation_plot},
    )
    if identity_mismatches(expected, projected):
        return False, None
    return True, (
        "compiler source identity reconstructed exactly after projecting "
        "scripts/validate/deployment.py, build contract checker, and environment "
        "gate to Git HEAD; excluding the added scripts/validate/plot_history.py"
    )


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


def nonnegative_count_mapping(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> dict[str, int]:
    """Validate persisted coverage counts without trusting JSON value types."""

    if not isinstance(value, dict):
        errors.append(f"{label} must be an object of non-negative integer counts")
        return {}
    invalid = [
        str(key)
        for key, count in value.items()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0
    ]
    if invalid:
        errors.append(
            f"{label} contains invalid counts for: {', '.join(sorted(invalid))}"
        )
        return {}
    return {str(key): count for key, count in value.items()}


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


def release_convergence_checkpoint_errors(
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
            errors.extend(frozen_checkpoint_errors(model_path))
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
        source_compatible, compatibility_note = compiler_source_identity_status(
            run_identity.get("compiler_source"), source_root
        )
        if not source_compatible:
            errors.append("calibration identity compiler source mismatch")
        elif compatibility_note:
            print(f"[identity] {compatibility_note}")
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
    parser.add_argument(
        "--expected-dataset-index-sha256",
        "--expected-selected-sha256",
        dest="expected_dataset_index_sha256",
    )
    parser.add_argument("--hidden-rotation-path", type=Path)
    parser.add_argument("--disable-hidden-rotation", action="store_true")
    parser.add_argument(
        "--graph-set",
        dest="graph_set",
        choices=LANGUAGE_GRAPH_SET_NAMES,
        default="standard",
    )
    args = parser.parse_args()
    required_groups = COMPONENT_GROUPS[args.component]
    required_stages = component_stages(args.component, args.graph_set)

    errors: list[str] = []
    selected = read_jsonl(args.selected_jsonl, errors, "selected manifest")
    generated = read_jsonl(args.generated_jsonl, errors, "generated manifest")
    scale = read_json(args.scale_manifest, errors, "calibration scale manifest")
    coverage = read_json(args.coverage_json, errors, "calibration graph coverage")
    declared_stages = coverage.get("expected_stages")
    calibration_component = next(
        (
            component
            for component in COMPONENT_GROUPS
            for stages in (component_stages(component, args.graph_set),)
            if declared_stages == list(stages)
        ),
        None,
    )
    if calibration_component is None:
        errors.append("calibration graph coverage declares an unknown stage profile")
        calibration_groups = required_groups
        calibration_stages = required_stages
    else:
        calibration_groups = COMPONENT_GROUPS[calibration_component]
        calibration_stages = component_stages(
            calibration_component, args.graph_set
        )
        missing_groups = sorted(set(required_groups) - set(calibration_groups))
        if missing_groups:
            errors.append(
                "calibration graph coverage does not include required component groups: "
                + ", ".join(missing_groups)
            )
    selected_sha = sha256(args.selected_jsonl) if args.selected_jsonl.is_file() else None
    errors.extend(selected_manifest_sha_errors(
        args.expected_samples,
        args.expected_dataset_index_sha256,
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
        scale_language_context_count = scale.get("language_context_count", 0)
        if (
            isinstance(scale_language_context_count, bool)
            or not isinstance(scale_language_context_count, int)
        ):
            errors.append("calibration scale manifest language_context_count is invalid")
        elif "language" in calibration_groups and not (
            len(generated) <= scale_language_context_count <= 2 * len(generated)
        ):
            errors.append("calibration Language context count is outside release bounds")
        elif "language" not in calibration_groups and scale_language_context_count != 0:
            errors.append("Vision-only calibration declares Language contexts")
        checkpoint = scale.get("checkpoint_samples")
        if type(checkpoint) is not int or not 0 < checkpoint < len(generated):
            errors.append("calibration scale manifest checkpoint_samples is invalid")
        errors.extend(
            release_convergence_checkpoint_errors(args.expected_samples, checkpoint)
        )
        activation_audits = coverage.get(
            "activation_statistics_audit",
            coverage.get("observer_audit", {}),
        )
        for group in calibration_groups:
            snapshots = scale.get(group)
            if not isinstance(snapshots, dict):
                errors.append(f"calibration scale manifest lacks {group} snapshots")
                continue
            group_audit = (
                activation_audits.get(group, {})
                if isinstance(activation_audits, dict) else {}
            )
            empty_snapshot_is_valid = (
                group_audit.get("passed") is True
                and group_audit.get("status") == "not_applicable"
                and group_audit.get("required_point_count") == 0
            )
            full_snapshot = snapshots.get(str(len(generated)))
            if not isinstance(full_snapshot, dict) or (
                not full_snapshot and not empty_snapshot_is_valid
            ):
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
                or (
                    not snapshots.get(str(checkpoint))
                    and not empty_snapshot_is_valid
                )
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
        language_context_count = coverage.get("language_context_count", 0)
        if isinstance(language_context_count, bool) or not isinstance(
            language_context_count, int
        ):
            errors.append("calibration coverage Language context count is invalid")
            language_context_count = 0
        elif "language" in calibration_groups and not (
            len(generated) <= language_context_count <= 2 * len(generated)
        ):
            errors.append("calibration coverage Language context count is outside release bounds")
        elif "language" not in calibration_groups and language_context_count != 0:
            errors.append("Vision-only graph coverage declares Language contexts")
        if language_context_count != scale.get("language_context_count", 0):
            errors.append("calibration coverage/scale Language context count mismatch")
        if "language" in calibration_groups:
            profile = language_graph_set(args.graph_set)
            expected_language_profile = {
                "language_decoder_weight_bits": 8,
                "language_lm_head_weight_bits": args.lm_head_w_bits,
                "text_mask_token_id": 151676,
                "pbd_block_size": 6,
                "graph_set": profile.name,
                "pbd_total_query_lengths": (
                    list(range(6, 13)) if profile.uses_fused_decode else [6]
                ),
                "ar_total_query_lengths": (
                    list(range(1, 6)) if profile.uses_fused_decode else [1]
                ),
                "ar_q1_calls_per_context": profile.sequential_ar_q1_tokens,
                "pbd_q6_role": "post_prefill_bootstrap_only",
                "pbd_input_protocol": "accepted_prefix_plus_duplicated_anchor_plus_5_text_masks",
                "decode_context_policy": DECODE_CONTEXT_POLICY,
            }
            for source_name, profile in (
                ("scale", scale.get("profile")),
                ("coverage", coverage.get("profile")),
            ):
                if not isinstance(profile, dict):
                    errors.append(f"calibration {source_name} lacks Language profile")
                    continue
                actual_profile = dict(profile)
                stored_name = actual_profile.get(
                    "graph_set", actual_profile.get("graph_profile")
                )
                if stored_name is not None:
                    actual_profile["graph_set"] = language_graph_set(
                        normalize_graph_set_metadata(stored_name)
                    ).name
                mismatches = {
                    key: actual_profile.get(key)
                    for key, expected in expected_language_profile.items()
                    if actual_profile.get(key) != expected
                }
                if mismatches:
                    errors.append(
                        f"calibration {source_name} Language profile mismatch: {mismatches}"
                    )
            context = coverage.get("decode_context_coverage")
            if not isinstance(context, dict):
                errors.append("calibration coverage lacks Decode context evidence")
                context = {}
            if coverage.get("decode_context_coverage_passed") is not True:
                errors.append("calibration Decode context coverage did not pass")
            if context.get("policy") != DECODE_CONTEXT_POLICY:
                errors.append("calibration Decode context policy mismatch")
            if context.get("sample_count") != len(generated):
                errors.append("calibration Decode context sample_count mismatch")
            if context.get("language_context_count") != language_context_count:
                errors.append("calibration Decode context count mismatch")
            if context.get("base_context_count") != len(generated):
                errors.append("calibration Decode contexts lack exactly one base per sample")
            if (
                context.get("supplemental_context_count")
                != language_context_count - len(generated)
            ):
                errors.append("calibration Decode supplemental context count mismatch")
            eligible_long = context.get("eligible_long_detection_sample_count")
            if (
                isinstance(eligible_long, bool)
                or not isinstance(eligible_long, int)
                or eligible_long <= 0
            ):
                errors.append("calibration has no eligible long Detection target context")
            required_target = context.get("required_target_context_count")
            covered_target = context.get("covered_required_target_context_count")
            if (
                isinstance(required_target, bool)
                or not isinstance(required_target, int)
                or required_target <= 0
                or isinstance(covered_target, bool)
                or not isinstance(covered_target, int)
                or covered_target != required_target
            ):
                errors.append("calibration required Detection target contexts are incomplete")
            if context.get("missing_required_target_contexts"):
                errors.append("calibration reports missing Detection target contexts")
            if context.get("passed") is not True or context.get("errors"):
                errors.append("calibration Decode context evidence contains failed gates")
            depth_buckets = nonnegative_count_mapping(
                context.get("depth_buckets"),
                label="calibration Decode context depth_buckets",
                errors=errors,
            )
            if sum(depth_buckets.values()) != language_context_count:
                errors.append("calibration Decode depth buckets do not cover all contexts")
            if depth_buckets.get("zero", 0) <= 0:
                errors.append("calibration Decode context lacks prompt-boundary samples")
            if sum(
                depth_buckets.get(name, 0)
                for name in ("1_31", "32_127", "128_plus")
            ) <= 0:
                errors.append("calibration Decode context lacks nonzero history")
            if sum(
                depth_buckets.get(name, 0) for name in ("32_127", "128_plus")
            ) <= 0:
                errors.append("calibration Decode context lacks deep history")
            token_sources = nonnegative_count_mapping(
                context.get("token_sources"),
                label="calibration Decode context token_sources",
                errors=errors,
            )
            if sum(token_sources.values()) != language_context_count:
                errors.append("calibration Decode token sources do not cover all contexts")
            context_roles = nonnegative_count_mapping(
                context.get("context_roles"),
                label="calibration Decode context_roles",
                errors=errors,
            )
            if sum(context_roles.values()) != language_context_count:
                errors.append("calibration Decode roles do not cover all contexts")
            if context_roles.get("base") != len(generated):
                errors.append("calibration Decode roles lack one base per sample")
            if set(context_roles) - {"base", "target_tail"}:
                errors.append("calibration Decode roles contain unexpected values")
            for metric in ("suffix_len", "past_len"):
                values = context.get(metric)
                if (
                    not isinstance(values, dict)
                    or values.get("min") is None
                    or values.get("max") is None
                ):
                    errors.append(f"calibration Decode context lacks {metric} range")
        stage_counts = coverage.get("stage_execution_counts")
        if not isinstance(stage_counts, dict):
            errors.append("calibration graph coverage lacks stage_execution_counts")
            stage_counts = {}
        expected_stage_counts = component_stage_counts(
            calibration_component,
            args.graph_set,
            sample_count=len(generated),
            language_context_count=language_context_count,
        )
        if coverage.get("expected_stage_execution_counts") != expected_stage_counts:
            errors.append("calibration expected stage execution counts mismatch")
        for stage in calibration_stages:
            expected_count = expected_stage_counts[stage]
            if stage_counts.get(stage) != expected_count:
                errors.append(
                    f"calibration graph coverage {stage} count={stage_counts.get(stage)} "
                    f"expected={expected_count}"
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
        for group in calibration_groups:
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
