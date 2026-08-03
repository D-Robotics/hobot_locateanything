#!/usr/bin/env python3
"""Collect LocateAnything activation statistics from prepared calibration tensors."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = REPO_ROOT / "compiler"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COMPILER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler.scripts.common.identity import (  # noqa: E402
    artifact_identities,
    atomic_json as atomic_identity_json,
    checkpoint_identity,
    file_identity,
    identity_mismatches,
    prepared_bundle_identity_errors,
    read_json,
    release_checkpoint_errors,
    rotation_identity,
    sha256_json,
    source_tree_identity,
    tokenizer_identity,
)

from leap_llm.apis.calibration.locateanything_replay import (  # noqa: E402
    ActivationTracker,
    DECODE_CONTEXT_POLICY,
    append_cache_updates,
    build_decode_inputs,
    build_prefill_inputs,
    build_right_aligned_caches,
    compare_snapshots,
    load_tensor_payload,
    read_generated_manifest,
    replay_sequential_ar_q1,
    select_decode_replay_contexts,
    select_pbd_tokens,
    sha256_file,
    summarize_decode_context_coverage,
)
from leap_llm.language_graphs import language_graph_set  # noqa: E402
from leap_llm.models.locateanything.hidden_rotation import load_hidden_rotation  # noqa: E402
from report import generate_activation_report  # noqa: E402


STANDARD_CONVERGENCE_CHECKPOINTS = (64, 128, 256, 512)
RELEASE_SAMPLE_COUNT = 1200
RELEASE_CONVERGENCE_CHECKPOINT = 512
RELEASE_SELECTED_MANIFEST_SHA256 = (
    "521c9203579b165b619934684ca0dd44f9a33dc9c68e0bb6abb17f481d17850b"
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def activation_statistics_audit(
    snapshot: dict[str, Any],
    *,
    required_point_count: int | None = None,
) -> dict[str, Any]:
    """Validate collected activation scales against the model's tracked points.

    A component with no static activation-scale modules is valid when the
    tracker reports zero required points.  An omitted count retains the old
    strict behavior and treats an empty snapshot as a failure.
    """
    observed_point_count = len(snapshot)
    if required_point_count is None:
        expected_point_count = observed_point_count
        empty_snapshot_is_valid = False
    else:
        if required_point_count < 0:
            raise ValueError("required_point_count must be non-negative")
        expected_point_count = required_point_count
        empty_snapshot_is_valid = required_point_count == 0
    point_count_mismatch = observed_point_count != expected_point_count
    unexecuted = [name for name, value in snapshot.items() if not value.get("executions")]
    zero_absmax = [
        name for name, value in snapshot.items()
        if value.get("kind") == "ConstFakeQuant"
        and (
            not isinstance(value.get("absmax"), (int, float))
            or not math.isfinite(float(value["absmax"]))
            or float(value["absmax"]) <= 0
        )
    ]
    invalid_norm = [
        name for name, value in snapshot.items()
        if value.get("kind") != "ConstFakeQuant"
        and (
            not isinstance(value.get("summax_hidden"), (int, float))
            or not math.isfinite(float(value["summax_hidden"]))
            or not isinstance(value.get("scale"), (int, float))
            or not math.isfinite(float(value["scale"]))
            or float(value["scale"]) <= 0
        )
    ]
    nonfinite = [
        name for name, value in snapshot.items() if value.get("nonfinite_count", 0) > 0
    ]
    has_valid_points = not (
        unexecuted or zero_absmax or invalid_norm or nonfinite
    )
    passed = (
        has_valid_points
        and not point_count_mismatch
        and (bool(snapshot) or empty_snapshot_is_valid)
    )
    return {
        "activation_point_count": observed_point_count,
        "required_point_count": expected_point_count,
        "point_count_mismatch": point_count_mismatch,
        # Deprecated compatibility field for existing deployment validators.
        "observer_count": observed_point_count,
        "unexecuted": unexecuted,
        "zero_absmax": zero_absmax,
        "invalid_norm": invalid_norm,
        "nonfinite": nonfinite,
        "status": "not_applicable" if passed and not snapshot else (
            "passed" if passed else "failed"
        ),
        "passed": passed,
    }


def resolve_convergence_checkpoints(
    specification: Any, total_samples: int
) -> tuple[list[int], list[int], list[int], int]:
    """Return configured, evaluated, skipped, and legacy checkpoint values."""

    if isinstance(specification, int):
        requested = [specification]
        saw_full = False
    else:
        values = specification if isinstance(specification, (list, tuple)) else [specification]
        requested = []
        saw_full = False
        for value in values:
            for token in str(value).split(","):
                token = token.strip().lower()
                if token == "full":
                    saw_full = True
                    continue
                if not token:
                    continue
                requested.append(int(token))
    if not requested and saw_full:
        return [], [], [], total_samples
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("checkpoint_samples must contain positive integers")

    legacy_checkpoint = max(requested)
    if len(requested) == 1:
        requested.extend(
            value
            for value in STANDARD_CONVERGENCE_CHECKPOINTS
            if value <= legacy_checkpoint
        )
    configured = sorted(set(requested))
    evaluated = [value for value in configured if value <= total_samples]
    skipped = [value for value in configured if value > total_samples]
    return configured, evaluated, skipped, legacy_checkpoint


def progress(records: list[dict[str, Any]], description: str):
    try:
        from tqdm import tqdm

        return tqdm(records, desc=description, unit="sample")
    except ImportError:
        return records


def run(args: argparse.Namespace) -> int:
    graph_set = language_graph_set(args.graph_set)
    if args.lm_head_w_bits != 8:
        raise RuntimeError("release activation calibration requires lm_head W8")
    if args.dtype != "float16":
        raise RuntimeError("release activation calibration requires float16")
    if args.max_samples != RELEASE_SAMPLE_COUNT:
        raise RuntimeError(
            f"release activation calibration requires {RELEASE_SAMPLE_COUNT} samples"
        )
    if args.chunk_size != 1024 or args.cache_len != 4096:
        raise RuntimeError(
            "release activation calibration requires chunk_size=1024 and cache_len=4096"
        )
    if args.image_token_id != 151665:
        raise RuntimeError(
            "release activation calibration requires image_token_id=151665"
        )

    selected_sha = sha256_file(args.selected_jsonl.resolve())
    if selected_sha != RELEASE_SELECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "release selected manifest SHA256 mismatch: "
            f"actual={selected_sha} expected={RELEASE_SELECTED_MANIFEST_SHA256}"
        )
    checkpoint_errors = release_checkpoint_errors(args.model_path)
    if checkpoint_errors:
        raise RuntimeError("; ".join(checkpoint_errors))
    prepare_errors = prepared_bundle_identity_errors(
        selected_jsonl=args.selected_jsonl,
        generated_jsonl=args.generated_jsonl,
        model_path=args.model_path,
        prepare_source_path=Path(__file__).with_name("prepare.py"),
        upstream_repo=args.upstream_repo,
        expected_sample_count=RELEASE_SAMPLE_COUNT,
    )
    if prepare_errors:
        raise RuntimeError(
            "prepared calibration identity check failed: "
            + "; ".join(prepare_errors)
        )

    from leap_llm.apis.model.locateanything_language import LocateAnythingLanguageApi

    manifest = args.generated_jsonl.resolve()
    records = read_generated_manifest(manifest, args.max_samples)
    random.Random(args.replay_seed).shuffle(records)
    (
        configured_checkpoints,
        evaluated_checkpoints,
        skipped_checkpoints,
        legacy_checkpoint,
    ) = resolve_convergence_checkpoints(args.checkpoint_samples, len(records))
    if legacy_checkpoint != RELEASE_CONVERGENCE_CHECKPOINT:
        raise RuntimeError(
            "release activation calibration requires the 512-sample "
            "convergence checkpoint"
        )
    snapshot_samples = sorted(set([*evaluated_checkpoints, len(records)]))
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    rotation, rotation_source = load_hidden_rotation(args.hidden_rotation_path, 2048)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_counts = dict(Counter(record["task"] for record in records))

    prepare_identity_path = manifest.parent / "prepare_run_identity.json"
    generation_summary_path = manifest.parent / "generation_summary.json"
    if not prepare_identity_path.is_file() or not generation_summary_path.is_file():
        raise RuntimeError(
            "prepared calibration tensors lack prepare identity/summary; rerun Prepare "
            "with the current code into a new output directory"
        )
    run_identity = {
        "schema_version": 1,
        "generated_manifest": file_identity(manifest),
        "selected_manifest": file_identity(args.selected_jsonl),
        "prepare_run_identity": file_identity(prepare_identity_path),
        "generation_summary": file_identity(generation_summary_path),
        "checkpoint": checkpoint_identity(args.model_path),
        "tokenizer": tokenizer_identity(args.model_path),
        "upstream_source": source_tree_identity(args.upstream_repo, {".py"}),
        "compiler_source": source_tree_identity(COMPILER_ROOT),
        "rotation": rotation_identity(args.hidden_rotation_path),
        "settings": {
            "device": args.device,
            "dtype": args.dtype,
            "component": args.component,
            "chunk_size": args.chunk_size,
            "cache_len": args.cache_len,
            "lm_head_w_bits": args.lm_head_w_bits,
            "sample_count": len(records),
            "convergence_checkpoints": configured_checkpoints,
            "legacy_checkpoint": legacy_checkpoint,
            "image_token_id": args.image_token_id,
            "replay_seed": args.replay_seed,
            "decode_context_policy": DECODE_CONTEXT_POLICY,
            "graph_set": graph_set.name,
        },
    }
    run_identity_sha256 = sha256_json(run_identity)
    identity_path = output_dir / "calibration_run_identity.json"
    durable_outputs = [
        output_dir / "calibration_scale_manifest.json",
        output_dir / "calibration_graph_coverage.json",
        output_dir / "scale_convergence.json",
        output_dir / f"scale_convergence_{legacy_checkpoint}_vs_{len(records)}.json",
    ]
    if args.resume and identity_path.is_file():
        previous = read_json(identity_path)
        previous_identity = previous.get("identity") if isinstance(previous, dict) else None
        mismatches = identity_mismatches(run_identity, previous_identity)
        if mismatches:
            raise RuntimeError(
                "calibration resume identity mismatch: " + ", ".join(mismatches[:12])
                + "; use a separate output directory"
            )
        if previous.get("status") == "complete":
            expected_artifacts = previous.get("artifacts")
            if not isinstance(expected_artifacts, dict):
                raise RuntimeError("completed calibration identity has no artifact catalog")
            actual_artifacts = artifact_identities(durable_outputs)
            artifact_mismatches = identity_mismatches(expected_artifacts, actual_artifacts)
            if artifact_mismatches:
                raise RuntimeError(
                    "completed calibration artifacts changed: "
                    + ", ".join(artifact_mismatches[:12])
                )
            print(
                f"[calibrate] resume identity matched; reused {len(records)} samples",
                flush=True,
            )
            return 0
    elif args.resume and any(path.exists() for path in durable_outputs):
        raise RuntimeError(
            "calibration outputs exist without calibration_run_identity.json; "
            "use a separate output directory"
        )
    elif not args.resume and (
        identity_path.exists() or any(path.exists() for path in durable_outputs)
    ):
        raise RuntimeError(
            "calibration output already contains run state; use --resume or a separate output directory"
        )
    atomic_identity_json(
        identity_path,
        {"schema_version": 1, "status": "running", "identity": run_identity},
    )

    vision_snapshots = {}
    vision_cosines = []
    vision_required_point_count = None
    language_snapshots = {}
    language_required_point_count = None
    stage_counts = Counter()
    decode_context_records: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []

    if args.component in {"all", "vision"}:
        from leap_llm.apis.model.locateanything_vision import LocateAnythingVisionApi

        print("\n================== VISION ACTIVATION STATISTICS ==================", flush=True)
        vision_api = LocateAnythingVisionApi(
            str(args.model_path.resolve()), str(output_dir / "vision_api"),
            image_width=672, image_height=672, device=args.device, w_bits=8,
            hidden_rotation_path=args.hidden_rotation_path, apply_hidden_rotation=True,
            export_only=True,
        )
        vision = vision_api.model.to(device=device, dtype=dtype).eval()
        vision.compile_mode(False)
        vision_tracker = ActivationTracker(vision, component="vision")
        vision_required_point_count = vision_tracker.tracked_module_count
        with torch.no_grad():
            for index, record in enumerate(progress(records, "Vision activation statistics"), 1):
                payload = load_tensor_payload(record)
                vision_tracker.stage = "vision"
                actual = vision(payload["vision_input"].to(device=device, dtype=dtype))
                expected = (payload["projected_visual_features"].float() @ rotation).to(
                    device=device, dtype=dtype
                )
                vision_cosines.append(float(torch.nn.functional.cosine_similarity(
                    actual.float().reshape(1, -1), expected.float().reshape(1, -1)
                ).item()))
                if index in snapshot_samples:
                    vision_snapshots[str(index)] = vision_tracker.snapshot(vision)
        activation_rows.extend(vision_tracker.activation_statistics(vision))
        vision_tracker.close()
        del vision, vision_api
        gc.collect()
        torch.cuda.empty_cache()

    if args.component in {"all", "language"}:
        print("\n================== LANGUAGE ACTIVATION STATISTICS ==================", flush=True)
        language_api = LocateAnythingLanguageApi(
            str(args.model_path.resolve()), str(output_dir / "language_api"),
            chunk_size=args.chunk_size, cache_len=args.cache_len, decode_seq_len=6,
            device=args.device, w_bits=8, lm_head_w_bits=args.lm_head_w_bits,
            hidden_rotation_path=args.hidden_rotation_path,
            apply_hidden_rotation=True, export_only=True,
            graph_set=graph_set.name,
        )
        language = language_api.text_model.to(device=device, dtype=dtype).eval()
        language.compile_mode(False)
        language_tracker = ActivationTracker(language, component="language")
        language_required_point_count = language_tracker.tracked_module_count
        num_layers = language.config.num_hidden_layers
        num_kv = language.config.num_key_value_heads
        head_dim = language.config.hidden_size // language.config.num_attention_heads
        zero_caches = [torch.zeros(
            (1, args.cache_len, num_kv, head_dim), device=device, dtype=dtype
        ) for _ in range(num_layers * 2)]
        with torch.no_grad():
            for index, record in enumerate(progress(records, "Language activation statistics"), 1):
                payload = load_tensor_payload(record)
                replay_contexts = select_decode_replay_contexts(
                    payload,
                    task=record["task"],
                    chunk_size=args.chunk_size,
                )
                for replay_context in replay_contexts:
                    language_tracker.stage = "prefill"
                    embeds, positions, mask, active_len = build_prefill_inputs(
                        language, payload, rotation, chunk_size=args.chunk_size,
                        cache_len=args.cache_len, image_token_id=args.image_token_id,
                        device=device, dtype=dtype,
                        suffix_token_ids=replay_context.suffix_token_ids,
                    )
                    if active_len != replay_context.past_len:
                        raise RuntimeError(
                            f"{replay_context.context_id}: Prefill active_len={active_len} "
                            f"does not match selected past_len={replay_context.past_len}"
                        )
                    decode_context_records.append(
                        replay_context.coverage_record(record["task"])
                    )
                    logits, new_keys, new_values = language(
                        embeds, positions, mask, *zero_caches
                    )
                    stage_counts["prefill"] += 1
                    del logits
                    cache_keys, cache_values = build_right_aligned_caches(
                        new_keys, new_values,
                        active_len=active_len,
                        cache_len=args.cache_len,
                    )

                    pbd_tokens = select_pbd_tokens(
                        payload,
                        6,
                        int(language.config.text_mask_token_id),
                        anchor_token_id=replay_context.anchor_token_id,
                    )
                    language_tracker.stage = "pbd_q6"
                    pbd_embeds, pbd_pos, pbd_mask = build_decode_inputs(
                        language, pbd_tokens, q_len=6, past_len=active_len,
                        cache_len=args.cache_len, is_pbd=True,
                        device=device, dtype=dtype,
                    )
                    pbd_out = language(
                        pbd_embeds, pbd_pos, pbd_mask,
                        *(cache_keys + cache_values),
                    )
                    stage_counts["pbd_q6"] += 1
                    del pbd_out, pbd_embeds, pbd_pos, pbd_mask

                    if graph_set.uses_fused_decode:
                        for prefix_len in range(1, 7):
                            prefix = list(replay_context.pending_token_ids[:prefix_len])
                            fused_tokens = [
                                *prefix,
                                prefix[-1],
                                *([int(language.config.text_mask_token_id)] * 5),
                            ]
                            q_len = prefix_len + 6
                            stage = f"pbd_q{q_len}"
                            language_tracker.stage = stage
                            fused_embeds, fused_pos, fused_mask = build_decode_inputs(
                                language, fused_tokens, q_len=q_len, past_len=active_len,
                                cache_len=args.cache_len, is_pbd=True,
                                pbd_prefix_len=prefix_len, device=device, dtype=dtype,
                            )
                            fused_out = language(
                                fused_embeds, fused_pos, fused_mask,
                                *(cache_keys + cache_values),
                            )
                            stage_counts[stage] += 1
                            del fused_out, fused_embeds, fused_pos, fused_mask

                        for q_len in range(1, 6):
                            ar_tokens = list(replay_context.pending_token_ids[:q_len])
                            stage = f"ar_q{q_len}"
                            language_tracker.stage = stage
                            ar_embeds, ar_pos, ar_mask = build_decode_inputs(
                                language, ar_tokens, q_len=q_len, past_len=active_len,
                                cache_len=args.cache_len, is_pbd=False,
                                device=device, dtype=dtype,
                            )
                            ar_out = language(
                                ar_embeds, ar_pos, ar_mask,
                                *(cache_keys + cache_values),
                            )
                            stage_counts[stage] += 1
                            del ar_out, ar_embeds, ar_pos, ar_mask
                    else:
                        language_tracker.stage = "ar_q1"
                        cache_keys, cache_values = replay_sequential_ar_q1(
                            language,
                            replay_context.pending_token_ids,
                            cache_keys,
                            cache_values,
                            active_len=active_len,
                            cache_len=args.cache_len,
                            device=device,
                            dtype=dtype,
                        )
                        stage_counts["ar_q1"] += len(
                            replay_context.pending_token_ids
                        )
                    del (
                        cache_keys, cache_values, new_keys, new_values,
                        embeds, positions, mask,
                    )
                if index in snapshot_samples:
                    language_snapshots[str(index)] = language_tracker.snapshot(language)
        activation_rows.extend(language_tracker.activation_statistics(language))
        language_tracker.close()
        del zero_caches, language, language_api
        gc.collect()
        torch.cuda.empty_cache()

    full_samples = len(records)
    language_context_count = len(decode_context_records)
    full = str(full_samples)
    audits = {}
    if vision_snapshots:
        audits["vision"] = activation_statistics_audit(
            vision_snapshots[full],
            required_point_count=vision_required_point_count,
        )
    if language_snapshots:
        audits["language"] = activation_statistics_audit(
            language_snapshots[full],
            required_point_count=language_required_point_count,
        )
    scale_manifest = {
        "schema_version": 3,
        "generated_manifest": str(manifest),
        "generated_manifest_sha256": sha256_file(manifest),
        "rotation_source": rotation_source,
        "rotation_file_sha256": (
            sha256_file(Path(args.hidden_rotation_path).resolve())
            if args.hidden_rotation_path else None
        ),
        "sample_count": full_samples,
        "language_context_count": language_context_count,
        "checkpoint_samples": legacy_checkpoint,
        "convergence_checkpoints": configured_checkpoints,
        "recorded_snapshot_samples": snapshot_samples,
        "skipped_convergence_checkpoints": skipped_checkpoints,
        "replay_order": "deterministic_shuffle",
        "replay_seed": args.replay_seed,
        "calibration_run_identity_sha256": run_identity_sha256,
        "task_counts": task_counts,
        "profile": {
            "component": args.component,
            "chunk_size": args.chunk_size,
            "cache_len": args.cache_len,
            "pbd_query_len": 6,
            "ar_query_len": 1,
            "graph_set": graph_set.name,
        },
    }
    if vision_snapshots:
        scale_manifest["vision"] = vision_snapshots
    if language_snapshots:
        scale_manifest["language"] = language_snapshots
        scale_manifest["profile"].update({
            "language_decoder_weight_bits": 8,
            "language_lm_head_weight_bits": args.lm_head_w_bits,
            "text_mask_token_id": 151676,
            "pbd_block_size": 6,
            "pbd_total_query_lengths": (
                list(range(6, 13)) if graph_set.uses_fused_decode else [6]
            ),
            "ar_total_query_lengths": (
                list(range(1, 6)) if graph_set.uses_fused_decode else [1]
            ),
            "ar_q1_calls_per_context": graph_set.sequential_ar_q1_tokens,
            "pbd_q6_role": "post_prefill_bootstrap_only",
            "pbd_input_protocol": "accepted_prefix_plus_duplicated_anchor_plus_5_text_masks",
            "decode_context_policy": DECODE_CONTEXT_POLICY,
        })
    expected_stages = []
    stage_execution_counts = {}
    expected_stage_execution_counts = {}
    if vision_snapshots:
        expected_stages.append("vision")
        stage_execution_counts["vision"] = full_samples
        expected_stage_execution_counts["vision"] = full_samples
    if language_snapshots:
        expected_stages.extend(graph_set.calibration_stages)
        stage_execution_counts.update(stage_counts)
        expected_stage_execution_counts.update(
            graph_set.calibration_execution_counts(language_context_count)
        )
    coverage = {
        "schema_version": 3,
        "generated_manifest_sha256": sha256_file(manifest),
        "sample_count": full_samples,
        "language_context_count": language_context_count,
        "checkpoint_samples": legacy_checkpoint,
        "convergence_checkpoints": configured_checkpoints,
        "recorded_snapshot_samples": snapshot_samples,
        "skipped_convergence_checkpoints": skipped_checkpoints,
        "task_counts": task_counts,
        "calibration_run_identity_sha256": run_identity_sha256,
        "profile": scale_manifest["profile"],
        "stage_execution_counts": stage_execution_counts,
        "expected_stage_execution_counts": expected_stage_execution_counts,
        "expected_stages": expected_stages,
        "all_stages_executed": stage_execution_counts == expected_stage_execution_counts,
        "activation_statistics_audit": audits,
        "activation_statistics_audit_passed": all(
            audit["passed"] for audit in audits.values()
        ),
    }
    if language_snapshots:
        coverage["decode_context_coverage"] = summarize_decode_context_coverage(
            decode_context_records,
            expected_samples=full_samples,
            cache_len=args.cache_len,
        )
        coverage["decode_context_coverage_passed"] = coverage[
            "decode_context_coverage"
        ]["passed"]
        if (
            coverage["decode_context_coverage"]["language_context_count"]
            != language_context_count
        ):
            raise RuntimeError("Language context count does not match coverage records")
    if vision_cosines:
        coverage["vision_cosine_min"] = min(vision_cosines)
        coverage["vision_cosine_mean"] = sum(vision_cosines) / len(vision_cosines)
    convergence = {
        "schema_version": 2,
        "configured_checkpoints": configured_checkpoints,
        "evaluated_checkpoints": snapshot_samples,
        "skipped_checkpoints": skipped_checkpoints,
        "full_samples": full_samples,
        "comparison_basis": "each checkpoint and each adjacent pair",
        "components": {},
    }
    component_snapshots = {
        name: snapshots
        for name, snapshots in (
            ("vision", vision_snapshots),
            ("language", language_snapshots),
        )
        if snapshots
    }
    for component, snapshots in component_snapshots.items():
        points = sorted(int(value) for value in snapshots)
        vs_full = [
            compare_snapshots(
                snapshots[str(value)],
                snapshots[full],
                first_samples=value,
                second_samples=full_samples,
            )
            for value in points
            if value != full_samples
        ]
        adjacent = [
            compare_snapshots(
                snapshots[str(before)],
                snapshots[str(after)],
                first_samples=before,
                second_samples=after,
            )
            for before, after in zip(points, points[1:])
        ]
        convergence["components"][component] = {
            "snapshot_samples": points,
            "vs_full": vs_full,
            "adjacent": adjacent,
        }
    atomic_json(output_dir / "calibration_scale_manifest.json", scale_manifest)
    atomic_json(output_dir / "calibration_graph_coverage.json", coverage)
    atomic_json(output_dir / "scale_convergence.json", convergence)
    if str(legacy_checkpoint) in vision_snapshots or str(legacy_checkpoint) in language_snapshots:
        legacy_convergence = {
            "schema_version": 1,
            "checkpoint_samples": legacy_checkpoint,
            "full_samples": full_samples,
        }
        for component, snapshots in component_snapshots.items():
            if str(legacy_checkpoint) in snapshots:
                legacy_convergence[component] = compare_snapshots(
                    snapshots[str(legacy_checkpoint)],
                    snapshots[full],
                    first_samples=legacy_checkpoint,
                    second_samples=full_samples,
                )
        atomic_json(
            output_dir / f"scale_convergence_{legacy_checkpoint}_vs_{full_samples}.json",
            legacy_convergence,
        )
    activation_report = generate_activation_report(
        output_dir,
        activation_rows,
        convergence,
        coverage,
        metadata={
            "generated_manifest": str(manifest),
            "generated_manifest_sha256": sha256_file(manifest),
            "sample_count": full_samples,
            "task_counts": task_counts,
        },
    )
    plot_status = activation_report.get("plots", {})
    if plot_status.get("status") != "generated":
        raise RuntimeError(
            "calibration report generation did not complete: "
            + str(plot_status.get("reason") or "required figures are missing")
        )
    print(json.dumps(coverage, sort_keys=True), flush=True)
    if not coverage["all_stages_executed"]:
        raise RuntimeError("not all required calibration graph paths were executed")
    if not coverage["activation_statistics_audit_passed"]:
        raise RuntimeError(
            "activation statistics audit found unexecuted, non-finite, or invalid scales"
        )
    if language_snapshots and not coverage["decode_context_coverage_passed"]:
        errors = coverage["decode_context_coverage"]["errors"]
        raise RuntimeError(
            "Language Decode context coverage failed: " + "; ".join(errors[:12])
        )
    atomic_identity_json(
        identity_path,
        {
            "schema_version": 1,
            "status": "complete",
            "identity": run_identity,
            "artifacts": artifact_identities(durable_outputs),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--generated-jsonl", type=Path, required=True)
    result.add_argument("--selected-jsonl", type=Path, required=True)
    result.add_argument("--upstream-repo", type=Path, required=True)
    result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--dtype", choices=["float16"], default="float16")
    result.add_argument(
        "--component", choices=["all", "vision", "language"], default="all"
    )
    result.add_argument("--chunk-size", type=int, default=1024)
    result.add_argument("--cache-len", type=int, default=4096)
    result.add_argument("--lm-head-w-bits", type=int, choices=[8], default=8)
    result.add_argument("--max-samples", type=int, default=RELEASE_SAMPLE_COUNT)
    result.add_argument(
        "--checkpoint-samples",
        default="64,128,256,512",
        help="comma-separated convergence checkpoints; full is always included",
    )
    result.add_argument("--image-token-id", type=int, default=151665)
    result.add_argument("--hidden-rotation-path")
    result.add_argument("--replay-seed", type=int, default=20260729)
    result.add_argument(
        "--graph-set",
        dest="graph_set",
        choices=("standard", "fused_decode"),
        default="standard",
    )
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
