#!/usr/bin/env python3
"""Validate LocateAnything fixed-graph Hybrid generation against annotations."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


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
from compiler.scripts.validate.evaluate_grounding import evaluate  # noqa: E402
from compiler.scripts.common.hybrid_generation import (  # noqa: E402
    FixedGraphHybridGenerator,
    HybridGenerationConfig,
    load_official_decoding,
    seed_generation,
)
from compiler.scripts.common.language import (  # noqa: E402
    AR_QUERY_LEN,
    CACHE_LEN,
    CHUNK_SIZE,
    PBD_QUERY_LEN,
    LanguageEagerRunner,
    create_language_model,
    language_quantization_policy,
    load_payload,
)


MODES = (
    "official_saved_hybrid",
    "adapted_float_hybrid",
    "quantized_eager_hybrid",
)
LEGACY_BASE_SEED = 20260718
GENERATION_CONFIG_SOURCE = (
    "compiler/quantize.py prepare defaults"
)
DEFAULT_MAX_NEW_TOKENS = 2048
SIX_DOMAIN_TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")


def discover_payloads(input_dir: Path) -> list[Path]:
    paths = sorted(path for path in input_dir.rglob("*.pt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no .pt Language payloads under {input_dir}")
    return paths


def deterministic_seed(base_seed: int, bundle_id: str, mode: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{bundle_id}|{mode}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _manifest_candidates(input_dir: Path) -> list[Path]:
    candidates = [
        input_dir / "generated.jsonl",
        input_dir.parent / "generated.jsonl",
        input_dir.parent.parent / "generated.jsonl",
    ]
    return list(dict.fromkeys(path.resolve() for path in candidates))


def load_generation_manifest(input_dir: Path) -> tuple[Path, dict[str, Any]]:
    for path in _manifest_candidates(input_dir):
        if not path.is_file():
            continue
        records: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                bundle_id = str(record.get("bundle_id") or "")
                if not bundle_id:
                    raise ValueError(f"{path}:{line_number}: missing bundle_id")
                if bundle_id in records:
                    raise ValueError(f"{path}:{line_number}: duplicate {bundle_id}")
                records[bundle_id] = record
        return path, records
    raise FileNotFoundError(
        f"generated.jsonl is required for official Hybrid validation under {input_dir}"
    )


def validate_manifest_coverage(
    payload_paths: list[Path], manifest_path: Path, manifest: dict[str, Any]
) -> None:
    payload_ids = {path.stem for path in payload_paths}
    manifest_ids = set(manifest)
    if payload_ids != manifest_ids:
        missing = sorted(payload_ids - manifest_ids)
        extra = sorted(manifest_ids - payload_ids)
        raise ValueError(
            f"payload/manifest set mismatch for {manifest_path}: "
            f"missing_manifest={missing[:3]} extra_manifest={extra[:3]}"
        )


def _metadata_candidates(input_dir: Path) -> list[Path]:
    roots = [input_dir, input_dir.parent, input_dir.parent.parent]
    return list(
        dict.fromkeys(
            [
                path.resolve()
                for root in roots
                for path in (
                    root / "d3_job_metadata.json",
                    root / "generation_summary.json",
                )
            ]
        )
    )


def load_generation_metadata(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata: dict[str, Any] = {}
    provenance: list[dict[str, str]] = []
    for path in _metadata_candidates(input_dir):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                metadata.setdefault(key, value)
            provenance.append({"path": str(path), "sha256": sha256(path)})
    return metadata, provenance


def official_seed(
    bundle_id: str, payload: dict[str, Any], manifest: dict[str, Any]
) -> tuple[int, str]:
    payload_seeds = payload.get("prediction_seeds", {})
    if isinstance(payload_seeds, dict) and isinstance(
        payload_seeds.get("hybrid"), int
    ):
        return int(payload_seeds["hybrid"]), "payload:prediction_seeds.hybrid"
    record = manifest.get(bundle_id, {})
    prediction = record.get("prediction", {}) if isinstance(record, dict) else {}
    hybrid = prediction.get("hybrid", {}) if isinstance(prediction, dict) else {}
    if isinstance(hybrid, dict) and isinstance(hybrid.get("seed"), int):
        return int(hybrid["seed"]), "generated.jsonl:prediction.hybrid.seed"
    return (
        deterministic_seed(LEGACY_BASE_SEED, bundle_id, "hybrid"),
        "derived_from_legacy_base_seed_20260718",
    )


def generation_config_from_payload(
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    max_new_tokens_override: int | None = None,
) -> tuple[HybridGenerationConfig, str]:
    raw = payload.get("generation_config")
    if not isinstance(raw, dict):
        metadata = metadata or {}
        max_new_tokens = int(
            metadata.get("max_new_tokens", HybridGenerationConfig().max_new_tokens)
        )
        source = (
            "d3_job_metadata.json:max_new_tokens + "
            f"{GENERATION_CONFIG_SOURCE}"
            if "max_new_tokens" in metadata
            else GENERATION_CONFIG_SOURCE
        )
        config = HybridGenerationConfig(max_new_tokens=max_new_tokens)
        config.validate()
    else:
        config = HybridGenerationConfig(
            max_new_tokens=int(raw["max_new_tokens"]),
            temperature=float(raw["temperature"]),
            top_p=float(raw["top_p"]),
            top_k=(int(raw["top_k"]) if raw.get("top_k") is not None else None),
            repetition_penalty=float(raw["repetition_penalty"]),
        )
        source = "payload:generation_config"
    if max_new_tokens_override is not None:
        config = HybridGenerationConfig(
            max_new_tokens=max_new_tokens_override,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )
        source = "cli:--max_new_tokens"
    config.validate()
    return config, source


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )


def decode_response(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.batch_decode(
        [token_ids], skip_special_tokens=False
    )[0]


def load_checkpoint_token_config(
    model_path: Path, official: Any
) -> tuple[dict[str, int], int, dict[str, str]]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = SimpleNamespace(**raw)
    text_config = raw.get("text_config")
    config.text_config = (
        SimpleNamespace(**text_config) if isinstance(text_config, dict) else None
    )
    token_ids = {
        str(name): int(value)
        for name, value in official.get_token_ids_from_config(config).items()
    }
    image_token_id = int(raw.get("image_token_index", 151665))
    return token_ids, image_token_id, {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
    }


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def sequence_comparison(left: list[int], right: list[int]) -> dict[str, Any]:
    prefix = common_prefix_length(left, right)
    denominator = max(len(left), len(right))
    return {
        "exact": left == right,
        "left_length": len(left),
        "right_length": len(right),
        "common_prefix_length": prefix,
        "common_prefix_rate": prefix / denominator if denominator else 1.0,
    }


def append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def pairwise_summary(samples: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    comparisons = [sample["sequence_comparisons"][key] for sample in samples]
    return {
        "records": len(comparisons),
        "exact_records": sum(bool(row["exact"]) for row in comparisons),
        "exact_rate": (
            sum(bool(row["exact"]) for row in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
        "mean_common_prefix_rate": (
            sum(float(row["common_prefix_rate"]) for row in comparisons)
            / len(comparisons)
            if comparisons
            else None
        ),
    }


def hybrid_control_flow_summary(
    samples: Iterable[dict[str, Any]], mode: str
) -> dict[str, Any]:
    results = [sample[mode] for sample in samples]
    patterns = Counter(
        step["pattern"]
        for result in results
        for step in result["steps"]
        if step["mode"] == "pbd"
    )
    pbd_calls = sum(patterns.values())
    error_boxes = int(patterns.get("error_box", 0))
    stop_reasons = Counter(str(result["stop_reason"]) for result in results)
    return {
        "records": len(results),
        "pbd_calls": pbd_calls,
        "pbd_pattern_counts": dict(sorted(patterns.items())),
        "pbd_direct_acceptance_rate": (
            (pbd_calls - error_boxes) / pbd_calls if pbd_calls else None
        ),
        "pbd_error_box_rate": error_boxes / pbd_calls if pbd_calls else None,
        "pbd_fused_prefix_calls": sum(
            int(result.get("pbd_fused_prefix_calls", 0)) for result in results
        ),
        "q1_commit_calls": sum(int(result["q1_commit_calls"]) for result in results),
        "ar_fallback_sample_calls": sum(
            int(result["ar_fallback_sample_calls"]) for result in results
        ),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "generation_limit_records": sum(
            stop_reasons.get(reason, 0)
            for reason in ("max_new_tokens", "model_max_length", "cache_limit")
        ),
    }


def write_summary_csv(
    path: Path,
    accuracy: dict[str, Any],
    sequence_comparisons: dict[str, Any],
    control_flow: dict[str, Any],
) -> None:
    fieldnames = [
        "row_type",
        "mode",
        "scope",
        "comparison",
        "records",
        "exact_rate",
        "mean_common_prefix_rate",
        "pbd_calls",
        "pbd_direct_acceptance_rate",
        "pbd_error_box_rate",
        "pbd_fused_prefix_calls",
        "q1_commit_calls",
        "ar_fallback_sample_calls",
        "generation_limit_records",
        "format_valid_rate",
        "structured_exact_match_rate",
        "label_f1",
        "box_precision",
        "box_recall",
        "box_f1",
        "single_box_mean_iou",
        "point_mean_distance_grid",
        "point_pck_0.05_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, values in accuracy["modes"].items():
            scopes = {"overall": values["overall"], **values["by_task"]}
            for scope, metrics in scopes.items():
                writer.writerow(
                    {
                        "row_type": "accuracy",
                        "mode": mode,
                        "scope": scope,
                        "records": metrics["records"],
                        "format_valid_rate": metrics["format"]["valid_rate"],
                        "structured_exact_match_rate": metrics[
                            "structured_exact_match_rate"
                        ],
                        "label_f1": metrics["label_ref"]["f1"],
                        "box_precision": metrics["box_iou"]["precision"],
                        "box_recall": metrics["box_iou"]["recall"],
                        "box_f1": metrics["box_iou"]["f1"],
                        "single_box_mean_iou": metrics["single_box_iou"][
                            "mean_iou"
                        ],
                        "point_mean_distance_grid": metrics["point"][
                            "target_mean_distance_grid"
                        ],
                        "point_pck_0.05_recall": metrics["point"]["pck"]
                        .get("0.05", {})
                        .get("recall"),
                    }
                )
        for comparison, metrics in sequence_comparisons.items():
            writer.writerow(
                {
                    "row_type": "sequence_comparison",
                    "comparison": comparison,
                    "records": metrics["records"],
                    "exact_rate": metrics["exact_rate"],
                    "mean_common_prefix_rate": metrics[
                        "mean_common_prefix_rate"
                    ],
                }
            )
        for mode, metrics in control_flow.items():
            writer.writerow(
                {
                    "row_type": "hybrid_control_flow",
                    "mode": mode,
                    "records": metrics["records"],
                    "pbd_calls": metrics["pbd_calls"],
                    "pbd_direct_acceptance_rate": metrics[
                        "pbd_direct_acceptance_rate"
                    ],
                    "pbd_error_box_rate": metrics["pbd_error_box_rate"],
                    "pbd_fused_prefix_calls": metrics["pbd_fused_prefix_calls"],
                    "q1_commit_calls": metrics["q1_commit_calls"],
                    "ar_fallback_sample_calls": metrics[
                        "ar_fallback_sample_calls"
                    ],
                    "generation_limit_records": metrics[
                        "generation_limit_records"
                    ],
                }
            )


def validate_payload_identity(
    path: Path, payload: dict[str, Any], manifest: dict[str, Any]
) -> str:
    bundle_id = str(payload.get("bundle_id") or path.stem)
    if path.stem != bundle_id:
        raise ValueError(f"{path}: filename and bundle_id disagree")
    record = manifest.get(bundle_id)
    if record is None:
        raise ValueError(f"{bundle_id}: missing generated manifest row")
    if record.get("status") != "complete":
        raise ValueError(f"{bundle_id}: generated manifest row is not complete")
    tensor_file = Path(str(record.get("tensor_file") or ""))
    if tensor_file.name != path.name:
        raise ValueError(f"{bundle_id}: tensor_file does not match payload filename")
    expected_sha256 = str(record.get("tensor_sha256") or "")
    if len(expected_sha256) != 64 or sha256(path) != expected_sha256:
        raise ValueError(f"{bundle_id}: payload SHA256 does not match manifest")
    return bundle_id


def official_saved_reference(
    bundle_id: str, payload: dict[str, Any], manifest: dict[str, Any]
) -> tuple[list[int], str]:
    prediction = manifest[bundle_id].get("prediction", {})
    hybrid = prediction.get("hybrid", {}) if isinstance(prediction, dict) else {}
    if not isinstance(hybrid, dict) or not isinstance(hybrid.get("answer"), str):
        raise ValueError(f"{bundle_id}: manifest has no original Hybrid answer")
    manifest_tokens = hybrid.get("token_ids")
    if not isinstance(manifest_tokens, list):
        raise ValueError(f"{bundle_id}: manifest has no Hybrid token_ids")
    saved = [int(token) for token in manifest_tokens]
    payload_saved = [
        int(token)
        for token in payload["prediction_token_ids"]["hybrid"].reshape(-1).tolist()
    ]
    if payload_saved != saved:
        raise ValueError(f"{bundle_id}: payload/manifest Hybrid token_ids disagree")
    return saved, hybrid["answer"]


def tensor_equivalence(reference: Any, candidate: Any) -> dict[str, Any]:
    """Report a compact numeric comparison without retaining large tensors."""

    import torch

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "status": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    left = reference.detach().float()
    right = candidate.detach().float()
    delta = right - left
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    cosine = torch.sum(left * right) / torch.clamp(
        left_norm * right_norm, min=torch.finfo(left.dtype).tiny
    )
    relative_l2 = torch.linalg.vector_norm(delta) / torch.clamp(
        left_norm, min=torch.finfo(left.dtype).tiny
    )
    result = {
        "status": "compared",
        "shape": list(reference.shape),
        "exact_equal": bool(torch.equal(reference, candidate)),
        "cosine": float(cosine.item()),
        "relative_l2": float(relative_l2.item()),
        "max_abs": float(delta.abs().max().item()),
    }
    if reference.ndim >= 1 and reference.shape[-1] > 1:
        result["top1_agreement"] = float(
            (left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item()
        )
    return result


def summarize_equivalence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [row for row in rows if row.get("status") == "compared"]
    if not compared:
        return {"status": "not_compared", "count": len(rows)}
    return {
        "status": "compared",
        "count": len(compared),
        "exact_equal_count": sum(bool(row["exact_equal"]) for row in compared),
        "min_cosine": min(float(row["cosine"]) for row in compared),
        "max_relative_l2": max(float(row["relative_l2"]) for row in compared),
        "max_abs": max(float(row["max_abs"]) for row in compared),
    }


def fused_prefix_equivalence_probe(
    generator: FixedGraphHybridGenerator,
    payload: dict[str, Any],
    prefix_tokens: list[int],
    *,
    quantized: bool,
    seed: int,
    config: HybridGenerationConfig,
) -> dict[str, Any]:
    """Compare legacy commit+q6 against one fused q(6+k) execution.

    This is independent of stochastic token sampling. The accepted prefix is
    processed once as ordinary causal decode and once as the causal prefix of a
    PBD graph; their K/V and the following q6 logits must agree in Float.
    """

    import torch

    if not 1 <= len(prefix_tokens) <= generator.pbd_query_len:
        raise ValueError(f"invalid accepted PBD prefix: {prefix_tokens}")
    token_ids = generator._validated_token_ids(payload)
    prefix_len = len(prefix_tokens)
    keys, values, history_len = generator._prefill(payload, quantized)
    causal_logits = causal_keys = causal_values = None
    committed_keys = committed_values = None
    legacy_logits = legacy_keys = legacy_values = None
    fused_logits = fused_keys = fused_values = None
    try:
        causal_logits, causal_keys, causal_values = generator._run_causal(
            prefix_tokens, keys, values, history_len, quantized
        )
        committed_keys, committed_values = generator._retain_prefix(
            keys, values, causal_keys, causal_values, prefix_len
        )
        legacy_tokens = [
            prefix_tokens[-1],
            *([token_ids["default_mask_token_id"]] * (generator.pbd_query_len - 1)),
        ]
        legacy_logits, legacy_keys, legacy_values = generator._run_pbd(
            legacy_tokens,
            committed_keys,
            committed_values,
            history_len + prefix_len,
            quantized,
            0,
        )
        fused_tokens = [*prefix_tokens, *legacy_tokens]
        fused_logits, fused_keys, fused_values = generator._run_pbd(
            fused_tokens,
            keys,
            values,
            history_len,
            quantized,
            prefix_len,
        )
        prefix_rows = [
            tensor_equivalence(reference[:, :prefix_len], candidate[:, :prefix_len])
            for reference, candidate in zip(causal_keys + causal_values,
                                            fused_keys + fused_values, strict=True)
        ]
        suffix_rows = [
            tensor_equivalence(reference, candidate[:, prefix_len:])
            for reference, candidate in zip(legacy_keys + legacy_values,
                                            fused_keys + fused_values, strict=True)
        ]
        generated = torch.cat(
            (
                payload["prompt_input_ids"].reshape(-1).to(
                    device=generator.device, dtype=torch.long
                ),
                torch.tensor(
                    prefix_tokens, device=generator.device, dtype=torch.long
                ),
            )
        ).unsqueeze(0)

        def decode_decision(logits: Any) -> dict[str, Any]:
            seed_generation(seed, generator.device)
            probabilities, confidence, sampled, decoded = (
                generator.official.sample_tokens(
                    logits,
                    generated,
                    token_ids,
                    keep_k=5,
                    generation_mode="hybrid",
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                )
            )
            use_sample = bool((decoded[0] == 0).all().item())
            selected = sampled[0] if use_sample else decoded[0]
            pattern = generator.official.handle_pattern(
                selected, token_ids, "hybrid"
            )
            result = {
                "pattern": str(pattern["type"]),
                "accepted_token_ids": [int(token) for token in pattern["tokens"]],
                "used_sample": use_sample,
            }
            del probabilities, confidence, sampled, decoded, selected
            return result

        legacy_decision = decode_decision(legacy_logits)
        fused_decision = decode_decision(fused_logits[:, prefix_len:])
        return {
            "status": "compared",
            "quantized": quantized,
            "prefix_token_ids": prefix_tokens,
            "prefix_len": prefix_len,
            "fused_q_len": prefix_len + generator.pbd_query_len,
            "causal_prefix_kv": summarize_equivalence(prefix_rows),
            "pbd_logits": tensor_equivalence(
                legacy_logits, fused_logits[:, prefix_len:]
            ),
            "pbd_suffix_kv": summarize_equivalence(suffix_rows),
            "legacy_decision": legacy_decision,
            "fused_decision": fused_decision,
            "decision_equal": legacy_decision == fused_decision,
        }
    finally:
        generator.emulator.set_enabled(False)
        del (
            keys,
            values,
            causal_logits,
            causal_keys,
            causal_values,
            committed_keys,
            committed_values,
            legacy_logits,
            legacy_keys,
            legacy_values,
            fused_logits,
            fused_keys,
            fused_values,
        )
        gc.collect()
        torch.cuda.empty_cache()


def pbd_probe_prefix(result: dict[str, Any]) -> list[int] | None:
    for step in result["steps"]:
        tokens = [int(token) for token in step["accepted_token_ids"]]
        if step["mode"] == "pbd" and step.get("pattern") != "im_end" and tokens:
            return tokens
    return None


def run(args: argparse.Namespace) -> int:
    import torch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    csv_path = output_dir / "report.csv"
    samples_path = output_dir / "samples.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    payload_paths = discover_payloads(args.input_dir.resolve())
    manifest_path, manifest = load_generation_manifest(args.input_dir.resolve())
    if args.one_per_task:
        selected_by_task: dict[str, Path] = {}
        for path in payload_paths:
            record = manifest.get(path.stem)
            task = str(record.get("task") or "") if isinstance(record, dict) else ""
            if task in SIX_DOMAIN_TASKS and task not in selected_by_task:
                selected_by_task[task] = path
        missing_tasks = [task for task in SIX_DOMAIN_TASKS if task not in selected_by_task]
        if missing_tasks:
            raise ValueError(
                f"input set lacks required six-domain tasks: {missing_tasks}"
            )
        payload_paths = [selected_by_task[task] for task in SIX_DOMAIN_TASKS]
        selected_ids = {path.stem for path in payload_paths}
        manifest = {key: value for key, value in manifest.items() if key in selected_ids}
    elif args.nums is not None:
        if args.nums <= 0:
            raise ValueError("--nums must be positive when supplied")
        payload_paths = payload_paths[: args.nums]
        selected_ids = {path.stem for path in payload_paths}
        missing_manifest = sorted(selected_ids - set(manifest))
        if missing_manifest:
            raise ValueError(
                f"selected payloads are missing from manifest: {missing_manifest[:3]}"
            )
        manifest = {key: value for key, value in manifest.items() if key in selected_ids}
    validate_manifest_coverage(payload_paths, manifest_path, manifest)
    generation_metadata, generation_provenance = load_generation_metadata(
        args.input_dir.resolve()
    )
    device = detect_float_device()
    if not device.startswith("cuda"):
        raise RuntimeError("official Hybrid validation requires an NVIDIA CUDA device")

    api = model = rotation = runner = tokenizer = None
    first_payload = load_payload(payload_paths[0])
    generation_config, generation_config_source = generation_config_from_payload(
        first_payload, generation_metadata, args.max_new_tokens
    )
    del first_payload
    official = load_official_decoding(args.model_path.resolve())
    tokenizer = load_tokenizer(args.model_path.resolve())
    checkpoint_token_ids, image_token_id, checkpoint_config = (
        load_checkpoint_token_config(args.model_path.resolve(), official)
    )
    tokenizer_model_max_length = int(tokenizer.model_max_length)
    if tokenizer_model_max_length <= 0:
        raise ValueError("tokenizer.model_max_length must be positive")
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
        capture_boundaries=False,
        capture_operators=False,
    )
    generator = FixedGraphHybridGenerator(
        model,
        rotation,
        runner.device,
        runner.dtype,
        runner.zero_caches,
        runner.emulator,
        official,
        chunk_size=CHUNK_SIZE,
        cache_len=CACHE_LEN,
        pbd_query_len=PBD_QUERY_LEN,
        image_token_id=image_token_id,
        token_ids=checkpoint_token_ids,
        model_max_length=tokenizer_model_max_length,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "method": "fixed_graph_hybrid_end_to_end_validation",
        "model_path": str(args.model_path.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "payload_count": len(payload_paths),
        "processed": 0,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": sha256(manifest_path),
        "generation_config": generation_config.as_dict(),
        "generation_config_source": generation_config_source,
        "generation_provenance": generation_provenance,
        "official_decoding": official.describe(),
        "checkpoint_config": checkpoint_config,
        "checkpoint_token_ids": checkpoint_token_ids,
        "image_token_id": image_token_id,
        "tokenizer_model_max_length": tokenizer_model_max_length,
        "graph_profile": {
            "prefill_chunk_size": CHUNK_SIZE,
            "cache_len": CACHE_LEN,
            "pbd_q_len": PBD_QUERY_LEN,
            "ar_q_len": AR_QUERY_LEN,
            "pbd_draft_kv": "discarded",
            "accepted_token_kv": "fused_into_next_mtp_or_ar_bridge",
        },
        "reference_scopes": {
            "official_saved_hybrid": (
                "original checkpoint answer string from generated.jsonl"
            ),
            "adapted_float_hybrid": (
                "fixed prefill/PBD/AR graph-family semantics with quantization disabled"
            ),
            "quantized_eager_hybrid": (
                "same graph-family semantics with Language QDQ enabled"
            ),
            "ground_truth": "profile_target_response from dataset annotation",
        },
        "sequence_comparison_scope": (
            "diagnostic only: the legacy official token list was re-tokenized from "
            "the answer string and stochastic Hybrid outputs are not an exact-token gate"
        ),
        "fused_equivalence_probe": bool(args.fused_equivalence_probe),
        "scale_manifest": str(scale_manifest),
        "scale_manifest_sha256": sha256(scale_manifest),
        "calibration": calibration,
        "weight_policy": language_quantization_policy(),
        "samples_jsonl": str(samples_path),
        "predictions_jsonl": str(predictions_path),
        "started_at": utc_now(),
    }
    atomic_json(report_path, report)
    samples: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    try:
        with samples_path.open("w", encoding="utf-8", newline="\n") as sample_file, predictions_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as prediction_file:
            for index, path in enumerate(payload_paths, 1):
                started = time.monotonic()
                payload = load_payload(path)
                bundle_id = validate_payload_identity(path, payload, manifest)
                payload_config, _ = generation_config_from_payload(
                    payload, generation_metadata, args.max_new_tokens
                )
                if payload_config != generation_config:
                    raise ValueError(
                        f"{bundle_id}: generation_config differs from the first payload"
                    )
                seed, seed_source = official_seed(bundle_id, payload, manifest)
                saved, saved_answer = official_saved_reference(
                    bundle_id, payload, manifest
                )
                float_result = generator.generate(
                    payload,
                    quantized=False,
                    seed=seed,
                    config=generation_config,
                )
                quantized_result = generator.generate(
                    payload,
                    quantized=True,
                    seed=seed,
                    config=generation_config,
                )
                fused_equivalence: dict[str, Any] | None = None
                if args.fused_equivalence_probe:
                    float_prefix = pbd_probe_prefix(float_result)
                    quantized_prefix = pbd_probe_prefix(quantized_result)
                    if float_prefix is None or quantized_prefix is None:
                        raise RuntimeError(
                            f"{bundle_id}: no non-terminal PBD prefix for fused probe"
                        )
                    fused_equivalence = {
                        "adapted_float": fused_prefix_equivalence_probe(
                            generator,
                            payload,
                            float_prefix,
                            quantized=False,
                            seed=seed,
                            config=generation_config,
                        ),
                        "quantized_eager": fused_prefix_equivalence_probe(
                            generator,
                            payload,
                            quantized_prefix,
                            quantized=True,
                            seed=seed,
                            config=generation_config,
                        ),
                    }
                float_tokens = float_result["response_token_ids"]
                quantized_tokens = quantized_result["response_token_ids"]
                prediction = {
                    "bundle_id": bundle_id,
                    "task": payload["task"],
                    "profile_target_response": payload["profile_target_response"],
                    "prediction": {
                        "official_saved_hybrid": {
                            "answer": saved_answer
                        },
                        "adapted_float_hybrid": {
                            "answer": decode_response(tokenizer, float_tokens)
                        },
                        "quantized_eager_hybrid": {
                            "answer": decode_response(tokenizer, quantized_tokens)
                        },
                    },
                }
                comparisons = {
                    "official_roundtrip_to_adapted_float": sequence_comparison(
                        saved, float_tokens
                    ),
                    "adapted_float_to_quantized_eager": sequence_comparison(
                        float_tokens, quantized_tokens
                    ),
                    "official_roundtrip_to_quantized_eager": sequence_comparison(
                        saved, quantized_tokens
                    ),
                }
                sample = {
                    "bundle_id": bundle_id,
                    "task": payload["task"],
                    "payload": str(path),
                    "payload_sha256": sha256(path),
                    "seed": seed,
                    "seed_source": seed_source,
                    "official_roundtrip_token_ids": saved,
                    "official_roundtrip_token_scope": (
                        "manifest answer re-tokenized during calibration generation; "
                        "diagnostic only, not the original sampled token stream"
                    ),
                    "official_saved_answer_source": (
                        "generated.jsonl:prediction.hybrid.answer"
                    ),
                    "official_saved_stop_reason": (
                        "im_end"
                        if saved and saved[-1] == checkpoint_token_ids["im_end_token_id"]
                        else (
                            "max_new_tokens"
                            if len(saved) >= generation_config.max_new_tokens
                            else "unknown"
                        )
                    ),
                    "adapted_float": float_result,
                    "quantized_eager": quantized_result,
                    "fused_equivalence": fused_equivalence,
                    "sequence_comparisons": comparisons,
                    "elapsed_seconds": time.monotonic() - started,
                }
                append_jsonl(sample_file, sample)
                append_jsonl(prediction_file, prediction)
                samples.append(sample)
                prediction_rows.append(prediction)
                report["processed"] = index
                if index % 10 == 0 or index == len(payload_paths):
                    atomic_json(report_path, report)
                print(
                    f"[{index}/{len(payload_paths)}] {bundle_id} "
                    f"{sample['elapsed_seconds']:.2f}s"
                )
                del payload, float_result, quantized_result
                gc.collect()
                torch.cuda.empty_cache()

        accuracy, accuracy_details = evaluate(
            prediction_rows,
            modes=MODES,
        )
        details_path = output_dir / "accuracy_details.jsonl"
        with details_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in accuracy_details:
                append_jsonl(handle, row)
        report["accuracy"] = accuracy
        report["accuracy_details_jsonl"] = str(details_path)
        report["sequence_comparisons"] = {
            key: pairwise_summary(samples, key)
            for key in (
                "official_roundtrip_to_adapted_float",
                "adapted_float_to_quantized_eager",
                "official_roundtrip_to_quantized_eager",
            )
        }
        report["hybrid_control_flow"] = {
            mode: hybrid_control_flow_summary(samples, mode)
            for mode in ("adapted_float", "quantized_eager")
        }
        report["official_saved_stop_reason_counts"] = dict(
            sorted(Counter(sample["official_saved_stop_reason"] for sample in samples).items())
        )
        write_summary_csv(
            csv_path,
            accuracy,
            report["sequence_comparisons"],
            report["hybrid_control_flow"],
        )
    except BaseException as error:
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        atomic_json(report_path, report)
        raise
    finally:
        if runner is not None:
            runner.close()
        del model, api, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    report["status"] = "completed"
    report["finished_at"] = utc_now()
    report["report_csv"] = str(csv_path)
    atomic_json(report_path, report)
    print(f"REPORT: {report_path}")
    print(f"CSV: {csv_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--input_dir", type=Path, required=True)
    root.add_argument("--output_dir", type=Path, required=True)
    root.add_argument("--model_path", type=Path, required=True)
    root.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="generation budget for this validation (default: 2048)",
    )
    selection = root.add_mutually_exclusive_group()
    selection.add_argument(
        "--nums",
        type=int,
        default=None,
        help="validate this many sorted payloads; default validates the full input set",
    )
    selection.add_argument(
        "--one_per_task",
        action="store_true",
        help="validate one deterministic sample from each of the six task domains",
    )
    root.add_argument(
        "--fused_equivalence_probe",
        action="store_true",
        help=(
            "compare legacy causal-prefix commit + q6 against fused q(6+k) "
            "for one accepted prefix per Float and Quantized sample"
        ),
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.output_dir.exists():
        raise FileExistsError(
            f"validation output already exists; choose a new --output_dir: "
            f"{args.output_dir}"
        )
    try:
        return run(args)
    except BaseException as error:
        if args.output_dir.is_dir():
            report_path = args.output_dir / "report.json"
            report: dict[str, Any] = {}
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    report = {}
            report.update(
                {
                    "schema_version": report.get("schema_version", 1),
                    "status": "failed",
                    "input_dir": str(args.input_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "model_path": str(args.model_path.resolve()),
                    "finished_at": utc_now(),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
            atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
