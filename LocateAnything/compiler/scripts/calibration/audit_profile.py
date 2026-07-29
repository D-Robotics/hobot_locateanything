#!/usr/bin/env python3
"""Audit LocateAnything Language calibration inputs against a fixed profile."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = REPO_ROOT / "compiler"
sys.path.insert(0, str(COMPILER_ROOT))

REPLAY_PATH = (
    COMPILER_ROOT / "leap_llm" / "apis" / "calibration" / "locateanything_replay.py"
)
REPLAY_SPEC = importlib.util.spec_from_file_location("locateanything_replay", REPLAY_PATH)
if REPLAY_SPEC is None or REPLAY_SPEC.loader is None:
    raise ImportError(f"cannot load replay helpers from {REPLAY_PATH}")
REPLAY = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(REPLAY)
load_tensor_payload = REPLAY.load_tensor_payload
read_generated_manifest = REPLAY.read_generated_manifest
sha256_file = REPLAY.sha256_file


def progress(items: Iterable[Any], *, total: int) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(items, total=total, desc="Language profile audit", unit="sample")
    except ImportError:
        return items


def percentile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty group")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    if args.chunk_size <= 0 or args.cache_len <= 0:
        raise ValueError("chunk_size and cache_len must be positive")
    if args.chunk_size > args.cache_len:
        raise ValueError("chunk_size cannot exceed cache_len")

    manifest = args.generated_jsonl.resolve()
    records = read_generated_manifest(manifest)
    output_dir = args.output_dir.resolve()
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    print("\n================== LANGUAGE PROFILE AUDIT ==================", flush=True)
    for index, record in enumerate(progress(records, total=len(records)), 1):
        payload = load_tensor_payload(record)
        input_ids = payload["prompt_input_ids"].reshape(-1).to(torch.long)
        attention_mask = payload["prompt_attention_mask"].reshape(-1)
        prompt_len = int(attention_mask.sum().item())
        image_tokens = int((input_ids == args.image_token_id).sum().item())
        projected = payload["projected_visual_features"]
        projected_tokens = int(projected.numel() // projected.shape[-1])
        task = str(record.get("task", "unknown"))
        bundle_id = str(record.get("bundle_id", Path(record["tensor_file"]).stem))

        checks = {
            "unpadded_prompt": prompt_len == input_ids.numel(),
            "prefill_fits": prompt_len <= args.chunk_size,
            "image_placeholders_match_features": image_tokens == projected_tokens,
            "expected_visual_tokens": image_tokens == args.visual_tokens,
            "pbd_fits_cache": prompt_len + args.pbd_query_len <= args.cache_len,
            "ar_fits_cache": prompt_len + args.ar_query_len <= args.cache_len,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            violations.append({
                "index": index,
                "bundle_id": bundle_id,
                "task": task,
                "failed_checks": failed,
            })

        prompt_lengths[task].append(prompt_len)
        sample_rows.append({
            "index": index,
            "bundle_id": bundle_id,
            "task": task,
            "prompt_tokens": prompt_len,
            "visual_tokens": image_tokens,
            "prefill_headroom": args.chunk_size - prompt_len,
            "cache_headroom_after_pbd": args.cache_len - prompt_len - args.pbd_query_len,
            "cache_headroom_after_ar": args.cache_len - prompt_len - args.ar_query_len,
            "passed": not failed,
        })

    all_lengths = [row["prompt_tokens"] for row in sample_rows]
    summaries: list[dict[str, Any]] = []
    for task, values in [("all", all_lengths), *sorted(prompt_lengths.items())]:
        stats = summarize(values)
        summaries.append({
            "task": task,
            **stats,
            "prefill_headroom_min": args.chunk_size - stats["max"],
            "cache_headroom_after_pbd_min": (
                args.cache_len - stats["max"] - args.pbd_query_len
            ),
        })

    report = {
        "schema_version": 1,
        "passed": not violations,
        "profile": {
            "batch_size": 1,
            "chunk_size": args.chunk_size,
            "cache_len": args.cache_len,
            "pbd_query_len": args.pbd_query_len,
            "ar_query_len": args.ar_query_len,
            "visual_tokens": args.visual_tokens,
            "use_sliding_window": False,
            "max_position_embeddings": 32768,
            "rope_theta": 1_000_000.0,
            "rope_scaling": None,
        },
        "semantics": {
            "prefill": "prompt_tokens must fit chunk_size",
            "pbd": "q=6 uses a separate decode graph; it consumes cache capacity, not prefill width",
            "ar": "q=1 uses a separate decode graph; it consumes cache capacity, not prefill width",
        },
        "generated_manifest": str(manifest),
        "generated_manifest_sha256": sha256_file(manifest),
        "sample_count": len(sample_rows),
        "task_counts": dict(sorted(Counter(row["task"] for row in sample_rows).items())),
        "summary": summaries,
        "violations": violations,
        "samples": sample_rows,
    }
    atomic_json(output_dir / "language_profile_audit.json", report)
    atomic_csv(output_dir / "language_profile_summary.csv", summaries)

    print("\n================== LANGUAGE PROFILE AUDIT COMPLETED ==================", flush=True)
    print(f"SAMPLES: {len(sample_rows)}", flush=True)
    print(f"VIOLATIONS: {len(violations)}", flush=True)
    print(f"PROMPT_TOKENS_MAX: {max(all_lengths)}", flush=True)
    print(f"PREFILL_HEADROOM_MIN: {args.chunk_size - max(all_lengths)}", flush=True)
    print(
        "CACHE_HEADROOM_AFTER_PBD_MIN: "
        f"{args.cache_len - max(all_lengths) - args.pbd_query_len}",
        flush=True,
    )
    print(f"REPORT: {output_dir / 'language_profile_audit.json'}", flush=True)
    print(f"SUMMARY: {output_dir / 'language_profile_summary.csv'}", flush=True)
    if violations:
        raise RuntimeError(f"language profile audit failed for {len(violations)} samples")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--generated-jsonl", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--chunk-size", type=int, default=1024)
    result.add_argument("--cache-len", type=int, default=4096)
    result.add_argument("--pbd-query-len", type=int, default=6)
    result.add_argument("--ar-query-len", type=int, default=1)
    result.add_argument("--image-token-id", type=int, default=151665)
    result.add_argument("--visual-tokens", type=int, default=576)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
