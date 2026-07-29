#!/usr/bin/env python3
"""Analyze completed LocateAnything validation stages in one output directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from compiler.scripts.validate.compare_pipeline import (
    CAPTURE_LEVELS,
    DEFAULT_OUTPUT_DIR,
    QUANTIZED_EAGER_STAGE,
    ModuleMetricAggregator,
    _safe_batch_name,
    atomic_json,
    compare_arrays,
    float_reference_sha256,
    positive_int,
    print_phase_summary,
    progress_bar,
    read_json,
    sample_task,
    select_records,
    utc_now,
    valid_completed_sample,
)


STAGE_ORDER = (
    "float",
    QUANTIZED_EAGER_STAGE,
    "exported_bc",
    "converted_bc",
    "hbm",
)
INTERMEDIATE_STAGES = (QUANTIZED_EAGER_STAGE, "exported_bc", "converted_bc")
LANGUAGE_DECISION_METRICS = (
    "shape",
    "status",
    "cosine",
    "relative_l2",
    "mae",
    "rmse",
    "max_abs",
    "top1_flip_rate",
    "topk",
    "topk_overlap",
    "reference_top1_margin",
    "candidate_top1_margin",
    "reference_top1_rank_in_candidate",
    "candidate_top1_rank_in_reference",
    "decisions",
)


def language_decision_evidence(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep decision-level logits evidence without duplicating KV diagnostics."""

    evidence: list[dict[str, Any]] = []
    for candidate_stage, entries in sample.get("intermediate", {}).items():
        for entry in entries:
            operation = str(entry.get("semantic_operation", ""))
            if operation != "logits" and not operation.startswith("logits.token_"):
                continue
            comparison = entry.get("comparison", {})
            evidence.append({
                "candidate_stage": candidate_stage,
                "module": entry.get("module"),
                "comparison": {
                    key: comparison[key]
                    for key in LANGUAGE_DECISION_METRICS
                    if key in comparison
                },
            })
    return evidence


def coordinate_evidence(sample: dict[str, Any]) -> dict[str, Any] | None:
    audit = sample.get("coordinate_audit")
    return audit if isinstance(audit, dict) else None


def write_domain_csv(
    path: Path,
    aggregate: ModuleMetricAggregator,
    domains: dict[str, ModuleMetricAggregator],
) -> None:
    columns = [
        "domain", "stage", "reference_sequence", "semantic_group",
        "semantic_operation", "module", "shape", "samples", "matched", "match_rate",
    ]
    for metric in ModuleMetricAggregator.METRICS:
        columns.extend(
            f"{metric}_{suffix}"
            for suffix in ("count", "mean", "min", "p05", "median", "p95", "max")
        )
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = [("all", aggregate), *sorted(domains.items())]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for domain, aggregator in grouped:
            for record in aggregator.records():
                row = {
                    key: record[key]
                    for key in (
                        "stage", "reference_sequence", "semantic_group",
                        "semantic_operation", "module", "samples", "matched",
                    )
                }
                row["domain"] = domain
                row["shape"] = "x".join(str(value) for value in record["shape"])
                row["match_rate"] = record["match_rate"]
                for metric, statistics in record["metrics"].items():
                    row.update(
                        {
                            f"{metric}_{name}": value
                            for name, value in statistics.items()
                        }
                    )
                writer.writerow(row)
    os.replace(temporary, path)


def selection_metadata(
    phase: str,
    nums: int | None,
    records: list[dict[str, Any]],
    coverage: dict[str, Any],
    stage_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_ids = [str(record["id"]) for record in records]
    digest = hashlib.sha256(
        json.dumps(sorted(selected_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "phase": phase,
        "requested_nums": nums if nums is not None else "all",
        "selected_ids": selected_ids,
        "selected_ids_sha256": digest,
        "coverage": coverage,
        "stage_levels": {
            stage: summary.get("level") for stage, summary in stage_summaries.items()
        },
    }


def load_completed_stages(
    output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    completed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for stage in STAGE_ORDER:
        state_path = output_dir / stage / "stage.json"
        if not state_path.is_file():
            continue
        summary = read_json(state_path)
        status = str(summary.get("status", "unknown"))
        if status != "completed":
            skipped[stage] = status
            continue
        completed[stage] = summary
    if "float" not in completed:
        raise FileNotFoundError("analysis requires a completed float stage")
    if len(completed) < 2:
        raise FileNotFoundError("analysis requires at least one completed candidate stage")
    return completed, skipped


def completed_output_path(
    output_dir: Path,
    stage: str,
    record: dict[str, Any],
    phase: str,
    capture_level: str,
) -> Path:
    sample_path = output_dir / stage / "samples" / f"{_safe_batch_name(record['id'])}.json"
    if not sample_path.is_file():
        raise FileNotFoundError(f"missing {stage} sample metadata: {sample_path}")
    sample = read_json(sample_path)
    output = sample.get("output")
    if not isinstance(output, str):
        safe_id = _safe_batch_name(str(record["id"]))
        candidates = [
            output_dir / stage / "outputs" / f"{safe_id}.npy",
            output_dir / stage / "outputs" / f"{safe_id}.npz",
        ]
        output_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = output_dir / stage / "outputs" / output_path
    if valid_completed_sample(
        sample_path,
        output_path,
        str(record["id"]),
        input_sha256=record.get("sha256"),
        phase=phase,
        capture_level=capture_level,
    ) is None:
        raise ValueError(f"invalid {stage} output for {record['id']}")
    return output_path


def load_named_outputs(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npy":
        return {"output": np.load(path, mmap_mode="r", allow_pickle=False)}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    raise ValueError(f"unsupported output format: {path}")


def final_output_rows(
    reference_stage: str,
    candidate_stage: str,
    reference_outputs: dict[str, np.ndarray],
    candidate_outputs: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_names = [name for name in reference_outputs if name in candidate_outputs]
    if not common_names:
        raise ValueError(f"no common outputs for {reference_stage}->{candidate_stage}")
    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for offset, name in enumerate(common_names):
        reference = reference_outputs[name]
        candidate = candidate_outputs[name]
        comparison = compare_arrays(reference, candidate)
        comparisons.append(
            {
                "name": name,
                "reference": reference_stage,
                "candidate": candidate_stage,
                "comparison": comparison,
            }
        )
        rows.append(
            {
                "reference_sequence": 10_000 + offset,
                "semantic_group": "FINAL OUTPUT",
                "semantic_operation": name,
                "module": name,
                "shape": list(reference.shape),
                "status": "matched",
                "comparison": comparison,
            }
        )
    return comparisons, rows


def run_analysis_collection(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    input_index = read_json(output_dir / "inputs.json")
    all_records = input_index["inputs"]
    stage_summaries, skipped_stages = load_completed_stages(output_dir)

    phases = {summary.get("phase") for summary in stage_summaries.values()}
    if None in phases or len(phases) != 1:
        raise ValueError(f"completed stages do not share one phase: {sorted(map(str, phases))}")
    phase = str(next(iter(phases)))
    for stage, summary in stage_summaries.items():
        if not isinstance(summary.get("selected_ids"), list):
            raise ValueError(f"stage {stage} is missing selected_ids metadata")

    fingerprints = {
        summary.get("input_set_sha256") for summary in stage_summaries.values()
    }
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError("stage input sets do not match")
    records_by_id = {str(record["id"]): record for record in all_records}
    for stage in INTERMEDIATE_STAGES:
        if stage not in stage_summaries:
            continue
        try:
            stage_records = [
                records_by_id[sample_id]
                for sample_id in stage_summaries[stage]["selected_ids"]
            ]
        except KeyError as error:
            raise ValueError(f"stage {stage} refers to an unknown input id") from error
        recorded_reference = stage_summaries[stage].get("float_reference_sha256")
        expected_reference = float_reference_sha256(
            output_dir, stage_summaries["float"], stage_records
        )
        if recorded_reference is not None and recorded_reference != expected_reference:
            raise ValueError(f"stage {stage} does not match the current Float reference")

    selected_sets = [set(summary["selected_ids"]) for summary in stage_summaries.values()]
    common_ids = set.intersection(*selected_sets)
    common_records = [record for record in all_records if record["id"] in common_ids]
    records, coverage = select_records(common_records, args.nums)
    metadata = selection_metadata(
        phase, args.nums, records, coverage, stage_summaries
    )
    selected_ids = set(metadata["selected_ids"])

    for record in records:
        for stage, summary in stage_summaries.items():
            capture_level = CAPTURE_LEVELS.get(str(summary.get("level")))
            if capture_level is None:
                raise ValueError(f"stage {stage} has an invalid level")
            completed_output_path(output_dir, stage, record, phase, capture_level)

    details_dir = output_dir / "samples"
    details_dir.mkdir(parents=True, exist_ok=True)
    aggregator = ModuleMetricAggregator()
    domain_aggregators: dict[str, ModuleMetricAggregator] = {}
    domain_counts: dict[str, int] = {}
    language_decisions: dict[str, list[dict[str, Any]]] = {}
    coordinate_decisions: dict[str, dict[str, Any]] = {}
    for record in records:
        domain = sample_task(record)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        domain_aggregators.setdefault(domain, ModuleMetricAggregator())
    for stage_name in INTERMEDIATE_STAGES:
        if stage_name not in stage_summaries:
            continue
        stage_samples = output_dir / stage_name / "samples"
        for record in records:
            if record["id"] in selected_ids:
                path = stage_samples / f"{_safe_batch_name(record['id'])}.json"
                sample = read_json(path)
                aggregator.restore_sample(sample)
                domain_aggregators[sample_task(record)].restore_sample(sample)
                if phase == "language":
                    language_decisions.setdefault(str(record["id"]), []).extend(
                        language_decision_evidence(sample)
                    )
                    audit = coordinate_evidence(sample)
                    if audit is not None:
                        coordinate_decisions[str(record["id"])] = audit

    ordered_stages = [stage for stage in STAGE_ORDER if stage in stage_summaries]
    transitions = list(zip(ordered_stages, ordered_stages[1:]))
    if "hbm" in stage_summaries and ("float", "hbm") not in transitions:
        transitions.append(("float", "hbm"))
    completed = 0
    phase_started = time.monotonic()
    progress = progress_bar(records, "[1/1] Analysis")
    for index, record in enumerate(progress, start=1):
        sample_id = record["id"]
        safe_id = _safe_batch_name(sample_id)
        comparisons: list[dict[str, Any]] = []
        for reference_stage, candidate_stage in transitions:
            reference_level = CAPTURE_LEVELS[str(stage_summaries[reference_stage]["level"])]
            candidate_level = CAPTURE_LEVELS[str(stage_summaries[candidate_stage]["level"])]
            reference_path = completed_output_path(
                output_dir, reference_stage, record, phase, reference_level
            )
            candidate_path = completed_output_path(
                output_dir, candidate_stage, record, phase, candidate_level
            )
            reference_outputs = load_named_outputs(reference_path)
            candidate_outputs = load_named_outputs(candidate_path)
            transition = f"{reference_stage}_to_{candidate_stage}"
            transition_comparisons, rows = final_output_rows(
                reference_stage, candidate_stage, reference_outputs, candidate_outputs
            )
            comparisons.extend(transition_comparisons)
            aggregator.add(transition, rows)
            domain_aggregators[sample_task(record)].add(transition, rows)
            del reference_outputs, candidate_outputs, transition_comparisons, rows
        detail = {
            "schema_version": 2,
            "status": "completed",
            "index": index - 1,
            "id": sample_id,
            "task": sample_task(record),
            "input": record,
            "final_outputs": comparisons,
            "finished_at": utc_now(),
        }
        if phase == "language":
            detail["language_decisions"] = language_decisions.get(str(sample_id), [])
            if str(sample_id) in coordinate_decisions:
                detail["coordinate_audit"] = coordinate_decisions[str(sample_id)]
        atomic_json(details_dir / f"{safe_id}.json", detail)
        completed += 1
        if completed % 10 == 0:
            write_domain_csv(output_dir / "report.csv", aggregator, domain_aggregators)
        cosines = [
            float(value)
            for item in comparisons
            if isinstance((value := item["comparison"].get("cosine")), (int, float))
        ]
        progress.set_postfix(
            sample=sample_id,
            min_cosine=f"{min(cosines, default=float('nan')):.6f}",
        )
    progress.close()
    write_domain_csv(output_dir / "report.csv", aggregator, domain_aggregators)
    module_records = aggregator.records()
    domain_reports = {
        domain: {
            "sample_count": domain_counts[domain],
            "modules": domain_aggregators[domain].records(),
        }
        for domain in sorted(domain_aggregators)
    }
    coordinate_modules = [
        record
        for record in module_records
        if record.get("semantic_group") == "COORDINATE OUTPUT"
    ]
    coordinate_domains = {
        domain: [
            record
            for record in report["modules"]
            if record.get("semantic_group") == "COORDINATE OUTPUT"
        ]
        for domain, report in domain_reports.items()
    }
    atomic_json(
        output_dir / "report.json",
        {
            "schema_version": 2,
            "status": "completed",
            **metadata,
            "input_count": len(all_records),
            "common_count": len(common_records),
            "selected_count": len(records),
            "completed": completed,
            "inputs": str((output_dir / "inputs.json").resolve()),
            "samples": str(details_dir.resolve()),
            "csv": str((output_dir / "report.csv").resolve()),
            "stages": stage_summaries,
            "skipped_stages": skipped_stages,
            "modules": module_records,
            "coordinates": {
                "sample_count": len(coordinate_decisions),
                "decision_count": sum(
                    int(audit.get("decision_count", 0))
                    for audit in coordinate_decisions.values()
                ),
                "modules": coordinate_modules,
                "domains": coordinate_domains,
            },
            "domains": domain_reports,
            "updated_at": utc_now(),
        },
    )
    print_phase_summary(
        "Analysis",
        len(records),
        completed,
        0,
        phase_started,
        output_dir,
        report=output_dir / "report.json",
        csv=output_dir / "report.csv",
    )
    print(f"[analysis] report={output_dir / 'report.json'}", flush=True)
    print(f"[analysis] csv={output_dir / 'report.csv'}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Analyze completed LocateAnything validation stages."
    )
    result.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory containing Float, Quantized Eager, BC, or HBM stage outputs",
    )
    result.add_argument(
        "--nums",
        type=positive_int,
        help="exact number of common inputs to analyze; omitted means all",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    return run_analysis_collection(parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
