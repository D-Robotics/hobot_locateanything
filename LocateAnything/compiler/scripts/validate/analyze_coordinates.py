#!/usr/bin/env python3
"""Summarize LocateAnything coordinate diagnostics without rerunning the model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


IOU_THRESHOLDS = (0.50, 0.75, 0.90, 0.95)
PIXEL_THRESHOLDS = (2.0, 5.0, 10.0, 50.0)
PAIRWISE_TOLERANCE = 0.01
WORST_CASE_COUNT = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def statistics(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, float | int] = {
        "count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
    }
    if finite:
        result.update(
            mean=sum(finite) / len(finite),
            min=min(finite),
            p05=percentile(finite, 0.05),
            median=percentile(finite, 0.50),
            p95=percentile(finite, 0.95),
            max=max(finite),
        )
    return result


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def coordinate_valid(decoded: dict[str, Any]) -> bool:
    expected = {"box": 4, "point": 2}.get(str(decoded.get("type")))
    return expected is not None and len(decoded.get("coordinate_values", [])) == expected


def decision_record(sample_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    resolved = decision.get("resolved", {})
    float_output = resolved.get("float", decision["float"])
    quantized_output = resolved.get("quantized_eager", decision["quantized_eager"])
    comparison = decision.get("resolved_comparison", decision["comparison"])
    position_rows = decision.get("position_diagnostics", [])
    misses = [
        row
        for row in position_rows
        if row.get("comparison", {}).get("float_token_top4_hit") == 0.0
    ]
    ranks = [
        float(row["comparison"]["float_token_rank_in_quantized"])
        for row in position_rows
        if "float_token_rank_in_quantized" in row.get("comparison", {})
    ]
    source = decision["source"]
    return {
        "sample_id": sample_id,
        "decision_index": int(decision["index"]),
        "source_kind": str(source["kind"]),
        "source_coordinates": source.get("coordinate_values", []),
        "float_type": str(float_output.get("type")),
        "float_coordinates": float_output.get("coordinate_values", []),
        "quantized_type": str(quantized_output.get("type")),
        "quantized_coordinates": quantized_output.get("coordinate_values", []),
        "float_coordinate_valid": coordinate_valid(float_output),
        "quantized_coordinate_valid": coordinate_valid(quantized_output),
        "structure_agreement": bool(comparison.get("structure_agreement", 0.0)),
        "coordinate_token_exact": comparison.get("coordinate_token_exact"),
        "pixel_mae": comparison.get("pixel_mae"),
        "pixel_max_abs": comparison.get("pixel_max_abs"),
        "box_iou": comparison.get("box_iou"),
        "point_distance_pixels": comparison.get("point_distance_pixels"),
        "float_ar_fallback": decision.get("ar_q1", {}).get("float") is not None,
        "quantized_ar_fallback": (
            decision.get("ar_q1", {}).get("quantized_eager") is not None
        ),
        "pbd_position_count": len(position_rows),
        "pbd_top4_miss_count": len(misses),
        "pbd_max_float_token_rank": max(ranks) if ranks else None,
        "pbd_top4_miss_positions": [int(row["position"]) for row in misses],
    }


def sample_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [row for row in records if row["source_kind"] == "box"]
    comparable = [row for row in boxes if row["box_iou"] is not None]
    ious = [float(row["box_iou"]) for row in comparable]
    pixel_errors = [
        float(row["pixel_max_abs"])
        for row in comparable
        if row["pixel_max_abs"] is not None
    ]
    return {
        "decision_count": len(records),
        "box_count": len(boxes),
        "comparable_box_count": len(comparable),
        "box_iou_mean": sum(ious) / len(ious) if ious else None,
        "box_iou_min": min(ious) if ious else None,
        "pixel_max_abs": max(pixel_errors) if pixel_errors else None,
        "coordinate_exact_rate": rate(
            sum(row["coordinate_token_exact"] == 1.0 for row in comparable),
            len(comparable),
        ),
        "structure_mismatch_count": sum(not row["structure_agreement"] for row in records),
        "pbd_top4_miss_count": sum(row["pbd_top4_miss_count"] for row in records),
        "float_ar_fallback_count": sum(row["float_ar_fallback"] for row in records),
        "quantized_ar_fallback_count": sum(
            row["quantized_ar_fallback"] for row in records
        ),
    }


def geometry_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [row for row in records if row["source_kind"] == "box"]
    comparable = [row for row in boxes if row["box_iou"] is not None]
    ious = [float(row["box_iou"]) for row in comparable]
    pixel_max = [float(row["pixel_max_abs"]) for row in comparable]
    summary: dict[str, Any] = {
        "box_count": len(boxes),
        "comparable_box_count": len(comparable),
        "coordinate_exact_rate": rate(
            sum(row["coordinate_token_exact"] == 1.0 for row in comparable),
            len(comparable),
        ),
        "box_iou": statistics(ious),
        "pixel_max_abs": statistics(pixel_max),
    }
    for threshold in IOU_THRESHOLDS:
        count = sum(value < threshold for value in ious)
        label = str(threshold).replace(".", "_")
        summary[f"box_iou_lt_{label}_count"] = count
        summary[f"box_iou_lt_{label}_rate"] = rate(count, len(ious))
    return summary


def experiment_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [row for row in records if row["source_kind"] == "box"]
    comparable = [row for row in boxes if row["box_iou"] is not None]
    ious = [float(row["box_iou"]) for row in comparable]
    pixel_mae = [float(row["pixel_mae"]) for row in comparable]
    pixel_max = [float(row["pixel_max_abs"]) for row in comparable]
    positions = sum(row["pbd_position_count"] for row in records)
    misses = sum(row["pbd_top4_miss_count"] for row in records)
    summary: dict[str, Any] = {
        "decision_count": len(records),
        "box_count": len(boxes),
        "comparable_box_count": len(comparable),
        "float_coordinate_valid_rate": rate(
            sum(row["float_coordinate_valid"] for row in records), len(records)
        ),
        "quantized_coordinate_valid_rate": rate(
            sum(row["quantized_coordinate_valid"] for row in records), len(records)
        ),
        "structure_agreement_rate": rate(
            sum(row["structure_agreement"] for row in records), len(records)
        ),
        "coordinate_exact_rate": rate(
            sum(row["coordinate_token_exact"] == 1.0 for row in comparable),
            len(comparable),
        ),
        "box_iou": statistics(ious),
        "pixel_mae": statistics(pixel_mae),
        "pixel_max_abs": statistics(pixel_max),
        "pbd_position_count": positions,
        "pbd_top4_miss_count": misses,
        "pbd_top4_miss_rate": rate(misses, positions),
        "pbd_decisions_with_top4_miss": sum(
            row["pbd_top4_miss_count"] > 0 for row in records
        ),
        "float_ar_fallback_count": sum(row["float_ar_fallback"] for row in records),
        "quantized_ar_fallback_count": sum(
            row["quantized_ar_fallback"] for row in records
        ),
        "decode_paths": {
            "no_ar": geometry_summary(
                [
                    row
                    for row in records
                    if not row["float_ar_fallback"]
                    and not row["quantized_ar_fallback"]
                ]
            ),
            "float_only_ar": geometry_summary(
                [
                    row
                    for row in records
                    if row["float_ar_fallback"]
                    and not row["quantized_ar_fallback"]
                ]
            ),
            "quantized_only_ar": geometry_summary(
                [
                    row
                    for row in records
                    if not row["float_ar_fallback"]
                    and row["quantized_ar_fallback"]
                ]
            ),
            "both_ar": geometry_summary(
                [
                    row
                    for row in records
                    if row["float_ar_fallback"]
                    and row["quantized_ar_fallback"]
                ]
            ),
        },
        "pbd_top4_groups": {
            "no_miss": geometry_summary(
                [row for row in records if row["pbd_top4_miss_count"] == 0]
            ),
            "has_miss": geometry_summary(
                [row for row in records if row["pbd_top4_miss_count"] > 0]
            ),
        },
    }
    for threshold in IOU_THRESHOLDS:
        count = sum(value < threshold for value in ious)
        label = str(threshold).replace(".", "_")
        summary[f"box_iou_lt_{label}_count"] = count
        summary[f"box_iou_lt_{label}_rate"] = rate(count, len(ious))
    for threshold in PIXEL_THRESHOLDS:
        count = sum(value > threshold for value in pixel_max)
        label = str(threshold).replace(".", "_")
        summary[f"pixel_max_gt_{label}_count"] = count
        summary[f"pixel_max_gt_{label}_rate"] = rate(count, len(pixel_max))
    return summary


def pairwise_summary(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(candidate))
    pairs = [
        (baseline[key], candidate[key])
        for key in keys
        if baseline[key]["source_kind"] == "box"
    ]
    comparable = [
        (left, right)
        for left, right in pairs
        if left["box_iou"] is not None and right["box_iou"] is not None
    ]
    deltas = [float(right["box_iou"]) - float(left["box_iou"]) for left, right in comparable]
    improved = sum(delta > PAIRWISE_TOLERANCE for delta in deltas)
    regressed = sum(delta < -PAIRWISE_TOLERANCE for delta in deltas)
    return {
        "matched_box_count": len(pairs),
        "comparable_box_count": len(comparable),
        "box_iou_delta": statistics(deltas),
        "box_iou_improved_gt_0_01_count": improved,
        "box_iou_improved_gt_0_01_rate": rate(improved, len(deltas)),
        "box_iou_regressed_gt_0_01_count": regressed,
        "box_iou_regressed_gt_0_01_rate": rate(regressed, len(deltas)),
        "box_iou_crossed_up_0_90_count": sum(
            float(left["box_iou"]) < 0.90 <= float(right["box_iou"])
            for left, right in comparable
        ),
        "box_iou_crossed_down_0_90_count": sum(
            float(right["box_iou"]) < 0.90 <= float(left["box_iou"])
            for left, right in comparable
        ),
        "coordinate_exact_gain_count": sum(
            left["coordinate_token_exact"] != 1.0
            and right["coordinate_token_exact"] == 1.0
            for left, right in comparable
        ),
        "coordinate_exact_loss_count": sum(
            left["coordinate_token_exact"] == 1.0
            and right["coordinate_token_exact"] != 1.0
            for left, right in comparable
        ),
        "quantized_valid_gain_count": sum(
            not left["quantized_coordinate_valid"]
            and right["quantized_coordinate_valid"]
            for left, right in pairs
        ),
        "quantized_valid_loss_count": sum(
            left["quantized_coordinate_valid"]
            and not right["quantized_coordinate_valid"]
            for left, right in pairs
        ),
        "pbd_top4_miss_delta": sum(
            right["pbd_top4_miss_count"] - left["pbd_top4_miss_count"]
            for left, right in pairs
        ),
    }


def worst_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes = [row for row in records if row["source_kind"] == "box"]
    return sorted(
        boxes,
        key=lambda row: (
            row["box_iou"] is not None,
            float(row["box_iou"]) if row["box_iou"] is not None else -1.0,
            -float(row["pixel_max_abs"] or 0.0),
        ),
    )[:WORST_CASE_COUNT]


def flatten_scalars(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            rows.extend(flatten_scalars(child_prefix, child))
        return rows
    if isinstance(value, (int, float)) or value is None:
        return [(prefix, value)]
    return []


def write_summary_csv(path: Path, analysis: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("experiment", "scope", "sample_id", "metric", "value"),
        )
        writer.writeheader()
        for name, experiment in analysis["experiments"].items():
            for metric, value in flatten_scalars("", experiment["summary"]):
                writer.writerow(
                    {
                        "experiment": name,
                        "scope": "experiment",
                        "sample_id": "",
                        "metric": metric,
                        "value": value,
                    }
                )
            for sample_id, summary in experiment["samples"].items():
                for metric, value in flatten_scalars("", summary):
                    writer.writerow(
                        {
                            "experiment": name,
                            "scope": "sample",
                            "sample_id": sample_id,
                            "metric": metric,
                            "value": value,
                        }
                    )
        for name, comparison in analysis["comparisons_to_baseline"].items():
            for metric, value in flatten_scalars("", comparison):
                writer.writerow(
                    {
                        "experiment": name,
                        "scope": "vs_baseline",
                        "sample_id": "",
                        "metric": metric,
                        "value": value,
                    }
                )
    temporary.replace(path)


def write_worst_csv(path: Path, analysis: dict[str, Any]) -> None:
    columns = (
        "experiment", "rank", "sample_id", "decision_index", "source_kind",
        "source_coordinates", "float_type", "float_coordinates", "quantized_type",
        "quantized_coordinates", "box_iou", "pixel_mae", "pixel_max_abs",
        "structure_agreement", "pbd_top4_miss_count", "pbd_max_float_token_rank",
        "float_ar_fallback", "quantized_ar_fallback",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, experiment in analysis["experiments"].items():
            for rank, row in enumerate(experiment["worst_cases"], 1):
                writer.writerow(
                    {
                        "experiment": name,
                        "rank": rank,
                        **{
                            column: json.dumps(row[column], ensure_ascii=True)
                            if isinstance(row.get(column), list)
                            else row.get(column)
                            for column in columns
                            if column not in {"experiment", "rank"}
                        },
                    }
                )
    temporary.replace(path)


def analyze(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    experiments: dict[str, Any] = {}
    record_maps: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for experiment in report.get("experiments", []):
        name = str(experiment["name"])
        records = [
            decision_record(str(sample["id"]), decision)
            for sample in experiment.get("samples", [])
            for decision in sample["coordinate_audit"].get("decisions", [])
        ]
        by_sample: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            by_sample.setdefault(str(row["sample_id"]), []).append(row)
        experiments[name] = {
            "summary": experiment_summary(records),
            "samples": {
                sample_id: sample_summary(rows)
                for sample_id, rows in sorted(by_sample.items())
            },
            "worst_cases": worst_cases(records),
        }
        record_maps[name] = {
            (str(row["sample_id"]), int(row["decision_index"])): row for row in records
        }
    baseline_name = next(iter(experiments), None)
    comparisons = (
        {
            name: pairwise_summary(record_maps[baseline_name], records)
            for name, records in record_maps.items()
            if name != baseline_name
        }
        if baseline_name is not None
        else {}
    )
    return {
        "schema_version": 1,
        "source_report": str(report_path.resolve()),
        "source_report_sha256": sha256(report_path),
        "source_status": report.get("status"),
        "source_suite": report.get("suite"),
        "baseline_experiment": baseline_name,
        "reference_contract": (
            "Float and quantized outputs are compared against the saved upstream "
            "hybrid prediction, not dataset ground-truth annotations."
        ),
        "experiments": experiments,
        "comparisons_to_baseline": comparisons,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("report", type=Path, help="coordinate diagnostic report.json")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report_path = args.report.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    analysis = analyze(report_path)
    output_dir = report_path.parent
    json_path = output_dir / "coordinate_analysis.json"
    summary_csv = output_dir / "coordinate_analysis.csv"
    worst_csv = output_dir / "coordinate_worst_cases.csv"
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    write_summary_csv(summary_csv, analysis)
    write_worst_csv(worst_csv, analysis)
    print(f"ANALYSIS: {json_path}")
    print(f"CSV: {summary_csv}")
    print(f"WORST CASES: {worst_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
