#!/usr/bin/env python3
"""Evaluate LocateAnything grounding responses from prepared Float or S600 JSONL.

The evaluator uses only the Python standard library.  Coordinates are expected
on LocateAnything's normalized 0..1000 grid.  A malformed prediction is kept in
the denominator and receives no grounding matches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TASKS = ("detection", "gui", "referring", "ocr", "layout", "pointing")
REF_RE = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
GEOMETRY_RE = re.compile(r"<box>((?:<-?\d+>){2}|(?:<-?\d+>){4})</box>")
COORD_RE = re.compile(r"<(-?\d+)>")
DEFAULT_PCK_THRESHOLDS = (0.05, 0.10)
GRID_MAX = 1000.0
GRID_DIAGONAL = math.sqrt(2.0) * GRID_MAX
# LocateAnything's tokenizer declares im_end as EOS. endoftext is PAD, and
# </s> is not the model EOS, so neither is silently accepted here.
ALLOWED_TERMINAL_TOKENS = ("<|im_end|>",)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            records.append(value)
    return records


def normalize_label(value: str) -> str:
    """Normalize presentation differences without erasing label semantics."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


@dataclass(frozen=True)
class Geometry:
    label: str
    coordinates: tuple[int, ...]

    @property
    def kind(self) -> str:
        return "point" if len(self.coordinates) == 2 else "box"


@dataclass(frozen=True)
class ParsedResponse:
    valid: bool
    refs: tuple[str, ...]
    geometries: tuple[Geometry, ...]
    error: str | None = None
    terminal_token: str | None = None


def parse_response(response: Any) -> ParsedResponse:
    if not isinstance(response, str) or not response.strip():
        return ParsedResponse(False, (), (), "response is empty or not a string")

    position = 0
    refs: list[str] = []
    geometries: list[Geometry] = []
    text = response.strip()
    terminal_token = None
    for candidate in ALLOWED_TERMINAL_TOKENS:
        if text.endswith(candidate):
            text = text[: -len(candidate)].rstrip()
            terminal_token = candidate
            break
    if not text:
        return ParsedResponse(False, (), (), "response has no grounding content", terminal_token)
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        ref_match = REF_RE.match(text, position)
        if ref_match is None:
            return ParsedResponse(
                False, tuple(refs), tuple(geometries), f"expected <ref> at offset {position}", terminal_token
            )
        raw_label = ref_match.group(1)
        label = normalize_label(raw_label)
        # OCR labels can legitimately contain comparison signs (for example
        # ``<ref><30</ref>``). The closing tag remains an unambiguous boundary.
        if not label:
            return ParsedResponse(False, tuple(refs), tuple(geometries), "empty ref label", terminal_token)
        refs.append(label)
        position = ref_match.end()

        geometry_count = 0
        while position < len(text):
            geometry_match = GEOMETRY_RE.match(text, position)
            if geometry_match is None:
                break
            coordinates = tuple(int(value) for value in COORD_RE.findall(geometry_match.group(1)))
            if any(value < 0 or value > 1000 for value in coordinates):
                return ParsedResponse(
                    False, tuple(refs), tuple(geometries), "coordinate outside 0..1000", terminal_token
                )
            if len(coordinates) == 4 and (
                coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]
            ):
                return ParsedResponse(False, tuple(refs), tuple(geometries), "degenerate box", terminal_token)
            geometries.append(Geometry(label, coordinates))
            geometry_count += 1
            position = geometry_match.end()
            while position < len(text) and text[position].isspace():
                position += 1
        if geometry_count == 0:
            return ParsedResponse(
                False, tuple(refs), tuple(geometries), "ref has no following geometry", terminal_token
            )

    return ParsedResponse(True, tuple(refs), tuple(geometries), terminal_token=terminal_token)


def box_iou(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def point_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def maximum_threshold_matching(
    predicted: list[Geometry],
    target: list[Geometry],
    score,
    threshold: float,
    higher_is_better: bool,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality label-aware bipartite matching."""
    adjacency: list[list[tuple[int, float]]] = []
    for prediction in predicted:
        candidates = []
        for target_index, expected in enumerate(target):
            if prediction.label != expected.label:
                continue
            value = score(prediction.coordinates, expected.coordinates)
            passes = value >= threshold if higher_is_better else value <= threshold
            if passes:
                candidates.append((target_index, value))
        candidates.sort(key=lambda item: item[1], reverse=higher_is_better)
        adjacency.append(candidates)

    target_to_prediction: dict[int, tuple[int, float]] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for target_index, value in adjacency[prediction_index]:
            if target_index in visited:
                continue
            visited.add(target_index)
            previous = target_to_prediction.get(target_index)
            if previous is None or augment(previous[0], visited):
                target_to_prediction[target_index] = (prediction_index, value)
                return True
        return False

    for prediction_index in range(len(predicted)):
        augment(prediction_index, set())
    return [
        (prediction_index, target_index, value)
        for target_index, (prediction_index, value) in sorted(target_to_prediction.items())
    ]


def nearest_point_assignment(predicted: list[Geometry], target: list[Geometry]) -> list[float]:
    """Return target-centric one-to-one distances; unmatched targets get a diagonal penalty."""
    available = set(range(len(predicted)))
    distances: list[float] = []
    # Rarest labels first prevents a common label from affecting deterministic ordering.
    order = sorted(range(len(target)), key=lambda index: (sum(p.label == target[index].label for p in predicted), index))
    assigned: dict[int, float] = {}
    for target_index in order:
        candidates = [
            (point_distance(predicted[index].coordinates, target[target_index].coordinates), index)
            for index in available
            if predicted[index].label == target[target_index].label
        ]
        if not candidates:
            assigned[target_index] = GRID_DIAGONAL
            continue
        distance, prediction_index = min(candidates)
        available.remove(prediction_index)
        assigned[target_index] = distance
    for target_index in range(len(target)):
        distances.append(assigned[target_index])
    return distances


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def prf(true_positive: int, predicted: int, target: int) -> dict[str, float | int | None]:
    precision = ratio(true_positive, predicted)
    recall = ratio(true_positive, target)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@dataclass
class Metrics:
    records: int = 0
    format_valid: int = 0
    structured_exact: int = 0
    ref_predicted: int = 0
    ref_target: int = 0
    ref_tp: int = 0
    ref_exact_records: int = 0
    box_predicted: int = 0
    box_target: int = 0
    box_tp: int = 0
    matched_box_ious: list[float] = field(default_factory=list)
    box_records: int = 0
    box_exact_records: int = 0
    single_box_records: int = 0
    single_box_ious: list[float] = field(default_factory=list)
    point_predicted: int = 0
    point_target: int = 0
    point_records: int = 0
    point_distances: list[float] = field(default_factory=list)
    pck_tp: dict[float, int] = field(default_factory=dict)
    parse_errors: Counter[str] = field(default_factory=Counter)

    def add(self, prediction: ParsedResponse, target: ParsedResponse, iou_threshold: float, pck: tuple[float, ...]) -> dict[str, Any]:
        self.records += 1
        if prediction.valid:
            self.format_valid += 1
        else:
            self.parse_errors[prediction.error or "unknown parse error"] += 1

        effective = prediction if prediction.valid else ParsedResponse(True, (), ())
        if effective.refs == target.refs and effective.geometries == target.geometries:
            self.structured_exact += 1

        predicted_refs = Counter(effective.refs)
        target_refs = Counter(target.refs)
        ref_tp = sum((predicted_refs & target_refs).values())
        self.ref_predicted += sum(predicted_refs.values())
        self.ref_target += sum(target_refs.values())
        self.ref_tp += ref_tp
        ref_exact = predicted_refs == target_refs
        self.ref_exact_records += int(ref_exact)

        predicted_boxes = [value for value in effective.geometries if value.kind == "box"]
        target_boxes = [value for value in target.geometries if value.kind == "box"]
        box_matches = maximum_threshold_matching(
            predicted_boxes, target_boxes, box_iou, iou_threshold, True
        )
        self.box_predicted += len(predicted_boxes)
        self.box_target += len(target_boxes)
        self.box_tp += len(box_matches)
        self.matched_box_ious.extend(value for _, _, value in box_matches)
        if target_boxes:
            self.box_records += 1
            self.box_exact_records += int(
                len(box_matches) == len(target_boxes) == len(predicted_boxes)
            )

        single_iou: float | None = None
        if len(target_boxes) == 1:
            self.single_box_records += 1
            if len(predicted_boxes) == 1 and predicted_boxes[0].label == target_boxes[0].label:
                single_iou = box_iou(predicted_boxes[0].coordinates, target_boxes[0].coordinates)
            else:
                single_iou = 0.0
            self.single_box_ious.append(single_iou)

        predicted_points = [value for value in effective.geometries if value.kind == "point"]
        target_points = [value for value in target.geometries if value.kind == "point"]
        point_distances: list[float] = []
        pck_hits: dict[str, int] = {}
        self.point_predicted += len(predicted_points)
        self.point_target += len(target_points)
        if target_points:
            self.point_records += 1
            point_distances = nearest_point_assignment(predicted_points, target_points)
            self.point_distances.extend(point_distances)
        for threshold in pck:
            matches = maximum_threshold_matching(
                predicted_points,
                target_points,
                point_distance,
                threshold * GRID_DIAGONAL,
                False,
            )
            self.pck_tp[threshold] = self.pck_tp.get(threshold, 0) + len(matches)
            pck_hits[f"{threshold:g}"] = len(matches)

        return {
            "format_valid": prediction.valid,
            "parse_error": prediction.error,
            "terminal_token": prediction.terminal_token,
            "structured_exact": effective.refs == target.refs and effective.geometries == target.geometries,
            "ref_exact": ref_exact,
            "target_refs": len(target.refs),
            "predicted_refs": len(effective.refs),
            "ref_matches": ref_tp,
            "target_boxes": len(target_boxes),
            "predicted_boxes": len(predicted_boxes),
            "box_matches": len(box_matches),
            "matched_box_ious": [value for _, _, value in box_matches],
            "single_box_iou": single_iou,
            "target_points": len(target_points),
            "predicted_points": len(predicted_points),
            "point_target_distances_grid": point_distances,
            "pck_hits": pck_hits,
        }

    def summary(self, iou_threshold: float, pck: tuple[float, ...]) -> dict[str, Any]:
        box = prf(self.box_tp, self.box_predicted, self.box_target)
        box.update({
            "iou_threshold": iou_threshold,
            "mean_iou_of_true_positives": (
                statistics.fmean(self.matched_box_ious) if self.matched_box_ious else None
            ),
            "applicable_records": self.box_records,
            "exact_set_record_rate": ratio(self.box_exact_records, self.box_records),
        })
        single = {
            "applicable_records": self.single_box_records,
            "mean_iou": statistics.fmean(self.single_box_ious) if self.single_box_ious else None,
            "success_rate_at_iou_threshold": ratio(
                sum(value >= iou_threshold for value in self.single_box_ious), self.single_box_records
            ),
        }
        point_metrics: dict[str, Any] = {
            "applicable_records": self.point_records,
            "predicted": self.point_predicted,
            "target": self.point_target,
            "target_mean_distance_grid": (
                statistics.fmean(self.point_distances) if self.point_distances else None
            ),
            "target_mean_distance_normalized_diagonal": (
                statistics.fmean(self.point_distances) / GRID_DIAGONAL if self.point_distances else None
            ),
            "target_median_distance_grid": (
                statistics.median(self.point_distances) if self.point_distances else None
            ),
            "pck": {},
        }
        for threshold in pck:
            point_metrics["pck"][f"{threshold:g}"] = {
                "normalized_diagonal_threshold": threshold,
                "grid_distance_threshold": threshold * GRID_DIAGONAL,
                **prf(self.pck_tp.get(threshold, 0), self.point_predicted, self.point_target),
            }
        return {
            "records": self.records,
            "format": {
                "valid": self.format_valid,
                "valid_rate": ratio(self.format_valid, self.records),
                "parse_errors": dict(sorted(self.parse_errors.items())),
            },
            "structured_exact_match_rate": ratio(self.structured_exact, self.records),
            "label_ref": {
                **prf(self.ref_tp, self.ref_predicted, self.ref_target),
                "exact_multiset_record_rate": ratio(self.ref_exact_records, self.records),
            },
            "box_iou": box,
            "single_box_iou": single,
            "point": point_metrics,
        }


def extract_answers(row: dict[str, Any]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for container_name in ("prediction", "predictions"):
        container = row.get(container_name)
        if isinstance(container, dict):
            for mode, value in container.items():
                if isinstance(value, str):
                    answers[str(mode)] = value
                elif isinstance(value, dict) and isinstance(value.get("answer"), str):
                    answers[str(mode)] = value["answer"]
    for mode in ("hybrid", "slow", "fast"):
        value = row.get(f"{mode}_answer")
        if isinstance(value, str):
            answers[mode] = value
    mode = row.get("mode") or row.get("generation_mode")
    for field_name in ("answer", "prediction_response", "response", "output"):
        if isinstance(row.get(field_name), str):
            answers[str(mode or "s600")] = row[field_name]
            break
    return answers


def target_response(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for field_name in ("profile_target_response", "target_response", "reference_response"):
        if isinstance(row.get(field_name), str) and row[field_name]:
            return row[field_name], field_name
    return None, None


def resolve_target(
    prediction: dict[str, Any], reference: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Prefer profile coordinates even when only the prediction row has them."""
    for owner, row in (("reference", reference), ("prediction", prediction)):
        value = row.get("profile_target_response")
        if isinstance(value, str) and value:
            return value, f"{owner}.profile_target_response"
    for owner, row in (("reference", reference), ("prediction", prediction)):
        for field_name in ("target_response", "reference_response"):
            value = row.get(field_name)
            if isinstance(value, str) and value:
                return value, f"{owner}.{field_name}"
    return None, None


def build_samples(
    prediction_rows: Iterable[dict[str, Any]], reference_rows: Iterable[dict[str, Any]] = ()
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(reference_rows, 1):
        bundle_id = str(row.get("bundle_id") or row.get("sample_id") or "")
        if not bundle_id:
            raise ValueError(f"reference row {index} has no bundle_id or sample_id")
        if bundle_id in references:
            raise ValueError(f"duplicate reference bundle_id: {bundle_id}")
        references[bundle_id] = row

    samples: dict[str, dict[str, Any]] = {}
    # A supplied reference manifest defines the evaluation universe. This keeps
    # entirely missing S600 outputs in the accuracy denominator.
    for bundle_id, reference in references.items():
        expected, target_field = resolve_target({}, reference)
        task = reference.get("task")
        if task not in TASKS:
            raise ValueError(f"{bundle_id}: missing or unsupported task {task!r}")
        if expected is None:
            raise ValueError(f"{bundle_id}: reference has no target response")
        samples[bundle_id] = {
            "bundle_id": bundle_id,
            "task": task,
            "target": expected,
            "target_field": target_field,
            "answers": {},
        }

    for index, row in enumerate(prediction_rows, 1):
        bundle_id = str(row.get("bundle_id") or row.get("sample_id") or "")
        if not bundle_id:
            raise ValueError(f"prediction row {index} has no bundle_id or sample_id")
        reference = references.get(bundle_id, {})
        if references and not reference:
            raise ValueError(f"{bundle_id}: prediction is absent from reference manifest")
        expected, target_field = resolve_target(row, reference)
        task = row.get("task") or reference.get("task")
        if task not in TASKS:
            raise ValueError(f"{bundle_id}: missing or unsupported task {task!r}")
        if expected is None:
            raise ValueError(
                f"{bundle_id}: no profile_target_response/target_response; provide --reference-jsonl"
            )
        sample = samples.setdefault(
            bundle_id,
            {
                "bundle_id": bundle_id,
                "task": task,
                "target": expected,
                "target_field": target_field,
                "answers": {},
            },
        )
        if (
            sample["target"] != expected
            and target_field == "prediction.profile_target_response"
            and sample["target_field"] != "reference.profile_target_response"
        ):
            sample["target"] = expected
            sample["target_field"] = target_field
        if sample["task"] != task or sample["target"] != expected:
            raise ValueError(f"{bundle_id}: inconsistent task or target across prediction rows")
        for mode, answer in extract_answers(row).items():
            if mode in sample["answers"]:
                raise ValueError(f"{bundle_id}: duplicate prediction for mode {mode!r}")
            sample["answers"][mode] = answer
    return samples


def evaluate(
    prediction_rows: Iterable[dict[str, Any]],
    reference_rows: Iterable[dict[str, Any]] = (),
    modes: Iterable[str] | None = None,
    iou_threshold: float = 0.5,
    pck_thresholds: Iterable[float] = DEFAULT_PCK_THRESHOLDS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = build_samples(prediction_rows, reference_rows)
    observed_modes = sorted({mode for sample in samples.values() for mode in sample["answers"]})
    selected_modes = tuple(dict.fromkeys(modes or observed_modes))
    if not selected_modes:
        raise ValueError("no prediction modes found")
    pck = tuple(sorted(set(float(value) for value in pck_thresholds)))
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    if not pck or any(value <= 0.0 or value > 1.0 for value in pck):
        raise ValueError("PCK thresholds must be in (0, 1]")

    target_field_counts: Counter[str] = Counter()
    for sample in samples.values():
        target_field_counts[sample["target_field"]] += 1
    summaries: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    for mode in selected_modes:
        overall = Metrics()
        by_task = {task: Metrics() for task in TASKS}
        available_overall = Metrics()
        available_by_task = {task: Metrics() for task in TASKS}
        for sample in samples.values():
            target = parse_response(sample["target"])
            if not target.valid:
                raise ValueError(
                    f"{sample['bundle_id']}: invalid reference in {sample['target_field']}: {target.error}"
                )
            answer = sample["answers"].get(mode)
            prediction = parse_response(answer)
            detail = overall.add(prediction, target, iou_threshold, pck)
            by_task[sample["task"]].add(prediction, target, iou_threshold, pck)
            if answer is not None:
                available_overall.add(prediction, target, iou_threshold, pck)
                available_by_task[sample["task"]].add(
                    prediction, target, iou_threshold, pck
                )
            details.append({
                "bundle_id": sample["bundle_id"],
                "task": sample["task"],
                "mode": mode,
                "prediction_present": answer is not None,
                "target_field": sample["target_field"],
                **detail,
            })
        summaries[mode] = {
            "prediction_coverage": {
                "present": sum(mode in sample["answers"] for sample in samples.values()),
                "total": len(samples),
                "rate": ratio(
                    sum(mode in sample["answers"] for sample in samples.values()), len(samples)
                ),
            },
            "overall": overall.summary(iou_threshold, pck),
            "by_task": {
                task: by_task[task].summary(iou_threshold, pck) for task in TASKS
            },
            "available_predictions_only": {
                "overall": available_overall.summary(iou_threshold, pck),
                "by_task": {
                    task: available_by_task[task].summary(iou_threshold, pck)
                    for task in TASKS
                },
            },
        }

    result = {
        "schema_version": 1,
        "record_count": len(samples),
        "modes": summaries,
        "configuration": {
            "coordinate_grid": [0, 1000],
            "iou_threshold": iou_threshold,
            "pck_normalization": "Euclidean distance divided by the 1000x1000 grid diagonal",
            "pck_thresholds": list(pck),
            "invalid_prediction_policy": "kept in denominator with zero parsed refs/geometries",
            "metric_denominators": {
                "overall": "all reference records; missing predictions score zero",
                "available_predictions_only": "records containing the evaluated mode; malformed present answers still score zero",
            },
            "matching_policy": "one-to-one, exact normalized label, maximum cardinality at threshold",
            "target_field_counts": dict(sorted(target_field_counts.items())),
        },
    }
    return result, details


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", "--generated-jsonl", dest="predictions_jsonl", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--details-jsonl", type=Path)
    parser.add_argument("--mode", action="append", help="Mode to score; repeat for multiple modes (default: all observed)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--pck-threshold", type=float, action="append")
    args = parser.parse_args()

    result, details = evaluate(
        read_jsonl(args.predictions_jsonl.resolve()),
        read_jsonl(args.reference_jsonl.resolve()) if args.reference_jsonl else (),
        modes=args.mode,
        iou_threshold=args.iou_threshold,
        pck_thresholds=args.pck_threshold or DEFAULT_PCK_THRESHOLDS,
    )
    result["predictions_jsonl"] = str(args.predictions_jsonl.resolve())
    result["reference_jsonl"] = str(args.reference_jsonl.resolve()) if args.reference_jsonl else None
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.details_jsonl:
        write_jsonl(args.details_jsonl, details)
    compact = {
        mode: {
            "records": values["overall"]["records"],
            "prediction_coverage_rate": values["prediction_coverage"]["rate"],
            "format_valid_rate": values["overall"]["format"]["valid_rate"],
            "available_only_records": values["available_predictions_only"]["overall"]["records"],
            "available_only_format_valid_rate": values["available_predictions_only"]["overall"]["format"]["valid_rate"],
            "label_ref_f1": values["overall"]["label_ref"]["f1"],
            "box_f1": values["overall"]["box_iou"]["f1"],
            "point_pck_0.05": values["overall"]["point"]["pck"].get("0.05", {}).get("recall"),
        }
        for mode, values in result["modes"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
