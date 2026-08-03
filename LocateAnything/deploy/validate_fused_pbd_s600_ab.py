#!/usr/bin/env python3
"""Compare standard and fused-decode Language HBMs on S600."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


RUN_COUNT = 5
BASE_GRAPHS = ("prefill", "decode", "decode_ar")
FUSED_PBD_GRAPHS = tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13))
FUSED_AR_GRAPHS = tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6))
EXPECTED_FUSED_GRAPHS = BASE_GRAPHS + FUSED_PBD_GRAPHS + FUSED_AR_GRAPHS
FUSED_DECODE_LOG = "[INFO] Language graph set=fused_decode"
STANDARD_LOG = "[INFO] Language graph set=standard"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER_PATTERN = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
BOX_RE = re.compile(
    rf"<box>\s*<({NUMBER_PATTERN})>\s*<({NUMBER_PATTERN})>\s*"
    rf"<({NUMBER_PATTERN})>\s*<({NUMBER_PATTERN})>\s*</box>"
)
POINT_RE = re.compile(
    rf"<point>\s*<({NUMBER_PATTERN})>\s*<({NUMBER_PATTERN})>\s*</point>"
)
POINT_BOX_RE = re.compile(
    rf"<box>\s*<({NUMBER_PATTERN})>\s*<({NUMBER_PATTERN})>\s*</box>"
)


class ValidationError(RuntimeError):
    """Raised when an A/B acceptance gate fails."""


@dataclass(frozen=True)
class GenerationOutput:
    stop_reason: str
    token_ids: tuple[int, ...]
    structured: str


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    stop_reason: str
    response_size: int
    prefill_tokens: int
    prefill_ms: float
    decode_ms: float


@dataclass(frozen=True)
class ModelRun:
    tag: str
    model_path: Path
    graph_names: tuple[str, ...]
    generations: tuple[GenerationOutput, ...]
    metrics: tuple[RequestMetrics, ...]
    log_path: Path
    log_lines: tuple[str, ...]


def parse_generation_text(text: str, source: str = "generation output") -> GenerationOutput:
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValidationError(f"{source}:{line_number}: malformed line")
        if key in fields:
            raise ValidationError(f"{source}:{line_number}: duplicate field {key!r}")
        fields[key] = value
    required = {"stop_reason", "token_ids", "structured"}
    missing = sorted(required - fields.keys())
    if missing:
        raise ValidationError(f"{source}: missing fields: {', '.join(missing)}")
    try:
        token_ids = tuple(int(value) for value in fields["token_ids"].split(",") if value)
    except ValueError as error:
        raise ValidationError(f"{source}: invalid token_ids") from error
    return GenerationOutput(fields["stop_reason"], token_ids, fields["structured"])


def parse_generation_file(path: Path) -> GenerationOutput:
    if not path.is_file():
        raise ValidationError(f"generation output was not created: {path}")
    return parse_generation_text(path.read_text(encoding="utf-8"), str(path))


def parse_graph_names(lines: Iterable[str]) -> tuple[str, ...]:
    prefix = "[ok] loaded graphs:"
    matches = [line[len(prefix):].strip().split() for line in lines if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValidationError(f"expected one non-empty graph list, found {len(matches)}")
    names = tuple(matches[0])
    if len(names) != len(set(names)):
        raise ValidationError("loaded graph list contains duplicate names")
    return names


def parse_result_line(line: str, expected_request_id: str) -> RequestMetrics:
    fields = line.split("\t")
    if len(fields) != 8 or fields[:2] != ["LAHBM/1", "RESULT"]:
        raise ValidationError(f"invalid RESULT frame: {line}")
    if fields[2] != expected_request_id:
        raise ValidationError(
            f"RESULT request id {fields[2]!r} does not match {expected_request_id!r}"
        )
    try:
        metrics = RequestMetrics(
            request_id=fields[2],
            stop_reason=fields[3],
            response_size=int(fields[4]),
            prefill_tokens=int(fields[5]),
            prefill_ms=float(fields[6]),
            decode_ms=float(fields[7]),
        )
    except ValueError as error:
        raise ValidationError(f"invalid numeric field in RESULT frame: {line}") from error
    if metrics.response_size < 0 or metrics.prefill_tokens <= 0:
        raise ValidationError(f"invalid token counts in RESULT frame: {line}")
    if not math.isfinite(metrics.prefill_ms) or metrics.prefill_ms <= 0.0:
        raise ValidationError(f"invalid prefill time in RESULT frame: {line}")
    if not math.isfinite(metrics.decode_ms) or metrics.decode_ms <= 0.0:
        raise ValidationError(f"invalid decode time in RESULT frame: {line}")
    return metrics


class ResidentLanguageServer:
    def __init__(
        self,
        runner: Path,
        model: Path,
        embedding: Path,
        graph_set: str,
        log_path: Path,
        startup_timeout: float,
        request_timeout: float,
    ) -> None:
        self.runner = runner
        self.model = model
        self.embedding = embedding
        self.graph_set = graph_set
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.process: subprocess.Popen[str] | None = None
        self.lines: list[str] = []
        self._line_queue: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        env = os.environ.copy()
        env.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                str(self.runner),
                "--model",
                str(self.model),
                "--embed",
                str(self.embedding),
                "--graph-set",
                self.graph_set,
                "--server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        while True:
            line = self._next_line(self.startup_timeout, "server startup")
            if line == "LAHBM/1\tREADY\tlanguage":
                return
            if line.startswith("[FAIL]"):
                raise ValidationError(f"Language server failed during startup: {line}")

    def _read_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        with self.log_path.open("w", encoding="utf-8", buffering=1) as log:
            for raw_line in self.process.stdout:
                log.write(raw_line)
                line = ANSI_ESCAPE_RE.sub("", raw_line.rstrip("\r\n"))
                self.lines.append(line)
                self._line_queue.put(line)
        self._line_queue.put(None)

    def _next_line(self, timeout: float, phase: str) -> str:
        try:
            line = self._line_queue.get(timeout=timeout)
        except queue.Empty as error:
            raise ValidationError(
                f"timed out after {timeout:.0f}s during {phase}; log: {self.log_path}"
            ) from error
        if line is None:
            code = self.process.poll() if self.process is not None else None
            raise ValidationError(
                f"Language server exited during {phase} with code {code}; log: {self.log_path}"
            )
        return line

    def request(
        self,
        request_id: str,
        token_path: Path,
        visual_path: Path,
        output_path: Path,
        max_new_tokens: int,
    ) -> tuple[GenerationOutput, RequestMetrics]:
        assert self.process is not None and self.process.stdin is not None
        for value in (request_id, str(token_path), str(visual_path), str(output_path)):
            if any(character in value for character in "\t\r\n"):
                raise ValidationError("server protocol fields must not contain tabs or newlines")
        frame = "\t".join(
            (
                "LAHBM/1",
                "RUN",
                request_id,
                str(token_path),
                str(visual_path),
                str(output_path),
                str(max_new_tokens),
                "hybrid",
            )
        )
        self.process.stdin.write(frame + "\n")
        self.process.stdin.flush()
        streamed_tokens: list[int] = []
        while True:
            line = self._next_line(self.request_timeout, f"request {request_id}")
            if line.startswith("LAHBM/1\tTOKEN\t"):
                fields = line.split("\t")
                if len(fields) != 4 or fields[2] != request_id:
                    raise ValidationError(f"invalid TOKEN frame: {line}")
                try:
                    streamed_tokens.append(int(fields[3]))
                except ValueError as error:
                    raise ValidationError(f"invalid token in TOKEN frame: {line}") from error
                continue
            if line.startswith("LAHBM/1\tERROR\t"):
                fields = line.split("\t", 4)
                if len(fields) >= 3 and fields[2] in {request_id, "0"}:
                    raise ValidationError(f"Language server rejected {request_id}: {line}")
            if line.startswith("LAHBM/1\tRESULT\t"):
                metrics = parse_result_line(line, request_id)
                generation = parse_generation_file(output_path)
                if generation.token_ids != tuple(streamed_tokens):
                    raise ValidationError(
                        f"{request_id}: streamed tokens differ from saved token_ids"
                    )
                return generation, metrics

    def shutdown(self, check: bool) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("LAHBM/1\tQUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        if self._reader is not None:
            self._reader.join(timeout=10)
        code = self.process.returncode
        if check and code != 0:
            raise ValidationError(f"Language server exited with code {code}; log: {self.log_path}")


def run_model(
    tag: str,
    runner: Path,
    model: Path,
    embedding: Path,
    graph_set: str,
    token_path: Path,
    visual_path: Path,
    output_dir: Path,
    max_new_tokens: int,
    startup_timeout: float,
    request_timeout: float,
) -> ModelRun:
    model_dir = output_dir / tag
    model_dir.mkdir()
    server = ResidentLanguageServer(
        runner,
        model,
        embedding,
        graph_set,
        model_dir / "server.log",
        startup_timeout,
        request_timeout,
    )
    generations: list[GenerationOutput] = []
    metrics: list[RequestMetrics] = []
    try:
        server.start()
        for index in range(1, RUN_COUNT + 1):
            generation, request_metrics = server.request(
                f"{tag}-{index:02d}",
                token_path,
                visual_path,
                model_dir / f"round_{index:02d}.txt",
                max_new_tokens,
            )
            generations.append(generation)
            metrics.append(request_metrics)
        server.shutdown(check=True)
    except Exception:
        server.shutdown(check=False)
        raise
    return ModelRun(
        tag=tag,
        model_path=model,
        graph_names=parse_graph_names(server.lines),
        generations=tuple(generations),
        metrics=tuple(metrics),
        log_path=server.log_path,
        log_lines=tuple(server.lines),
    )


def _validate_model_run(run: ModelRun) -> None:
    if len(run.generations) != RUN_COUNT or len(run.metrics) != RUN_COUNT:
        raise ValidationError(f"{run.tag}: expected {RUN_COUNT} completed requests")
    first = run.generations[0]
    for index, (generation, metrics) in enumerate(
        zip(run.generations, run.metrics), 1
    ):
        if generation.stop_reason != "im_end":
            raise ValidationError(
                f"{run.tag} round {index}: stop_reason={generation.stop_reason!r}, expected 'im_end'"
            )
        if not generation.token_ids:
            raise ValidationError(f"{run.tag} round {index}: empty token_ids")
        if generation != first:
            raise ValidationError(f"{run.tag}: output changed at round {index}")
        if metrics.stop_reason != generation.stop_reason:
            raise ValidationError(f"{run.tag} round {index}: RESULT/output stop_reason mismatch")
        if metrics.response_size != len(generation.token_ids):
            raise ValidationError(f"{run.tag} round {index}: RESULT/output token count mismatch")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values[1:]))


def _localization_parts(
    structured: str,
) -> tuple[str, tuple[tuple[float, float, float, float], ...], tuple[tuple[float, float], ...]]:
    boxes = tuple(tuple(float(value) for value in match.groups()) for match in BOX_RE.finditer(structured))
    point_boxes = tuple(
        tuple(float(value) for value in match.groups())
        for match in POINT_BOX_RE.finditer(structured)
    )
    tagged_points = tuple(
        tuple(float(value) for value in match.groups())
        for match in POINT_RE.finditer(structured)
    )
    points = point_boxes + tagged_points
    skeleton = BOX_RE.sub("<box><coord><coord><coord><coord></box>", structured)
    skeleton = POINT_BOX_RE.sub("<box><coord><coord></box>", skeleton)
    skeleton = POINT_RE.sub("<point><coord><coord></point>", skeleton)
    return skeleton, boxes, points


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 1.0 if left == right else 0.0
    return intersection / union


def _match_box_ious(
    left_boxes: tuple[tuple[float, float, float, float], ...],
    right_boxes: tuple[tuple[float, float, float, float], ...],
    threshold: float,
) -> tuple[float, ...]:
    scores = tuple(
        tuple(_box_iou(left, right) for right in right_boxes)
        for left in left_boxes
    )
    right_match = [-1] * len(right_boxes)

    def augment(left_index: int, seen: set[int]) -> bool:
        candidates = sorted(
            range(len(right_boxes)),
            key=lambda right_index: scores[left_index][right_index],
            reverse=True,
        )
        for right_index in candidates:
            if right_index in seen or scores[left_index][right_index] < threshold:
                continue
            seen.add(right_index)
            if right_match[right_index] < 0 or augment(right_match[right_index], seen):
                right_match[right_index] = left_index
                return True
        return False

    for left_index in range(len(left_boxes)):
        augment(left_index, set())
    pairs = sorted(
        (left_index, right_index)
        for right_index, left_index in enumerate(right_match)
        if left_index >= 0
    )
    return tuple(scores[left_index][right_index] for left_index, right_index in pairs)


def compare_localization_outputs(
    old: GenerationOutput,
    new: GenerationOutput,
    min_box_iou: float,
    max_point_distance: float,
) -> dict[str, object]:
    if old.stop_reason != new.stop_reason:
        raise ValidationError("old/new stop_reason differ")
    old_skeleton, old_boxes, old_points = _localization_parts(old.structured)
    new_skeleton, new_boxes, new_points = _localization_parts(new.structured)
    if old_skeleton != new_skeleton:
        raise ValidationError("old/new labels or localization output structure differ")
    if len(old_boxes) != len(new_boxes) or len(old_points) != len(new_points):
        raise ValidationError("old/new localization result counts differ")

    box_ious = _match_box_ious(old_boxes, new_boxes, min_box_iou)
    if len(box_ious) != len(old_boxes):
        raise ValidationError(
            f"old/new box IoU below {min_box_iou:.3f}: "
            f"matched {len(box_ious)}/{len(old_boxes)}"
        )
    point_distances = tuple(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left, right in zip(old_points, new_points)
    )
    if any(value > max_point_distance for value in point_distances):
        raise ValidationError(
            f"old/new point distance exceeds {max_point_distance:.3f}: "
            + ", ".join(f"{value:.6f}" for value in point_distances)
        )
    return {
        "token_ids_exact": old.token_ids == new.token_ids,
        "structured_exact": old.structured == new.structured,
        "box_ious": list(box_ious),
        "point_distances": list(point_distances),
    }


def _run_summary(run: ModelRun) -> dict[str, object]:
    prefill_median = _median([item.prefill_ms for item in run.metrics])
    decode_median = _median([item.decode_ms for item in run.metrics])
    language_median = _median(
        [item.prefill_ms + item.decode_ms for item in run.metrics]
    )
    return {
        "model": str(run.model_path),
        "log": str(run.log_path),
        "graph_names": list(run.graph_names),
        "rounds": [
            {
                "request_id": item.request_id,
                "stop_reason": item.stop_reason,
                "response_size": item.response_size,
                "prefill_tokens": item.prefill_tokens,
                "prefill_ms": item.prefill_ms,
                "decode_ms": item.decode_ms,
            }
            for item in run.metrics
        ],
        "warmup_rounds_excluded": 1,
        "prefill_median_ms": prefill_median,
        "decode_median_ms": decode_median,
        "language_median_ms": language_median,
    }


def validate_ab(
    old: ModelRun,
    new: ModelRun,
    min_box_iou: float = 0.90,
    max_point_distance: float = 10.0,
) -> dict[str, object]:
    _validate_model_run(old)
    _validate_model_run(new)
    old_graphs = set(old.graph_names)
    new_graphs = set(new.graph_names)
    if not set(BASE_GRAPHS) <= old_graphs:
        raise ValidationError(f"old HBM is missing base graphs: {sorted(set(BASE_GRAPHS) - old_graphs)}")
    if set(EXPECTED_FUSED_GRAPHS) <= old_graphs:
        raise ValidationError("old HBM already contains the complete fused graph family")
    if len(new.graph_names) != len(EXPECTED_FUSED_GRAPHS) or new_graphs != set(
        EXPECTED_FUSED_GRAPHS
    ):
        missing = sorted(set(EXPECTED_FUSED_GRAPHS) - new_graphs)
        extra = sorted(new_graphs - set(EXPECTED_FUSED_GRAPHS))
        raise ValidationError(
            f"new HBM graph set mismatch: count={len(new.graph_names)} "
            f"missing={missing} extra={extra}"
        )
    if old.log_lines.count(STANDARD_LOG) < RUN_COUNT:
        raise ValidationError("old HBM did not use the standard graph set in every round")
    if FUSED_DECODE_LOG in old.log_lines:
        raise ValidationError("old HBM unexpectedly used the fused_decode graph set")
    if new.log_lines.count(FUSED_DECODE_LOG) < RUN_COUNT:
        raise ValidationError("new HBM did not use the fused_decode graph set in every round")
    if STANDARD_LOG in new.log_lines:
        raise ValidationError("new HBM unexpectedly used the standard graph set")
    output_comparisons: list[dict[str, object]] = []
    for index, (old_output, new_output) in enumerate(
        zip(old.generations, new.generations), 1
    ):
        try:
            comparison = compare_localization_outputs(
                old_output, new_output, min_box_iou, max_point_distance
            )
        except ValidationError as error:
            raise ValidationError(f"round {index}: {error}") from error
        comparison["round"] = index
        output_comparisons.append(comparison)
        if old.metrics[index - 1].prefill_tokens != new.metrics[index - 1].prefill_tokens:
            raise ValidationError(f"round {index}: old/new prefill token counts differ")

    old_summary = _run_summary(old)
    new_summary = _run_summary(new)
    speedup = {
        "prefill": old_summary["prefill_median_ms"] / new_summary["prefill_median_ms"],
        "decode": old_summary["decode_median_ms"] / new_summary["decode_median_ms"],
        "language": old_summary["language_median_ms"] / new_summary["language_median_ms"],
    }
    return {
        "schema_version": 2,
        "passed": True,
        "run_count": RUN_COUNT,
        "output_comparison": {
            "policy": "labels-and-structure-exact; boxes-by-iou; points-by-distance",
            "min_box_iou": min_box_iou,
            "max_point_distance": max_point_distance,
            "exact_token_rounds": sum(
                bool(item["token_ids_exact"]) for item in output_comparisons
            ),
            "exact_structured_rounds": sum(
                bool(item["structured_exact"]) for item in output_comparisons
            ),
            "rounds": output_comparisons,
        },
        "old": old_summary,
        "new": new_summary,
        "speedup_old_over_new": speedup,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _input_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValidationError(f"{label} is missing or empty: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValidationError(f"{label} is not executable: {resolved}")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--embed", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-box-iou", type=float, default=0.90)
    parser.add_argument("--max-point-distance", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_created = False
    try:
        if args.max_new_tokens <= 0:
            raise ValidationError("--max-new-tokens must be positive")
        if args.startup_timeout <= 0 or args.request_timeout <= 0:
            raise ValidationError("timeouts must be positive")
        if not 0.0 <= args.min_box_iou <= 1.0:
            raise ValidationError("--min-box-iou must be in [0, 1]")
        if args.max_point_distance < 0.0:
            raise ValidationError("--max-point-distance must be non-negative")
        runner = _input_file(args.runner, "Language runner", executable=True)
        old_model = _input_file(args.old_model, "old Language HBM")
        new_model = _input_file(args.new_model, "new Language HBM")
        embedding = _input_file(args.embed, "embedding")
        token_path = _input_file(args.tokens, "prompt token payload")
        visual_path = _input_file(args.visual, "visual feature payload")
        if old_model == new_model:
            raise ValidationError("old and new HBM paths must differ")
        output_dir.mkdir(parents=True, exist_ok=False)
        output_created = True
        old = run_model(
            "old",
            runner,
            old_model,
            embedding,
            "standard",
            token_path,
            visual_path,
            output_dir,
            args.max_new_tokens,
            args.startup_timeout,
            args.request_timeout,
        )
        new = run_model(
            "new",
            runner,
            new_model,
            embedding,
            "fused_decode",
            token_path,
            visual_path,
            output_dir,
            args.max_new_tokens,
            args.startup_timeout,
            args.request_timeout,
        )
        report = validate_ab(
            old,
            new,
            min_box_iou=args.min_box_iou,
            max_point_distance=args.max_point_distance,
        )
        report["inputs"] = {
            "runner": str(runner),
            "embedding": str(embedding),
            "tokens": str(token_path),
            "visual": str(visual_path),
            "max_new_tokens": args.max_new_tokens,
        }
        _write_json(output_dir / "report.json", report)
        speedup = report["speedup_old_over_new"]
        print("[PASS] fused-PBD S600 A/B acceptance")
        comparison = report["output_comparison"]
        print(f"[PASS] localization-equivalent outputs: {RUN_COUNT}/{RUN_COUNT}")
        print(f"[INFO] exact token rounds: {comparison['exact_token_rounds']}/{RUN_COUNT}")
        print(f"[PERF] prefill speedup: {speedup['prefill']:.3f}x")
        print(f"[PERF] decode speedup: {speedup['decode']:.3f}x")
        print(f"[PERF] language speedup: {speedup['language']:.3f}x")
        print(f"[OUTPUT] {output_dir / 'report.json'}")
        return 0
    except Exception as error:
        if output_created:
            _write_json(
                output_dir / "failure.json",
                {
                    "schema_version": 1,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
