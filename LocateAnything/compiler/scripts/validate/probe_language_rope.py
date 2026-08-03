#!/usr/bin/env python3
"""Numerically validate batch=1 Language RoPE cache gathering in exported BC."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "locateanything.language-rope-gather-probe.v1"
CACHE_POSITIONS = 16
CACHE_CHANNELS = 128
POSITION_PATTERNS = {
    1: (7,),
    6: (5, 1, 7, 0, 9, 3),
}


def stage(name: str) -> None:
    print(f"\n================== {name} ==================", flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sentinel_cache(
    positions: int = CACHE_POSITIONS,
    channels: int = CACHE_CHANNELS,
) -> np.ndarray:
    if positions < 1 or channels < 1:
        raise ValueError("sentinel cache dimensions must be positive")
    position = np.arange(positions, dtype=np.float32)[:, None]
    channel = np.arange(channels, dtype=np.float32)[None, :]
    return 1000.0 * position + channel


def probe_position_ids(q_len: int) -> np.ndarray:
    try:
        pattern = POSITION_PATTERNS[q_len]
    except KeyError as error:
        raise ValueError(f"unsupported q_len: {q_len}") from error
    return np.asarray(pattern, dtype=np.int32).reshape(1, 1, q_len)


def expected_gather(cache: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
    cache = np.asarray(cache)
    position_ids = np.asarray(position_ids)
    if cache.ndim != 2:
        raise ValueError(f"cache must be rank 2, got {cache.shape}")
    if position_ids.ndim != 3 or position_ids.shape[:2] != (1, 1):
        raise ValueError(
            f"position_ids must have shape [1,1,q], got {position_ids.shape}"
        )
    flat = position_ids.reshape(-1)
    if np.any(flat < 0) or np.any(flat >= cache.shape[0]):
        raise ValueError("position_ids contain an out-of-range cache index")
    return np.ascontiguousarray(cache[flat].reshape(1, 1, flat.size, cache.shape[1]))


def compare_output(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {
            "matched": False,
            "status": "shape_mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "mismatch_count": None,
            "max_abs": None,
        }
    difference = np.abs(
        expected.astype(np.float64, copy=False) - actual.astype(np.float64, copy=False)
    )
    mismatch_count = int(np.count_nonzero(expected != actual))
    return {
        "matched": mismatch_count == 0,
        "status": "matched" if mismatch_count == 0 else "value_mismatch",
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "mismatch_count": mismatch_count,
        "max_abs": float(difference.max(initial=0.0)),
    }


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def execute_exported_bc(
    compiler: Any,
    path: Path,
    graph_name: str,
    position_ids: np.ndarray,
) -> dict[str, Any]:
    load = getattr(compiler, "load", None)
    if not callable(load):
        return {"status": "unsupported", "reason": "hbdk4.compiler.load is unavailable"}

    module = load(str(path))
    functions = getattr(module, "functions", None)
    if functions is None:
        return {"status": "unsupported", "reason": "loaded BC exposes no functions"}
    function = next(
        (item for item in functions if str(getattr(item, "name", "")) == graph_name),
        None,
    )
    if function is None:
        raise RuntimeError(f"exported BC does not contain graph {graph_name}")
    feed = getattr(function, "feed", None)
    if not callable(feed):
        return {"status": "unsupported", "reason": "HBDK function.feed is unavailable"}

    inputs = list(function.inputs)
    outputs = list(function.outputs)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"unexpected graph signature: {len(inputs)} inputs, {len(outputs)} outputs"
        )
    input_descriptor = inputs[0]
    output_descriptor = outputs[0]
    feed_values = {
        str(input_descriptor.name): position_ids.astype(
            input_descriptor.type.np_dtype, copy=False
        )
    }
    result = feed(inputs=feed_values)
    return {
        "status": "executed",
        "output": np.asarray(result[str(output_descriptor.name)]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/diagnostics/language_rope_gather"),
        help="Empty directory for q=1/q=6 exported BC files and report.json.",
    )
    return parser


def run(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "passed": None,
        "sentinel": {
            "formula": "cache_cos[position, channel] = 1000 * position + channel",
            "shape": [CACHE_POSITIONS, CACHE_CHANNELS],
            "dtype": "float32",
        },
        "graphs": {},
        "error": None,
    }

    exit_code = 0
    try:
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root / "compiler"))

        import torch
        import hbdk4.compiler as hb
        from hbdk4.compiler import leap, save
        from leap_llm.nn.utils import Model

        cache_numpy = sentinel_cache()

        class LanguageRopeGatherProbe(Model):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "cache_cos",
                    torch.from_numpy(cache_numpy.copy()),
                    persistent=True,
                )

            def build(self, position_ids: Any) -> Any:
                batch, _, q_len = position_ids.type.shape
                indices = leap.reshape(position_ids, (batch, -1))
                indices = leap.transpose(indices, (1, 0))
                gathered = leap.gather_nd(self.cache_cos, indices, 0)
                return leap.reshape(gathered, (batch, 1, q_len, -1))

            def forward(self, position_ids: Any) -> Any:
                batch, _, q_len = position_ids.shape
                indices = position_ids.reshape(batch, q_len)
                return self.cache_cos[indices].reshape(
                    batch, 1, q_len, self.cache_cos.shape[1]
                )

        model = LanguageRopeGatherProbe()
        model.compile_mode(True)
        mismatches: list[str] = []
        executed = 0

        for q_len in POSITION_PATTERNS:
            graph_name = f"language_rope_gather_q{q_len}"
            exported_path = output_dir / f"{graph_name}.bc"
            position_ids = probe_position_ids(q_len)
            expected = expected_gather(cache_numpy, position_ids)

            stage(f"EXPORT LANGUAGE ROPE GATHER Q={q_len}")
            exported = model.export_module(
                [leap.TensorType([1, 1, q_len], leap.int32)],
                name=graph_name,
                save_path=None,
                high_precision_qpp=True,
            )
            save(exported, str(exported_path))
            graph_report: dict[str, Any] = {
                "q_len": q_len,
                "graph": graph_name,
                "position_ids": position_ids.reshape(-1).tolist(),
                "expected_shape": list(expected.shape),
                "expected": expected.tolist(),
                "artifact": artifact(exported_path),
            }

            stage(f"EXECUTE EXPORTED BC Q={q_len}")
            execution = execute_exported_bc(hb, exported_path, graph_name, position_ids)
            if execution["status"] == "executed":
                executed += 1
                actual = execution.pop("output")
                comparison = compare_output(expected, actual)
                execution["comparison"] = comparison
                execution["actual"] = actual.tolist()
                if not comparison["matched"]:
                    mismatches.append(graph_name)
                print(
                    f"Q={q_len}: {comparison['status']} "
                    f"mismatches={comparison['mismatch_count']} "
                    f"max_abs={comparison['max_abs']}",
                    flush=True,
                )
            else:
                print(f"Q={q_len}: numeric execution unavailable: {execution['reason']}")
            graph_report["execution"] = execution
            report["graphs"][f"q{q_len}"] = graph_report

        if mismatches:
            raise RuntimeError(
                "exported BC RoPE gather mismatch: " + ", ".join(mismatches)
            )
        if executed != len(POSITION_PATTERNS):
            raise RuntimeError(
                f"numeric BC execution unavailable for "
                f"{len(POSITION_PATTERNS) - executed} graph(s)"
            )
        report["passed"] = True
        report["status"] = "passed"
        stage("PROBE COMPLETED")
        print(f"STATUS: {report['status']}")
        print(f"NUMERIC_GRAPHS: {executed}/{len(POSITION_PATTERNS)}")
    except Exception as error:
        exit_code = 2
        report["status"] = "failed"
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        stage("PROBE FAILED")
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        report["finished_at"] = utc_now()
        atomic_json(report_path, report)
        print(f"REPORT: {report_path}", flush=True)
    return exit_code


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args.output_dir)
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
