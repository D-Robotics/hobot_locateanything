#!/usr/bin/env python3
"""Run independent LocateAnything Float, Quantized Eager, BC, and HBM stages."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "compiler"))

from compiler.scripts.common.trace import (
    TRACE_MODES,
    TraceRecorder,
    canonical_name,
    iter_tensors,
    semantic_location,
    trace_torch_modules,
)
from compiler.scripts.common.quantization import (
    EMULATION_VERSION,
    LIMITATION as QUANTIZATION_LIMITATION,
    SCHEME as QUANTIZATION_SCHEME,
    BoundaryCapture,
    QuantizationEmulator,
)
from compiler.scripts.common.language import (
    EMULATION_VERSION as LANGUAGE_EMULATION_VERSION,
    LIMITATION as LANGUAGE_QUANTIZATION_LIMITATION,
    SCHEME as LANGUAGE_QUANTIZATION_SCHEME,
    LANGUAGE_BC_GRAPHS,
    LanguageBCArtifact,
    LanguageBCRunner,
    LanguageEagerRunner,
    create_language_model,
    load_language_bc_artifacts,
    load_payload as load_language_payload,
    language_quantization_policy,
    resolve_language_bc_paths,
)
from compiler.scripts.common.coordinates import (
    AUDIT_VERSION as COORDINATE_AUDIT_VERSION,
    coordinate_metric_rows,
)

VOCAB, HIDDEN, CACHE, GROUPS, HEAD_DIM, LAYERS = 152681, 2048, 4096, 2, 128, 36
Q_LENGTHS = {"visual": None, "prefill": 1024, "decode": 6, "decode_ar": 1}
BC_MODES = {"exported-bc": "exported_bc", "converted-bc": "converted_bc"}
QUANTIZED_EAGER_MODE = "quantized-eager"
QUANTIZED_EAGER_STAGE = "quantized_eager"
MODES = ("float", QUANTIZED_EAGER_MODE, *BC_MODES, "hbm", "analysis")
LEVELS = ("small", "medium", "high")
PHASES = ("vision", "language", "full_model")
CAPTURE_LEVELS = {"small": "final", "medium": "boundary", "high": "deep"}
CANDIDATE_STAGES = ("exported_bc", "converted_bc", "hbm")
VISION_OUTPUT_SHAPE = (1, 576, 2048)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "workspace" / "evaluation" / "pipeline"
DEFAULT_CONFIG = REPO_ROOT / "compiler" / "config.yaml"


class SimpleProgress:
    def __init__(self, items: list[Any], description: str) -> None:
        self.items = items
        self.description = description
        self.total = len(items)
        self.postfix: dict[str, Any] = {}

    def __iter__(self):
        step = max(1, self.total // 100)
        for index, item in enumerate(self.items, start=1):
            yield item
            if index == self.total or index % step == 0:
                details = " ".join(f"{key}={value}" for key, value in self.postfix.items())
                print(f"{self.description}: {index}/{self.total} {details}", flush=True)

    def set_postfix(self, values: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.postfix = dict(values or {})
        self.postfix.update(kwargs)

    def close(self) -> None:
        return None


def progress_bar(items: list[Any], description: str) -> Any:
    label = description.upper()
    print(f"\n================== {label} ==================", flush=True)
    try:
        from tqdm import tqdm

        return tqdm(items, desc=description, unit="sample", dynamic_ncols=True)
    except ImportError:
        return SimpleProgress(items, description)


def print_phase_summary(
    name: str,
    total: int,
    processed: int,
    resumed: int,
    started: float,
    output: Path,
    *,
    elapsed_seconds: float | None = None,
    **details: Any,
) -> None:
    elapsed = time.monotonic() - started if elapsed_seconds is None else elapsed_seconds
    print(f"\n================== {name.upper()} COMPLETED ==================", flush=True)
    print(f"TOTAL: {total}", flush=True)
    print(f"PROCESSED: {processed}", flush=True)
    print(f"RESUMED: {resumed}", flush=True)
    print(f"ELAPSED_SECONDS: {elapsed:.3f}", flush=True)
    print(f"RATE: {processed / elapsed:.3f} sample/s" if elapsed else "RATE: N/A", flush=True)
    for key, value in details.items():
        print(f"{key.upper()}: {value}", flush=True)
    print(f"OUTPUT: {output.resolve()}\n", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str | None:
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    values = {f"output_{index:03d}": value for index, value in enumerate(arrays)}
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def atomic_named_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_named_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
    os.replace(temporary, path)


def atomic_binary(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(np.ascontiguousarray(value).tobytes())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def discover_inputs(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".jsonl":
        discovered = []
        for record in read_jsonl(path):
            tensor_path = (path.parent / record["tensor_file"]).resolve()
            actual_sha256 = sha256(tensor_path)
            declared_sha256 = record.get("tensor_sha256")
            if declared_sha256 is not None and declared_sha256 != actual_sha256:
                raise ValueError(f"tensor SHA256 mismatch: {tensor_path}")
            discovered.append(
                {
                    "id": str(record.get("bundle_id") or tensor_path.stem),
                    "path": str(tensor_path),
                    "sha256": actual_sha256,
                    "task": record.get("task"),
                    "source": record.get("source"),
                    "image": record.get("image"),
                    "image_sha256": record.get("image_sha256"),
                }
            )
        return discovered
    if path.is_file():
        return [{"id": path.stem, "path": str(path), "sha256": sha256(path)}]
    if not path.is_dir():
        raise FileNotFoundError(path)
    for manifest in (path / "generated.jsonl", path / "generated" / "generated.jsonl"):
        if manifest.is_file():
            return discover_inputs(manifest)
    files = sorted(
        item
        for suffix in ("*.pt", "*.pth", "*.npy", "*.npz")
        for item in path.rglob(suffix)
        if item.is_file()
    )
    if not files:
        raise ValueError(f"no supported inputs found under {path}")
    started = time.monotonic()
    progress = progress_bar(files, "Input integrity")
    records = []
    for item in progress:
        records.append(
            {"id": item.stem, "path": str(item.resolve()), "sha256": sha256(item)}
        )
        progress.set_postfix(file=item.name)
    progress.close()
    print_phase_summary("Input integrity", len(files), len(files), 0, started, path)
    return records


def prepare_input_index(output_dir: Path, input_dir: Path) -> list[dict[str, Any]]:
    records = discover_inputs(input_dir)
    identity = [{"id": item["id"], "sha256": item.get("sha256")} for item in records]
    index_path = output_dir / "inputs.json"
    if index_path.is_file():
        existing = read_json(index_path)
        existing_identity = [
            {"id": item["id"], "sha256": item.get("sha256")}
            for item in existing.get("inputs", [])
        ]
        if existing_identity != identity:
            raise ValueError(f"{index_path} belongs to a different input set")
        if existing.get("inputs") != records or existing.get("source") != str(input_dir.resolve()):
            atomic_json(
                index_path,
                {
                    **existing,
                    "source": str(input_dir.resolve()),
                    "count": len(records),
                    "inputs": records,
                    "updated_at": utc_now(),
                },
            )
    else:
        atomic_json(
            index_path,
            {
                "schema_version": 1,
                "source": str(input_dir.resolve()),
                "count": len(records),
                "inputs": records,
                "created_at": utc_now(),
            },
        )
    return records


def load_npz(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return [np.asarray(payload[name]) for name in sorted(payload.files)]


def describe_arrays(arrays: list[np.ndarray], names: list[str] | None = None) -> list[dict[str, Any]]:
    labels = names or [f"output_{index:03d}" for index in range(len(arrays))]
    if len(labels) != len(arrays):
        raise ValueError(f"expected {len(arrays)} output names, got {len(labels)}")
    return [
        {
            "index": index,
            "name": labels[index],
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": value.nbytes,
        }
        for index, value in enumerate(arrays)
    ]


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "status": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    left = reference.reshape(-1)
    right = candidate.reshape(-1)
    dot = left_sq = right_sq = left_sum = right_sum = abs_sum = diff_sq = 0.0
    max_abs = 0.0
    left_min = right_min = float("inf")
    left_max = right_max = float("-inf")
    left_nonzero = right_nonzero = False
    exact_equal = True
    for start in range(0, left.size, 1_000_000):
        left_chunk = left[start:start + 1_000_000].astype(np.float64)
        right_chunk = right[start:start + 1_000_000].astype(np.float64)
        if not np.isfinite(left_chunk).all() or not np.isfinite(right_chunk).all():
            return {"status": "nonfinite"}
        delta = right_chunk - left_chunk
        exact_equal = exact_equal and bool(np.array_equal(left_chunk, right_chunk))
        dot += float(np.dot(left_chunk, right_chunk))
        left_sq += float(np.dot(left_chunk, left_chunk))
        right_sq += float(np.dot(right_chunk, right_chunk))
        left_sum += float(left_chunk.sum())
        right_sum += float(right_chunk.sum())
        abs_sum += float(np.abs(delta).sum())
        diff_sq += float(np.dot(delta, delta))
        max_abs = max(max_abs, float(np.abs(delta).max()))
        left_min = min(left_min, float(left_chunk.min()))
        left_max = max(left_max, float(left_chunk.max()))
        right_min = min(right_min, float(right_chunk.min()))
        right_max = max(right_max, float(right_chunk.max()))
        left_nonzero = left_nonzero or bool(np.any(left_chunk != 0))
        right_nonzero = right_nonzero or bool(np.any(right_chunk != 0))
    left_norm = left_sq**0.5
    right_norm = right_sq**0.5
    left_mean = left_sum / left.size
    right_mean = right_sum / right.size
    top1 = None
    if reference.ndim >= 2 and reference.shape[-1] > 1:
        top1 = float(np.mean(np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1)))
    return {
        "status": "compared",
        "shape": list(reference.shape),
        "cosine": dot / max(left_norm * right_norm, np.finfo(np.float64).tiny),
        "relative_l2": diff_sq**0.5 / max(left_norm, np.finfo(np.float64).tiny),
        "mae": abs_sum / left.size,
        "rmse": (diff_sq / left.size) ** 0.5,
        "max_abs": max_abs,
        "top1_agreement": top1,
        "exact_equal": exact_equal,
        "reference_nonzero": left_nonzero,
        "candidate_nonzero": right_nonzero,
        "reference_range": [left_min, left_max],
        "candidate_range": [right_min, right_max],
        "reference_mean": left_mean,
        "candidate_mean": right_mean,
        "reference_std": max(left_sq / left.size - left_mean**2, 0.0) ** 0.5,
        "candidate_std": max(right_sq / right.size - right_mean**2, 0.0) ** 0.5,
    }


def load_visual_input(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "vision_input" not in payload:
            raise ValueError(f"{path}: PyTorch bundle lacks 'vision_input'")
        value = payload["vision_input"].detach().cpu().numpy()
    elif path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "vision_input" not in payload:
                raise ValueError(f"{path}: NPZ bundle lacks 'vision_input'")
            value = np.asarray(payload["vision_input"])
    else:
        raise ValueError(f"unsupported input format {path.suffix}; use .pt, .pth, .npy, or .npz")
    value = np.asarray(value, dtype=np.float16)
    if value.shape != (1, 2304, 588):
        raise ValueError(f"vision input must be [1,2304,588], got {value.shape}")
    return value


def create_float_visual_model(
    model_dir: Path,
    output_dir: Path,
    device: str,
) -> tuple[Any, Any]:
    import torch
    from leap_llm.apis.model.locateanything_vision import LocateAnythingVisionApi

    output_dir.mkdir(parents=True, exist_ok=True)
    api = LocateAnythingVisionApi(
        str(model_dir),
        str(output_dir),
        image_width=672,
        image_height=672,
        device=device,
        vit_core_num=[4],
        apply_hidden_rotation=True,
        export_only=True,
    )
    model = api.model.to(device=device, dtype=torch.float16).eval()
    model.compile_mode(False)
    return api, model


def resolve_scale_manifest(
    phase: str = "vision", override: Path | None = None
) -> Path:
    if override is not None:
        manifest = override.expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        return manifest
    calibration_root = REPO_ROOT / "workspace" / "calibration" / "current"
    candidates = sorted(calibration_root.glob("**/calibration_scale_manifest.json"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "calibration_scale_manifest.json was not found under "
            f"{calibration_root}; run observer replay first"
        )
    choices = ", ".join(str(path) for path in candidates)
    raise ValueError(
        "multiple calibration scale manifests were found; pass --scale-manifest: "
        f"{choices}"
    )


def release_language_graphs(config_path: Path = DEFAULT_CONFIG) -> tuple[str, ...]:
    """Return the ordered Language graph catalog fixed by compiler/config.yaml."""

    from compiler.quantize import load_config

    config = load_config(config_path)
    return tuple(str(graph) for graph in config["language"]["graphs"])


def validate_language_hbm_catalog(
    actual: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    if actual == expected:
        return
    missing = [graph for graph in expected if graph not in actual]
    unexpected = [graph for graph in actual if graph not in expected]
    details = []
    if missing:
        details.append(f"missing={missing}")
    if unexpected:
        details.append(f"unexpected={unexpected}")
    if not missing and not unexpected:
        details.append("catalog order differs")
    raise ValueError(
        "Language release HBM graph catalog must exactly match the ordered "
        f"compiler configuration ({', '.join(details)}); "
        f"expected={list(expected)}, found={list(actual)}"
    )


def restore_calibration_scales(
    model: Any, manifest_path: Path, phase: str = "vision"
) -> dict[str, Any]:
    from leap_llm.apis.calibration.locateanything_replay import apply_scale_manifest

    return apply_scale_manifest(model, manifest_path, phase)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    inputs_path = run_dir / manifest["inputs_file"]
    if sha256(inputs_path) != manifest["inputs_sha256"]:
        raise ValueError(f"input bundle hash mismatch: {inputs_path}")
    inputs = load_npz(inputs_path)
    if describe_arrays(inputs) != manifest["inputs"]:
        raise ValueError(f"input bundle metadata mismatch: {inputs_path}")
    return manifest, inputs


def tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.type.shape)


def tensor_dtype(tensor: Any) -> np.dtype:
    return np.dtype(tensor.type.np_dtype)


def load_artifact(kind: str, path: Path, graph: str) -> tuple[Any, list[Any], list[Any]]:
    import hbdk4.compiler as hb

    if kind == "bc":
        module = hb.load(str(path))
        functions = {str(item.name): item for item in module.functions}
        if graph not in functions:
            raise ValueError(f"BC lacks graph {graph}; found {sorted(functions)}")
        function = functions[graph]
        return function, list(function.inputs), list(function.outputs)
    hbm = hb.Hbm(str(path))
    graphs = {str(item.name): item for item in hbm.graphs}
    if graph not in graphs:
        raise ValueError(f"HBM lacks graph {graph}; found {sorted(graphs)}")
    function = graphs[graph]
    return function, list(function.inputs), list(function.outputs)


def simulator_snapshot() -> tuple[int, int, float] | None:
    if os.name != "posix" or not hasattr(os, "getpgrp"):
        return None
    try:
        result = subprocess.run(
            ["ps", "-eo", "pgid=,comm=,pcpu="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    process_group = os.getpgrp()
    qemu = workers = 0
    cpu = 0.0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            group = int(fields[0])
            process_cpu = float(fields[2])
        except ValueError:
            continue
        if group != process_group:
            continue
        command = fields[1]
        if command.startswith("qemu-system"):
            qemu += 1
        elif command.startswith("hbcm-module"):
            workers += 1
        else:
            continue
        cpu += process_cpu
    return qemu, workers, cpu


class SimulatorHeartbeat:
    def __init__(self, interval: int) -> None:
        self.interval = interval
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "SimulatorHeartbeat":
        print("[hbm] progress=unknown; waiting for the simulator", flush=True)
        if self.interval > 0:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            elapsed = int(time.monotonic() - self.started)
            snapshot = simulator_snapshot()
            details = ""
            if snapshot is not None:
                qemu, workers, cpu = snapshot
                details = f" qemu={qemu} workers={workers} host_cpu={cpu:.1f}%"
            print(
                f"[hbm] elapsed={elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:"
                f"{elapsed % 60:02d} progress=unknown{details}",
                flush=True,
            )


class OperationHeartbeat:
    def __init__(self, label: str, interval: int = 30) -> None:
        self.label = label
        self.interval = interval
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "OperationHeartbeat":
        if self.interval > 0:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            elapsed = int(time.monotonic() - self.started)
            print(
                f"[bc] {self.label} elapsed={elapsed // 60:02d}:{elapsed % 60:02d} "
                "status=running",
                flush=True,
            )


def detect_hbm_backend(
    machine: str | None = None,
    hobot_root: Path = Path("/usr/hobot"),
) -> str:
    architecture = (machine or platform.machine()).lower()
    if architecture in {"aarch64", "arm64"} and hobot_root.is_dir():
        return "s600_bpu"
    return "hbdk_x86_simulator"


def execute_s600_hbm(
    path: Path,
    graph: str,
    inputs: list[np.ndarray],
    run_dir: Path,
    trace: TraceRecorder | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if graph != "visual":
        raise ValueError("the S600 runner supports only the visual graph")
    if len(inputs) != 1 or inputs[0].shape != (1, 2304, 588):
        raise ValueError("S600 visual input must be one [1,2304,588] tensor")
    runner = REPO_ROOT / "deploy" / "run_vision_hbm.sh"
    board_input = run_dir / "vision_input.f16.bin"
    board_output = run_dir / "s600_vision_output.f16.bin"
    atomic_binary(board_input, inputs[0].astype(np.float16, copy=False))
    command = [
        str(runner), "--model", str(path), "--input", str(board_input),
        "--output", str(board_output),
    ]
    if not os.access(runner, os.X_OK):
        command.insert(0, "sh")
    subprocess.run(command, check=True)
    output = np.fromfile(board_output, dtype=np.dtype("<f2"))
    expected = int(np.prod(VISION_OUTPUT_SHAPE))
    if output.size != expected:
        raise ValueError(f"S600 output has {output.size} values, expected {expected}")
    output = output.reshape(VISION_OUTPUT_SHAPE)
    return [output], {
        "backend": "s600_bpu",
        "artifact": {"kind": "hbm", "path": str(path.resolve())},
        "board_output": {"path": str(board_output.resolve()), "bytes": board_output.stat().st_size},
        "output_names": ["output_000"],
    }


class S600VisionSession:
    PROTOCOL = "LAHBM/1"
    LOAD_TIMEOUT_SECONDS = 300
    INFERENCE_TIMEOUT_SECONDS = 600

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        self.binary = REPO_ROOT / "deploy" / "build" / "vision_hbm_runner"
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.inferences = 0
        self.load_seconds = 0.0
        self.inference_ms: list[float] = []
        self.request_id = 0

    def start(self) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"persistent S600 runner not built: {self.binary}")
        environment = os.environ.copy()
        environment.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                [str(self.binary), "--model", str(self.model_path), "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=os.name == "posix",
            )
            self.reader = threading.Thread(target=self._read_output, daemon=True)
            self.reader.start()
            deadline = time.monotonic() + self.LOAD_TIMEOUT_SECONDS
            while True:
                line = self._readline_until(deadline, "load")
                if line == f"{self.PROTOCOL}\tREADY\tvisual":
                    self.load_seconds = time.monotonic() - started
                    print(
                        f"[hbm] persistent S600 session ready; "
                        f"model_load_seconds={self.load_seconds:.3f}",
                        flush=True,
                    )
                    return
                print(f"[hbm:init] {line}", flush=True)
        except BaseException:
            self.close(graceful=False)
            raise

    def run(self, input_path: Path, output_path: Path) -> float:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("S600 session is not running")
        self.request_id += 1
        request_id = str(self.request_id)
        request = (
            f"{self.PROTOCOL}\tRUN\t{request_id}\t"
            f"{input_path.resolve()}\t{output_path.resolve()}"
        )
        if any(character in request for character in ("\n", "\r")):
            raise ValueError("S600 request paths cannot contain newlines")
        if any("\t" in str(path.resolve()) for path in (input_path, output_path)):
            raise ValueError("S600 request paths cannot contain tabs")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        output_path.with_name(output_path.name + ".tmp").unlink(missing_ok=True)
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.INFERENCE_TIMEOUT_SECONDS
        while True:
            line = self._readline_until(deadline, "inference")
            if line.startswith(f"{self.PROTOCOL}\tRESULT\t"):
                fields = line.split("\t")
                if len(fields) != 5 or fields[2] != request_id:
                    raise RuntimeError(f"invalid S600 result frame: {line}")
                elapsed_ms = float(fields[3])
                expected_bytes = int(np.prod(VISION_OUTPUT_SHAPE)) * 2
                if int(fields[4]) != expected_bytes:
                    raise RuntimeError(f"invalid S600 output size in frame: {line}")
                self.inferences += 1
                self.inference_ms.append(elapsed_ms)
                return elapsed_ms
            if line.startswith(f"{self.PROTOCOL}\tERROR\t"):
                raise RuntimeError(f"S600 runner failed: {line}")

    def close(self, graceful: bool = True) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if graceful and process.stdin is not None and process.poll() is None:
                process.stdin.write(f"{self.PROTOCOL}\tQUIT\n")
                process.stdin.flush()
            process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._terminate(process)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def _read_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            self.lines.put(None)
            return
        try:
            for line in process.stdout:
                self.lines.put(line.rstrip("\r\n"))
        finally:
            self.lines.put(None)

    def _readline(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            process = self.process
            if process is not None:
                self._terminate(process)
            raise TimeoutError(f"S600 runner did not respond within {timeout}s") from error
        if line is None:
            code = self.process.poll() if self.process is not None else None
            raise RuntimeError(f"S600 runner exited before responding, code={code}")
        return line

    def _readline_until(self, deadline: float, operation: str) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process = self.process
            if process is not None:
                self._terminate(process)
            raise TimeoutError(f"S600 {operation} exceeded its absolute timeout")
        return self._readline(remaining)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> "S600VisionSession":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _capture_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if str(tensor.dtype).endswith("bfloat16"):
            tensor = tensor.float()
        return np.array(tensor.numpy(), copy=True, order="C")
    return None


class FloatActivationCapture:
    def __init__(self, model: Any, level: str = "boundary") -> None:
        if level not in {"boundary", "deep"}:
            raise ValueError(f"unsupported capture level: {level}")
        self.model = model
        self.level = level
        self.entries: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def __enter__(self) -> "FloatActivationCapture":
        semantic_boundaries = re.compile(
            r"^(?:patch_embed|final_layernorm|merger|blocks\.\d+)$"
        )

        def hook(name: str, type_name: str):
            def capture(_module: Any, _inputs: Any, outputs: Any) -> None:
                for tensor_path, value in iter_tensors(outputs, "output"):
                    array = _capture_array(value)
                    if array is None:
                        continue
                    group, operation = semantic_location("float", "torch_module", name)
                    self.entries.append(
                        {
                            "name": canonical_name(name),
                            "type": type_name,
                            "tensor_path": tensor_path,
                            "shape": list(array.shape),
                            "semantic_group": group,
                            "semantic_operation": operation,
                            "array": array,
                        }
                    )
            return capture

        for name, module in self.model.named_modules():
            if not name:
                continue
            is_leaf = not any(module.children())
            is_boundary = bool(semantic_boundaries.fullmatch(name))
            if self.level == "boundary" and not is_boundary:
                continue
            if self.level == "deep" and not is_leaf and not is_boundary:
                continue
            self.handles.append(module.register_forward_hook(hook(name, type(module).__name__)))
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()

    def append_final(self, output: np.ndarray) -> None:
        self.entries.append(
            {
                "name": "visual",
                "type": "LocateAnything",
                "tensor_path": "output",
                "shape": list(output.shape),
                "semantic_group": "FINAL OUTPUT",
                "semantic_operation": "VISUAL OUTPUT",
                "array": np.ascontiguousarray(output),
            }
        )


class BCActivationCapture:
    def __init__(self, references: list[dict[str, Any]]) -> None:
        self.references = {
            (entry["name"], tuple(entry["shape"])): entry for entry in references
        }
        self.reference_names = {entry["name"] for entry in references}
        self.candidates: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
        self.sequence = 0
        self.callback_comparison_seconds = 0.0

    def callback(
        self,
        op: Any,
        results: Any,
        raw_name: str,
        name: str,
        sequence: int,
    ) -> bool:
        type_name = str(getattr(op, "type", type(op).__name__))
        for tensor_path, value in iter_tensors(results, "output"):
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            key = (name, tuple(int(item) for item in shape))
            reference = self.references.get(key)
            if reference is None:
                continue
            array = _capture_array(value)
            if array is None:
                continue
            compare_started = time.monotonic()
            comparison = compare_arrays(reference["array"], array)
            self.callback_comparison_seconds += time.monotonic() - compare_started
            self.candidates[key] = {
                "sequence": sequence,
                "name": raw_name,
                "canonical_name": name,
                "type": type_name,
                "tensor_path": tensor_path,
                "comparison": comparison,
            }
        return True


class BCCallbackDispatcher:
    def __init__(self) -> None:
        self.active: BCActivationCapture | None = None
        self.name_cache: dict[str, str] = {}

    def __call__(self, op: Any, results: Any, operands: Any) -> bool:
        if self.active is None:
            return True
        if str(getattr(op, "type", "")) == "func.func":
            return True
        raw_name = str(getattr(op, "name", "<unnamed>"))
        name = self.name_cache.get(raw_name)
        if name is None:
            name = canonical_name(raw_name)
            self.name_cache[raw_name] = name
        sequence = self.active.sequence
        self.active.sequence += 1
        if name not in self.active.reference_names:
            return True
        return self.active.callback(op, results, raw_name, name, sequence)


def compare_activation_captures(
    references: list[dict[str, Any]],
    capture: BCActivationCapture,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence, reference in enumerate(references):
        key = (reference["name"], tuple(reference["shape"]))
        candidate = capture.candidates.get(key)
        row: dict[str, Any] = {
            "reference_sequence": sequence,
            "semantic_group": reference["semantic_group"],
            "semantic_operation": reference["semantic_operation"],
            "module": reference["name"],
            "reference_type": reference["type"],
            "shape": reference["shape"],
        }
        if candidate is None:
            row["status"] = "unmatched"
        else:
            row.update(
                status="matched",
                candidate_sequence=candidate["sequence"],
                candidate_type=candidate["type"],
                comparison=candidate["comparison"],
            )
        rows.append(row)
    return rows


def execute_loaded_bc(
    artifact: Any,
    dispatcher: BCCallbackDispatcher,
    input_descriptors: list[Any],
    output_descriptors: list[Any],
    value: np.ndarray,
    capture: BCActivationCapture,
) -> np.ndarray:
    descriptor = input_descriptors[0]
    feed = {str(descriptor.name): value.astype(tensor_dtype(descriptor), copy=False)}
    dispatcher.active = capture
    try:
        raw = artifact.feed(inputs=feed)
    finally:
        dispatcher.active = None
    return np.asarray(raw[str(output_descriptors[0].name)])


def execute_loaded_bc_final(
    artifact: Any,
    input_descriptors: list[Any],
    output_descriptors: list[Any],
    value: np.ndarray,
) -> np.ndarray:
    descriptor = input_descriptors[0]
    feed = {str(descriptor.name): value.astype(tensor_dtype(descriptor), copy=False)}
    raw = artifact.feed(inputs=feed)
    return np.asarray(raw[str(output_descriptors[0].name)])


class ModuleMetricAggregator:
    METRICS = (
        "cosine", "relative_l2", "mae", "rmse", "max_abs", "top1_agreement",
        "top1_flip_rate", "topk_overlap", "reference_top1_margin",
        "candidate_top1_margin", "reference_top1_rank_in_candidate",
        "candidate_top1_rank_in_reference",
        "structure_agreement", "float_valid", "candidate_valid",
        "float_fallback", "candidate_fallback", "coordinate_token_exact",
        "coordinate_mae", "coordinate_max_abs", "pixel_mae", "pixel_max_abs",
        "box_iou", "point_distance_pixels",
        "comparable", "token_exact", "token_delta", "token_abs_delta",
        "pixel_delta", "pixel_abs_delta", "float_token_top4_hit",
        "float_token_rank_in_quantized",
        "float_token_probability_in_quantized",
        "selected_minus_float_logit_margin",
        "top1_minus_float_logit_margin",
    )

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}

    def add(self, stage: str, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            key = (stage, entry["module"], tuple(entry["shape"]))
            aggregate = self.rows.setdefault(
                key,
                {
                    "stage": stage,
                    "reference_sequence": entry["reference_sequence"],
                    "semantic_group": entry["semantic_group"],
                    "semantic_operation": entry["semantic_operation"],
                    "module": entry["module"],
                    "shape": entry["shape"],
                    "samples": 0,
                    "matched": 0,
                    "metrics": {name: [] for name in self.METRICS},
                },
            )
            aggregate["samples"] += 1
            if entry.get("status") != "matched":
                continue
            aggregate["matched"] += 1
            comparison = entry["comparison"]
            for name in self.METRICS:
                value = comparison.get(name)
                if value is not None:
                    aggregate["metrics"][name].append(float(value))

    def restore_sample(self, sample: dict[str, Any]) -> None:
        for section in ("intermediate", "diagnostic"):
            for stage, entries in sample.get(section, {}).items():
                self.add(stage, entries)

    def records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for aggregate in sorted(
            self.rows.values(), key=lambda item: (item["stage"], item["reference_sequence"])
        ):
            record = {
                key: aggregate[key]
                for key in (
                    "stage", "reference_sequence", "semantic_group", "semantic_operation",
                    "module", "shape", "samples", "matched",
                )
            }
            record["match_rate"] = aggregate["matched"] / aggregate["samples"]
            record["metrics"] = {}
            for metric, values in aggregate["metrics"].items():
                if not values:
                    continue
                array = np.asarray(values, dtype=np.float64)
                record["metrics"][metric] = {
                    "count": int(array.size),
                    "mean": float(array.mean()),
                    "min": float(array.min()),
                    "p05": float(np.percentile(array, 5)),
                    "median": float(np.median(array)),
                    "p95": float(np.percentile(array, 95)),
                    "max": float(array.max()),
                }
            records.append(record)
        return records


def write_batch_csv(path: Path, aggregator: ModuleMetricAggregator) -> None:
    columns = [
        "stage", "reference_sequence", "semantic_group", "semantic_operation",
        "module", "shape", "samples", "matched", "match_rate",
    ]
    for metric in ModuleMetricAggregator.METRICS:
        columns.extend(
            f"{metric}_{suffix}"
            for suffix in ("count", "mean", "min", "p05", "median", "p95", "max")
        )
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for aggregate in aggregator.records():
            row = {
                key: aggregate[key]
                for key in (
                    "stage", "reference_sequence", "semantic_group", "semantic_operation",
                    "module", "samples", "matched",
                )
            }
            row["shape"] = "x".join(str(value) for value in aggregate["shape"])
            row["match_rate"] = aggregate["matched"] / aggregate["samples"]
            for metric, statistics in aggregate["metrics"].items():
                row.update(
                    {
                        f"{metric}_{name}": value
                        for name, value in statistics.items()
                    }
                )
            writer.writerow(row)
    os.replace(temporary, path)


StageAction = Callable[[], tuple[list[np.ndarray], dict[str, Any]]]
TracedAction = Callable[[TraceRecorder | None], tuple[list[np.ndarray], dict[str, Any]]]


def traced_action(
    run_dir: Path,
    stage: str,
    mode: str,
    action: TracedAction,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if mode == "off":
        return action(None)
    recorder = TraceRecorder(run_dir, stage, mode)
    with recorder:
        outputs, details = action(recorder)
    details["trace"] = recorder.summary
    return outputs, details


def execute_stage(
    run_dir: Path,
    name: str,
    metadata: dict[str, Any],
    action: StageAction,
) -> dict[str, Any]:
    state_path = run_dir / f"{name}.json"
    output_path = run_dir / f"{name}.npz"
    record = {
        "schema_version": 2,
        "stage": name,
        "status": "running",
        "started_at": utc_now(),
        **metadata,
    }
    atomic_json(state_path, record)
    started = time.monotonic()
    try:
        outputs, details = action()
        output_names = details.pop("output_names", None)
        atomic_npz(output_path, outputs)
    except Exception as error:
        record.update(
            status="failed",
            finished_at=utc_now(),
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(state_path, record)
        raise
    record.update(
        status="completed",
        finished_at=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        output_file=output_path.name,
        output_sha256=sha256(output_path),
        outputs=describe_arrays(outputs, output_names),
        **details,
    )
    atomic_json(state_path, record)
    print(f"[{name}] completed in {record['elapsed_seconds']:.2f}s", flush=True)
    print(f"[{name}] output={output_path}", flush=True)
    print(f"[{name}] state={state_path}", flush=True)
    return record


def run_float(args: argparse.Namespace) -> int:
    manifest, inputs = prepare_run(args)
    metadata = {
        "graph": manifest["graph"],
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
    }
    def action(trace: TraceRecorder | None) -> tuple[list[np.ndarray], dict[str, Any]]:
        if trace is not None:
            trace.record("input", "model_input", "ModelInput", (), inputs)
        outputs = build_float_reference(
            args.model_dir,
            manifest["graph"],
            inputs,
            args.run_dir / "work",
            args.device,
            trace,
        )
        if trace is not None:
            trace.record(
                manifest["graph"],
                "float_output",
                "LocateAnything",
                (),
                outputs,
            )
        return outputs, {}

    execute_stage(
        args.run_dir,
        "float",
        metadata,
        lambda: traced_action(args.run_dir, "float", args.trace, action),
    )
    return 0


def run_bc(args: argparse.Namespace) -> int:
    manifest, inputs = load_run(args.run_dir)
    stage_name = BC_MODES[args.mode]
    execute_stage(
        args.run_dir,
        stage_name,
        {"graph": manifest["graph"]},
        lambda: traced_action(
            args.run_dir,
            stage_name,
            args.trace,
            lambda trace: execute_artifact(
                "bc", args.artifact, manifest["graph"], inputs, trace=trace
            ),
        ),
    )
    return 0


def run_hbm(args: argparse.Namespace) -> int:
    manifest, inputs = load_run(args.run_dir)
    backend = detect_hbm_backend()

    def action(trace: TraceRecorder | None) -> tuple[list[np.ndarray], dict[str, Any]]:
        if backend == "s600_bpu":
            return execute_s600_hbm(
                args.artifact,
                manifest["graph"],
                inputs,
                args.run_dir,
                trace,
            )
        print(
            f"[hbm] backend=hbdk_x86_simulator machine={platform.machine()}",
            flush=True,
        )
        outputs, details = execute_artifact(
            "hbm",
            args.artifact,
            manifest["graph"],
            inputs,
            heartbeat=args.heartbeat,
            trace=trace,
        )
        details.update(backend=backend, machine=platform.machine())
        return outputs, details

    execute_stage(
        args.run_dir,
        "hbm",
        {"graph": manifest["graph"], "backend": backend},
        lambda: traced_action(
            args.run_dir,
            "hbm",
            args.trace,
            action,
        ),
    )
    return 0


def run_batch(args: argparse.Namespace) -> int:
    import torch

    records = read_jsonl(args.input_manifest)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if not records:
        raise ValueError(f"no records found in {args.input_manifest}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    details_dir = args.run_dir / "samples"
    details_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.run_dir / "report.csv"
    summary_path = args.run_dir / "report.json"
    aggregator = ModuleMetricAggregator()
    manifest_sha256 = sha256(args.input_manifest)
    completed: set[str] = set()
    for path in sorted(details_dir.glob("*.json")):
        sample = read_json(path)
        if sample.get("status") == "completed":
            completed.add(sample["bundle_id"])
            aggregator.restore_sample(sample)

    api, model = create_float_visual_model(
        args.model_dir, args.run_dir / "work" / "float_model", args.device
    )
    exported, exported_inputs, exported_outputs = load_artifact(
        "bc", args.exported_bc, "visual"
    )
    converted, converted_inputs, converted_outputs = load_artifact(
        "bc", args.converted_bc, "visual"
    )
    exported_dispatcher = BCCallbackDispatcher()
    converted_dispatcher = BCCallbackDispatcher()
    exported.register_callback(exported_dispatcher)
    converted.register_callback(converted_dispatcher)

    def write_summary(status: str, current: str | None = None) -> None:
        atomic_json(
            summary_path,
            {
                "schema_version": 1,
                "status": status,
                "input_manifest": str(args.input_manifest.resolve()),
                "input_manifest_sha256": manifest_sha256,
                "expected_samples": len(records),
                "completed_samples": len(completed),
                "current_sample": current,
                "details_directory": str(details_dir.resolve()),
                "aggregate_csv": str(csv_path.resolve()),
                "model_dir": str(args.model_dir.resolve()),
                "exported_bc": str(args.exported_bc.resolve()),
                "converted_bc": str(args.converted_bc.resolve()),
                "updated_at": utc_now(),
            },
        )

    write_summary("running")
    for index, record in enumerate(records, start=1):
        bundle_id = str(record["bundle_id"])
        if bundle_id in completed:
            print(f"[batch] {index}/{len(records)} skip={bundle_id}", flush=True)
            continue
        tensor_path = (args.input_manifest.parent / record["tensor_file"]).resolve()
        detail_path = details_dir / f"{_safe_batch_name(bundle_id)}.json"
        started = time.monotonic()
        vision_input = float_output = exported_output = converted_output = None
        float_capture = exported_rows = converted_rows = detail = None
        write_summary("running", bundle_id)
        try:
            vision_input = load_visual_input(tensor_path)
            with FloatActivationCapture(model) as float_capture, torch.no_grad():
                float_tensor = model(torch.from_numpy(vision_input).to(args.device))
            float_output = float_tensor.detach().float().cpu().numpy()
            del float_tensor

            exported_capture = BCActivationCapture(float_capture.entries)
            exported_output = execute_loaded_bc(
                exported, exported_dispatcher, exported_inputs, exported_outputs,
                vision_input, exported_capture
            )
            exported_rows = compare_activation_captures(
                float_capture.entries, exported_capture
            )
            exported_rows.append(
                _final_batch_row(float_output, exported_output, len(float_capture.entries))
            )
            del exported_capture

            converted_capture = BCActivationCapture(float_capture.entries)
            converted_output = execute_loaded_bc(
                converted, converted_dispatcher, converted_inputs, converted_outputs,
                vision_input, converted_capture
            )
            converted_rows = compare_activation_captures(
                float_capture.entries, converted_capture
            )
            converted_rows.append(
                _final_batch_row(float_output, converted_output, len(float_capture.entries))
            )
            del converted_capture

            detail = {
                "schema_version": 1,
                "status": "completed",
                "sample_index": index - 1,
                "bundle_id": bundle_id,
                "task": record.get("task"),
                "source": record.get("source"),
                "image": record.get("image"),
                "image_sha256": record.get("image_sha256"),
                "tensor_file": str(tensor_path),
                "tensor_sha256": record.get("tensor_sha256"),
                "intermediate": {
                    "float_to_exported_bc": exported_rows,
                    "float_to_converted_bc": converted_rows,
                },
                "elapsed_seconds": time.monotonic() - started,
                "finished_at": utc_now(),
            }
            atomic_json(detail_path, detail)
            aggregator.add("float_to_exported_bc", exported_rows)
            aggregator.add("float_to_converted_bc", converted_rows)
            completed.add(bundle_id)
            write_batch_csv(csv_path, aggregator)
            write_summary("running")
            print(
                f"[batch] {index}/{len(records)} completed={bundle_id} "
                f"elapsed={detail['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            atomic_json(
                detail_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "sample_index": index - 1,
                    "bundle_id": bundle_id,
                    "tensor_file": str(tensor_path),
                    "error": f"{type(error).__name__}: {error}",
                    "finished_at": utc_now(),
                },
            )
            write_summary("failed", bundle_id)
            raise
        finally:
            if float_capture is not None:
                float_capture.entries.clear()
            vision_input = float_output = exported_output = converted_output = None
            float_capture = exported_rows = converted_rows = detail = None
            gc.collect()
            torch.cuda.empty_cache()

    write_batch_csv(csv_path, aggregator)
    write_summary("completed")
    del model, api
    print(f"[batch] completed={len(completed)}/{len(records)}", flush=True)
    print(f"[batch] summary={summary_path}", flush=True)
    print(f"[batch] csv={csv_path}", flush=True)
    print(f"[batch] samples={details_dir}", flush=True)
    return 0


def _safe_batch_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def _final_batch_row(
    reference: np.ndarray,
    candidate: np.ndarray,
    sequence: int,
) -> dict[str, Any]:
    return {
        "reference_sequence": sequence,
        "semantic_group": "FINAL OUTPUT",
        "semantic_operation": "VISUAL OUTPUT",
        "module": "visual",
        "reference_type": "LocateAnything",
        "shape": list(reference.shape),
        "status": "matched",
        "candidate_type": "BCGraphOutput",
        "comparison": compare_arrays(reference, candidate),
    }


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = value.astype(np.float64, copy=False)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "nonzero": bool(np.any(array != 0)),
        "finite": bool(np.isfinite(array).all()),
    }


def detect_float_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def input_set_sha256(input_index: Path) -> str | None:
    if not input_index.is_file():
        return None
    inputs = read_json(input_index).get("inputs", [])
    identity = sorted(
        (
        {"id": str(item["id"]), "sha256": item.get("sha256")}
        for item in inputs
        ),
        key=lambda item: item["id"],
    )
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stage_identity(
    stage_dir: Path,
    model_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    input_index = stage_dir.parent / "inputs.json"
    identity = {
        "model": str(model_path),
        "model_sha256": path_sha256(model_path),
        "input_set_sha256": input_set_sha256(input_index),
    }
    identity.update(run_metadata or {})
    return identity


def validate_stage_identity(
    stage_dir: Path,
    model_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = stage_identity(stage_dir, model_path, run_metadata)
    state_path = stage_dir / "stage.json"
    if not state_path.is_file():
        samples_dir = stage_dir / "samples"
        if samples_dir.is_dir() and any(samples_dir.glob("*.json")):
            raise ValueError(
                f"{stage_dir} contains partial results without model identity; "
                "choose another --output_dir"
            )
        return current
    previous = read_json(state_path)
    if Path(previous["model"]).resolve() != Path(current["model"]):
        raise ValueError(
            f"{stage_dir} contains results for {previous['model']}; "
            f"choose another --output_dir for {current['model']}"
        )
    samples_dir = stage_dir / "samples"
    has_samples = samples_dir.is_dir() and any(samples_dir.glob("*.json"))
    immutable_keys = (
        "model_sha256", "input_set_sha256", "phase", "level",
        "selected_ids", "selected_ids_sha256", "float_reference_sha256",
        "scale_manifest_sha256", "emulation_version", "artifact_set_sha256",
    )
    for key in immutable_keys:
        if key not in current:
            continue
        value = current[key]
        if key not in previous:
            if has_samples:
                raise ValueError(
                    f"{stage_dir} has samples but its stage identity is missing {key}; "
                    "choose another --output_dir"
                )
            continue
        if previous[key] != value:
            raise ValueError(f"{stage_dir} identity mismatch for {key}; choose another --output_dir")
    return current


def begin_stage(
    stage_dir: Path,
    stage: str,
    model_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = validate_stage_identity(stage_dir, model_path, run_metadata)
    state_path = stage_dir / "stage.json"
    previous = read_json(state_path) if state_path.is_file() else {}
    atomic_json(
        state_path,
        {
            "schema_version": 2,
            "stage": stage,
            "status": "running",
            **identity,
            "started_at": previous.get("started_at", utc_now()),
            "updated_at": utc_now(),
        },
    )
    return identity


def valid_completed_sample(
    sample_path: Path,
    output_path: Path,
    sample_id: str,
    required_field: str | None = None,
    *,
    input_sha256: str | None = None,
    phase: str | None = None,
    capture_level: str | None = None,
) -> dict[str, Any] | None:
    if not sample_path.is_file() or not output_path.is_file():
        return None
    try:
        sample = read_json(sample_path)
    except (OSError, json.JSONDecodeError):
        return None
    if sample.get("status") != "completed" or sample.get("id") != sample_id:
        return None
    if required_field is not None and not sample.get(required_field):
        return None
    if input_sha256 is not None and sample.get("input_sha256") != input_sha256:
        return None
    if phase is not None and sample.get("phase") != phase:
        return None
    if capture_level is not None and sample.get("capture_level") != capture_level:
        return None
    expected = sample.get("output_sha256")
    if not isinstance(expected, str) or sha256(output_path) != expected:
        return None
    return sample


def sample_task(record: dict[str, Any]) -> str:
    task = str(record.get("task") or "").strip().lower()
    if task:
        return task
    match = re.match(r"^\d+-([a-z][a-z0-9_]*)-", str(record["id"]))
    return match.group(1) if match else "unknown"


def select_records(
    records: list[dict[str, Any]], nums: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError("no inputs are available")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("input ids must be unique")
    safe_ids = [_safe_batch_name(sample_id) for sample_id in ids]
    if len(safe_ids) != len(set(safe_ids)):
        raise ValueError("input ids collide after filename normalization")
    if nums is not None and (nums <= 0 or nums > len(records)):
        raise ValueError(
            f"--nums must be between 1 and the {len(records)} available inputs"
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(sample_task(record), []).append(record)
    grouped = {
        task: sorted(task_records, key=lambda item: str(item["id"]))
        for task, task_records in sorted(grouped.items())
    }
    available_by_task = {task: len(task_records) for task, task_records in grouped.items()}

    if nums is None or nums == len(records):
        selected = list(records)
        policy = "all"
    else:
        quotas = {task: 0 for task in grouped}
        tasks = list(grouped)
        remaining = nums
        if nums >= len(tasks):
            for task in tasks:
                quotas[task] = 1
            remaining -= len(tasks)

        capacities = {
            task: len(grouped[task]) - quotas[task]
            for task in tasks
        }
        capacity_total = sum(capacities.values())
        remainders: list[tuple[int, str]] = []
        if remaining:
            for task in tasks:
                quotient, remainder = divmod(remaining * capacities[task], capacity_total)
                quotas[task] += quotient
                remainders.append((remainder, task))
            left = nums - sum(quotas.values())
            for _remainder, task in sorted(remainders, key=lambda item: (-item[0], item[1])):
                if not left:
                    break
                if quotas[task] < len(grouped[task]):
                    quotas[task] += 1
                    left -= 1
        if sum(quotas.values()) != nums:
            raise RuntimeError("could not allocate the requested input count")

        selected_ids: set[str] = set()
        for task, quota in quotas.items():
            task_records = grouped[task]
            for index in range(quota):
                position = ((2 * index + 1) * len(task_records)) // (2 * quota)
                selected_ids.add(str(task_records[position]["id"]))
        selected = sorted(
            (record for record in records if str(record["id"]) in selected_ids),
            key=lambda record: str(record["id"]),
        )
        policy = "deterministic_stratified"

    selected_by_task: dict[str, int] = {task: 0 for task in grouped}
    for record in selected:
        selected_by_task[sample_task(record)] += 1
    coverage = {
        "selection_policy": policy,
        "available_count": len(records),
        "selected_count": len(selected),
        "available_by_task": available_by_task,
        "selected_by_task": selected_by_task,
    }
    return selected, coverage


def selection_metadata(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = [str(record["id"]) for record in records]
    digest = hashlib.sha256(
        json.dumps(sorted(selected_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "phase": args.phase,
        "level": args.level,
        "requested_nums": args.nums if args.nums is not None else "all",
        "selected_ids": selected_ids,
        "selected_ids_sha256": digest,
        "coverage": coverage,
    }


def float_reference_sha256(
    output_dir: Path,
    float_stage: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    outputs = []
    for record in records:
        sample_path = (
            output_dir / "float" / "samples" / f"{_safe_batch_name(record['id'])}.json"
        )
        sample = read_json(sample_path)
        outputs.append({"id": str(record["id"]), "output_sha256": sample["output_sha256"]})
    payload = {
        "model": float_stage.get("model"),
        "model_sha256": float_stage.get("model_sha256"),
        "input_set_sha256": float_stage.get("input_set_sha256"),
        "phase": float_stage.get("phase"),
        "level": float_stage.get("level"),
        "outputs": outputs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def activation_statistics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sequence": sequence,
            "semantic_group": entry["semantic_group"],
            "semantic_operation": entry["semantic_operation"],
            "module": entry["name"],
            "type": entry["type"],
            "tensor_path": entry["tensor_path"],
            "statistics": _array_summary(entry["array"]),
        }
        for sequence, entry in enumerate(entries)
    ]


def quantization_metric_rows(
    entries: list[dict[str, Any]],
    *,
    sequence_start: int,
    reference_type: str,
    candidate_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, entry in enumerate(entries):
        name = str(entry["module"])
        comparison = entry["comparison"]
        group, operation = semantic_location(
            QUANTIZED_EAGER_STAGE, "torch_module", name
        )
        shape = comparison.get("shape") or comparison.get("reference_shape") or []
        row = {
            "reference_sequence": sequence_start + offset,
            "semantic_group": group,
            "semantic_operation": operation,
            "module": name,
            "shape": shape,
            "status": (
                "matched" if comparison.get("status") == "compared"
                else comparison.get("status", "unmatched")
            ),
            "reference_type": reference_type,
            "candidate_type": candidate_type,
            "comparison": comparison,
        }
        details = {
            key: value
            for key, value in entry.items()
            if key not in {"module", "comparison"}
        }
        if details:
            row["quantization"] = details
        rows.append(row)
    return rows


def run_quantized_eager_sample(
    model: Any,
    emulator: QuantizationEmulator,
    boundaries: BoundaryCapture,
    vision_input: np.ndarray,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    value = torch.from_numpy(vision_input).to(device)
    boundary_entries: list[dict[str, Any]] = []
    operator_entries: list[dict[str, Any]] = []
    try:
        if boundaries.enabled:
            emulator.set_enabled(False)
            boundaries.begin_reference()
            with torch.no_grad():
                reference = model(value)
            del reference

        boundaries.begin_candidate()
        emulator.set_enabled(True)
        with torch.no_grad():
            candidate = model(value)
        output = candidate.detach().float().cpu().numpy()
        del candidate
        boundary_entries = list(boundaries.rows)
        operator_entries = list(emulator.rows)
    finally:
        emulator.set_enabled(False)
        boundaries.finish_sample()
        del value

    boundary_rows = quantization_metric_rows(
        boundary_entries,
        sequence_start=0,
        reference_type="FloatEagerBoundary",
        candidate_type="QuantizedEagerBoundary",
    )
    operator_rows = quantization_metric_rows(
        operator_entries,
        sequence_start=0,
        reference_type="FloatOperator",
        candidate_type="QuantizedEagerOperator",
    )
    return output, boundary_rows, operator_rows


def run_float_final(model: Any, vision_input: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        tensor = model(torch.from_numpy(vision_input).to(device))
    output = tensor.detach().float().cpu().numpy()
    del tensor
    return output


def run_float_sample(
    model: Any,
    vision_input: np.ndarray,
    device: str,
    capture_level: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import torch

    with FloatActivationCapture(model, capture_level) as capture, torch.no_grad():
        tensor = model(torch.from_numpy(vision_input).to(device))
    output = tensor.detach().float().cpu().numpy()
    del tensor
    statistics = activation_statistics(capture.entries)
    capture.entries.clear()
    return output, statistics


def run_bc_sample(
    float_model: Any,
    artifact: Any,
    dispatcher: BCCallbackDispatcher,
    input_descriptors: list[Any],
    output_descriptors: list[Any],
    vision_input: np.ndarray,
    capture_level: str,
    heartbeat_label: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    import torch

    float_started = time.monotonic()
    with FloatActivationCapture(float_model, capture_level) as reference, torch.no_grad():
        float_tensor = float_model(torch.from_numpy(vision_input).to(next(float_model.parameters()).device))
    float_output = float_tensor.detach().float().cpu().numpy()
    del float_tensor
    float_seconds = time.monotonic() - float_started
    capture = BCActivationCapture(reference.entries)
    bc_started = time.monotonic()
    with OperationHeartbeat(heartbeat_label):
        candidate_output = execute_loaded_bc(
            artifact, dispatcher, input_descriptors, output_descriptors, vision_input, capture
        )
    bc_seconds = time.monotonic() - bc_started
    row_build_started = time.monotonic()
    rows = compare_activation_captures(reference.entries, capture)
    rows.append(_final_batch_row(float_output, candidate_output, len(reference.entries)))
    row_build_seconds = time.monotonic() - row_build_started
    callback_comparison_seconds = capture.callback_comparison_seconds
    comparison_seconds = callback_comparison_seconds + row_build_seconds
    reference.entries.clear()
    del float_output, capture, reference
    return candidate_output, rows, {
        "float_forward_seconds": float_seconds,
        "bc_feed_seconds": bc_seconds,
        "comparison_seconds": comparison_seconds,
        "callback_comparison_seconds": callback_comparison_seconds,
        "row_build_seconds": row_build_seconds,
    }


def language_metric_rows(
    entries: list[dict[str, Any]],
    *,
    reference_type: str,
    candidate_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence, entry in enumerate(entries):
        comparison = entry["comparison"]
        stage = str(entry.get("stage", "language"))
        module = str(entry["module"])
        rows.append({
            "reference_sequence": sequence,
            "semantic_group": stage.upper(),
            "semantic_operation": module,
            "module": f"{stage}/{module}",
            "shape": comparison.get("shape", []),
            "status": (
                "matched" if comparison.get("status") == "compared"
                else comparison.get("status", "unmatched")
            ),
            "reference_type": reference_type,
            "candidate_type": candidate_type,
            "comparison": comparison,
            "quantization": {
                key: value
                for key, value in entry.items()
                if key not in {"stage", "module", "comparison"}
            },
        })
    return rows


def named_npz_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: _array_summary(value) for name, value in arrays.items()}


def run_language_float_collection(args: argparse.Namespace) -> int:
    import torch

    device = detect_float_device()
    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    capture_level = CAPTURE_LEVELS[args.level]
    stage_dir = args.output_dir / "float"
    identity = begin_stage(stage_dir, "float", args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    api, model, rotation = create_language_model(
        args.model_path, stage_dir / "work" / "model", device
    )
    runner = LanguageEagerRunner(
        model,
        rotation,
        device,
        quantized=False,
        capture_boundaries=False,
        capture_operators=False,
    )
    completed = processed = resumed = 0
    started_all = time.monotonic()
    progress = progress_bar(records, "Language Float Prefill + PBD + AR")
    try:
        for index, record in enumerate(progress, 1):
            sample_id = str(record["id"])
            safe_id = _safe_batch_name(sample_id)
            sample_path = samples_dir / f"{safe_id}.json"
            output_path = outputs_dir / f"{safe_id}.npz"
            existing = valid_completed_sample(
                sample_path,
                output_path,
                sample_id,
                input_sha256=record.get("sha256"),
                phase=args.phase,
                capture_level=capture_level,
            )
            if existing is not None:
                completed += 1
                resumed += 1
                progress.set_postfix(sample=sample_id, resumed=resumed)
                continue
            sample_started = time.monotonic()
            payload = load_language_payload(Path(record["path"]))
            result = runner.run(payload)
            atomic_named_npz(output_path, result.outputs)
            elapsed = time.monotonic() - sample_started
            atomic_json(
                sample_path,
                {
                    "schema_version": 2,
                    "status": "completed",
                    "phase": args.phase,
                    "level": args.level,
                    "capture_level": capture_level,
                    "index": index - 1,
                    "id": sample_id,
                    "input": record,
                    "input_sha256": record.get("sha256"),
                    "output": str(output_path.resolve()),
                    "output_sha256": sha256(output_path),
                    "statistics": named_npz_statistics(result.outputs),
                    "executed_stages": list(result.outputs),
                    "elapsed_seconds": elapsed,
                    "finished_at": utc_now(),
                },
            )
            completed += 1
            processed += 1
            progress.set_postfix(
                sample=sample_id,
                elapsed=f"{elapsed:.1f}s",
                stages=len(result.outputs),
            )
            del payload, result
            if processed % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        progress.close()
        runner.close()
    print_phase_summary(
        "Language Float Prefill + PBD + AR",
        len(records),
        processed,
        resumed,
        started_all,
        stage_dir,
        level=args.level,
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": "float",
            "status": "completed",
            **identity,
            "device": device,
            "profile": {"chunk_size": 1024, "cache_len": 4096, "pbd_q": 6, "ar_q": 1},
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": completed,
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "updated_at": utc_now(),
        },
    )
    del model, api
    return 0


def run_language_quantized_eager_collection(args: argparse.Namespace) -> int:
    import torch

    device = detect_float_device()
    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    capture_level = CAPTURE_LEVELS[args.level]
    float_stage_path = args.output_dir / "float" / "stage.json"
    if not float_stage_path.is_file():
        raise FileNotFoundError(f"run --mode float first: {float_stage_path}")
    float_stage = read_json(float_stage_path)
    if float_stage.get("status") != "completed" or float_stage.get("phase") != "language":
        raise ValueError("run the matching Language Float stage to completion first")
    if Path(float_stage["model"]).resolve() != args.model_path.resolve():
        raise ValueError("Language Float and quantized-eager checkpoints differ")
    for record in records:
        safe_id = _safe_batch_name(str(record["id"]))
        if valid_completed_sample(
            args.output_dir / "float" / "samples" / f"{safe_id}.json",
            args.output_dir / "float" / "outputs" / f"{safe_id}.npz",
            str(record["id"]),
            input_sha256=record.get("sha256"),
            phase="language",
            capture_level=CAPTURE_LEVELS[str(float_stage["level"])],
        ) is None:
            raise ValueError(f"invalid Language Float reference for {record['id']}")

    scale_manifest = resolve_scale_manifest(
        "language", getattr(args, "scale_manifest", None)
    )
    metadata.update({
        "scale_manifest": str(scale_manifest),
        "scale_manifest_sha256": sha256(scale_manifest),
        "emulation_version": LANGUAGE_EMULATION_VERSION,
        "coordinate_audit_version": COORDINATE_AUDIT_VERSION,
    })
    stage_dir = args.output_dir / QUANTIZED_EAGER_STAGE
    identity = begin_stage(stage_dir, QUANTIZED_EAGER_STAGE, args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    aggregator = ModuleMetricAggregator()
    completed: set[str] = set()
    for record in records:
        sample_id = str(record["id"])
        safe_id = _safe_batch_name(sample_id)
        existing = valid_completed_sample(
            samples_dir / f"{safe_id}.json",
            outputs_dir / f"{safe_id}.npz",
            sample_id,
            required_field="coordinate_audit",
            input_sha256=record.get("sha256"),
            phase="language",
            capture_level=capture_level,
        )
        if existing is not None:
            completed.add(sample_id)
            aggregator.restore_sample(existing)

    api, model, rotation = create_language_model(
        args.model_path, stage_dir / "work" / "model", device
    )
    restored_scales = restore_calibration_scales(model, scale_manifest, "language")
    runner = LanguageEagerRunner(
        model,
        rotation,
        device,
        quantized=True,
        capture_boundaries=capture_level != "final",
        capture_operators=capture_level == "deep",
    )
    processed = resumed = 0
    started_all = time.monotonic()
    progress = progress_bar(records, "Language quantized eager Prefill + PBD + AR")
    try:
        for index, record in enumerate(progress, 1):
            sample_id = str(record["id"])
            safe_id = _safe_batch_name(sample_id)
            if sample_id in completed:
                resumed += 1
                progress.set_postfix(sample=sample_id, resumed=resumed)
                continue
            sample_started = time.monotonic()
            payload = load_language_payload(Path(record["path"]))
            result = runner.run(payload)
            coordinate_audit = runner.audit_coordinates(payload)
            output_path = outputs_dir / f"{safe_id}.npz"
            atomic_named_npz(output_path, result.outputs)
            main_rows = language_metric_rows(
                [*result.comparisons, *result.boundaries],
                reference_type="LanguageFloat",
                candidate_type="LanguageQuantizedEager",
            )
            main_rows.extend(coordinate_metric_rows(coordinate_audit))
            main_rows.extend(language_metric_rows(
                coordinate_audit.get("boundaries", []),
                reference_type="LanguageFloatCoordinateBoundary",
                candidate_type="LanguageQuantizedEagerCoordinateBoundary",
            ))
            operator_rows = language_metric_rows(
                result.operators,
                reference_type="FloatOperator",
                candidate_type="QuantizedEagerOperator",
            )
            coordinate_operators = coordinate_audit.get("operators", {})
            operator_rows.extend(language_metric_rows(
                [
                    *coordinate_operators.get("prefill", []),
                    *coordinate_operators.get("pbd_q6", []),
                ],
                reference_type="FloatCoordinateOperator",
                candidate_type="QuantizedEagerCoordinateOperator",
            ))
            elapsed = time.monotonic() - sample_started
            detail = {
                "schema_version": 2,
                "status": "completed",
                "phase": "language",
                "level": args.level,
                "capture_level": capture_level,
                "index": index - 1,
                "id": sample_id,
                "input": record,
                "input_sha256": record.get("sha256"),
                "output": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "statistics": named_npz_statistics(result.outputs),
                "intermediate": {QUANTIZED_EAGER_STAGE: main_rows},
                "coordinate_audit": coordinate_audit,
                "diagnostic": (
                    {f"{QUANTIZED_EAGER_STAGE}_diagnostic": operator_rows}
                    if operator_rows else {}
                ),
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
            }
            atomic_json(samples_dir / f"{safe_id}.json", detail)
            aggregator.restore_sample(detail)
            completed.add(sample_id)
            processed += 1
            if processed % 10 == 0:
                write_batch_csv(stage_dir / "modules.csv", aggregator)
            progress.set_postfix(
                sample=sample_id,
                elapsed=f"{elapsed:.1f}s",
                comparisons=len(main_rows),
                coordinates=coordinate_audit["decision_count"],
                operators=len(operator_rows),
            )
            del payload, result, coordinate_audit, main_rows, operator_rows, detail
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        progress.close()
        runner.close()
    write_batch_csv(stage_dir / "modules.csv", aggregator)
    weights_path = stage_dir / "weights.json"
    if runner.emulator is not None and runner.emulator.weight_rows:
        atomic_json(weights_path, {
            "schema_version": 1,
            "scheme": LANGUAGE_QUANTIZATION_SCHEME,
            "modules": runner.emulator.weight_rows,
        })
    print_phase_summary(
        "Language quantized eager Prefill + PBD + AR",
        len(records),
        processed,
        resumed,
        started_all,
        stage_dir,
        modules_csv=stage_dir / "modules.csv",
        level=args.level,
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": QUANTIZED_EAGER_STAGE,
            "status": "completed",
            **identity,
            "device": device,
            "profile": {"chunk_size": 1024, "cache_len": 4096, "pbd_q": 6, "ar_q": 1},
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": len(completed),
            "calibration": restored_scales,
            "emulation": {
                "version": LANGUAGE_EMULATION_VERSION,
                "scheme": LANGUAGE_QUANTIZATION_SCHEME,
                "limitation": LANGUAGE_QUANTIZATION_LIMITATION,
                "policy": language_quantization_policy(),
                "decode_cache_source": "float_prefill",
                "coordinate_audit": {
                    "version": COORDINATE_AUDIT_VERSION,
                    "method": "teacher_forced_hybrid_pbd_q6_with_ar_q1_fallback",
                    "cache_source": "stage_specific_teacher_forced_prefill",
                },
            },
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "modules_csv": str((stage_dir / "modules.csv").resolve()),
            "updated_at": utc_now(),
        },
    )
    del model, api
    return 0


def run_float_collection(args: argparse.Namespace) -> int:
    import torch

    if args.phase == "language":
        return run_language_float_collection(args)

    device = detect_float_device()
    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    capture_level = CAPTURE_LEVELS[args.level]
    stage_dir = args.output_dir / "float"
    identity = begin_stage(stage_dir, "float", args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    api, model = create_float_visual_model(
        args.model_path, stage_dir / "work" / "model", device
    )
    completed = phase_processed = phase_resumed = 0
    phase_started = time.monotonic()
    label = {
        "final": "Float final outputs",
        "boundary": "Float major modules",
        "deep": "Float leaf operators",
    }[capture_level]
    progress = progress_bar(records, label)
    for index, record in enumerate(progress, start=1):
        sample_path = samples_dir / f"{_safe_batch_name(record['id'])}.json"
        output_path = outputs_dir / f"{_safe_batch_name(record['id'])}.npy"
        existing = valid_completed_sample(
            sample_path,
            output_path,
            record["id"],
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level=capture_level,
        )
        if existing is not None:
            completed += 1
            phase_resumed += 1
            progress.set_postfix(sample=record["id"], resumed=phase_resumed)
            continue
        started = time.monotonic()
        vision_input = load_visual_input(Path(record["path"]))
        if capture_level == "final":
            output = run_float_final(model, vision_input, device)
            activations: list[dict[str, Any]] = []
        else:
            output, activations = run_float_sample(
                model, vision_input, device, capture_level
            )
        elapsed = time.monotonic() - started
        atomic_npy(output_path, output)
        atomic_json(
            sample_path,
            {
                "schema_version": 2,
                "status": "completed",
                "phase": args.phase,
                "level": args.level,
                "capture_level": capture_level,
                "index": index - 1,
                "id": record["id"],
                "input": record,
                "input_sha256": record.get("sha256"),
                "output": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "statistics": _array_summary(output),
                "activations": activations,
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
            },
        )
        completed += 1
        phase_processed += 1
        del vision_input, output
        if phase_processed % 25 == 0:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        progress.set_postfix(
            sample=record["id"], elapsed=f"{elapsed:.1f}s", activations=len(activations)
        )
    progress.close()
    print_phase_summary(
        label, len(records), phase_processed, phase_resumed,
        phase_started, stage_dir, level=args.level,
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": "float",
            "status": "completed",
            **identity,
            "device": device,
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": completed,
            "capture_policy": {
                "small": "final output only",
                "medium": "patch embedding, blocks, final layernorm, and merger",
                "high": "leaf operators and major module boundaries",
            },
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "updated_at": utc_now(),
        },
    )
    del model, api
    return 0


def run_quantized_eager_collection(args: argparse.Namespace) -> int:
    import torch

    if args.phase == "language":
        return run_language_quantized_eager_collection(args)

    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    capture_level = CAPTURE_LEVELS[args.level]
    float_stage_path = args.output_dir / "float" / "stage.json"
    if not float_stage_path.is_file():
        raise FileNotFoundError(f"run --mode float first: {float_stage_path}")
    float_stage = read_json(float_stage_path)
    if float_stage.get("status") != "completed":
        raise ValueError("run --mode float to completion before quantized-eager")
    if float_stage.get("phase") != args.phase:
        raise ValueError("Float and quantized-eager must use the same --phase")
    float_capture_level = CAPTURE_LEVELS.get(str(float_stage.get("level")))
    if float_capture_level is None:
        raise ValueError("Float stage is missing valid level metadata; run Float again")
    float_model_path = Path(float_stage["model"])
    if float_model_path.resolve() != args.model_path.resolve():
        raise ValueError("quantized-eager must use the same checkpoint as Float")
    if float_stage.get("model_sha256") != path_sha256(float_model_path):
        raise ValueError("Float checkpoint contents changed; run Float again")
    for record in records:
        safe_id = _safe_batch_name(record["id"])
        if valid_completed_sample(
            args.output_dir / "float" / "samples" / f"{safe_id}.json",
            args.output_dir / "float" / "outputs" / f"{safe_id}.npy",
            record["id"],
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level=float_capture_level,
        ) is None:
            raise ValueError(f"invalid Float reference output for {record['id']}")

    scale_manifest = resolve_scale_manifest(
        "vision", getattr(args, "scale_manifest", None)
    )
    metadata.update(
        {
            "float_reference_sha256": float_reference_sha256(
                args.output_dir, float_stage, records
            ),
            "scale_manifest": str(scale_manifest),
            "scale_manifest_sha256": sha256(scale_manifest),
            "emulation_version": EMULATION_VERSION,
        }
    )
    stage_dir = args.output_dir / QUANTIZED_EAGER_STAGE
    identity = begin_stage(stage_dir, QUANTIZED_EAGER_STAGE, args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    weights_path = stage_dir / "weights.json"
    samples_dir.mkdir(parents=True, exist_ok=True)
    aggregator = ModuleMetricAggregator()
    completed: set[str] = set()
    for record in records:
        sample_id = str(record["id"])
        safe_id = _safe_batch_name(sample_id)
        sample = valid_completed_sample(
            samples_dir / f"{safe_id}.json",
            outputs_dir / f"{safe_id}.npy",
            sample_id,
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level=capture_level,
        )
        if sample is not None:
            completed.add(sample_id)
            aggregator.restore_sample(sample)

    device = detect_float_device()
    api, model = create_float_visual_model(
        args.model_path, stage_dir / "work" / "model", device
    )
    restored_scales = restore_calibration_scales(model, scale_manifest)
    emulator = QuantizationEmulator(model, capture_operators=capture_level == "deep")
    boundaries = BoundaryCapture(model, enabled=capture_level != "final")
    phase_started = time.monotonic()
    phase_processed = phase_resumed = 0
    last_boundary_count = last_operator_count = 0
    label = {
        "final": "Quantized eager final outputs",
        "boundary": "Quantized eager major modules",
        "deep": "Quantized eager leaf operators",
    }[capture_level]
    progress = progress_bar(records, label)
    try:
        for index, record in enumerate(progress, start=1):
            sample_id = str(record["id"])
            if sample_id in completed:
                phase_resumed += 1
                progress.set_postfix(sample=sample_id, resumed=phase_resumed)
                continue
            started = time.monotonic()
            vision_input = load_visual_input(Path(record["path"]))
            output, boundary_rows, operator_rows = run_quantized_eager_sample(
                model, emulator, boundaries, vision_input, device
            )
            reference_path = (
                args.output_dir
                / "float"
                / "outputs"
                / f"{_safe_batch_name(sample_id)}.npy"
            )
            reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
            final_row = _final_batch_row(reference, output, len(boundary_rows))
            final_row["candidate_type"] = "QuantizedEagerOutput"
            intermediate_rows = [*boundary_rows, final_row]
            del reference

            elapsed = time.monotonic() - started
            output_path = outputs_dir / f"{_safe_batch_name(sample_id)}.npy"
            atomic_npy(output_path, output)
            detail = {
                "schema_version": 2,
                "status": "completed",
                "phase": args.phase,
                "level": args.level,
                "capture_level": capture_level,
                "index": index - 1,
                "id": sample_id,
                "input": record,
                "input_sha256": record.get("sha256"),
                "output": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "statistics": _array_summary(output),
                "intermediate": {QUANTIZED_EAGER_STAGE: intermediate_rows},
                "diagnostic": (
                    {f"{QUANTIZED_EAGER_STAGE}_diagnostic": operator_rows}
                    if operator_rows else {}
                ),
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
            }
            atomic_json(samples_dir / f"{_safe_batch_name(sample_id)}.json", detail)
            aggregator.restore_sample(detail)
            completed.add(sample_id)
            phase_processed += 1
            last_boundary_count = len(intermediate_rows)
            last_operator_count = len(operator_rows)
            if phase_processed % 25 == 0:
                write_batch_csv(stage_dir / "modules.csv", aggregator)
            progress.set_postfix(
                sample=sample_id,
                elapsed=f"{elapsed:.1f}s",
                boundaries=len(intermediate_rows),
                operators=len(operator_rows),
            )
            del vision_input, output, boundary_rows, operator_rows, intermediate_rows, detail
            if phase_processed % 25 == 0:
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        progress.close()
        boundaries.close()
        emulator.close()

    write_batch_csv(stage_dir / "modules.csv", aggregator)
    weight_rows = list(emulator.weight_rows)
    if weight_rows:
        atomic_json(
            weights_path,
            {
                "schema_version": 1,
                "scheme": QUANTIZATION_SCHEME,
                "modules": weight_rows,
            },
        )
    print_phase_summary(
        label,
        len(records),
        phase_processed,
        phase_resumed,
        phase_started,
        stage_dir,
        modules_csv=stage_dir / "modules.csv",
        boundary_comparisons_per_sample=last_boundary_count,
        operator_comparisons_per_sample=last_operator_count,
        level=args.level,
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": QUANTIZED_EAGER_STAGE,
            "status": "completed",
            **identity,
            "float_model": float_stage["model"],
            "device": device,
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": len(completed),
            "calibration": restored_scales,
            "emulation": {
                "version": EMULATION_VERSION,
                "scheme": QUANTIZATION_SCHEME,
                "limitation": QUANTIZATION_LIMITATION,
            },
            "capture_policy": {
                "small": "final output only",
                "medium": "patch embedding, blocks, final layernorm, and merger",
                "high": "major boundaries plus local quantized operators",
            },
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "modules_csv": str((stage_dir / "modules.csv").resolve()),
            "weights": str(weights_path.resolve()) if weights_path.is_file() else None,
            "updated_at": utc_now(),
        },
    )
    del model, api
    return 0


def _language_bc_artifact_identity(paths: dict[str, Path]) -> tuple[dict[str, Any], str]:
    identities: dict[Path, dict[str, Any]] = {}
    files: dict[str, Any] = {}
    for graph, path in paths.items():
        resolved = path.resolve()
        if resolved not in identities:
            identities[resolved] = {
                "path": str(resolved),
                "sha256": sha256(resolved),
                "bytes": resolved.stat().st_size,
            }
        files[graph] = identities[resolved]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def load_language_hbm_artifacts(
    model_path: Path,
) -> tuple[Any, dict[str, LanguageBCArtifact]]:
    """Validate the release catalog, then expose the three numerical test graphs."""

    import hbdk4.compiler as hb

    hbm = hb.Hbm(str(model_path))
    catalog = tuple(str(function.name) for function in hbm.graphs)
    validate_language_hbm_catalog(catalog, release_language_graphs())
    functions = {str(function.name): function for function in hbm.graphs}
    artifacts = {
        graph: LanguageBCArtifact(
            graph=graph,
            path=model_path,
            function=functions[graph],
            inputs=list(functions[graph].inputs),
            outputs=list(functions[graph].outputs),
        )
        for graph in LANGUAGE_BC_GRAPHS
    }
    return hbm, artifacts


def _language_bc_metric_rows(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    candidate_type: str,
) -> list[dict[str, Any]]:
    if not candidate:
        raise ValueError("Language BC did not execute any graph for this sample")
    if not set(candidate).issubset(reference):
        raise ValueError(
            "Language BC produced outputs absent from Float: "
            f"{sorted(set(candidate) - set(reference))}"
        )
    semantic_stages = {
        "prefill_logits": "prefill",
        "pbd_logits": "pbd_q6",
        "ar_logits": "ar_q1",
    }
    entries = []
    for name in candidate:
        stage = semantic_stages[name]
        entries.append({
            "stage": stage,
            "module": "logits",
            "kind": "output",
            "comparison": compare_arrays(reference[name], candidate[name]),
        })
        if name == "pbd_logits":
            entries.extend(
                {
                    "stage": stage,
                    "module": f"logits.token_{index}",
                    "kind": "output",
                    "comparison": compare_arrays(
                        reference[name][:, index], candidate[name][:, index]
                    ),
                }
                for index in range(reference[name].shape[1])
            )
    return language_metric_rows(
        entries,
        reference_type="LanguageFloat",
        candidate_type=candidate_type,
    )


def run_language_bc_collection(args: argparse.Namespace) -> int:
    import torch

    if args.level != "small":
        raise ValueError("Language compiled artifacts support --level small only")
    is_hbm = args.mode == "hbm"
    stage_name = "hbm" if is_hbm else BC_MODES[args.mode]
    capture_level = CAPTURE_LEVELS[args.level]
    all_records = prepare_input_index(args.output_dir, args.input_dir)

    float_stage_path = args.output_dir / "float" / "stage.json"
    if not float_stage_path.is_file():
        raise FileNotFoundError(f"run Language --mode float first: {float_stage_path}")
    float_stage = read_json(float_stage_path)
    if float_stage.get("status") != "completed" or float_stage.get("phase") != "language":
        raise ValueError("run the matching Language Float stage to completion first")
    float_model_path = Path(float_stage["model"])
    if float_stage.get("model_sha256") != path_sha256(float_model_path):
        raise ValueError("Language Float checkpoint contents changed; run Float again")
    float_capture_level = CAPTURE_LEVELS.get(str(float_stage.get("level")))
    if float_capture_level is None:
        raise ValueError("Language Float stage is missing level metadata; run Float again")

    converted = args.mode == "converted-bc"
    resolved_paths = (
        {graph: args.model_path.resolve() for graph in LANGUAGE_BC_GRAPHS}
        if is_hbm
        else resolve_language_bc_paths(args.model_path, converted=converted)
    )
    eligible_records = all_records
    if set(resolved_paths) == {"decode_ar"}:
        eligible_records = []
        for record in all_records:
            output_path = (
                args.output_dir / "float" / "outputs"
                / f"{_safe_batch_name(str(record['id']))}.npz"
            )
            if output_path.is_file():
                with np.load(output_path, allow_pickle=False) as archive:
                    if "ar_logits" in archive.files:
                        eligible_records.append(record)
        if not eligible_records:
            raise ValueError("no Language Float samples contain AR q=1 reference logits")
    records, coverage = select_records(eligible_records, args.nums)
    coverage["graph_selection"] = list(resolved_paths)
    coverage["source_input_count"] = len(all_records)
    metadata = selection_metadata(args, records, coverage)
    for record in records:
        safe_id = _safe_batch_name(str(record["id"]))
        if valid_completed_sample(
            args.output_dir / "float" / "samples" / f"{safe_id}.json",
            args.output_dir / "float" / "outputs" / f"{safe_id}.npz",
            str(record["id"]),
            input_sha256=record.get("sha256"),
            phase="language",
            capture_level=float_capture_level,
        ) is None:
            raise ValueError(f"invalid Language Float reference for {record['id']}")
    metadata["float_reference_sha256"] = float_reference_sha256(
        args.output_dir, float_stage, records
    )

    artifact_files, artifact_set_sha256 = _language_bc_artifact_identity(resolved_paths)
    metadata["artifact_set_sha256"] = artifact_set_sha256
    stage_dir = args.output_dir / stage_name
    canonical_artifact = next(iter(resolved_paths.values()))
    identity = begin_stage(stage_dir, stage_name, canonical_artifact, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)

    aggregator = ModuleMetricAggregator()
    completed: set[str] = set()
    for record in records:
        sample_id = str(record["id"])
        safe_id = _safe_batch_name(sample_id)
        sample = valid_completed_sample(
            samples_dir / f"{safe_id}.json",
            outputs_dir / f"{safe_id}.npz",
            sample_id,
            input_sha256=record.get("sha256"),
            phase="language",
            capture_level=capture_level,
        )
        if sample is not None:
            completed.add(sample_id)
            aggregator.restore_sample(sample)

    device = detect_float_device()
    api, model, rotation = create_language_model(
        float_model_path, stage_dir / "work" / "float_model", device
    )
    hbm = None
    if is_hbm:
        hbm, artifacts = load_language_hbm_artifacts(args.model_path)
    else:
        artifacts = load_language_bc_artifacts(
            args.model_path,
            converted=converted,
            loader=load_artifact,
        )
    runner = LanguageBCRunner(model, rotation, device, artifacts)
    artifact_contract = runner.describe_artifacts()
    candidate_type = (
        "LanguageHBM"
        if is_hbm
        else "LanguageConvertedBC" if converted else "LanguageExportedBC"
    )

    processed = resumed = 0
    phase_started = time.monotonic()
    label = f"Language {stage_name} Prefill + PBD + AR final outputs"
    progress = progress_bar(records, label)
    try:
        for index, record in enumerate(progress, 1):
            sample_id = str(record["id"])
            safe_id = _safe_batch_name(sample_id)
            if sample_id in completed:
                resumed += 1
                progress.set_postfix(sample=sample_id, resumed=resumed)
                continue
            started = time.monotonic()
            payload = load_language_payload(Path(record["path"]))
            with OperationHeartbeat(f"{stage_name} Language sample={sample_id}"):
                result = runner.run(payload)
            reference_path = args.output_dir / "float" / "outputs" / f"{safe_id}.npz"
            reference = load_named_npz(reference_path)
            rows = _language_bc_metric_rows(
                reference, result.outputs, candidate_type=candidate_type
            )
            output_path = outputs_dir / f"{safe_id}.npz"
            atomic_named_npz(output_path, result.outputs)
            elapsed = time.monotonic() - started
            detail = {
                "schema_version": 2,
                "status": "completed",
                "phase": "language",
                "level": args.level,
                "capture_level": capture_level,
                "index": index - 1,
                "id": sample_id,
                "input": record,
                "input_sha256": record.get("sha256"),
                "output": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "statistics": named_npz_statistics(result.outputs),
                "executed_stages": list(result.outputs),
                "execution": result.execution,
                "intermediate": {stage_name: rows},
                "timings": result.timings,
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
            }
            atomic_json(samples_dir / f"{safe_id}.json", detail)
            aggregator.restore_sample(detail)
            completed.add(sample_id)
            processed += 1
            if processed % 10 == 0:
                write_batch_csv(stage_dir / "modules.csv", aggregator)
            progress.set_postfix(
                sample=sample_id,
                elapsed=f"{elapsed:.1f}s",
                graphs=len(result.outputs),
            )
            del payload, result, reference, rows, detail
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        progress.close()
        runner.close()

    write_batch_csv(stage_dir / "modules.csv", aggregator)
    print_phase_summary(
        label,
        len(records),
        processed,
        resumed,
        phase_started,
        stage_dir,
        level=args.level,
        graphs=",".join(resolved_paths),
        decode_cache_source="float_prefill",
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": stage_name,
            "status": "completed",
            **identity,
            "float_model": float_stage["model"],
            "device": device,
            "profile": {"chunk_size": 1024, "cache_len": 4096, "pbd_q": 6, "ar_q": 1},
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": len(completed),
            "capture_policy": {"small": "selected graph final logits only"},
            "cache_policy": {
                graph: "zero" if graph == "prefill" else "float_prefill"
                for graph in resolved_paths
            } | {
                "purpose": "isolate each Decode BC graph from Prefill BC error",
            },
            "artifact_files": artifact_files,
            "artifacts": artifact_contract,
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "modules_csv": str((stage_dir / "modules.csv").resolve()),
            "updated_at": utc_now(),
        },
    )
    del artifacts, hbm, model, api
    return 0


def run_bc_collection(args: argparse.Namespace) -> int:
    import torch

    stage_name = BC_MODES[args.mode]
    capture_level = CAPTURE_LEVELS[args.level]
    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    float_stage_path = args.output_dir / "float" / "stage.json"
    if not float_stage_path.is_file():
        raise FileNotFoundError(f"run --mode float first: {float_stage_path}")
    float_stage = read_json(float_stage_path)
    if float_stage.get("status") != "completed":
        raise ValueError("run --mode float to completion before running BC")
    if float_stage.get("phase") != args.phase:
        raise ValueError("Float and BC phases do not match")
    float_model_path = Path(float_stage["model"])
    if float_stage.get("model_sha256") != path_sha256(float_model_path):
        raise ValueError("Float checkpoint contents changed; run Float again")
    float_capture_level = CAPTURE_LEVELS.get(str(float_stage.get("level")))
    if float_capture_level is None:
        raise ValueError("Float stage is missing level metadata; run Float again")
    for record in records:
        safe_id = _safe_batch_name(record["id"])
        if valid_completed_sample(
            args.output_dir / "float" / "samples" / f"{safe_id}.json",
            args.output_dir / "float" / "outputs" / f"{safe_id}.npy",
            record["id"],
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level=float_capture_level,
        ) is None:
            raise ValueError(f"invalid Float reference output for {record['id']}")
    metadata["float_reference_sha256"] = float_reference_sha256(
        args.output_dir, float_stage, records
    )

    stage_dir = args.output_dir / stage_name
    identity = begin_stage(stage_dir, stage_name, args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    aggregator = ModuleMetricAggregator()
    completed: set[str] = set()
    for record in records:
        sample_id = str(record["id"])
        safe_id = _safe_batch_name(sample_id)
        sample = valid_completed_sample(
            samples_dir / f"{safe_id}.json",
            outputs_dir / f"{safe_id}.npy",
            sample_id,
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level=capture_level,
        )
        if sample is not None:
            completed.add(sample["id"])
            aggregator.restore_sample(sample)

    device: str | None = None
    api = float_model = dispatcher = None
    if capture_level != "final":
        device = detect_float_device()
        api, float_model = create_float_visual_model(
            float_model_path, stage_dir / "work" / "float_model", device
        )
    artifact, input_descriptors, output_descriptors = load_artifact(
        "bc", args.model_path, "visual"
    )
    if capture_level != "final":
        dispatcher = BCCallbackDispatcher()
        artifact.register_callback(dispatcher)

    phase_started = time.monotonic()
    phase_processed = phase_resumed = 0
    last_row_count = 0
    label = {
        "final": f"{stage_name} final outputs",
        "boundary": f"{stage_name} major modules",
        "deep": f"{stage_name} leaf operators",
    }[capture_level]
    progress = progress_bar(records, label)
    for index, record in enumerate(progress, start=1):
        sample_id = record["id"]
        if sample_id in completed:
            phase_resumed += 1
            progress.set_postfix(sample=sample_id, resumed=phase_resumed)
            continue
        started = time.monotonic()
        vision_input = load_visual_input(Path(record["path"]))
        float_output_path = (
            args.output_dir / "float" / "outputs" / f"{_safe_batch_name(sample_id)}.npy"
        )
        if capture_level == "final":
            feed_started = time.monotonic()
            with OperationHeartbeat(f"{stage_name} final sample={sample_id}"):
                candidate_output = execute_loaded_bc_final(
                    artifact, input_descriptors, output_descriptors, vision_input
                )
            feed_seconds = time.monotonic() - feed_started
            compare_started = time.monotonic()
            reference_output = np.load(float_output_path, mmap_mode="r", allow_pickle=False)
            rows = [_final_batch_row(reference_output, candidate_output, 0)]
            timings = {
                "bc_feed_seconds": feed_seconds,
                "comparison_seconds": time.monotonic() - compare_started,
            }
            del reference_output
        else:
            assert float_model is not None and dispatcher is not None
            candidate_output, rows, timings = run_bc_sample(
                float_model,
                artifact,
                dispatcher,
                input_descriptors,
                output_descriptors,
                vision_input,
                capture_level,
                f"{stage_name} {capture_level} sample={sample_id}",
            )
        elapsed = time.monotonic() - started
        output_path = outputs_dir / f"{_safe_batch_name(sample_id)}.npy"
        persist_started = time.monotonic()
        atomic_npy(output_path, candidate_output)
        detail = {
            "schema_version": 2,
            "status": "completed",
            "phase": args.phase,
            "level": args.level,
            "capture_level": capture_level,
            "index": index - 1,
            "id": sample_id,
            "input": record,
            "input_sha256": record.get("sha256"),
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "statistics": _array_summary(candidate_output),
            "intermediate": {stage_name: rows},
            "timings": timings,
            "elapsed_seconds": elapsed,
            "finished_at": utc_now(),
        }
        atomic_json(samples_dir / f"{_safe_batch_name(sample_id)}.json", detail)
        aggregator.add(stage_name, rows)
        completed.add(sample_id)
        phase_processed += 1
        if phase_processed % 25 == 0:
            write_batch_csv(stage_dir / "modules.csv", aggregator)
        persist_seconds = time.monotonic() - persist_started
        last_row_count = len(rows)
        progress.set_postfix(
            sample=sample_id,
            elapsed=f"{elapsed:.1f}s",
            bc_feed=f"{timings['bc_feed_seconds']:.1f}s",
            compare=f"{timings['comparison_seconds']:.1f}s",
            persist=f"{persist_seconds:.1f}s",
        )
        del vision_input, candidate_output, rows, timings, detail
        if phase_processed % 25 == 0:
            gc.collect()
            if device is not None and device.startswith("cuda"):
                torch.cuda.empty_cache()
    progress.close()
    write_batch_csv(stage_dir / "modules.csv", aggregator)
    print_phase_summary(
        label, len(records), phase_processed, phase_resumed,
        phase_started, stage_dir, modules_csv=stage_dir / "modules.csv",
        comparisons_per_sample=last_row_count if completed else 0,
        level=args.level,
    )
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": stage_name,
            "status": "completed",
            **identity,
            "float_model": float_stage["model"],
            "device": device,
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": len(completed),
            "capture_policy": {
                "small": "final output only; HBDK callbacks disabled",
                "medium": "major module boundaries; HBDK callbacks enabled",
                "high": "leaf operators and major boundaries; HBDK callbacks enabled",
            },
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "modules_csv": str((stage_dir / "modules.csv").resolve()),
            "updated_at": utc_now(),
        },
    )
    del artifact
    if float_model is not None:
        del float_model, api
    return 0


def run_hbm_collection(args: argparse.Namespace) -> int:
    all_records = prepare_input_index(args.output_dir, args.input_dir)
    records, coverage = select_records(all_records, args.nums)
    metadata = selection_metadata(args, records, coverage)
    stage_dir = args.output_dir / "hbm"
    previous_state_path = stage_dir / "stage.json"
    previous_state = read_json(previous_state_path) if previous_state_path.is_file() else {}
    previous_runtime = previous_state.get("runtime", {})
    identity = begin_stage(stage_dir, "hbm", args.model_path, metadata)
    samples_dir = stage_dir / "samples"
    outputs_dir = stage_dir / "outputs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    resumable_ids = {
        record["id"]
        for record in records
        if valid_completed_sample(
            samples_dir / f"{_safe_batch_name(record['id'])}.json",
            outputs_dir / f"{_safe_batch_name(record['id'])}.npy",
            record["id"],
            input_sha256=record.get("sha256"),
            phase=args.phase,
            capture_level="final",
        )
        is not None
    }
    pending_count = len(records) - len(resumable_ids)
    backend = detect_hbm_backend()
    artifact = input_descriptors = output_descriptors = None
    board_session: S600VisionSession | None = None
    load_started = time.monotonic()
    print("\n================== [1/2] HBM MODEL LOAD ==================", flush=True)
    if pending_count:
        if backend == "hbdk_x86_simulator":
            artifact, input_descriptors, output_descriptors = load_artifact(
                "hbm", args.model_path, "visual"
            )
        else:
            board_session = S600VisionSession(args.model_path)
            board_session.start()
    print_phase_summary(
        "HBM model load", 1, int(bool(pending_count)), int(not pending_count),
        load_started, args.model_path, backend=backend,
        model_load_seconds=(
            f"{board_session.load_seconds:.3f}" if board_session is not None else "N/A"
        ),
    )
    completed = 0
    phase_started = time.monotonic()
    phase_processed = phase_resumed = 0
    collection_seconds = 0.0
    session_close_seconds = 0.0
    progress = progress_bar(records, "[2/2] HBM final outputs")
    try:
        for index, record in enumerate(progress, start=1):
            sample_id = record["id"]
            sample_path = samples_dir / f"{_safe_batch_name(sample_id)}.json"
            output_path = outputs_dir / f"{_safe_batch_name(sample_id)}.npy"
            if sample_id in resumable_ids:
                completed += 1
                phase_resumed += 1
                progress.set_postfix(sample=sample_id, resumed=phase_resumed)
                continue
            started = time.monotonic()
            inference_ms: float | None = None
            vision_input = load_visual_input(Path(record["path"]))
            if backend == "hbdk_x86_simulator":
                assert artifact is not None and input_descriptors is not None and output_descriptors is not None
                descriptor = input_descriptors[0]
                feed = {
                    str(descriptor.name): vision_input.astype(tensor_dtype(descriptor), copy=False)
                }
                with SimulatorHeartbeat(30):
                    raw = artifact.feed(feed)
                output = np.asarray(raw[str(output_descriptors[0].name)])
            else:
                assert board_session is not None
                scratch = stage_dir / "work" / "current"
                board_input = scratch / "vision_input.f16.bin"
                board_output = scratch / "vision_output.f16.bin"
                atomic_binary(board_input, vision_input)
                inference_ms = board_session.run(board_input, board_output)
                output = np.fromfile(board_output, dtype=np.dtype("<f2"))
                expected = int(np.prod(VISION_OUTPUT_SHAPE))
                if output.size != expected:
                    raise ValueError(f"S600 output has {output.size} values, expected {expected}")
                output = output.reshape(VISION_OUTPUT_SHAPE)
            elapsed = time.monotonic() - started
            atomic_npy(output_path, output)
            atomic_json(
                sample_path,
                {
                    "schema_version": 2,
                    "status": "completed",
                    "phase": args.phase,
                    "level": args.level,
                    "capture_level": "final",
                    "index": index - 1,
                    "id": sample_id,
                    "input": record,
                    "input_sha256": record.get("sha256"),
                    "backend": backend,
                    "output": str(output_path.resolve()),
                    "output_sha256": sha256(output_path),
                    "statistics": _array_summary(output),
                    "inference_ms": inference_ms,
                    "elapsed_seconds": elapsed,
                    "finished_at": utc_now(),
                },
            )
            completed += 1
            phase_processed += 1
            del vision_input, output
            progress.set_postfix(
                sample=sample_id,
                elapsed=f"{elapsed:.1f}s",
                inference_ms=f"{inference_ms:.1f}" if inference_ms is not None else "N/A",
            )
    finally:
        progress.close()
        collection_seconds = time.monotonic() - phase_started
        if board_session is not None:
            close_started = time.monotonic()
            board_session.close()
            session_close_seconds = time.monotonic() - close_started
    invocation_inference_total_ms = (
        sum(board_session.inference_ms) if board_session is not None else 0.0
    )
    invocation_inference_mean_ms = (
        invocation_inference_total_ms / board_session.inferences
        if board_session is not None and board_session.inferences
        else 0.0
    )
    invocation_graph_execute_time_ratio = (
        invocation_inference_total_ms / (collection_seconds * 1000.0)
        if collection_seconds and board_session is not None
        else 0.0
    )
    print_phase_summary(
        "HBM final outputs", len(records), phase_processed, phase_resumed,
        phase_started, stage_dir, elapsed_seconds=collection_seconds, backend=backend,
        model_loads=1 if pending_count else 0,
        inference_total_ms=f"{invocation_inference_total_ms:.3f}",
        inference_mean_ms=f"{invocation_inference_mean_ms:.3f}",
        graph_execute_time_ratio=f"{100.0 * invocation_graph_execute_time_ratio:.2f}%",
        session_close_seconds=f"{session_close_seconds:.3f}",
    )
    recorded_inference_ms = []
    for record in records:
        sample = read_json(samples_dir / f"{_safe_batch_name(record['id'])}.json")
        if sample.get("inference_ms") is not None:
            recorded_inference_ms.append(float(sample["inference_ms"]))
    if pending_count == 0 and previous_runtime:
        runtime_summary = dict(previous_runtime)
    else:
        cumulative_collection_seconds = (
            float(previous_runtime.get("collection_seconds", 0.0)) + collection_seconds
        )
        cumulative_inference_total_ms = sum(recorded_inference_ms)
        cumulative_inference_mean_ms = (
            cumulative_inference_total_ms / len(recorded_inference_ms)
            if recorded_inference_ms else 0.0
        )
        runtime_summary = {
            "persistent_session": backend == "s600_bpu",
            "model_loads": int(previous_runtime.get("model_loads", 0)) + int(bool(pending_count)),
            "inferences": len(recorded_inference_ms) if backend == "s600_bpu" else completed,
            "model_load_seconds": (
                float(previous_runtime.get("model_load_seconds") or 0.0)
                + (board_session.load_seconds if board_session is not None else 0.0)
            ),
            "inference_total_ms": cumulative_inference_total_ms,
            "inference_mean_ms": cumulative_inference_mean_ms,
            "collection_seconds": cumulative_collection_seconds,
            "session_close_seconds": (
                float(previous_runtime.get("session_close_seconds", 0.0))
                + session_close_seconds
            ),
            "graph_execute_time_ratio": (
                cumulative_inference_total_ms / (cumulative_collection_seconds * 1000.0)
                if cumulative_collection_seconds and backend == "s600_bpu" else 0.0
            ),
            "graph_execute_time_ratio_definition": (
                "sum(ExecuteGraphByName time) / collection wall time; "
                "not hardware BPU utilization"
            ),
        }
    runtime_summary["last_invocation"] = {
        "processed": phase_processed,
        "resumed": phase_resumed,
        "collection_seconds": collection_seconds,
        "model_loads": int(bool(pending_count)),
        "inference_total_ms": invocation_inference_total_ms,
        "inference_mean_ms": invocation_inference_mean_ms,
        "session_close_seconds": session_close_seconds,
    }
    atomic_json(
        stage_dir / "stage.json",
        {
            "schema_version": 2,
            "stage": "hbm",
            "status": "completed",
            **identity,
            "backend": backend,
            "runtime": runtime_summary,
            "input_count": len(all_records),
            "selected_count": len(records),
            "completed": completed,
            "samples": str(samples_dir.resolve()),
            "outputs": str(outputs_dir.resolve()),
            "updated_at": utc_now(),
        },
    )
    return 0


def load_stage_outputs(run_dir: Path, name: str) -> tuple[dict[str, Any], list[np.ndarray]]:
    state_path = run_dir / f"{name}.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    record = read_json(state_path)
    if record.get("status") != "completed":
        raise ValueError(f"stage {name} is {record.get('status')}")
    output_path = run_dir / record["output_file"]
    if sha256(output_path) != record["output_sha256"]:
        raise ValueError(f"stage output hash mismatch: {output_path}")
    outputs = load_npz(output_path)
    if describe_arrays(outputs, [item["name"] for item in record["outputs"]]) != record["outputs"]:
        raise ValueError(f"stage output metadata mismatch: {output_path}")
    return record, outputs


def _trace_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _trace_outputs(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events_path = Path(trace["events_file"])
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    entries: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        event_name = canonical_name(event["name"])
        for tensor in event["outputs"]:
            entries.append(
                {
                    "sequence": event["sequence"],
                    "name": event["name"],
                    "canonical_name": event_name,
                    "normalized_name": _trace_name(event_name),
                    "type": event["type"],
                    "semantic_group": event.get("semantic_group"),
                    "semantic_operation": event.get("semantic_operation"),
                    "tensor": tensor,
                }
            )
    return entries


def compare_intermediate_traces(
    reference_trace: dict[str, Any],
    candidate_trace: dict[str, Any],
) -> dict[str, Any]:
    reference_entries = _trace_outputs(reference_trace)
    candidate_entries = _trace_outputs(candidate_trace)
    candidates: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    for entry in candidate_entries:
        key = (entry["normalized_name"], tuple(entry["tensor"]["shape"]))
        candidates.setdefault(key, []).append(entry)

    occurrences: dict[tuple[str, tuple[int, ...]], int] = {}
    comparisons: list[dict[str, Any]] = []
    matched_candidate_ids: set[tuple[int, str]] = set()
    numeric = matched = 0
    for reference in reference_entries:
        key = (reference["normalized_name"], tuple(reference["tensor"]["shape"]))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        choices = candidates.get(key, [])
        item: dict[str, Any] = {
            "reference_sequence": reference["sequence"],
            "reference_name": reference["name"],
            "canonical_name": reference["canonical_name"],
            "shape": reference["tensor"]["shape"],
            "occurrence": occurrence,
            "semantic_group": reference.get("semantic_group"),
            "semantic_operation": reference.get("semantic_operation"),
            "reference_type": reference["type"],
        }
        if occurrence >= len(choices):
            item["status"] = "unmatched"
            comparisons.append(item)
            continue
        candidate = choices[-1] if occurrence == 0 else choices[min(occurrence, len(choices) - 1)]
        matched += 1
        matched_candidate_ids.add((candidate["sequence"], candidate["tensor"]["path"]))
        item.update(
            status="matched",
            candidate_sequence=candidate["sequence"],
            candidate_name=candidate["name"],
            candidate_type=candidate["type"],
            candidate_choices=len(choices),
            selection_policy="last_output_for_module_and_shape",
        )
        reference_file = reference["tensor"].get("file")
        candidate_file = candidate["tensor"].get("file")
        if reference_file and candidate_file:
            reference_array = np.load(reference_file, mmap_mode="r", allow_pickle=False)
            candidate_array = np.load(candidate_file, mmap_mode="r", allow_pickle=False)
            item["comparison"] = compare_arrays(reference_array, candidate_array)
            numeric += 1
        else:
            item["comparison"] = {
                "status": "metadata_only",
                "reason": "use --trace full in both stages for numeric comparison",
            }
        comparisons.append(item)

    candidate_only = sum(
        (entry["sequence"], entry["tensor"]["path"]) not in matched_candidate_ids
        for entry in candidate_entries
    )
    return {
        "status": "compared",
        "reference_outputs": len(reference_entries),
        "candidate_outputs": len(candidate_entries),
        "matched": matched,
        "numeric_comparisons": numeric,
        "unmatched_reference": len(reference_entries) - matched,
        "unmatched_candidate": candidate_only,
        "outputs": comparisons,
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    manifest, _ = load_run(run_dir)
    reference_record, reference_outputs = load_stage_outputs(run_dir, "float")
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "completed",
        "component": manifest["component"],
        "graph": manifest["graph"],
        "run_dir": str(run_dir.resolve()),
        "input": manifest,
        "reference": reference_record,
        "candidates": {},
        "intermediate": {},
        "pipeline_comparisons": [],
        "missing_stages": [],
        "created_at": utc_now(),
    }
    completed_outputs: dict[str, list[np.ndarray]] = {"float": reference_outputs}
    for name in CANDIDATE_STAGES:
        state_path = run_dir / f"{name}.json"
        if not state_path.is_file():
            report["missing_stages"].append(name)
            continue
        candidate_record = read_json(state_path)
        if candidate_record.get("status") != "completed":
            report["candidates"][name] = {"stage": candidate_record, "status": "pending"}
            continue
        candidate_record, candidate_outputs = load_stage_outputs(run_dir, name)
        if len(candidate_outputs) != len(reference_outputs):
            report["candidates"][name] = {
                "status": "incompatible",
                "stage": candidate_record,
                "reason": "output_count_mismatch",
                "reference_count": len(reference_outputs),
                "candidate_count": len(candidate_outputs),
            }
            continue
        completed_outputs[name] = candidate_outputs
        comparisons: list[dict[str, Any]] = []
        for index, (reference, candidate) in enumerate(
            zip(reference_outputs, candidate_outputs, strict=True)
        ):
            if index == 0 or candidate.dtype.kind == "f":
                comparison = compare_arrays(reference, candidate)
            else:
                comparison = {
                    "status": "skipped",
                    "reason": "quantized KV output lacks a dequantization scale",
                }
            comparisons.append(
                {
                    "index": index,
                    "name": candidate_record["outputs"][index]["name"],
                    "comparison": comparison,
                }
            )
        report["candidates"][name] = {
            "status": "compared",
            "stage": candidate_record,
            "outputs": comparisons,
        }
        if (
            name in ("exported_bc", "converted_bc")
            and reference_record.get("trace")
            and candidate_record.get("trace")
        ):
            report["intermediate"][name] = compare_intermediate_traces(
                reference_record["trace"], candidate_record["trace"]
            )

    for reference_name, candidate_name in (
        ("float", "exported_bc"),
        ("exported_bc", "converted_bc"),
        ("converted_bc", "hbm"),
    ):
        reference_values = completed_outputs.get(reference_name)
        candidate_values = completed_outputs.get(candidate_name)
        if reference_values is None or candidate_values is None:
            continue
        if len(reference_values) != len(candidate_values):
            comparison = {
                "status": "output_count_mismatch",
                "reference_count": len(reference_values),
                "candidate_count": len(candidate_values),
            }
        else:
            comparison = compare_arrays(reference_values[0], candidate_values[0])
        report["pipeline_comparisons"].append(
            {
                "reference": reference_name,
                "candidate": candidate_name,
                "comparison": comparison,
            }
        )
    if report["missing_stages"] or any(
        item["status"] != "compared" for item in report["candidates"].values()
    ):
        report["status"] = "partial"
    return report


def run_report(args: argparse.Namespace) -> int:
    def metric_text(value: Any) -> str:
        return f"{float(value):.6g}" if isinstance(value, (int, float)) else str(value)

    report = build_report(args.run_dir)
    output = args.output or args.run_dir / "report.json"
    csv_output = output.with_suffix(".csv")
    intermediate_csv_output = output.with_name(output.stem + "_intermediate.csv")
    write_pipeline_csv(csv_output, report["pipeline_comparisons"])
    write_intermediate_csv(intermediate_csv_output, report["intermediate"])
    report["pipeline_csv"] = str(csv_output.resolve())
    report["intermediate_csv"] = str(intermediate_csv_output.resolve())
    atomic_json(output, report)
    print(f"[report] status={report['status']}", flush=True)
    print(f"[report] output={output}", flush=True)
    print(f"[report] csv={csv_output}", flush=True)
    print(f"[report] intermediate_csv={intermediate_csv_output}", flush=True)
    for name, candidate in report["candidates"].items():
        if candidate["status"] != "compared" or not candidate["outputs"]:
            print(f"[report] {name}: {candidate['status']}", flush=True)
            continue
        metric = candidate["outputs"][0]["comparison"]
        print(
            f"[report] {name}: status={metric['status']} "
            f"cosine={metric.get('cosine')} relative_l2={metric.get('relative_l2')}",
            flush=True,
        )
    if report["pipeline_comparisons"]:
        print("", flush=True)
        print(
            "| Comparison | Shape | Cosine | Relative L2 | MAE | RMSE | Max Abs | Feature Argmax | Exact |",
            flush=True,
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---|", flush=True)
        for item in report["pipeline_comparisons"]:
            metric = item["comparison"]
            label = f"{item['reference']} -> {item['candidate']}"
            print(
                f"| {label} | {metric.get('shape', 'N/A')} "
                f"| {metric_text(metric.get('cosine', 'N/A'))} "
                f"| {metric_text(metric.get('relative_l2', 'N/A'))} "
                f"| {metric_text(metric.get('mae', 'N/A'))} "
                f"| {metric_text(metric.get('rmse', 'N/A'))} "
                f"| {metric_text(metric.get('max_abs', 'N/A'))} "
                f"| {metric_text(metric.get('top1_agreement', 'N/A'))} "
                f"| {metric.get('exact_equal', 'N/A')} |",
                flush=True,
            )
        print("", flush=True)
        print(
            "| Comparison | Reference Range | Candidate Range | Reference Mean | Candidate Mean | Reference Std | Candidate Std |",
            flush=True,
        )
        print("|---|---|---|---:|---:|---:|---:|", flush=True)
        for item in report["pipeline_comparisons"]:
            metric = item["comparison"]
            label = f"{item['reference']} -> {item['candidate']}"
            print(
                f"| {label} | {metric.get('reference_range', 'N/A')} "
                f"| {metric.get('candidate_range', 'N/A')} "
                f"| {metric_text(metric.get('reference_mean', 'N/A'))} "
                f"| {metric_text(metric.get('candidate_mean', 'N/A'))} "
                f"| {metric_text(metric.get('reference_std', 'N/A'))} "
                f"| {metric_text(metric.get('candidate_std', 'N/A'))} |",
                flush=True,
            )
    return 0


def write_pipeline_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    columns = [
        "reference",
        "candidate",
        "status",
        "shape",
        "cosine",
        "relative_l2",
        "mae",
        "rmse",
        "max_abs",
        "feature_argmax_agreement",
        "exact_equal",
        "reference_min",
        "reference_max",
        "candidate_min",
        "candidate_max",
        "reference_mean",
        "candidate_mean",
        "reference_std",
        "candidate_std",
        "reference_nonzero",
        "candidate_nonzero",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in comparisons:
            metric = item["comparison"]
            reference_range = metric.get("reference_range", [None, None])
            candidate_range = metric.get("candidate_range", [None, None])
            shape = metric.get("shape")
            writer.writerow(
                {
                    "reference": item["reference"],
                    "candidate": item["candidate"],
                    "status": metric.get("status"),
                    "shape": "x".join(str(value) for value in shape) if shape else None,
                    "cosine": metric.get("cosine"),
                    "relative_l2": metric.get("relative_l2"),
                    "mae": metric.get("mae"),
                    "rmse": metric.get("rmse"),
                    "max_abs": metric.get("max_abs"),
                    "feature_argmax_agreement": metric.get("top1_agreement"),
                    "exact_equal": metric.get("exact_equal"),
                    "reference_min": reference_range[0],
                    "reference_max": reference_range[1],
                    "candidate_min": candidate_range[0],
                    "candidate_max": candidate_range[1],
                    "reference_mean": metric.get("reference_mean"),
                    "candidate_mean": metric.get("candidate_mean"),
                    "reference_std": metric.get("reference_std"),
                    "candidate_std": metric.get("candidate_std"),
                    "reference_nonzero": metric.get("reference_nonzero"),
                    "candidate_nonzero": metric.get("candidate_nonzero"),
                }
            )
    os.replace(temporary, path)


def write_intermediate_csv(path: Path, stages: dict[str, Any]) -> None:
    columns = [
        "stage",
        "reference_sequence",
        "semantic_group",
        "semantic_operation",
        "reference_name",
        "reference_type",
        "candidate_sequence",
        "candidate_name",
        "candidate_type",
        "candidate_choices",
        "selection_policy",
        "shape",
        "status",
        "cosine",
        "relative_l2",
        "mae",
        "rmse",
        "max_abs",
        "feature_argmax_agreement",
        "exact_equal",
        "reference_min",
        "reference_max",
        "candidate_min",
        "candidate_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stage_name, stage in stages.items():
            for item in stage.get("outputs", []):
                metric = item.get("comparison", {})
                reference_range = metric.get("reference_range", [None, None])
                candidate_range = metric.get("candidate_range", [None, None])
                shape = item.get("shape")
                writer.writerow(
                    {
                        "stage": stage_name,
                        "reference_sequence": item.get("reference_sequence"),
                        "semantic_group": item.get("semantic_group"),
                        "semantic_operation": item.get("semantic_operation"),
                        "reference_name": item.get("reference_name"),
                        "reference_type": item.get("reference_type"),
                        "candidate_sequence": item.get("candidate_sequence"),
                        "candidate_name": item.get("candidate_name"),
                        "candidate_type": item.get("candidate_type"),
                        "candidate_choices": item.get("candidate_choices"),
                        "selection_policy": item.get("selection_policy"),
                        "shape": "x".join(str(value) for value in shape) if shape else None,
                        "status": item.get("status"),
                        "cosine": metric.get("cosine"),
                        "relative_l2": metric.get("relative_l2"),
                        "mae": metric.get("mae"),
                        "rmse": metric.get("rmse"),
                        "max_abs": metric.get("max_abs"),
                        "feature_argmax_agreement": metric.get("top1_agreement"),
                        "exact_equal": metric.get("exact_equal"),
                        "reference_min": reference_range[0],
                        "reference_max": reference_range[1],
                        "candidate_min": candidate_range[0],
                        "candidate_max": candidate_range[1],
                    }
                )
    os.replace(temporary, path)


def positive_int(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("--nums must be a positive integer")
    return int(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Validate LocateAnything Vision or Language across execution stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  --mode float        --level small --phase vision --input_dir INPUTS --model_path CHECKPOINT
  --mode quantized-eager --level small --phase vision --input_dir INPUTS --model_path CHECKPOINT
  --mode float        --level small --phase language --input_dir GENERATED --model_path CHECKPOINT
  --mode quantized-eager --level medium --phase language --input_dir GENERATED --model_path CHECKPOINT
  --mode exported-bc  --level small --phase language --input_dir GENERATED --model_path LANGUAGE_BC_OR_DIR
  --mode converted-bc --level small --phase language --input_dir GENERATED --model_path LANGUAGE_BC_OR_DIR
  --mode hbm          --level small --phase language --input_dir GENERATED --model_path LANGUAGE_HBM
  --mode exported-bc  --level small --phase vision --input_dir INPUTS --model_path MODEL.visual.bc
  --mode converted-bc --level small --phase vision --input_dir INPUTS --model_path MODEL.visual_convert.bc
  --mode hbm          --level small --phase vision --input_dir INPUTS --model_path MODEL.hbm
  --mode analysis     --output_dir OUTPUTS
Add --nums N to process an exact subset; omit it to process all inputs.
All modes share --output_dir. The default is:
  workspace/evaluation/pipeline
""",
    )
    root.add_argument("--mode", choices=MODES, required=True, help="one execution stage")
    root.add_argument(
        "--level", choices=LEVELS, default="small",
        help="small=final output, medium=major modules, high=leaf operators",
    )
    root.add_argument(
        "--phase", choices=PHASES, default="vision",
        help="model phase; Language BC uses the Prefill, PBD q=6, and AR q=1 graph bundle",
    )
    root.add_argument(
        "--nums", type=positive_int,
        help="exact number of inputs to process; omitted means all discovered inputs",
    )
    root.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="shared stage outputs and analysis reports",
    )
    root.add_argument(
        "--scale-manifest", "--scale_manifest", dest="scale_manifest", type=Path,
        help="frozen calibration Scale manifest; required when no single current manifest exists",
    )
    root.add_argument(
        "--input_dir", type=Path,
        help="directory scanned recursively for calibration tensor inputs",
    )
    root.add_argument(
        "--model_path", type=Path,
        help="checkpoint, BC, or HBM selected by --mode",
    )
    return root


def validate_args(root: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.phase == "full_model":
        root.error("--phase full_model is not implemented")
    if args.mode == "analysis":
        return
    language_modes = {"float", QUANTIZED_EAGER_MODE, *BC_MODES, "hbm"}
    if args.phase == "language" and args.mode not in language_modes:
        root.error("Language supports Float, quantized-eager, BC, and HBM modes")
    if args.phase == "language" and args.mode in {*BC_MODES, "hbm"} and args.level != "small":
        root.error("Language compiled artifacts support --level small only")
    if args.mode == "hbm" and args.level != "small":
        root.error("HBM exposes only final graph outputs; use --level small")
    if args.input_dir is None or args.model_path is None:
        root.error(f"--mode {args.mode} requires --input_dir and --model_path")
    if not args.input_dir.is_dir():
        root.error(f"--input_dir is not a directory: {args.input_dir}")
    if not args.model_path.exists():
        root.error(f"--model_path does not exist: {args.model_path}")


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    validate_args(root, args)
    if args.mode == "float":
        return run_float_collection(args)
    if args.mode == QUANTIZED_EAGER_MODE:
        return run_quantized_eager_collection(args)
    if args.mode in BC_MODES:
        if args.phase == "language":
            return run_language_bc_collection(args)
        return run_bc_collection(args)
    if args.mode == "hbm":
        if args.phase == "language":
            return run_language_bc_collection(args)
        return run_hbm_collection(args)
    if args.mode == "analysis":
        from compiler.scripts.validate.analyze_pipeline import run_analysis_collection

        return run_analysis_collection(args)
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
