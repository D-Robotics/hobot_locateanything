"""Structured, interruption-tolerant tensor tracing for LA verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

TRACE_MODES = ("off", "summary", "full")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_name(name: str) -> str:
    module_paths = re.findall(
        r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+(?![A-Za-z0-9_])",
        name,
    )
    if module_paths:
        value = module_paths[-1]
    else:
        quoted = [item.strip("\\") for item in re.findall(r'"([^"]+)"', name)]
        value = quoted[-1] if quoted else name
    for prefix in ("model.", "_orig_mod."):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def _is_tensor(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    module = type(value).__module__
    if module.startswith("torch") and hasattr(value, "detach"):
        return True
    tensor_type = getattr(value, "type", None)
    return hasattr(tensor_type, "shape") and hasattr(tensor_type, "np_dtype")


def iter_tensors(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if _is_tensor(value):
        yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_tensors(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_tensors(item, f"{path}.{index}")
        return
    hidden = getattr(value, "last_hidden_state", None)
    if hidden is not None:
        yield from iter_tensors(hidden, f"{path}.last_hidden_state")


def _to_numpy(value: Any) -> tuple[np.ndarray | None, str, str, list[int]]:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return array, str(array.dtype), "cpu", list(array.shape)
    module = type(value).__module__
    if module.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach()
        original_dtype = str(tensor.dtype).replace("torch.", "")
        device = str(tensor.device)
        cpu = tensor.cpu()
        if original_dtype == "bfloat16":
            cpu = cpu.float()
        array = np.ascontiguousarray(cpu.numpy())
        return array, original_dtype, device, list(tensor.shape)
    tensor_type = getattr(value, "type", None)
    if hasattr(tensor_type, "shape") and hasattr(tensor_type, "np_dtype"):
        return (
            None,
            str(np.dtype(tensor_type.np_dtype)),
            "descriptor",
            [int(item) for item in tensor_type.shape],
        )
    raise TypeError(f"unsupported tensor value: {type(value)}")


def _sample(array: np.ndarray, count: int = 4) -> dict[str, list[Any]]:
    flat = array.reshape(-1)

    def values(items: np.ndarray) -> list[Any]:
        result: list[Any] = []
        for item in items:
            scalar = item.item()
            if isinstance(scalar, float) and not np.isfinite(scalar):
                result.append(None)
            else:
                result.append(scalar)
        return result

    return {"head": values(flat[:count]), "tail": values(flat[-count:])}


def _statistics(array: np.ndarray) -> dict[str, Any]:
    if array.size == 0 or array.dtype.kind not in "biufc":
        return {"elements": int(array.size)}
    flat = array.reshape(-1)
    finite = nonzero = count = 0
    total = square_total = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    absolute_maximum = 0.0
    for start in range(0, flat.size, 1_000_000):
        chunk = flat[start:start + 1_000_000]
        if np.iscomplexobj(chunk):
            chunk = np.abs(chunk)
        values = chunk.astype(np.float64, copy=False)
        mask = np.isfinite(values)
        finite += int(mask.sum())
        nonzero += int(np.count_nonzero(values))
        valid = values[mask]
        if valid.size == 0:
            continue
        count += int(valid.size)
        total += float(valid.sum())
        square_total += float(np.dot(valid, valid))
        minimum = min(minimum, float(valid.min()))
        maximum = max(maximum, float(valid.max()))
        absolute_maximum = max(absolute_maximum, float(np.abs(valid).max()))
    result: dict[str, Any] = {
        "elements": int(flat.size),
        "finite_ratio": finite / flat.size,
        "nonzero_ratio": nonzero / flat.size,
        "sample": _sample(array),
    }
    if count:
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        result.update(
            min=minimum,
            max=maximum,
            mean=mean,
            std=variance**0.5,
            absmax=absolute_maximum,
            l2=square_total**0.5,
        )
    return result


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[-80:] or "tensor"


def semantic_location(stage: str, kind: str, name: str) -> tuple[str, str]:
    """Map implementation paths to the actual LocateAnything execution stages."""
    value = canonical_name(name)
    lowered = value.lower()
    if kind == "model_input":
        return "INPUT", ""
    if kind.endswith("_output") or kind == "float_graph":
        return "FINAL OUTPUT", ""

    block = re.search(r"(?:encoder\.)?blocks\.(\d+)(?:\.(.+))?$", lowered)
    if block:
        operation = (block.group(2) or "BLOCK OUTPUT").replace(".", " / ").upper()
        operation = operation.replace("WQKV", "QKV")
        return f"BLOCK {int(block.group(1)) + 1}", operation
    layer = re.search(r"(?:language_model\.)?layers\.(\d+)(?:\.(.+))?$", lowered)
    if layer:
        operation = (layer.group(2) or "BLOCK OUTPUT").replace(".", " / ").upper()
        return f"DECODER BLOCK {int(layer.group(1)) + 1}", operation
    if "patch_embed" in lowered:
        suffix = value.rsplit("patch_embed", 1)[-1].strip(".")
        operation = suffix.replace(".", " / ").upper() if suffix else "PATCH EMBEDDING OUTPUT"
        return "PATCH EMBEDDING", operation
    if "final_layernorm" in lowered:
        return "FINAL LAYERNORM", "FINAL LAYERNORM"
    if "merger" in lowered or "mlp1" in lowered:
        suffix = value.rsplit("merger", 1)[-1].strip(".")
        operation = suffix.replace(".", " / ").upper() if suffix else "PATCH MERGER OUTPUT"
        return "PATCH MERGER / PROJECTOR", operation
    if "embed_tokens" in lowered:
        return "TOKEN EMBEDDING", "EMBED TOKENS"
    if lowered.endswith(".lm_head") or lowered == "lm_head":
        return "LM HEAD", "LM HEAD"
    if lowered.endswith(".norm") or lowered == "norm":
        return "FINAL RMSNORM", "RMSNORM"
    return f"{stage.upper()} OPERATIONS", value.replace(".", " / ").upper()


class TraceRecorder:
    def __init__(self, run_dir: Path, stage: str, mode: str) -> None:
        if mode not in TRACE_MODES or mode == "off":
            raise ValueError(f"trace recorder requires summary or full mode, got {mode}")
        self.stage = stage
        self.mode = mode
        self.directory = run_dir / "traces" / stage
        self.tensor_directory = self.directory / "tensors"
        self.events_path = self.directory / "events.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        if mode == "full":
            self.tensor_directory.mkdir(parents=True, exist_ok=True)
        self.handle = self.events_path.open("w", encoding="utf-8")
        self.started = time.monotonic()
        self.sequence = 0
        self.tensor_count = 0
        self.dumped_files: dict[str, str] = {}
        self.dumped_bytes = 0
        self.closed = False
        self._summary: dict[str, Any] | None = None

    def __enter__(self) -> "TraceRecorder":
        print(f"[trace:{self.stage}] started mode={self.mode}", flush=True)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.handle.flush()
        self.handle.close()
        self.closed = True
        self._summary = {
            "mode": self.mode,
            "events": self.sequence,
            "tensors": self.tensor_count,
            "unique_tensor_files": len(self.dumped_files),
            "dumped_bytes": self.dumped_bytes,
            "events_file": str(self.events_path.resolve()),
            "summary_file": str(self.summary_path.resolve()),
            "tensor_dir": str(self.tensor_directory.resolve()) if self.mode == "full" else None,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        _atomic_json(self.summary_path, self._summary)
        print("\n==== TRACE SAVED ====", flush=True)
        print(f"STAGE: {self.stage}", flush=True)
        print(f"EVENTS: {self.sequence}", flush=True)
        print(f"TENSORS: {self.tensor_count}", flush=True)
        print(f"UNIQUE_TENSOR_FILES: {len(self.dumped_files)}", flush=True)
        print(f"SAVED_BYTES: {self.dumped_bytes}", flush=True)
        print(f"EVENTS_JSONL: {self.events_path.resolve()}", flush=True)
        print(f"SUMMARY_JSON: {self.summary_path.resolve()}", flush=True)
        if self.mode == "full":
            print(f"TENSOR_DIRECTORY: {self.tensor_directory.resolve()}", flush=True)
        print("=========================\n", flush=True)

    @property
    def summary(self) -> dict[str, Any]:
        if not self.closed:
            raise RuntimeError("trace summary is available after close")
        assert self._summary is not None
        return self._summary

    def _tensor_record(self, name: str, value: Any) -> dict[str, Any]:
        array, original_dtype, device, shape = _to_numpy(value)
        record: dict[str, Any] = {
            "path": name,
            "shape": shape,
            "dtype": original_dtype,
            "device": device,
            "materialized": array is not None,
        }
        self.tensor_count += 1
        if array is None:
            return record

        raw_digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        identity = hashlib.sha256()
        identity.update(f"{original_dtype}:{shape}:{raw_digest}".encode("utf-8"))
        tensor_id = identity.hexdigest()
        record.update(
            tensor_id=tensor_id,
            storage_dtype=str(array.dtype),
            bytes=array.nbytes,
            sha256=raw_digest,
            statistics=_statistics(array),
        )
        if self.mode == "full":
            relative = self.dumped_files.get(tensor_id)
            if relative is None:
                filename = f"{tensor_id[:16]}-{_safe_name(name)}.npy"
                path = self.tensor_directory / filename
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, array, allow_pickle=False)
                os.replace(temporary, path)
                relative = str(path.resolve())
                self.dumped_files[tensor_id] = relative
                self.dumped_bytes += path.stat().st_size
            record["file"] = relative
        return record

    def record(
        self,
        name: str,
        kind: str,
        type_name: str,
        inputs: Any,
        outputs: Any,
        duration_ms: float | None = None,
    ) -> None:
        input_records = [
            self._tensor_record(path, tensor)
            for path, tensor in iter_tensors(inputs, "input")
        ]
        output_records = [
            self._tensor_record(path, tensor)
            for path, tensor in iter_tensors(outputs, "output")
        ]
        event = {
            "sequence": self.sequence,
            "elapsed_ms": (time.monotonic() - self.started) * 1000.0,
            "stage": self.stage,
            "kind": kind,
            "name": name,
            "canonical_name": canonical_name(name),
            "type": type_name,
            "duration_ms": duration_ms,
            "inputs": input_records,
            "outputs": output_records,
        }
        event["semantic_group"], event["semantic_operation"] = semantic_location(
            self.stage, kind, name
        )
        self.handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        self.sequence += 1
        self._print_event(event, input_records, output_records)

    def _print_event(
        self,
        event: dict[str, Any],
        input_records: list[dict[str, Any]],
        output_records: list[dict[str, Any]],
    ) -> None:
        group = event["semantic_group"]
        if group != getattr(self, "_active_group", None):
            print(f"\n==== {group} ====", flush=True)
            self._active_group = group
            self._operation_index = 0
        if event["kind"] == "model_input":
            self._print_tensors("INPUT", output_records)
            return
        self._operation_index = getattr(self, "_operation_index", 0) + 1
        operation = event["semantic_operation"] or event["type"]
        print(f"=== {self._operation_index}. {operation} ===", flush=True)
        print(f"KIND: {event['kind']}", flush=True)
        print(f"NAME: {event['name']}", flush=True)
        print(f"TYPE: {event['type']}", flush=True)
        duration = event["duration_ms"]
        duration_text = f"{duration:.3f}" if duration is not None else "N/A"
        print(f"DURATION_MS: {duration_text}", flush=True)
        self._print_tensors("INPUT", input_records)
        self._print_tensors("OUTPUT", output_records)
        print("================================", flush=True)

    @staticmethod
    def _print_tensors(role: str, records: list[dict[str, Any]]) -> None:
        def display(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.8g}"
            if isinstance(value, list):
                return "[" + ", ".join(display(item) for item in value) + "]"
            return str(value)

        if not records:
            print(f"==== {role}: NONE ====", flush=True)
            return
        for index, tensor in enumerate(records, start=1):
            stats = tensor.get("statistics", {})
            sample = stats.get("sample", {})
            print(f"==== {role} {index}/{len(records)} ====", flush=True)
            print(f"PATH: {tensor['path']}", flush=True)
            print(f"SHAPE: {tensor['shape']}", flush=True)
            print(f"DTYPE: {tensor['dtype']}", flush=True)
            print(f"DEVICE: {tensor['device']}", flush=True)
            print(f"ELEMENTS: {display(stats.get('elements', 'N/A'))}", flush=True)
            print(f"MIN: {display(stats.get('min', 'N/A'))}", flush=True)
            print(f"MAX: {display(stats.get('max', 'N/A'))}", flush=True)
            print(f"MEAN: {display(stats.get('mean', 'N/A'))}", flush=True)
            print(f"STD: {display(stats.get('std', 'N/A'))}", flush=True)
            print(f"ABSMAX: {display(stats.get('absmax', 'N/A'))}", flush=True)
            print(f"L2: {display(stats.get('l2', 'N/A'))}", flush=True)
            print(f"FINITE_RATIO: {display(stats.get('finite_ratio', 'N/A'))}", flush=True)
            print(f"NONZERO_RATIO: {display(stats.get('nonzero_ratio', 'N/A'))}", flush=True)
            print(f"SAMPLE_HEAD: {display(sample.get('head', 'N/A'))}", flush=True)
            print(f"SAMPLE_TAIL: {display(sample.get('tail', 'N/A'))}", flush=True)

    def bc_callback(self, op: Any, results: Any, operands: Any) -> bool:
        if str(getattr(op, "type", "")) == "func.func":
            return True
        self.record(
            str(getattr(op, "name", "<unnamed>")),
            "bc_op",
            str(getattr(op, "type", type(op).__name__)),
            operands,
            results,
        )
        return True


@contextmanager
def trace_torch_modules(model: Any, recorder: TraceRecorder | None) -> Iterator[None]:
    if recorder is None:
        yield
        return
    handles: list[Any] = []
    starts: dict[str, list[float]] = {}

    def pre_hook(name: str):
        def hook(_module: Any, _inputs: Any) -> None:
            starts.setdefault(name, []).append(time.monotonic())

        return hook

    def post_hook(name: str, type_name: str):
        def hook(_module: Any, inputs: Any, outputs: Any) -> None:
            stack = starts.get(name, [])
            started = stack.pop() if stack else time.monotonic()
            recorder.record(
                name,
                "torch_module",
                type_name,
                inputs,
                outputs,
                duration_ms=(time.monotonic() - started) * 1000.0,
            )

        return hook

    semantic_boundaries = re.compile(
        r"(?:^|\.)(?:patch_embed|merger|blocks\.\d+|layers\.\d+|self_attn|mlp)$"
    )
    for name, module in model.named_modules():
        is_leaf = not any(module.children())
        if not name or (not is_leaf and not semantic_boundaries.search(name)):
            continue
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook(name, type(module).__name__)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
