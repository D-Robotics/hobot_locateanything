"""Structural metadata used to resume long-running calibration jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path: Path, *, normalize_text: bool = False) -> dict[str, Any]:
    del normalize_text
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"required file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size}


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"checkpoint index is missing: {index_path}")
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"checkpoint index has no weight_map: {index_path}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards = {name: file_identity(model_path / name) for name in shard_names}
    return {
        "model_path": str(model_path),
        "index": file_identity(index_path),
        "config": file_identity(model_path / "config.json"),
        "shards": shards,
    }


def release_checkpoint_errors(model_path: Path) -> list[str]:
    """Validate that the model directory contains a complete checkpoint."""

    try:
        checkpoint_identity(model_path)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return [f"checkpoint is incomplete: {exc}"]
    return []


def tokenizer_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    files = {
        name: file_identity(model_path / name)
        for name in TOKENIZER_FILES
        if (model_path / name).is_file()
    }
    if "tokenizer_config.json" not in files:
        raise RuntimeError(f"tokenizer_config.json is missing from {model_path}")
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files.keys():
        raise RuntimeError(f"model directory has no complete tokenizer files: {model_path}")
    return {"model_path": str(model_path), "files": files}


def rotation_identity(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"mode": "built-in"}
    resolved = Path(path).expanduser().resolve()
    return {"mode": "file", "file": file_identity(resolved)}


def identity_mismatches(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                mismatches.append(path)
            else:
                mismatches.extend(identity_mismatches(expected[key], actual[key], path))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [prefix]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatches.extend(identity_mismatches(left, right, f"{prefix}[{index}]"))
        return mismatches
    return [] if expected == actual else [prefix or "<root>"]


def artifact_identities(paths: Iterable[Path]) -> dict[str, Any]:
    return {
        path.resolve().name: file_identity(path.resolve())
        for path in paths
    }


def prepared_bundle_identity_errors(
    *,
    selected_jsonl: Path,
    generated_jsonl: Path,
    model_path: Path,
    prepare_source_path: Path,
    upstream_repo: Path | None = None,
    expected_sample_count: int | None = None,
) -> list[str]:
    """Validate prepared calibration structure without content fingerprints."""

    del prepare_source_path, upstream_repo
    selected_jsonl = selected_jsonl.resolve()
    generated_jsonl = generated_jsonl.resolve()
    identity_path = generated_jsonl.parent / "prepare_run_identity.json"
    summary_path = generated_jsonl.parent / "generation_summary.json"
    required = (
        ("selected data", selected_jsonl),
        ("generated data", generated_jsonl),
        ("Prepare metadata", identity_path),
        ("Prepare summary", summary_path),
    )
    errors = [f"{label} is missing: {path}" for label, path in required if not path.is_file()]
    errors.extend(release_checkpoint_errors(model_path))
    try:
        tokenizer_identity(model_path)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(f"tokenizer is incomplete: {exc}")
    if errors:
        return errors

    try:
        prepare_identity = read_json(identity_path)
        summary = read_json(summary_path)
        if not isinstance(prepare_identity, dict):
            return ["Prepare metadata must be a JSON object"]
        if not isinstance(summary, dict):
            return ["Prepare summary must be a JSON object"]
        if summary.get("generated_manifest") != generated_jsonl.name:
            errors.append("Prepare summary generated data filename mismatch")
        for field in ("fixed_profile", "generation_config"):
            left = prepare_identity.get(field)
            right = summary.get(field)
            if left is not None and right is not None and identity_mismatches(left, right):
                errors.append(f"Prepare metadata/summary {field} mismatch")

        selected_count = sum(
            bool(line.strip())
            for line in selected_jsonl.read_text(encoding="utf-8").splitlines()
        )
        generated_records = [
            json.loads(line)
            for line in generated_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if expected_sample_count is not None:
            if selected_count != expected_sample_count:
                errors.append(
                    f"selected data has {selected_count} records; expected {expected_sample_count}"
                )
            if len(generated_records) != expected_sample_count:
                errors.append(
                    f"generated data has {len(generated_records)} records; "
                    f"expected {expected_sample_count}"
                )
            if summary.get("sample_count") != expected_sample_count:
                errors.append("Prepare summary sample_count does not match the required count")
        for index, record in enumerate(generated_records, 1):
            tensor_value = record.get("tensor_file")
            if not isinstance(tensor_value, str) or not tensor_value:
                errors.append(f"generated data row {index} has no tensor_file")
                continue
            tensor_path = (generated_jsonl.parent / tensor_value).resolve()
            try:
                tensor_path.relative_to(generated_jsonl.parent.resolve())
            except ValueError:
                errors.append(f"generated data row {index} tensor escapes its directory")
                continue
            if not tensor_path.is_file() or tensor_path.stat().st_size == 0:
                errors.append(f"prepared tensor is missing or empty: {tensor_value}")
            if len(errors) >= 20:
                break
    except (OSError, ValueError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot validate prepared calibration data: {exc}")
    return errors
