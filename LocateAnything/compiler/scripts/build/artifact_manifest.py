#!/usr/bin/env python3
"""Write and validate provenance for reusable exported BC artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} expects NAME=VALUE; got {value!r}")
        if key in parsed:
            raise ValueError(f"duplicate {option} key: {key}")
        parsed[key] = raw
    return parsed


def required_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing or empty: {resolved}")
    return resolved


def source_tree_identity(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"compiler source directory is missing: {root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not {"__pycache__", ".pytest_cache"} & set(path.parts)
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        total_bytes += len(content)
    return {
        "path": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def installed_distributions(import_name: str) -> dict[str, str]:
    distributions = importlib.metadata.packages_distributions().get(import_name, [])
    versions: dict[str, str] = {}
    for distribution in sorted(set(distributions)):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions or {import_name: "unavailable"}


def toolchain_identity() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "hbdk4": installed_distributions("hbdk4"),
        "leap_llm": installed_distributions("leap_llm"),
        "torch": installed_distributions("torch"),
    }


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"model directory is missing: {model_path}")
    metadata = []
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_path / name
        if path.is_file():
            metadata.append(
                {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    weights = []
    for path in sorted(model_path.glob("*.safetensors")):
        stat = path.stat()
        weights.append(
            {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    if not weights:
        raise RuntimeError(f"checkpoint has no safetensors weights: {model_path}")
    return {"path": str(model_path), "metadata": metadata, "weights": weights}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = parse_pairs(args.artifact, "--artifact")
    fields = parse_pairs(args.field, "--field")
    artifact_identity = {}
    for name, raw_path in sorted(artifacts.items()):
        path = required_file(Path(raw_path), f"artifact {name}")
        stat = path.stat()
        artifact_identity[name] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    scale = required_file(args.scale_manifest, "calibration scale manifest")
    rotation: dict[str, Any]
    if args.disable_hidden_rotation:
        rotation = {"mode": "disabled"}
    elif args.hidden_rotation_path:
        path = required_file(args.hidden_rotation_path, "hidden rotation")
        rotation = {"mode": "file", "path": str(path), "sha256": sha256_file(path)}
    else:
        rotation = {"mode": "built-in"}
    return {
        "schema_version": 1,
        "component": args.component,
        "contract": dict(sorted(fields.items())),
        "compiler_source": source_tree_identity(args.source_root),
        "toolchain": toolchain_identity(),
        "model": checkpoint_identity(args.model_path),
        "calibration_scale_manifest": {
            "path": str(scale),
            "sha256": sha256_file(scale),
        },
        "hidden_rotation": rotation,
        "artifacts": artifact_identity,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "check"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--component", choices=("vision", "language"), required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--scale_manifest", type=Path, required=True)
    parser.add_argument("--hidden_rotation_path", type=Path)
    parser.add_argument("--disable_hidden_rotation", action="store_true")
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    payload = build_payload(args)
    if args.action == "write":
        atomic_json(args.manifest, payload)
        print(f"[PASS] wrote reusable BC manifest: {args.manifest}", flush=True)
        return 0
    if not args.manifest.is_file():
        print(f"[MISS] reusable BC manifest is missing: {args.manifest}", flush=True)
        return 1
    try:
        recorded = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[MISS] reusable BC manifest is invalid: {exc}", flush=True)
        return 1
    if recorded != payload:
        changed = sorted(
            key for key in set(recorded) | set(payload)
            if recorded.get(key) != payload.get(key)
        )
        print(
            "[MISS] reusable BC provenance changed: " + ", ".join(changed),
            flush=True,
        )
        return 1
    print(f"[PASS] reusable BC provenance: {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
