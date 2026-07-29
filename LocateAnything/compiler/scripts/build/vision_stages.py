#!/usr/bin/env python3
"""Continue the LocateAnything Vision build from an exported visual BC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hbdk4.compiler import load, save
from hbdk4.compiler.hbm import Hbm, Hbo

from leap_llm.nn.utils import Model


INPUT_SHAPE = (1, 2304, 588)
OUTPUT_SHAPE = (1, 576, 2048)


def heading(value: str) -> None:
    print(f"\n================== {value} ==================", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_digest(path: Path) -> None:
    sidecar = digest_path(path)
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(sha256_file(path) + "\n", encoding="ascii")
    os.replace(temporary, sidecar)


def digest_matches(path: Path) -> bool:
    sidecar = digest_path(path)
    if not sidecar.is_file():
        return False
    try:
        return sidecar.read_text(encoding="ascii").strip() == sha256_file(path)
    except OSError:
        return False


def invalidate_stage_digests(root: Path, artifact_prefix: str) -> int:
    removed = 0
    for sidecar in root.glob(f"{artifact_prefix}*.sha256"):
        if sidecar.is_file():
            sidecar.unlink()
            removed += 1
    return removed


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_visual_bc(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"visual BC is missing: {path}")
    module = load(str(path))
    functions = list(module.functions)
    if len(functions) != 1 or str(functions[0].name) != "visual":
        names = [str(function.name) for function in functions]
        raise RuntimeError(f"visual BC must contain only graph 'visual'; got {names}")
    function = functions[0]
    if len(function.inputs) != 1 or len(function.outputs) != 1:
        raise RuntimeError(
            "visual BC must expose one input and one output; "
            f"got {len(function.inputs)} and {len(function.outputs)}"
        )
    input_shape = tuple(function.inputs[0].type.shape)
    output_shape = tuple(function.outputs[0].type.shape)
    if input_shape != INPUT_SHAPE or output_shape != OUTPUT_SHAPE:
        raise RuntimeError(
            f"visual BC shape mismatch: input={input_shape} output={output_shape}; "
            f"expected {INPUT_SHAPE} -> {OUTPUT_SHAPE}"
        )
    print(
        f"[PASS] visual: input={input_shape} output={output_shape}",
        flush=True,
    )


def valid_function(path: Path, expected_name: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0 or not digest_matches(path):
        return False
    try:
        module = load(str(path))
        functions = list(module.functions)
        return len(functions) == 1 and str(functions[0].name) == expected_name
    except Exception:
        return False


def valid_hbo(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0 or not digest_matches(path):
        return False
    try:
        Hbo(str(path))
        return True
    except Exception:
        return False


def hbm_contract_matches(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        model = Hbm(str(path))
        graphs = {str(graph.name): graph for graph in model.graphs}
        if set(graphs) != {"visual"}:
            return False
        graph = graphs["visual"]
        if len(graph.inputs) != 1 or len(graph.outputs) != 1:
            return False
        return (
            tuple(graph.inputs[0].type.shape) == INPUT_SHAPE
            and tuple(graph.outputs[0].type.shape) == OUTPUT_SHAPE
        )
    except Exception:
        return False


def valid_hbm(path: Path) -> bool:
    return digest_matches(path) and hbm_contract_matches(path)


def validate_manifest(path: Path, source: Path, args: argparse.Namespace) -> bool:
    payload = {
        "schema_version": 2,
        "source_bc": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "march": args.march,
        "core_num": args.core_num,
        "jobs": args.jobs,
        "hbm_path": str(args.hbm_path),
    }
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous == payload:
            return True
        atomic_json(path, payload)
        return False
    atomic_json(path, payload)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_path", type=Path, required=True)
    parser.add_argument("--hbm_path", type=Path, required=True)
    parser.add_argument("--march", default="nash-p")
    parser.add_argument("--core_num", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.bc_path = args.bc_path.resolve()
    args.hbm_path = args.hbm_path.resolve()
    args.hbm_path.parent.mkdir(parents=True, exist_ok=True)

    heading("SOURCE CONTRACT")
    validate_visual_bc(args.bc_path)
    if args.check_only:
        heading("SOURCE CONTRACT PASSED")
        return 0

    manifest = args.hbm_path.with_suffix(".compile_manifest.json")
    manifest_compatible = validate_manifest(manifest, args.bc_path, args)
    if args.resume and not manifest_compatible:
        removed = invalidate_stage_digests(
            args.hbm_path.parent,
            args.hbm_path.stem,
        )
        print(
            "[RESUME] source or compile contract changed; rebuilding Converted BC, "
            f"HBO, and HBM from the reusable source BC (invalidated={removed})",
            flush=True,
        )
        args.resume = False

    converted_path = args.hbm_path.with_suffix(".visual_convert.bc")
    if args.resume and valid_function(converted_path, "visual"):
        print(f"[RESUME] converted visual: {converted_path}", flush=True)
    else:
        heading("CONVERT VISUAL")
        converted = Model.convert_mlir(
            load(str(args.bc_path)),
            enable_vpu=True,
            march=args.march,
            dynamic_quant=True,
        )
        function = converted.functions[0]
        if str(function.name) != "visual":
            raise RuntimeError(
                f"converted function is {function.name}, expected visual"
            )
        function.remove_io_op(["Dequantize", "Quantize"])
        temporary = converted_path.with_name(converted_path.stem + ".partial.bc")
        save(converted, str(temporary))
        os.replace(temporary, converted_path)
        write_digest(converted_path)
        print(f"[PASS] converted visual: {converted_path}", flush=True)

    hbo_path = args.hbm_path.with_suffix(".visual.hbo")
    if args.resume and valid_hbo(hbo_path):
        print(f"[RESUME] HBO visual core={args.core_num}: {hbo_path}", flush=True)
    else:
        heading(f"COMPILE VISUAL CORE={args.core_num}")
        module = load(str(converted_path))
        temporary = hbo_path.with_name(hbo_path.stem + ".partial.hbo")
        kwargs = {
            "march": args.march,
            "jobs": args.jobs,
            "progress_bar": True,
            "max_time_per_fc": 0.0,
            "opt": 2,
            "debug": False,
            "advice": 0.0,
            "balance": 100,
            "input_no_padding": True,
            "output_no_padding": True,
            "core_num": args.core_num,
        }
        if args.core_num > 1:
            kwargs["max_l2m_size"] = 25165824
        Model.compile_hbo(module, save_path=str(temporary), **kwargs)
        os.replace(temporary, hbo_path)
        Hbo(str(hbo_path))
        write_digest(hbo_path)
        print(f"[PASS] HBO visual core={args.core_num}: {hbo_path}", flush=True)

    if args.resume and valid_hbm(args.hbm_path):
        print(f"[RESUME] HBM: {args.hbm_path}", flush=True)
        return 0

    heading(f"LINK {args.hbm_path.name}")
    temporary = args.hbm_path.with_name(args.hbm_path.stem + ".partial.hbm")
    Model.link_models([Hbo(str(hbo_path))], str(temporary))
    os.replace(temporary, args.hbm_path)
    if not hbm_contract_matches(args.hbm_path):
        raise RuntimeError(f"linked HBM graph contract mismatch: {args.hbm_path}")
    write_digest(args.hbm_path)
    print(f"[PASS] HBM: {args.hbm_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
