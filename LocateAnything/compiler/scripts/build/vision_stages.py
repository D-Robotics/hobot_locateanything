#!/usr/bin/env python3
"""Continue the LocateAnything Vision build from an exported visual BC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hbdk4.compiler import load, save
from hbdk4.compiler.hbm import Hbm, Hbo

from leap_llm.nn.utils import Model


INPUT_SHAPE = (1, 2304, 588)
OUTPUT_SHAPE = (1, 576, 2048)
IO_DTYPE = "float16"


def heading(value: str) -> None:
    print(f"\n================== {value} ==================", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_dtype(value: Any) -> str:
    tensor_type = getattr(value, "type", None)
    raw = getattr(tensor_type, "np_dtype", None)
    if raw is None:
        raise RuntimeError("visual tensor descriptor has no np_dtype")
    text = str(raw).lower()
    for dtype in ("float16", "float32", "int8", "uint8", "int16", "int32", "int64"):
        if dtype in text:
            return dtype
    raise RuntimeError(f"unsupported visual tensor dtype: {raw!r}")


def validate_visual_function(function: Any) -> None:
    if str(function.name) != "visual":
        raise RuntimeError(f"visual graph name mismatch: {function.name}")
    if len(function.inputs) != 1 or len(function.outputs) != 1:
        raise RuntimeError(
            "visual graph must expose one input and one output; "
            f"got {len(function.inputs)} and {len(function.outputs)}"
        )
    input_shape = tuple(function.inputs[0].type.shape)
    output_shape = tuple(function.outputs[0].type.shape)
    input_dtype = canonical_dtype(function.inputs[0])
    output_dtype = canonical_dtype(function.outputs[0])
    if input_shape != INPUT_SHAPE or output_shape != OUTPUT_SHAPE:
        raise RuntimeError(
            f"visual graph shape mismatch: input={input_shape} output={output_shape}; "
            f"expected {INPUT_SHAPE} -> {OUTPUT_SHAPE}"
        )
    if input_dtype != IO_DTYPE or output_dtype != IO_DTYPE:
        raise RuntimeError(
            f"visual graph dtype mismatch: input={input_dtype} output={output_dtype}; "
            f"expected {IO_DTYPE} -> {IO_DTYPE}"
        )


def validate_visual_bc(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"visual BC is missing: {path}")
    module = load(str(path))
    functions = list(module.functions)
    if len(functions) != 1 or str(functions[0].name) != "visual":
        names = [str(function.name) for function in functions]
        raise RuntimeError(f"visual BC must contain only graph 'visual'; got {names}")
    function = functions[0]
    validate_visual_function(function)
    input_shape = tuple(function.inputs[0].type.shape)
    output_shape = tuple(function.outputs[0].type.shape)
    print(
        f"[PASS] visual: input={input_shape}/{IO_DTYPE} "
        f"output={output_shape}/{IO_DTYPE}",
        flush=True,
    )


def valid_function(path: Path, expected_name: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        module = load(str(path))
        functions = list(module.functions)
        if len(functions) != 1 or str(functions[0].name) != expected_name:
            return False
        validate_visual_function(functions[0])
        return True
    except Exception:
        return False


def valid_hbo(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
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
        validate_visual_function(graph)
        return True
    except Exception:
        return False


def valid_hbm(path: Path) -> bool:
    return hbm_contract_matches(path)


def write_compile_manifest(path: Path, source: Path, args: argparse.Namespace) -> None:
    payload = {
        "schema_version": 3,
        "source_bc": {
            "path": str(source),
            "bytes": source.stat().st_size,
        },
        "march": args.march,
        "core_num": args.core_num,
        "jobs": args.jobs,
        "hbm_path": str(args.hbm_path),
    }
    atomic_json(path, payload)


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
    write_compile_manifest(manifest, args.bc_path, args)

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
        validate_visual_bc(converted_path)
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
    print(f"[PASS] HBM: {args.hbm_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
