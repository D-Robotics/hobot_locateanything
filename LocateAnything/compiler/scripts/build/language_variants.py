#!/usr/bin/env python3
"""Convert and compile LocateAnything Language graph-family variants.

Legacy inputs contain Prefill, PBD q=6, and AR q=1. Fused-PBD inputs also
contain PBD q=7..12 and causal AR bridge q=2..5 profiles. Shared PBD graphs are
compiled once; the AR graph family is compiled for each requested core count.
"""

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


BASE_EXPECTED = {
    "prefill": ((1, 1, 152681), (1, 1024, 2, 128)),
    "decode": ((1, 6, 152681), (1, 6, 2, 128)),
    "decode_ar": ((1, 1, 152681), (1, 1, 2, 128)),
}
FUSED_PBD_STAGES = tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13))
FUSED_AR_STAGES = tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6))
FUSED_STAGES = FUSED_PBD_STAGES + FUSED_AR_STAGES
KNOWN_STAGES = set(BASE_EXPECTED) | set(FUSED_STAGES)


def expected_contract(name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if name in BASE_EXPECTED:
        return BASE_EXPECTED[name]
    prefix, value = name.rsplit("q", 1)
    q_len = int(value)
    if prefix not in {"decode_pbd_", "decode_ar_"}:
        raise ValueError(f"unsupported Language graph: {name}")
    return (1, q_len, 152681), (1, q_len, 2, 128)


def heading(value: str) -> None:
    print(f"\n================== {value} ==================", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_bc(bc_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(bc_dir.glob("*.bc")):
        if path.name.endswith("_convert.bc"):
            continue
        module = load(str(path))
        functions = list(module.functions)
        if len(functions) != 1:
            raise RuntimeError(f"{path} contains {len(functions)} functions")
        function = functions[0]
        name = str(function.name)
        if name not in KNOWN_STAGES:
            continue
        if name in discovered:
            raise RuntimeError(f"duplicate {name} BC: {discovered[name]} and {path}")
        if len(function.inputs) != 75 or len(function.outputs) != 73:
            raise RuntimeError(
                f"{name} contract is {len(function.inputs)} inputs and "
                f"{len(function.outputs)} outputs; expected 75 and 73"
            )
        logits_shape = tuple(function.outputs[0].type.shape)
        cache_shape = tuple(function.outputs[1].type.shape)
        expected_logits, expected_cache = expected_contract(name)
        if logits_shape != expected_logits or cache_shape != expected_cache:
            raise RuntimeError(
                f"{name} output mismatch: logits={logits_shape}, "
                f"cache={cache_shape}; expected {(expected_logits, expected_cache)}"
            )
        discovered[name] = path.resolve()
        print(
            f"[PASS] {name}: logits={logits_shape} cache={cache_shape} "
            f"inputs=75 outputs=73",
            flush=True,
        )
    missing = sorted(set(BASE_EXPECTED) - set(discovered))
    if missing:
        raise RuntimeError(f"missing BC graphs in {bc_dir}: {missing}")
    fused_present = set(discovered) & set(FUSED_STAGES)
    if fused_present and fused_present != set(FUSED_STAGES):
        missing_fused = sorted(set(FUSED_STAGES) - fused_present)
        raise RuntimeError(
            "fused PBD graph family is incomplete: "
            f"present={sorted(fused_present)} missing={missing_fused}"
        )
    return discovered


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_or_create_manifest(
    path: Path,
    source_bc: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "source_bc": {
            name: {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
            for name, source in source_bc.items()
        },
        "march": args.march,
        "prefill_core_num": args.prefill_core_num,
        "decode_core_num": args.decode_core_num,
        "ar_core_nums": args.ar_core_nums,
        "jobs": args.jobs,
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(
                f"{path} belongs to a different source/configuration; "
                "use a new output directory"
            )
    else:
        atomic_json(path, payload)
    return payload


def valid_function(path: Path, expected_name: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        module = load(str(path))
        functions = list(module.functions)
        return len(functions) == 1 and str(functions[0].name) == expected_name
    except Exception:
        return False


def convert_stage(source: Path, destination: Path, name: str,
                  march: str, resume: bool) -> None:
    if resume and valid_function(destination, name):
        print(f"[RESUME] converted {name}: {destination}", flush=True)
        return
    heading(f"CONVERT {name.upper()}")
    converted = Model.convert_mlir(
        load(str(source)),
        enable_vpu=True,
        march=march,
        dynamic_quant=True,
    )
    function = converted.functions[0]
    if str(function.name) != name:
        raise RuntimeError(f"converted function is {function.name}, expected {name}")
    function.remove_io_op(["Dequantize", "Quantize"])
    temporary = destination.with_name(destination.stem + ".partial.bc")
    save(converted, str(temporary))
    os.replace(temporary, destination)
    print(f"[PASS] converted {name}: {destination}", flush=True)


def valid_hbo(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        Hbo(str(path))
        return True
    except Exception:
        return False


def compile_stage(converted_bc: Path, destination: Path, name: str,
                  core_num: int, args: argparse.Namespace) -> None:
    if args.resume and valid_hbo(destination):
        print(f"[RESUME] HBO {name} core={core_num}: {destination}", flush=True)
        return
    heading(f"COMPILE {name.upper()} CORE={core_num}")
    module = load(str(converted_bc))
    temporary = destination.with_name(destination.stem + ".partial.hbo")
    kwargs = {
        "march": args.march,
        "jobs": args.jobs,
        "progress_bar": True,
        "max_time_per_fc": 0.0,
        "opt": 2,
        "debug": False,
        "advice": 0.0,
        "balance": 100,
        "enable_hpc": True,
        "input_no_padding": True,
        "output_no_padding": True,
        "core_num": core_num,
    }
    if core_num > 1:
        kwargs["max_l2m_size"] = 25165824
    Model.compile_hbo(module, save_path=str(temporary), **kwargs)
    os.replace(temporary, destination)
    Hbo(str(destination))
    print(f"[PASS] HBO {name} core={core_num}: {destination}", flush=True)


def link_variant(hbos: list[Path], destination: Path,
                 resume: bool) -> None:
    if resume and destination.is_file() and destination.stat().st_size > 0:
        try:
            Hbm(str(destination))
            print(f"[RESUME] HBM: {destination}", flush=True)
            return
        except Exception:
            pass
    heading(f"LINK {destination.name}")
    temporary = destination.with_name(destination.stem + ".partial.hbm")
    Model.link_models([Hbo(str(path)) for path in hbos], str(temporary))
    os.replace(temporary, destination)
    Hbm(str(destination))
    print(f"[PASS] HBM: {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--march", default="nash-p")
    parser.add_argument("--prefill_core_num", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--decode_core_num", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument(
        "--ar_core_nums", type=int, nargs="+", choices=(1, 2, 4),
        default=[1, 2, 4],
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--convert_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.bc_dir = args.bc_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.ar_core_nums = sorted(set(args.ar_core_nums))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted_dir = args.output_dir / "converted_bc"
    hbo_dir = args.output_dir / "hbo"
    converted_dir.mkdir(exist_ok=True)
    hbo_dir.mkdir(exist_ok=True)

    heading("SOURCE CONTRACT")
    source_bc = discover_bc(args.bc_dir)
    validate_or_create_manifest(
        args.output_dir / "compile_manifest.json", source_bc, args,
    )

    stage_order = [
        name
        for name in ("prefill", "decode", *FUSED_PBD_STAGES,
                     "decode_ar", *FUSED_AR_STAGES)
        if name in source_bc
    ]
    converted = {
        name: converted_dir / f"{name}_convert.bc" for name in stage_order
    }
    for name in stage_order:
        convert_stage(source_bc[name], converted[name], name, args.march, args.resume)
    if args.convert_only:
        heading("CONVERT ONLY COMPLETED")
        return 0

    prefill_hbo = hbo_dir / f"prefill_core{args.prefill_core_num}.hbo"
    compile_stage(
        converted["prefill"], prefill_hbo, "prefill",
        args.prefill_core_num, args,
    )
    shared_hbos = [prefill_hbo]
    for name in ("decode", *FUSED_PBD_STAGES):
        if name not in converted:
            continue
        hbo = hbo_dir / f"{name}_core{args.decode_core_num}.hbo"
        compile_stage(converted[name], hbo, name, args.decode_core_num, args)
        shared_hbos.append(hbo)
    for ar_core in args.ar_core_nums:
        ar_hbos: list[Path] = []
        for name in ("decode_ar", *FUSED_AR_STAGES):
            if name not in converted:
                continue
            ar_hbo = hbo_dir / f"{name}_core{ar_core}.hbo"
            compile_stage(converted[name], ar_hbo, name, ar_core, args)
            ar_hbos.append(ar_hbo)
        fused_suffix = "_fusedpbd" if set(FUSED_STAGES) <= set(source_bc) else ""
        hbm = args.output_dir / (
            "LocateAnything-3B_language_chunk_1024_cache_4096_"
            "decoder_w8_lmhead_w8_nash-p_"
            f"prefill{args.prefill_core_num}_pbd{args.decode_core_num}_ar{ar_core}"
            f"{fused_suffix}.hbm"
        )
        link_variant([*shared_hbos, *ar_hbos], hbm, args.resume)

    heading("ALL VARIANTS COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
