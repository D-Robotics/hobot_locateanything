#!/usr/bin/env python3
"""Convert and compile LocateAnything Language graph sets.

The standard set contains Prefill, PBD q=6, and AR q=1. The fused-decode set
also contains PBD q=7..12 and causal AR bridge q=2..5 graphs. Shared PBD graphs
are compiled once; the AR graph family is compiled for each requested core count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hbdk4.compiler import load, save
from hbdk4.compiler.hbm import Hbm, Hbo

from leap_llm.nn.utils import Model
from leap_llm.language_graphs import (
    LANGUAGE_GRAPH_SET_NAMES,
    LANGUAGE_GRAPH_SETS,
    language_graph_set,
)
from compiler.scripts.common.progress import StageProgress  # noqa: E402


BASE_EXPECTED = {
    "prefill": ((1, 1, 152681), (1, 1024, 2, 128)),
    "decode": ((1, 6, 152681), (1, 6, 2, 128)),
    "decode_ar": ((1, 1, 152681), (1, 1, 2, 128)),
}
KNOWN_STAGES = {
    graph
    for graph_set in LANGUAGE_GRAPH_SETS.values()
    for graph in graph_set.graphs
}
VOCAB_SIZE = 152681
HIDDEN_SIZE = 2048
NUM_LAYERS = 36
CACHE_LEN = 4096
NUM_KV_HEADS = 2
HEAD_DIM = 128
CACHE_TENSOR_COUNT = NUM_LAYERS * 2


def expected_contract(name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if name in BASE_EXPECTED:
        return BASE_EXPECTED[name]
    prefix, value = name.rsplit("q", 1)
    q_len = int(value)
    if prefix not in {"decode_pbd_", "decode_ar_"}:
        raise ValueError(f"unsupported Language graph: {name}")
    return (1, q_len, 152681), (1, q_len, 2, 128)


def query_length(name: str) -> int:
    if name == "prefill":
        return 1024
    if name == "decode":
        return 6
    if name == "decode_ar":
        return 1
    if name not in KNOWN_STAGES:
        raise ValueError(f"unsupported Language graph: {name}")
    return int(name.rsplit("q", 1)[1])


def graph_label(name: str) -> str:
    if name == "prefill":
        return "Prefill"
    if name == "decode":
        return "Decode PBD q=6"
    if name == "decode_ar":
        return "Decode AR q=1"
    if name.startswith("decode_pbd_q"):
        return f"Decode PBD q={query_length(name)}"
    if name.startswith("decode_ar_q"):
        return f"Decode AR q={query_length(name)}"
    raise ValueError(f"unsupported Language graph: {name}")


def _canonical_dtype(value: Any) -> str:
    tensor_type = getattr(value, "type", None)
    raw = getattr(tensor_type, "np_dtype", None)
    if raw is None:
        raise RuntimeError("tensor descriptor has no np_dtype")
    text = str(raw).lower()
    for dtype in (
        "float16", "float32", "int8", "uint8", "int16", "int32", "int64"
    ):
        if dtype in text:
            return dtype
    raise RuntimeError(f"unsupported tensor dtype: {raw!r}")


def _descriptor_contract(value: Any) -> tuple[tuple[int, ...], str]:
    tensor_type = getattr(value, "type", None)
    shape = tuple(getattr(tensor_type, "shape", ()))
    if not shape or not all(isinstance(axis, int) and axis > 0 for axis in shape):
        raise RuntimeError(f"tensor has non-static shape: {shape}")
    return shape, _canonical_dtype(value)


def expected_io_contract(
    name: str,
    *,
    cache_dtype: str = "float32",
) -> tuple[list[tuple[tuple[int, ...], str]], list[tuple[tuple[int, ...], str]]]:
    q_len = query_length(name)
    logits_shape, update_shape = expected_contract(name)
    cache_shape = (1, CACHE_LEN, NUM_KV_HEADS, HEAD_DIM)
    inputs = [
        ((1, q_len, HIDDEN_SIZE), "float16"),
        ((1, 1, q_len), "int32"),
        ((1, q_len, CACHE_LEN), "float16"),
        *[(cache_shape, cache_dtype) for _ in range(CACHE_TENSOR_COUNT)],
    ]
    outputs = [
        (logits_shape, "float16"),
        *[(update_shape, cache_dtype) for _ in range(CACHE_TENSOR_COUNT)],
    ]
    return inputs, outputs


def validate_graph_contract(
    function: Any,
    name: str,
    *,
    cache_dtype: str = "float32",
) -> None:
    actual_name = str(function.name)
    if actual_name != name:
        raise RuntimeError(
            f"Language graph name mismatch: expected {name}, got {actual_name}"
        )
    expected_inputs, expected_outputs = expected_io_contract(
        name, cache_dtype=cache_dtype
    )
    if len(function.inputs) != len(expected_inputs):
        raise RuntimeError(
            f"{name} contract has {len(function.inputs)} inputs; "
            f"expected {len(expected_inputs)}"
        )
    if len(function.outputs) != len(expected_outputs):
        raise RuntimeError(
            f"{name} contract has {len(function.outputs)} outputs; "
            f"expected {len(expected_outputs)}"
        )
    for direction, actual, expected in (
        ("input", function.inputs, expected_inputs),
        ("output", function.outputs, expected_outputs),
    ):
        for index, (descriptor, expected_descriptor) in enumerate(
            zip(actual, expected)
        ):
            try:
                actual_descriptor = _descriptor_contract(descriptor)
            except RuntimeError as exc:
                raise RuntimeError(f"{name} {direction}[{index}]: {exc}") from exc
            if actual_descriptor != expected_descriptor:
                raise RuntimeError(
                    f"{name} {direction}[{index}] mismatch: "
                    f"got {actual_descriptor}, expected {expected_descriptor}"
                )


def heading(value: str) -> None:
    print(f"\n================== {value} ==================", flush=True)


def discover_bc(
    bc_dir: Path,
    *,
    artifact_prefix: str | None = None,
    graph_set: str = "standard",
) -> dict[str, Path]:
    expected = set(language_graph_set(graph_set).graphs)
    discovered: dict[str, Path] = {}
    if artifact_prefix:
        candidates = [
            bc_dir / f"{artifact_prefix}.{name}.bc"
            for name in sorted(KNOWN_STAGES)
            if (bc_dir / f"{artifact_prefix}.{name}.bc").is_file()
        ]
    else:
        candidates = [
            path for path in sorted(bc_dir.glob("*.bc"))
            if not path.name.endswith(("_convert.bc", ".partial.bc"))
        ]
    for path in candidates:
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
        validate_graph_contract(function, name)
        logits_shape = tuple(function.outputs[0].type.shape)
        cache_shape = tuple(function.outputs[1].type.shape)
        discovered[name] = path.resolve()
        print(
            f"[PASS] {name}: logits={logits_shape} cache={cache_shape} "
            f"inputs=75 outputs=73",
            flush=True,
        )
    missing = sorted(expected - set(discovered))
    if missing:
        raise RuntimeError(f"missing BC graphs in {bc_dir}: {missing}")
    unexpected = sorted(set(discovered) - expected)
    if unexpected:
        raise RuntimeError(
            f"{graph_set} Language BC contains unexpected graphs: {unexpected}"
        )
    return discovered


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_compile_manifest(
    path: Path,
    source_bc: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema_version": 3,
        "source_bc": {
            name: {
                "path": str(source),
                "bytes": source.stat().st_size,
            }
            for name, source in source_bc.items()
        },
        "march": args.march,
        "prefill_core_num": args.prefill_core_num,
        "decode_core_num": args.decode_core_num,
        "ar_core_nums": args.ar_core_nums,
        "jobs": args.jobs,
        "graph_set": args.graph_set,
        "hbm_path": str(args.hbm_path) if args.hbm_path else None,
    }
    atomic_json(path, payload)


def valid_function(path: Path, expected_name: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        module = load(str(path))
        functions = list(module.functions)
        if len(functions) != 1:
            return False
        validate_graph_contract(functions[0], expected_name, cache_dtype="int8")
        return True
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
    # Removing the boundary quantization wrappers exposes the integer KV cache
    # contract consumed by HBO compilation and the linked HBM.
    validate_graph_contract(function, name, cache_dtype="int8")
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
    functions = list(module.functions)
    if len(functions) != 1:
        raise RuntimeError(f"{converted_bc} contains {len(functions)} functions")
    validate_graph_contract(functions[0], name, cache_dtype="int8")
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


def hbm_contract_matches(path: Path, expected_names: list[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        model = Hbm(str(path))
        graphs = {str(graph.name): graph for graph in model.graphs}
        if set(graphs) != set(expected_names):
            return False
        for name in expected_names:
            graph = graphs[name]
            validate_graph_contract(graph, name, cache_dtype="int8")
        return True
    except Exception:
        return False


def valid_hbm(path: Path, expected_names: list[str]) -> bool:
    return hbm_contract_matches(path, expected_names)


def link_variant(hbos: list[Path], destination: Path,
                 resume: bool, expected_names: list[str]) -> None:
    if resume and valid_hbm(destination, expected_names):
        print(f"[RESUME] HBM: {destination}", flush=True)
        return
    if resume and destination.is_file() and destination.stat().st_size > 0:
        try:
            Hbm(str(destination))
            print(f"[STALE] HBM graph contract mismatch: {destination}", flush=True)
        except Exception:
            pass
    heading(f"LINK {destination.name}")
    temporary = destination.with_name(destination.stem + ".partial.hbm")
    Model.link_models([Hbo(str(path)) for path in hbos], str(temporary))
    os.replace(temporary, destination)
    if not hbm_contract_matches(destination, expected_names):
        raise RuntimeError(f"linked HBM graph contract mismatch: {destination}")
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
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument(
        "--graph-set",
        dest="graph_set",
        choices=LANGUAGE_GRAPH_SET_NAMES,
        default="standard",
    )
    parser.add_argument(
        "--hbm_path", type=Path,
        help="Exact release HBM path; also selects its matching BC prefix.",
    )
    parser.add_argument("--embedding_path", type=Path)
    parser.add_argument("--expected_embedding_bytes", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.bc_dir = args.bc_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.hbm_path:
        args.hbm_path = args.hbm_path.resolve()
    if args.embedding_path:
        args.embedding_path = args.embedding_path.resolve()
    args.ar_core_nums = sorted(set(args.ar_core_nums))
    if args.hbm_path and len(args.ar_core_nums) != 1:
        raise RuntimeError("--hbm_path requires exactly one --ar_core_nums value")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.hbm_path:
        args.hbm_path.parent.mkdir(parents=True, exist_ok=True)
        converted_dir = args.hbm_path.parent
        hbo_dir = args.hbm_path.parent
        artifact_prefix = args.hbm_path.stem
        manifest_path = args.hbm_path.with_suffix(".compile_manifest.json")
    else:
        converted_dir = args.output_dir / "converted_bc"
        hbo_dir = args.output_dir / "hbo"
        converted_dir.mkdir(exist_ok=True)
        hbo_dir.mkdir(exist_ok=True)
        artifact_prefix = None
        manifest_path = args.output_dir / "compile_manifest.json"

    heading("SOURCE CONTRACT")
    source_bc = discover_bc(
        args.bc_dir,
        artifact_prefix=artifact_prefix,
        graph_set=args.graph_set,
    )
    if args.embedding_path:
        if not args.embedding_path.is_file():
            raise RuntimeError(f"token embedding is missing: {args.embedding_path}")
        if (
            args.expected_embedding_bytes is not None
            and args.embedding_path.stat().st_size != args.expected_embedding_bytes
        ):
            raise RuntimeError(
                f"token embedding size is {args.embedding_path.stat().st_size}; "
                f"expected {args.expected_embedding_bytes}"
            )
    if args.check_only:
        heading("SOURCE CONTRACT PASSED")
        return 0
    write_compile_manifest(manifest_path, source_bc, args)

    stage_order = list(language_graph_set(args.graph_set).graphs)
    pbd_stages = [
        graph for graph in stage_order
        if graph == "decode" or graph.startswith("decode_pbd_q")
    ]
    ar_stages = [graph for graph in stage_order if graph.startswith("decode_ar")]
    total_stages = len(stage_order)
    if not args.convert_only:
        total_stages += 1 + len(pbd_stages)
        total_stages += len(args.ar_core_nums) * (len(ar_stages) + 1)
    progress = StageProgress(total_stages, "Language build")
    if args.hbm_path:
        converted = {
            name: args.hbm_path.with_suffix(f".{name}_convert.bc")
            for name in stage_order
        }
    else:
        converted = {
            name: converted_dir / f"{name}_convert.bc" for name in stage_order
        }
    for name in stage_order:
        with progress.stage(f"Convert {graph_label(name)}"):
            convert_stage(source_bc[name], converted[name], name, args.march, args.resume)
    if args.convert_only:
        heading("CONVERT ONLY COMPLETED")
        return 0

    prefill_hbo = (
        args.hbm_path.with_suffix(".prefill.hbo")
        if args.hbm_path
        else hbo_dir / f"prefill_core{args.prefill_core_num}.hbo"
    )
    with progress.stage(f"Compile {graph_label('prefill')} HBO"):
        compile_stage(
            converted["prefill"], prefill_hbo, "prefill",
            args.prefill_core_num, args,
        )
    shared_hbos = {"prefill": prefill_hbo}
    for name in pbd_stages:
        if name not in converted:
            continue
        hbo = (
            args.hbm_path.with_suffix(f".{name}.hbo")
            if args.hbm_path
            else hbo_dir / f"{name}_core{args.decode_core_num}.hbo"
        )
        with progress.stage(f"Compile {graph_label(name)} HBO"):
            compile_stage(converted[name], hbo, name, args.decode_core_num, args)
        shared_hbos[name] = hbo
    for ar_core in args.ar_core_nums:
        ar_hbos: dict[str, Path] = {}
        for name in ar_stages:
            if name not in converted:
                continue
            ar_hbo = (
                args.hbm_path.with_suffix(f".{name}.hbo")
                if args.hbm_path
                else hbo_dir / f"{name}_core{ar_core}.hbo"
            )
            with progress.stage(f"Compile {graph_label(name)} HBO"):
                compile_stage(converted[name], ar_hbo, name, ar_core, args)
            ar_hbos[name] = ar_hbo
        graph_set_suffix = "_fused_decode" if args.graph_set == "fused_decode" else ""
        hbm = args.hbm_path or args.output_dir / (
            "LocateAnything-3B_language_chunk_1024_cache_4096_"
            "decoder_w8_lmhead_w8_nash-p_"
            f"prefill{args.prefill_core_num}_pbd{args.decode_core_num}_ar{ar_core}"
            f"{graph_set_suffix}.hbm"
        )
        all_hbos = {**shared_hbos, **ar_hbos}
        with progress.stage(f"Link Language HBM (AR cores={ar_core})"):
            link_variant(
                [all_hbos[name] for name in stage_order],
                hbm,
                args.resume,
                stage_order,
            )

    heading("ALL VARIANTS COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
