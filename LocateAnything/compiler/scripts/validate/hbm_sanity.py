#!/usr/bin/env python3
"""Fail-closed descriptor and optional x86-simulator checks for LA HBMs.

The descriptor-only mode is the normal preflight: it loads HBM metadata but
does not allocate graph tensors or execute the simulator.  Simulation is an
explicit opt-in because the Language logits and KV tensors are large.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MASK_VALUE = -32768
LANGUAGE_GRAPH_QUERIES = {
    "prefill": 1024,
    "decode": 6,
    "decode_ar": 1,
    **{f"decode_pbd_q{q_len}": q_len for q_len in range(7, 13)},
    **{f"decode_ar_q{q_len}": q_len for q_len in range(2, 6)},
}
GRAPH_ORDER = ("visual", *LANGUAGE_GRAPH_QUERIES)


def _expected_logits_query(graph_name: str, input_query: int) -> int:
    """Return the compiled logits length for a Language graph.

    Prefill consumes the full static chunk to build KV state, but its compiled
    graph projects only the final hidden row through lm_head. Decode graphs
    retain one logits row per input position.
    """
    return 1 if graph_name == "prefill" else input_query


@dataclass(frozen=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class GraphDescriptor:
    name: str
    inputs: tuple[TensorDescriptor, ...]
    outputs: tuple[TensorDescriptor, ...]


@dataclass(frozen=True)
class ExpectedProfile:
    image_size: int = 672
    patch_size: int = 14
    spatial_merge: int = 2
    patch_channels: int = 3
    hidden_size: int = 2048
    vocab_size: int = 152681
    prefill_query: int = 1024
    cache_length: int = 4096
    pbd_query: int = 6
    ar_query: int = 1
    decoder_layers: int = 36
    cache_groups: int = 2
    head_dim: int = 128

    @property
    def patch_vector(self) -> int:
        return self.patch_channels * self.patch_size * self.patch_size

    @property
    def patch_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def visual_tokens(self) -> int:
        return self.patch_tokens // (self.spatial_merge**2)

    @property
    def cache_tensor_count(self) -> int:
        return self.decoder_layers * 2


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _shape(tensor: TensorDescriptor) -> tuple[int, ...]:
    _require(all(isinstance(x, int) and x > 0 for x in tensor.shape),
             f"{tensor.name}: shape must contain positive static integers, got {tensor.shape}")
    return tensor.shape


def _canonical_dtype(tensor: TensorDescriptor) -> str:
    value = tensor.dtype.lower().replace("numpy.", "").replace("<class '", "").replace("'>", "")
    for name in ("float16", "float32", "int8", "uint8", "int16", "int32", "int64"):
        if name in value:
            return name
    return value


def validate_descriptor_contract(
    graphs: Mapping[str, GraphDescriptor], profile: ExpectedProfile
) -> dict[str, Any]:
    """Validate the complete release graph contract using metadata only."""
    missing = [name for name in GRAPH_ORDER if name not in graphs]
    _require(not missing, f"required graph(s) missing: {', '.join(missing)}")
    unexpected = sorted(set(graphs) - set(GRAPH_ORDER))
    _require(not unexpected, f"unexpected graph(s): {', '.join(unexpected)}")

    visual = graphs["visual"]
    _require(len(visual.inputs) == 1, "visual: expected exactly one input")
    _require(len(visual.outputs) >= 1, "visual: expected at least one output")
    vin = _shape(visual.inputs[0])
    vout = _shape(visual.outputs[0])
    _require(vin == (1, profile.patch_tokens, profile.patch_vector),
             f"visual input: expected {(1, profile.patch_tokens, profile.patch_vector)}, got {vin}")
    _require(vout == (1, profile.visual_tokens, profile.hidden_size),
             f"visual output: expected {(1, profile.visual_tokens, profile.hidden_size)}, got {vout}")
    _require(_canonical_dtype(visual.inputs[0]) == "float16", "visual input must be float16")
    _require(_canonical_dtype(visual.outputs[0]) == "float16", "visual output must be float16")

    language_summary: dict[str, Any] = {}
    common_cache_dtype: str | None = None
    expected_queries = {
        **LANGUAGE_GRAPH_QUERIES,
        "prefill": profile.prefill_query,
        "decode": profile.pbd_query,
        "decode_ar": profile.ar_query,
    }
    for graph_name, expected_q in expected_queries.items():
        graph = graphs[graph_name]
        expected_logits_q = _expected_logits_query(graph_name, expected_q)
        expected_io = 3 + profile.cache_tensor_count
        _require(len(graph.inputs) == expected_io,
                 f"{graph_name}: expected {expected_io} inputs, got {len(graph.inputs)}")
        _require(len(graph.outputs) == 1 + profile.cache_tensor_count,
                 f"{graph_name}: expected {1 + profile.cache_tensor_count} outputs, got {len(graph.outputs)}")

        embed, position, mask = map(_shape, graph.inputs[:3])
        _require(embed == (1, expected_q, profile.hidden_size),
                 f"{graph_name} embeddings: expected {(1, expected_q, profile.hidden_size)}, got {embed}")
        _require(position == (1, 1, expected_q),
                 f"{graph_name} position IDs: expected {(1, 1, expected_q)}, got {position}")
        _require(mask == (1, expected_q, profile.cache_length),
                 f"{graph_name} attention mask: expected {(1, expected_q, profile.cache_length)}, got {mask}")
        _require(_canonical_dtype(graph.inputs[0]) == "float16",
                 f"{graph_name} embeddings must be float16")
        _require(_canonical_dtype(graph.inputs[1]) == "int32",
                 f"{graph_name} position IDs must be int32")
        _require(_canonical_dtype(graph.inputs[2]) == "float16",
                 f"{graph_name} attention mask must be float16")

        cache_shapes = {_shape(t) for t in graph.inputs[3:]}
        expected_cache = (1, profile.cache_length, profile.cache_groups, profile.head_dim)
        _require(cache_shapes == {expected_cache},
                 f"{graph_name} cache inputs: expected only {expected_cache}, got {sorted(cache_shapes)}")
        cache_dtypes = {_canonical_dtype(t) for t in graph.inputs[3:]}
        _require(len(cache_dtypes) == 1,
                 f"{graph_name}: cache input dtypes are inconsistent: {sorted(cache_dtypes)}")
        cache_dtype = next(iter(cache_dtypes))
        _require(cache_dtype == "int8",
                 f"{graph_name}: linked HBM cache dtype must be int8, got {cache_dtype}")
        if common_cache_dtype is None:
            common_cache_dtype = cache_dtype
        _require(cache_dtype == common_cache_dtype,
                 f"{graph_name}: cache dtype {cache_dtype} differs from {common_cache_dtype}")

        logits = _shape(graph.outputs[0])
        expected_logits = (1, expected_logits_q, profile.vocab_size)
        _require(logits == expected_logits,
                 f"{graph_name} logits: expected {expected_logits}, got {logits}")
        _require(_canonical_dtype(graph.outputs[0]) == "float16",
                 f"{graph_name} logits must be float16")
        expected_update = (1, expected_q, profile.cache_groups, profile.head_dim)
        _require(all(_shape(t) == expected_update for t in graph.outputs[1:]),
                 f"{graph_name}: one or more cache update shapes differ from {expected_update}")
        update_dtypes = {_canonical_dtype(t) for t in graph.outputs[1:]}
        _require(update_dtypes == {cache_dtype},
                 f"{graph_name}: cache update dtype must match cache input dtype {cache_dtype}")
        language_summary[graph_name] = {
            "query_length": expected_q,
            "logits_query_length": expected_logits_q,
            "input_count": len(graph.inputs),
            "output_count": len(graph.outputs),
            "cache_dtype": cache_dtype,
        }

    return {
        "profile": asdict(profile),
        "derived": {
            "patch_vector": profile.patch_vector,
            "patch_tokens": profile.patch_tokens,
            "visual_tokens": profile.visual_tokens,
            "cache_tensor_count": profile.cache_tensor_count,
        },
        "language_graphs": language_summary,
    }


def validate_embed_file(path: Path, profile: ExpectedProfile, *, scan_finite: bool) -> dict[str, Any]:
    _require(path.is_file(), f"embedding file missing: {path}")
    expected_bytes = profile.vocab_size * profile.hidden_size * 2
    actual_bytes = path.stat().st_size
    _require(actual_bytes == expected_bytes,
             f"embedding bytes: expected {expected_bytes}, got {actual_bytes}")
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": actual_bytes,
        "shape": [profile.vocab_size, profile.hidden_size],
        "dtype": "float16",
    }
    if scan_finite:
        import numpy as np

        table = np.memmap(path, dtype=np.float16, mode="r", shape=(profile.vocab_size, profile.hidden_size))
        chunk_rows = 1024
        finite = True
        for start in range(0, profile.vocab_size, chunk_rows):
            if not np.isfinite(table[start:start + chunk_rows]).all():
                finite = False
                break
        del table
        _require(finite, "embedding table contains NaN or Inf")
        result["finite_full_scan"] = True
    return result


def _dtype_name(dtype: Any) -> str:
    return str(getattr(dtype, "name", dtype))


def describe_graph(graph: Any) -> GraphDescriptor:
    def tensor_desc(tensor: Any) -> TensorDescriptor:
        return TensorDescriptor(
            name=str(tensor.name),
            shape=tuple(int(x) for x in tensor.type.shape),
            dtype=_dtype_name(tensor.type.np_dtype),
        )

    return GraphDescriptor(
        name=str(graph.name),
        inputs=tuple(tensor_desc(t) for t in graph.inputs),
        outputs=tuple(tensor_desc(t) for t in graph.outputs),
    )


def load_hbm_descriptors(vision_hbm: Path, language_hbm: Path) -> tuple[dict[str, GraphDescriptor], dict[str, Any], dict[str, Any]]:
    _require(vision_hbm.is_file(), f"Vision HBM missing: {vision_hbm}")
    _require(language_hbm.is_file(), f"Language HBM missing: {language_hbm}")
    try:
        import hbdk4.compiler as hb
    except ImportError as exc:
        raise RuntimeError("hbdk4 is required to read HBM descriptors; run this on the compiler host") from exc

    vhbm = hb.Hbm(str(vision_hbm))
    lhbm = hb.Hbm(str(language_hbm))
    vgraphs = {str(g.name): g for g in vhbm.graphs}
    lgraphs = {str(g.name): g for g in lhbm.graphs}
    unexpected_vision = sorted(set(vgraphs) - {"visual"})
    unexpected_language = sorted(set(lgraphs) - set(LANGUAGE_GRAPH_QUERIES))
    _require(
        not unexpected_vision,
        f"Vision HBM contains unexpected graph(s): {', '.join(unexpected_vision)}",
    )
    _require(
        not unexpected_language,
        f"Language HBM contains unexpected graph(s): {', '.join(unexpected_language)}",
    )
    graphs: dict[str, GraphDescriptor] = {}
    runtime_graphs: dict[str, Any] = {}
    for name in GRAPH_ORDER:
        source = vgraphs if name == "visual" else lgraphs
        if name in source:
            graphs[name] = describe_graph(source[name])
            runtime_graphs[name] = source[name]
    metadata = {
        "vision": {"march": str(vhbm.march_name), "toolkit": str(vhbm.toolkit_version)},
        "language": {"march": str(lhbm.march_name), "toolkit": str(lhbm.toolkit_version)},
    }
    return graphs, runtime_graphs, metadata


def _build_graph_inputs(graph: Any, graph_name: str, embed_path: Path, profile: ExpectedProfile) -> dict[str, Any]:
    import numpy as np

    q_len = tuple(int(x) for x in graph.inputs[0].type.shape)[1]
    feed: dict[str, Any] = {}
    rng = np.random.default_rng(42)
    for index, inp in enumerate(graph.inputs):
        shape = tuple(int(x) for x in inp.type.shape)
        dtype = inp.type.np_dtype
        if graph_name == "visual":
            value = rng.normal(0.0, 1.0, size=shape).astype(dtype)
        elif index == 0:
            table = np.memmap(embed_path, dtype=np.float16, mode="r",
                              shape=(profile.vocab_size, profile.hidden_size))
            token_ids = np.arange(q_len, dtype=np.int64) % profile.vocab_size
            value = np.asarray(table[token_ids][None, :, :], dtype=dtype)
            del table
        elif index == 1:
            if graph_name == "decode" and q_len == profile.pbd_query:
                # The upstream PBD window is the real tail token followed by
                # text-mask slots.  Start at one so the shared -1 offset never
                # creates an invalid negative RoPE position in this cold probe.
                value = np.arange(1, q_len + 1, dtype=dtype)[None, None, :]
                value[..., -q_len:] -= 1
            else:
                value = np.arange(q_len, dtype=dtype)[None, None, :]
        elif index == 2:
            value = np.full(shape, MASK_VALUE, dtype=dtype)
            if graph_name == "prefill":
                for row in range(q_len):
                    value[0, row, :row + 1] = 0
            else:
                value[...] = 0
        else:
            value = np.zeros(shape, dtype=dtype)
        _require(tuple(value.shape) == shape, f"{graph_name}/{inp.name}: generated wrong shape")
        feed[str(inp.name)] = value
    return feed


def simulate_graphs(runtime_graphs: Mapping[str, Any], embed_path: Path, profile: ExpectedProfile) -> dict[str, Any]:
    import numpy as np

    evidence: dict[str, Any] = {}
    for graph_name in GRAPH_ORDER:
        graph = runtime_graphs[graph_name]
        feed = _build_graph_inputs(graph, graph_name, embed_path, profile)
        started = time.monotonic()
        outputs = graph.feed(feed)
        elapsed = time.monotonic() - started
        output_evidence = []
        for output in graph.outputs:
            name = str(output.name)
            _require(name in outputs, f"{graph_name}: simulator omitted output {name}")
            array = outputs[name]
            declared = tuple(int(x) for x in output.type.shape)
            _require(tuple(array.shape) == declared,
                     f"{graph_name}/{name}: runtime shape {array.shape} != descriptor {declared}")
            _require(bool(np.isfinite(array).all()), f"{graph_name}/{name}: contains NaN or Inf")
            output_evidence.append({
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "min": float(array.min()),
                "max": float(array.max()),
            })
        logits = outputs[str(graph.outputs[0].name)] if graph_name != "visual" else None
        if logits is not None:
            _require(bool(np.any(logits != 0)), f"{graph_name}: logits are entirely zero")
        evidence[graph_name] = {"elapsed_seconds": elapsed, "outputs": output_evidence}
    return evidence


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-hbm", type=Path, required=True)
    parser.add_argument("--language-hbm", type=Path, required=True)
    parser.add_argument("--embed-bin", type=Path, required=True)
    parser.add_argument("--mode", choices=("descriptor-only", "simulate"), default="descriptor-only",
                        help="descriptor-only performs no graph execution (default)")
    parser.add_argument("--report-json", type=Path, help="atomically write machine-readable evidence")
    parser.add_argument("--sha256", action="store_true", help="hash all three potentially large artifacts")
    parser.add_argument("--skip-embed-finite-scan", action="store_true",
                        help="size is still checked; use only when a full finite scan is intentionally deferred")
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--spatial-merge", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=152681)
    parser.add_argument("--prefill-query", type=int, default=1024)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--pbd-query", type=int, default=6)
    parser.add_argument("--ar-query", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=36)
    parser.add_argument("--cache-groups", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    return parser


def _profile_from_args(args: argparse.Namespace) -> ExpectedProfile:
    values = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "spatial_merge": args.spatial_merge,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "prefill_query": args.prefill_query,
        "cache_length": args.cache_length,
        "pbd_query": args.pbd_query,
        "ar_query": args.ar_query,
        "decoder_layers": args.decoder_layers,
        "cache_groups": args.cache_groups,
        "head_dim": args.head_dim,
    }
    _require(all(value > 0 for value in values.values()), "all profile dimensions must be positive")
    _require(args.image_size % args.patch_size == 0, "image size must be divisible by patch size")
    grid = args.image_size // args.patch_size
    _require(grid % args.spatial_merge == 0, "patch grid must be divisible by spatial merge")
    return ExpectedProfile(**values)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = _profile_from_args(args)
        graphs, runtime_graphs, metadata = load_hbm_descriptors(args.vision_hbm, args.language_hbm)
        contract = validate_descriptor_contract(graphs, profile)
        embed = validate_embed_file(args.embed_bin, profile, scan_finite=not args.skip_embed_finite_scan)
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "pass",
            "mode": args.mode,
            "artifacts": {
                "vision_hbm": {"path": str(args.vision_hbm.resolve()), "bytes": args.vision_hbm.stat().st_size},
                "language_hbm": {"path": str(args.language_hbm.resolve()), "bytes": args.language_hbm.stat().st_size},
                "embed_bin": embed,
            },
            "hbm_metadata": metadata,
            "contract": contract,
            "graphs": {name: asdict(graph) for name, graph in graphs.items()},
        }
        if args.sha256:
            report["artifacts"]["vision_hbm"]["sha256"] = sha256_file(args.vision_hbm)
            report["artifacts"]["language_hbm"]["sha256"] = sha256_file(args.language_hbm)
            report["artifacts"]["embed_bin"]["sha256"] = sha256_file(args.embed_bin)
        if args.mode == "simulate":
            report["simulation"] = simulate_graphs(runtime_graphs, args.embed_bin, profile)
        if args.report_json:
            _write_report(args.report_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("[PASS] LocateAnything HBM sanity contract verified", file=sys.stderr)
        return 0
    except Exception as exc:  # Vendor HBDK exceptions do not share a stable base class.
        if args.report_json:
            try:
                _write_report(args.report_json, {
                    "schema_version": 1,
                    "status": "fail",
                    "mode": args.mode,
                    "error": str(exc),
                })
            except OSError as report_exc:
                print(f"[FAIL] could not write failure evidence: {report_exc}", file=sys.stderr)
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
