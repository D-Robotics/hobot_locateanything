from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "compiler" / "scripts" / "build"


def load_stage_module(monkeypatch, filename: str, module_name: str):
    hbdk4 = ModuleType("hbdk4")
    compiler = ModuleType("hbdk4.compiler")
    hbm = ModuleType("hbdk4.compiler.hbm")
    compiler.load = lambda _path: None
    compiler.save = lambda _module, _path: None
    hbm.Hbm = type("Hbm", (), {"__init__": lambda self, _path: None})
    hbm.Hbo = type("Hbo", (), {"__init__": lambda self, _path: None})
    hbdk4.compiler = compiler

    leap_llm = ModuleType("leap_llm")
    leap_nn = ModuleType("leap_llm.nn")
    leap_utils = ModuleType("leap_llm.nn.utils")
    leap_utils.Model = type("Model", (), {})

    monkeypatch.setitem(sys.modules, "hbdk4", hbdk4)
    monkeypatch.setitem(sys.modules, "hbdk4.compiler", compiler)
    monkeypatch.setitem(sys.modules, "hbdk4.compiler.hbm", hbm)
    monkeypatch.setitem(sys.modules, "leap_llm", leap_llm)
    monkeypatch.setitem(sys.modules, "leap_llm.nn", leap_nn)
    monkeypatch.setitem(sys.modules, "leap_llm.nn.utils", leap_utils)

    spec = importlib.util.spec_from_file_location(module_name, BUILD / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def value(shape, dtype="float16"):
    return SimpleNamespace(type=SimpleNamespace(shape=shape, np_dtype=dtype))


def language_module(
    name: str,
    logits_shape: tuple[int, ...],
    cache_shape: tuple[int, ...],
    cache_dtype: str = "float32",
):
    query = cache_shape[1]
    cache_input_shape = (1, 4096, 2, 128)
    function = SimpleNamespace(
        name=name,
        inputs=[
            value((1, query, 2048)),
            value((1, 1, query), "int32"),
            value((1, query, 4096)),
            *[value(cache_input_shape, cache_dtype) for _ in range(72)],
        ],
        outputs=[
            value(logits_shape),
            *[value(cache_shape, cache_dtype) for _ in range(72)],
        ],
    )
    return SimpleNamespace(functions=[function])


def test_language_bc_check_requires_all_13_release_graphs(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_variants_test"
    )
    prefix = "release_language_fusedpbd"
    contracts = dict(module.BASE_EXPECTED)
    contracts.update(
        {name: module.expected_contract(name) for name in module.FUSED_STAGES}
    )
    by_path = {}
    for name, (logits, cache) in contracts.items():
        path = tmp_path / f"{prefix}.{name}.bc"
        path.write_bytes(name.encode("ascii"))
        by_path[str(path)] = language_module(name, logits, cache)
    module.load = lambda path: by_path[path]

    discovered = module.discover_bc(
        tmp_path, artifact_prefix=prefix, require_fused=True
    )
    assert set(discovered) == set(module.KNOWN_STAGES)
    assert len(discovered) == 13

    missing = tmp_path / f"{prefix}.decode_pbd_q12.bc"
    missing.unlink()
    with pytest.raises(RuntimeError, match="graph family is incomplete"):
        module.discover_bc(tmp_path, artifact_prefix=prefix, require_fused=True)


def test_language_graph_contract_checks_last_kv_and_every_dtype(monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_full_contract_test"
    )
    function = language_module(
        "decode_pbd_q12", (1, 12, 152681), (1, 12, 2, 128)
    ).functions[0]
    module.validate_graph_contract(function, "decode_pbd_q12")

    function.inputs[-1] = value((1, 4095, 2, 128), "float32")
    with pytest.raises(RuntimeError, match=r"input\[74\] mismatch"):
        module.validate_graph_contract(function, "decode_pbd_q12")

    function = language_module(
        "decode_pbd_q12", (1, 12, 152681), (1, 12, 2, 128)
    ).functions[0]
    function.outputs[-1] = value((1, 12, 2, 128), "float16")
    with pytest.raises(RuntimeError, match=r"output\[72\] mismatch"):
        module.validate_graph_contract(function, "decode_pbd_q12")

    function = language_module(
        "decode_pbd_q12", (1, 12, 152681), (1, 12, 2, 128)
    ).functions[0]
    function.inputs[1] = value((1, 1, 12), "int64")
    with pytest.raises(RuntimeError, match=r"input\[1\] mismatch"):
        module.validate_graph_contract(function, "decode_pbd_q12")

    function = language_module(
        "decode_pbd_q12", (1, 12, 152681), (1, 12, 2, 128), "int8"
    ).functions[0]
    with pytest.raises(RuntimeError, match=r"input\[3\] mismatch"):
        module.validate_graph_contract(function, "decode_pbd_q12")


def test_language_resume_invalidates_changed_bc_identity(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_manifest_test"
    )
    source = tmp_path / "release.prefill.bc"
    source.write_bytes(b"first")
    manifest = tmp_path / "release.compile_manifest.json"
    args = argparse.Namespace(
        march="nash-p",
        prefill_core_num=4,
        decode_core_num=4,
        ar_core_nums=[4],
        jobs=16,
        require_fused=True,
        hbm_path=tmp_path / "release.hbm",
        resume=False,
    )
    assert module.validate_or_create_manifest(manifest, {"prefill": source}, args) is False
    assert module.validate_or_create_manifest(manifest, {"prefill": source}, args) is True

    source.write_bytes(b"changed")
    args.resume = True
    assert module.validate_or_create_manifest(manifest, {"prefill": source}, args) is False


def test_language_resume_skips_completed_converted_hbo_and_hbm(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_stage_resume_test"
    )
    converted = tmp_path / "release.prefill_convert.bc"
    hbo = tmp_path / "release.prefill.hbo"
    hbm = tmp_path / "release.hbm"
    for path in (converted, hbo, hbm):
        path.write_bytes(b"complete")
        module.write_digest(path)
    module.load = lambda _path: language_module(
        "prefill", (1, 1, 152681), (1, 1024, 2, 128), "float32"
    )
    module.Hbo = lambda _path: object()
    hbm_graph = language_module(
        "prefill", (1, 1, 152681), (1, 1024, 2, 128), "int8"
    ).functions[0]
    module.Hbm = lambda _path: SimpleNamespace(graphs=[hbm_graph])

    class FailModel:
        @staticmethod
        def convert_mlir(*_args, **_kwargs):
            raise AssertionError("completed Converted BC was rebuilt")

        @staticmethod
        def compile_hbo(*_args, **_kwargs):
            raise AssertionError("completed HBO was rebuilt")

        @staticmethod
        def link_models(*_args, **_kwargs):
            raise AssertionError("completed HBM was relinked")

    module.Model = FailModel
    module.convert_stage(
        tmp_path / "source.prefill.bc", converted, "prefill", "nash-p", True
    )
    module.compile_stage(
        converted,
        hbo,
        "prefill",
        4,
        SimpleNamespace(resume=True, march="nash-p", jobs=16),
    )
    module.link_variant([hbo], hbm, True, ["prefill"])


def test_vision_bc_check_enforces_release_shapes(tmp_path, monkeypatch):
    module = load_stage_module(monkeypatch, "vision_stages.py", "vision_stages_test")
    bc = tmp_path / "release.visual.bc"
    bc.write_bytes(b"bc")
    function = SimpleNamespace(
        name="visual",
        inputs=[value((1, 2304, 588))],
        outputs=[value((1, 576, 2048))],
    )
    module.load = lambda _path: SimpleNamespace(functions=[function])
    module.validate_visual_bc(bc)

    function.outputs[0] = value((1, 575, 2048))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        module.validate_visual_bc(bc)


@pytest.mark.parametrize("side", ("input", "output"))
def test_vision_bc_check_enforces_fp16_io(tmp_path, monkeypatch, side):
    module = load_stage_module(
        monkeypatch, "vision_stages.py", f"vision_dtype_{side}_test"
    )
    bc = tmp_path / "release.visual.bc"
    bc.write_bytes(b"bc")
    function = SimpleNamespace(
        name="visual",
        inputs=[value((1, 2304, 588))],
        outputs=[value((1, 576, 2048))],
    )
    getattr(function, f"{side}s")[0] = value(
        (1, 2304, 588) if side == "input" else (1, 576, 2048),
        "float32",
    )
    module.load = lambda _path: SimpleNamespace(functions=[function])
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        module.validate_visual_bc(bc)


def test_vision_hbm_completion_rejects_non_fp16_io(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "vision_stages.py", "vision_hbm_dtype_test"
    )
    hbm = tmp_path / "release.hbm"
    hbm.write_bytes(b"hbm")
    graph = SimpleNamespace(
        name="visual",
        inputs=[value((1, 2304, 588), "float16")],
        outputs=[value((1, 576, 2048), "float32")],
    )
    module.Hbm = lambda _path: SimpleNamespace(graphs=[graph])
    assert module.hbm_contract_matches(hbm) is False


def test_vision_resume_invalidates_changed_bc_identity(tmp_path, monkeypatch):
    module = load_stage_module(monkeypatch, "vision_stages.py", "vision_manifest_test")
    source = tmp_path / "release.visual.bc"
    source.write_bytes(b"first")
    manifest = tmp_path / "release.compile_manifest.json"
    args = argparse.Namespace(
        march="nash-p",
        core_num=4,
        jobs=16,
        hbm_path=tmp_path / "release.hbm",
        resume=False,
    )

    assert module.validate_manifest(manifest, source, args) is False
    assert module.validate_manifest(manifest, source, args) is True
    source.write_bytes(b"changed")
    args.resume = True
    assert module.validate_manifest(manifest, source, args) is False


def test_resume_requires_matching_stage_digest(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_digest_test"
    )
    converted = tmp_path / "release.prefill_convert.bc"
    converted.write_bytes(b"converted")
    module.load = lambda _path: language_module(
        "prefill", (1, 1, 152681), (1, 1024, 2, 128), "float32"
    )

    assert module.valid_function(converted, "prefill") is False
    module.write_digest(converted)
    assert module.valid_function(converted, "prefill") is True
    converted.write_bytes(b"tampered")
    assert module.valid_function(converted, "prefill") is False


def test_changed_contract_invalidates_only_matching_stage_digests(tmp_path, monkeypatch):
    module = load_stage_module(
        monkeypatch, "language_variants.py", "language_invalidation_test"
    )
    matching = tmp_path / "release.prefill.hbo.sha256"
    matching.write_text("old\n", encoding="ascii")
    unrelated = tmp_path / "other.prefill.hbo.sha256"
    unrelated.write_text("keep\n", encoding="ascii")

    assert module.invalidate_stage_digests(tmp_path, "release") == 1
    assert not matching.exists()
    assert unrelated.is_file()


def test_shell_wrappers_reuse_bc_and_forward_resume_to_stage_builders():
    vision = (BUILD / "vision.sh").read_text(encoding="utf-8")
    language = (BUILD / "language.sh").read_text(encoding="utf-8")

    for source, stage_script in (
        (vision, "vision_stages.py"),
        (language, "language_variants.py"),
    ):
        assert 'RESUME="${RESUME:-0}"' in source
        assert "if check_bc; then" in source
        assert "--export_only" in source
        assert "RESUME_ARGS+=(--resume)" in source
        assert stage_script in source
        assert "artifact_manifest.py" in source
        assert 'bash "$0"' in source

    assert "--require_fused" in language
    assert "EXPECTED_EMBEDDING_BYTES=$((152681 * 2048 * 2))" in language


def test_bc_provenance_rejects_changed_calibration(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    weight = model / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"weights")
    scale = tmp_path / "scales.json"
    scale.write_text('{"scale": 1}\n', encoding="utf-8")
    bc = tmp_path / "visual.bc"
    bc.write_bytes(b"bc")
    source_root = tmp_path / "compiler"
    source_root.mkdir()
    source_file = source_root / "adapter.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = tmp_path / "visual.bc_manifest.json"
    script = BUILD / "artifact_manifest.py"
    common = [
        "--manifest", str(manifest),
        "--component", "vision",
        "--model_path", str(model),
        "--source_root", str(source_root),
        "--scale_manifest", str(scale),
        "--field", "w_bits=8",
        "--artifact", f"visual={bc}",
    ]

    written = subprocess.run(
        [sys.executable, str(script), "write", *common],
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["model"]["weights"][0]["sha256"] == hashlib.sha256(
        b"weights"
    ).hexdigest()
    checked = subprocess.run(
        [sys.executable, str(script), "check", *common],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    scale.write_text('{"scale": 2}\n', encoding="utf-8")
    stale = subprocess.run(
        [sys.executable, str(script), "check", *common],
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 1
    assert "calibration_scale_manifest" in stale.stdout

    scale.write_text('{"scale": 1}\n', encoding="utf-8")
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    stale_source = subprocess.run(
        [sys.executable, str(script), "check", *common],
        capture_output=True,
        text=True,
    )
    assert stale_source.returncode == 1
    assert "compiler_source" in stale_source.stdout

    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    weight_stat = weight.stat()
    weight.write_bytes(b"changed")
    os.utime(weight, ns=(weight_stat.st_atime_ns, weight_stat.st_mtime_ns))
    stale_weight = subprocess.run(
        [sys.executable, str(script), "check", *common],
        capture_output=True,
        text=True,
    )
    assert stale_weight.returncode == 1
    assert "model" in stale_weight.stdout

    weight.write_bytes(b"weights")
    bc.write_bytes(b"changed bc")
    stale_bc = subprocess.run(
        [sys.executable, str(script), "check", *common],
        capture_output=True,
        text=True,
    )
    assert stale_bc.returncode == 1
    assert "artifacts" in stale_bc.stdout
