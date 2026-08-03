import importlib.util
import csv
import io
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "compiler" / "scripts" / "validate/compare_pipeline.py"
SPEC = importlib.util.spec_from_file_location("la_pipeline_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ANALYSIS_SCRIPT = (
    Path(__file__).parents[1] / "compiler" / "scripts" / "validate/analyze_pipeline.py"
)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "la_pipeline_analysis", ANALYSIS_SCRIPT
)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def test_compare_arrays_reports_core_metrics():
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = MODULE.compare_arrays(value, value.copy())
    assert result["status"] == "compared"
    assert result["cosine"] == pytest.approx(1.0)
    assert result["relative_l2"] == pytest.approx(0.0)
    assert result["top1_agreement"] == pytest.approx(1.0)
    assert result["exact_equal"] is True
    assert result["reference_mean"] == pytest.approx(2.5)
    assert result["candidate_mean"] == pytest.approx(2.5)
    assert result["reference_std"] == pytest.approx(np.std(value.astype(np.float64)))


def test_visual_input_accepts_npy_and_npz(tmp_path):
    value = np.zeros((1, 2304, 588), dtype=np.float16)
    npy = tmp_path / "input.npy"
    npz = tmp_path / "input.npz"
    np.save(npy, value)
    np.savez(npz, vision_input=value)
    assert MODULE.load_visual_input(npy).shape == value.shape
    assert MODULE.load_visual_input(npz).shape == value.shape


def test_stage_outputs_are_atomic_and_hash_checked(tmp_path):
    output = np.arange(12, dtype=np.float32).reshape(3, 4)
    MODULE.execute_stage(tmp_path, "float", {"graph": "visual"}, lambda: ([output], {}))
    record, values = MODULE.load_stage_outputs(tmp_path, "float")
    assert record["status"] == "completed"
    np.testing.assert_array_equal(values[0], output)


def test_report_compares_completed_stages_and_marks_missing_hbm(tmp_path):
    inputs = [np.zeros((1, 2304, 588), dtype=np.float16)]
    inputs_path = tmp_path / "inputs.npz"
    MODULE.atomic_npz(inputs_path, inputs)
    MODULE.atomic_json(
        tmp_path / "run.json",
        {
            "schema_version": 2,
            "component": "vision",
            "graph": "visual",
            "source": {"kind": "test"},
            "token_ids": [],
            "inputs_file": inputs_path.name,
            "inputs_sha256": MODULE.sha256(inputs_path),
            "inputs": MODULE.describe_arrays(inputs),
        },
    )
    reference = np.array([[1.0, 2.0]], dtype=np.float32)
    MODULE.execute_stage(tmp_path, "float", {"graph": "visual"}, lambda: ([reference], {}))
    MODULE.execute_stage(
        tmp_path,
        "exported_bc",
        {"graph": "visual"},
        lambda: ([reference.copy()], {"artifact": {"kind": "bc"}}),
    )
    MODULE.execute_stage(
        tmp_path,
        "converted_bc",
        {"graph": "visual"},
        lambda: ([reference.copy()], {"artifact": {"kind": "bc"}}),
    )

    report = MODULE.build_report(tmp_path)
    assert report["status"] == "partial"
    assert report["missing_stages"] == ["hbm"]
    assert report["candidates"]["exported_bc"]["outputs"][0]["comparison"]["cosine"] == pytest.approx(1.0)
    assert report["candidates"]["converted_bc"]["outputs"][0]["comparison"]["relative_l2"] == pytest.approx(0.0)
    assert [
        (item["reference"], item["candidate"])
        for item in report["pipeline_comparisons"]
    ] == [("float", "exported_bc"), ("exported_bc", "converted_bc")]
    assert all("assessment" not in item for item in report["pipeline_comparisons"])


def test_parser_exposes_quantized_eager_with_shared_bc_implementation():
    assert MODULE.MODES == (
        "float", "quantized-eager", "exported-bc", "converted-bc", "hbm", "analysis"
    )
    parsed = MODULE.parser().parse_args(
        [
            "--mode", "exported-bc",
            "--output_dir", "output",
            "--input_dir", "inputs",
            "--model_path", "model.bc",
        ]
    )
    assert parsed.mode == "exported-bc"
    assert parsed.level == "small"
    assert parsed.phase == "vision"
    assert parsed.nums is None
    assert parsed.output_dir == Path("output")
    assert parsed.input_dir == Path("inputs")
    assert parsed.model_path == Path("model.bc")

    parsed = MODULE.parser().parse_args(
        [
            "--mode", "converted-bc",
            "--input_dir", "inputs",
            "--model_path", "model_convert.bc",
        ]
    )
    assert parsed.mode == "converted-bc"
    assert MODULE.BC_MODES["exported-bc"] != MODULE.BC_MODES["converted-bc"]

    parsed = MODULE.parser().parse_args(
        [
            "--mode", "quantized-eager",
            "--input_dir", "inputs",
            "--model_path", "checkpoint",
        ]
    )
    assert parsed.mode == "quantized-eager"

    analyzed = ANALYSIS.parser().parse_args(["--output_dir", "output"])
    assert analyzed.output_dir == Path("output")
    assert analyzed.nums is None

    parsed = MODULE.parser().parse_args(["--mode", "analysis", "--output_dir", "output"])
    assert parsed.mode == "analysis"
    assert parsed.output_dir == Path("output")


def test_scale_manifest_search_uses_current_calibration_profile(tmp_path, monkeypatch):
    calibration_root = tmp_path / "artifacts" / "calibration" / "current"
    manifest = calibration_root / "statistics" / "calibration_scale_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)

    assert MODULE.resolve_scale_manifest("language") == manifest.resolve()


def test_scale_manifest_override_is_explicit_and_phase_independent(tmp_path):
    manifest = tmp_path / "calibration_scale_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    assert MODULE.resolve_scale_manifest("vision", manifest) == manifest.resolve()
    assert MODULE.resolve_scale_manifest("language", manifest) == manifest.resolve()


def test_release_language_graph_catalog_is_ordered_and_complete():
    legacy = MODULE.release_language_graphs()
    graphs = MODULE.release_language_graphs(graph_set="fused_decode")

    assert legacy == ("prefill", "decode", "decode_ar")
    assert len(graphs) == 13
    assert graphs[:3] == ("prefill", "decode", "decode_ar")
    assert graphs[3:9] == tuple(f"decode_pbd_q{q_len}" for q_len in range(7, 13))
    assert graphs[9:] == tuple(f"decode_ar_q{q_len}" for q_len in range(2, 6))


def test_release_language_graph_catalog_rejects_missing_or_reordered_graphs():
    expected = MODULE.release_language_graphs(graph_set="fused_decode")

    with pytest.raises(ValueError, match="missing=.*decode_ar_q5"):
        MODULE.validate_language_hbm_catalog(expected[:-1], expected)

    reordered = (expected[1], expected[0], *expected[2:])
    with pytest.raises(ValueError, match="catalog order differs"):
        MODULE.validate_language_hbm_catalog(reordered, expected)


def test_language_hbm_loader_validates_release_catalog_before_exposing_test_graphs(
    tmp_path, monkeypatch
):
    import sys
    from types import ModuleType

    expected = MODULE.release_language_graphs(graph_set="fused_decode")
    container = SimpleNamespace(
        graphs=[
            SimpleNamespace(name=name, inputs=[], outputs=[])
            for name in expected
        ]
    )
    compiler = ModuleType("hbdk4.compiler")
    compiler.Hbm = lambda _path: container
    hbdk4 = ModuleType("hbdk4")
    hbdk4.compiler = compiler
    monkeypatch.setitem(sys.modules, "hbdk4", hbdk4)
    monkeypatch.setitem(sys.modules, "hbdk4.compiler", compiler)

    model = tmp_path / "language.hbm"
    model.write_bytes(b"hbm")
    loaded, artifacts = MODULE.load_language_hbm_artifacts(model, "fused_decode")

    assert loaded is container
    assert tuple(artifacts) == MODULE.LANGUAGE_BC_GRAPHS

    container.graphs.pop()
    with pytest.raises(ValueError, match="missing=.*decode_ar_q5"):
        MODULE.load_language_hbm_artifacts(model, "fused_decode")


def test_hbm_backend_detection_requires_arm64_and_hobot_runtime(tmp_path):
    assert MODULE.detect_hbm_backend("x86_64", tmp_path) == "hbdk_x86_simulator"
    assert MODULE.detect_hbm_backend("aarch64", tmp_path) == "s600_bpu"
    assert MODULE.detect_hbm_backend("aarch64", tmp_path / "missing") == "hbdk_x86_simulator"


def test_s600_hbm_runner_imports_raw_fp16_output(tmp_path, monkeypatch):
    artifact = tmp_path / "vision.hbm"
    artifact.write_bytes(b"hbm")
    runner = MODULE.REPO_ROOT / "deploy" / "run_vision_hbm.sh"
    inputs = [np.zeros((1, 2304, 588), dtype=np.float16)]

    def run(command, check):
        assert check is True
        output_path = Path(command[command.index("--output") + 1])
        np.ones(MODULE.VISION_OUTPUT_SHAPE, dtype=np.float16).tofile(output_path)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    outputs, details = MODULE.execute_s600_hbm(
        artifact, "visual", inputs, tmp_path, trace=None
    )

    assert runner.is_file()
    assert outputs[0].shape == MODULE.VISION_OUTPUT_SHAPE
    assert outputs[0].dtype == np.float16
    assert details["backend"] == "s600_bpu"
    assert details["board_output"]["bytes"] == 2359296


def test_s600_persistent_session_uses_versioned_request_ids(tmp_path, monkeypatch):
    class Buffer(io.StringIO):
        def close(self):
            pass

    class Process:
        def __init__(self):
            self.stdin = Buffer()
            self.stdout = Buffer(
                "runtime initialization\n"
                "LAHBM/1\tREADY\tvisual\n"
                "LAHBM/1\tRESULT\t1\t12.500\t2359296\n"
                "LAHBM/1\tRESULT\t2\t11.250\t2359296\n"
            )
            self.pid = 123
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    process = Process()

    def popen(command, **kwargs):
        assert command[-1] == "--server"
        assert kwargs["start_new_session"] is (os.name != "nt")
        return process

    monkeypatch.setattr(MODULE.subprocess, "Popen", popen)
    binary = tmp_path / "vision_hbm_runner"
    binary.write_bytes(b"runner")
    binary.chmod(0o755)
    model = tmp_path / "model.hbm"
    model.write_bytes(b"hbm")
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"input")
    output_path = tmp_path / "output.bin"
    session = MODULE.S600VisionSession(model)
    session.binary = binary

    session.start()
    assert session.run(input_path, output_path) == pytest.approx(12.5)
    assert session.run(input_path, output_path) == pytest.approx(11.25)
    session.close()

    requests = process.stdin.getvalue()
    assert "LAHBM/1\tRUN\t1\t" in requests
    assert "LAHBM/1\tRUN\t2\t" in requests
    assert requests.endswith("LAHBM/1\tQUIT\n")
    assert session.inferences == 2
    assert session.inference_ms == [12.5, 11.25]


def test_hbm_full_resume_preserves_runtime_history(tmp_path, monkeypatch):
    record = {"id": "0000-detection-sample", "path": "unused.npy", "sha256": None}
    output_dir = tmp_path / "output"
    model = tmp_path / "vision.hbm"
    model.write_bytes(b"hbm")
    args = SimpleNamespace(
        mode="hbm",
        level="small",
        phase="vision",
        nums=None,
        output_dir=output_dir,
        input_dir=tmp_path / "inputs",
        model_path=model,
    )
    selected, coverage = MODULE.select_records([record], None)
    metadata = MODULE.selection_metadata(args, selected, coverage)
    stage_dir = output_dir / "hbm"
    identity = MODULE.stage_identity(stage_dir, model, metadata)
    output_path = stage_dir / "outputs" / f"{record['id']}.npy"
    MODULE.atomic_npy(output_path, np.ones((1, 2), dtype=np.float16))
    MODULE.atomic_json(
        stage_dir / "samples" / f"{record['id']}.json",
        {
            "status": "completed",
            "id": record["id"],
            "phase": "vision",
            "capture_level": "final",
            "output_sha256": MODULE.sha256(output_path),
            "inference_ms": 12.5,
        },
    )
    previous_runtime = {
        "persistent_session": True,
        "model_loads": 1,
        "inferences": 1,
        "model_load_seconds": 2.0,
        "inference_total_ms": 12.5,
        "inference_mean_ms": 12.5,
        "collection_seconds": 3.0,
        "session_close_seconds": 0.2,
        "graph_execute_time_ratio": 12.5 / 3000.0,
    }
    MODULE.atomic_json(
        stage_dir / "stage.json",
        {"status": "completed", "stage": "hbm", **identity, "runtime": previous_runtime},
    )
    monkeypatch.setattr(MODULE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(MODULE, "detect_hbm_backend", lambda: "s600_bpu")
    monkeypatch.setattr(
        MODULE,
        "S600VisionSession",
        lambda *_: pytest.fail("a fully resumed HBM stage must not reload the model"),
    )

    assert MODULE.run_hbm_collection(args) == 0
    runtime = MODULE.read_json(stage_dir / "stage.json")["runtime"]
    assert runtime["model_loads"] == 1
    assert runtime["inference_total_ms"] == pytest.approx(12.5)
    assert runtime["collection_seconds"] == pytest.approx(3.0)
    assert runtime["last_invocation"]["processed"] == 0
    assert runtime["last_invocation"]["resumed"] == 1


def test_s600_session_uses_an_absolute_deadline():
    session = MODULE.S600VisionSession(Path("vision.hbm"))
    with pytest.raises(TimeoutError, match="absolute timeout"):
        session._readline_until(MODULE.time.monotonic() - 1.0, "inference")


def test_trace_recorder_persists_jsonl_and_deduplicates_full_tensors(tmp_path):
    value = np.arange(8, dtype=np.float16).reshape(2, 4)
    recorder = MODULE.TraceRecorder(tmp_path, "float", "full")
    with recorder:
        recorder.record("blocks.0", "torch_module", "Block", (value,), value)
    summary = recorder.summary
    events = [json.loads(line) for line in Path(summary["events_file"]).read_text().splitlines()]
    assert summary["events"] == 1
    assert summary["tensors"] == 2
    assert summary["unique_tensor_files"] == 1
    assert events[0]["inputs"][0]["statistics"]["max"] == pytest.approx(7.0)
    assert Path(events[0]["outputs"][0]["file"]).is_file()


def test_trace_console_is_structured_and_defers_storage_paths(tmp_path, capsys):
    value = np.arange(4, dtype=np.float16).reshape(1, 4)
    recorder = MODULE.TraceRecorder(tmp_path, "float", "full")
    with recorder:
        recorder.record("blocks.0", "torch_module", "Block", (value,), value, 1.25)

    output = capsys.readouterr().out
    event_output, saved_output = output.split("==== TRACE SAVED ====")
    assert "==== BLOCK 1 ====" in event_output
    assert "=== 1. BLOCK OUTPUT ===" in event_output
    assert "==== INPUT 1/1 ====" in event_output
    assert "==== OUTPUT 1/1 ====" in event_output
    assert "SHAPE: [1, 4]" in event_output
    assert "DTYPE: float16" in event_output
    assert "MIN: 0" in event_output
    assert "MAX: 3" in event_output
    assert "sha256" not in event_output.lower()
    assert ".npy" not in event_output
    assert "EVENTS_JSONL:" in saved_output
    assert "SUMMARY_JSON:" in saved_output
    assert "TENSOR_DIRECTORY:" in saved_output


def test_trace_semantic_groups_match_locateanything_vision_flow(tmp_path, capsys):
    value = np.ones((1, 2), dtype=np.float16)
    recorder = MODULE.TraceRecorder(tmp_path, "float", "summary")
    with recorder:
        recorder.record("input", "model_input", "ModelInput", (), value)
        recorder.record("vision_model.patch_embed.proj", "torch_module", "Linear", (value,), value)
        recorder.record("vision_model.encoder.blocks.0.norm0", "torch_module", "LayerNorm", (value,), value)
        recorder.record("vision_model.encoder.blocks.0.wqkv", "torch_module", "Linear", (value,), value)
        recorder.record("vision_model.encoder.blocks.1.norm1", "torch_module", "LayerNorm", (value,), value)
        recorder.record("vision_model.final_layernorm", "torch_module", "LayerNorm", (value,), value)
        recorder.record("vision_model.merger.mlp1.1", "torch_module", "Linear", (value,), value)
        recorder.record("visual", "float_output", "LocateAnything", (), value)
    output = capsys.readouterr().out
    assert "==== INPUT ====" in output
    assert "==== PATCH EMBEDDING ====" in output
    assert "==== BLOCK 1 ====" in output
    assert "=== 2. QKV ===" in output
    assert "==== BLOCK 2 ====" in output
    assert "==== FINAL LAYERNORM ====" in output
    assert "==== PATCH MERGER / PROJECTOR ====" in output
    assert "==== FINAL OUTPUT ====" in output


def test_trace_summary_mode_records_bc_operands_without_tensor_files(tmp_path):
    class Op:
        name = 'loc("blocks.0.attn")'
        type = "leap.matmul"

    recorder = MODULE.TraceRecorder(tmp_path, "exported_bc", "summary")
    with recorder:
        assert recorder.bc_callback(
            Op(),
            [np.ones((1, 2), dtype=np.float16)],
            [np.zeros((1, 2), dtype=np.float16)],
        )
    event = json.loads(Path(recorder.summary["events_file"]).read_text().splitlines()[0])
    assert event["canonical_name"] == "blocks.0.attn"
    assert "file" not in event["outputs"][0]


def test_canonical_name_extracts_module_from_escaped_hbdk_location():
    name = (
        'loc(fused<#hbdk.track<layerName = \\"b30.conv2d_id_18\\">>'
        '[\\"/repo/linear.py\\":145:19, \\"patch_embed.proj\\"])'
    )
    assert MODULE.canonical_name(name) == "patch_embed.proj"


def test_torch_trace_records_nested_modules(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    recorder = MODULE.TraceRecorder(tmp_path, "float", "summary")
    with recorder, MODULE.trace_torch_modules(model, recorder):
        model(torch.ones((1, 4)))
    events = [json.loads(line) for line in Path(recorder.summary["events_file"]).read_text().splitlines()]
    assert [event["name"] for event in events] == ["0", "1"]
    assert all(event["duration_ms"] is not None for event in events)


def test_torch_trace_records_leaf_modules_only(tmp_path):
    torch = pytest.importorskip("torch")

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

        def forward(self, value):
            return self.inner(value)

    model = Outer()
    recorder = MODULE.TraceRecorder(tmp_path, "float", "summary")
    with recorder, MODULE.trace_torch_modules(model, recorder):
        model(torch.ones((1, 4)))
    events = [
        json.loads(line)
        for line in Path(recorder.summary["events_file"]).read_text().splitlines()
    ]
    assert [event["name"] for event in events] == ["inner.0", "inner.1"]


def test_torch_trace_keeps_locateanything_stage_boundaries(tmp_path):
    torch = pytest.importorskip("torch")

    class Patch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.proj(value) + 1

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm0 = torch.nn.LayerNorm(4)

        def forward(self, value):
            return value + self.norm0(value)

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = Patch()
            self.blocks = torch.nn.ModuleList([Block()])

        def forward(self, value):
            return self.blocks[0](self.patch_embed(value))

    model = Vision()
    recorder = MODULE.TraceRecorder(tmp_path, "float", "summary")
    with recorder, MODULE.trace_torch_modules(model, recorder):
        model(torch.ones((1, 4)))
    events = [
        json.loads(line)
        for line in Path(recorder.summary["events_file"]).read_text().splitlines()
    ]
    assert [event["name"] for event in events] == [
        "patch_embed.proj",
        "patch_embed",
        "blocks.0.norm0",
        "blocks.0",
    ]
    assert events[1]["semantic_operation"] == "PATCH EMBEDDING OUTPUT"
    assert events[3]["semantic_operation"] == "BLOCK OUTPUT"


def test_intermediate_trace_comparison_uses_name_shape_and_occurrence(tmp_path):
    reference = MODULE.TraceRecorder(tmp_path, "float", "full")
    with reference:
        reference.record(
            "model.blocks.0.attn",
            "torch_module",
            "Attention",
            (),
            np.ones((1, 2), dtype=np.float16),
        )
    candidate = MODULE.TraceRecorder(tmp_path, "exported_bc", "full")
    with candidate:
        candidate.record(
            'loc("blocks.0.attn")',
            "bc_op",
            "leap.attention",
            (),
            np.ones((1, 2), dtype=np.float16),
        )
    result = MODULE.compare_intermediate_traces(reference.summary, candidate.summary)
    assert result["matched"] == 1
    assert result["numeric_comparisons"] == 1
    assert result["outputs"][0]["comparison"]["cosine"] == pytest.approx(1.0)


def test_report_rejects_mismatched_output_counts(tmp_path):
    inputs = [np.zeros((1, 2304, 588), dtype=np.float16)]
    inputs_path = tmp_path / "inputs.npz"
    MODULE.atomic_npz(inputs_path, inputs)
    MODULE.atomic_json(
        tmp_path / "run.json",
        {
            "schema_version": 2,
            "component": "vision",
            "graph": "visual",
            "source": {"kind": "test"},
            "token_ids": [],
            "inputs_file": inputs_path.name,
            "inputs_sha256": MODULE.sha256(inputs_path),
            "inputs": MODULE.describe_arrays(inputs),
        },
    )
    MODULE.execute_stage(
        tmp_path,
        "float",
        {"graph": "visual"},
        lambda: ([np.ones((1, 2)), np.ones((1, 2))], {}),
    )
    MODULE.execute_stage(
        tmp_path,
        "exported_bc",
        {"graph": "visual"},
        lambda: ([np.ones((1, 2))], {}),
    )
    report = MODULE.build_report(tmp_path)
    assert report["candidates"]["exported_bc"]["status"] == "incompatible"


def test_pipeline_csv_contains_flat_numeric_comparisons(tmp_path):
    path = tmp_path / "report.csv"
    MODULE.write_pipeline_csv(
        path,
        [
            {
                "reference": "float",
                "candidate": "exported_bc",
                "comparison": {
                    "status": "compared",
                    "shape": [1, 576, 2048],
                    "cosine": 0.82,
                    "relative_l2": 0.61,
                    "mae": 0.56,
                    "rmse": 0.74,
                    "max_abs": 5.1,
                    "top1_agreement": 0.31,
                    "exact_equal": False,
                    "reference_range": [-7.2, 6.6],
                    "candidate_range": [-8.2, 7.7],
                    "reference_mean": -0.003,
                    "candidate_mean": -0.002,
                    "reference_std": 1.0,
                    "candidate_std": 1.1,
                    "reference_nonzero": True,
                    "candidate_nonzero": True,
                },
            }
        ],
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reference"] == "float"
    assert rows[0]["candidate"] == "exported_bc"
    assert rows[0]["shape"] == "1x576x2048"
    assert float(rows[0]["cosine"]) == pytest.approx(0.82)


def test_intermediate_csv_preserves_model_execution_order(tmp_path):
    path = tmp_path / "report_intermediate.csv"
    MODULE.write_intermediate_csv(
        path,
        {
            "converted_bc": {
                "outputs": [
                    {
                        "reference_sequence": 3,
                        "semantic_group": "BLOCK 1",
                        "semantic_operation": "NORM0",
                        "reference_name": "blocks.0.norm0",
                        "reference_type": "LayerNorm",
                        "candidate_sequence": 27,
                        "candidate_name": "hbdk.layernorm",
                        "candidate_type": "hbdk.layernorm",
                        "candidate_choices": 2,
                        "selection_policy": "last_output_for_module_and_shape",
                        "shape": [1, 2304, 1152],
                        "status": "matched",
                        "comparison": {"cosine": 0.95, "relative_l2": 0.2},
                    }
                ]
            }
        },
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["stage"] == "converted_bc"
    assert rows[0]["semantic_group"] == "BLOCK 1"
    assert rows[0]["semantic_operation"] == "NORM0"
    assert float(rows[0]["cosine"]) == pytest.approx(0.95)


def test_batch_aggregate_csv_summarizes_all_samples_by_module(tmp_path):
    aggregator = MODULE.ModuleMetricAggregator()
    base = {
        "reference_sequence": 3,
        "semantic_group": "BLOCK 1",
        "semantic_operation": "NORM0",
        "module": "blocks.0.norm0",
        "shape": [1, 2304, 1152],
        "status": "matched",
    }
    aggregator.add(
        "float_to_exported_bc",
        [{**base, "comparison": {"cosine": 0.8, "relative_l2": 0.6, "mae": 0.5}}],
    )
    aggregator.add(
        "float_to_exported_bc",
        [{**base, "comparison": {"cosine": 1.0, "relative_l2": 0.0, "mae": 0.0}}],
    )
    path = tmp_path / "report.csv"
    MODULE.write_batch_csv(path, aggregator)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["samples"] == "2"
    assert rows[0]["matched"] == "2"
    assert float(rows[0]["cosine_mean"]) == pytest.approx(0.9)
    assert float(rows[0]["cosine_min"]) == pytest.approx(0.8)
    assert float(rows[0]["cosine_max"]) == pytest.approx(1.0)


def test_batch_capture_compares_last_matching_bc_output():
    reference = {
        "name": "blocks.0.norm0",
        "type": "LayerNorm",
        "tensor_path": "output",
        "shape": [1, 2],
        "semantic_group": "BLOCK 1",
        "semantic_operation": "NORM0",
        "array": np.ones((1, 2), dtype=np.float16),
    }

    class Op:
        name = 'loc("blocks.0.norm0")'
        type = "hbdk.layernorm"

    capture = MODULE.BCActivationCapture([reference])
    dispatcher = MODULE.BCCallbackDispatcher()
    dispatcher.active = capture
    dispatcher(Op(), [np.zeros((1, 2), dtype=np.float16)], [])
    dispatcher(Op(), [np.ones((1, 2), dtype=np.float16)], [])
    rows = MODULE.compare_activation_captures([reference], capture)
    assert rows[0]["comparison"]["exact_equal"] is True
    assert rows[0]["candidate_sequence"] == 1


def test_bc_dispatcher_skips_unrelated_operator_results_before_materialization():
    reference = {
        "name": "blocks.0.norm0",
        "shape": [1, 2],
        "array": np.ones((1, 2), dtype=np.float16),
    }

    class Op:
        name = 'loc("blocks.1.mlp")'
        type = "hbdk.linear"

    class Results:
        def __iter__(self):
            raise AssertionError("unrelated results must not be inspected")

    capture = MODULE.BCActivationCapture([reference])
    dispatcher = MODULE.BCCallbackDispatcher()
    dispatcher.active = capture

    assert dispatcher(Op(), Results(), []) is True
    assert capture.candidates == {}
    assert capture.sequence == 1


def test_canonical_name_extracts_hbdk_module_location():
    exported = (
        'loc(fused<#hbdk.track<layerName = "", layerId = 0>>'
        '[\\"/repo/leap_llm/nn/modules/linear.py\\":145:19, '
        '\\"blocks.0.wqkv\\"])'
    )
    converted = (
        'loc(fused<#hbdk.track<layerName = "b30.conv2d_id_18", layerId = 18>>'
        '[\\"/repo/leap_llm/nn/modules/linear.py\\":145:19, '
        '\\"patch_embed.proj\\"])'
    )

    assert MODULE.canonical_name(exported) == "blocks.0.wqkv"
    assert MODULE.canonical_name(converted) == "patch_embed.proj"


def test_input_directory_is_discovered_recursively(tmp_path):
    nested = tmp_path / "generated" / "tensors"
    nested.mkdir(parents=True)
    first = nested / "b.npy"
    second = nested / "a.npz"
    np.save(first, np.zeros((1, 2304, 588), dtype=np.float16))
    np.savez(second, vision_input=np.zeros((1, 2304, 588), dtype=np.float16))

    records = MODULE.discover_inputs(tmp_path)

    assert [Path(item["path"]).name for item in records] == ["a.npz", "b.npy"]
    assert [item["sha256"] for item in records] == [
        MODULE.sha256(second),
        MODULE.sha256(first),
    ]


def test_input_index_rejects_changed_file_contents(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    tensor = input_dir / "sample.npy"
    np.save(tensor, np.zeros((1, 2), dtype=np.float16))
    output_dir = tmp_path / "output"
    MODULE.prepare_input_index(output_dir, input_dir)

    np.save(tensor, np.ones((1, 2), dtype=np.float16))
    with pytest.raises(ValueError, match="different input set"):
        MODULE.prepare_input_index(output_dir, input_dir)


def test_input_index_accepts_same_content_at_a_new_absolute_path(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    value = np.zeros((1, 2), dtype=np.float16)
    np.save(first_dir / "sample.npy", value)
    np.save(second_dir / "sample.npy", value)
    output_dir = tmp_path / "output"

    MODULE.prepare_input_index(output_dir, first_dir)
    records = MODULE.prepare_input_index(output_dir, second_dir)

    assert Path(records[0]["path"]).parent == second_dir
    index = MODULE.read_json(output_dir / "inputs.json")
    assert Path(index["inputs"][0]["path"]).parent == second_dir


def test_input_index_preserves_manifest_metadata(tmp_path):
    input_dir = tmp_path / "inputs"
    tensor_dir = input_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    tensor = tensor_dir / "sample.npy"
    np.save(tensor, np.zeros((1, 2304, 588), dtype=np.float16))
    manifest = input_dir / "generated.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "bundle_id": "sample-1",
                "tensor_file": "tensors/sample.npy",
                "task": "detection",
                "source": "SKU110K",
                "image": "images/sample.jpg",
            }
        ) + "\n",
        encoding="utf-8",
    )

    records = MODULE.prepare_input_index(tmp_path / "output", input_dir)
    index = json.loads((tmp_path / "output" / "inputs.json").read_text(encoding="utf-8"))

    assert records[0]["task"] == "detection"
    assert records[0]["sha256"] == MODULE.sha256(tensor)
    assert index["inputs"][0]["source"] == "SKU110K"


def test_manifest_rejects_an_incorrect_declared_tensor_hash(tmp_path):
    tensor = tmp_path / "sample.npy"
    np.save(tensor, np.zeros((1, 2), dtype=np.float16))
    manifest = tmp_path / "generated.jsonl"
    manifest.write_text(
        json.dumps({"bundle_id": "sample", "tensor_file": tensor.name, "tensor_sha256": "0" * 64}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tensor SHA256 mismatch"):
        MODULE.discover_inputs(manifest)


def test_parser_has_default_output_and_scale_manifest_aliases():
    root = MODULE.parser()
    parsed = root.parse_args(["--mode", "float"])
    assert parsed.output_dir == MODULE.DEFAULT_OUTPUT_DIR
    option_names = {
        option
        for action in root._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert option_names == {
        "--mode", "--level", "--phase", "--nums",
        "--output_dir", "--scale-manifest", "--scale_manifest",
        "--graph-set", "--input_dir", "--model_path",
    }

    dashed = root.parse_args(["--mode", "float", "--scale-manifest", "scale.json"])
    underscored = root.parse_args(
        ["--mode", "float", "--scale_manifest", "scale.json"]
    )
    assert dashed.scale_manifest == underscored.scale_manifest == Path("scale.json")


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", "+1"])
def test_parser_rejects_invalid_nums(value):
    with pytest.raises(SystemExit):
        MODULE.parser().parse_args(["--mode", "float", "--nums", value])


def test_parser_accepts_explicit_nums():
    parsed = MODULE.parser().parse_args(
        ["--mode", "float", "--level", "high", "--nums", "42"]
    )
    assert parsed.level == "high"
    assert parsed.nums == 42


def test_unimplemented_phase_and_hbm_detail_fail_closed():
    root = MODULE.parser()
    language = root.parse_args(["--mode", "float", "--phase", "language"])
    with pytest.raises(SystemExit):
        MODULE.validate_args(root, language)

    hbm_medium = root.parse_args(["--mode", "hbm", "--level", "medium"])
    with pytest.raises(SystemExit):
        MODULE.validate_args(root, hbm_medium)


def test_float_device_is_selected_automatically(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert MODULE.detect_float_device() == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert MODULE.detect_float_device() == "cpu"


def test_stage_resume_rejects_a_different_model(tmp_path):
    stage_dir = tmp_path / "float"
    first_model = tmp_path / "first-model"
    second_model = tmp_path / "second-model"
    first_model.mkdir()
    second_model.mkdir()
    MODULE.atomic_json(stage_dir / "stage.json", {"model": str(first_model.resolve())})

    MODULE.validate_stage_identity(stage_dir, first_model)
    with pytest.raises(ValueError, match="another --output_dir"):
        MODULE.validate_stage_identity(stage_dir, second_model)


def test_stage_resume_rejects_changed_model_contents(tmp_path):
    output_dir = tmp_path / "output"
    MODULE.atomic_json(output_dir / "inputs.json", {"inputs": []})
    stage_dir = output_dir / "hbm"
    model = tmp_path / "vision.hbm"
    model.write_bytes(b"first")
    MODULE.begin_stage(stage_dir, "hbm", model)
    model.write_bytes(b"second")

    with pytest.raises(ValueError, match="model_sha256"):
        MODULE.validate_stage_identity(stage_dir, model)


def test_stage_resume_rejects_changed_checkpoint_directory_contents(tmp_path):
    output_dir = tmp_path / "output"
    MODULE.atomic_json(output_dir / "inputs.json", {"inputs": []})
    stage_dir = output_dir / "float"
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"first")
    MODULE.begin_stage(stage_dir, "float", model)
    weight.write_bytes(b"second")

    with pytest.raises(ValueError, match="model_sha256"):
        MODULE.validate_stage_identity(stage_dir, model)


def test_stage_resume_rejects_changed_level_or_selected_ids(tmp_path):
    output_dir = tmp_path / "output"
    MODULE.atomic_json(output_dir / "inputs.json", {"inputs": []})
    stage_dir = output_dir / "float"
    model = tmp_path / "model"
    model.mkdir()
    first = {
        "phase": "vision",
        "level": "small",
        "requested_nums": "all",
        "selected_ids": ["sample"],
    }
    MODULE.begin_stage(stage_dir, "float", model, first)

    changed = {**first, "level": "medium", "selected_ids": ["other"]}
    with pytest.raises(ValueError, match="identity mismatch for level"):
        MODULE.validate_stage_identity(stage_dir, model, changed)


def test_stage_resume_rejects_legacy_partial_results_without_identity(tmp_path):
    stage_dir = tmp_path / "hbm"
    MODULE.atomic_json(stage_dir / "samples" / "sample.json", {"status": "completed"})
    model = tmp_path / "vision.hbm"
    model.write_bytes(b"hbm")

    with pytest.raises(ValueError, match="partial results without model identity"):
        MODULE.validate_stage_identity(stage_dir, model)


def test_stage_resume_rejects_samples_when_stage_fingerprint_is_missing(tmp_path):
    output_dir = tmp_path / "output"
    MODULE.atomic_json(output_dir / "inputs.json", {"inputs": []})
    stage_dir = output_dir / "hbm"
    model = tmp_path / "vision.hbm"
    model.write_bytes(b"hbm")
    MODULE.atomic_json(
        stage_dir / "stage.json",
        {"model": str(model.resolve()), "status": "running"},
    )
    MODULE.atomic_json(stage_dir / "samples" / "sample.json", {"status": "completed"})

    with pytest.raises(ValueError, match="missing model_sha256"):
        MODULE.validate_stage_identity(stage_dir, model)


def test_completed_sample_requires_matching_output_hash(tmp_path):
    sample_path = tmp_path / "sample.json"
    output_path = tmp_path / "sample.npy"
    MODULE.atomic_npy(output_path, np.ones((1, 2), dtype=np.float16))
    MODULE.atomic_json(
        sample_path,
        {
            "status": "completed",
            "id": "sample",
            "output_sha256": MODULE.sha256(output_path),
        },
    )
    assert MODULE.valid_completed_sample(sample_path, output_path, "sample") is not None

    MODULE.atomic_npy(output_path, np.zeros((1, 2), dtype=np.float16))
    assert MODULE.valid_completed_sample(sample_path, output_path, "sample") is None


def test_json_aggregate_preserves_module_statistics():
    aggregator = MODULE.ModuleMetricAggregator()
    aggregator.add(
        "float_to_exported_bc",
        [
            {
                "reference_sequence": 1,
                "semantic_group": "BLOCK 1",
                "semantic_operation": "QKV",
                "module": "blocks.0.attn.qkv",
                "shape": [1, 2],
                "status": "matched",
                "comparison": {"cosine": 0.8, "mae": 0.5},
            }
        ],
    )

    record = aggregator.records()[0]
    assert record["module"] == "blocks.0.attn.qkv"
    assert record["metrics"]["cosine"]["mean"] == pytest.approx(0.8)
    assert record["metrics"]["mae"]["max"] == pytest.approx(0.5)


def test_analysis_preserves_language_decisions_and_writes_domain_csv(tmp_path):
    comparison = {
        "shape": [1, 8],
        "status": "compared",
        "cosine": 0.99,
        "top1_flip_rate": 0.0,
        "reference_top1_margin": 1.25,
        "candidate_top1_margin": 1.0,
        "decisions": [{"position": [0], "top1_flip": False}],
    }
    sample = {
        "intermediate": {
            "quantized_eager": [
                {
                    "semantic_operation": "logits",
                    "module": "prefill/logits",
                    "comparison": comparison,
                },
                {
                    "semantic_operation": "layers.0.key",
                    "module": "prefill/layers.0.key",
                    "comparison": {"cosine": 0.95},
                },
            ]
        }
    }
    assert ANALYSIS.language_decision_evidence(sample) == [
        {
            "candidate_stage": "quantized_eager",
            "module": "prefill/logits",
            "comparison": comparison,
        }
    ]

    entry = {
        "reference_sequence": 0,
        "semantic_group": "PREFILL",
        "semantic_operation": "logits",
        "module": "prefill/logits",
        "shape": [1, 8],
        "status": "matched",
        "comparison": comparison,
    }
    aggregate = MODULE.ModuleMetricAggregator()
    detection = MODULE.ModuleMetricAggregator()
    aggregate.add("quantized_eager", [entry])
    detection.add("quantized_eager", [entry])
    path = tmp_path / "report.csv"
    ANALYSIS.write_domain_csv(path, aggregate, {"detection": detection})
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["domain"] for row in rows] == ["all", "detection"]
    assert rows[1]["top1_flip_rate_mean"] == "0.0"


def test_analysis_reads_flat_stage_directories_and_writes_reports_at_root(tmp_path):
    record = {"id": "sample", "path": "sample.pt", "sha256": None}
    MODULE.atomic_json(
        tmp_path / "inputs.json",
        {"inputs": [record]},
    )
    input_fingerprint = MODULE.input_set_sha256(tmp_path / "inputs.json")
    value = np.ones((1, 2), dtype=np.float16)
    for stage in ("float", "exported_bc", "converted_bc", "hbm"):
        output_path = tmp_path / stage / "outputs" / "sample.npy"
        MODULE.atomic_npy(output_path, value)
        MODULE.atomic_json(
            tmp_path / stage / "stage.json",
            {
                "stage": stage,
                "status": "completed",
                "model": f"{stage}.model",
                "phase": "vision",
                "level": "small",
                "selected_ids": ["sample"],
                "input_set_sha256": input_fingerprint,
            },
        )
        MODULE.atomic_json(
            tmp_path / stage / "samples" / "sample.json",
            {
                "status": "completed",
                "id": "sample",
                "phase": "vision",
                "capture_level": "final",
                "output_sha256": MODULE.sha256(output_path),
                "intermediate": {},
            },
        )
    float_stage = MODULE.read_json(tmp_path / "float" / "stage.json")
    reference_sha256 = MODULE.float_reference_sha256(tmp_path, float_stage, [record])
    for stage in ("exported_bc", "converted_bc"):
        state_path = tmp_path / stage / "stage.json"
        state = MODULE.read_json(state_path)
        state["float_reference_sha256"] = reference_sha256
        MODULE.atomic_json(state_path, state)

    args = ANALYSIS.parser().parse_args(["--output_dir", str(tmp_path)])
    assert ANALYSIS.run_analysis_collection(args) == 0

    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.csv").is_file()
    assert not (tmp_path / "stages").exists()


def test_analysis_uses_only_common_stage_ids_and_ignores_stale_bc_samples(tmp_path):
    records = [
        {"id": sample_id, "path": f"{sample_id}.npy", "sha256": None}
        for sample_id in ("a", "b", "c")
    ]
    MODULE.atomic_json(tmp_path / "inputs.json", {"inputs": records})
    input_fingerprint = MODULE.input_set_sha256(tmp_path / "inputs.json")
    stage_ids = {
        "float": ["a", "b", "c"],
        "exported_bc": ["a", "b"],
        "converted_bc": ["b", "c"],
        "hbm": ["a", "b", "c"],
    }
    for stage, sample_ids in stage_ids.items():
        MODULE.atomic_json(
            tmp_path / stage / "stage.json",
            {
                "stage": stage,
                "status": "completed",
                "model": f"{stage}.model",
                "phase": "vision",
                "level": "small",
                "selected_ids": sample_ids,
                "input_set_sha256": input_fingerprint,
            },
        )
        for sample_id in sample_ids:
            output_path = tmp_path / stage / "outputs" / f"{sample_id}.npy"
            MODULE.atomic_npy(output_path, np.ones((1, 2), dtype=np.float16))
            MODULE.atomic_json(
                tmp_path / stage / "samples" / f"{sample_id}.json",
                {
                    "status": "completed",
                    "id": sample_id,
                    "phase": "vision",
                    "capture_level": "final",
                    "output_sha256": MODULE.sha256(output_path),
                    "intermediate": {},
                },
            )

    float_stage = MODULE.read_json(tmp_path / "float" / "stage.json")
    records_by_id = {record["id"]: record for record in records}
    for stage in ("exported_bc", "converted_bc"):
        state_path = tmp_path / stage / "stage.json"
        state = MODULE.read_json(state_path)
        state["float_reference_sha256"] = MODULE.float_reference_sha256(
            tmp_path,
            float_stage,
            [records_by_id[sample_id] for sample_id in state["selected_ids"]],
        )
        MODULE.atomic_json(state_path, state)
    MODULE.atomic_json(
        tmp_path / "exported_bc" / "samples" / "stale.json",
        {
            "status": "completed",
            "id": "stale",
            "intermediate": {
                "stale": [
                    {
                        "reference_sequence": 0,
                        "semantic_group": "STALE",
                        "semantic_operation": "STALE",
                        "module": "stale",
                        "shape": [1],
                        "status": "matched",
                        "comparison": {"cosine": 0.0},
                    }
                ]
            },
        },
    )

    args = ANALYSIS.parser().parse_args(["--output_dir", str(tmp_path)])
    assert ANALYSIS.run_analysis_collection(args) == 0
    report = MODULE.read_json(tmp_path / "report.json")
    assert report["selected_ids"] == ["b"]
    assert report["common_count"] == 1
    assert "stale" not in {row["stage"] for row in report["modules"]}

    too_many = ANALYSIS.parser().parse_args(
        ["--nums", "2", "--output_dir", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="1 available inputs"):
        ANALYSIS.run_analysis_collection(too_many)


def test_analysis_supports_direct_float_to_hbm_without_bc_stages(tmp_path):
    record = {"id": "sample", "path": "sample.npy", "sha256": None}
    MODULE.atomic_json(tmp_path / "inputs.json", {"inputs": [record]})
    input_fingerprint = MODULE.input_set_sha256(tmp_path / "inputs.json")
    value = np.ones((1, 2), dtype=np.float16)
    for stage in ("float", "hbm"):
        output_path = tmp_path / stage / "outputs" / "sample.npy"
        MODULE.atomic_npy(output_path, value)
        MODULE.atomic_json(
            tmp_path / stage / "stage.json",
            {
                "stage": stage,
                "status": "completed",
                "model": f"{stage}.model",
                "phase": "vision",
                "level": "small",
                "selected_ids": ["sample"],
                "input_set_sha256": input_fingerprint,
            },
        )
        MODULE.atomic_json(
            tmp_path / stage / "samples" / "sample.json",
            {
                "status": "completed",
                "id": "sample",
                "phase": "vision",
                "capture_level": "final",
                "output_sha256": MODULE.sha256(output_path),
            },
        )

    args = ANALYSIS.parser().parse_args(["--output_dir", str(tmp_path)])
    assert ANALYSIS.run_analysis_collection(args) == 0
    report = MODULE.read_json(tmp_path / "report.json")
    assert {row["stage"] for row in report["modules"]} == {"float_to_hbm"}


def test_analysis_inserts_quantized_eager_between_float_and_hbm(tmp_path):
    record = {"id": "sample", "path": "sample.npy", "sha256": None}
    MODULE.atomic_json(tmp_path / "inputs.json", {"inputs": [record]})
    input_fingerprint = MODULE.input_set_sha256(tmp_path / "inputs.json")
    values = {
        "float": np.array([[1.0, 2.0]], dtype=np.float32),
        "quantized_eager": np.array([[1.0, 1.9]], dtype=np.float32),
        "hbm": np.array([[1.0, 1.8]], dtype=np.float32),
    }
    for stage, value in values.items():
        level = "medium" if stage == "quantized_eager" else "small"
        capture_level = "boundary" if stage == "quantized_eager" else "final"
        output_path = tmp_path / stage / "outputs" / "sample.npy"
        MODULE.atomic_npy(output_path, value)
        MODULE.atomic_json(
            tmp_path / stage / "stage.json",
            {
                "stage": stage,
                "status": "completed",
                "model": f"{stage}.model",
                "phase": "vision",
                "level": level,
                "selected_ids": ["sample"],
                "input_set_sha256": input_fingerprint,
            },
        )
        MODULE.atomic_json(
            tmp_path / stage / "samples" / "sample.json",
            {
                "status": "completed",
                "id": "sample",
                "phase": "vision",
                "capture_level": capture_level,
                "output_sha256": MODULE.sha256(output_path),
                "intermediate": {},
            },
        )
    MODULE.atomic_json(
        tmp_path / "converted_bc" / "stage.json",
        {"stage": "converted_bc", "status": "running"},
    )

    float_stage = MODULE.read_json(tmp_path / "float" / "stage.json")
    quantized_stage_path = tmp_path / "quantized_eager" / "stage.json"
    quantized_stage = MODULE.read_json(quantized_stage_path)
    quantized_stage["float_reference_sha256"] = MODULE.float_reference_sha256(
        tmp_path, float_stage, [record]
    )
    MODULE.atomic_json(quantized_stage_path, quantized_stage)

    args = ANALYSIS.parser().parse_args(["--output_dir", str(tmp_path)])
    assert ANALYSIS.run_analysis_collection(args) == 0
    report = MODULE.read_json(tmp_path / "report.json")
    assert {row["stage"] for row in report["modules"]} == {
        "float_to_quantized_eager",
        "quantized_eager_to_hbm",
        "float_to_hbm",
    }
    assert report["stage_levels"] == {
        "float": "small", "quantized_eager": "medium", "hbm": "small"
    }
    assert report["skipped_stages"] == {"converted_bc": "running"}


def test_analysis_rejects_mixed_phases_before_comparing_outputs(tmp_path):
    MODULE.atomic_json(tmp_path / "inputs.json", {"inputs": []})
    for stage, phase in (("float", "vision"), ("hbm", "language")):
        MODULE.atomic_json(
            tmp_path / stage / "stage.json",
            {
                "stage": stage,
                "status": "completed",
                "phase": phase,
                "level": "small",
                "selected_ids": [],
                "input_set_sha256": "same-input-set",
            },
        )

    args = ANALYSIS.parser().parse_args(["--output_dir", str(tmp_path)])
    with pytest.raises(ValueError, match="do not share one phase"):
        ANALYSIS.run_analysis_collection(args)


def test_explicit_nums_selects_exact_stratified_count():
    counts = {
        "detection": 240,
        "gui": 180,
        "referring": 120,
        "ocr": 120,
        "layout": 100,
        "pointing": 60,
    }
    records = [
        {"id": f"{index:04d}-{task}-sample", "task": task}
        for task, count in counts.items()
        for index in range(count)
    ]
    selected, coverage = MODULE.select_records(records, 42)
    selected_counts = Counter(
        MODULE.sample_task(record) for record in selected
    )

    assert len(selected) == 42
    assert selected_counts == {
        "detection": 12,
        "gui": 9,
        "referring": 6,
        "ocr": 6,
        "layout": 5,
        "pointing": 4,
    }
    assert coverage["selected_count"] == 42
    assert coverage["selection_policy"] == "deterministic_stratified"


def test_nums_defaults_to_all_and_rejects_more_than_available():
    records = [
        {"id": "0000-detection-a", "task": "detection"},
        {"id": "0001-gui-b", "task": "gui"},
    ]

    selected, coverage = MODULE.select_records(records, None)
    assert selected == records
    assert coverage["selection_policy"] == "all"
    with pytest.raises(ValueError, match="2 available inputs"):
        MODULE.select_records(records, 3)


def test_partial_selection_is_deterministic_across_input_order():
    records = [
        {"id": f"{index:04d}-{task}-sample", "task": task}
        for task in ("detection", "gui", "referring", "ocr", "layout", "pointing")
        for index in range(10)
    ]

    forward, _ = MODULE.select_records(records, 14)
    reverse, _ = MODULE.select_records(list(reversed(records)), 14)
    assert [record["id"] for record in forward] == [record["id"] for record in reverse]
    assert len(forward) == 14


@pytest.mark.parametrize(
    ("level", "capture_level"),
    [("small", "final"), ("medium", "boundary"), ("high", "deep")],
)
def test_float_collection_routes_level_without_extra_diagnostics(
    tmp_path, monkeypatch, level, capture_level
):
    record = {"id": "0000-detection-sample", "path": "unused.npy", "sha256": None}
    model_path = tmp_path / "model"
    model_path.mkdir()
    args = SimpleNamespace(
        mode="float",
        level=level,
        phase="vision",
        nums=None,
        output_dir=tmp_path / "output",
        input_dir=tmp_path / "inputs",
        model_path=model_path,
    )
    calls: list[str] = []
    monkeypatch.setattr(MODULE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(MODULE, "detect_float_device", lambda: "cpu")
    monkeypatch.setattr(MODULE, "create_float_visual_model", lambda *_: (object(), object()))
    monkeypatch.setattr(MODULE, "load_visual_input", lambda *_: np.ones((1, 2), dtype=np.float16))

    def final(*_args):
        calls.append("final")
        return np.ones((1, 2), dtype=np.float32)

    def captured(_model, _value, _device, actual_level):
        calls.append(actual_level)
        return np.ones((1, 2), dtype=np.float32), [{"module": actual_level}]

    monkeypatch.setattr(MODULE, "run_float_final", final)
    monkeypatch.setattr(MODULE, "run_float_sample", captured)

    assert MODULE.run_float_collection(args) == 0
    assert calls == [capture_level]
    sample = MODULE.read_json(
        args.output_dir / "float" / "samples" / "0000-detection-sample.json"
    )
    assert sample["capture_level"] == capture_level
    assert len(sample["activations"]) == (0 if level == "small" else 1)


def test_quantized_eager_collection_uses_float_reference_and_writes_stage(
    tmp_path, monkeypatch
):
    record = {"id": "0000-detection-sample", "path": "unused.npy", "sha256": None}
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model"
    model_path.mkdir()
    float_output = output_dir / "float" / "outputs" / f"{record['id']}.npy"
    MODULE.atomic_npy(float_output, np.ones((1, 2), dtype=np.float32))
    MODULE.atomic_json(
        output_dir / "float" / "samples" / f"{record['id']}.json",
        {
            "status": "completed",
            "id": record["id"],
            "phase": "vision",
            "capture_level": "final",
            "output_sha256": MODULE.sha256(float_output),
        },
    )
    MODULE.atomic_json(
        output_dir / "float" / "stage.json",
        {
            "status": "completed",
            "phase": "vision",
            "level": "small",
            "model": str(model_path.resolve()),
            "model_sha256": MODULE.path_sha256(model_path),
        },
    )
    scale_manifest = tmp_path / "calibration_scale_manifest.json"
    scale_manifest.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        mode="quantized-eager",
        level="small",
        phase="vision",
        nums=None,
        output_dir=output_dir,
        input_dir=tmp_path / "inputs",
        model_path=model_path,
    )

    class FakeEmulator:
        def __init__(self, _model, capture_operators):
            assert capture_operators is False
            self.weight_rows = []

        def close(self):
            return None

    class FakeBoundaries:
        def __init__(self, _model, enabled):
            assert enabled is False

        def close(self):
            return None

    monkeypatch.setattr(MODULE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(MODULE, "resolve_scale_manifest", lambda *_: scale_manifest)
    monkeypatch.setattr(MODULE, "detect_float_device", lambda: "cpu")
    monkeypatch.setattr(MODULE, "create_float_visual_model", lambda *_: (object(), object()))
    monkeypatch.setattr(
        MODULE, "restore_calibration_scales",
        lambda *_: {"group": "vision", "sample_count": 820, "applied_modules": 108},
    )
    monkeypatch.setattr(MODULE, "QuantizationEmulator", FakeEmulator)
    monkeypatch.setattr(MODULE, "BoundaryCapture", FakeBoundaries)
    monkeypatch.setattr(
        MODULE, "load_visual_input", lambda *_: np.ones((1, 2), dtype=np.float16)
    )
    monkeypatch.setattr(
        MODULE,
        "run_quantized_eager_sample",
        lambda *_: (np.ones((1, 2), dtype=np.float32), [], []),
    )

    assert MODULE.run_quantized_eager_collection(args) == 0
    stage = MODULE.read_json(output_dir / "quantized_eager" / "stage.json")
    assert stage["status"] == "completed"
    assert stage["calibration"]["applied_modules"] == 108
    assert "reciprocal-LUT" in stage["emulation"]["limitation"]
    assert (output_dir / "quantized_eager" / "modules.csv").is_file()


def test_bc_small_uses_final_only_path_without_callback_or_float_reload(tmp_path, monkeypatch):
    record = {"id": "0000-detection-sample", "path": "unused.npy", "sha256": None}
    output_dir = tmp_path / "output"
    float_output = output_dir / "float" / "outputs" / f"{record['id']}.npy"
    MODULE.atomic_npy(float_output, np.ones((1, 2), dtype=np.float32))
    MODULE.atomic_json(
        output_dir / "float" / "samples" / f"{record['id']}.json",
        {
            "status": "completed",
            "id": record["id"],
            "phase": "vision",
            "capture_level": "final",
            "output_sha256": MODULE.sha256(float_output),
        },
    )
    float_model = tmp_path / "float-model"
    float_model.mkdir()
    MODULE.atomic_json(
        output_dir / "float" / "stage.json",
        {
            "status": "completed",
            "phase": "vision",
            "level": "small",
            "model": str(float_model),
            "model_sha256": MODULE.path_sha256(float_model),
        },
    )
    bc_model = tmp_path / "model.bc"
    bc_model.write_bytes(b"bc")
    args = SimpleNamespace(
        mode="exported-bc",
        level="small",
        phase="vision",
        nums=None,
        output_dir=output_dir,
        input_dir=tmp_path / "inputs",
        model_path=bc_model,
    )
    calls: list[str] = []
    monkeypatch.setattr(MODULE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(MODULE, "load_visual_input", lambda *_: np.ones((1, 2), dtype=np.float16))
    monkeypatch.setattr(MODULE, "load_artifact", lambda *_: (object(), [], []))

    def execute_final(*_args):
        calls.append("final")
        return np.ones((1, 2), dtype=np.float32)

    monkeypatch.setattr(MODULE, "execute_loaded_bc_final", execute_final)
    monkeypatch.setattr(
        MODULE, "create_float_visual_model",
        lambda *_: pytest.fail("small BC must not reload Float"),
    )
    monkeypatch.setattr(
        MODULE, "run_bc_sample",
        lambda *_: pytest.fail("small BC must not register intermediate capture"),
    )

    assert MODULE.run_bc_collection(args) == 0
    assert calls == ["final"]
    stage = MODULE.read_json(output_dir / "exported_bc" / "stage.json")
    assert stage["level"] == "small"
    assert stage["selected_count"] == 1


def test_bc_rejects_phase_that_differs_from_float(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    MODULE.atomic_json(
        output_dir / "float" / "stage.json",
        {"status": "completed", "phase": "language", "level": "small"},
    )
    args = SimpleNamespace(
        mode="exported-bc",
        level="high",
        phase="vision",
        nums=1,
        output_dir=output_dir,
        input_dir=tmp_path / "inputs",
        model_path=tmp_path / "model.bc",
    )
    monkeypatch.setattr(
        MODULE,
        "prepare_input_index",
        lambda *_: [{"id": "0000-detection-sample", "path": "unused", "sha256": None}],
    )

    with pytest.raises(ValueError, match="phases do not match"):
        MODULE.run_bc_collection(args)


@pytest.mark.parametrize(("level", "capture_level"), [("medium", "boundary"), ("high", "deep")])
def test_bc_detailed_levels_register_one_callback(
    tmp_path, monkeypatch, level, capture_level
):
    record = {"id": "0000-detection-sample", "path": "unused.npy", "sha256": None}
    output_dir = tmp_path / "output"
    float_output = output_dir / "float" / "outputs" / f"{record['id']}.npy"
    MODULE.atomic_npy(float_output, np.ones((1, 2), dtype=np.float32))
    MODULE.atomic_json(
        output_dir / "float" / "samples" / f"{record['id']}.json",
        {
            "status": "completed",
            "id": record["id"],
            "phase": "vision",
            "capture_level": "final",
            "output_sha256": MODULE.sha256(float_output),
        },
    )
    float_model_path = tmp_path / "float-model"
    float_model_path.mkdir()
    MODULE.atomic_json(
        output_dir / "float" / "stage.json",
        {
            "status": "completed",
            "phase": "vision",
            "level": "small",
            "model": str(float_model_path),
            "model_sha256": MODULE.path_sha256(float_model_path),
        },
    )
    bc_model = tmp_path / "model.bc"
    bc_model.write_bytes(b"bc")
    args = SimpleNamespace(
        mode="exported-bc",
        level=level,
        phase="vision",
        nums=None,
        output_dir=output_dir,
        input_dir=tmp_path / "inputs",
        model_path=bc_model,
    )

    class Artifact:
        def __init__(self):
            self.callbacks = []

        def register_callback(self, callback):
            self.callbacks.append(callback)

    artifact = Artifact()
    calls: list[str] = []
    monkeypatch.setattr(MODULE, "prepare_input_index", lambda *_: [record])
    monkeypatch.setattr(MODULE, "detect_float_device", lambda: "cpu")
    monkeypatch.setattr(MODULE, "create_float_visual_model", lambda *_: (object(), object()))
    monkeypatch.setattr(MODULE, "load_artifact", lambda *_: (artifact, [], []))
    monkeypatch.setattr(MODULE, "load_visual_input", lambda *_: np.ones((1, 2), dtype=np.float16))
    monkeypatch.setattr(
        MODULE,
        "execute_loaded_bc_final",
        lambda *_: pytest.fail("detailed BC must not use final-only execution"),
    )

    def run_sample(_model, _artifact, _dispatcher, _inputs, _outputs, _value, actual, _label):
        calls.append(actual)
        value = np.ones((1, 2), dtype=np.float32)
        return value, [MODULE._final_batch_row(value, value, 0)], {
            "bc_feed_seconds": 0.1,
            "comparison_seconds": 0.01,
        }

    monkeypatch.setattr(MODULE, "run_bc_sample", run_sample)

    assert MODULE.run_bc_collection(args) == 0
    assert calls == [capture_level]
    assert len(artifact.callbacks) == 1


def test_metric_restore_includes_boundary_and_diagnostic_rows():
    row = {
        "reference_sequence": 1,
        "semantic_group": "BLOCK 1",
        "semantic_operation": "BLOCK OUTPUT",
        "module": "blocks.0",
        "shape": [1, 2],
        "status": "matched",
        "comparison": {"cosine": 1.0},
    }
    aggregator = MODULE.ModuleMetricAggregator()
    aggregator.restore_sample(
        {
            "intermediate": {"exported_bc": [row]},
            "diagnostic": {"exported_bc_diagnostic": [row]},
        }
    )

    assert {record["stage"] for record in aggregator.records()} == {
        "exported_bc", "exported_bc_diagnostic"
    }


def test_activation_statistics_discards_tensor_data():
    array = np.array([[1.0, 2.0]], dtype=np.float16)
    records = MODULE.activation_statistics(
        [
            {
                "name": "blocks.0",
                "type": "LocateAnythingVisionBlock",
                "tensor_path": "output",
                "semantic_group": "BLOCK 1",
                "semantic_operation": "BLOCK OUTPUT",
                "array": array,
            }
        ]
    )

    assert records[0]["module"] == "blocks.0"
    assert records[0]["statistics"]["shape"] == [1, 2]
    assert records[0]["statistics"]["mean"] == pytest.approx(1.5)
    assert "array" not in records[0]


def test_boundary_capture_selects_only_model_stage_boundaries():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Sequential(torch.nn.Linear(2, 2))
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())]
            )
            self.final_layernorm = torch.nn.LayerNorm(2)
            self.merger = torch.nn.Sequential(torch.nn.Linear(2, 2))

        def forward(self, value):
            value = self.patch_embed(value)
            value = self.blocks[0](value)
            value = self.final_layernorm(value)
            return self.merger(value)

    model = Model().eval()
    with MODULE.FloatActivationCapture(model, "boundary") as capture, torch.no_grad():
        model(torch.ones((1, 2)))

    assert [entry["name"] for entry in capture.entries] == [
        "patch_embed", "blocks.0", "final_layernorm", "merger"
    ]


def test_phase_output_uses_full_width_stage_headers(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "tqdm", None)
    progress = MODULE.progress_bar(["sample"], "Float boundaries")
    for _ in progress:
        progress.set_postfix(sample="sample")
    progress.close()
    MODULE.print_phase_summary(
        "Float boundaries", 1, 1, 0, MODULE.time.monotonic(), tmp_path,
        boundaries_per_sample=30,
    )

    output = capsys.readouterr().out
    assert "================== FLOAT BOUNDARIES ==================" in output
    assert "================== FLOAT BOUNDARIES COMPLETED ==================" in output
    assert "BOUNDARIES_PER_SAMPLE: 30" in output


def test_phase_summary_accepts_elapsed_time_before_runtime_shutdown(tmp_path, capsys):
    MODULE.print_phase_summary(
        "HBM final outputs", 2, 2, 0, MODULE.time.monotonic(), tmp_path,
        elapsed_seconds=0.25,
        session_close_seconds="5.000",
    )

    output = capsys.readouterr().out
    assert "ELAPSED_SECONDS: 0.250" in output
    assert "RATE: 8.000 sample/s" in output
    assert "SESSION_CLOSE_SECONDS: 5.000" in output
