import importlib.util
from pathlib import Path
import sys
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "deploy" / "benchmark_s600_runtime.py"
SPEC = importlib.util.spec_from_file_location("benchmark_s600_runtime", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench)


PERF_LOG = """noise
[callback] END: <ref>cat</ref><box><10><20><30><40></box>
[perf] vit_infer=30ms prefill_tokens=1024 prefill_tps=6400 decode_tokens=12 decode_tps=60 ttft=210ms tpot=16.7ms e2e=500ms
[demo] xlm_infer ret=0
"""


def test_parse_runtime_performance_uses_last_complete_record():
    parsed = bench.parse_runtime_performance(PERF_LOG + PERF_LOG.replace("e2e=500", "e2e=600"))
    assert parsed["vit_infer_ms"] == 30.0
    assert parsed["prefill_tokens"] == 1024
    assert parsed["decode_tps"] == 60.0
    assert parsed["e2e_ms"] == 600.0


def test_runtime_metrics_preserve_measured_and_derived_provenance():
    metrics = bench.build_runtime_metrics(PERF_LOG, 550.0)
    assert metrics["runtime_e2e_ms"] == {
        "value": 500.0, "status": "measured",
        "source": "xlm_result_s.performance", "unit": "ms",
    }
    assert metrics["prefill_ms"]["status"] == "derived"
    assert metrics["prefill_ms"]["value"] == 160.0
    assert metrics["bps"]["status"] == "derived"
    assert metrics["bps"]["value"] == 2.0


def test_missing_runtime_fields_are_unavailable_not_zero():
    metrics = bench.build_runtime_metrics("plain output", 25.0)
    assert metrics["runtime_e2e_ms"]["status"] == "unavailable"
    assert metrics["runtime_e2e_ms"]["value"] is None
    assert metrics["bps"]["status"] == "unavailable"


def test_only_canonical_point_or_box_frames_count():
    response = "<box><1><2></box> junk <box><1><2><3><4></box> <box><1></box> <box><9999><2></box>"
    assert bench.count_structured_boxes(response) == 2
    assert bench.count_structured_boxes(None) is None


def test_summary_excludes_warmup_and_does_not_invent_semantic_rate():
    def record(kind, ok, latency):
        return {
            "run_kind": kind,
            "success": {"process": ok, "semantic": None},
            "metrics": {"wall_latency_ms": bench.measured(latency, "clock", "ms")},
        }

    summary = bench.summarize_runs(
        [record("warmup", True, 999), record("measured", True, 10), record("measured", False, 20)],
        semantic_configured=False,
    )
    assert summary["run_count"] == 2
    assert summary["process_success_rate"] == 0.5
    assert summary["semantic_success_rate"] is None
    assert summary["valid_run_count_for_metrics"] == 1
    assert summary["metrics"]["wall_latency_ms"]["mean"] == 10.0


def test_custom_metric_validation(tmp_path):
    metrics = bench.parse_custom_metrics([f"bpu0={tmp_path / 'load'}"])
    assert metrics[0][0] == "bpu0"
    try:
        bench.parse_custom_metrics(["missing-separator"])
    except ValueError as exc:
        assert "NAME=PATH" in str(exc)
    else:
        raise AssertionError("invalid metric was accepted")


def test_main_writes_complete_evidence_for_fast_command(tmp_path):
    output = tmp_path / "evidence"
    fake = (
        "print('[callback] END: <ref>cat</ref><box><1><2><3><4></box>');"
        "print('[perf] vit_infer=1ms prefill_tokens=10 prefill_tps=100 "
        "decode_tokens=2 decode_tps=20 ttft=2ms tpot=3ms e2e=200ms');"
        "print('[demo] xlm_infer ret=0')"
    )
    status = bench.main([
        "--runs", "1", "--warmup", "0", "--output-dir", str(output),
        "--semantic-regex", r"<ref>cat</ref>", "--", sys.executable, "-c", fake,
    ])
    assert status == 0
    assert (output / "summary.json").is_file()
    assert (output / "runs.jsonl").is_file()
    assert (output / "resource_samples.jsonl").is_file()
    run = __import__("json").loads((output / "runs.jsonl").read_text())
    assert run["success"] == {
        "process": True,
        "semantic": True,
        "semantic_criterion": r"<ref>cat</ref>",
    }


def test_parse_hrut_somstatus_fixture_preserves_units():
    fixture = Path(__file__).parent / "fixtures" / "hrut_somstatus_s600.txt"
    parsed = bench.parse_vendor_status(fixture.read_text(encoding="utf-8"))
    assert parsed["temperature_c"] == {"cpu": 43.2, "bpu": 44.5}
    assert parsed["voltage_v"] == {"vdd_cpu": 0.825, "vdd_bpu": 0.75}
    assert parsed["bpu_ratio"]["bpu0"] == {"value": 17.0, "unit": "vendor_ratio"}
    assert parsed["bpu_ratio"]["bpu1"] == {"value": 22.0, "unit": "%"}


def test_parse_direct_vendor_fields_and_metric_units():
    parsed = bench.parse_vendor_status(
        "temperature: 41000 mC\nvoltage soc: 900000 uV\nbpu3 ratio=12.5%\n"
    )
    assert parsed["temperature_c"] == {"soc": 41.0}
    assert parsed["voltage_v"] == {"soc": 0.9}
    assert parsed["bpu_ratio"]["bpu3"] == {"value": 12.5, "unit": "%"}


def test_vendor_command_failure_is_unavailable_not_zero(tmp_path):
    command = tmp_path / "hrut_somstatus"
    with mock.patch.object(bench.subprocess, "run") as run:
        run.return_value = mock.Mock(returncode=3, stdout="", stderr="device busy\n")
        result = bench.collect_vendor_status(command, 0.5)
    assert result["status"] == "unavailable"
    assert result["bpu_ratio"] == {}
    assert result["exit_code"] == 3
    assert "device busy" in result["raw_output"]


def test_vendor_command_success_is_structured_from_fixture(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "hrut_somstatus_s600.txt"
    command = tmp_path / "hrut_somstatus"
    completed = mock.Mock(
        returncode=0, stdout=fixture.read_text(encoding="utf-8"), stderr=""
    )
    with mock.patch.object(bench.subprocess, "run", return_value=completed):
        result = bench.collect_vendor_status(command, 0.5)
    assert result["status"] == "measured"
    assert result["temperature_c"]["cpu"] == 43.2
    assert result["voltage_v"]["vdd_cpu"] == 0.825
    assert result["bpu_ratio"]["bpu3"]["value"] == 5.0


def test_vendor_command_auto_resolution_requires_executable_file(tmp_path):
    command = tmp_path / "hrut_somstatus"
    command.write_text("fixture", encoding="utf-8")
    assert bench.resolve_vendor_status_command(str(command), disabled=True) is None
    with mock.patch.object(bench.os, "access", return_value=False):
        assert bench.resolve_vendor_status_command(str(command), disabled=False) is None
