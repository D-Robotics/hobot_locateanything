from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


replay = _load(
    "locateanything_activation_replay",
    ROOT / "compiler/leap_llm/apis/calibration/locateanything_replay.py",
)
report = _load(
    "locateanything_activation_report",
    ROOT / "compiler/scripts/calibration/report.py",
)
calibrate = _load(
    "locateanything_calibrate",
    ROOT / "compiler/scripts/calibration/calibrate.py",
)


class UpdatingRange(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("absmax", torch.tensor(0.0))

    def forward(self, value):
        self.absmax.copy_(torch.maximum(self.absmax, value.abs().max()))
        return value


class FixedRange(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("absmax", torch.tensor(value))

    def forward(self, value):
        return value


def test_activation_tracker_separates_stages_and_max_hits_from_clipping():
    model = nn.Module()
    model.point = UpdatingRange()
    tracker = replay.ActivationTracker(model, component="language")

    tracker.stage = "prefill"
    model.point(torch.tensor([0.0, -1.0, 2.0, 2.0]))
    tracker.stage = "ar_q1"
    model.point(torch.tensor([-3.0, 1.0]))

    rows = tracker.activation_statistics(model)
    snapshot = tracker.snapshot(model)["point"]
    tracker.close()

    assert [(row["component"], row["graph_stage"], row["module_name"]) for row in rows] == [
        ("language", "ar_q1", "point"),
        ("language", "prefill", "point"),
    ]
    prefill = next(row for row in rows if row["graph_stage"] == "prefill")
    assert prefill["observed_elements"] == 4
    assert prefill["finite_elements"] == 4
    assert prefill["execution_count"] == 1
    assert prefill["max_hit_rate"] == 0.5
    assert prefill["clipping_rate"] == 0.0
    assert prefill["clipping_rate_exact"] is True
    assert prefill["p99_abs"] == pytest.approx(2.0)
    assert "saturation_rate" not in prefill
    assert snapshot["observed_elements"] == 6
    assert snapshot["max_hit_rate"] == pytest.approx(1 / 6)
    assert snapshot["clipping_rate"] == 0.0


def test_activation_tracker_counts_nonfinite_and_exact_fixed_range_clipping():
    model = nn.Module()
    model.point = FixedRange(1.5)
    tracker = replay.ActivationTracker(
        model, component="vision", fixed_clipping_ranges={"point": 1.5}
    )
    tracker.stage = "vision"
    model.point(torch.tensor([1.0, 2.0, 3.0, float("nan"), float("inf")]))

    row = tracker.activation_statistics(model)[0]
    snapshot = tracker.snapshot(model)["point"]
    tracker.close()

    assert row["observed_elements"] == 5
    assert row["finite_elements"] == 3
    assert row["nonfinite_count"] == 2
    assert row["min"] == 1.0
    assert row["max"] == 3.0
    assert row["mean"] == 2.0
    assert row["std"] == pytest.approx((2 / 3) ** 0.5)
    assert row["clipping_rate"] == pytest.approx(2 / 3)
    assert row["clipping_rate_exact"] is True
    assert snapshot["absmax"] == 1.5
    assert snapshot["activation_absmax"] == 3.0


def test_convergence_checkpoints_include_standard_points_and_skip_oversized():
    configured, evaluated, skipped, legacy = calibrate.resolve_convergence_checkpoints(
        "512", 300
    )

    assert configured == [64, 128, 256, 512]
    assert evaluated == [64, 128, 256]
    assert skipped == [512]
    assert legacy == 512
    assert calibrate.resolve_convergence_checkpoints("full", 23) == (
        [], [], [], 23
    )


def test_activation_statistics_audit_rejects_an_empty_snapshot():
    audit = calibrate.activation_statistics_audit({})

    assert audit["activation_point_count"] == 0
    assert audit["passed"] is False


def test_activation_statistics_audit_accepts_empty_snapshot_when_no_static_points_exist():
    audit = calibrate.activation_statistics_audit({}, required_point_count=0)

    assert audit["activation_point_count"] == 0
    assert audit["required_point_count"] == 0
    assert audit["point_count_mismatch"] is False
    assert audit["status"] == "not_applicable"
    assert audit["passed"] is True


def test_activation_statistics_audit_rejects_missing_required_static_points():
    audit = calibrate.activation_statistics_audit(
        {}, required_point_count=1
    )

    assert audit["point_count_mismatch"] is True
    assert audit["status"] == "failed"
    assert audit["passed"] is False


def test_scale_convergence_uses_scale_for_norm_activation_points():
    result = replay.compare_snapshots(
        {"norm": {"kind": "RMSNorm", "scale": 1.0, "activation_absmax": 20.0}},
        {"norm": {"kind": "RMSNorm", "scale": 2.0, "activation_absmax": 40.0}},
        first_samples=64,
        second_samples=128,
    )

    assert result["layers"][0]["metric"] == "scale"
    assert result["layers"][0]["relative_drift"] == 0.5
    assert result["from_samples"] == 64
    assert result["to_samples"] == 128


def test_activation_report_writes_json_csv_and_four_png_files(tmp_path):
    rows = [{
        "component": "vision",
        "graph_stage": "vision",
        "module_name": "point",
        "kind": "ConstFakeQuant",
        "min": -2.0,
        "max": 2.0,
        "absmax": 2.0,
        "mean": 0.0,
        "std": 1.0,
        "p99_abs": 1.8,
        "p999_abs": 2.0,
        "observed_elements": 4,
        "finite_elements": 4,
        "nonfinite_count": 0,
        "execution_count": 1,
        "max_hit_rate": 0.25,
        "clipping_range_abs": 2.0,
        "clipping_rate": 0.0,
        "clipping_rate_exact": True,
    }]
    convergence = {
        "components": {
            "vision": {
                "vs_full": [{"from_samples": 64, "p95_relative_drift": 0.03}]
            }
        }
    }
    coverage = {
        "expected_stages": ["vision"],
        "stage_execution_counts": {"vision": 64},
    }

    result = report.generate_activation_report(
        tmp_path, rows, convergence, coverage, metadata={"sample_count": 64}
    )

    payload = json.loads((tmp_path / "activation_stats.json").read_text(encoding="utf-8"))
    assert payload["activation_point_count"] == 1
    assert payload["percentile_estimator"]["diagnostic_only"] is True
    assert (tmp_path / "activation_stats.csv").is_file()
    if result["plots"]["status"] == "generated":
        assert all((tmp_path / name).is_file() for name in report.PNG_NAMES)
    else:
        assert (tmp_path / "activation_report_skipped.json").is_file()


def test_activation_report_records_explicit_skip_without_matplotlib(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    result = report.generate_activation_report(
        tmp_path,
        [],
        {"components": {}},
        {"expected_stages": [], "stage_execution_counts": {}},
    )

    assert result["plots"]["status"] == "skipped"
    skipped = json.loads(
        (tmp_path / "activation_report_skipped.json").read_text(encoding="utf-8")
    )
    assert skipped["required_pngs"] == list(report.PNG_NAMES)
    assert (tmp_path / "activation_stats.json").is_file()
    assert (tmp_path / "activation_stats.csv").is_file()
