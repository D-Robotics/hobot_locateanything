"""Tests for the formal compiler-host environment gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_SCRIPT = ROOT / "compiler/scripts/common/environment.py"


def load_environment():
    spec = importlib.util.spec_from_file_location("environment_gate_test", ENVIRONMENT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def healthy_resources(*, memory_gib: int = 128, disk_gib: int = 256, cores: int = 32):
    gib = 1024 ** 3
    return {
        "memory": {"available": True, "available_bytes": memory_gib * gib},
        "disk": {"available": True, "free_bytes": disk_gib * gib},
        "cpu": {
            "available_cores": cores,
            "load_average_available": True,
            "load_1m": cores * 0.5,
            "load_5m": cores * 0.5,
        },
        "cuda": {
            "available": True,
            "devices": [{"index": 0, "memory_free_bytes": 24 * gib}],
        },
    }


def test_version_contract_allows_only_cuda_local_suffixes():
    module = load_environment()
    assert module.version_matches("2.8.0+cu128", "2.8.0", allow_local_suffix=True)
    assert module.version_matches("0.23.0+cu128", "0.23.0", allow_local_suffix=True)
    assert not module.version_matches("2.8.1+cu128", "2.8.0", allow_local_suffix=True)
    assert not module.version_matches("4.57.6+local", "4.57.6", allow_local_suffix=False)


def test_resource_profiles_enforce_memory_disk_jobs_and_cpu_load():
    module = load_environment()
    assert module.resource_failures(
        "build", healthy_resources(), requested_jobs=16
    ) == []

    low_memory = healthy_resources(memory_gib=95)
    assert any(
        "96 GiB" in failure
        for failure in module.resource_failures("build", low_memory, requested_jobs=16)
    )

    low_disk = healthy_resources(disk_gib=159)
    assert any(
        "160 GiB" in failure
        for failure in module.resource_failures("build", low_disk, requested_jobs=16)
    )

    low_cuda = healthy_resources()
    low_cuda["cuda"]["devices"][0]["memory_free_bytes"] = 15 * 1024 ** 3
    assert any(
        "16 GiB" in failure
        for failure in module.resource_failures("build", low_cuda, requested_jobs=16)
    )

    overloaded = healthy_resources(cores=16)
    overloaded["cpu"].update({"load_1m": 25.0, "load_5m": 21.0})
    failures = module.resource_failures("build", overloaded, requested_jobs=17)
    assert any("requested jobs" in failure for failure in failures)
    assert any("idle cores" in failure for failure in failures)
    assert any("5-minute CPU load" in failure for failure in failures)
    assert any(
        "must be positive" in failure
        for failure in module.resource_failures(
            "build", healthy_resources(), requested_jobs=0
        )
    )


def test_formal_profiles_fail_closed_when_resource_metrics_are_unavailable():
    module = load_environment()
    resources = {
        "memory": {"available": False},
        "disk": {"available": False},
        "cpu": {"available_cores": None, "load_average_available": False},
        "cuda": {"available": False},
    }
    failures = module.resource_failures("calibrate", resources, requested_jobs=None)
    assert "available host memory could not be measured" in failures
    assert "free disk space could not be measured" in failures
    assert "available CPU cores could not be measured" in failures
    assert "CUDA device memory could not be measured" in failures


def test_resource_gate_checks_the_requested_gpu_not_the_freest_gpu():
    module = load_environment()
    resources = healthy_resources()
    resources["cuda"]["devices"] = [
        {"index": 0, "memory_free_bytes": 8 * 1024 ** 3},
        {"index": 1, "memory_free_bytes": 24 * 1024 ** 3},
    ]
    failures = module.resource_failures(
        "build", resources, requested_jobs=16, requested_cuda_index=0
    )
    assert any("cuda:0" in failure and "16 GiB" in failure for failure in failures)
    assert module.resource_failures(
        "build", resources, requested_jobs=16, requested_cuda_index=1
    ) == []


def test_cuda_floor_can_be_explicitly_overridden(monkeypatch):
    module = load_environment()
    monkeypatch.setenv(module.CUDA_FLOOR_OVERRIDE_ENV, "12")
    resources = healthy_resources()
    resources["cuda"]["devices"][0]["memory_free_bytes"] = 14 * 1024 ** 3

    requirements = module.effective_resource_requirements("calibrate")
    assert requirements["minimum_free_cuda_bytes"] == 12 * 1024 ** 3
    assert requirements["minimum_free_cuda_override"] == {
        "environment": module.CUDA_FLOOR_OVERRIDE_ENV,
        "gib": 12,
    }
    assert module.resource_failures(
        "calibrate",
        resources,
        requested_jobs=None,
        requirements=requirements,
    ) == []


def test_cuda_floor_override_rejects_invalid_values(monkeypatch):
    module = load_environment()
    monkeypatch.setenv(module.CUDA_FLOOR_OVERRIDE_ENV, "0")
    try:
        module.effective_resource_requirements("prepare")
    except ValueError as exc:
        assert module.CUDA_FLOOR_OVERRIDE_ENV in str(exc)
    else:
        raise AssertionError("invalid CUDA floor override was accepted")


def test_build_profile_accepts_only_the_pinned_toolchain(monkeypatch, tmp_path, capsys):
    module = load_environment()
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    versions = {
        name: requirement["expected"]
        for name, requirement in {
            **module.RUNTIME_DISTRIBUTIONS,
            **module.BUILD_DISTRIBUTIONS,
        }.items()
    }
    monkeypatch.setattr(module.sys, "argv", [
        str(ENVIRONMENT_SCRIPT),
        "--profile", "build",
        "--model-path", str(model),
        "--selected-jsonl", str(manifest),
        "--resource-path", str(tmp_path),
        "--requested-jobs", "16",
        "--device", "cuda:0",
        "--require-cuda",
    ])
    monkeypatch.setattr(module.sys, "version_info", (3, 10, 18, "final", 0))
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.10.18")
    monkeypatch.setattr(module, "module_available", lambda _name: True)
    monkeypatch.setattr(module, "distribution_version", versions.get)
    monkeypatch.setattr(module, "memory_state", lambda: healthy_resources()["memory"])
    monkeypatch.setattr(module, "cpu_state", lambda: healthy_resources()["cpu"])
    monkeypatch.setattr(module, "disk_state", lambda _path: healthy_resources()["disk"])
    monkeypatch.setattr(
        module,
        "cuda_state",
        lambda _installed: {
            "available": True,
            "devices": [{"index": 0, "memory_free_bytes": 24 * 1024 ** 3}],
        },
    )
    monkeypatch.setattr(module, "import_probe", lambda _modules: {"passed": True})
    monkeypatch.setattr(module, "command_probe", lambda _command: {"passed": True})

    assert module.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["distributions"]["hbdk4-compiler"]["actual"] == (
        "4.10.2a2.dev202603180400+4c23b55.develop"
    )
    assert report["requested_jobs"] == 16
    assert report["requested_device"] == "cuda:0"

    versions["hbdk4-compiler"] = "4.10.1"
    assert module.main() == 1
    failed = json.loads(capsys.readouterr().out)
    assert any("hbdk4-compiler" in failure for failure in failed["failures"])


def test_formal_wrappers_call_the_matching_environment_profile():
    prepare = (ROOT / "compiler/scripts/calibration/prepare.sh").read_text(encoding="utf-8")
    calibrate = (ROOT / "compiler/scripts/calibration/calibrate.sh").read_text(encoding="utf-8")
    assert "--profile prepare" in prepare
    assert '--resource-path "$OUTPUT_DIR"' in prepare
    assert "--profile calibrate" in calibrate
    assert "calibration_environment.json" in calibrate
    assert '--device "$DEVICE"' in prepare
    assert '--device "$DEVICE"' in calibrate

    for name in ("vision.sh", "language.sh"):
        build = (ROOT / "compiler/scripts/build" / name).read_text(encoding="utf-8")
        assert "--profile build" in build
        assert '--requested-jobs "$JOBS"' in build
        assert '--device "$DEVICE"' in build
        assert "environment gate failed" in build
