from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "compiler" / "quantize.py"
CONFIG = ROOT / "compiler" / "config.yaml"


def load_cli():
    spec = importlib.util.spec_from_file_location("quantize_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_config_fixes_release_contract():
    module = load_cli()
    config = module.load_config(CONFIG)
    assert config["language"]["chunk_size"] == 1024
    assert config["language"]["cache_len"] == 4096
    assert config["language"]["decoder_w_bits"] == 8
    assert config["language"]["lm_head_w_bits"] == 8
    assert config["language"]["fused_pbd"] is True
    assert len(config["language"]["graphs"]) == 13
    assert config["vision"]["w_bits"] == 8
    assert config["model"]["patch_size"] == 14
    assert config["model"]["spatial_merge"] == 2
    assert config["model"]["checkpoint_sha256"] == module.EXPECTED_CHECKPOINT_SHA256
    assert config["model"]["checkpoint_index_sha256"] == (
        module.EXPECTED_CHECKPOINT_INDEX_SHA256
    )
    assert config["paths"]["upstream_source"] == "workspace/upstream/Embodied"
    assert config["paths"]["build_root"] == "workspace/builds/release_candidate"
    assert config["calibration"]["sample_count"] == 1200
    assert config["calibration"]["checkpoint_samples"] == 512
    assert config["calibration"]["max_new_tokens"] == 1024
    assert config["calibration"]["image_token_id"] == 151665
    assert config["calibration"]["prepare_dtype"] == "bfloat16"
    assert config["calibration"]["calibrate_dtype"] == "float16"
    assert config["calibration"]["selected_manifest_sha256"] == (
        "22cc670b2b600b2e5ea3dfbc3d169c07540ef108a0e2a135d8b20f949ed62b03"
    )
    assert config["calibration"]["task_counts"] == {
        "detection": 620,
        "gui": 180,
        "referring": 120,
        "ocr": 120,
        "layout": 100,
        "pointing": 60,
    }
    assert config["calibration"]["source_role_counts"] == {
        "coco_multicategory_detection": 500,
        "dense_retail_detection": 120,
        "existing_non_detection": 580,
    }
    assert config["calibration"]["coco_stratum_counts"] == {
        "single": 200,
        "double": 220,
        "multi": 80,
    }


def test_rotation_validation_uses_the_release_tensor_contract():
    source = (ROOT / "compiler/scripts/validate/rotation.py").read_text(encoding="utf-8")
    assert "config.cache_len = 4096" in source
    assert "config.w_bits = 8" in source
    assert "torch.randn(1, 2304, 588" in source
    assert "config.cache_len = 2048" not in source


def test_graph_contract_rejects_duplicates_and_reordering():
    module = load_cli()
    config = module.load_config(CONFIG)
    config["language"]["graphs"][-1] = config["language"]["graphs"][-2]
    try:
        module.validate_config(config)
    except module.ConfigurationError as error:
        assert "canonical 13 graphs" in str(error)
    else:
        raise AssertionError("duplicate graph name was accepted")


def test_release_config_rejects_a_legacy_calibration_count():
    module = load_cli()
    config = module.load_config(CONFIG)
    config["calibration"]["sample_count"] = 820
    try:
        module.validate_config(config)
    except module.ConfigurationError as error:
        assert "sample_count=1200" in str(error)
    else:
        raise AssertionError("legacy calibration count was accepted")


def test_release_config_rejects_the_legacy_generation_limit():
    module = load_cli()
    config = module.load_config(CONFIG)
    config["calibration"]["max_new_tokens"] = 512
    try:
        module.validate_config(config)
    except module.ConfigurationError as error:
        assert "max_new_tokens=1024" in str(error)
    else:
        raise AssertionError("legacy calibration generation limit was accepted")


def test_release_config_rejects_processor_or_dtype_drift():
    module = load_cli()
    mutations = (
        ("model", "patch_size", 16, "patch_size=14"),
        ("model", "spatial_merge", 1, "spatial_merge=2"),
        ("calibration", "image_token_id", 151666, "image_token_id=151665"),
        ("calibration", "prepare_dtype", "float16", "bfloat16"),
        ("calibration", "calibrate_dtype", "bfloat16", "float16"),
    )
    for section, key, value, expected in mutations:
        config = module.load_config(CONFIG)
        config[section][key] = value
        try:
            module.validate_config(config)
        except module.ConfigurationError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"{section}.{key} drift was accepted")


def test_all_public_commands_have_help():
    assert run_cli("--help").returncode == 0
    for command in ("prepare", "calibrate", "build", "verify"):
        result = run_cli(command, "--help")
        assert result.returncode == 0, result.stderr
        assert "--progress" in result.stdout
        assert "--resume" in result.stdout
        assert "--dry-run" in result.stdout


def test_build_dry_run_forwards_fixed_contract():
    result = run_cli(
        "build", "--component", "all", "--target", "hbm", "--dry-run", "--progress", "log"
    )
    assert result.returncode == 0, result.stderr
    assert "CACHE_LEN=4096" in result.stdout
    assert "W_BITS=8" in result.stdout
    assert "LM_HEAD_W_BITS=8" in result.stdout
    assert "IMAGE_WIDTH=672" in result.stdout
    assert "IMAGE_HEIGHT=672" in result.stdout
    assert "FUSED_PBD_PROFILES=1" in result.stdout
    assert (
        "EXPECTED_SELECTED_MANIFEST_SHA256="
        "22cc670b2b600b2e5ea3dfbc3d169c07540ef108a0e2a135d8b20f949ed62b03"
    ) in result.stdout
    assert "WAIT=1" in result.stdout
    assert "DETACH=0" in result.stdout
    assert "language_graphs\": 13" in result.stdout
    assert "[dry-run] no command executed" in result.stdout


def test_build_all_plan_waits_for_vision_before_language():
    module = load_cli()
    config = module.load_config(CONFIG)
    args = module.build_parser().parse_args(
        ["build", "--component", "all", "--target", "hbm", "--dry-run"]
    )
    steps = module.build_plan(args, config)
    assert [step.label for step in steps] == [
        "build vision through hbm",
        "build language through hbm",
    ]
    assert all(step.env["WAIT"] == "1" for step in steps)
    assert all(step.env["DETACH"] == "0" for step in steps)
    assert all(
        step.env["EXPECTED_SELECTED_MANIFEST_SHA256"]
        == config["calibration"]["selected_manifest_sha256"]
        for step in steps
    )


def test_workspace_and_scoped_roots_rebase_compiler_paths(tmp_path, monkeypatch):
    module = load_cli()
    config = module.load_config(CONFIG)
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("LA_WORKSPACE", str(workspace))
    assert module.resolve_path(config, "generated_jsonl") == (
        workspace / "calibration/current/generated/generated.jsonl"
    ).resolve()

    calibration = tmp_path / "calibration-data"
    builds = tmp_path / "compiler-builds"
    evaluation = tmp_path / "evaluation-data"
    monkeypatch.setenv("LA_CALIBRATION_ROOT", str(calibration))
    monkeypatch.setenv("LA_BUILD_ROOT", str(builds))
    monkeypatch.setenv("LA_EVALUATION_ROOT", str(evaluation))
    assert module.resolve_path(config, "selected_jsonl") == (
        calibration / "current/selected.jsonl"
    ).resolve()
    assert module.resolve_path(config, "build_root") == (
        builds / "release_candidate"
    ).resolve()
    assert module.resolve_path(config, "verification_root") == (
        evaluation / "release_candidate"
    ).resolve()


def test_direct_cli_and_model_path_overrides_take_precedence(tmp_path, monkeypatch):
    module = load_cli()
    config = module.load_config(CONFIG)
    monkeypatch.setenv("LA_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("LA_MODEL_ROOT", str(tmp_path / "models"))
    direct_model = tmp_path / "exact-model"
    monkeypatch.setenv("LA_MODEL_PATH", str(direct_model))
    assert module.resolve_path(config, "model") == direct_model.resolve()

    direct_output = tmp_path / "exact-build"
    assert module.resolve_path(
        config, "build_root", str(direct_output)
    ) == direct_output.resolve()


def test_verify_plan_honors_evaluation_root(tmp_path, monkeypatch):
    module = load_cli()
    config = module.load_config(CONFIG)
    evaluation = tmp_path / "evaluation"
    monkeypatch.setenv("LA_EVALUATION_ROOT", str(evaluation))
    args = module.build_parser().parse_args(["verify", "--level", "task", "--dry-run"])
    command = module.verify_plan(args, config)[0].command
    assert str(evaluation / "release_candidate/predictions.jsonl") in command
    assert str(evaluation / "current/selected.jsonl") in command


def test_build_all_executes_serially_and_stops_after_failure(monkeypatch):
    module = load_cli()
    config = module.load_config(CONFIG)
    args = module.build_parser().parse_args(
        ["build", "--component", "all", "--target", "hbm"]
    )
    steps = module.build_plan(args, config)
    calls = []

    def fail_vision(command, **kwargs):
        calls.append(Path(command[1]).name)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fail_vision)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/bash")
    assert module.run_plan(steps, args, config) == 7
    assert calls == ["vision.sh"]


def test_each_build_target_is_accepted_in_dry_run():
    for target in ("bc", "hbm"):
        result = run_cli(
            "build", "--component", "language", "--target", target, "--dry-run"
        )
        assert result.returncode == 0, result.stderr
        assert f"BUILD_TARGET={target}" in result.stdout


def test_dry_run_never_invokes_an_external_command(monkeypatch):
    module = load_cli()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("dry-run invoked an external command")

    monkeypatch.setattr(module.subprocess, "run", unexpected_call)
    commands = (
        ["prepare", "--dry-run"],
        ["calibrate", "--dry-run"],
        ["build", "--component", "all", "--target", "hbm", "--dry-run"],
        ["verify", "--component", "all", "--level", "all", "--dry-run"],
    )
    for command in commands:
        assert module.main(command) == 0


def test_build_resume_is_forwarded_to_each_component():
    module = load_cli()
    config = module.load_config(CONFIG)
    args = module.build_parser().parse_args(
        ["build", "--component", "all", "--target", "hbm", "--resume"]
    )
    steps = module.build_plan(args, config)
    assert [step.env["RESUME"] for step in steps] == ["1", "1"]


def test_hbo_is_not_exposed_as_a_false_stop_target():
    result = run_cli("build", "--component", "language", "--target", "hbo", "--dry-run")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_calibrate_dry_run_uses_full_graph_replay_and_cache_4096():
    result = run_cli("calibrate", "--component", "all", "--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout.replace("\\", "/")
    assert "compiler/scripts/calibration/calibrate.sh" in output
    assert "CALIBRATION_COMPONENT=all" in output
    assert "SELECTED_JSONL=" in output
    assert "UPSTREAM_REPO=" in output
    assert "CACHE_LEN=4096" in output
    assert "MAX_SAMPLES=1200" in output
    assert "CHECKPOINT_SAMPLES=512" in output


def test_calibrate_cli_rejects_nonrelease_overrides():
    result = run_cli("calibrate", "--max-samples", "1199", "--dry-run")
    assert result.returncode != 0
    assert "--max-samples=1200" in result.stderr


def test_cli_rejects_nonrelease_dtype_overrides():
    prepare = run_cli("prepare", "--dtype", "float16", "--dry-run")
    calibrate = run_cli("calibrate", "--dtype", "bfloat16", "--dry-run")
    assert prepare.returncode != 0
    assert calibrate.returncode != 0
    assert "invalid choice" in prepare.stderr
    assert "invalid choice" in calibrate.stderr


def test_external_hidden_rotation_is_forwarded_to_both_builds(tmp_path):
    module = load_cli()
    config = module.load_config(CONFIG)
    rotation = tmp_path / "rotation.pt"
    rotation.write_bytes(b"rotation")
    config["calibration"]["hidden_rotation_path"] = str(rotation)
    args = module.build_parser().parse_args(
        ["build", "--component", "all", "--target", "bc", "--dry-run"]
    )
    steps = module.build_plan(args, config)
    assert [step.env["HIDDEN_ROTATION_PATH"] for step in steps] == [
        str(rotation.resolve()), str(rotation.resolve())
    ]


def test_cli_uses_standardized_script_layout():
    prepare = run_cli("prepare", "--dry-run")
    build = run_cli("build", "--component", "all", "--dry-run")
    verify = run_cli("verify", "--level", "all", "--dry-run")
    prepare_output = prepare.stdout.replace("\\", "/")
    build_output = build.stdout.replace("\\", "/")
    verify_output = verify.stdout.replace("\\", "/")
    assert "compiler/scripts/calibration/preflight.py" in prepare_output
    assert "compiler/scripts/calibration/prepare.sh" in prepare_output
    assert "compiler/scripts/build/vision.sh" in build_output
    assert "compiler/scripts/build/language.sh" in build_output
    assert "compiler/scripts/validate/deployment.py" in verify_output
    assert "compiler/scripts/validate/compare_pipeline.py" in verify_output
    assert "--scale_manifest" in verify.stdout
    assert "compiler/scripts/validate/evaluate_grounding.py" in verify_output


def test_prepare_preflight_only_bypasses_gpu_generation_wrapper():
    result = run_cli("prepare", "--preflight-only", "--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout.replace("\\", "/")
    assert "compiler/scripts/calibration/preflight.py" in output
    assert "compiler/scripts/calibration/prepare.sh" not in output
    assert "DEVICE=" not in output
    assert "DTYPE=" not in output
    assert "MAX_NEW_TOKENS=" not in output
    assert "--report-json" in output
    assert "no CUDA or model inference" in output


def test_prepare_always_runs_static_preflight_before_model_generation():
    result = run_cli("prepare", "--dry-run")
    assert result.returncode == 0, result.stderr
    output = result.stdout.replace("\\", "/")
    assert "[plan 1/2] validate frozen Prepare inputs" in output
    assert "compiler/scripts/calibration/preflight.py" in output
    assert "[plan 2/2] prepare calibration tensors" in output
    assert output.index("preflight.py") < output.index("prepare.sh")
