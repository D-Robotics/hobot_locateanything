"""Tests for the Prepare and Calibrate job wrappers."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATE_WRAPPER = REPO_ROOT / "compiler/scripts/calibration/calibrate.sh"
PREPARE_WRAPPER = REPO_ROOT / "compiler/scripts/calibration/prepare.sh"


def _bash() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is unavailable")
    return executable


def _passing_environment_script(tmp_path: Path) -> Path:
    script = tmp_path / "passing_environment.py"
    script.write_text("print('{\"passed\": true}')\n", encoding="utf-8")
    return script


def _write_manifest(path: Path, count: int) -> None:
    path.write_text("{}\n" * count, encoding="utf-8")


def _terminate_after_marker(
    wrapper: Path, env: dict[str, str], marker: Path
) -> subprocess.CompletedProcess[str]:
    wrapper_arg = shlex.quote(wrapper.as_posix())
    marker_arg = shlex.quote(marker.as_posix())
    command = f"""
{wrapper_arg} &
wrapper_pid=$!
for _ in $(seq 1 200); do
  [[ -f {marker_arg} ]] && break
  sleep 0.05
done
if [[ ! -f {marker_arg} ]]; then
  kill -KILL "$wrapper_pid" 2>/dev/null || true
  wait "$wrapper_pid" 2>/dev/null || true
  exit 99
fi
kill -TERM "$wrapper_pid"
wait "$wrapper_pid"
wrapper_exit=$?
printf 'wrapper_exit=%s\n' "$wrapper_exit"
exit 0
"""
    return subprocess.run(
        [_bash(), "-c", command],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_calibration_wrappers_have_valid_bash_syntax():
    for wrapper in (PREPARE_WRAPPER, CALIBRATE_WRAPPER):
        result = subprocess.run(
            [_bash(), "-n", wrapper.as_posix()], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_calibration_wrappers_reenter_through_bash_when_detached():
    for wrapper in (PREPARE_WRAPPER, CALIBRATE_WRAPPER):
        text = wrapper.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "setsid nohup env DETACH=0" in text
        assert 'bash "$0"' in text
        assert "trap finish_job EXIT" in text
        assert "trap 'cancel_job SIGTERM 143' TERM" in text
        assert "trap 'cancel_job SIGINT 130' INT" in text
        assert "trap 'cancel_job SIGHUP 129' HUP" in text
        assert 'local temporary="${EXIT_PATH}.tmp.$$"' in text
        assert 'mv -f "$temporary" "$EXIT_PATH"' in text


def test_prepare_preflight_covers_checkpoint_runtime_dependencies():
    text = PREPARE_WRAPPER.read_text(encoding="utf-8")
    for module in ("cv2", "decord", "lmdb", "packaging", "peft", "requests", "torchvision"):
        assert f"--required-module {module}" in text


def test_prepare_wrapper_only_resumes_when_requested():
    text = PREPARE_WRAPPER.read_text(encoding="utf-8")
    assert 'RESUME=${RESUME:-0}' in text
    assert 'if [[ "$RESUME" == "1" ]]' in text
    assert 'RESUME_ARGS+=(--resume)' in text
    assert '--seed "$SEED" \\\n  "${RESUME_ARGS[@]}"' in text


def test_prepare_wrapper_forwards_release_geometry_from_environment():
    text = PREPARE_WRAPPER.read_text(encoding="utf-8")
    expected = {
        "IMAGE_WIDTH": ("672", '--image-width "$IMAGE_WIDTH"'),
        "IMAGE_HEIGHT": ("672", '--image-height "$IMAGE_HEIGHT"'),
        "RESIZE_MODE": ("letterbox", '--resize-mode "$RESIZE_MODE"'),
        "LETTERBOX_FILL": ("128", '--letterbox-fill "$LETTERBOX_FILL"'),
        "PATCH_SIZE": ("14", '--patch-size "$PATCH_SIZE"'),
        "MERGE_SIZE": ("2", '--merge-size "$MERGE_SIZE"'),
        "HIDDEN_SIZE": ("2048", '--hidden-size "$HIDDEN_SIZE"'),
        "PREFILL_LIMIT": ("1024", '--prefill-limit "$PREFILL_LIMIT"'),
    }
    for name, (default, argument) in expected.items():
        assert f"{name}=${{{name}:-{default}}}" in text
        assert argument in text


def test_prepare_preflight_failure_writes_atomic_failed_terminal_state(tmp_path):
    selected = tmp_path / "selected.jsonl"
    selected.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    upstream = tmp_path / "upstream"
    output = tmp_path / "prepared"
    logs = tmp_path / "logs"
    for directory in (model, upstream, logs):
        directory.mkdir()
    fake_environment = tmp_path / "fake_environment.py"
    fake_environment.write_text(
        "import json\nprint(json.dumps({'passed': False}))\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    prepare_marker = tmp_path / "prepare_started"
    fake_prepare = tmp_path / "must_not_prepare.py"
    fake_prepare.write_text(
        f"from pathlib import Path\nPath({prepare_marker.as_posix()!r}).touch()\n",
        encoding="utf-8",
    )
    exit_path = logs / "prepare.exit.txt"
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "ENVIRONMENT_SCRIPT": fake_environment.as_posix(),
        "PREPARE_SCRIPT": fake_prepare.as_posix(),
        "SELECTED_JSONL": selected.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "UPSTREAM_REPO": upstream.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "LOG_PATH": (logs / "prepare.log").as_posix(),
        "EXIT_PATH": exit_path.as_posix(),
    }
    result = subprocess.run(
        [_bash(), PREPARE_WRAPPER.as_posix()],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert not prepare_marker.exists()
    assert (output / "prepare_environment.json").is_file()
    exit_record = exit_path.read_text(encoding="utf-8")
    assert "status=failed" in exit_record
    assert "exit_code=7" in exit_record
    metadata = json.loads(
        (output / "prepare_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["phase"] == "prepare"
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 7


def test_prepare_sigterm_writes_cancelled_terminal_state(tmp_path):
    selected = tmp_path / "selected.jsonl"
    selected.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    upstream = tmp_path / "upstream"
    output = tmp_path / "prepared"
    logs = tmp_path / "logs"
    for directory in (model, upstream, logs):
        directory.mkdir()
    fake_environment = tmp_path / "fake_environment.py"
    fake_environment.write_text("print('{\"passed\": true}')\n", encoding="utf-8")
    marker = tmp_path / "prepare_started"
    fake_prepare = tmp_path / "slow_prepare.py"
    fake_prepare.write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({marker.as_posix()!r}).touch()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    exit_path = logs / "prepare.exit.txt"
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "ENVIRONMENT_SCRIPT": fake_environment.as_posix(),
        "PREPARE_SCRIPT": fake_prepare.as_posix(),
        "SELECTED_JSONL": selected.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "UPSTREAM_REPO": upstream.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "LOG_PATH": (logs / "prepare.log").as_posix(),
        "EXIT_PATH": exit_path.as_posix(),
    }
    result = _terminate_after_marker(PREPARE_WRAPPER, env, marker)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrapper_exit=143" in result.stdout
    exit_record = exit_path.read_text(encoding="utf-8")
    assert "status=cancelled" in exit_record
    assert "signal=SIGTERM" in exit_record
    assert "exit_code=143" in exit_record
    metadata = json.loads(
        (output / "prepare_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "cancelled"
    assert metadata["signal"] == "SIGTERM"


def test_calibrate_help_exposes_current_defaults_only():
    result = subprocess.run(
        [_bash(), CALIBRATE_WRAPPER.as_posix(), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MAX_SAMPLES         1200" in result.stdout
    assert "CACHE_LEN           4096" in result.stdout
    assert "LM_HEAD_W_BITS      8" in result.stdout
    assert "V6" not in result.stdout
    assert "820" not in result.stdout


def test_calibrate_wrapper_forwards_profile_and_requires_complete_graph_family(tmp_path):
    generated = tmp_path / "generated.jsonl"
    _write_manifest(generated, 1200)
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "statistics"
    logs = tmp_path / "logs"
    logs.mkdir()
    fake_replay = tmp_path / "fake_replay.py"
    fake_replay.write_text(
        """
import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--generated-jsonl'); p.add_argument('--selected-jsonl'); p.add_argument('--upstream-repo')
p.add_argument('--model-path'); p.add_argument('--output-dir')
p.add_argument('--device'); p.add_argument('--dtype'); p.add_argument('--chunk-size')
p.add_argument('--cache-len'); p.add_argument('--max-samples', type=int)
p.add_argument('--checkpoint-samples', type=int); p.add_argument('--image-token-id')
p.add_argument('--component'); p.add_argument('--lm-head-w-bits', type=int)
p.add_argument('--replay-seed', type=int)
p.add_argument('--hidden-rotation-path', default=None)
a = p.parse_args()
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
language_stages = [
  'prefill',
  *(f'pbd_q{q_len}' for q_len in range(6, 13)),
  *(f'ar_q{q_len}' for q_len in range(1, 6)),
]
expected_stages = []
if a.component in {'all', 'vision'}: expected_stages.append('vision')
if a.component in {'all', 'language'}: expected_stages.extend(language_stages)
decode_context = {
  'policy': 'bundle_hash_structural_boundary_v1',
  'sample_count': a.max_samples,
  'passed': True,
  'errors': [],
  'suffix_len': {'min': 0, 'max': 64},
  'past_len': {'min': 600, 'max': 664},
  'depth_buckets': {'zero': 1, '1_31': 0, '32_127': a.max_samples - 1, '128_plus': 0},
  'token_sources': {'target': a.max_samples},
}
coverage = {
  'generated_manifest_sha256': hashlib.sha256(Path(a.generated_jsonl).read_bytes()).hexdigest(),
  'sample_count': a.max_samples,
  'checkpoint_samples': a.checkpoint_samples,
  'task_counts': {'detection': a.max_samples},
  'stage_sample_counts': {stage: a.max_samples for stage in expected_stages},
  'expected_stages': expected_stages,
  'all_stages_executed': True, 'observer_audit_passed': True,
  'decode_context_coverage': decode_context,
  'decode_context_coverage_passed': True,
}
(out / 'calibration_graph_coverage.json').write_text(json.dumps(coverage))
generated = Path(a.generated_jsonl)
(out / 'calibration_scale_manifest.json').write_text(json.dumps({
  'sample_count': a.max_samples, 'checkpoint_samples': a.checkpoint_samples,
  'task_counts': {'detection': a.max_samples},
  'generated_manifest': str(generated),
  'generated_manifest_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
  'replay_seed': a.replay_seed,
  'profile': {
    'component': a.component,
    'language_lm_head_weight_bits': a.lm_head_w_bits,
    'decode_context_policy': 'bundle_hash_structural_boundary_v1',
  }}))
(out / f'scale_convergence_{a.checkpoint_samples}_vs_{a.max_samples}.json').write_text(
  json.dumps({'checkpoint_samples': a.checkpoint_samples, 'full_samples': a.max_samples}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "ENVIRONMENT_SCRIPT": _passing_environment_script(tmp_path).as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "SELECTED_JSONL": generated.as_posix(),
        "UPSTREAM_REPO": model.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "MAX_SAMPLES": "1200",
        "CHECKPOINT_SAMPLES": "512",
        "LOG_PATH": (logs / "calibrate.log").as_posix(),
        "EXIT_PATH": (logs / "calibrate.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), CALIBRATE_WRAPPER.as_posix()],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "independent postflight passed" in result.stdout
    exit_record = (logs / "calibrate.exit.txt").read_text(encoding="utf-8")
    assert "status=succeeded" in exit_record
    assert "exit_code=0" in exit_record
    metadata = json.loads(
        (output / "calibration_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "succeeded"
    assert metadata["component"] == "all"
    assert metadata["lm_head_w_bits"] == 8
    assert metadata["replay_seed"] == 20260729
    assert len(metadata["expected_graph_paths"]) == 14
    assert metadata["expected_graph_paths"][0:3] == ["vision", "prefill", "pbd_q6"]


def test_calibrate_wrapper_fails_preflight_before_replay_for_short_manifest(tmp_path):
    generated = tmp_path / "generated.jsonl"
    generated.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "statistics"
    fake_replay = tmp_path / "must_not_run.py"
    fake_replay.write_text("raise RuntimeError('replay was started')\n", encoding="utf-8")
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "ENVIRONMENT_SCRIPT": _passing_environment_script(tmp_path).as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "SELECTED_JSONL": generated.as_posix(),
        "UPSTREAM_REPO": model.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "MAX_SAMPLES": "1200",
        "CHECKPOINT_SAMPLES": "512",
        "LOG_PATH": (tmp_path / "logs/calibrate.log").as_posix(),
        "EXIT_PATH": (tmp_path / "logs/calibrate.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), CALIBRATE_WRAPPER.as_posix()],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "generated manifest has 1 records, expected 1200" in result.stdout
    assert "replay was started" not in result.stdout + result.stderr
    exit_record = (tmp_path / "logs/calibrate.exit.txt").read_text(encoding="utf-8")
    assert "status=failed" in exit_record
    assert "exit_code=0" not in exit_record
    metadata = json.loads(
        (output / "calibration_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"


def test_calibrate_sigterm_writes_cancelled_terminal_state(tmp_path):
    generated = tmp_path / "generated.jsonl"
    _write_manifest(generated, 1200)
    model = tmp_path / "model"
    output = tmp_path / "statistics"
    logs = tmp_path / "logs"
    model.mkdir()
    logs.mkdir()
    marker = tmp_path / "calibrate_started"
    fake_replay = tmp_path / "slow_calibrate.py"
    fake_replay.write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({marker.as_posix()!r}).touch()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    exit_path = logs / "calibrate.exit.txt"
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "ENVIRONMENT_SCRIPT": _passing_environment_script(tmp_path).as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "SELECTED_JSONL": generated.as_posix(),
        "UPSTREAM_REPO": model.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "MAX_SAMPLES": "1200",
        "CHECKPOINT_SAMPLES": "512",
        "LOG_PATH": (logs / "calibrate.log").as_posix(),
        "EXIT_PATH": exit_path.as_posix(),
    }
    result = _terminate_after_marker(CALIBRATE_WRAPPER, env, marker)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrapper_exit=143" in result.stdout
    exit_record = exit_path.read_text(encoding="utf-8")
    assert "status=cancelled" in exit_record
    assert "signal=SIGTERM" in exit_record
    assert "exit_code=143" in exit_record
    metadata = json.loads(
        (output / "calibration_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "cancelled"
    assert metadata["signal"] == "SIGTERM"


def test_calibrate_wrapper_rejects_nonrelease_sample_contract_before_replay(tmp_path):
    generated = tmp_path / "generated.jsonl"
    _write_manifest(generated, 1199)
    model = tmp_path / "model"
    model.mkdir()
    marker = tmp_path / "replay_started"
    fake_replay = tmp_path / "must_not_run.py"
    fake_replay.write_text(
        f"from pathlib import Path\nPath({marker.as_posix()!r}).touch()\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "ENVIRONMENT_SCRIPT": _passing_environment_script(tmp_path).as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "SELECTED_JSONL": generated.as_posix(),
        "UPSTREAM_REPO": model.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": (tmp_path / "statistics").as_posix(),
        "MAX_SAMPLES": "1199",
        "CHECKPOINT_SAMPLES": "512",
        "LOG_PATH": (tmp_path / "logs/calibrate.log").as_posix(),
        "EXIT_PATH": (tmp_path / "logs/calibrate.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), CALIBRATE_WRAPPER.as_posix()], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "requires MAX_SAMPLES=1200 and CHECKPOINT_SAMPLES=512" in result.stdout
    assert not marker.exists()


def test_calibrate_wrapper_rejects_lm_head_w4_before_replay(tmp_path):
    generated = tmp_path / "generated.jsonl"
    _write_manifest(generated, 1200)
    model = tmp_path / "model"
    model.mkdir()
    marker = tmp_path / "replay_started"
    fake_replay = tmp_path / "must_not_run.py"
    fake_replay.write_text(
        f"from pathlib import Path\nPath({marker.as_posix()!r}).touch()\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "ENVIRONMENT_SCRIPT": _passing_environment_script(tmp_path).as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "SELECTED_JSONL": generated.as_posix(),
        "UPSTREAM_REPO": model.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": (tmp_path / "statistics").as_posix(),
        "MAX_SAMPLES": "1200",
        "CHECKPOINT_SAMPLES": "512",
        "LM_HEAD_W_BITS": "4",
        "LOG_PATH": (tmp_path / "logs/calibrate.log").as_posix(),
        "EXIT_PATH": (tmp_path / "logs/calibrate.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), CALIBRATE_WRAPPER.as_posix()],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires LM_HEAD_W_BITS=8" in result.stdout
    assert not marker.exists()
