from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "compiler/scripts/calibration/calibrate.sh"


def _bash() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is unavailable")
    return executable


def test_d4_wrapper_has_valid_bash_syntax():
    result = subprocess.run([_bash(), "-n", WRAPPER.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_calibrate_help_exposes_current_defaults_only():
    result = subprocess.run(
        [_bash(), WRAPPER.as_posix(), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "MAX_SAMPLES         620" in result.stdout
    assert "CACHE_LEN           4096" in result.stdout
    assert "LM_HEAD_W_BITS      8" in result.stdout
    assert "V6" not in result.stdout
    assert "820" not in result.stdout


def test_calibrate_wrapper_forwards_profile_and_requires_complete_graph_family(tmp_path):
    generated = tmp_path / "generated.jsonl"
    generated.write_text("{}\n{}\n{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "d4"
    logs = tmp_path / "logs"
    logs.mkdir()
    fake_replay = tmp_path / "fake_replay.py"
    fake_replay.write_text(
        """
import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--generated-jsonl'); p.add_argument('--model-path'); p.add_argument('--output-dir')
p.add_argument('--device'); p.add_argument('--dtype'); p.add_argument('--chunk-size')
p.add_argument('--cache-len'); p.add_argument('--max-samples', type=int)
p.add_argument('--checkpoint-samples', type=int); p.add_argument('--image-token-id')
p.add_argument('--stage'); p.add_argument('--lm-head-w-bits', type=int)
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
if a.stage in {'all', 'vision'}: expected_stages.append('vision')
if a.stage in {'all', 'language'}: expected_stages.extend(language_stages)
coverage = {
  'generated_manifest_sha256': hashlib.sha256(Path(a.generated_jsonl).read_bytes()).hexdigest(),
  'sample_count': a.max_samples,
  'checkpoint_samples': a.checkpoint_samples,
  'task_counts': {'detection': a.max_samples},
  'stage_sample_counts': {stage: a.max_samples for stage in expected_stages},
  'expected_stages': expected_stages,
  'all_stages_executed': True, 'observer_audit_passed': True,
}
(out / 'calibration_graph_coverage.json').write_text(json.dumps(coverage))
generated = Path(a.generated_jsonl)
(out / 'calibration_scale_manifest.json').write_text(json.dumps({
  'sample_count': a.max_samples, 'checkpoint_samples': a.checkpoint_samples,
  'task_counts': {'detection': a.max_samples},
  'generated_manifest': str(generated),
  'generated_manifest_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
  'replay_seed': a.replay_seed,
  'profile': {'stage': a.stage, 'language_lm_head_weight_bits': a.lm_head_w_bits}}))
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
        "GENERATED_JSONL": generated.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "MAX_SAMPLES": "3",
        "CHECKPOINT_SAMPLES": "2",
        "LOG_PATH": (logs / "d4.log").as_posix(),
        "EXIT_PATH": (logs / "d4.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), WRAPPER.as_posix()], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "independent postflight passed" in result.stdout
    assert "exit_code=0" in (logs / "d4.exit.txt").read_text(encoding="utf-8")
    metadata = json.loads(
        (output / "calibration_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "passed"
    assert metadata["component"] == "all"
    assert metadata["lm_head_w_bits"] == 8
    assert metadata["replay_seed"] == 20260729
    assert len(metadata["expected_graph_paths"]) == 14
    assert metadata["expected_graph_paths"][0:3] == ["vision", "prefill", "pbd_q6"]


def test_d4_wrapper_fails_preflight_before_replay_for_short_manifest(tmp_path):
    generated = tmp_path / "generated.jsonl"
    generated.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "d4"
    fake_replay = tmp_path / "must_not_run.py"
    fake_replay.write_text("raise RuntimeError('replay was started')\n", encoding="utf-8")
    env = {
        **os.environ,
        "REPO_ROOT": REPO_ROOT.as_posix(),
        "PYTHON_BIN": os.environ.get("PYTHON", "python"),
        "REPLAY_SCRIPT": fake_replay.as_posix(),
        "GENERATED_JSONL": generated.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "OUTPUT_DIR": output.as_posix(),
        "MAX_SAMPLES": "2",
        "CHECKPOINT_SAMPLES": "1",
        "LOG_PATH": (tmp_path / "logs/d4.log").as_posix(),
        "EXIT_PATH": (tmp_path / "logs/d4.exit.txt").as_posix(),
    }
    result = subprocess.run(
        [_bash(), WRAPPER.as_posix()], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "fewer than MAX_SAMPLES=2" in result.stdout
    assert "replay was started" not in result.stdout + result.stderr
    assert "exit_code=0" not in (tmp_path / "logs/d4.exit.txt").read_text(encoding="utf-8")
    metadata = json.loads(
        (output / "calibration_job_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
