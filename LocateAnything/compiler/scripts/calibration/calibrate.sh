#!/usr/bin/env bash
# Reproducible activation calibration with preflight checks and durable evidence.
set -euo pipefail

if [[ ${1:-} == "--help" ]]; then
  cat <<'EOF'
Run LocateAnything activation calibration.

Required variables:
  GENERATED_JSONL  prepared calibration manifest
  OUTPUT_DIR       activation statistics output directory

Release defaults:
  MODEL_PATH          workspace/models/LocateAnything-3B
  STAGE               all
  MAX_SAMPLES         1200
  CHECKPOINT_SAMPLES  512
  CHUNK_SIZE          1024
  CACHE_LEN           4096
  LM_HEAD_W_BITS      8
  REPLAY_SEED         20260729

Optional variables include REPO_ROOT, PYTHON_BIN, REPLAY_SCRIPT, DEVICE, DTYPE,
IMAGE_TOKEN_ID, HIDDEN_ROTATION_PATH, JOB_NAME, LOG_PATH, EXIT_PATH, and META_PATH.
Set PREFLIGHT_ONLY=1 to validate inputs without a GPU run.
EOF
  exit 0
fi

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../../.." && pwd)"}
PYTHON_BIN=${PYTHON_BIN:-python3}
REPLAY_SCRIPT=${REPLAY_SCRIPT:-"$REPO_ROOT/compiler/scripts/calibration/calibrate.py"}
GENERATED_JSONL=${GENERATED_JSONL:?set GENERATED_JSONL to the prepared calibration manifest}
MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/workspace/models/LocateAnything-3B"}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to the activation calibration output directory}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
CHUNK_SIZE=${CHUNK_SIZE:-1024}
CACHE_LEN=${CACHE_LEN:-4096}
STAGE=${STAGE:-all}
LM_HEAD_W_BITS=${LM_HEAD_W_BITS:-8}
REPLAY_SEED=${REPLAY_SEED:-20260729}
MAX_SAMPLES=${MAX_SAMPLES:-1200}
CHECKPOINT_SAMPLES=${CHECKPOINT_SAMPLES:-512}
IMAGE_TOKEN_ID=${IMAGE_TOKEN_ID:-151665}
HIDDEN_ROTATION_PATH=${HIDDEN_ROTATION_PATH:-}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/workspace/logs"
JOB_NAME=${JOB_NAME:-"$(basename "$OUTPUT_DIR")_calibrate"}
LOG_PATH=${LOG_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.log"}
EXIT_PATH=${EXIT_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.exit.txt"}
META_PATH=${META_PATH:-"$OUTPUT_DIR/calibration_job_metadata.json"}
PID_PATH=${PID_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.pid"}
LAUNCH_LOG=${LAUNCH_LOG:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.launcher.log"}
STARTED_AT=$(date --iso-8601=seconds)
mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$EXIT_PATH")" "$(dirname "$META_PATH")"

if [[ "${DETACH:-0}" == "1" ]]; then
  setsid nohup env DETACH=0 bash "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[calibrate] detached_pid=$child_pid"
  echo "[calibrate] launcher_log=$LAUNCH_LOG"
  exit 0
fi

ACTIVE_CHILD_PID=""
CANCEL_SIGNAL=""
CANCEL_EXIT_CODE=""

write_exit_record() {
  local state=$1
  local exit_code=${2:-}
  local signal_name=${3:-}
  local completed_at=${4:-}
  local temporary="${EXIT_PATH}.tmp.$$"
  {
    printf 'status=%s\nexecution_mode=%s\nstarted_at=%s\n' \
      "$state" "$([[ "$PREFLIGHT_ONLY" == "1" ]] && echo preflight_only || echo replay)" \
      "$STARTED_AT"
    [[ -z "$exit_code" ]] || printf 'exit_code=%s\n' "$exit_code"
    [[ -z "$signal_name" ]] || printf 'signal=%s\n' "$signal_name"
    [[ -z "$completed_at" ]] || printf 'completed_at=%s\n' "$completed_at"
  } > "$temporary"
  mv -f "$temporary" "$EXIT_PATH"
}

write_initial_metadata() {
  "$PYTHON_BIN" - "$META_PATH" "$STARTED_AT" "$GENERATED_JSONL" "$MODEL_PATH" \
    "$OUTPUT_DIR" "$REPLAY_SCRIPT" "$DEVICE" "$DTYPE" "$CHUNK_SIZE" "$CACHE_LEN" \
    "$MAX_SAMPLES" "$CHECKPOINT_SAMPLES" "$IMAGE_TOKEN_ID" "$STAGE" \
    "$LM_HEAD_W_BITS" "$REPLAY_SEED" "$HIDDEN_ROTATION_PATH" "$LOG_PATH" \
    "$PREFLIGHT_ONLY" <<'PY'
import json, os, socket, sys
from pathlib import Path

(path, started_at, generated, model, output, replay, device, dtype, chunk, cache,
 max_samples, checkpoint, image_token, stage, lm_head_w_bits, replay_seed,
 rotation, log_path, preflight_only) = sys.argv[1:]
language_stages = [
    "prefill",
    *(f"pbd_q{q_len}" for q_len in range(6, 13)),
    *(f"ar_q{q_len}" for q_len in range(1, 6)),
]
expected_graph_paths = []
if stage in {"all", "vision"}:
    expected_graph_paths.append("vision")
if stage in {"all", "language"}:
    expected_graph_paths.extend(language_stages)
value = {
    "schema_version": 1,
    "phase": "calibrate",
    "component": stage,
    "status": "running",
    "execution_mode": "preflight_only" if preflight_only == "1" else "replay",
    "started_at": started_at,
    "hostname": socket.gethostname(),
    "wrapper_pid": os.getppid(),
    "generated_jsonl": generated,
    "model_path": model,
    "output_dir": output,
    "replay_script": replay,
    "device": device,
    "dtype": dtype,
    "chunk_size": int(chunk),
    "cache_len": int(cache),
    "max_samples": int(max_samples),
    "checkpoint_samples": int(checkpoint),
    "image_token_id": int(image_token),
    "lm_head_w_bits": int(lm_head_w_bits),
    "replay_seed": int(replay_seed),
    "hidden_rotation_path": rotation or None,
    "expected_graph_paths": expected_graph_paths,
    "log_path": log_path,
}
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

finish_job() {
  local exit_code=$?
  trap - EXIT TERM INT HUP
  set +e
  local state=failed
  local signal_name=""
  if [[ -n "$CANCEL_SIGNAL" ]]; then
    state=cancelled
    signal_name=$CANCEL_SIGNAL
    exit_code=$CANCEL_EXIT_CODE
  elif [[ "$exit_code" -eq 0 ]]; then
    state=succeeded
  fi
  local completed_at
  completed_at=$(date --iso-8601=seconds)
  write_exit_record "$state" "$exit_code" "$signal_name" "$completed_at"
  "$PYTHON_BIN" - "$META_PATH" "$state" "$exit_code" "$signal_name" "$completed_at" <<'PY' || true
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_file():
    value = json.loads(path.read_text(encoding="utf-8"))
else:
    value = {"schema_version": 1, "phase": "calibrate"}
value.update({
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "signal": sys.argv[4] or None,
    "completed_at": sys.argv[5],
})
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
  echo "[calibrate] status=$state exit_code=$exit_code exit_record=$EXIT_PATH" | tee -a "$LOG_PATH"
  exit "$exit_code"
}

cancel_job() {
  CANCEL_SIGNAL=$1
  CANCEL_EXIT_CODE=$2
  if [[ -n "$ACTIVE_CHILD_PID" ]] && kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
    kill -s "$CANCEL_SIGNAL" "$ACTIVE_CHILD_PID" 2>/dev/null || true
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
  fi
  exit "$CANCEL_EXIT_CODE"
}
trap finish_job EXIT
trap 'cancel_job SIGTERM 143' TERM
trap 'cancel_job SIGINT 130' INT
trap 'cancel_job SIGHUP 129' HUP

write_exit_record running
write_initial_metadata

echo "[calibrate] manifest=$GENERATED_JSONL" | tee -a "$LOG_PATH"
echo "[calibrate] output=$OUTPUT_DIR stage=$STAGE device=$DEVICE dtype=$DTYPE samples=$MAX_SAMPLES checkpoint=$CHECKPOINT_SAMPLES lm_head_w_bits=$LM_HEAD_W_BITS replay_seed=$REPLAY_SEED" | tee -a "$LOG_PATH"

"$PYTHON_BIN" - "$REPLAY_SCRIPT" "$GENERATED_JSONL" "$MODEL_PATH" \
  "$MAX_SAMPLES" "$CHECKPOINT_SAMPLES" "$STAGE" "$LM_HEAD_W_BITS" \
  "$REPLAY_SEED" "$HIDDEN_ROTATION_PATH" 2>&1 <<'PY' | tee -a "$LOG_PATH"
import sys
from pathlib import Path

replay, manifest, model, max_samples, checkpoint, stage, lm_head_w_bits, replay_seed, rotation = sys.argv[1:]
max_samples, checkpoint = int(max_samples), int(checkpoint)
errors = []
if not Path(replay).is_file():
    errors.append(f"replay script is not a file: {replay}")
if not Path(manifest).is_file():
    errors.append(f"generated manifest is not a file: {manifest}")
if not Path(model).is_dir():
    errors.append(f"model path is not a directory: {model}")
if rotation and not Path(rotation).is_file():
    errors.append(f"hidden rotation is not a file: {rotation}")
if max_samples <= 0:
    errors.append("MAX_SAMPLES must be positive")
if checkpoint <= 0 or checkpoint >= max_samples:
    errors.append("CHECKPOINT_SAMPLES must be positive and smaller than MAX_SAMPLES")
if stage not in {"all", "vision", "language"}:
    errors.append(f"STAGE must be all, vision, or language; got {stage}")
if int(lm_head_w_bits) not in {4, 8}:
    errors.append(f"LM_HEAD_W_BITS must be 4 or 8; got {lm_head_w_bits}")
if int(replay_seed) < 0:
    errors.append(f"REPLAY_SEED must be non-negative; got {replay_seed}")
if Path(manifest).is_file():
    with Path(manifest).open("r", encoding="utf-8") as handle:
        record_count = sum(bool(line.strip()) for line in handle)
    if record_count < max_samples:
        errors.append(f"manifest has {record_count} records, fewer than MAX_SAMPLES={max_samples}")
if errors:
    raise SystemExit("calibration preflight failed:\n- " + "\n- ".join(errors))
print(f"[calibration preflight] passed: records>={max_samples}, checkpoint={checkpoint}", flush=True)
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[calibrate] preflight-only; activation replay was not started" | tee -a "$LOG_PATH"
  exit 0
fi

replay_args=(
  --generated-jsonl "$GENERATED_JSONL"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --stage "$STAGE"
  --chunk-size "$CHUNK_SIZE"
  --cache-len "$CACHE_LEN"
  --lm-head-w-bits "$LM_HEAD_W_BITS"
  --max-samples "$MAX_SAMPLES"
  --checkpoint-samples "$CHECKPOINT_SAMPLES"
  --image-token-id "$IMAGE_TOKEN_ID"
  --replay-seed "$REPLAY_SEED"
)
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  replay_args+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$REPLAY_SCRIPT" "${replay_args[@]}" \
  > >(tee -a "$LOG_PATH") 2>&1 &
ACTIVE_CHILD_PID=$!
set +e
wait "$ACTIVE_CHILD_PID"
replay_status=$?
ACTIVE_CHILD_PID=""
set -e
if [[ "$replay_status" -ne 0 ]]; then
  echo "[calibrate] replay failed with exit_code=$replay_status" | tee -a "$LOG_PATH"
  exit "$replay_status"
fi

# A zero replay exit is accepted only when all durable calibration evidence agrees.
"$PYTHON_BIN" - "$OUTPUT_DIR" "$MAX_SAMPLES" "$CHECKPOINT_SAMPLES" "$STAGE" \
  "$LM_HEAD_W_BITS" "$REPLAY_SEED" 2>&1 <<'PY' | tee -a "$LOG_PATH"
import hashlib, json, sys
from pathlib import Path

output = Path(sys.argv[1])
max_samples, checkpoint = map(int, sys.argv[2:4])
stage = sys.argv[4]
lm_head_w_bits = int(sys.argv[5])
replay_seed = int(sys.argv[6])
coverage_path = output / "calibration_graph_coverage.json"
manifest_path = output / "calibration_scale_manifest.json"
convergence_path = output / f"scale_convergence_{checkpoint}_vs_{max_samples}.json"
missing = [str(path) for path in (coverage_path, manifest_path, convergence_path) if not path.is_file()]
if missing:
    raise SystemExit("calibration postflight missing artifacts:\n- " + "\n- ".join(missing))
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
counts = coverage.get("stage_sample_counts", {})
errors = []
if coverage.get("sample_count") != max_samples:
    errors.append(f"coverage sample_count={coverage.get('sample_count')} expected {max_samples}")
if coverage.get("checkpoint_samples") != checkpoint:
    errors.append("coverage checkpoint_samples does not match the requested run")
if coverage.get("task_counts") != manifest.get("task_counts"):
    errors.append("coverage task_counts does not match the scale manifest")
language_stages = [
    "prefill",
    *(f"pbd_q{q_len}" for q_len in range(6, 13)),
    *(f"ar_q{q_len}" for q_len in range(1, 6)),
]
expected_stages = []
if stage in {"all", "vision"}:
    expected_stages.append("vision")
if stage in {"all", "language"}:
    expected_stages.extend(language_stages)
for graph_stage in expected_stages:
    if counts.get(graph_stage) != max_samples:
        errors.append(f"{graph_stage} count={counts.get(graph_stage)} expected {max_samples}")
if coverage.get("expected_stages") != expected_stages:
    errors.append("expected_stages does not match the complete graph-family contract")
if coverage.get("all_stages_executed") is not True:
    errors.append("all_stages_executed is not true")
audit_passed = coverage.get(
    "activation_statistics_audit_passed",
    coverage.get("observer_audit_passed"),
)
if audit_passed is not True:
    errors.append("activation_statistics_audit_passed is not true")
if manifest.get("sample_count") != max_samples or manifest.get("checkpoint_samples") != checkpoint:
    errors.append("scale manifest sample/checkpoint counts do not match the requested run")
profile = manifest.get("profile", {})
if profile.get("stage") != stage:
    errors.append(f"scale manifest stage={profile.get('stage')} expected {stage}")
if stage in {"all", "language"} and profile.get("language_lm_head_weight_bits") != lm_head_w_bits:
    errors.append("scale manifest lm_head weight bits do not match the requested run")
if manifest.get("replay_seed") != replay_seed:
    errors.append("scale manifest replay_seed does not match the requested run")
generated_path = Path(manifest.get("generated_manifest", ""))
if not generated_path.is_file():
    errors.append(f"scale manifest generated_manifest is unavailable: {generated_path}")
else:
    digest = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    if manifest.get("generated_manifest_sha256") != digest:
        errors.append("scale manifest generated_manifest_sha256 does not match its input")
    if coverage.get("generated_manifest_sha256") != digest:
        errors.append("coverage generated_manifest_sha256 does not match its input")
if convergence.get("checkpoint_samples") != checkpoint or convergence.get("full_samples") != max_samples:
    errors.append("convergence sample/checkpoint counts do not match the requested run")
if errors:
    raise SystemExit("calibration postflight failed:\n- " + "\n- ".join(errors))
print(
    f"[calibration postflight] passed: samples={max_samples}, "
    f"component={stage}, stages={len(expected_stages)}, "
    f"checkpoint={checkpoint}, observers=passed",
    flush=True,
)
PY

echo "[calibrate] replay and independent postflight passed" | tee -a "$LOG_PATH"
