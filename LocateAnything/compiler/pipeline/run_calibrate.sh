#!/usr/bin/env bash
# Run LocateAnything activation calibration and record the run status.
set -euo pipefail

if [[ ${1:-} == "--help" ]]; then
  cat <<'EOF'
Run LocateAnything activation calibration.

Required variables:
  GENERATED_JSONL       prepared calibration records
  SELECTED_JSONL        calibration dataset records
  LOCATEANYTHING_SOURCE directory containing locateanything_worker.py
  OUTPUT_DIR            activation statistics output directory
  MAX_SAMPLES           number of records to replay
  CHECKPOINT_SAMPLES    convergence checkpoint below MAX_SAMPLES

Optional variables include REPO_ROOT, PYTHON_BIN, REPLAY_SCRIPT, DEVICE, DTYPE,
IMAGE_TOKEN_ID, HIDDEN_ROTATION_PATH, JOB_NAME, LOG_PATH, EXIT_PATH, META_PATH,
RESUME, DETACH, and DETAILED_STATISTICS.
EOF
  exit 0
fi

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
PYTHON_BIN=${PYTHON_BIN:-python3}
REPLAY_SCRIPT=${REPLAY_SCRIPT:-"$REPO_ROOT/compiler/pipeline/calibrate.py"}
GENERATED_JSONL=${GENERATED_JSONL:?set GENERATED_JSONL to the prepared calibration records}
SELECTED_JSONL=${SELECTED_JSONL:?set SELECTED_JSONL to the calibration dataset records}
LOCATEANYTHING_SOURCE=${LOCATEANYTHING_SOURCE:?set LOCATEANYTHING_SOURCE to the worker source directory}
MODEL_PATH=${MODEL_PATH:-"$REPO_ROOT/compiler/models/LocateAnything-3B"}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to the activation statistics directory}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-float16}
CHUNK_SIZE=${CHUNK_SIZE:-768}
CACHE_LEN=${CACHE_LEN:-4096}
CALIBRATION_COMPONENT=${CALIBRATION_COMPONENT:-all}
VISION_W_BITS=${VISION_W_BITS:-8}
LANGUAGE_W_BITS=${LANGUAGE_W_BITS:-8}
LM_HEAD_W_BITS=${LM_HEAD_W_BITS:-8}
SAMPLING_BACKEND=${SAMPLING_BACKEND:-bpu}
SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-0.7}
SAMPLING_TOP_P=${SAMPLING_TOP_P:-0.9}
SAMPLING_REPETITION_PENALTY=${SAMPLING_REPETITION_PENALTY:-1.1}
REPLAY_SEED=${REPLAY_SEED:-20260729}
MAX_SAMPLES=${MAX_SAMPLES:?set MAX_SAMPLES to the prepared dataset size}
CHECKPOINT_SAMPLES=${CHECKPOINT_SAMPLES:?set CHECKPOINT_SAMPLES below MAX_SAMPLES}
IMAGE_TOKEN_ID=${IMAGE_TOKEN_ID:-151665}
HIDDEN_ROTATION_PATH=${HIDDEN_ROTATION_PATH:-}
RESUME=${RESUME:-0}
DETACH=${DETACH:-0}
DETAILED_STATISTICS=${DETAILED_STATISTICS:-0}

[[ "$DTYPE" == "float16" || "$DTYPE" == "bfloat16" ]] || {
  echo "DTYPE must be float16 or bfloat16; got $DTYPE" >&2
  exit 1
}
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || {
  echo "RESUME must be 0 or 1; got $RESUME" >&2
  exit 1
}
[[ "$DETACH" == "0" || "$DETACH" == "1" ]] || {
  echo "DETACH must be 0 or 1; got $DETACH" >&2
  exit 1
}
[[ "$DETAILED_STATISTICS" == "0" || "$DETAILED_STATISTICS" == "1" ]] || {
  echo "DETAILED_STATISTICS must be 0 or 1; got $DETAILED_STATISTICS" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/compiler/outputs/logs"
JOB_NAME=${JOB_NAME:-"$(basename "$OUTPUT_DIR")_calibrate"}
LOG_PATH=${LOG_PATH:-"$REPO_ROOT/compiler/outputs/logs/${JOB_NAME}.log"}
EXIT_PATH=${EXIT_PATH:-"$REPO_ROOT/compiler/outputs/logs/${JOB_NAME}.exit.txt"}
META_PATH=${META_PATH:-"$OUTPUT_DIR/calibration_job_metadata.json"}
PID_PATH=${PID_PATH:-"$REPO_ROOT/compiler/outputs/logs/${JOB_NAME}.pid"}
LAUNCH_LOG=${LAUNCH_LOG:-"$REPO_ROOT/compiler/outputs/logs/${JOB_NAME}.launcher.log"}
STARTED_AT=$(date --iso-8601=seconds)
mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$EXIT_PATH")" "$(dirname "$META_PATH")"if [[ "$DETACH" == "1" ]]; then
  setsid nohup env DETACH=0 RESUME="$RESUME" bash "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
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
    printf 'status=%s\nexecution_mode=replay\nstarted_at=%s\n' "$state" "$STARTED_AT"
    [[ -z "$exit_code" ]] || printf 'exit_code=%s\n' "$exit_code"
    [[ -z "$signal_name" ]] || printf 'signal=%s\n' "$signal_name"
    [[ -z "$completed_at" ]] || printf 'completed_at=%s\n' "$completed_at"
  } > "$temporary"
  mv -f "$temporary" "$EXIT_PATH"
}

write_initial_metadata() {
  "$PYTHON_BIN" - "$META_PATH" "$STARTED_AT" "$GENERATED_JSONL" "$SELECTED_JSONL" \
    "$LOCATEANYTHING_SOURCE" "$MODEL_PATH" "$OUTPUT_DIR" "$REPLAY_SCRIPT" \
    "$DEVICE" "$DTYPE" "$CHUNK_SIZE" "$CACHE_LEN" "$MAX_SAMPLES" \
    "$CHECKPOINT_SAMPLES" "$IMAGE_TOKEN_ID" "$CALIBRATION_COMPONENT" \
    "$VISION_W_BITS" "$LANGUAGE_W_BITS" "$LM_HEAD_W_BITS" "$REPLAY_SEED" \
  "$HIDDEN_ROTATION_PATH" "$LOG_PATH" "$DETAILED_STATISTICS" "$SAMPLING_BACKEND" \
  "$SAMPLING_TEMPERATURE" "$SAMPLING_TOP_P" "$SAMPLING_REPETITION_PENALTY" "$REPO_ROOT" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

(
    path, started_at, generated, selected, source, model, output, replay, device,
    dtype, chunk, cache, max_samples, checkpoint, image_token, component,
    vision_w_bits, language_w_bits, lm_head_w_bits, replay_seed, rotation,
    log_path, detailed_statistics, sampling_backend, sampling_temperature,
    sampling_top_p, sampling_repetition_penalty, repo_root,
) = sys.argv[1:]
sys.path.insert(0, str(Path(repo_root) / "compiler"))
from model.graphs import language_graph_set

profile = language_graph_set()
expected_graph_paths = []
if component in {"all", "vision"}:
    expected_graph_paths.append("vision")
if component in {"all", "language"}:
    expected_graph_paths.extend(profile.calibration_stages)
value = {
    "schema_version": 1,
    "phase": "calibrate",
    "component": component,
    "status": "running",
    "execution_mode": "replay",
    "started_at": started_at,
    "hostname": socket.gethostname(),
    "wrapper_pid": os.getppid(),
    "generated_jsonl": generated,
    "selected_jsonl": selected,
    "locateanything_source": source,
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
    "vision_w_bits": int(vision_w_bits),
    "language_w_bits": int(language_w_bits),
    "lm_head_w_bits": int(lm_head_w_bits),
    "replay_seed": int(replay_seed),
    "graph_set": profile.name,
    "detailed_statistics": detailed_statistics == "1",
    "sampling_backend": sampling_backend,
    "sampling_temperature": float(sampling_temperature),
    "sampling_top_p": float(sampling_top_p),
    "sampling_repetition_penalty": float(sampling_repetition_penalty),
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
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
value.update({
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "signal": sys.argv[4] or None,
    "completed_at": sys.argv[5],
})
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
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
echo "[calibrate] output=$OUTPUT_DIR component=$CALIBRATION_COMPONENT graph_set=fused_decode device=$DEVICE dtype=$DTYPE samples=$MAX_SAMPLES checkpoint=$CHECKPOINT_SAMPLES vision_w_bits=$VISION_W_BITS language_w_bits=$LANGUAGE_W_BITS lm_head_w_bits=$LM_HEAD_W_BITS detailed_statistics=$DETAILED_STATISTICS replay_seed=$REPLAY_SEED" | tee -a "$LOG_PATH"

replay_args=(
  --generated-jsonl "$GENERATED_JSONL"
  --selected-jsonl "$SELECTED_JSONL"
  --source-dir "$LOCATEANYTHING_SOURCE"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --component "$CALIBRATION_COMPONENT"
  --chunk-size "$CHUNK_SIZE"
  --cache-len "$CACHE_LEN"
  --vision-w-bits "$VISION_W_BITS"
  --language-w-bits "$LANGUAGE_W_BITS"
  --lm-head-w-bits "$LM_HEAD_W_BITS"
  --max-samples "$MAX_SAMPLES"
  --checkpoint-samples "$CHECKPOINT_SAMPLES"
  --image-token-id "$IMAGE_TOKEN_ID"
  --replay-seed "$REPLAY_SEED"
  --sampling-backend "$SAMPLING_BACKEND"
  --sampling-temperature "$SAMPLING_TEMPERATURE"
  --sampling-top-p "$SAMPLING_TOP_P"
  --sampling-repetition-penalty "$SAMPLING_REPETITION_PENALTY"
)
if [[ "$DETAILED_STATISTICS" == "1" ]]; then
  replay_args+=(--detailed-statistics)
fi
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  replay_args+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$RESUME" == "1" ]]; then
  replay_args+=(--resume)
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

echo "[calibrate] replay completed" | tee -a "$LOG_PATH"
