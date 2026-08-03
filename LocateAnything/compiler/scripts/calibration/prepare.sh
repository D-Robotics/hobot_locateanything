#!/usr/bin/env bash
# Reproducible calibration-input preparation with durable job metadata and logs.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../../.." && pwd)"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$REPO_ROOT/compiler/scripts/calibration/prepare.py"}
SELECTED_JSONL=${SELECTED_JSONL:?set SELECTED_JSONL to the selected dataset index}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new or resume-compatible output directory}
UPSTREAM_REPO=${UPSTREAM_REPO:?set UPSTREAM_REPO to the Embodied source directory}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the LocateAnything-3B checkpoint}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
IMAGE_WIDTH=${IMAGE_WIDTH:-672}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-672}
RESIZE_MODE=${RESIZE_MODE:-letterbox}
LETTERBOX_FILL=${LETTERBOX_FILL:-128}
PATCH_SIZE=${PATCH_SIZE:-14}
MERGE_SIZE=${MERGE_SIZE:-2}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
PREFILL_LIMIT=${PREFILL_LIMIT:-1024}
SLOW_SAMPLES=${SLOW_SAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
SEED=${SEED:-20260729}
RESUME=${RESUME:-0}

[[ "$DTYPE" == "bfloat16" ]] || {
  echo "release Prepare requires DTYPE=bfloat16; got $DTYPE"
  exit 1
}

[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || {
  echo "RESUME must be 0 or 1"
  exit 1
}

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/artifacts/logs"
JOB_NAME=${JOB_NAME:-"$(basename "$OUTPUT_DIR")_prepare"}
LOG_PATH=${LOG_PATH:-"$REPO_ROOT/artifacts/logs/${JOB_NAME}.log"}
EXIT_PATH=${EXIT_PATH:-"$REPO_ROOT/artifacts/logs/${JOB_NAME}.exit.txt"}
META_PATH=${META_PATH:-"$OUTPUT_DIR/prepare_job_metadata.json"}
PID_PATH=${PID_PATH:-"$REPO_ROOT/artifacts/logs/${JOB_NAME}.pid"}
LAUNCH_LOG=${LAUNCH_LOG:-"$REPO_ROOT/artifacts/logs/${JOB_NAME}.launcher.log"}
ENVIRONMENT_PATH=${ENVIRONMENT_PATH:-"$OUTPUT_DIR/prepare_environment.json"}
ENVIRONMENT_SCRIPT=${ENVIRONMENT_SCRIPT:-"$REPO_ROOT/compiler/scripts/common/environment.py"}

if [[ "${DETACH:-0}" == "1" ]]; then
  setsid nohup env DETACH=0 bash "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[prepare] detached_pid=$child_pid"
  echo "[prepare] launcher_log=$LAUNCH_LOG"
  exit 0
fi

STARTED_AT=$(date --iso-8601=seconds)
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
    printf 'status=%s\nstarted_at=%s\n' "$state" "$STARTED_AT"
    [[ -z "$exit_code" ]] || printf 'exit_code=%s\n' "$exit_code"
    [[ -z "$signal_name" ]] || printf 'signal=%s\n' "$signal_name"
    [[ -z "$completed_at" ]] || printf 'completed_at=%s\n' "$completed_at"
  } > "$temporary"
  mv -f "$temporary" "$EXIT_PATH"
}

update_terminal_metadata() {
  local state=$1
  local exit_code=$2
  local signal_name=$3
  local completed_at=$4
  "$PYTHON_BIN" - "$META_PATH" "$state" "$exit_code" "$signal_name" "$completed_at" <<'PY' || true
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_file():
    value = json.loads(path.read_text(encoding="utf-8"))
else:
    value = {"schema_version": 1, "phase": "prepare"}
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
  update_terminal_metadata "$state" "$exit_code" "$signal_name" "$completed_at"
  echo "[prepare] status=$state exit_code=$exit_code exit_record=$EXIT_PATH" | tee -a "$LOG_PATH"
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

"$PYTHON_BIN" - "$META_PATH" <<PY
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "phase": "prepare",
    "status": "running",
    "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "hostname": socket.gethostname(),
    "wrapper_pid": os.getppid(),
    "selected_jsonl": "${SELECTED_JSONL}",
    "output_dir": "${OUTPUT_DIR}",
    "upstream_repo": "${UPSTREAM_REPO}",
    "model_path": "${MODEL_PATH}",
    "prepare_script": "${PREPARE_SCRIPT}",
    "device": "${DEVICE}",
    "dtype": "${DTYPE}",
    "image_width": int("${IMAGE_WIDTH}"),
    "image_height": int("${IMAGE_HEIGHT}"),
    "resize_mode": "${RESIZE_MODE}",
    "letterbox_fill": int("${LETTERBOX_FILL}"),
    "patch_size": int("${PATCH_SIZE}"),
    "merge_size": int("${MERGE_SIZE}"),
    "hidden_size": int("${HIDDEN_SIZE}"),
    "prefill_limit": int("${PREFILL_LIMIT}"),
    "slow_samples": int("${SLOW_SAMPLES}"),
    "max_new_tokens": int("${MAX_NEW_TOKENS}"),
    "seed": int("${SEED}"),
    "resume": bool(int("${RESUME}")),
    "log_path": "${LOG_PATH}",
    "pid_path": "${PID_PATH}",
    "launcher_log": "${LAUNCH_LOG}",
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

environment_temporary="${ENVIRONMENT_PATH}.tmp.$$"
set +e
"$PYTHON_BIN" "$ENVIRONMENT_SCRIPT" \
  --profile prepare \
  --model-path "$MODEL_PATH" \
  --selected-jsonl "$SELECTED_JSONL" \
  --upstream-repo "$UPSTREAM_REPO" \
  --resource-path "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --require-cuda \
  --required-module cv2 \
  --required-module decord \
  --required-module lmdb \
  --required-module packaging \
  --required-module peft \
  --required-module requests \
  --required-module torchvision \
  > "$environment_temporary"
environment_status=$?
set -e
mv -f "$environment_temporary" "$ENVIRONMENT_PATH"
if [[ "$environment_status" -ne 0 ]]; then
  echo "[prepare] environment preflight failed exit_code=$environment_status" | tee -a "$LOG_PATH"
  exit "$environment_status"
fi

echo "[prepare] manifest=$SELECTED_JSONL" | tee -a "$LOG_PATH"
echo "[prepare] output=$OUTPUT_DIR device=$DEVICE dtype=$DTYPE slow_samples=$SLOW_SAMPLES resume=$RESUME" | tee -a "$LOG_PATH"

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PREPARE_SCRIPT" generate \
  --selected-jsonl "$SELECTED_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --upstream-repo "$UPSTREAM_REPO" \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --image-width "$IMAGE_WIDTH" \
  --image-height "$IMAGE_HEIGHT" \
  --resize-mode "$RESIZE_MODE" \
  --letterbox-fill "$LETTERBOX_FILL" \
  --patch-size "$PATCH_SIZE" \
  --merge-size "$MERGE_SIZE" \
  --hidden-size "$HIDDEN_SIZE" \
  --prefill-limit "$PREFILL_LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --slow-samples "$SLOW_SAMPLES" \
  --seed "$SEED" \
  "${RESUME_ARGS[@]}" > >(tee -a "$LOG_PATH") 2>&1 &
ACTIVE_CHILD_PID=$!
set +e
wait "$ACTIVE_CHILD_PID"
status=$?
ACTIVE_CHILD_PID=""
set -e
exit "$status"
