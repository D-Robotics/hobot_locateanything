#!/usr/bin/env bash
# Reproducible calibration-input preparation with durable job metadata and logs.
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$0")/../../.." && pwd)"}
PYTHON_BIN=${PYTHON_BIN:-python3}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-"$REPO_ROOT/compiler/scripts/calibration/prepare.py"}
SELECTED_JSONL=${SELECTED_JSONL:?set SELECTED_JSONL to the selected bundle manifest}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new or resume-compatible output directory}
UPSTREAM_REPO=${UPSTREAM_REPO:?set UPSTREAM_REPO to the Embodied source directory}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the LocateAnything-3B checkpoint}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
SLOW_SAMPLES=${SLOW_SAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
SEED=${SEED:-20260720}

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/workspace/logs"
JOB_NAME=${JOB_NAME:-"$(basename "$OUTPUT_DIR")_prepare"}
LOG_PATH=${LOG_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.log"}
EXIT_PATH=${EXIT_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.exit.txt"}
META_PATH="$OUTPUT_DIR/prepare_job_metadata.json"
PID_PATH=${PID_PATH:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.pid"}
LAUNCH_LOG=${LAUNCH_LOG:-"$REPO_ROOT/workspace/logs/${JOB_NAME}.launcher.log"}

if [[ "${DETACH:-0}" == "1" ]]; then
  setsid nohup env DETACH=0 "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[prepare] detached_pid=$child_pid"
  echo "[prepare] launcher_log=$LAUNCH_LOG"
  exit 0
fi

printf 'status=running\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "$EXIT_PATH"

"$PYTHON_BIN" "$REPO_ROOT/compiler/scripts/common/environment.py" \
  --model-path "$MODEL_PATH" \
  --selected-jsonl "$SELECTED_JSONL" \
  --upstream-repo "$UPSTREAM_REPO" \
  --require-cuda \
  --required-module decord \
  --required-module lmdb \
  > "$OUTPUT_DIR/d3_environment.json"

"$PYTHON_BIN" - "$META_PATH" <<PY
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "hostname": socket.gethostname(),
    "pid": os.getppid(),
    "selected_jsonl": "${SELECTED_JSONL}",
    "output_dir": "${OUTPUT_DIR}",
    "upstream_repo": "${UPSTREAM_REPO}",
    "model_path": "${MODEL_PATH}",
    "prepare_script": "${PREPARE_SCRIPT}",
    "device": "${DEVICE}",
    "dtype": "${DTYPE}",
    "slow_samples": int("${SLOW_SAMPLES}"),
    "max_new_tokens": int("${MAX_NEW_TOKENS}"),
    "seed": int("${SEED}"),
    "log_path": "${LOG_PATH}",
    "pid_path": "${PID_PATH}",
    "launcher_log": "${LAUNCH_LOG}",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "[prepare] manifest=$SELECTED_JSONL" | tee -a "$LOG_PATH"
echo "[prepare] output=$OUTPUT_DIR device=$DEVICE dtype=$DTYPE slow_samples=$SLOW_SAMPLES" | tee -a "$LOG_PATH"

set +e
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PREPARE_SCRIPT" generate \
  --selected-jsonl "$SELECTED_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --upstream-repo "$UPSTREAM_REPO" \
  --model-path "$MODEL_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --image-width 672 \
  --image-height 672 \
  --resize-mode letterbox \
  --prefill-limit 1024 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --slow-samples "$SLOW_SAMPLES" \
  --seed "$SEED" \
  --resume 2>&1 | tee -a "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\ncompleted_at=%s\n' "$status" "$(date --iso-8601=seconds)" > "$EXIT_PATH"
echo "[prepare] exit_code=$status" | tee -a "$LOG_PATH"
exit "$status"
