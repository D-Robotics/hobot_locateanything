#!/usr/bin/env bash
# Compile a baseline HBM for qwen2_5-vl-3b (nash-p, w4) using the OELLM S600
# toolchain. Runs in the background via nohup so long compiles survive SSH
# disconnects. Environment variables override the defaults below.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/oe_locateanything}"
LEAP_LLM_SRC="${LEAP_LLM_SRC:-$REPO_ROOT/main/language/leap_llm_src}"

MODEL_NAME="${MODEL_NAME:-qwen2_5-vl-3b}"
MARCH="${MARCH:-nash-p}"
W_BITS="${W_BITS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CACHE_LEN="${CACHE_LEN:-1024}"
IMAGE_WIDTH="${IMAGE_WIDTH:-448}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-448}"
DEVICE="${DEVICE:-cuda:0}"
VIT_CORE_NUM="${VIT_CORE_NUM:-4}"
PREFILL_CORE_NUM="${PREFILL_CORE_NUM:-4}"
DECODE_CORE_NUM="${DECODE_CORE_NUM:-4}"
JOBS="${JOBS:-16}"

INPUT_MODEL_PATH="${INPUT_MODEL_PATH:-$REPO_ROOT/main/language/baseline_weights/Qwen2.5-VL-3B-Instruct}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-$REPO_ROOT/main/language/baseline_outputs/${MODEL_NAME}_${MARCH}_w${W_BITS}}"
CALIB_JSON="${CALIB_JSON:-leap_llm/apis/calibration/calibration_data/mmstar/conversation.json}"
CONDA_ENV="${CONDA_ENV:-oellm}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/main/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/baseline_${MODEL_NAME}.log}"

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo "conda.sh not found: $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$LEAP_LLM_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ -f "$CALIB_JSON" ]] || { echo "calib json missing (relative to $LEAP_LLM_SRC): $CALIB_JSON"; exit 1; }
command -v oellm_build >/dev/null || { echo "oellm_build not on PATH in env $CONDA_ENV"; exit 1; }

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_MODEL_PATH")"

if pgrep -f "oellm_build.*--model_name $MODEL_NAME" >/dev/null; then
  echo "an oellm_build for $MODEL_NAME is already running:"
  pgrep -af "oellm_build.*--model_name $MODEL_NAME"
  exit 2
fi

echo "cwd:           $(pwd)"
echo "conda env:     $CONDA_ENV"
echo "input:         $INPUT_MODEL_PATH"
echo "output:        $OUTPUT_MODEL_PATH"
echo "calib:         $CALIB_JSON"
echo "log:           $LOG_FILE"
echo

nohup oellm_build \
  --model_name "$MODEL_NAME" \
  --march "$MARCH" \
  --input_model_path "$INPUT_MODEL_PATH" \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --w_bits "$W_BITS" \
  --chunk_size "$CHUNK_SIZE" \
  --cache_len "$CACHE_LEN" \
  --image_width "$IMAGE_WIDTH" \
  --image_height "$IMAGE_HEIGHT" \
  --calib_json_path "$CALIB_JSON" \
  --device "$DEVICE" \
  --vit_core_num "$VIT_CORE_NUM" \
  --prefill_core_num "$PREFILL_CORE_NUM" \
  --decode_core_num "$DECODE_CORE_NUM" \
  --jobs "$JOBS" \
  >"$LOG_FILE" 2>&1 &

PID=$!
disown

echo "PID=$PID"
echo "tail -f $LOG_FILE   # to follow"
echo "kill $PID           # to stop"

sleep 8
if kill -0 "$PID" 2>/dev/null; then
  echo
  echo "process alive ✓"
else
  echo
  echo "PROCESS DEAD after 8s — check $LOG_FILE"
  exit 3
fi
