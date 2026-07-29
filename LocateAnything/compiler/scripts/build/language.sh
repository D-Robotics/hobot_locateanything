#!/usr/bin/env bash
# Compile LocateAnything Language with Decoder W8 and lm_head W8 via oellm_build.
# Uses the LocateAnythingLanguageApi registered under model_name locateanything-lm-3b
# in the LA subsystem of the OELLM ecosystem. Environment variables override defaults.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}"
LEAP_LLM_SRC="${LEAP_LLM_SRC:-$REPO_ROOT/compiler}"

MODEL_NAME="${MODEL_NAME:-locateanything-lm-3b}"
MARCH="${MARCH:-nash-p}"
W_BITS="${W_BITS:-8}"
LM_HEAD_W_BITS="${LM_HEAD_W_BITS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
CACHE_LEN="${CACHE_LEN:-4096}"
DECODE_SEQ_LEN="${DECODE_SEQ_LEN:-6}"   # PBD q_len=6
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-}"
DEVICE="${DEVICE:-cuda:0}"
PREFILL_CORE_NUM="${PREFILL_CORE_NUM:-4}"
DECODE_CORE_NUM="${DECODE_CORE_NUM:-4}"
AR_CORE_NUM="${AR_CORE_NUM:-$DECODE_CORE_NUM}"
JOBS="${JOBS:-16}"
HIDDEN_ROTATION_PATH="${HIDDEN_ROTATION_PATH:-}"
DISABLE_HIDDEN_ROTATION="${DISABLE_HIDDEN_ROTATION:-0}"
BUILD_TARGET="${BUILD_TARGET:-}"
EXPORT_ONLY_INPUT="${EXPORT_ONLY:-}"
FUSED_PBD_PROFILES="${FUSED_PBD_PROFILES:-1}"
WAIT="${WAIT:-1}"
DETACH="${DETACH:-0}"

BUILD_ID="${BUILD_ID:-release}"
INPUT_MODEL_PATH="${INPUT_MODEL_PATH:-$REPO_ROOT/workspace/models/LocateAnything-3B}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-$REPO_ROOT/workspace/builds/$BUILD_ID/language}"
CALIB_JSON="${CALIB_JSON:?set CALIB_JSON to the selected calibration manifest}"
GENERATED_JSON="${GENERATED_JSON:?set GENERATED_JSON to the prepared calibration manifest}"
CALIBRATION_SCALE_MANIFEST="${CALIBRATION_SCALE_MANIFEST:-}"
CALIBRATION_COVERAGE_JSON="${CALIBRATION_COVERAGE_JSON:-}"
CONDA_ENV="${CONDA_ENV:-oellm_clean}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/workspace/builds/$BUILD_ID/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/language.log}"

if [[ -z "$BUILD_TARGET" ]]; then
  [[ "$EXPORT_ONLY_INPUT" == "1" ]] && BUILD_TARGET=bc || BUILD_TARGET=hbm
fi
case "$BUILD_TARGET" in
  bc) EXPECTED_EXPORT_ONLY=1 ;;
  hbm) EXPECTED_EXPORT_ONLY=0 ;;
  *) echo "BUILD_TARGET must be bc or hbm; got $BUILD_TARGET"; exit 1 ;;
esac
if [[ -n "$EXPORT_ONLY_INPUT" && "$EXPORT_ONLY_INPUT" != "$EXPECTED_EXPORT_ONLY" ]]; then
  echo "EXPORT_ONLY=$EXPORT_ONLY_INPUT conflicts with BUILD_TARGET=$BUILD_TARGET"
  exit 1
fi
EXPORT_ONLY="$EXPECTED_EXPORT_ONLY"
[[ "$WAIT" == "0" || "$WAIT" == "1" ]] || { echo "WAIT must be 0 or 1"; exit 1; }
[[ "$DETACH" == "0" || "$DETACH" == "1" ]] || { echo "DETACH must be 0 or 1"; exit 1; }
PID_PATH="${PID_PATH:-$LOG_DIR/language_${BUILD_TARGET}.pid}"
EXIT_PATH="${EXIT_PATH:-$LOG_DIR/language_${BUILD_TARGET}.exit.txt}"
LAUNCH_LOG="${LAUNCH_LOG:-$LOG_DIR/language_${BUILD_TARGET}.launcher.log}"

mkdir -p "$LOG_DIR"
if [[ "$DETACH" == "1" || "$WAIT" == "0" ]]; then
  setsid nohup env DETACH=0 WAIT=1 BUILD_TARGET="$BUILD_TARGET" \
    EXPORT_ONLY="$EXPORT_ONLY" "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[build:language] detached_pid=$child_pid"
  echo "[build:language] launcher_log=$LAUNCH_LOG"
  exit 0
fi

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo "conda.sh not found: $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$LEAP_LLM_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ -f "$CALIB_JSON" ]] || { echo "calib json missing: $CALIB_JSON"; exit 1; }
[[ -f "$GENERATED_JSON" ]] || { echo "D3 generated json missing: $GENERATED_JSON"; exit 1; }
[[ "$W_BITS" == "4" || "$W_BITS" == "8" ]] || { echo "W_BITS must be 4 or 8"; exit 1; }
[[ "$LM_HEAD_W_BITS" == "4" || "$LM_HEAD_W_BITS" == "8" ]] || { echo "LM_HEAD_W_BITS must be 4 or 8"; exit 1; }
[[ "$AR_CORE_NUM" == "1" || "$AR_CORE_NUM" == "2" || "$AR_CORE_NUM" == "4" ]] || { echo "AR_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ -n "$CALIBRATION_SCALE_MANIFEST" && -f "$CALIBRATION_SCALE_MANIFEST" ]] || { echo "activation scale manifest missing; set CALIBRATION_SCALE_MANIFEST"; exit 1; }
[[ -n "$CALIBRATION_COVERAGE_JSON" && -f "$CALIBRATION_COVERAGE_JSON" ]] || { echo "calibration graph coverage missing; set CALIBRATION_COVERAGE_JSON"; exit 1; }
[[ "$DISABLE_HIDDEN_ROTATION" == "0" || "$DISABLE_HIDDEN_ROTATION" == "1" ]] || { echo "DISABLE_HIDDEN_ROTATION must be 0 or 1"; exit 1; }
[[ "$FUSED_PBD_PROFILES" == "0" || "$FUSED_PBD_PROFILES" == "1" ]] || { echo "FUSED_PBD_PROFILES must be 0 or 1"; exit 1; }
command -v oellm_build >/dev/null || { echo "oellm_build not on PATH in env $CONDA_ENV"; exit 1; }

VALIDATION_ARGS=()
if [[ -n "$EXPECTED_SAMPLES" ]]; then
  VALIDATION_ARGS+=(--expected-samples "$EXPECTED_SAMPLES")
fi
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  VALIDATION_ARGS+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  VALIDATION_ARGS+=(--disable-hidden-rotation)
fi

python "$REPO_ROOT/compiler/scripts/validate/deployment.py" \
  --component language \
  --selected-jsonl "$CALIB_JSON" \
  --generated-jsonl "$GENERATED_JSON" \
  --scale-manifest "$CALIBRATION_SCALE_MANIFEST" \
  --coverage-json "$CALIBRATION_COVERAGE_JSON" \
  --image-width 672 --image-height 672 \
  --chunk-size "$CHUNK_SIZE" --cache-len "$CACHE_LEN" --decode-seq-len "$DECODE_SEQ_LEN" \
  --lm-head-w-bits "$LM_HEAD_W_BITS" \
  "${VALIDATION_ARGS[@]}"

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
echo "generated:     $GENERATED_JSON"
echo "scale:         $CALIBRATION_SCALE_MANIFEST"
echo "coverage:      $CALIBRATION_COVERAGE_JSON"
echo "weights:       decoder W$W_BITS; lm_head W$LM_HEAD_W_BITS"
echo "graph profile: fused_pbd=$FUSED_PBD_PROFILES"
echo "cores:         prefill=$PREFILL_CORE_NUM pbd_q6=$DECODE_CORE_NUM ar_q1=$AR_CORE_NUM"
echo "target:        $BUILD_TARGET wait=$WAIT detach=$DETACH"
echo "log:           $LOG_FILE"
echo

EXTRA_ARGS=()
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  EXTRA_ARGS+=(--hidden_rotation_path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  EXTRA_ARGS+=(--disable_hidden_rotation)
fi
if [[ "$EXPORT_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--export_only)
fi
if [[ "$FUSED_PBD_PROFILES" == "1" ]]; then
  EXTRA_ARGS+=(--fused_pbd_profiles)
fi

printf 'status=running\ntarget=%s\nstarted_at=%s\n' \
  "$BUILD_TARGET" "$(date --iso-8601=seconds)" > "$EXIT_PATH"
set +e
env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" oellm_build \
  --model_name "$MODEL_NAME" \
  --march "$MARCH" \
  --input_model_path "$INPUT_MODEL_PATH" \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --w_bits "$W_BITS" \
  --lm_head_w_bits "$LM_HEAD_W_BITS" \
  --chunk_size "$CHUNK_SIZE" \
  --cache_len "$CACHE_LEN" \
  --decode_seq_len "$DECODE_SEQ_LEN" \
  --calibration_scale_manifest "$CALIBRATION_SCALE_MANIFEST" \
  --device "$DEVICE" \
  --prefill_core_num "$PREFILL_CORE_NUM" \
  --decode_core_num "$DECODE_CORE_NUM" \
  --ar_core_num "$AR_CORE_NUM" \
  --jobs "$JOBS" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\ntarget=%s\ncompleted_at=%s\n' \
  "$status" "$BUILD_TARGET" "$(date --iso-8601=seconds)" > "$EXIT_PATH"
if [[ "$status" -ne 0 ]]; then
  echo "[build:language] failed exit_code=$status log=$LOG_FILE"
  exit "$status"
fi
echo "[build:language] completed target=$BUILD_TARGET log=$LOG_FILE"
