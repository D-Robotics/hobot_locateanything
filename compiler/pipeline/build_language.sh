#!/usr/bin/env bash
# Compile LocateAnything Language through the external OE_LLM/HBDK environment.
# Uses the LocateAnythingLanguageApi registered under model_name locateanything-lm-3b
# in the LA subsystem of the OELLM ecosystem. Run through compiler/quantize.py.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_ADAPTER=${BUILD_ADAPTER:?run this stage through compiler/quantize.py}
COMPILER_SRC="${COMPILER_SRC:-$REPO_ROOT/compiler}"

MODEL_NAME="${MODEL_NAME:-locateanything-lm-3b}"
MARCH="${MARCH:?set by compiler/quantize.py}"
W_BITS="${W_BITS:?set by compiler/quantize.py}"
LM_HEAD_W_BITS="${LM_HEAD_W_BITS:?set by compiler/quantize.py}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-bpu}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"
SAMPLING_REPETITION_PENALTY="${SAMPLING_REPETITION_PENALTY:-1.1}"
CHUNK_SIZE="${CHUNK_SIZE:?set by compiler/quantize.py}"
CACHE_LEN="${CACHE_LEN:?set by compiler/quantize.py}"
DECODE_SEQ_LEN="${DECODE_SEQ_LEN:?set by compiler/quantize.py}"
DEVICE="${DEVICE:?set by compiler/quantize.py}"
PREFILL_CORE_NUM="${PREFILL_CORE_NUM:?set by compiler/quantize.py}"
DECODE_CORE_NUM="${DECODE_CORE_NUM:?set by compiler/quantize.py}"
AR_CORE_NUM="${AR_CORE_NUM:?set by compiler/quantize.py}"
JOBS="${JOBS:?set by compiler/quantize.py}"
HIDDEN_ROTATION_PATH="${HIDDEN_ROTATION_PATH:-}"
DISABLE_HIDDEN_ROTATION="${DISABLE_HIDDEN_ROTATION:-0}"
BUILD_TARGET="${BUILD_TARGET:?set by compiler/quantize.py}"
WAIT="${WAIT:?set by compiler/quantize.py}"
DETACH="${DETACH:?set by compiler/quantize.py}"
RESUME="${RESUME:?set by compiler/quantize.py}"

INPUT_MODEL_PATH="${INPUT_MODEL_PATH:?set by compiler/quantize.py}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:?set by compiler/quantize.py}"
LANGUAGE_HBM_NAME="${LANGUAGE_HBM_NAME:?set by compiler/quantize.py}"
EMBEDDING_NAME="${EMBEDDING_NAME:?set by compiler/quantize.py}"
CALIBRATION_SCALE_MANIFEST="${CALIBRATION_SCALE_MANIFEST:-}"

LOG_DIR="${LOG_DIR:?set by compiler/quantize.py}"
LOG_FILE="${LOG_FILE:?set by compiler/quantize.py}"

case "$BUILD_TARGET" in
  bc) EXPORT_ONLY=1 ;;
  hbm) EXPORT_ONLY=0 ;;
  *) echo "BUILD_TARGET must be bc or hbm; got $BUILD_TARGET"; exit 1 ;;
esac
[[ "$WAIT" == "0" || "$WAIT" == "1" ]] || { echo "WAIT must be 0 or 1"; exit 1; }
[[ "$DETACH" == "0" || "$DETACH" == "1" ]] || { echo "DETACH must be 0 or 1"; exit 1; }
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || { echo "RESUME must be 0 or 1"; exit 1; }
PID_PATH="${PID_PATH:-$LOG_DIR/language_${BUILD_TARGET}.pid}"
EXIT_PATH="${EXIT_PATH:-$LOG_DIR/language_${BUILD_TARGET}.exit.txt}"
LAUNCH_LOG="${LAUNCH_LOG:-$LOG_DIR/language_${BUILD_TARGET}.launcher.log}"

mkdir -p "$LOG_DIR"
if [[ "$DETACH" == "1" || "$WAIT" == "0" ]]; then
  setsid nohup env DETACH=0 WAIT=1 RESUME="$RESUME" BUILD_TARGET="$BUILD_TARGET" \
    EXPORT_ONLY="$EXPORT_ONLY" bash "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[build:language] detached_pid=$child_pid"
  echo "[build:language] launcher_log=$LAUNCH_LOG"
  exit 0
fi

cd "$COMPILER_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ "$W_BITS" == "4" || "$W_BITS" == "8" ]] || {
  echo "W_BITS must be 4 or 8; got $W_BITS"
  exit 1
}
[[ "$LM_HEAD_W_BITS" == "4" || "$LM_HEAD_W_BITS" == "8" ]] || {
  echo "LM_HEAD_W_BITS must be 4 or 8; got $LM_HEAD_W_BITS"
  exit 1
}
[[ "$DECODE_SEQ_LEN" == "6" ]] || {
  echo "the default fused graph catalog requires DECODE_SEQ_LEN=6"
  exit 1
}
[[ "$PREFILL_CORE_NUM" == "1" || "$PREFILL_CORE_NUM" == "2" || "$PREFILL_CORE_NUM" == "4" ]] || { echo "PREFILL_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ "$DECODE_CORE_NUM" == "1" || "$DECODE_CORE_NUM" == "2" || "$DECODE_CORE_NUM" == "4" ]] || { echo "DECODE_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ "$AR_CORE_NUM" == "1" || "$AR_CORE_NUM" == "2" || "$AR_CORE_NUM" == "4" ]] || { echo "AR_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ -n "$CALIBRATION_SCALE_MANIFEST" && -f "$CALIBRATION_SCALE_MANIFEST" ]] || { echo "activation scale manifest missing; set CALIBRATION_SCALE_MANIFEST"; exit 1; }
[[ "$DISABLE_HIDDEN_ROTATION" == "0" || "$DISABLE_HIDDEN_ROTATION" == "1" ]] || { echo "DISABLE_HIDDEN_ROTATION must be 0 or 1"; exit 1; }

mkdir -p "$(dirname "$OUTPUT_MODEL_PATH")"

echo "cwd:           $(pwd)"
echo "python:        $PYTHON_BIN"
echo "input:         $INPUT_MODEL_PATH"
echo "output:        $OUTPUT_MODEL_PATH"
echo "scale:         $CALIBRATION_SCALE_MANIFEST"
echo "weights:       decoder W$W_BITS; lm_head W$LM_HEAD_W_BITS"
echo "Language graphs: fused_decode (13 graphs)"
echo "cores:         prefill=$PREFILL_CORE_NUM pbd_q6=$DECODE_CORE_NUM ar_q1=$AR_CORE_NUM"
echo "target:        $BUILD_TARGET wait=$WAIT detach=$DETACH resume=$RESUME"
echo "log:           $LOG_FILE"
echo

EXTRA_ARGS=()
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  EXTRA_ARGS+=(--hidden_rotation_path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  EXTRA_ARGS+=(--disable_hidden_rotation)
fi
HBM_PATH="$OUTPUT_MODEL_PATH/$LANGUAGE_HBM_NAME"
EMBEDDING_PATH="$OUTPUT_MODEL_PATH/$EMBEDDING_NAME"
EXPECTED_EMBEDDING_BYTES=$((152681 * 2048 * 2))
STAGE_ARGS=(
  --bc_dir "$OUTPUT_MODEL_PATH"
  --output_dir "$OUTPUT_MODEL_PATH"
  --hbm_path "$HBM_PATH"
  --embedding_path "$EMBEDDING_PATH"
  --expected_embedding_bytes "$EXPECTED_EMBEDDING_BYTES"
  --march "$MARCH"
  --prefill_core_num "$PREFILL_CORE_NUM"
  --decode_core_num "$DECODE_CORE_NUM"
  --ar_core_nums "$AR_CORE_NUM"
  --jobs "$JOBS"
  --chunk-size "$CHUNK_SIZE"
  --cache-len "$CACHE_LEN"
  --language-w-bits "$W_BITS"
  --lm-head-w-bits "$LM_HEAD_W_BITS"
  --sampling-backend "$SAMPLING_BACKEND"
  --sampling-temperature "$SAMPLING_TEMPERATURE"
  --sampling-top-p "$SAMPLING_TOP_P"
  --sampling-repetition-penalty "$SAMPLING_REPETITION_PENALTY"
)
mapfile -t LANGUAGE_GRAPHS < <(
  "$PYTHON_BIN" -m model.graphs
)
[[ "${#LANGUAGE_GRAPHS[@]}" -gt 0 ]] || {
  echo "Language graph catalog is empty"
  exit 1
}
validate_bc() {
  env PYTHONPATH="$COMPILER_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
    "$REPO_ROOT/compiler/pipeline/language_build.py" \
    "${STAGE_ARGS[@]}" --check_only
}

export_bc() {
  echo "[build:language] exporting the fused_decode Language BC graph family"
  env PYTHONUNBUFFERED=1 PYTHONPATH="$COMPILER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$BUILD_ADAPTER" \
    --model_name "$MODEL_NAME" \
    --march "$MARCH" \
    --input_model_path "$INPUT_MODEL_PATH" \
    --output_model_path "$OUTPUT_MODEL_PATH" \
    --w_bits "$W_BITS" \
    --lm_head_w_bits "$LM_HEAD_W_BITS" \
    --sampling_backend "$SAMPLING_BACKEND" \
    --sampling_temperature "$SAMPLING_TEMPERATURE" \
    --sampling_top_p "$SAMPLING_TOP_P" \
    --sampling_repetition_penalty "$SAMPLING_REPETITION_PENALTY" \
    --chunk_size "$CHUNK_SIZE" \
    --cache_len "$CACHE_LEN" \
    --decode_seq_len "$DECODE_SEQ_LEN" \
    --calibration_scale_manifest "$CALIBRATION_SCALE_MANIFEST" \
    --device "$DEVICE" \
    --prefill_core_num "$PREFILL_CORE_NUM" \
    --decode_core_num "$DECODE_CORE_NUM" \
    --ar_core_num "$AR_CORE_NUM" \
    --jobs "$JOBS" \
    --export_only \
    "${EXTRA_ARGS[@]}"
}

run_build() (
  set -e
  if [[ "$BUILD_TARGET" == "bc" ]]; then
    if [[ "$RESUME" == "1" ]] && validate_bc; then
      echo "[RESUME] Language BC graphs already complete: $OUTPUT_MODEL_PATH"
      return 0
    fi
    export_bc
    validate_bc
    return 0
  fi

  if validate_bc; then
    echo "[REUSE] Language HBM build will consume existing ${#LANGUAGE_GRAPHS[@]}-graph BC family"
  else
    echo "[build:language] complete BC graph family not found; exporting it first"
    export_bc
    validate_bc
  fi

  RESUME_ARGS=()
  [[ "$RESUME" == "1" ]] && RESUME_ARGS+=(--resume)
  env PYTHONUNBUFFERED=1 PYTHONPATH="$COMPILER_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
    "$REPO_ROOT/compiler/pipeline/language_build.py" \
    "${STAGE_ARGS[@]}" "${RESUME_ARGS[@]}"
)

printf 'status=running\ntarget=%s\nstarted_at=%s\n' \
  "$BUILD_TARGET" "$(date --iso-8601=seconds)" > "$EXIT_PATH"
set +e
run_build 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\ntarget=%s\ncompleted_at=%s\n' \
  "$status" "$BUILD_TARGET" "$(date --iso-8601=seconds)" > "$EXIT_PATH"
if [[ "$status" -ne 0 ]]; then
  echo "[build:language] failed exit_code=$status log=$LOG_FILE"
  exit "$status"
fi
echo "[build:language] completed target=$BUILD_TARGET log=$LOG_FILE"
