#!/usr/bin/env bash
# Compile LocateAnything vision tower (MoonViT-SO-400M) HBM (nash-p, W8) via oellm_build.
# Uses the LocateAnythingVisionApi registered under model_name locateanything-vit-3b
# in the LA subsystem of the OELLM ecosystem.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEAP_LLM_SRC="${LEAP_LLM_SRC:-$REPO_ROOT/compiler}"

MODEL_NAME="${MODEL_NAME:-locateanything-vit-3b}"
MARCH="${MARCH:?set by compiler/quantize.py}"
W_BITS="${W_BITS:?set by compiler/quantize.py}"
# The release profile is a fixed 672x672 canvas: 2304 patches -> 576 tokens.
# Source images are letterboxed by the calibration/runtime preprocessor.
IMAGE_WIDTH="${IMAGE_WIDTH:?set by compiler/quantize.py}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:?set by compiler/quantize.py}"
DEVICE="${DEVICE:?set by compiler/quantize.py}"
VIT_CORE_NUM="${VIT_CORE_NUM:?set by compiler/quantize.py}"
JOBS="${JOBS:?set by compiler/quantize.py}"
CHUNK_SIZE="${CHUNK_SIZE:?set by compiler/quantize.py}"
CACHE_LEN="${CACHE_LEN:?set by compiler/quantize.py}"
HIDDEN_ROTATION_PATH="${HIDDEN_ROTATION_PATH:-}"
DISABLE_HIDDEN_ROTATION="${DISABLE_HIDDEN_ROTATION:-0}"
BUILD_TARGET="${BUILD_TARGET:?set by compiler/quantize.py}"
WAIT="${WAIT:?set by compiler/quantize.py}"
DETACH="${DETACH:?set by compiler/quantize.py}"
RESUME="${RESUME:?set by compiler/quantize.py}"

INPUT_MODEL_PATH="${INPUT_MODEL_PATH:?set by compiler/quantize.py}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:?set by compiler/quantize.py}"
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
PID_PATH="${PID_PATH:-$LOG_DIR/vision_${BUILD_TARGET}.pid}"
EXIT_PATH="${EXIT_PATH:-$LOG_DIR/vision_${BUILD_TARGET}.exit.txt}"
LAUNCH_LOG="${LAUNCH_LOG:-$LOG_DIR/vision_${BUILD_TARGET}.launcher.log}"

mkdir -p "$LOG_DIR"
if [[ "$DETACH" == "1" || "$WAIT" == "0" ]]; then
  setsid nohup env DETACH=0 WAIT=1 RESUME="$RESUME" BUILD_TARGET="$BUILD_TARGET" \
    EXPORT_ONLY="$EXPORT_ONLY" bash "$0" >"$LAUNCH_LOG" 2>&1 </dev/null &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$PID_PATH"
  echo "[build:vision] detached_pid=$child_pid"
  echo "[build:vision] launcher_log=$LAUNCH_LOG"
  exit 0
fi

cd "$LEAP_LLM_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ -n "$CALIBRATION_SCALE_MANIFEST" && -f "$CALIBRATION_SCALE_MANIFEST" ]] || { echo "activation scale manifest missing; set CALIBRATION_SCALE_MANIFEST"; exit 1; }
[[ "$DISABLE_HIDDEN_ROTATION" == "0" || "$DISABLE_HIDDEN_ROTATION" == "1" ]] || { echo "DISABLE_HIDDEN_ROTATION must be 0 or 1"; exit 1; }

[[ "$W_BITS" == "8" ]] || {
  echo "LocateAnything Vision currently supports W8 only; got W_BITS=$W_BITS"
  exit 1
}

mkdir -p "$(dirname "$OUTPUT_MODEL_PATH")"

echo "cwd:           $(pwd)"
echo "python:        $PYTHON_BIN"
echo "input:         $INPUT_MODEL_PATH"
echo "output:        $OUTPUT_MODEL_PATH"
echo "scale:         $CALIBRATION_SCALE_MANIFEST"
echo "image_wh:      ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
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
MODEL_BASENAME=$(basename "$INPUT_MODEL_PATH")
HBM_PATH="$OUTPUT_MODEL_PATH/${MODEL_BASENAME}_vision_${IMAGE_WIDTH}x${IMAGE_HEIGHT}_w${W_BITS}_${MARCH}_corenum_${VIT_CORE_NUM}.hbm"
BC_PATH="${HBM_PATH%.hbm}.visual.bc"
STAGE_ARGS=(
  --bc_path "$BC_PATH"
  --hbm_path "$HBM_PATH"
  --march "$MARCH"
  --core_num "$VIT_CORE_NUM"
  --jobs "$JOBS"
)
validate_bc() {
  env PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
    "$REPO_ROOT/compiler/scripts/build/vision_stages.py" \
    "${STAGE_ARGS[@]}" --check_only
}

export_bc() {
  echo "[build:vision] exporting visual BC"
  env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" oellm_build \
    --model_name "$MODEL_NAME" \
    --march "$MARCH" \
    --input_model_path "$INPUT_MODEL_PATH" \
    --output_model_path "$OUTPUT_MODEL_PATH" \
    --w_bits "$W_BITS" \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    --calibration_scale_manifest "$CALIBRATION_SCALE_MANIFEST" \
    --device "$DEVICE" \
    --vit_core_num "$VIT_CORE_NUM" \
    --jobs "$JOBS" \
    --export_only \
    "${EXTRA_ARGS[@]}"
}

run_build() (
  set -e
  if [[ "$BUILD_TARGET" == "bc" ]]; then
    if [[ "$RESUME" == "1" ]] && validate_bc; then
      echo "[RESUME] Vision BC graph already complete: $BC_PATH"
      return 0
    fi
    export_bc
    validate_bc
    return 0
  fi

  if validate_bc; then
    echo "[REUSE] Vision HBM build will consume existing BC: $BC_PATH"
  else
    echo "[build:vision] complete BC graph not found; exporting it first"
    export_bc
    validate_bc
  fi

  RESUME_ARGS=()
  [[ "$RESUME" == "1" ]] && RESUME_ARGS+=(--resume)
  env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" \
    "$REPO_ROOT/compiler/scripts/build/vision_stages.py" \
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
  echo "[build:vision] failed exit_code=$status log=$LOG_FILE"
  exit "$status"
fi
echo "[build:vision] completed target=$BUILD_TARGET log=$LOG_FILE"
