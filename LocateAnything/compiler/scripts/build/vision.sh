#!/usr/bin/env bash
# Compile LocateAnything vision tower (MoonViT-SO-400M) HBM (nash-p, W8) via oellm_build.
# Uses the LocateAnythingVisionApi registered under model_name locateanything-vit-3b
# in the LA subsystem of the OELLM ecosystem.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}"
LEAP_LLM_SRC="${LEAP_LLM_SRC:-$REPO_ROOT/compiler}"

MODEL_NAME="${MODEL_NAME:-locateanything-vit-3b}"
MARCH="${MARCH:-nash-p}"
W_BITS="${W_BITS:-8}"
# The release profile is a fixed 672x672 canvas: 2304 patches -> 576 tokens.
# Source images are letterboxed by the calibration/runtime preprocessor.
IMAGE_WIDTH="${IMAGE_WIDTH:-672}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-672}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1200}"
EXPECTED_SELECTED_MANIFEST_SHA256="${EXPECTED_SELECTED_MANIFEST_SHA256:-22cc670b2b600b2e5ea3dfbc3d169c07540ef108a0e2a135d8b20f949ed62b03}"
DEVICE="${DEVICE:-cuda:0}"
VIT_CORE_NUM="${VIT_CORE_NUM:-4}"
JOBS="${JOBS:-16}"
HIDDEN_ROTATION_PATH="${HIDDEN_ROTATION_PATH:-}"
DISABLE_HIDDEN_ROTATION="${DISABLE_HIDDEN_ROTATION:-0}"
BUILD_TARGET="${BUILD_TARGET:-}"
EXPORT_ONLY_INPUT="${EXPORT_ONLY:-}"
WAIT="${WAIT:-1}"
DETACH="${DETACH:-0}"
RESUME="${RESUME:-0}"

BUILD_ID="${BUILD_ID:-release}"
INPUT_MODEL_PATH="${INPUT_MODEL_PATH:-$REPO_ROOT/workspace/models/LocateAnything-3B}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-$REPO_ROOT/workspace/builds/$BUILD_ID/vision}"
CALIB_JSON="${CALIB_JSON:?set CALIB_JSON to the selected calibration manifest}"
GENERATED_JSON="${GENERATED_JSON:?set GENERATED_JSON to the prepared calibration manifest}"
CALIBRATION_SCALE_MANIFEST="${CALIBRATION_SCALE_MANIFEST:-}"
CALIBRATION_COVERAGE_JSON="${CALIBRATION_COVERAGE_JSON:-}"
CONDA_ENV="${CONDA_ENV:-oellm_clean}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/workspace/builds/$BUILD_ID/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vision.log}"

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

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo "conda.sh not found: $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$LEAP_LLM_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ -f "$CALIB_JSON" ]] || { echo "calib json missing (relative to $LEAP_LLM_SRC): $CALIB_JSON"; exit 1; }
[[ -f "$GENERATED_JSON" ]] || { echo "prepared calibration manifest missing: $GENERATED_JSON"; exit 1; }
[[ -n "$CALIBRATION_SCALE_MANIFEST" && -f "$CALIBRATION_SCALE_MANIFEST" ]] || { echo "activation scale manifest missing; set CALIBRATION_SCALE_MANIFEST"; exit 1; }
[[ -n "$CALIBRATION_COVERAGE_JSON" && -f "$CALIBRATION_COVERAGE_JSON" ]] || { echo "calibration graph coverage missing; set CALIBRATION_COVERAGE_JSON"; exit 1; }
[[ "$DISABLE_HIDDEN_ROTATION" == "0" || "$DISABLE_HIDDEN_ROTATION" == "1" ]] || { echo "DISABLE_HIDDEN_ROTATION must be 0 or 1"; exit 1; }
command -v oellm_build >/dev/null || { echo "oellm_build not on PATH in env $CONDA_ENV"; exit 1; }
[[ "$W_BITS" == "8" ]] || {
  echo "LocateAnything Vision currently supports W8 only; got W_BITS=$W_BITS"
  exit 1
}

VALIDATION_ARGS=()
if [[ -n "$EXPECTED_SAMPLES" ]]; then
  VALIDATION_ARGS+=(--expected-samples "$EXPECTED_SAMPLES")
fi
if [[ -n "$EXPECTED_SELECTED_MANIFEST_SHA256" ]]; then
  VALIDATION_ARGS+=(--expected-selected-sha256 "$EXPECTED_SELECTED_MANIFEST_SHA256")
fi
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  VALIDATION_ARGS+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  VALIDATION_ARGS+=(--disable-hidden-rotation)
fi

python "$REPO_ROOT/compiler/scripts/validate/deployment.py" \
  --component vision \
  --selected-jsonl "$CALIB_JSON" \
  --generated-jsonl "$GENERATED_JSON" \
  --scale-manifest "$CALIBRATION_SCALE_MANIFEST" \
  --coverage-json "$CALIBRATION_COVERAGE_JSON" \
  --image-width "$IMAGE_WIDTH" --image-height "$IMAGE_HEIGHT" \
  --chunk-size 1024 --cache-len 4096 --decode-seq-len 6 \
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
echo "calib_json:    $CALIB_JSON"
echo "generated:     $GENERATED_JSON"
echo "scale:         $CALIBRATION_SCALE_MANIFEST"
echo "coverage:      $CALIBRATION_COVERAGE_JSON"
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
BC_MANIFEST="${HBM_PATH%.hbm}.bc_manifest.json"
STAGE_ARGS=(
  --bc_path "$BC_PATH"
  --hbm_path "$HBM_PATH"
  --march "$MARCH"
  --core_num "$VIT_CORE_NUM"
  --jobs "$JOBS"
)
PROVENANCE_ARGS=(
  --manifest "$BC_MANIFEST"
  --component vision
  --model_path "$INPUT_MODEL_PATH"
  --scale_manifest "$CALIBRATION_SCALE_MANIFEST"
  --field "image_width=$IMAGE_WIDTH"
  --field "image_height=$IMAGE_HEIGHT"
  --field "w_bits=$W_BITS"
  --field "march=$MARCH"
  --artifact "visual=$BC_PATH"
)
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  PROVENANCE_ARGS+=(--hidden_rotation_path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  PROVENANCE_ARGS+=(--disable_hidden_rotation)
fi

validate_bc() {
  env PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" python \
    "$REPO_ROOT/compiler/scripts/build/vision_stages.py" \
    "${STAGE_ARGS[@]}" --check_only
}

check_bc() {
  python "$REPO_ROOT/compiler/scripts/build/artifact_manifest.py" \
    check "${PROVENANCE_ARGS[@]}" && validate_bc
}

record_bc() {
  validate_bc
  python "$REPO_ROOT/compiler/scripts/build/artifact_manifest.py" \
    write "${PROVENANCE_ARGS[@]}"
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
    if [[ "$RESUME" == "1" ]] && check_bc; then
      echo "[RESUME] Vision BC contract already complete: $BC_PATH"
      return 0
    fi
    export_bc
    record_bc
    return 0
  fi

  if check_bc; then
    echo "[REUSE] Vision HBM build will consume existing BC: $BC_PATH"
  else
    echo "[build:vision] compatible BC not found; exporting it first"
    export_bc
    record_bc
  fi

  RESUME_ARGS=()
  [[ "$RESUME" == "1" ]] && RESUME_ARGS+=(--resume)
  env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" python \
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
