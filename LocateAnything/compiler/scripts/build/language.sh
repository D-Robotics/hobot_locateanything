#!/usr/bin/env bash
# Compile LocateAnything Language with Decoder W8 and lm_head W8 via oellm_build.
# Uses the LocateAnythingLanguageApi registered under model_name locateanything-lm-3b
# in the LA subsystem of the OELLM ecosystem. Environment variables override defaults.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LEAP_LLM_SRC="${LEAP_LLM_SRC:-$REPO_ROOT/compiler}"

MODEL_NAME="${MODEL_NAME:-locateanything-lm-3b}"
MARCH="${MARCH:-nash-p}"
W_BITS="${W_BITS:-8}"
LM_HEAD_W_BITS="${LM_HEAD_W_BITS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
CACHE_LEN="${CACHE_LEN:-4096}"
DECODE_SEQ_LEN="${DECODE_SEQ_LEN:-6}"   # PBD q_len=6
EXPECTED_SAMPLES=1200
EXPECTED_SELECTED_MANIFEST_SHA256="22cc670b2b600b2e5ea3dfbc3d169c07540ef108a0e2a135d8b20f949ed62b03"
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
RESUME="${RESUME:-0}"

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
ENVIRONMENT_PATH="${ENVIRONMENT_PATH:-$LOG_DIR/language_environment.json}"
ENVIRONMENT_SCRIPT="${ENVIRONMENT_SCRIPT:-$REPO_ROOT/compiler/scripts/common/environment.py}"

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

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[[ -f "$CONDA_SH" ]] || { echo "conda.sh not found: $CONDA_SH"; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$LEAP_LLM_SRC"

[[ -d "$INPUT_MODEL_PATH" ]] || { echo "input model missing: $INPUT_MODEL_PATH"; exit 1; }
[[ -f "$CALIB_JSON" ]] || { echo "calib json missing: $CALIB_JSON"; exit 1; }
[[ -f "$GENERATED_JSON" ]] || { echo "prepared calibration manifest missing: $GENERATED_JSON"; exit 1; }
[[ "$W_BITS" == "8" ]] || { echo "LocateAnything Language release requires W_BITS=8"; exit 1; }
[[ "$LM_HEAD_W_BITS" == "8" ]] || { echo "LocateAnything Language release requires LM_HEAD_W_BITS=8"; exit 1; }
[[ "$CHUNK_SIZE" == "1024" && "$CACHE_LEN" == "4096" && "$DECODE_SEQ_LEN" == "6" ]] || {
  echo "LocateAnything Language release requires CHUNK_SIZE=1024 CACHE_LEN=4096 DECODE_SEQ_LEN=6"
  exit 1
}
[[ "$FUSED_PBD_PROFILES" == "1" ]] || { echo "LocateAnything Language release requires FUSED_PBD_PROFILES=1"; exit 1; }
[[ "$PREFILL_CORE_NUM" == "1" || "$PREFILL_CORE_NUM" == "2" || "$PREFILL_CORE_NUM" == "4" ]] || { echo "PREFILL_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ "$DECODE_CORE_NUM" == "1" || "$DECODE_CORE_NUM" == "2" || "$DECODE_CORE_NUM" == "4" ]] || { echo "DECODE_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ "$AR_CORE_NUM" == "1" || "$AR_CORE_NUM" == "2" || "$AR_CORE_NUM" == "4" ]] || { echo "AR_CORE_NUM must be 1, 2, or 4"; exit 1; }
[[ -n "$CALIBRATION_SCALE_MANIFEST" && -f "$CALIBRATION_SCALE_MANIFEST" ]] || { echo "activation scale manifest missing; set CALIBRATION_SCALE_MANIFEST"; exit 1; }
[[ -n "$CALIBRATION_COVERAGE_JSON" && -f "$CALIBRATION_COVERAGE_JSON" ]] || { echo "calibration graph coverage missing; set CALIBRATION_COVERAGE_JSON"; exit 1; }
[[ "$DISABLE_HIDDEN_ROTATION" == "0" || "$DISABLE_HIDDEN_ROTATION" == "1" ]] || { echo "DISABLE_HIDDEN_ROTATION must be 0 or 1"; exit 1; }

environment_temporary="${ENVIRONMENT_PATH}.tmp.$$"
set +e
"$PYTHON_BIN" "$ENVIRONMENT_SCRIPT" \
  --profile build \
  --model-path "$INPUT_MODEL_PATH" \
  --selected-jsonl "$CALIB_JSON" \
  --resource-path "$OUTPUT_MODEL_PATH" \
  --requested-jobs "$JOBS" \
  --device "$DEVICE" \
  --require-cuda \
  > "$environment_temporary"
environment_status=$?
set -e
mv -f "$environment_temporary" "$ENVIRONMENT_PATH"
if [[ "$environment_status" -ne 0 ]]; then
  echo "[build:language] environment gate failed exit_code=$environment_status report=$ENVIRONMENT_PATH"
  exit "$environment_status"
fi

VALIDATION_ARGS=(
  --expected-samples "$EXPECTED_SAMPLES"
  --expected-selected-sha256 "$EXPECTED_SELECTED_MANIFEST_SHA256"
)
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  VALIDATION_ARGS+=(--hidden-rotation-path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  VALIDATION_ARGS+=(--disable-hidden-rotation)
fi

python "$REPO_ROOT/compiler/scripts/validate/deployment.py" \
  --component language \
  --model-path "$INPUT_MODEL_PATH" \
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
if [[ "$FUSED_PBD_PROFILES" == "1" ]]; then
  EXTRA_ARGS+=(--fused_pbd_profiles)
fi

MODEL_BASENAME=$(basename "$INPUT_MODEL_PATH")
HBM_STEM="${MODEL_BASENAME}_language_chunk_${CHUNK_SIZE}_cache_${CACHE_LEN}_decoder_w${W_BITS}_lmhead_w${LM_HEAD_W_BITS}_${MARCH}_corenum_${PREFILL_CORE_NUM}_${DECODE_CORE_NUM}"
if [[ "$AR_CORE_NUM" != "$DECODE_CORE_NUM" ]]; then
  HBM_STEM="${HBM_STEM}_ar${AR_CORE_NUM}"
fi
if [[ "$FUSED_PBD_PROFILES" == "1" ]]; then
  HBM_STEM="${HBM_STEM}_fusedpbd"
fi
HBM_PATH="$OUTPUT_MODEL_PATH/${HBM_STEM}.hbm"
EMBEDDING_PATH="$OUTPUT_MODEL_PATH/${MODEL_BASENAME}_embed_tokens.bin"
EXPECTED_EMBEDDING_BYTES=$((152681 * 2048 * 2))
BC_MANIFEST="${HBM_PATH%.hbm}.bc_manifest.json"
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
)
if [[ "$FUSED_PBD_PROFILES" == "1" ]]; then
  STAGE_ARGS+=(--require_fused)
fi
LANGUAGE_GRAPHS=(
  prefill decode decode_ar
  decode_pbd_q7 decode_pbd_q8 decode_pbd_q9
  decode_pbd_q10 decode_pbd_q11 decode_pbd_q12
  decode_ar_q2 decode_ar_q3 decode_ar_q4 decode_ar_q5
)
PROVENANCE_ARGS=(
  --manifest "$BC_MANIFEST"
  --component language
  --model_path "$INPUT_MODEL_PATH"
  --scale_manifest "$CALIBRATION_SCALE_MANIFEST"
  --field "chunk_size=$CHUNK_SIZE"
  --field "cache_len=$CACHE_LEN"
  --field "pbd_query_len=$DECODE_SEQ_LEN"
  --field "ar_query_len=1"
  --field "decoder_w_bits=$W_BITS"
  --field "lm_head_w_bits=$LM_HEAD_W_BITS"
  --field "fused_pbd=$FUSED_PBD_PROFILES"
  --field "march=$MARCH"
  --artifact "embed_tokens=$EMBEDDING_PATH"
)
for graph in "${LANGUAGE_GRAPHS[@]}"; do
  PROVENANCE_ARGS+=(--artifact "$graph=${HBM_PATH%.hbm}.${graph}.bc")
done
if [[ -n "$HIDDEN_ROTATION_PATH" ]]; then
  PROVENANCE_ARGS+=(--hidden_rotation_path "$HIDDEN_ROTATION_PATH")
fi
if [[ "$DISABLE_HIDDEN_ROTATION" == "1" ]]; then
  PROVENANCE_ARGS+=(--disable_hidden_rotation)
fi

validate_bc() {
  env PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" python \
    "$REPO_ROOT/compiler/scripts/build/language_variants.py" \
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
  echo "[build:language] exporting the complete Language BC graph family"
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
    --export_only \
    "${EXTRA_ARGS[@]}"
}

run_build() (
  set -e
  if [[ "$BUILD_TARGET" == "bc" ]]; then
    if [[ "$RESUME" == "1" ]] && check_bc; then
      echo "[RESUME] Language BC contract already complete: $OUTPUT_MODEL_PATH"
      return 0
    fi
    export_bc
    record_bc
    return 0
  fi

  if check_bc; then
    echo "[REUSE] Language HBM build will consume existing 13-graph BC family"
  else
    echo "[build:language] compatible BC family not found; exporting it first"
    export_bc
    record_bc
  fi

  RESUME_ARGS=()
  [[ "$RESUME" == "1" ]] && RESUME_ARGS+=(--resume)
  env PYTHONUNBUFFERED=1 PYTHONPATH="$LEAP_LLM_SRC${PYTHONPATH:+:$PYTHONPATH}" python \
    "$REPO_ROOT/compiler/scripts/build/language_variants.py" \
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
