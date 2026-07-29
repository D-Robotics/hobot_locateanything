#!/bin/sh
set -eu

usage() {
  echo "usage: $0 --model VISION.hbm --input vision_input.f16.bin --output vision_output.f16.bin" >&2
}

MODEL=
INPUT=
OUTPUT=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --input) INPUT=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    *) usage; exit 1 ;;
  esac
done

[ -n "$MODEL" ] && [ -f "$MODEL" ] || { usage; exit 1; }
[ -n "$INPUT" ] && [ -f "$INPUT" ] || { usage; exit 1; }
[ -n "$OUTPUT" ] || { usage; exit 1; }

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BINARY=$SCRIPT_DIR/build/vision_hbm_runner
[ -x "$BINARY" ] || {
  echo "runner not built: $BINARY" >&2
  echo "build with: cmake -S $SCRIPT_DIR -B $SCRIPT_DIR/build && cmake --build $SCRIPT_DIR/build --target vision_hbm_runner -j4" >&2
  exit 2
}

mkdir -p "$(dirname -- "$OUTPUT")"
export HB_DNN_USER_DEFINED_L2M_SIZES=${HB_DNN_USER_DEFINED_L2M_SIZES:-6:6:6:6}

echo "[vision-hbm] model=$MODEL"
echo "[vision-hbm] input=$INPUT"
echo "[vision-hbm] output=$OUTPUT"
echo "[vision-hbm] l2m=$HB_DNN_USER_DEFINED_L2M_SIZES"
sha256sum "$MODEL" "$INPUT"

"$BINARY" --model "$MODEL" --input "$INPUT" --output "$OUTPUT"
sha256sum "$OUTPUT" | tee "$OUTPUT.sha256"
