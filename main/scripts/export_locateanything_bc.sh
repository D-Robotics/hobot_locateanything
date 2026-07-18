#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/oe_locateanything}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/main/outputs/export_validation}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/main/logs}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "[1/2] Export LocateAnything Language prefill/decode BC"
EXPORT_ONLY=1 \
OUTPUT_MODEL_PATH="$OUTPUT_ROOT/language" \
LOG_FILE="$LOG_DIR/locateanything_language_export.log" \
  "$REPO_ROOT/main/scripts/compile_locateanything_language.sh"

echo "Language export runs detached. Follow with:"
echo "  tail -f $LOG_DIR/locateanything_language_export.log"
echo
echo "Run the Vision export after the Language log contains:"
echo "  [LocateAnythingLanguageApi] export-only validation passed"
echo
echo "Command:"
echo "  EXPORT_ONLY=1 OUTPUT_MODEL_PATH=$OUTPUT_ROOT/vision LOG_FILE=$LOG_DIR/locateanything_vision_export.log $REPO_ROOT/main/scripts/compile_locateanything_vit.sh"
