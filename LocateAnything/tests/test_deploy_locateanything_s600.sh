#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_SCRIPT=${SCRIPT_DIR}/../deploy/deploy_locateanything_s600.sh
bash -n "$DEPLOY_SCRIPT"

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/la-deploy-test.XXXXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/deploy" "$WORK_DIR/tokenizer"
printf 'cmake_minimum_required(VERSION 3.10)\n' >"$WORK_DIR/deploy/CMakeLists.txt"
cp "$SCRIPT_DIR/../deploy/run_locateanything.py" "$WORK_DIR/deploy/"
cp "$SCRIPT_DIR/../deploy/run_locateanything_interactive.py" "$WORK_DIR/deploy/"
cp "$SCRIPT_DIR/../deploy/LocateAnything" "$WORK_DIR/deploy/"
printf 'tokenizer fixture\n' >"$WORK_DIR/tokenizer/tokenizer.json"
printf 'vision\n' >"$WORK_DIR/vision.hbm"
printf 'language\n' >"$WORK_DIR/language.hbm"
printf 'embedding\n' >"$WORK_DIR/embed.bin"
cp "$SCRIPT_DIR/../deploy/runtime_config.json" "$WORK_DIR/config.json"

OUTPUT=$(bash "$DEPLOY_SCRIPT" \
  --release la-test-001 \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" \
  --ssh-target test@s600 \
  --dest-root /home/test/releases \
  --dry-run)

grep -q '\[deploy\]\[DRY-RUN\].*no SSH/SCP command was run' <<<"$OUTPUT"
grep -q '/home/test/releases/la-test-001' <<<"$OUTPUT"
grep -q 'artifacts/LocateAnything-3B_vision.hbm' <<<"$OUTPUT"

printf '#!/bin/sh\r\nexit 0\r\n' >"$WORK_DIR/deploy/LocateAnything"
if bash "$DEPLOY_SCRIPT" \
  --release la-test-crlf \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" \
  --ssh-target test@s600 \
  --dest-root /home/test/releases >"$WORK_DIR/crlf.log" 2>&1; then
  echo "deployment with CRLF shell entrypoint unexpectedly passed" >&2
  exit 1
fi
grep -q 'deployment shell script contains CRLF line endings' "$WORK_DIR/crlf.log"
cp "$SCRIPT_DIR/../deploy/LocateAnything" "$WORK_DIR/deploy/LocateAnything"

python3 - "$WORK_DIR/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
config.update({
    "default_max_new_tokens": 99,
    "default_nms_iou": 0.75,
    "telemetry_interval_ms": 1250,
    "runner_startup_timeout_seconds": 45,
})
path.write_text(json.dumps(config))
PY
FLEX_OUTPUT=$(bash "$DEPLOY_SCRIPT" \
  --release la-test-flexible-runtime \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" \
  --ssh-target test@s600 \
  --dest-root /home/test/releases \
  --dry-run)
grep -q '\[deploy\]\[DRY-RUN\]' <<<"$FLEX_OUTPUT"

mv "$WORK_DIR/tokenizer/tokenizer.json" "$WORK_DIR/tokenizer/not-a-tokenizer.txt"
if bash "$DEPLOY_SCRIPT" \
  --release la-test-missing-tokenizer \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" >/dev/null 2>&1; then
  echo "deployment without tokenizer.json unexpectedly passed" >&2
  exit 1
fi
mv "$WORK_DIR/tokenizer/not-a-tokenizer.txt" "$WORK_DIR/tokenizer/tokenizer.json"

python3 - "$WORK_DIR/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
config["runner_startup_timeout_seconds"] = float("nan")
path.write_text(json.dumps(config))
PY
if bash "$DEPLOY_SCRIPT" \
  --release la-test-nan-timeout \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" >/dev/null 2>&1; then
  echo "deployment with NaN startup timeout unexpectedly passed" >&2
  exit 1
fi
cp "$SCRIPT_DIR/../deploy/runtime_config.json" "$WORK_DIR/config.json"

if bash "$DEPLOY_SCRIPT" \
  --release ../unsafe \
  --vision-hbm "$WORK_DIR/vision.hbm" \
  --language-hbm "$WORK_DIR/language.hbm" \
  --embed-bin "$WORK_DIR/embed.bin" \
  --runtime-config "$WORK_DIR/config.json" \
  --deploy-dir "$WORK_DIR/deploy" \
  --tokenizer-dir "$WORK_DIR/tokenizer" >/dev/null 2>&1; then
  echo "unsafe release unexpectedly passed" >&2
  exit 1
fi

grep -q 'sha256sum -c checksums.sha256' "$DEPLOY_SCRIPT"
grep -q 'target already exists' "$DEPLOY_SCRIPT"
grep -q 'staging directory already exists' "$DEPLOY_SCRIPT"
grep -q 'cmake -S deploy -B deploy/build' "$DEPLOY_SCRIPT"
grep -q 'runtime_checksums.sha256' "$DEPLOY_SCRIPT"
grep -q 'test -f tokenizer/tokenizer.json' "$DEPLOY_SCRIPT"
grep -q '/home/sunrise/oe_locateanything/LocateAnything/artifacts/releases' "$DEPLOY_SCRIPT"
grep -q 'LA_REMOTE_RELEASE_DIR' "$DEPLOY_SCRIPT"
grep -q 'generated runtime config contains an unexpected model directory' "$DEPLOY_SCRIPT"
if grep -Eq 'rsync|scp[^#]*-[A-Za-z]*[a-zA-Z]*C|rm[[:space:]]+-rf.*(DEST_ROOT|STAGING_DIR|target|stage)' "$DEPLOY_SCRIPT"; then
  echo "resume or dangerous remote deletion primitive found" >&2
  exit 1
fi

printf '[PASS] deploy_locateanything_s600.sh syntax and dry-run safety checks\n'
