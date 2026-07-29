#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_SCRIPT=${SCRIPT_DIR}/../deploy/deploy_locateanything_s600.sh
bash -n "$DEPLOY_SCRIPT"

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/la-deploy-test.XXXXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/deploy" "$WORK_DIR/tokenizer"
printf 'cmake_minimum_required(VERSION 3.10)\n' >"$WORK_DIR/deploy/CMakeLists.txt"
printf 'tokenizer fixture\n' >"$WORK_DIR/tokenizer/tokenizer.json"
printf 'vision\n' >"$WORK_DIR/vision.hbm"
printf 'language\n' >"$WORK_DIR/language.hbm"
printf 'embedding\n' >"$WORK_DIR/embed.bin"
cat >"$WORK_DIR/config.json" <<'EOF'
{"vocab_size":152681,"embed_dim":2048,"image_height":672,"image_width":672}
EOF

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

grep -q 'sha256sum -c DEPLOY_MANIFEST.sha256' "$DEPLOY_SCRIPT"
grep -q 'target already exists' "$DEPLOY_SCRIPT"
grep -q 'staging directory already exists' "$DEPLOY_SCRIPT"
if grep -Eq 'rsync|scp[^#]*-[A-Za-z]*[a-zA-Z]*C|rm[[:space:]]+-rf.*(DEST_ROOT|STAGING_DIR|target|stage)' "$DEPLOY_SCRIPT"; then
  echo "resume or dangerous remote deletion primitive found" >&2
  exit 1
fi

printf '[PASS] deploy_locateanything_s600.sh syntax and dry-run safety checks\n'
