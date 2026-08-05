#!/usr/bin/env bash
# Copy the selected model files into the checked-out LocateAnything project.
set -euo pipefail

SSH_TARGET=${LA_S600_SSH_TARGET:-sunrise@10.112.133.20}
DEST_ROOT=${LA_S600_DEST_ROOT:-/home/sunrise/oe_locateanything/LocateAnything}
SSH_PORT=${LA_S600_SSH_PORT:-22}
VISION_HBM=
LANGUAGE_HBM=
EMBED_BIN=
EXECUTE=0
SKIP_BUILD=0
SSH_BIN=${SSH_BIN:-ssh}
SCP_BIN=${SCP_BIN:-scp}

usage() {
  cat <<'EOF'
Usage: deploy.sh --vision-hbm FILE --language-hbm FILE --embed-bin FILE [options]

Copy model assets into the checked-out S600 project and rebuild its runtime.
The target layout is:
  PROJECT/models/LocateAnything-3B/
  PROJECT/deploy/build/

Options:
  --ssh-target USER@HOST    S600 target (env: LA_S600_SSH_TARGET)
  --ssh-port PORT           SSH port (env: LA_S600_SSH_PORT; default: 22)
  --dest-root ABS_DIR       project root on S600
                             (env: LA_S600_DEST_ROOT)
  --skip-build              upload assets without rebuilding the runners
  --dry-run                 validate local inputs only (default)
  --execute                 upload assets and rebuild the S600 runtime
  -h, --help                show this help
EOF
}

die() {
  printf '[deploy][FAIL] %s\n' "$*" >&2
  exit 2
}

need_arg() {
  [[ $# -ge 2 && -n ${2:-} ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vision-hbm) need_arg "$@"; VISION_HBM=$2; shift 2 ;;
    --language-hbm) need_arg "$@"; LANGUAGE_HBM=$2; shift 2 ;;
    --embed-bin) need_arg "$@"; EMBED_BIN=$2; shift 2 ;;
    --ssh-target) need_arg "$@"; SSH_TARGET=$2; shift 2 ;;
    --ssh-port) need_arg "$@"; SSH_PORT=$2; shift 2 ;;
    --dest-root) need_arg "$@"; DEST_ROOT=$2; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run) EXECUTE=0; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for pair in \
  "vision HBM|$VISION_HBM" \
  "language HBM|$LANGUAGE_HBM" \
  "embedding table|$EMBED_BIN"; do
  label=${pair%%|*}
  path=${pair#*|}
  [[ -f $path && -r $path ]] || die "$label is not a readable file: $path"
done

[[ $SSH_PORT =~ ^[0-9]+$ && $SSH_PORT -ge 1 && $SSH_PORT -le 65535 ]] || \
  die "invalid SSH port: $SSH_PORT"

for command in "$SSH_BIN" "$SCP_BIN"; do
  command -v "$command" >/dev/null 2>&1 || die "required command is unavailable: $command"
done

printf '[deploy] project root: %s\n' "$DEST_ROOT"
printf '[deploy] model files:  %s\n' "$DEST_ROOT/models/LocateAnything-3B"
printf '[deploy] runtime:      %s/deploy/build\n' "$DEST_ROOT"
if [[ $EXECUTE -eq 0 ]]; then
  printf '[deploy][DRY-RUN] local input validation passed; no SSH/SCP command was run.\n'
  exit 0
fi

SSH_ARGS=(-p "$SSH_PORT" -o BatchMode=yes)
SCP_ARGS=(-P "$SSH_PORT" -o BatchMode=yes)
MODEL_DIR="$DEST_ROOT/models/LocateAnything-3B"
STAGING_DIR="$DEST_ROOT/models/.incoming-locateanything"

printf '[deploy][1/4] reserving model staging directory\n'
"$SSH_BIN" "${SSH_ARGS[@]}" "$SSH_TARGET" sh -s -- "$DEST_ROOT" <<'REMOTE_PREPARE'
set -eu
root=$1
model_dir=$root/models/LocateAnything-3B
staging=$root/models/.incoming-locateanything
mkdir -p "$root/models"
if [ -e "$staging" ]; then
  echo "[remote][FAIL] staging directory already exists: $staging" >&2
  exit 73
fi
mkdir "$staging"
mkdir -p "$model_dir"
REMOTE_PREPARE

printf '[deploy][2/4] transferring model files\n'
"$SCP_BIN" "${SCP_ARGS[@]}" -- "$VISION_HBM" \
  "$SSH_TARGET:${STAGING_DIR}/LocateAnything-3B_vision.hbm"
"$SCP_BIN" "${SCP_ARGS[@]}" -- "$LANGUAGE_HBM" \
  "$SSH_TARGET:${STAGING_DIR}/LocateAnything-3B_language.hbm"
"$SCP_BIN" "${SCP_ARGS[@]}" -- "$EMBED_BIN" \
  "$SSH_TARGET:${STAGING_DIR}/LocateAnything-3B_embed_tokens.bin"

printf '[deploy][3/4] publishing model files\n'
"$SSH_BIN" "${SSH_ARGS[@]}" "$SSH_TARGET" sh -s -- "$DEST_ROOT" <<'REMOTE_PUBLISH'
set -eu
root=$1
model_dir=$root/models/LocateAnything-3B
staging=$root/models/.incoming-locateanything
for name in LocateAnything-3B_vision.hbm LocateAnything-3B_language.hbm LocateAnything-3B_embed_tokens.bin; do
  test -s "$staging/$name"
  mv -f "$staging/$name" "$model_dir/$name"
done
rmdir "$staging"
REMOTE_PUBLISH

if [[ $SKIP_BUILD -eq 0 ]]; then
  printf '[deploy][4/4] rebuilding S600 runtime\n'
  "$SSH_BIN" "${SSH_ARGS[@]}" "$SSH_TARGET" sh -s -- "$DEST_ROOT" <<'REMOTE_BUILD'
set -eu
root=$1
test -f "$root/deploy/CMakeLists.txt"
command -v cmake >/dev/null 2>&1
cmake -S "$root/deploy" -B "$root/deploy/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/deploy/build" --target vision_hbm_runner language_hbm_runner -j4
test -x "$root/deploy/build/vision_hbm_runner"
test -x "$root/deploy/build/language_hbm_runner"
REMOTE_BUILD
else
  printf '[deploy][4/4] runtime build skipped\n'
fi

printf '[deploy][PASS] model files are in %s\n' "$MODEL_DIR"
