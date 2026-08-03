#!/usr/bin/env bash
# Safely publish one immutable LocateAnything artifact set to an S600 board.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

SSH_TARGET=${LA_S600_SSH_TARGET:-sunrise@10.112.133.20}
DEST_ROOT=${LA_S600_DEST_ROOT:-/home/sunrise/oe_locateanything/LocateAnything/artifacts/releases}
SSH_PORT=${LA_S600_SSH_PORT:-22}
IDENTITY_FILE=
DEPLOY_DIR=${REPO_ROOT}/deploy
VISION_HBM=
LANGUAGE_HBM=
EMBED_BIN=
RUNTIME_CONFIG=
TOKENIZER_DIR=
RELEASE=
EXECUTE=0
SSH_BIN=${SSH_BIN:-ssh}
SCP_BIN=${SCP_BIN:-scp}

usage() {
  cat <<'EOF'
Usage: deploy_locateanything_s600.sh --release NAME --vision-hbm FILE \
       --language-hbm FILE --embed-bin FILE --runtime-config FILE \
       --tokenizer-dir DIR [options]

Required:
  --release NAME            Immutable release name (letters, digits, . _ -)
  --vision-hbm FILE         LocateAnything Vision HBM
  --language-hbm FILE       LocateAnything Language HBM
  --embed-bin FILE          LocateAnything embedding table
  --runtime-config FILE     JSON runtime config used as the source template
  --tokenizer-dir DIR       Complete tokenizer directory

Options:
  --deploy-dir DIR          S600 deployment source tree (default: deploy)
  --ssh-target USER@HOST    S600 SSH target (env: LA_S600_SSH_TARGET)
  --ssh-port PORT           SSH port (env: LA_S600_SSH_PORT; default: 22)
  --identity-file FILE      SSH private key
  --dest-root ABS_DIR       Version parent on S600
                             (env: LA_S600_DEST_ROOT)
  --dry-run                 Build and verify the local plan only (default)
  --execute                 Connect, transfer, verify, and atomically publish
  -h, --help                Show this help

The target is DEST_ROOT/NAME. Existing final or .incoming-NAME directories are
always rejected. Interrupted transfers are never resumed or automatically
deleted. Every transferred file is checked on S600 by byte count and SHA256
before the two ARM64 runners are built in staging and the directory is renamed
to the final release directory.
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
    --release) need_arg "$@"; RELEASE=$2; shift 2 ;;
    --vision-hbm) need_arg "$@"; VISION_HBM=$2; shift 2 ;;
    --language-hbm) need_arg "$@"; LANGUAGE_HBM=$2; shift 2 ;;
    --embed-bin) need_arg "$@"; EMBED_BIN=$2; shift 2 ;;
    --runtime-config) need_arg "$@"; RUNTIME_CONFIG=$2; shift 2 ;;
    --tokenizer-dir) need_arg "$@"; TOKENIZER_DIR=$2; shift 2 ;;
    --deploy-dir) need_arg "$@"; DEPLOY_DIR=$2; shift 2 ;;
    --ssh-target) need_arg "$@"; SSH_TARGET=$2; shift 2 ;;
    --ssh-port) need_arg "$@"; SSH_PORT=$2; shift 2 ;;
    --identity-file) need_arg "$@"; IDENTITY_FILE=$2; shift 2 ;;
    --dest-root) need_arg "$@"; DEST_ROOT=$2; shift 2 ;;
    --dry-run) EXECUTE=0; shift ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n $RELEASE ]] || die "--release is required"
[[ $RELEASE =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "unsafe release name: $RELEASE"
[[ $RELEASE != *..* ]] || die "release name must not contain '..'"
[[ $SSH_TARGET =~ ^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$ ]] || die "unsafe SSH target: $SSH_TARGET"
[[ $SSH_PORT =~ ^[0-9]+$ && $SSH_PORT -ge 1 && $SSH_PORT -le 65535 ]] || die "invalid SSH port: $SSH_PORT"
[[ $DEST_ROOT =~ ^/[A-Za-z0-9._/-]+$ && $DEST_ROOT != *..* && $DEST_ROOT != */ ]] || die "unsafe --dest-root: $DEST_ROOT"

for pair in \
  "vision HBM|$VISION_HBM" \
  "language HBM|$LANGUAGE_HBM" \
  "embedding table|$EMBED_BIN" \
  "runtime config|$RUNTIME_CONFIG"; do
  label=${pair%%|*}
  path=${pair#*|}
  [[ -f $path && -r $path ]] || die "$label is not a readable file: $path"
done
for pair in "deployment source|$DEPLOY_DIR" "tokenizer|$TOKENIZER_DIR"; do
  label=${pair%%|*}
  path=${pair#*|}
  [[ -d $path && -r $path ]] || die "$label is not a readable directory: $path"
  if find "$path" -type l -print -quit | grep -q .; then
    die "$label directory contains a symlink; materialize regular files first: $path"
  fi
done
for required in \
  "$DEPLOY_DIR/CMakeLists.txt" \
  "$DEPLOY_DIR/run_locateanything.py" \
  "$DEPLOY_DIR/run_locateanything_interactive.py" \
  "$DEPLOY_DIR/LocateAnything" \
  "$TOKENIZER_DIR/tokenizer.json"; do
  [[ -f $required && -r $required ]] || die "runtime payload file is missing: $required"
done
[[ -z $IDENTITY_FILE || -f $IDENTITY_FILE ]] || die "identity file is missing: $IDENTITY_FILE"

for command in python3 sha256sum stat tar mktemp; do
  command -v "$command" >/dev/null 2>&1 || die "required command is unavailable: $command"
done

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/la-s600-deploy.XXXXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/config" "$WORK_DIR/bundles"

FINAL_DIR=${DEST_ROOT}/${RELEASE}
STAGING_DIR=${DEST_ROOT}/.incoming-${RELEASE}

printf '[deploy][1/7] generating version-bound runtime config\n'
python3 - "$RUNTIME_CONFIG" "$WORK_DIR/config/locateanything_3b_config.json" "$FINAL_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

source, output, release_dir = map(Path, sys.argv[1:])
try:
    config = json.loads(source.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid runtime config: {exc}")
if not isinstance(config, dict):
    raise SystemExit("runtime config must be a JSON object")
required = {
    "model_type": "LocateAnything-3B",
    "vocab_size": 152681,
    "embed_dim": 2048,
    "image_height": 672,
    "image_width": 672,
    "patch_size": 14,
    "visual_tokens": 576,
    "prefill_chunk": 1024,
    "cache_len": 4096,
    "pbd_query_len": 6,
    "ar_query_len": 1,
    "default_generation_mode": "hybrid",
    "l2m_sizes": "6:6:6:6",
    "vit_bpu_core": [0, 1, 2, 3],
    "prefill_bpu_core": [0, 1, 2, 3],
    "decode_bpu_core": [0, 1, 2, 3],
}
for key, expected in required.items():
    if config.get(key) != expected:
        raise SystemExit(f"runtime config {key}={config.get(key)!r}; expected {expected}")
if config.get("language_graph_set") not in {"standard", "fused_decode"}:
    raise SystemExit(
        "runtime config language_graph_set must be standard or fused_decode"
    )
for key in (
    "default_max_new_tokens", "default_nms_iou", "telemetry_interval_ms",
    "runner_startup_timeout_seconds",
):
    if key not in config:
        raise SystemExit(f"runtime config is missing {key}")
try:
    if int(config["default_max_new_tokens"]) <= 0:
        raise ValueError
    if not 0.0 <= float(config["default_nms_iou"]) <= 1.0:
        raise ValueError
    if int(config["telemetry_interval_ms"]) < 250:
        raise ValueError
    startup_timeout = float(config["runner_startup_timeout_seconds"])
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError
except (TypeError, ValueError):
    raise SystemExit("runtime config has invalid generation, NMS, telemetry, or startup timeout values")
base = str(release_dir).rstrip("/")
config.update({
    "model_dir": f"{base}/artifacts/",
    "vit_model_file": "LocateAnything-3B_vision.hbm",
    "llm_model_file": "LocateAnything-3B_language.hbm",
    "embed_weight_file_path": "LocateAnything-3B_embed_tokens.bin",
    "vocabulary_path": f"{base}/tokenizer/",
})
output.write_text(
    json.dumps(config, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

printf '[deploy][2/7] packaging reusable deployment source and tokenizer\n'
tar --exclude='./build' --exclude='./demo_build' --exclude='./oellm_runtime' \
  --exclude='*/__pycache__' --exclude='*.pyc' \
  -cf "$WORK_DIR/bundles/deploy-source.tar" -C "$DEPLOY_DIR" .
tar -cf "$WORK_DIR/bundles/tokenizer.tar" -C "$TOKENIZER_DIR" .

declare -a LOCAL_FILES=(
  "$VISION_HBM"
  "$LANGUAGE_HBM"
  "$EMBED_BIN"
  "$WORK_DIR/config/locateanything_3b_config.json"
  "$WORK_DIR/bundles/deploy-source.tar"
  "$WORK_DIR/bundles/tokenizer.tar"
)
declare -a REMOTE_FILES=(
  "artifacts/LocateAnything-3B_vision.hbm"
  "artifacts/LocateAnything-3B_language.hbm"
  "artifacts/LocateAnything-3B_embed_tokens.bin"
  "config/locateanything_3b_config.json"
  "bundles/deploy-source.tar"
  "bundles/tokenizer.tar"
)

CHECKSUMS=$WORK_DIR/checksums.sha256
FILE_SIZES=$WORK_DIR/file_sizes.tsv
: >"$CHECKSUMS"
: >"$FILE_SIZES"
printf '[deploy][3/7] hashing six source payloads\n'
for index in "${!LOCAL_FILES[@]}"; do
  source_file=${LOCAL_FILES[$index]}
  remote_file=${REMOTE_FILES[$index]}
  digest=$(sha256sum "$source_file" | awk '{print $1}')
  bytes=$(stat -c '%s' "$source_file")
  printf '%s  %s\n' "$digest" "$remote_file" >>"$CHECKSUMS"
  printf '%s  %s\n' "$bytes" "$remote_file" >>"$FILE_SIZES"
  printf '  [%d/6] %-52s %12s bytes  %s\n' "$((index + 1))" "$remote_file" "$bytes" "$digest"
done

cat >"$WORK_DIR/release_metadata.txt" <<EOF
release=${RELEASE}
destination=${FINAL_DIR}
source_host=$(hostname 2>/dev/null || printf unknown)
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
deployment_policy=immutable-staged-sha256-and-byte-verified
EOF

printf '\n[deploy] target:  %s:%s\n' "$SSH_TARGET" "$FINAL_DIR"
printf '[deploy] staging: %s:%s\n' "$SSH_TARGET" "$STAGING_DIR"
if [[ $EXECUTE -eq 0 ]]; then
  printf '[deploy][DRY-RUN] local payload and checksum validation passed; no SSH/SCP command was run.\n'
  printf '[deploy][DRY-RUN] rerun with --execute to create and publish this immutable release.\n'
  exit 0
fi

SSH_ARGS=(-p "$SSH_PORT" -o BatchMode=yes)
SCP_ARGS=(-P "$SSH_PORT" -o BatchMode=yes)
if [[ -n $IDENTITY_FILE ]]; then
  SSH_ARGS+=(-i "$IDENTITY_FILE")
  SCP_ARGS+=(-i "$IDENTITY_FILE")
fi

printf '[deploy][4/7] reserving a new remote staging directory\n'
"$SSH_BIN" "${SSH_ARGS[@]}" "$SSH_TARGET" sh -s -- "$DEST_ROOT" "$RELEASE" <<'REMOTE_PREFLIGHT'
set -eu
root=$1
release=$2
target=$root/$release
stage=$root/.incoming-$release
umask 077
mkdir -p "$root"
if [ -e "$target" ]; then
  echo "[remote][FAIL] target already exists: $target" >&2
  exit 73
fi
if [ -e "$stage" ]; then
  echo "[remote][FAIL] staging directory already exists: $stage" >&2
  echo "[remote][FAIL] inspect it manually; automatic deletion/resume is forbidden" >&2
  exit 74
fi
mkdir "$stage"
mkdir "$stage/artifacts" "$stage/config" "$stage/bundles"
REMOTE_PREFLIGHT

printf '[deploy][5/7] transferring payloads without overwrite or resume\n'
for index in "${!LOCAL_FILES[@]}"; do
  printf '  [transfer %d/6] %s\n' "$((index + 1))" "${REMOTE_FILES[$index]}"
  "$SCP_BIN" "${SCP_ARGS[@]}" -- "${LOCAL_FILES[$index]}" \
    "${SSH_TARGET}:${STAGING_DIR}/${REMOTE_FILES[$index]}"
done
for metadata in checksums.sha256 file_sizes.tsv release_metadata.txt; do
  "$SCP_BIN" "${SCP_ARGS[@]}" -- "$WORK_DIR/$metadata" \
    "${SSH_TARGET}:${STAGING_DIR}/$metadata"
done

printf '[deploy][6/7] verifying payloads and building ARM64 runtime in staging\n'
"$SSH_BIN" "${SSH_ARGS[@]}" "$SSH_TARGET" sh -s -- "$DEST_ROOT" "$RELEASE" <<'REMOTE_VERIFY'
set -eu
root=$1
release=$2
target=$root/$release
stage=$root/.incoming-$release
[ ! -e "$target" ] || { echo "[remote][FAIL] target appeared during transfer" >&2; exit 75; }
[ -d "$stage" ] || { echo "[remote][FAIL] staging directory is missing" >&2; exit 76; }
cd "$stage"
sha256sum -c checksums.sha256
while read -r expected relative; do
  actual=$(wc -c <"$relative" | tr -d ' ')
  [ "$actual" = "$expected" ] || {
    echo "[remote][FAIL] byte mismatch: $relative expected=$expected actual=$actual" >&2
    exit 77
  }
done <file_sizes.tsv
mkdir deploy tokenizer
tar -xf bundles/deploy-source.tar -C deploy
tar -xf bundles/tokenizer.tar -C tokenizer
test -f deploy/CMakeLists.txt
test -f deploy/run_locateanything.py
test -f deploy/run_locateanything_interactive.py
test -f deploy/LocateAnything
test -f tokenizer/tokenizer.json
command -v cmake >/dev/null 2>&1 || {
  echo "[remote][FAIL] cmake is required to build the S600 runtime" >&2
  exit 78
}
cmake -S deploy -B deploy/build -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/build \
  --target vision_hbm_runner language_hbm_runner \
  -j4
test -x deploy/build/vision_hbm_runner
test -x deploy/build/language_hbm_runner
sha256sum \
  deploy/build/vision_hbm_runner \
  deploy/build/language_hbm_runner >runtime_checksums.sha256
{
  printf 'build_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  cmake --version
} >runtime_environment.txt
printf 'verified_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>release_metadata.txt
printf 'status=sha256-byte-verified-and-runtime-built\n' >>release_metadata.txt
mv "$stage" "$target"
printf '[remote][PASS] immutable release published: %s\n' "$target"
REMOTE_VERIFY

printf '[deploy][7/7] deployment complete: %s:%s\n' "$SSH_TARGET" "$FINAL_DIR"
printf '[deploy] checksums.sha256 and file_sizes.tsv remain with the deployed release.\n'
