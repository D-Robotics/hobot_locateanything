#!/bin/sh
# Repeat the real LocateAnything CLI while collecting S600 resource evidence.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PRODUCT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
RUNS=10
WARMUP=1
INTERVAL=0.2
TIMEOUT=600
OUTPUT_DIR="$PRODUCT_ROOT/workspace/benchmarks/s600-$(date -u +%Y%m%dT%H%M%SZ)"
PROMPT="/detect cat"
GENERATION_MODE=hybrid
MAX_NEW_TOKENS=2048
SEMANTIC_REGEX=
IMAGE=
METRICS=

usage() {
  cat <<'EOF'
Usage: run_s600_benchmark.sh --image FILE [options]

Options:
  --prompt TEXT              LocateAnything task command (default: /detect cat)
  --generation-mode MODE    hybrid or slow (default: hybrid)
  --max-new-tokens N        generation limit (default: 2048)
  --runs N                  measured repetitions (default: 10)
  --warmup N                warm-up repetitions (default: 1)
  --interval SECONDS        resource sample interval (default: 0.2)
  --timeout SECONDS         per-run timeout (default: 600)
  --output-dir DIR          benchmark evidence directory
  --semantic-regex REGEX    required output pattern
  --bpu-metric NAME=PATH    extra board metric, repeatable
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2 ;;
    --prompt) PROMPT=$2; shift 2 ;;
    --generation-mode) GENERATION_MODE=$2; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --semantic-regex) SEMANTIC_REGEX=$2; shift 2 ;;
    --bpu-metric) METRICS="${METRICS}${METRICS:+
}$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$IMAGE" ] && [ -f "$IMAGE" ] || {
  echo "--image must name a readable file" >&2
  exit 2
}
[ "$GENERATION_MODE" = hybrid ] || [ "$GENERATION_MODE" = slow ] || {
  echo "--generation-mode must be hybrid or slow" >&2
  exit 2
}

set -- python3 "$SCRIPT_DIR/benchmark_s600_runtime.py" \
  --runs "$RUNS" --warmup "$WARMUP" --interval "$INTERVAL" \
  --timeout "$TIMEOUT" --output-dir "$OUTPUT_DIR" \
  --artifact "$IMAGE"
if [ -n "$SEMANTIC_REGEX" ]; then
  set -- "$@" --semantic-regex "$SEMANTIC_REGEX"
fi
old_ifs=$IFS
IFS='
'
for metric in $METRICS; do
  set -- "$@" --bpu-metric "$metric"
done
IFS=$old_ifs
set -- "$@" -- python3 "$SCRIPT_DIR/run_locateanything.py" \
  --image "$IMAGE" --prompt "$PROMPT" \
  --generation-mode "$GENERATION_MODE" --max-new-tokens "$MAX_NEW_TOKENS"

echo "[benchmark] output=$OUTPUT_DIR"
exec "$@"
