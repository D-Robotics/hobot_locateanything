#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "${1:-}" = "--interactive" ]; then
  shift
  exec python3 "$SCRIPT_DIR/run_locateanything_interactive.py" "$@"
fi
exec python3 "$SCRIPT_DIR/run_locateanything.py" "$@"
