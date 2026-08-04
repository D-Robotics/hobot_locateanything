#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEPLOY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
INSTALL_DIR=${1:-"$HOME/.local/bin"}
COMMAND="$INSTALL_DIR/LocateAnything"
LAUNCHER="$DEPLOY_ROOT/bin/LocateAnything"

mkdir -p "$INSTALL_DIR"
ln -sfn "$LAUNCHER" "$COMMAND"
chmod +x "$LAUNCHER"

printf 'Installed: %s\n' "$COMMAND"
case ":${PATH:-}:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    printf 'Add this directory to PATH:\n'
    printf '  export PATH="%s:$PATH"\n' "$INSTALL_DIR"
    ;;
esac
