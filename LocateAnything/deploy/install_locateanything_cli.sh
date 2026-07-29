#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALL_DIR=${1:-"$HOME/.local/bin"}
COMMAND="$INSTALL_DIR/LocateAnything"

mkdir -p "$INSTALL_DIR"
ln -sfn "$SCRIPT_DIR/LocateAnything" "$COMMAND"
chmod +x "$SCRIPT_DIR/LocateAnything"

printf 'Installed: %s\n' "$COMMAND"
case ":${PATH:-}:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    printf 'Add this directory to PATH:\n'
    printf '  export PATH="%s:$PATH"\n' "$INSTALL_DIR"
    ;;
esac
