#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all \
  -v /home/kangjie.xu/oe_locateanything:/workspace/oe_locateanything \
  -w /workspace/oe_locateanything \
  locateanything_oellm_s600:1.0.5 \
  "$@"
