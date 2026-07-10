#!/bin/sh
# Run vision_dummy_test on S600 with the L2 memspace env var that the
# oellm_runtime run_vlm.sh reference sets. Without this, hbDNN reports
# "L2 memory not enough, user-assigned l2 memspace size: [0,0,0,0]" and
# hbUCPWaitTaskDone fails with code -200003.
#
# Usage:
#   ./run_vision_dummy_test.sh [path/to/vision.hbm]
# Default hbm path resolves to the standard S600 deploy layout.

set -e

# Match oellm_runtime/examples/vlm_demo/run_vlm.sh's L2M allocation.
export HB_DNN_USER_DEFINED_L2M_SIZES=${HB_DNN_USER_DEFINED_L2M_SIZES:-6:6:6:6}

HBM=${1:-/home/sunrise/oe_locateanything/main/vision/outputs/locateanything-vit-3b_nash-p_w4/LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build
if [ ! -x "${BUILD_DIR}/vision_dummy_test" ]; then
  echo "binary not built; run: cd ${SCRIPT_DIR} && mkdir build && cd build && cmake .. && make" >&2
  exit 1
fi

echo "[run_vision_dummy_test] HB_DNN_USER_DEFINED_L2M_SIZES=${HB_DNN_USER_DEFINED_L2M_SIZES}"
echo "[run_vision_dummy_test] hbm: ${HBM}"
exec "${BUILD_DIR}/vision_dummy_test" "${HBM}"
