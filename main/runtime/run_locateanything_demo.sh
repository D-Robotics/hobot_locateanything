#!/bin/sh
# Run locateanything_demo on S600.
# Sets the L2 memspace env var (KNOWN_ISSUES #017) + LD_LIBRARY_PATH for
# libxlm.so's bundled deps.
SDK_ROOT=$(cd "$(dirname "$0")/../oellm_runtime" && pwd)
export LD_LIBRARY_PATH=${SDK_ROOT}/lib:${LD_LIBRARY_PATH:-}
export HB_DNN_USER_DEFINED_L2M_SIZES=${HB_DNN_USER_DEFINED_L2M_SIZES:-6:6:6:6}

BUILD_DIR=$(cd "$(dirname "$0")" && pwd)/demo_build
CONFIG=$(dirname "$0")/locateanything_3b_config.json
IMAGE=${1:-$(dirname "$0")/../../../main/examples/test-cat.jpg}
PROMPT=${2:-"Please locate the cat and output the bounding box."}

echo "[run] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[run] L2M=${HB_DNN_USER_DEFINED_L2M_SIZES}"
echo "[run] binary=${BUILD_DIR}/locateanything_demo"
echo "[run] config=${CONFIG}"
echo "[run] image=${IMAGE}"

exec "${BUILD_DIR}/locateanything_demo" -c "${CONFIG}" -i "${IMAGE}" -p "${PROMPT}"
