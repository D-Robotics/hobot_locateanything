#!/bin/bash

ARCH=$(uname -m)
case "${ARCH}" in
  aarch64|arm64|armv7l|armv8*|arm*)
    echo "Detected native ARM system (${ARCH}), using system gcc/g++"
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    ;;
  *)
    if [ "${LINARO_GCC_ROOT}" ]; then
      LINARO_GCC_ROOT=${LINARO_GCC_ROOT}
    else
      echo "Please set environment LINARO_GCC_ROOT correctly"
      LINARO_GCC_ROOT=/opt/aarch64/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
    fi

    export CC="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-gcc"
    export CXX="${LINARO_GCC_ROOT}/bin/aarch64-none-linux-gnu-g++"
    ;;
esac

if [ -d "build" ]; then
  rm -rf build
fi

mkdir build
cd build
cmake ..
make
