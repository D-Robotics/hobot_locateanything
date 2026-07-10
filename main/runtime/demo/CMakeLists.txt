# LocateAnything demo — 仿 oellm_runtime/examples/vlm_demo/CMakeLists.txt
# Vendored from vlm_demo_CMakeLists.txt with modifications: target name +
# 指向我们自己的 oellm_runtime SDK (../oellm_runtime/).

cmake_minimum_required(VERSION 3.15)
project(locateanything_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# S600 上 oellm_runtime SDK 在 repo 根的 oellm_runtime/
get_filename_component(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../oellm_runtime" ABSOLUTE)
include_directories(${SDK_ROOT}/include)
link_directories(${SDK_ROOT}/lib)

add_executable(locateanything_demo locateanything_demo.cc)
target_link_libraries(locateanything_demo xlm)
# vlm_demo CMakeLists 用 -Wl,-unresolved-symbols=ignore-in-shared-libs,
# 因为 libxlm.so 依赖的很多 .so 在运行时由 LD_LIBRARY_PATH 提供.
target_link_options(locateanything_demo PRIVATE
  -Wl,-unresolved-symbols=ignore-in-shared-libs)
