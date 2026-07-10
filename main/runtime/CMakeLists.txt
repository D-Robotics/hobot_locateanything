# LocateAnything C++ host runtime — CMake build.
#
# Phase 1 scope: a self-contained sanity test that loads vision.hbm and
# runs the "visual" graph with a dummy input. No external deps (no Python,
# no pybind11, no image codec yet).
#
# Build (on S600):
#   cd main/runtime
#   mkdir build && cd build
#   cmake ..
#   make -j4
#   ./vision_dummy_test ../path/to/vision.hbm
#
# Or out-of-source with a custom hbm path:
#   ./vision_dummy_test /home/sunrise/oe_locateanything/main/vision/outputs/locateanything-vit-3b_nash-p_w4/LocateAnything-3B_vision_448x448_w8_nash-p_corenum_4.hbm

cmake_minimum_required(VERSION 3.10)
project(locateanything_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# D-Robotics hobot-dnn ships headers under /usr/include/hobot and
# libraries under /usr/hobot/lib on S600. We rely on the system include
# path for headers and explicitly pass -L for the lib.
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64_be")
  message(WARNING "locateanything_runtime is intended for S600 (aarch64). "
                  "Building on ${CMAKE_SYSTEM_PROCESSOR} may still link but the "
                  "BPU calls will fail at runtime.")
endif()

# Search /usr/hobot for the hbrt4 / hbucp / hb_dnn C API.
find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h
          PATHS /usr/include /usr/local/include)
find_library(HBDNN_LIB hbdnn       PATHS /usr/hobot/lib /usr/lib)
find_library(HBUCP_LIB  hbucp       PATHS /usr/hobot/lib /usr/lib)
find_library(HBRT4_LIB  hbrt4       PATHS /usr/hobot/lib /usr/lib)
find_library(HBTL_LIB  hbtl         PATHS /usr/hobot/lib /usr/lib)
find_library(HBDSP_PLUGIN_LIB hb_dsp_plugin PATHS /usr/hobot/lib /usr/lib)
find_library(HBHPL_LIB hbhpl        PATHS /usr/hobot/lib /usr/lib)
find_library(HBVP_LIB  hbvp        PATHS /usr/hobot/lib /usr/lib)

if(NOT HOBOT_INCLUDE_DIR)
  message(FATAL_ERROR "hobot/dnn/hb_dnn.h not found — install the hobot-dnn "
                       "deb package: sudo apt-get install -y hobot-dnn")
endif()

# The runtime library (Phase 1 only has hbm_session + image_preprocess).
add_library(locateanything_runtime STATIC
  src/hbm_session.cpp
  src/image_preprocess.cpp)
target_include_directories(locateanything_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${HOBOT_INCLUDE_DIR})

# Link the D-Robotics runtime libraries. The wrapper only needs hb_dnn +
# hb_ucp directly, but on S600 those transitively depend on hbrt4 and the
# underlying ucp/vp/pl/dsp_plugin libs — list them all to be safe.
set(LA_RT_LIBS)
if(HBDNN_LIB)
  list(APPEND LA_RT_LIBS ${HBDNN_LIB})
endif()
if(HBUCP_LIB)
  list(APPEND LA_RT_LIBS ${HBUCP_LIB})
endif()
if(HBRT4_LIB)
  list(APPEND LA_RT_LIBS ${HBRT4_LIB})
endif()
if(HBTL_LIB)
  list(APPEND LA_RT_LIBS ${HBTL_LIB})
endif()
if(HBHPL_LIB)
  list(APPEND LA_RT_LIBS ${HBHPL_LIB})
endif()
if(HBVP_LIB)
  list(APPEND LA_RT_LIBS ${HBVP_LIB})
endif()
if(HBDSP_PLUGIN_LIB)
  list(APPEND LA_RT_LIBS ${HBDSP_PLUGIN_LIB})
endif()
target_link_libraries(locateanything_runtime PUBLIC ${LA_RT_LIBS})

# --- Phase 1 sanity executable ------------------------------------------------
add_executable(vision_dummy_test example/vision_dummy_test.cpp)
target_link_libraries(vision_dummy_test PRIVATE locateanything_runtime)
