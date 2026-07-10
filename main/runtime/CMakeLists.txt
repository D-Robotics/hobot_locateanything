# LocateAnything C++ host runtime — CMake build.
#
# Phase 1 (done): vision.hbm sanity via HbmSession + image_preprocess.
# Phase 2 (in progress): language hbm loop — embed_lookup + attention_mask
#   + position_ids + kv_cache + pbd_generate. See docs/RUNTIME_ARCHITECTURE.md
#   for the overall split-HBM design and docs/KNOWN_ISSUES.md #019 for the
#   decision to self-implement rather than use libxlm.so.
#
# Build (on S600):
#   cd main/runtime
#   mkdir build && cd build
#   cmake ..
#   make -j4
#   ../run_vision_dummy_test.sh        # Phase 1 sanity
#   ./embed_lookup_test <embed_tokens.bin>   # Phase 2 embed unit test

cmake_minimum_required(VERSION 3.10)
project(locateanything_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# S600 symbol homes (verified by nm -D, see KNOWN_ISSUES #015):
#   hbDNN*     -> libdnn.so      (in /usr/hobot/lib)
#   hbUCP*     -> libhbucp.so    (in /usr/hobot/lib)
find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h
          PATHS /usr/include /usr/local/include)
find_library(HBDNN_LIB dnn            PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBUCP_LIB  hbucp         PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBRT4_LIB  hbrt4         PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBTL_LIB  hbtl           PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBDSP_PLUGIN_LIB hbdsp_plugin PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBHPL_LIB hbhpl        PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBVP_LIB  hbvp        PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)

if(NOT HOBOT_INCLUDE_DIR)
  message(FATAL_ERROR "hobot/dnn/hb_dnn.h not found — install the hobot-dnn "
                       "deb package: sudo apt-get install -y hobot-dnn")
endif()

# The runtime library. Phase 2 adds embed_lookup; attention_mask /
# position_ids / kv_cache / pbd_generate land in subsequent commits.
add_library(locateanything_runtime STATIC
  src/hbm_session.cpp
  src/image_preprocess.cpp
  src/embed_lookup.cpp
  src/attention_mask.cpp)
target_include_directories(locateanything_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${HOBOT_INCLUDE_DIR})

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

# --- Phase 2 embed_lookup unit test -------------------------------------------
add_executable(embed_lookup_test example/embed_lookup_test.cpp)
target_link_libraries(embed_lookup_test PRIVATE locateanything_runtime)

# --- Phase 2 attention_mask unit test ----------------------------------------
add_executable(attention_mask_test example/attention_mask_test.cpp)
target_link_libraries(attention_mask_test PRIVATE locateanything_runtime)
