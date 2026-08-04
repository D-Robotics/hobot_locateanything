# LocateAnything S600 runtime and HBM executables.

cmake_minimum_required(VERSION 3.10)
project(locateanything_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

add_library(locateanything_runtime STATIC
  src/hbm_session.cpp
  src/embed_lookup.cpp
  src/attention_mask.cpp
  src/hybrid_decoder.cpp
  src/kv_cache_ring.cpp
  src/position_ids.cpp)
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

# Executables used by the Python CLI.
add_executable(vision_hbm_runner apps/vision_hbm_runner.cpp)
target_link_libraries(vision_hbm_runner PRIVATE locateanything_runtime)

add_executable(language_hbm_runner apps/language_hbm_runner.cpp)
target_link_libraries(language_hbm_runner PRIVATE locateanything_runtime)
