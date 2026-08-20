# Copyright (c) 2025 D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
project(hobot_locateanything)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(hbm_img_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

set(HOBOT_INCLUDE_PATHS
  ${CMAKE_SYSROOT}/usr/hobot/include
  ${CMAKE_SYSROOT}/usr/include
  /usr/hobot/include
  /usr/include
  /usr/local/include)
set(HOBOT_LIBRARY_PATHS
  ${CMAKE_SYSROOT}/usr/hobot/lib
  ${CMAKE_SYSROOT}/usr/lib/hbbpu
  ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
  ${CMAKE_SYSROOT}/usr/lib
  /usr/hobot/lib
  /usr/lib/hbbpu
  /usr/lib/aarch64-linux-gnu
  /usr/lib)

find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h PATHS ${HOBOT_INCLUDE_PATHS})
find_library(HBDNN_LIB dnn PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBUCP_LIB hbucp PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBRT4_LIB hbrt4 PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBTL_LIB hbtl PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBDSP_PLUGIN_LIB hbdsp_plugin
  PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBHPL_LIB hbhpl PATHS ${HOBOT_LIBRARY_PATHS})
find_library(HBVP_LIB hbvp PATHS ${HOBOT_LIBRARY_PATHS})
if(NOT HOBOT_INCLUDE_DIR)
  message(FATAL_ERROR "hobot/dnn/hb_dnn.h not found; build this package in the S600 TROS environment")
endif()
if(NOT HBDNN_LIB)
  message(FATAL_ERROR "libdnn not found; build this package in the S600 TROS environment")
endif()

set(HOBOT_RUNTIME_LIBS)
foreach(library HBDNN_LIB HBUCP_LIB HBRT4_LIB HBTL_LIB HBDSP_PLUGIN_LIB HBHPL_LIB HBVP_LIB)
  if(${library})
    list(APPEND HOBOT_RUNTIME_LIBS ${${library}})
  endif()
endforeach()

add_library(hobot_locateanything_package STATIC src/package_paths.cpp)
target_include_directories(hobot_locateanything_package PUBLIC include)
ament_target_dependencies(hobot_locateanything_package ament_index_cpp)

add_library(hobot_locateanything_runtime STATIC
  src/runtime/attention_mask.cpp
  src/runtime/embedding.cpp
  src/runtime/hbm.cpp
  src/runtime/decoder.cpp
  src/runtime/kv_cache.cpp
  src/runtime/language.cpp
  src/runtime/position.cpp
  src/runtime/vision.cpp)
target_include_directories(hobot_locateanything_runtime PUBLIC
  include
  ${HOBOT_INCLUDE_DIR})
target_link_libraries(hobot_locateanything_runtime PUBLIC
  ${HOBOT_RUNTIME_LIBS}
  Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/runtime/decoder.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()

add_library(hobot_locateanything_inference STATIC
  src/inference.cpp
  src/processing/image.cpp
  src/processing/postprocess.cpp
  src/processing/prompt.cpp
  src/processing/tokenizer.cpp)
target_include_directories(hobot_locateanything_inference PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hobot_locateanything_inference PUBLIC
  hobot_locateanything_runtime
  ${OpenCV_LIBS}
  Threads::Threads)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/locateanything_node.cpp)
target_link_libraries(${PROJECT_NAME}
  hobot_locateanything_inference
  hobot_locateanything_package)
ament_target_dependencies(${PROJECT_NAME}
  ai_msgs cv_bridge hbm_img_msgs rclcpp sensor_msgs std_msgs)

add_executable(console src/cli.cpp)
target_link_libraries(console
  hobot_locateanything_inference
  hobot_locateanything_package
  yaml-cpp::yaml-cpp)

install(TARGETS
  ${PROJECT_NAME}
  console
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config/
  DESTINATION lib/${PROJECT_NAME}/config)

install(DIRECTORY image/
  DESTINATION lib/${PROJECT_NAME}/image)

# Model weights are downloaded after installation. Tokenizer files ship with the package.
install(FILES
  models/tokenizer/vocab.json
  models/tokenizer/merges.txt
  models/tokenizer/added_tokens.json
  DESTINATION lib/${PROJECT_NAME}/models/tokenizer)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
