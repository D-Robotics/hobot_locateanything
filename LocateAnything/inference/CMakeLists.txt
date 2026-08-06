cmake_minimum_required(VERSION 3.10)
project(locateanything)

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
find_package(cv_bridge REQUIRED)
find_package(hbm_img_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h
  PATHS /usr/include /usr/local/include)
find_library(HBDNN_LIB dnn PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBUCP_LIB hbucp PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBRT4_LIB hbrt4 PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBTL_LIB hbtl PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBDSP_PLUGIN_LIB hbdsp_plugin
  PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBHPL_LIB hbhpl PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(HBVP_LIB hbvp PATHS /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
if(NOT HOBOT_INCLUDE_DIR)
  message(FATAL_ERROR "hobot/dnn/hb_dnn.h not found; build this package in the S600 TROS environment")
endif()

set(HOBOT_RUNTIME_LIBS)
foreach(library HBDNN_LIB HBUCP_LIB HBRT4_LIB HBTL_LIB HBDSP_PLUGIN_LIB HBHPL_LIB HBVP_LIB)
  if(${library})
    list(APPEND HOBOT_RUNTIME_LIBS ${${library}})
  endif()
endforeach()

add_library(locateanything_hbm STATIC
  src/attention_mask.cpp
  src/embed_lookup.cpp
  src/hbm_session.cpp
  src/hybrid_decoder.cpp
  src/kv_cache_ring.cpp
  src/position_ids.cpp)
target_include_directories(locateanything_hbm PUBLIC include ${HOBOT_INCLUDE_DIR})
target_link_libraries(locateanything_hbm PUBLIC ${HOBOT_RUNTIME_LIBS} Threads::Threads)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/hybrid_decoder.cpp PROPERTIES
    COMPILE_OPTIONS "-march=armv8.2-a+fp16")
endif()

add_library(locateanything_inference STATIC
  src/image_preprocessor.cpp
  src/inference_session.cpp
  src/postprocessor.cpp
  src/prompt_builder.cpp
  src/runner_process.cpp
  src/tokenizer.cpp)
target_include_directories(locateanything_inference PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(locateanything_inference PUBLIC ${OpenCV_LIBS} Threads::Threads)

add_executable(vision_runner src/vision_runner.cpp)
target_link_libraries(vision_runner PRIVATE locateanything_hbm)

add_executable(language_runner src/language_runner.cpp)
target_link_libraries(language_runner PRIVATE locateanything_hbm)

add_executable(inference src/inference_node.cpp)
target_link_libraries(inference locateanything_inference)
ament_target_dependencies(inference
  ament_index_cpp cv_bridge hbm_img_msgs rclcpp sensor_msgs std_msgs)

add_executable(console src/console.cpp)
target_link_libraries(console locateanything_inference yaml-cpp::yaml-cpp)
ament_target_dependencies(console ament_index_cpp)

install(TARGETS
  vision_runner
  language_runner
  inference
  console
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch models
  DESTINATION share/${PROJECT_NAME})

ament_package()
