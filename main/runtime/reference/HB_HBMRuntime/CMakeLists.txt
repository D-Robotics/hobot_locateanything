cmake_minimum_required(VERSION 3.18)
project(hbm_runtime LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# set(CMAKE_BUILD_TYPE Debug)

find_package(pybind11 REQUIRED)
find_library(UCP_LIB hbucp HINTS /usr/hobot/lib/)
find_library(DNN_LIB dnn HINTS /usr/hobot/lib)
include_directories(
  /usr/include/hobot
  /usr/include/hobot/dnn
  /usr/local/lib/python3.10/dist-packages/numpy/core/include
)

add_library(HB_HBMRuntime MODULE
  src/HB_HBMRuntime.cc
  src/HB_RuntimeUtils.cc
  src/HBMRuntimeBinding.cc
)

# add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
target_include_directories(HB_HBMRuntime PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(HB_HBMRuntime PRIVATE pybind11::module ${UCP_LIB} ${DNN_LIB})
set_target_properties(HB_HBMRuntime PROPERTIES OUTPUT_NAME "HB_HBMRuntime" PREFIX "")
# install compiled module into the same subdir inside wheel
install(TARGETS HB_HBMRuntime
        LIBRARY DESTINATION hbm_runtime)
# install pure-python files (scikit-build can do this automatically,
# but adding it is harmless and explicit)
install(DIRECTORY hbm_runtime/ DESTINATION hbm_runtime
        FILES_MATCHING PATTERN "*.py")

install(FILES hbm_runtime/py.typed
        DESTINATION hbm_runtime)
