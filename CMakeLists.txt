cmake_minimum_required(VERSION 3.20)
project(pkt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared") # prevents -lcudadevrt /
# -lcudart_static issues set(CMAKE_CUDA_ARCHITECTURES 89) # for targeting
# set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
set(CMAKE_CUDA_ARCHITECTURES 86) # for targeting
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

message(DEBUG "CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: \
        ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")

# ----------------------------------------------------------------------------
# Profiling / framework experiment toggles (compile-time)
# ----------------------------------------------------------------------------
option(PK_PROFILE_RANGES "Enable device-side range timing counters" OFF)
option(PK_CONTROLLER_SKIP_BLOCKED "Controller skips blocked tasks instead of waiting" OFF)
option(PK_CLAIMED_READY_FIRST "Claimed ready-first controller scheduler" OFF)
option(PK_SEGMENT_GATE "Restrict claimed scheduler scan to phase segments" OFF)
set(PK_SEGMENT_WINDOW_SIZE "4" CACHE STRING "Segment gate active window size")
option(PK_RELAX_PUBLISH_FENCE "Replace __threadfence with release/acquire publish protocol" OFF)
option(PK_DEP_POLL_ATOMIC_LOAD "Poll dependencies with atomic acquire loads (no RMW)" ON)
option(PK_TASK_ACCOUNTING "Count tasks issued by controller (debug)" OFF)
option(PK_PERF_COUNTERS "Enable per-block persistent kernel perf counters" OFF)

# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG v4.3.11
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

# -------------------------------------------------------------
# Lib
# -------------------------------------------------------------
add_library(pk INTERFACE)
target_include_directories(pk INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(
  pk INTERFACE
  PK_PROFILE_RANGES=$<IF:$<BOOL:${PK_PROFILE_RANGES}>,1,0>
  PK_CONTROLLER_SKIP_BLOCKED=$<IF:$<BOOL:${PK_CONTROLLER_SKIP_BLOCKED}>,1,0>
  PK_CLAIMED_READY_FIRST=$<IF:$<BOOL:${PK_CLAIMED_READY_FIRST}>,1,0>
  PK_SEGMENT_GATE=$<IF:$<BOOL:${PK_SEGMENT_GATE}>,1,0>
  PK_SEGMENT_WINDOW_SIZE=${PK_SEGMENT_WINDOW_SIZE}
  PK_RELAX_PUBLISH_FENCE=$<IF:$<BOOL:${PK_RELAX_PUBLISH_FENCE}>,1,0>
  PK_DEP_POLL_ATOMIC_LOAD=$<IF:$<BOOL:${PK_DEP_POLL_ATOMIC_LOAD}>,1,0>
  PK_TASK_ACCOUNTING=$<IF:$<BOOL:${PK_TASK_ACCOUNTING}>,1,0>
  PK_PERF_COUNTERS=$<IF:$<BOOL:${PK_PERF_COUNTERS}>,1,0>)

# -------------------------------------------------------------
# Exe tgts
# -------------------------------------------------------------
file(GLOB FILES examples/*.cu)
foreach(file_path ${FILES})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${file_path})
  message("Building: ${file_path}")
  add_executable(${FILE_NAME} ${file_path})
  # target_link_libraries(${FILE_NAME} PRIVATE nanobench)
  # target_link_libraries(vadd PRIVATE CUDA::cublas CUDA::cudart)
  target_link_libraries(${FILE_NAME} PRIVATE pk cuda cudart)
endforeach()

# -------------------------------------------------------------
# Tests
# -------------------------------------------------------------
enable_testing()

file(GLOB TEST_FILES tests/*.cu)
foreach(test_path ${TEST_FILES})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" TEST_NAME ${test_path})
  message("Building test: ${test_path}")
  add_executable(${TEST_NAME} ${test_path})
  target_link_libraries(${TEST_NAME} PRIVATE pk cuda cudart)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
