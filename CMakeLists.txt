cmake_minimum_required(VERSION 4.1)
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

message(
  DEBUG
  "CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}"
)

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

# -------------------------------------------------------------
# Exe tgts
# -------------------------------------------------------------
file(GLOB FILES examples/*.cu)
foreach(FILE_PATH ${FILES})
  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
  message("Building: ${FILE_PATH}")
  add_executable(${FILE_NAME} ${FILE_PATH})
  # target_link_libraries(${FILE_NAME} PRIVATE nanobench)
  # target_link_libraries(vadd PRIVATE CUDA::cublas CUDA::cudart)
  # target_link_libraries(${FILE_NAME} PRIVATE cuda cudart)
  target_link_libraries(${FILE_NAME} PRIVATE pk cuda cudart)
endforeach()

