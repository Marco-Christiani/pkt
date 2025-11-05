cmake_minimum_required(VERSION 3.26)
project(pkt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

# -------------------------------------------------------------
# CUDA runtime linking:
# - force shared cudart (otherwise CMake may try static -> missing libcudadevrt.a on Nix)
# -------------------------------------------------------------
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")        # prevents -lcudadevrt / -lcudart_static issues
# set(CMAKE_CUDA_ARCHITECTURES 89) # for targeting
# set(CMAKE_CUDA_ARCHITECTURES 80;86;89)
set(CMAKE_CUDA_ARCHITECTURES 86) # for targeting

# suppress deprecated-arch warnings
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")

# -------------------------------------------------------------
# Libs
# -------------------------------------------------------------
add_library(megakernel_builder src/megakernel_builder.cpp)
set_target_properties(megakernel_builder PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(megakernel_builder PUBLIC src)

# -------------------------------------------------------------
# Exe tgts
# -------------------------------------------------------------
add_executable(vadd src/vadd.cu)
add_executable(phasebits_demo src/phasebits_demo.cpp)
add_executable(phasebits_demo_threaded src/phasebits_demo_threaded.cpp)
add_executable(mlp_cpu src/mlp_cpu.cpp)
add_executable(mlp src/mlp.cu)
add_executable(linear_train_nowarpspec src/linear_train_nowarpspec.cu)
add_executable(linear_train_warp_spec src/linear_train_warp_spec.cu)
add_executable(train_mlp_pk src/train_mlp_pk.cu)
add_executable(test_builder src/test_builder.cu)
add_executable(mlp_megakernel src/mlp_megakernel.cu)
add_executable(mlp_train_megakernel src/mlp_train_megakernel.cu)
add_executable(mlp_train_simple src/mlp_train_simple.cu)

message(STATUS "CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")

# dir w/ cuda_bf16.h
# find_path(CUDA_INCLUDE_PATH cuda_bf16.h PATHS ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
# message(STATUS "CUDA_INCLUDE_PATH (with cuda_bf16.h): ${CUDA_INCLUDE_PATH}")
#
# if(CUDA_INCLUDE_PATH)
#   target_compile_definitions(test_builder PRIVATE CUDA_INCLUDE_DIR="${CUDA_INCLUDE_PATH}")
#   target_compile_definitions(mlp_megakernel PRIVATE CUDA_INCLUDE_DIR="${CUDA_INCLUDE_PATH}")
# else()
#   message(FATAL_ERROR "Could not find CUDA include directory with cuda_bf16.h")
# endif()
#
# target_link_libraries(test_builder PRIVATE megakernel_builder ${CUDA_DRIVER_LIBRARY} ${NVRTC_LIBRARY})
# target_link_libraries(mlp_megakernel PRIVATE megakernel_builder ${CUDA_DRIVER_LIBRARY} ${NVRTC_LIBRARY})
#
# target_link_libraries(vadd PRIVATE CUDA::cublas CUDA::cudart)


# ----------------------------------------------------------------------------
# CUTLASS / CUTE
# ----------------------------------------------------------------------------
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass")
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/include/cute)

message(STATUS "CUTE include path: ${CUTLASS_DIR}/include/cute")

# ----------------------------------------------------------------------------
# cute exes
# ----------------------------------------------------------------------------
add_executable(mlp_cute src/mlp_cute_megakernel.cu)
target_link_libraries(mlp_cute PRIVATE cuda cudart)
target_include_directories(mlp_cute PRIVATE ${CUTLASS_DIR}/include)
target_include_directories(mlp_cute PRIVATE ${CUTLASS_DIR}/include/cute)
