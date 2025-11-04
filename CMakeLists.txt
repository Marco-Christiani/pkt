cmake_minimum_required(VERSION 3.26)
project(pkt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Generate the `compile_commands.json` file.
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
set(CMAKE_CUDA_ARCHITECTURES 86) # for targeting

# suppress deprecated-arch warnings
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

# -------------------------------------------------------------
# Executable targets
# -------------------------------------------------------------
add_executable(vadd src/vadd.cu)
add_executable(phasebits_demo src/phasebits_demo.cpp)
add_executable(phasebits_demo_threaded src/phasebits_demo_threaded.cpp)
add_executable(mlp_cpu src/mlp_cpu.cpp)
add_executable(mlp src/mlp.cu)

# target_link_libraries(vadd PRIVATE CUDA::cublas CUDA::cudart)

