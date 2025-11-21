project(
  'pkt',
  ['cpp', 'cuda'],
  default_options: [
    'cpp_std=c++20',
    'cuda_std=c++20',
  ],
  meson_version: '>=1.4.0',
)

cuda = meson.get_compiler('cuda')
cuda_args = [
  '-Wno-deprecated-gpu-targets',
  '-arch=sm_86',
]

add_project_arguments(cuda_args, language: 'cuda')

add_project_link_arguments('-cudart', 'shared', language: 'cuda')

# Exe tgts
executable('vadd', ['src/vadd.cu'])
executable('phasebits_demo', ['src/phasebits_demo.cpp'])
executable('phasebits_demo_threaded', ['src/phasebits_demo_threaded.cpp'])
executable('mlp_cpu', ['src/mlp_cpu.cpp'])

# CUTLASS+CUTE tgt
cutes = files(
  # 'src/cute_mlp.cu',
  'src/cute_wmma.cu',
  'src/mlp_cute_megakernel.cu',
  # 'src/test_one_layer_cute.cu',
  'src/test_simple_cute_gemm.cu',
  'src/test_simple_cute_gemm2.cu',
)

foreach f : cutes
  message('Found cute src: ' + f.full_path())
  executable(
    f.full_path().split('/')[-1].split('.')[0],
    [f],
    # link_args: ['-lcuda', '-lcudart'],
  )
endforeach

# executable(
#   'mlp_cute',
#   ['src/mlp_cute_megakernel.cu'],
#   link_args: ['-lcuda', '-lcudart'],
# )

# executable(
#   'test_minimal_cute',
#   ['src/test_minimal_cute.cu'],
#   link_args: ['-lcuda', '-lcudart'],
# )

# executable(
#   'test_one_layer_cute',
#   ['src/test_one_layer_cute.cu'],
#   link_args: ['-lcuda', '-lcudart'],
# )

# executable(
#   'test_simple_cute_gemm',
#   ['src/test_simple_cute_gemm.cu'],
#   link_args: ['-lcuda', '-lcudart'],
# )

# executable(
#   'test_simple_cute_gemm2',
#   ['src/test_simple_cute_gemm2.cu'],
#   link_args: ['-lcuda', '-lcudart'],
# )

# nanobench = subproject('nanobench')
# executable('bench', 'bench.cpp', dependencies: [nanobench.dependency()])

# nanobench = subproject('nanobench')
nanobench_inc = include_directories('subprojects/nanobench/src/include')

executable('day1',
  ['warmups/day1.cu'],
  include_directories: [nanobench_inc],
)
