project(
  'pkt',
  ['cpp', 'cuda'],
  default_options: [
    'cpp_std=c++20',
    'cuda_std=c++20',
  ],
)

cuda = meson.get_compiler('cuda')
cuda_args = [
  '--expt-relaxed-constexpr',
  '--use_fast_math',
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
executable(
  'mlp_cute',
  ['src/mlp_cute_megakernel.cu'],
  link_args: ['-lcuda', '-lcudart'],
)

