# messing w meson+ninja...
# meson setup build
# meson compile -C build
project(
  'pkt',
  ['cpp', 'cuda'],
  default_options: ['cpp_std=c++20', 'cuda_std=c++20'],
)
add_project_link_arguments('-cudart', 'shared', language: 'cuda')

executable('vadd', ['src/vadd.cu'])
executable('main',   ['src/main.cpp'])
