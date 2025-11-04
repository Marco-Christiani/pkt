#!/bin/bash
# Auto-generate .clangd configuration for C++/CUDA projects

set -e

# prints repo root or exits non-zero if not in a git repo
repo_root() {
  git rev-parse --show-toplevel 2>/dev/null
}

cd_to_repo_root() {
  root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    tput setaf 1
    printf 'not inside a git repository\n'
    tput sgr0
    return 1
  }
  cd "$root" || return 1
}
self_path="$(realpath "$(dirname "$0")")"

echo "Generating .clangd configuration..."
echo "cd from $self_path to repo root @ $(repo_root)"

cd_to_repo_root

# Find cpp standard library paths
CXX_PATHS=$(echo | g++ -x c++ -E -v - 2>&1 | grep "^ /" | grep "c++" | head -2)

if [ -z "$CXX_PATHS" ]; then
  echo "ERROR: Could not find C++ standard library paths"
  echo "Make sure g++ is installed: which g++"
  exit 1
fi

echo "Found C++ paths:"
echo "$CXX_PATHS"

# For CUDA: Find CUDA headers
CUDA_PATH=""
if [ -d "/usr/local/cuda/include" ]; then
  CUDA_PATH="/usr/local/cuda/include"
elif [ -d "/opt/cuda/include" ]; then
  CUDA_PATH="/opt/cuda/include"
else
  # Try to find in Nix store or elsewhere
  CUDA_RUNTIME=$(find /nix/store /usr -name "cuda_runtime.h" 2>/dev/null | head -1)
  if [ -n "$CUDA_RUNTIME" ]; then
    CUDA_PATH=$(dirname "$CUDA_RUNTIME")
  fi
fi

if [ -n "$CUDA_PATH" ]; then
  echo "Found CUDA path: $CUDA_PATH"
  CUDA_INCLUDE="    - -I$CUDA_PATH"
  CUDA_SECTION="

---

# CUDA-specific configuration for .cu and .cuh files
If:
  PathMatch: [.*\\.cu, .*\\.cuh]
CompileFlags:
  Add:
    - --cuda-gpu-arch=sm_89
    - -xcuda
    - --no-cuda-version-check"
else
  echo "CUDA not found (this is OK for C++-only projects)"
  CUDA_INCLUDE=""
  CUDA_SECTION=""
fi

# Generate .clangd
cat >.clangd <<EOF
CompileFlags:
  CompilationDatabase: build
  Add:
$(echo "$CXX_PATHS" | sed 's/^ */    - -I/')
$CUDA_INCLUDE
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code*$CUDA_SECTION

Diagnostics:
  UnusedIncludes: None
  MissingIncludes: None
EOF

echo "Generated .clangd"
echo "Next: Generate compile_commands.json: cmake -B build && ln -sf build/compile_commands.json ."
