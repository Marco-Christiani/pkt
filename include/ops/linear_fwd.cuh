#pragma once

#include <cstdio>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

struct LinearForwardArgs {
  const float* W;
  const float* x;
  float* y;
  int m;
  int n;
};

template <> struct OpTraits<OpCode::LinearForward> {
  using Args = LinearForwardArgs;

  __device__ static void load(const Args& args, BlockRuntime& br, int slot_idx, int lane) {
    (void)args;
    (void)br;
    (void)slot_idx;
    (void)lane;
  }

  __device__ static void compute(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                                 int compute_warp_idx, int num_compute_warps) {
    (void)br;
    (void)slot_idx;

    int tid = compute_warp_idx * 32 + lane;
    int total_threads = num_compute_warps * 32;
    for (int row = tid; row < args.m; row += total_threads) {
      float sum = 0.0f;
      for (int col = 0; col < args.n; ++col) {
        sum += args.W[row * args.n + col] * args.x[col];
      }
      args.y[row] = sum;
    }
  }

  __device__ static void store(const Args& args, BlockRuntime& br, int slot_idx, int lane) {
    (void)args;
    (void)br;
    (void)slot_idx;
    (void)lane;
  }
};

} // namespace pk
