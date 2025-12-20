#pragma once

#include <cstdio>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

// Computes y = W @ x for a batch of inputs
// W: [m, n] row-major
// x: [batch, n]
// y: [batch, m]
struct LinearForwardArgs {
  const float* W;
  const float* x;
  float* y;
  int batch;
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
    int total_rows = args.batch * args.m;
    for (int idx = tid; idx < total_rows; idx += total_threads) {
      int b = idx / args.m;
      int row = idx % args.m;
      float sum = 0.0f;
      const float* x_row = args.x + b * args.n;
      for (int col = 0; col < args.n; ++col) {
        sum += args.W[row * args.n + col] * x_row[col];
      }
      args.y[b * args.m + row] = sum;
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
