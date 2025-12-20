#pragma once

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

// Computes weight gradient: dW = sum_b dy_b @ x_b^T
// dy: [batch, m]
// x: [batch, n]
// dW: [m, n] row-major (accumulated)
struct LinearBackwardArgs {
  const float* dy;
  const float* x;
  float* dW;
  int batch;
  int m;
  int n;
};

template <> struct OpTraits<OpCode::LinearBackward> {
  using Args = LinearBackwardArgs;

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
    int total_elements = args.m * args.n;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
      int row = idx / args.n;
      int col = idx % args.n;
      float grad = 0.0f;
      for (int b = 0; b < args.batch; ++b) {
        float dyb = args.dy[b * args.m + row];
        float xb = args.x[b * args.n + col];
        grad += dyb * xb;
      }
      atomicAdd(&args.dW[idx], grad);
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
