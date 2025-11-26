#pragma once
#include <cstdio>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

// Computes MSE loss and gradient
// L = sum((y - target)^2) / m
// dy = 2 * (y - target) / m
struct MSELossArgs {
  const float* y;      // [m]
  const float* target; // [m]
  float* dy;           // [m]
  float* loss;         // [1]
  int m;
};

template <> struct OpTraits<OpCode::MSELoss> {
  using Args = MSELossArgs;

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
    float partial_loss = 0.0f;

    for (int i = tid; i < args.m; i += total_threads) {
      float diff = args.y[i] - args.target[i];
      args.dy[i] = 2.0f * diff / float(args.m);
      partial_loss += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      partial_loss += __shfl_down_sync(0xffffffff, partial_loss, offset);
    }

    if (lane == 0) {
      atomicAdd(args.loss, partial_loss / float(args.m));
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
