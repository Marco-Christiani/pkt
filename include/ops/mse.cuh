#pragma once
#include <cstdio>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

// Computes MSE loss and gradient
// L = sum((y - target)^2) / (m * batch)
// dy = 2 * (y - target) / (m * batch)
struct MSELossArgs {
  const float* y;      // [batch, m]
  const float* target; // [batch, m]
  float* dy;           // [batch, m]
  float* loss;         // [1]
  int batch;
  int m;
};

template <> struct OpTraits<OpCode::MSELoss> {
  using Args = MSELossArgs;

  __device__ static void load(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                              unsigned long long* page_reuse_hits,
                              unsigned long long* page_refill_count) {
    (void)args;
    (void)br;
    (void)slot_idx;
    (void)lane;
    (void)page_reuse_hits;
    (void)page_refill_count;
  }

  __device__ static void compute(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                                 int compute_warp_idx, int num_compute_warps) {
    (void)br;
    (void)slot_idx;

    int tid = compute_warp_idx * 32 + lane;
    int total_threads = num_compute_warps * 32;
    float partial_loss = 0.0f;
    const int elems = args.batch * args.m;
    const float scale = 2.0f / float(args.batch * args.m);

    for (int i = tid; i < elems; i += total_threads) {
      float diff = args.y[i] - args.target[i];
      args.dy[i] = scale * diff;
      partial_loss += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      partial_loss += __shfl_down_sync(0xffffffff, partial_loss, offset);
    }

    if (lane == 0) {
      atomicAdd(args.loss, partial_loss / float(elems));
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
