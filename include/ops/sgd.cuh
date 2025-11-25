#pragma once

#include <cstdio>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

struct SGDUpdateArgs {
  float* W;
  const float* dW;
  float lr;
  int size;
};

template <> struct OpTraits<OpCode::SGDUpdate> {
  using Args = SGDUpdateArgs;

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

    for (int i = tid; i < args.size; i += total_threads) {
      args.W[i] -= args.lr * args.dW[i];
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
