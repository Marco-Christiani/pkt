#pragma once

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

// Zeros a region of memory
struct ZeroMemoryArgs {
  float* ptr; // [size]
  int size;
};

template <> struct OpTraits<OpCode::ZeroMemory> {
  using Args = ZeroMemoryArgs;

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

    for (int i = tid; i < args.size; i += total_threads) {
      args.ptr[i] = 0.0f;
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
