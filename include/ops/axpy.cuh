#pragma once

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"

namespace pk {

struct AxpyArgs {
  const float* x;  // 8 bytes, offset 0 (aligned)
  float* y;        // 8 bytes, offset 8
  float a;         // 4 bytes, offset 16
  int n;           // 4 bytes, offset 20
  // 24b
};

template <> struct OpTraits<OpCode::Axpy> {
  using Args = AxpyArgs;

  // TODO: prefetch data to SMEM (eg prefect x[] and y[] to br.pages)
  //   for now, skip prefetch, compute accesses global memory directly
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

    const int threads_per_group = num_compute_warps * Config::kWarpSize;
    const int thread_id = compute_warp_idx * Config::kWarpSize + lane;

    for (int i = thread_id; i < args.n; i += threads_per_group) {
      const float xi = args.x[i];
      const float yi = args.y[i];
      args.y[i] = args.a * xi + yi;
    }
  }

  // TODO: smem writeback to gbm. for now, compute wrote directly to y[], nothing to do here
  __device__ static void store(const Args& args, BlockRuntime& br, int slot_idx, int lane) {
    (void)args;
    (void)br;
    (void)slot_idx;
    (void)lane;
  }
};

} // namespace mk
