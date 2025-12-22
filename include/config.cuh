#pragma once

#include <cstdint>

namespace pk {

// Core runtime settings
struct Config {
  static constexpr int kWarpSize = 32;
  static constexpr int kPipelineStages = 4; // Ring buffer depth

  // Task argument payload size
  static constexpr int kArgBytes = 64;

  // Warp roles within a block
  static constexpr int kControllerWarp = 0;
  static constexpr int kLoaderWarp = 1;
  static constexpr int kStorerWarp = 2;
  static constexpr int kFirstComputeWarp = 3;
  static constexpr int kNumComputeWarps = 4;

  // Derived: total warps and threads per block
  static constexpr int kTotalWarps = kFirstComputeWarp + kNumComputeWarps;
  static constexpr int kThreadsPerBlock = kTotalWarps * kWarpSize;

  // Global queue chunk size for atomic allocation
  // Larger chunks reduce contention but increase load imbalance
  static constexpr int kChunkSize = 32;

  // Paging configuration for SMEM staging
  // Enable when using tiled ops that stage into SMEM pages.
  static constexpr int kPagesPerSlot = 2;     // two pages for ping-pong
  static constexpr int kMaxLogicalPages = kPipelineStages * kPagesPerSlot;
  static constexpr int kPageWords = 1024;     // 4 KB per page to hold x tiles up to 1024
  static constexpr int kPageBytes = kPageWords * sizeof(int);

  // GEMM friendly chunking
  static constexpr int kGemmKChunk = 16;
  static constexpr int kLinearFwdThreadNFrag = 4;
  static constexpr int kLinearBwdThreadNFrag = 4;

#if defined(PK_USE_TENSOR_CORES)
  static constexpr bool kUseTensorCores = true;
#else
  static constexpr bool kUseTensorCores = false;
#endif
};

// Experiment toggles (profiling / framework)
// Keep defaults conservative and correctness-preserving.
#ifndef PK_DEP_POLL_ATOMIC_LOAD
#define PK_DEP_POLL_ATOMIC_LOAD 1
#endif

#ifndef PK_CONTROLLER_SKIP_BLOCKED
#define PK_CONTROLLER_SKIP_BLOCKED 0
#endif

#ifndef PK_CLAIMED_READY_FIRST
#define PK_CLAIMED_READY_FIRST 0
#endif

#ifndef PK_SEGMENT_GATE
#define PK_SEGMENT_GATE 0
#endif

#ifndef PK_SEGMENT_WINDOW_SIZE
#define PK_SEGMENT_WINDOW_SIZE 4
#endif

#ifndef PK_SEGMENT_WINDOW_MAX
#define PK_SEGMENT_WINDOW_MAX 16
#endif

#ifndef PK_PROFILE_RANGES
#define PK_PROFILE_RANGES 0
#endif

#ifndef PK_RELAX_PUBLISH_FENCE
#define PK_RELAX_PUBLISH_FENCE 0
#endif

#ifndef PK_TASK_ACCOUNTING
#define PK_TASK_ACCOUNTING 0
#endif

#ifndef PK_PERF_COUNTERS
#define PK_PERF_COUNTERS 0
#endif

// Warp role helpers
struct WarpRoles {
  int controller{Config::kControllerWarp};
  int loader{Config::kLoaderWarp};
  int storer{Config::kStorerWarp};
  int first_compute{Config::kFirstComputeWarp};
  int num_compute{Config::kNumComputeWarps};
};

// Basic device helpers
__device__ inline int lane_id() {
  return threadIdx.x & (Config::kWarpSize - 1);
}

__device__ inline int warp_id() {
  return threadIdx.x / Config::kWarpSize;
}

__device__ inline bool is_controller(int wid, const WarpRoles& roles) {
  return wid == roles.controller;
}

__device__ inline bool is_loader(int wid, const WarpRoles& roles) {
  return wid == roles.loader;
}

__device__ inline bool is_storer(int wid, const WarpRoles& roles) {
  return wid == roles.storer;
}

__device__ inline bool is_compute(int wid, const WarpRoles& roles) {
  return wid >= roles.first_compute && wid < roles.first_compute + roles.num_compute;
}

__device__ inline int compute_warp_index(int wid, const WarpRoles& roles) {
  return wid - roles.first_compute;
}

} // namespace pk
