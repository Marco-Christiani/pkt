#pragma once

#include <cstdint>
#include <cstdio>

#include <cuda/atomic>

#include "config.cuh"
#include "task.cuh"

namespace pk {

// Pipeline slot phase state machine
enum class SlotPhase : int {
  Empty = 0, // Slot is free, controller can fill
  Loading,   // Loader warp is fetching data
  Loaded,    // Data ready, compute can begin
  Computing, // Compute warps are processing
  Computed,  // Compute done, ready to store
  Storing,   // Storer warp is writing results
  Done,      // Complete, ready to be marked Empty
};

// Holds one in-flight task
// NOTE: will 128-byte alignment prevent false sharing? eh, why not. TODO: profile
struct alignas(128) PipelineSlot {
  Task task;
  cuda::atomic<int, cuda::thread_scope_block> phase;
  cuda::atomic<int, cuda::thread_scope_block> task_seq;
  cuda::atomic<int, cuda::thread_scope_block> compute_warps_done;

  __device__ inline void reset() {
    phase.store((int)SlotPhase::Empty, cuda::memory_order_relaxed);
    task_seq.store(-1, cuda::memory_order_relaxed);
    compute_warps_done.store(0, cuda::memory_order_relaxed);
  }
};

// Wait for slot phase to reach target using atomic acquire semantics
__device__ inline void wait_for_phase(cuda::atomic<int, cuda::thread_scope_block>* phase, SlotPhase target) {
  int backoff_ns = 10;
  int target_val = (int)target;
  while (phase->load(cuda::memory_order_acquire) != target_val) {
    __nanosleep(backoff_ns);
    backoff_ns = (backoff_ns < 500) ? (backoff_ns * 2) : 1000;
  }
}

// Ring buffer index advance
__device__ inline int ring_advance(int index) {
  // bit-mask if kPipelineStages is power of 2
  if constexpr ((Config::kPipelineStages & (Config::kPipelineStages - 1)) == 0) {
    return (index + 1) & (Config::kPipelineStages - 1);
  } else {
    return (index + 1) % Config::kPipelineStages;
  }
}

} // namespace pk
