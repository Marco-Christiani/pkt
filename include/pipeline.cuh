#pragma once

#include <cstdint>
#include <cstdio>

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
  volatile SlotPhase phase;
  volatile int task_seq;
  volatile int compute_warps_done;

  // TODO: paging support
  // Will hold logical->physical page mappings
  // int logical_to_physical[Config::kMaxLogicalPages];
  // int page_epoch;

  __device__ inline void reset() {
    phase = SlotPhase::Empty;
    task_seq = -1;
    compute_warps_done = 0;
  }
};

__device__ inline void fence_release() {
  __threadfence_block();
}

__device__ inline void fence_acquire() {
  __threadfence_block();
}

// Phase transition with release semantics (writer)
__device__ inline void phase_transition_release(volatile SlotPhase* phase, SlotPhase new_phase) {
  __threadfence_block(); // release fence: publish earlier writes
  *phase = new_phase;    // visible after the release
}

// Phase read with acquire semantics (reader)
__device__ inline SlotPhase phase_load_acquire(volatile SlotPhase* phase) {
  SlotPhase p = *phase;  // read phase first
  __threadfence_block(); // acquire fence: subsequent loads observe published writes
  return p;
}

// Adaptive spin-wait on phase transition - exponential
template <typename Pred> __device__ inline void spin_adaptive(Pred predicate) {
  int backoff_ns = 10;
  while (!predicate()) {
    __nanosleep(backoff_ns);
    backoff_ns = (backoff_ns < 500) ? (backoff_ns * 2) : 1000; // cap at 1000ns
  }
}

// __device__ inline void wait_for_phase(volatile SlotPhase* phase, SlotPhase target) {
//   spin_adaptive([&]() { return phase_load_acquire(phase) == target; })
// }
__device__ inline void wait_for_phase(volatile SlotPhase* phase, SlotPhase target) {
  int spins = 0;
  spin_adaptive([&]() {
    spins++;
    if (spins % 10000 == 0) {
      printf("wait_for_phase: spins=%d, phase=%d, target=%d\n", spins, (int)*phase, (int)target);
    }
    return phase_load_acquire(phase) == target;
  });
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
