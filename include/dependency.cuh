#pragma once

#include <cstdint>

#include "task.cuh"

namespace pk {

// Simple global dependency tracker.
// Tracks per-buffer completion counts so consumers can wait until a required
// number of producers have marked the buffer ready.
struct DependencyState {
  static constexpr int kMaxBuffers = 256;

  volatile int buffer_ready_count[kMaxBuffers];
  volatile int buffer_epoch[kMaxBuffers];

  __device__ inline void init() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < kMaxBuffers) {
      buffer_ready_count[tid] = 0;
      buffer_epoch[tid] = 0;
    }
  }

  __device__ inline void mark_ready(int buffer_id, int epoch) {
    if (buffer_id != 0) {
      atomicAdd((int*)&buffer_ready_count[buffer_id], 1);
      atomicMax((int*)&buffer_epoch[buffer_id], epoch);
    }
  }

  __device__ inline int get_ready_count(int buffer_id) const {
    if (buffer_id == 0)
      return 0;
    return atomicAdd((int*)&buffer_ready_count[buffer_id], 0);
  }

  __device__ inline bool is_ready(const Task& task) const {
    if (task.header.buffer_read_id == 0) {
      return true; // No dependency declared
    }
    int ready = get_ready_count(task.header.buffer_read_id);
    int epoch = atomicAdd((int*)&buffer_epoch[task.header.buffer_read_id], 0);
    return ready >= static_cast<int>(task.header.wait_count) &&
           epoch >= static_cast<int>(task.header.read_epoch);
  }
};

} // namespace pk
