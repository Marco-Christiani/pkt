#pragma once

#include <cstdint>

#include <cuda/atomic>

#include "task.cuh"

namespace pk {

// Simple global dependency tracker.
// Tracks per-buffer completion counts so consumers can wait until a required
// number of producers have marked the buffer ready.
struct DependencyState {
  static constexpr int kMaxBuffers = 256;

  // Backed by plain ints so Runtime can reset with cudaMemset().
  int buffer_ready_count[kMaxBuffers];
  int buffer_epoch[kMaxBuffers];

  __device__ inline void init() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < kMaxBuffers) {
      buffer_ready_count[tid] = 0;
      buffer_epoch[tid] = 0;
    }
  }

  __device__ inline void mark_ready(int buffer_id, int epoch) {
    if (buffer_id != 0) {
#if PK_RELAX_PUBLISH_FENCE
      // Publish protocol experiment:
      // - caller omits __threadfence()
      // - producer stores epoch then performs a release increment on the ready counter
      // - consumers poll using acquire loads
      cuda::atomic_ref<int, cuda::thread_scope_device> epoch_ref(buffer_epoch[buffer_id]);
      // Keep epoch monotonic even if tasks complete out-of-order.
      int cur = epoch_ref.load(cuda::memory_order_relaxed);
      while (cur < epoch &&
             !epoch_ref.compare_exchange_weak(cur, epoch, cuda::memory_order_relaxed)) {
      }

      cuda::atomic_ref<int, cuda::thread_scope_device> count_ref(buffer_ready_count[buffer_id]);
      count_ref.fetch_add(1, cuda::memory_order_release);
#else
      atomicAdd((int*)&buffer_ready_count[buffer_id], 1);
      atomicMax((int*)&buffer_epoch[buffer_id], epoch);
#endif
    }
  }

  __device__ inline int get_ready_count(int buffer_id) const {
    if (buffer_id == 0)
      return 0;
#if PK_DEP_POLL_ATOMIC_LOAD
    cuda::atomic_ref<const int, cuda::thread_scope_device> ref(buffer_ready_count[buffer_id]);
    return ref.load(cuda::memory_order_acquire);
#else
    return atomicAdd((int*)&buffer_ready_count[buffer_id], 0);
#endif
  }

  __device__ inline bool is_ready(const Task& task) const {
    if (task.header.buffer_read_id == 0) {
      return true; // No dependency declared
    }
    const int buffer_id = task.header.buffer_read_id;
    int ready = get_ready_count(buffer_id);
#if PK_DEP_POLL_ATOMIC_LOAD
    cuda::atomic_ref<const int, cuda::thread_scope_device> epoch_ref(buffer_epoch[buffer_id]);
    int epoch = epoch_ref.load(cuda::memory_order_acquire);
#else
    int epoch = atomicAdd((int*)&buffer_epoch[buffer_id], 0);
#endif
    return ready >= static_cast<int>(task.header.wait_count) &&
           epoch >= static_cast<int>(task.header.read_epoch);
  }
};

} // namespace pk
