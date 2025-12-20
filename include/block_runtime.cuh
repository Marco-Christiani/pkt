#pragma once

#include "config.cuh"
#include "dependency.cuh"
#include "pipeline.cuh"
#include "queue.cuh"

namespace pk {

// For SMEM tile staging
// When paging is enabled (Config::kMaxLogicalPages > 0), holds tile data in SMEM
struct Page {
  int data[Config::kPageWords > 0 ? Config::kPageWords : 1];

  // Typed view of page data
  template <typename T> __device__ inline T* as() {
    static_assert(sizeof(T) % 4 == 0, "T must be 4-byte aligned");
    return reinterpret_cast<T*>(data);
  }

  template <typename T> __device__ inline const T* as() const {
    static_assert(sizeof(T) % 4 == 0, "T must be 4-byte aligned");
    return reinterpret_cast<const T*>(data);
  }
};

// Per-block runtime state
// Execution context for all warps in a block
struct BlockRuntime {
  // SMEM
  PipelineSlot* slots{nullptr};
  int num_slots{Config::kPipelineStages};

  // Warp role configuration
  WarpRoles roles{};

  // Global work queue
  GlobalQueue queue{};

  // SMEM pages for data staging
  Page* pages{nullptr};
  int num_pages{0};

  // Termination flag - set by controller when all work is done
  volatile int* done{nullptr};

  // Global dependency tracking
  DependencyState* deps{nullptr};

  // Debug
  int tasks_processed{0};

  // Helpers
  __device__ inline PipelineSlot& slot(int index) { return slots[index]; }

  __device__ inline const PipelineSlot& slot(int index) const { return slots[index]; }

  __device__ inline int next_slot(int current) const { return ring_advance(current); }
};

} // namespace mk
