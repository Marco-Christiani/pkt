#pragma once

#include <cstdint>

#include "config.cuh"
#include "task.cuh"

namespace pk {

// Global task queue shared across all blocks
// Uses atomic chunking for less contention
struct GlobalQueue {
  Task* tasks{nullptr};     // Device pointer to task array
  int total{0};             // Total number of tasks
  int* next_index{nullptr}; // Device pointer to atomic counter
};

// Chunked allocation from global queue
// The idea is instead of atomicAdd(counter, 1) per task (200 cycles each)
//   we atomicAdd(counter, chunk) once per chunk (~6 cycles per task).
//
// E.g., 80 SMs, 10k tasks
//   Per-task atomic: 80 SMs * 10k atomics
//   Chunked atomic: 80 SMs * (10k/32) atomics
template <int ChunkSize = Config::kChunkSize>
__device__ inline void dequeue_chunk(GlobalQueue* queue, int& chunk_begin, int& chunk_end) {
  const int start = atomicAdd(queue->next_index, ChunkSize);  // allocates ChunkSize tasks

  chunk_begin = start;
  chunk_end = (start + ChunkSize <= queue->total) ? (start + ChunkSize) : queue->total;
}

// Get pointer to specific task (after dequeuing chunk)
__device__ inline Task* get_task(GlobalQueue* queue, int index) {
  if (index >= queue->total) {
    return nullptr;
  }
  return &queue->tasks[index];
}

} // namespace mk
