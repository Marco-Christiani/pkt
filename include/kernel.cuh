#pragma once

#include "block_runtime.cuh"
#include "config.cuh"
#include "dispatch.cuh"
#include "pipeline.cuh"
#include "queue.cuh"

namespace pk {

// Controller warp fills pipeline with tasks from global queue.
__device__ void controller_loop(BlockRuntime& br, int lane) {
  int slot_idx = 0;

  while (true) {
    // Allocate chunk of tasks from global queue and broadcast to all lanes
    int chunk_begin, chunk_end;
    if (lane == 0) {
      dequeue_chunk(&br.queue, chunk_begin, chunk_end);
    }
    chunk_begin = __shfl_sync(0xffffffff, chunk_begin, 0);
    chunk_end = __shfl_sync(0xffffffff, chunk_end, 0);

    if (chunk_begin >= chunk_end) {
      break;
    }

    // Fill slots with tasks from this chunk
    for (int task_idx = chunk_begin; task_idx < chunk_end; ++task_idx) {
      PipelineSlot& slot = br.slot(slot_idx);
      wait_for_phase(&slot.phase, SlotPhase::Empty);

      if (lane == 0) { // copy into slot
        Task* task = get_task(&br.queue, task_idx);
        if (task != nullptr) {
          slot.task = *task;
          slot.compute_warps_done = 0;
        }
      }
      __syncwarp();

      if (lane == 0) { // transition to loading
        phase_transition_release(&slot.phase, SlotPhase::Loading);
      }

      slot_idx = br.next_slot(slot_idx);
    }
  }

  if (lane == 0) { // signal done to other warps
    fence_acquire();
    *br.done = 1;
  }
}

// Loader warp loads data for tasks (TODO: prefetch to SMEM?)
__device__ void loader_loop(BlockRuntime& br, int lane) {
  int slot_idx = 0;

  while (true) {
    PipelineSlot& slot = br.slot(slot_idx);

    // Wait for Loading phase
    SlotPhase phase = phase_load_acquire(&slot.phase);
    if (phase != SlotPhase::Loading) {
      if (phase == SlotPhase::Empty) {
        // Check termination flag
        if (*br.done) {
          return;
        }
        __nanosleep(100);
        continue;
      }
      slot_idx = br.next_slot(slot_idx);
      continue;
    }

    dispatch_load(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {  // Transition to Loaded
      phase_transition_release(&slot.phase, SlotPhase::Loaded);
    }

    slot_idx = br.next_slot(slot_idx);
  }
}

// Compute warps: process tasks in parallel
__device__ void compute_loop(BlockRuntime& br, int lane, int compute_warp_idx,
                             int num_compute_warps) {
  int slot_idx = 0;

  while (true) {
    PipelineSlot& slot = br.slot(slot_idx);

    while (true) { // wait for Loaded or Computing
      SlotPhase phase = phase_load_acquire(&slot.phase);
      if (phase == SlotPhase::Loaded || phase == SlotPhase::Computing) {
        break;
      }
      if (phase == SlotPhase::Empty && *br.done) {
        return;
      }
      __nanosleep(100);
    }

    if (compute_warp_idx == 0 && lane == 0) { // first warp transitions to Computing
      if (slot.phase == SlotPhase::Loaded) {  // transition if still Loaded, avoid double transition
        phase_transition_release(&slot.phase, SlotPhase::Computing);
      }
    }

    dispatch_compute(slot.task, br, slot_idx, lane, compute_warp_idx, num_compute_warps);
    __syncwarp();

    if (lane == 0) { // done++
      int done = atomicAdd(const_cast<int*>(&slot.compute_warps_done), 1);

      if (done + 1 == num_compute_warps) { // last warp transition to Computed
        phase_transition_release(&slot.phase, SlotPhase::Computed);
      }
    }

    slot_idx = br.next_slot(slot_idx);
  }
}

// Storer warp writes results back to GBM
__device__ void storer_loop(BlockRuntime& br, int lane) {
  int slot_idx = 0;

  while (true) {
    PipelineSlot& slot = br.slot(slot_idx);

    SlotPhase phase = phase_load_acquire(&slot.phase);
    if (phase != SlotPhase::Computed) { // wait for Computed
      if (phase == SlotPhase::Empty) {
        if (*br.done) {
          return;
        }
        __nanosleep(100);
        continue;
      }
      slot_idx = br.next_slot(slot_idx);
      continue;
    }

    if (lane == 0) { // lane 0 transition to Storing
      phase_transition_release(&slot.phase, SlotPhase::Storing);
    }

    dispatch_store(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) { // Mark slot as empty for reuse
      phase_transition_release(&slot.phase, SlotPhase::Empty);
    }

    slot_idx = br.next_slot(slot_idx);
  }
}

// Blocks spin, processing tasks from the global queue.
// Warps specializations: controller, loader, computer, storer
//
// Pipeline flow:
//   1. Controller: dequeues tasks from global queue -> fills slots
//   2. Loader: waits for filled slots -> loads data (Empty -> Loaded)
//   3. Compute: waits for loaded slots -> processes (Loaded -> Computed)
//   4. Storer: waits for computed slots -> stores results (Computed -> Empty)
__global__ void persistent_kernel(GlobalQueue queue) {
  __shared__ PipelineSlot slots[Config::kPipelineStages];
  __shared__ int done_flag;
  __shared__ int tasks_remaining;

  // Allocate pages for tile data staging (use size 1 when disabled to avoid zero-sized array)
  __shared__ Page pages[Config::kMaxLogicalPages > 0 ? Config::kMaxLogicalPages : 1];

  // Set up block runtime
  BlockRuntime br;
  br.slots = slots;
  br.num_slots = Config::kPipelineStages;
  br.queue = queue;
  br.done = &done_flag;
  br.pages = Config::kMaxLogicalPages > 0 ? pages : nullptr;
  br.num_pages = Config::kMaxLogicalPages;

  const int tid = threadIdx.x;
  if (tid == 0) {
    done_flag = 0;
    tasks_remaining = queue.total;
  }
  if (tid < br.num_slots) {
    br.slot(tid).reset();
  }
  __syncthreads();

  const int wid = warp_id();
  const int lane = lane_id();

  if (is_controller(wid, br.roles)) {
    controller_loop(br, lane);
  } else if (is_loader(wid, br.roles)) {
    loader_loop(br, lane);
  } else if (is_storer(wid, br.roles)) {
    storer_loop(br, lane);
  } else if (is_compute(wid, br.roles)) {
    const int compute_idx = compute_warp_index(wid, br.roles);
    compute_loop(br, lane, compute_idx, br.roles.num_compute);
  }
}

} // namespace mk
