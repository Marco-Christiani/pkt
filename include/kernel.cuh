#pragma once

#include <cstdio>

#include "block_runtime.cuh"
#include "config.cuh"
#include "dispatch.cuh"
#include "pipeline.cuh"
#include "queue.cuh"

namespace pk {

__device__ void controller_loop(BlockRuntime& br, int lane) {
  int seq = 0; // global task sequence number

  while (true) {
    // Allocate chunk of tasks from global queue and broadcast to all lanes
    int chunk_begin, chunk_end;
    if (lane == 0) {
      dequeue_chunk(&br.queue, chunk_begin, chunk_end);
    }
    chunk_begin = __shfl_sync(0xffffffff, chunk_begin, 0);
    chunk_end = __shfl_sync(0xffffffff, chunk_end, 0);

    if (chunk_begin >= chunk_end)
      break;

    // Fill slots with tasks from this chunk
    for (int task_idx = chunk_begin; task_idx < chunk_end; ++task_idx, ++seq) {
      int slot_idx = seq % br.num_slots;
      PipelineSlot& slot = br.slot(slot_idx);

      // Wait for slot to be empty (storer finished with it)
      wait_for_phase(&slot.phase, SlotPhase::Empty);

      if (lane == 0) { // transition to loading
        Task* task = get_task(&br.queue, task_idx);
        slot.task = *task;
        slot.compute_warps_done = 0;
        slot.task_seq = seq; // Tag with sequence number
        __threadfence_block();
        slot.phase = SlotPhase::Loading;
      }
      __syncwarp();
    }
  }

  if (lane == 0) { // signal done to other warps

    __threadfence_block();
    *br.done = 1;
  }
}

__device__ void loader_loop(BlockRuntime& br, int lane) {
  int seq = 0; // Loader tracks which sequence it's waiting for

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Spin until this slot has our sequence AND is in Loading phase
    while (true) {
      int current_seq = slot.task_seq;
      SlotPhase phase = slot.phase;

      if (*br.done && current_seq < seq)
        return; // No more work coming

      if (current_seq == seq && phase == SlotPhase::Loading)
        break;

      __nanosleep(50);
    }

    dispatch_load(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {
      __threadfence_block();
      slot.phase = SlotPhase::Loaded;
    }

    seq++;
  }
}

__device__ void compute_loop(BlockRuntime& br, int lane, int compute_warp_idx,
                             int num_compute_warps) {
  int seq = 0;

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Wait for our sequence to be Loaded or Computing
    while (true) {
      int current_seq = slot.task_seq;
      SlotPhase phase = slot.phase;

      if (*br.done && current_seq < seq)
        return;

      if (current_seq == seq && (phase == SlotPhase::Loaded || phase == SlotPhase::Computing))
        break;

      __nanosleep(50);
    }

    // First compute warp transitions Loaded -> Computing
    if (compute_warp_idx == 0 && lane == 0) {
      SlotPhase expected = SlotPhase::Loaded;
      if (slot.phase == expected) {
        slot.phase = SlotPhase::Computing;
        __threadfence_block();
      }
    }
    __syncwarp();

    dispatch_compute(slot.task, br, slot_idx, lane, compute_warp_idx, num_compute_warps);
    __syncwarp();

    // Last compute warp transitions Computing -> Computed
    if (lane == 0) {
      int done = atomicAdd((int*)&slot.compute_warps_done, 1);
      if (done + 1 == num_compute_warps) {
        __threadfence_block();
        slot.phase = SlotPhase::Computed;
      }
    }

    seq++;
  }
}

__device__ void storer_loop(BlockRuntime& br, int lane) {
  int seq = 0;

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Wait for our sequence to be Computed
    while (true) {
      int current_seq = slot.task_seq;
      SlotPhase phase = slot.phase;

      if (*br.done && current_seq < seq)
        return;

      if (current_seq == seq && phase == SlotPhase::Computed)
        break;

      __nanosleep(50);
    }

    dispatch_store(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {
      slot.task_seq = -1; // Clear sequence
      __threadfence_block();
      slot.phase = SlotPhase::Empty;
    }

    seq++;
  }
}
__global__ void persistent_kernel(GlobalQueue queue) {
  __shared__ PipelineSlot slots[Config::kPipelineStages];
  __shared__ int done_flag;
  __shared__ int tasks_remaining;
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

} // namespace pk
