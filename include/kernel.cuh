#pragma once

#include <cstdio>

#include "block_runtime.cuh"
#include "config.cuh"
#include "dependency.cuh"
#include "dispatch.cuh"
#include "pipeline.cuh"
#include "queue.cuh"

#ifndef PK_DEBUG
#define PK_DEBUG 0
#endif

#if PK_DEBUG
#define PK_PRINTF(...) printf(__VA_ARGS__)
#else
#define PK_PRINTF(...) ((void)0)
#endif

namespace pk {

__device__ void controller_loop(BlockRuntime& br, int lane) {
  int seq = 0; // global task sequence number

  while (true) {
    // Allocate chunk of tasks from global queue and broadcast to all lanes
    int chunk_begin = 0, chunk_end = 0;
    if (lane == 0) {
      dequeue_chunk(&br.queue, chunk_begin, chunk_end);
    }
    __syncwarp();  // Ensure lane 0 completes before shuffle
    chunk_begin = __shfl_sync(0xffffffff, chunk_begin, 0);
    chunk_end = __shfl_sync(0xffffffff, chunk_end, 0);

    if (lane == 0 && seq == 0) {
      PK_PRINTF("BLK%d CONTROLLER: seq=%d chunk [%d, %d)\n", blockIdx.x, seq, chunk_begin, chunk_end);
    }

    if (chunk_begin >= chunk_end)
      break;

    // Fill slots with tasks from this chunk
    for (int task_idx = chunk_begin; task_idx < chunk_end; ++task_idx, ++seq) {
      int slot_idx = seq % br.num_slots;
      PipelineSlot& slot = br.slot(slot_idx);

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: processing seq=%d task_idx=%d slot=%d phase=%d\n",
                  blockIdx.x, seq, task_idx, slot_idx, (int)slot.phase);
      }

      // Wait for slot to be empty (storer finished with it)
      wait_for_phase(&slot.phase, SlotPhase::Empty);

      Task* task = get_task(&br.queue, task_idx);

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: got task %d, opcode=%d, buf_write=%d\n",
                  blockIdx.x, task_idx, (int)task->header.opcode, task->header.buffer_write_id);
      }

      // Wait for declared dependencies to be satisfied
      if (lane == 0 && br.deps) {
        int spins = 0;
        while (!br.deps->is_ready(*task)) {
          if (spins == 0 && task_idx < 5) {
            int ready = br.deps->get_ready_count(task->header.buffer_read_id);
            PK_PRINTF("CONTROLLER: task %d waiting for buffer %d: ready=%d wait=%d\n",
                      task_idx, task->header.buffer_read_id, ready, task->header.wait_count);
          }
          spins++;
          __nanosleep(100);
        }
      }
      __syncwarp();

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: task %d deps satisfied, starting\n", blockIdx.x, task_idx);
      }

      if (lane == 0) { // transition to loading
        slot.task = *task;
        slot.compute_warps_done.store(0, cuda::memory_order_relaxed);

        // Publish task with release semantics so loader/compute/storer see all writes
        slot.task_seq.store(seq, cuda::memory_order_release);
        slot.phase.store((int)SlotPhase::Loading, cuda::memory_order_release);
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
      // Acquire semantics ensure we see controller's task initialization
      int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
      int phase = slot.phase.load(cuda::memory_order_acquire);

      if (*br.done && current_seq < seq)
        return; // No more work coming

      if (current_seq == seq && phase == (int)SlotPhase::Loading)
        break;

      __nanosleep(50);
    }

    dispatch_load(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {
      slot.phase.store((int)SlotPhase::Loaded, cuda::memory_order_release);
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
    int spins = 0;
    int current_seq;
    int phase;
    while (true) {
      // Acquire semantics ensure we see loader's data and phase transition
      current_seq = slot.task_seq.load(cuda::memory_order_acquire);
      phase = slot.phase.load(cuda::memory_order_acquire);

      if (compute_warp_idx == 0 && lane == 0 && seq <= 2 && (spins == 0 || spins % 1000 == 0)) {
        PK_PRINTF("BLK%d COMPUTE: spin=%d waiting seq=%d, slot_seq=%d, phase=%d\n",
                  blockIdx.x, spins, seq, current_seq, phase);
      }

      if (*br.done && current_seq < seq) {
        if (compute_warp_idx == 0 && lane == 0 && seq <= 2) {
          PK_PRINTF("BLK%d COMPUTE: seq=%d exiting, done=%d current_seq=%d\n",
                    blockIdx.x, seq, *br.done, current_seq);
        }
        return;
      }

      if (current_seq == seq && (phase == (int)SlotPhase::Loaded || phase == (int)SlotPhase::Computing))
        break;

      __nanosleep(50);
      spins++;
    }

    if (compute_warp_idx == 0 && lane == 0 && seq <= 2) {
      PK_PRINTF("BLK%d COMPUTE: seq=%d broke wait, current_seq=%d phase=%d\n",
                blockIdx.x, seq, current_seq, phase);
    }

    // First compute warp transitions Loaded -> Computing
    if (compute_warp_idx == 0 && lane == 0) {
      int expected = (int)SlotPhase::Loaded;
      int desired = (int)SlotPhase::Computing;
      // Use compare_exchange to handle race with other blocks
      if (slot.phase.compare_exchange_strong(expected, desired, cuda::memory_order_acq_rel)) {
        if (seq <= 2) PK_PRINTF("BLK%d COMPUTE: task seq=%d opcode=%d transition Loaded -> Computing\n",
                                blockIdx.x, seq, (int)slot.task.header.opcode);
      }
    }
    __syncwarp();

    dispatch_compute(slot.task, br, slot_idx, lane, compute_warp_idx, num_compute_warps);
    __syncwarp();

    // Last compute warp transitions Computing -> Computed
    if (lane == 0) {
      int done = slot.compute_warps_done.fetch_add(1, cuda::memory_order_acq_rel);
      if (done + 1 == num_compute_warps) {
        slot.compute_warps_done.store(0, cuda::memory_order_relaxed);  // Reset for next task
        slot.phase.store((int)SlotPhase::Computed, cuda::memory_order_release);
        if (slot_idx == 0) PK_PRINTF("BLK%d COMPUTE: task seq=%d opcode=%d completed\n",
                                     blockIdx.x, seq, (int)slot.task.header.opcode);
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

    // Wait for our sequence to be Computing or Computed
    while (true) {
      // Acquire semantics ensure we see compute's results
      int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
      int phase = slot.phase.load(cuda::memory_order_acquire);

      if (*br.done && current_seq < seq)
        return;

      if (current_seq == seq &&
          (phase == (int)SlotPhase::Computing || phase == (int)SlotPhase::Computed))
        break;

      __nanosleep(50);
    }

    wait_for_phase(&slot.phase, SlotPhase::Computed);

    dispatch_store(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {
      if (slot_idx == 0) PK_PRINTF("BLK%d STORER: task seq=%d opcode=%d buf_write=%d epoch=%d done\n",
                                   blockIdx.x, seq, (int)slot.task.header.opcode,
                                   slot.task.header.buffer_write_id, slot.task.header.write_epoch);

      // Publish data to global memory before marking dependency ready
      __threadfence();  // Device-wide fence for global memory visibility
      if (br.deps) {
        br.deps->mark_ready(slot.task.header.buffer_write_id,
                            static_cast<int>(slot.task.header.write_epoch));
      }

      // Mark slot as empty with release semantics so controller sees it
      slot.task_seq.store(-1, cuda::memory_order_relaxed);
      slot.phase.store((int)SlotPhase::Empty, cuda::memory_order_release);
    }

    seq++;
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
__global__ void persistent_kernel(GlobalQueue queue, DependencyState* deps) {
  __shared__ PipelineSlot slots[Config::kPipelineStages];
  __shared__ int done_flag;
  // Allocate pages for SMEM staging
  __shared__ Page pages[Config::kPipelineStages * Config::kPagesPerSlot];
  __shared__ PageMeta page_meta[Config::kPipelineStages * Config::kPagesPerSlot];

  // Set up block runtime
  BlockRuntime br;
  br.slots = slots;
  br.num_slots = Config::kPipelineStages;
  br.queue = queue;
  br.done = &done_flag;
  br.pages = pages;
  br.num_pages = Config::kPipelineStages * Config::kPagesPerSlot;
  br.page_meta = page_meta;
  br.deps = deps;

  const int tid = threadIdx.x;
  if (tid == 0) {
    done_flag = 0;
  }
  if (tid < br.num_slots) {
    br.slot(tid).reset();
  }
  if (tid < br.num_pages) {
    br.page_meta[tid].reset();
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
