#pragma once

#include <cstdio>

#include "block_runtime.cuh"
#include "config.cuh"
#include "dependency.cuh"
#include "dispatch.cuh"
#include "pipeline.cuh"
#include "profile_ranges.cuh"
#include "queue.cuh"
#include "segment.cuh"

#if PK_PERF_COUNTERS
#include "perf_counters.cuh"
#endif

#ifndef PK_DEBUG
#define PK_DEBUG 0
#endif

#if PK_DEBUG
#define PK_PRINTF(...) printf(__VA_ARGS__)
#else
#define PK_PRINTF(...) ((void)0)
#endif

namespace pk {

#if PK_SEGMENT_GATE && !PK_CLAIMED_READY_FIRST
#error "PK_SEGMENT_GATE requires PK_CLAIMED_READY_FIRST"
#endif

__device__ inline void profile_add(ProfileCounters* prof, ProfileRange r,
                                   unsigned long long cycles) {
#if PK_PROFILE_RANGES
  if (prof) {
    atomicAdd(&prof->cycles[static_cast<int>(r)], cycles);
  }
#else
  (void)prof;
  (void)r;
  (void)cycles;
#endif
}

__device__ inline void task_account(unsigned long long* task_counter) {
#if PK_TASK_ACCOUNTING
  if (task_counter) {
    atomicAdd(task_counter, 1ull);
  }
#else
  (void)task_counter;
#endif
}

__device__ void controller_loop(BlockRuntime& br, int lane, ProfileCounters* prof
#if PK_CONTROLLER_SKIP_BLOCKED
                                ,
                                int* pending
#endif
                                ,
                                unsigned long long* task_counter
#if PK_PERF_COUNTERS
                                ,
                                PerfBlock* perf_block
#endif
#if PK_CLAIMED_READY_FIRST
                                ,
                                unsigned int* claimed,
                                unsigned int* cursor,
                                unsigned int* claimed_total
#endif
#if PK_SEGMENT_GATE
                                ,
                                SegmentDesc* segments,
                                int num_segments,
                                int* seg_lo,
                                int* seg_hi,
                                int* window_size,
                                int* segment_completed
#endif
) {
  int seq = 0; // global task sequence number

#if PK_PERF_COUNTERS
  unsigned long long ctrl_dequeue_cycles = 0;
  unsigned long long ctrl_scan_cycles = 0;
  unsigned long long ctrl_claim_cycles = 0;
  unsigned long long ctrl_wait_cycles = 0;
  unsigned long long ctrl_issue_cycles = 0;

  unsigned long long ctrl_scans_total = 0;
  unsigned long long ctrl_scan_len_sum = 0;
  unsigned long long ctrl_scan_len_max = 0;
  unsigned long long ctrl_found_ready_count = 0;
  unsigned long long ctrl_deferred_count = 0;
  unsigned long long ctrl_no_ready_rounds = 0;

  unsigned long long ctrl_claim_attempts = 0;
  unsigned long long ctrl_claim_success = 0;
  unsigned long long ctrl_claim_fail = 0;

  unsigned long long ctrl_window_occupancy_sum = 0;
  unsigned long long ctrl_window_occupancy_samples = 0;

  unsigned long long pending_push_count = 0;
  unsigned long long pending_pop_count = 0;
  unsigned long long pending_max_depth = 0;
  unsigned long long pending_full_count = 0;

  unsigned long long tasks_issued = 0;
#endif

#if PK_CLAIMED_READY_FIRST
  if (br.queue.total <= 0) {
    if (lane == 0) {
      __threadfence_block();
      *br.done = 1;
    }
    return;
  }

  // Claimed ready-first scheduler: scan a bounded window starting from a global cursor,
  // claim ready tasks via CAS, and only backoff when the window yields nothing.
  constexpr unsigned int kLookahead = 256;
  unsigned int backoff_ns = 50;
  constexpr unsigned int kMaxBackoffNs = 50000;

#if PK_SEGMENT_GATE
  int window_lo_local = -1;
  int window_hi_local = -1;
  unsigned int seg_cursors[PK_SEGMENT_WINDOW_MAX];
  int seg_ids[PK_SEGMENT_WINDOW_MAX];
  for (int i = 0; i < PK_SEGMENT_WINDOW_MAX; ++i) {
    seg_cursors[i] = 0;
    seg_ids[i] = -1;
  }
#endif

  while (true) {
    const int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Ensure we have slot capacity before claiming a task.
#if PK_PERF_COUNTERS
    unsigned long long t_slot_wait0 = 0;
    if (lane == 0 && perf_block) {
      t_slot_wait0 = clock64();
    }
#endif
    wait_for_phase(&slot.phase, SlotPhase::Empty);
#if PK_PERF_COUNTERS
    if (lane == 0 && perf_block) {
      ctrl_wait_cycles += clock64() - t_slot_wait0;
    }
#endif

    int task_idx = -1;
    if (lane == 0) {
      const unsigned int total = static_cast<unsigned int>(br.queue.total);
      cuda::atomic_ref<unsigned int, cuda::thread_scope_device> claimed_total_ref(*claimed_total);

      unsigned int begin = 0;
      unsigned int end = total;
#if PK_SEGMENT_GATE
      int lo = atomicAdd(seg_lo, 0);
      if (lo >= num_segments) {
        // Segment window termination: only exit when all tasks completed.
        unsigned int completed_sum = 0;
        for (int s = 0; s < num_segments; ++s) {
          const int c = atomicAdd(&segment_completed[s], 0);
#if PK_DEBUG
          const SegmentDesc desc = segments[s];
          const int seg_size = desc.end - desc.begin;
          if (c > seg_size) {
            asm volatile("trap;");
          }
#endif
          if (c > 0) {
            completed_sum += static_cast<unsigned int>(c);
          }
        }
        if (completed_sum >= total) {
#if PK_DEBUG
          if (completed_sum != total) {
            asm volatile("trap;");
          }
#endif
          task_idx = -2;
        }
        // Never scan outside the bounded segment window.
        begin = 0;
        end = 0;
      } else {
        int win = atomicAdd(window_size, 0);
        if (win < 1) {
          win = 1;
        }
        if (win > PK_SEGMENT_WINDOW_MAX) {
          win = PK_SEGMENT_WINDOW_MAX;
        }

        // Advance window based on completion of the lowest segment.
        const SegmentDesc lo_desc = segments[lo];
        const int lo_size = lo_desc.end - lo_desc.begin;
        const int lo_completed = atomicAdd(&segment_completed[lo], 0);
        if (lo_size > 0 && lo_completed == lo_size) {
          if (atomicCAS(reinterpret_cast<int*>(seg_lo), lo, lo + 1) == lo) {
            const int new_lo = lo + 1;
            const int new_hi = (new_lo + win < num_segments) ? (new_lo + win) : num_segments;
            atomicExch(reinterpret_cast<int*>(seg_hi), new_hi);
            lo = new_lo;
          }
          task_idx = -3; // window moved; refresh
        }

#if PK_DEBUG
        if (lo_completed > lo_size) {
          asm volatile("trap;");
        }
#endif

        int hi = atomicAdd(seg_hi, 0);
        int desired_hi = lo + win;
        if (desired_hi < lo + 1) {
          desired_hi = lo + 1;
        }
        if (desired_hi > num_segments) {
          desired_hi = num_segments;
        }
        if (hi != desired_hi) {
          atomicExch(reinterpret_cast<int*>(seg_hi), desired_hi);
        }
        hi = desired_hi;

        // Sample average window occupancy (seg_hi - seg_lo).
#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_window_occupancy_sum += static_cast<unsigned long long>(hi - lo);
          ctrl_window_occupancy_samples += 1;
        }
#endif

        // Refresh local per-segment cursors when window changes.
        if (lo != window_lo_local || hi != window_hi_local) {
          window_lo_local = lo;
          window_hi_local = hi;
          const int count = hi - lo;
          for (int i = 0; i < count; ++i) {
            const int s = lo + i;
            if (seg_ids[i] != s) {
              const SegmentDesc desc = segments[s];
              seg_cursors[i] = static_cast<unsigned int>(desc.begin);
              seg_ids[i] = s;
            }
          }
          for (int i = count; i < PK_SEGMENT_WINDOW_MAX; ++i) {
            seg_ids[i] = -1;
            seg_cursors[i] = 0;
          }
        }

        // Clamp scan domain to a contiguous range, but prefer low segments first.
        const SegmentDesc first_desc = segments[lo];
        const SegmentDesc last_desc = segments[hi - 1];
        begin = static_cast<unsigned int>(first_desc.begin);
        end = static_cast<unsigned int>(last_desc.end);
        if (begin > total) {
          begin = total;
        }
        if (end > total) {
          end = total;
        }
        if (end < begin) {
          end = begin;
        }
      }
#endif
      if (task_idx != -2 && task_idx != -3) {
        const unsigned int seg_len = end - begin;
        const unsigned int scan_n = (seg_len < kLookahead) ? seg_len : kLookahead;
#if !PK_SEGMENT_GATE
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> cursor_ref(*cursor);
        unsigned int base = cursor_ref.load(cuda::memory_order_relaxed);
        if (base < begin || base >= end) {
          base = begin;
          atomicExch(reinterpret_cast<unsigned int*>(cursor), base);
        }
#endif

#if PK_PERF_COUNTERS
        const unsigned long long t_scan0 = perf_block ? clock64() : 0;
#endif
        unsigned int scanned = 0;
#if PK_SEGMENT_GATE
        const int lo = window_lo_local;
        const int hi = window_hi_local;
        const int seg_count = hi - lo;
        unsigned int remaining = scan_n;
        for (int seg_off = 0; seg_off < seg_count && task_idx == -1 && remaining > 0; ++seg_off) {
          const int seg = lo + seg_off;
          const SegmentDesc desc = segments[seg];
          unsigned int sbegin = static_cast<unsigned int>(desc.begin);
          unsigned int send = static_cast<unsigned int>(desc.end);
          if (sbegin > total) {
            sbegin = total;
          }
          if (send > total) {
            send = total;
          }
          if (send < sbegin) {
            send = sbegin;
          }
          const unsigned int slen = send - sbegin;
          if (slen == 0) {
            continue;
          }

          unsigned int base = seg_cursors[seg_off];
          if (base < sbegin || base >= send) {
            base = sbegin;
            seg_cursors[seg_off] = base;
          }

          const int segs_left = seg_count - seg_off;
          unsigned int per = remaining / static_cast<unsigned int>(segs_left);
          if (per == 0) {
            per = 1;
          }
          if (per > remaining) {
            per = remaining;
          }
          if (per > slen) {
            per = slen;
          }

          const unsigned int rel0 = base - sbegin;
          bool claimed_here = false;
          unsigned int claimed_idx = base;
          for (unsigned int off = 0; off < per; ++off) {
            const unsigned int idx = sbegin + ((rel0 + off) % slen);
            scanned += 1;

            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> claimed_ref(claimed[idx]);
            if (claimed_ref.load(cuda::memory_order_relaxed) != 0) {
              continue;
            }

            Task* task = get_task(&br.queue, static_cast<int>(idx));
            if (!task) {
              continue;
            }
            if (br.deps && !br.deps->is_ready(*task)) {
              continue;
            }

#if PK_PERF_COUNTERS
            if (perf_block) {
              ctrl_claim_attempts += 1;
            }
            const unsigned long long t_claim0 = perf_block ? clock64() : 0;
#endif
            const unsigned int prev =
                atomicCAS(reinterpret_cast<unsigned int*>(&claimed[idx]), 0u, 1u);
#if PK_PERF_COUNTERS
            if (perf_block) {
              ctrl_claim_cycles += clock64() - t_claim0;
            }
#endif
            if (prev == 0u) {
#if PK_PERF_COUNTERS
              if (perf_block) {
                ctrl_claim_success += 1;
              }
#endif
              task_idx = static_cast<int>(idx);
              claimed_here = true;
              claimed_idx = idx;

              atomicAdd(reinterpret_cast<unsigned int*>(claimed_total), 1u);
              break;
            }

#if PK_PERF_COUNTERS
            if (perf_block) {
              ctrl_claim_fail += 1;
            }
#endif
          }

          if (claimed_here) {
            unsigned int next = claimed_idx + 1u;
            if (next >= send) {
              next = sbegin;
            }
            seg_cursors[seg_off] = next;
          } else {
            const unsigned int next = sbegin + ((rel0 + per) % slen);
            seg_cursors[seg_off] = next;
          }

          remaining -= per;
        }
#else
        for (unsigned int off = 0; off < scan_n; ++off) {
          unsigned int idx = base + off;
          if (idx >= end) {
            idx = begin + (idx - end);
          }
          scanned += 1;

          cuda::atomic_ref<unsigned int, cuda::thread_scope_device> claimed_ref(claimed[idx]);
          if (claimed_ref.load(cuda::memory_order_relaxed) != 0) {
            continue;
          }

          Task* task = get_task(&br.queue, static_cast<int>(idx));
          if (!task) {
            continue;
          }
          if (br.deps && !br.deps->is_ready(*task)) {
            continue;
          }

#if PK_PERF_COUNTERS
          if (perf_block) {
            ctrl_claim_attempts += 1;
          }
          const unsigned long long t_claim0 = perf_block ? clock64() : 0;
#endif
          const unsigned int prev =
              atomicCAS(reinterpret_cast<unsigned int*>(&claimed[idx]), 0u, 1u);
#if PK_PERF_COUNTERS
          if (perf_block) {
            ctrl_claim_cycles += clock64() - t_claim0;
          }
#endif
          if (prev == 0u) {
#if PK_PERF_COUNTERS
            if (perf_block) {
              ctrl_claim_success += 1;
            }
#endif
            task_idx = static_cast<int>(idx);
            unsigned int next = idx + 1u;
            if (next >= end) {
              next = begin;
            }
            cursor_ref.store(next, cuda::memory_order_relaxed);

            atomicAdd(reinterpret_cast<unsigned int*>(claimed_total), 1u);
            break;
          }

#if PK_PERF_COUNTERS
          if (perf_block) {
            ctrl_claim_fail += 1;
          }
#endif
        }
#endif

#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_scan_cycles += clock64() - t_scan0;
          ctrl_scans_total += 1;
          ctrl_scan_len_sum += scanned;
          if (scanned > ctrl_scan_len_max) {
            ctrl_scan_len_max = scanned;
          }
          if (task_idx != -1) {
            ctrl_found_ready_count += 1;
          }
        }
#endif

        if (task_idx == -1) {
#if PK_PERF_COUNTERS
          if (perf_block) {
            ctrl_no_ready_rounds += 1;
          }
#endif
#if !PK_SEGMENT_GATE
          if (scan_n > 0 && end > begin) {
            unsigned int next = base + scan_n;
            const unsigned int span = end - begin;
            if (next >= end) {
              next = begin + ((next - end) % span);
            }
            cursor_ref.store(next, cuda::memory_order_relaxed);
          }
#endif
        }

#if !PK_SEGMENT_GATE
        const unsigned int total_claimed = claimed_total_ref.load(cuda::memory_order_relaxed);
        if (task_idx == -1 && total_claimed >= total) {
          task_idx = -2; // global done sentinel (broadcast to warp)
        }
#endif
      }
    }

    task_idx = __shfl_sync(0xffffffff, task_idx, 0);
    if (task_idx == -2) {
      break;
    }
    if (task_idx == -3) {
      continue;
    }
    if (task_idx == -1) {
      if (lane == 0) {
#if PK_PERF_COUNTERS
        const unsigned long long t_wait0 = perf_block ? clock64() : 0;
#endif
        __nanosleep(backoff_ns);
        if (backoff_ns < kMaxBackoffNs) {
          backoff_ns *= 2;
          if (backoff_ns > kMaxBackoffNs) {
            backoff_ns = kMaxBackoffNs;
          }
        }
#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_wait_cycles += clock64() - t_wait0;
        }
#endif
      }
      __syncwarp();
      continue;
    }

    backoff_ns = 50;

    Task* task = get_task(&br.queue, task_idx);
    if (!task) {
      continue;
    }

    if (lane == 0) {
#if PK_PERF_COUNTERS
      const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
      slot.task = *task;
#if !PK_SEGMENT_GATE
      slot.task.header.user_tag = static_cast<std::uint32_t>(task_idx);
#endif
      slot.compute_warps_done.store(0, cuda::memory_order_relaxed);
      slot.task_seq.store(seq, cuda::memory_order_release);
      slot.phase.store((int)SlotPhase::Loading, cuda::memory_order_release);
      task_account(task_counter);
#if PK_PERF_COUNTERS
      if (perf_block) {
        ctrl_issue_cycles += clock64() - t0_perf;
        tasks_issued += 1;
      }
#endif
    }
    __syncwarp();
    seq++;
  }

#if PK_PERF_COUNTERS
  if (lane == 0 && perf_block) {
    perf_block->ctrl_dequeue_cycles = ctrl_dequeue_cycles;
    perf_block->ctrl_scan_cycles = ctrl_scan_cycles;
    perf_block->ctrl_claim_cycles = ctrl_claim_cycles;
    perf_block->ctrl_wait_cycles = ctrl_wait_cycles;
    perf_block->ctrl_issue_cycles = ctrl_issue_cycles;

    perf_block->ctrl_scans_total = ctrl_scans_total;
    perf_block->ctrl_scan_len_sum = ctrl_scan_len_sum;
    perf_block->ctrl_scan_len_max = ctrl_scan_len_max;
    perf_block->ctrl_found_ready_count = ctrl_found_ready_count;
    perf_block->ctrl_deferred_count = ctrl_deferred_count;
    perf_block->ctrl_no_ready_rounds = ctrl_no_ready_rounds;

    perf_block->ctrl_claim_attempts = ctrl_claim_attempts;
    perf_block->ctrl_claim_success = ctrl_claim_success;
    perf_block->ctrl_claim_fail = ctrl_claim_fail;

    perf_block->ctrl_window_occupancy_sum = ctrl_window_occupancy_sum;
    perf_block->ctrl_window_occupancy_samples = ctrl_window_occupancy_samples;

    perf_block->pending_push_count = pending_push_count;
    perf_block->pending_pop_count = pending_pop_count;
    perf_block->pending_max_depth = pending_max_depth;
    perf_block->pending_full_count = pending_full_count;

    perf_block->tasks_issued = tasks_issued;
  }
#endif

  if (lane == 0) {
    __threadfence_block();
    *br.done = 1;
  }
  return;
#endif

#if PK_CONTROLLER_SKIP_BLOCKED
  // Simple local pending list of task indices that were dequeued but not yet ready.
  // Fixed-size to keep changes minimal; overflow falls back to blocking wait.
  constexpr int kPendingCap = 2048;
  int pending_head = 0;
  int pending_tail = 0;
  int pending_count = 0;

  auto pending_push = [&](int task_idx) {
    if (pending_count >= kPendingCap) {
#if PK_PERF_COUNTERS
      pending_full_count += 1;
#endif
      return false;
    }
    pending[pending_tail] = task_idx;
    pending_tail = (pending_tail + 1) % kPendingCap;
    pending_count++;
#if PK_PERF_COUNTERS
    pending_push_count += 1;
    if (static_cast<unsigned long long>(pending_count) > pending_max_depth) {
      pending_max_depth = static_cast<unsigned long long>(pending_count);
    }
#endif
    return true;
  };

  auto pending_try_pop_ready = [&]() -> int {
    if (pending_count == 0) {
      return -1;
    }
    // Round-robin scan: pop head, if not ready push back to tail.
    // This keeps the implementation simple and bounded.
    const int initial = pending_count;
#if PK_PERF_COUNTERS
    ctrl_scans_total += 1;
    ctrl_scan_len_sum += static_cast<unsigned long long>(initial);
    if (static_cast<unsigned long long>(initial) > ctrl_scan_len_max) {
      ctrl_scan_len_max = static_cast<unsigned long long>(initial);
    }
#endif
    for (int i = 0; i < initial; ++i) {
      int task_idx = pending[pending_head];
      pending_head = (pending_head + 1) % kPendingCap;
      pending_count--;

      Task* task = get_task(&br.queue, task_idx);
      const bool ready = (task && (!br.deps || br.deps->is_ready(*task)));
      if (ready) {
#if PK_PERF_COUNTERS
        pending_pop_count += 1;
        ctrl_found_ready_count += 1;
#endif
        return task_idx;
      }
      pending[pending_tail] = task_idx;
      pending_tail = (pending_tail + 1) % kPendingCap;
      pending_count++;
    }
#if PK_PERF_COUNTERS
    ctrl_no_ready_rounds += 1;
#endif
    return -1;
  };

  auto pending_any_ready = [&]() -> bool {
    if (pending_count == 0) {
      return false;
    }
    for (int i = 0; i < pending_count; ++i) {
      int idx = (pending_head + i) % kPendingCap;
      int task_idx = pending[idx];
      Task* task = get_task(&br.queue, task_idx);
      if (task && (!br.deps || br.deps->is_ready(*task))) {
        return true;
      }
    }
    return false;
  };
#endif

  while (true) {
    // Prefer issuing already-dequeued ready work, to avoid blocking on a single unready task.
#if PK_CONTROLLER_SKIP_BLOCKED
    // Backpressure guard: if pending is near full, stop dequeuing new tasks until
    // some blocked work becomes ready (prevents pathological pending growth).
    constexpr int kPendingHighWater = kPendingCap - Config::kChunkSize;
    int pending_backpressure = 0;
    if (lane == 0 && pending_count >= kPendingHighWater) {
      pending_backpressure = 1;
#if PK_PERF_COUNTERS
      const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
      const unsigned long long t0 = clock64();
      while (pending_count >= kPendingHighWater && !pending_any_ready()) {
        __nanosleep(100);
      }
      profile_add(prof, ProfileRange::ControllerDepsWait, clock64() - t0);
#if PK_PERF_COUNTERS
      if (perf_block) {
        ctrl_wait_cycles += clock64() - t0_perf;
      }
#endif
    }
    pending_backpressure = __shfl_sync(0xffffffff, pending_backpressure, 0);
    if (pending_backpressure) {
      continue;
    }

    int pending_task_idx = -1;
    if (lane == 0) {
#if PK_PERF_COUNTERS
      const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
      pending_task_idx = pending_try_pop_ready();
#if PK_PERF_COUNTERS
      if (perf_block) {
        ctrl_scan_cycles += clock64() - t0_perf;
      }
#endif
    }
    pending_task_idx = __shfl_sync(0xffffffff, pending_task_idx, 0);
    if (pending_task_idx != -1) {
      int slot_idx = seq % br.num_slots;
      PipelineSlot& slot = br.slot(slot_idx);

#if PK_PERF_COUNTERS
      unsigned long long t_wait_slot = 0;
      if (lane == 0 && perf_block) {
        t_wait_slot = clock64();
      }
#endif
      wait_for_phase(&slot.phase, SlotPhase::Empty);
#if PK_PERF_COUNTERS
      if (lane == 0 && perf_block) {
        ctrl_wait_cycles += clock64() - t_wait_slot;
      }
#endif
      Task* task = get_task(&br.queue, pending_task_idx);

      if (lane == 0) {
#if PK_PERF_COUNTERS
        const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
        slot.task = *task;
#if !PK_SEGMENT_GATE
        slot.task.header.user_tag = static_cast<std::uint32_t>(pending_task_idx);
#endif
        slot.compute_warps_done.store(0, cuda::memory_order_relaxed);
        slot.task_seq.store(seq, cuda::memory_order_release);
        slot.phase.store((int)SlotPhase::Loading, cuda::memory_order_release);
        task_account(task_counter);
#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_issue_cycles += clock64() - t0_perf;
          tasks_issued += 1;
        }
#endif
      }
      __syncwarp();
      seq++;
      continue;
    }
#endif

    // Allocate chunk of tasks from global queue and broadcast to all lanes
    int chunk_begin = 0, chunk_end = 0;
    if (lane == 0) {
#if PK_PERF_COUNTERS
      const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
      const unsigned long long t0 = clock64();
      dequeue_chunk(&br.queue, chunk_begin, chunk_end);
      profile_add(prof, ProfileRange::ControllerQueuePop, clock64() - t0);
#if PK_PERF_COUNTERS
      if (perf_block) {
        ctrl_dequeue_cycles += clock64() - t0_perf;
      }
#endif
    }
    __syncwarp();  // Ensure lane 0 completes before shuffle
    chunk_begin = __shfl_sync(0xffffffff, chunk_begin, 0);
    chunk_end = __shfl_sync(0xffffffff, chunk_end, 0);

    if (lane == 0 && seq == 0) {
      PK_PRINTF("BLK%d CONTROLLER: seq=%d chunk [%d, %d)\n", blockIdx.x, seq, chunk_begin, chunk_end);
    }

    if (chunk_begin >= chunk_end) {
#if PK_CONTROLLER_SKIP_BLOCKED
      if (lane == 0 && pending_count > 0) {
        // Queue drained, but we still have blocked work. Wait until something becomes ready.
        const unsigned long long t0 = clock64();
        while (pending_count > 0 && !pending_any_ready()) {
          __nanosleep(100);
        }
        profile_add(prof, ProfileRange::ControllerDepsWait, clock64() - t0);
      }
      int has_pending = 0;
      if (lane == 0) {
        has_pending = (pending_count > 0) ? 1 : 0;
      }
      has_pending = __shfl_sync(0xffffffff, has_pending, 0);
      if (has_pending) {
        continue;
      }
#endif
      break;
    }

    // Fill slots with tasks from this chunk
    for (int task_idx = chunk_begin; task_idx < chunk_end; ++task_idx) {
      int slot_idx = seq % br.num_slots;
      PipelineSlot& slot = br.slot(slot_idx);

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: processing seq=%d task_idx=%d slot=%d phase=%d\n",
                  blockIdx.x, seq, task_idx, slot_idx, (int)slot.phase);
      }

      // Wait for slot to be empty (storer finished with it)
#if PK_PERF_COUNTERS
      unsigned long long t_wait_slot = 0;
      if (lane == 0 && perf_block) {
        t_wait_slot = clock64();
      }
#endif
      wait_for_phase(&slot.phase, SlotPhase::Empty);
#if PK_PERF_COUNTERS
      if (lane == 0 && perf_block) {
        ctrl_wait_cycles += clock64() - t_wait_slot;
      }
#endif

      Task* task = get_task(&br.queue, task_idx);

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: got task %d, opcode=%d, buf_write=%d\n",
                  blockIdx.x, task_idx, (int)task->header.opcode, task->header.buffer_write_id);
      }

      // Wait for declared dependencies to be satisfied
#if PK_CONTROLLER_SKIP_BLOCKED
      int deferred = 0;
      if (lane == 0 && br.deps && !br.deps->is_ready(*task)) {
        deferred = pending_push(task_idx) ? 1 : 0;
#if PK_PERF_COUNTERS
        if (deferred && perf_block) {
          ctrl_deferred_count += 1;
        }
#endif
      }
      deferred = __shfl_sync(0xffffffff, deferred, 0);
      if (deferred) {
        continue;
      }
#endif

      if (lane == 0 && br.deps) {
#if PK_PERF_COUNTERS
        const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
        const unsigned long long t0 = clock64();

#if PK_CONTROLLER_SKIP_BLOCKED
        if (!br.deps->is_ready(*task)) {
          // pending overflow: fall back to blocking wait.
        }
#endif

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
        profile_add(prof, ProfileRange::ControllerDepsWait, clock64() - t0);
#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_wait_cycles += clock64() - t0_perf;
        }
#endif
      }
      __syncwarp();

      if (lane == 0 && task_idx <= 2) {
        PK_PRINTF("BLK%d CONTROLLER: task %d deps satisfied, starting\n", blockIdx.x, task_idx);
      }

      if (lane == 0) { // transition to loading
#if PK_PERF_COUNTERS
        const unsigned long long t0_perf = perf_block ? clock64() : 0;
#endif
        slot.task = *task;
#if !PK_SEGMENT_GATE
        slot.task.header.user_tag = static_cast<std::uint32_t>(task_idx);
#endif
        slot.compute_warps_done.store(0, cuda::memory_order_relaxed);

        // Publish task with release semantics so loader/compute/storer see all writes
        slot.task_seq.store(seq, cuda::memory_order_release);
        slot.phase.store((int)SlotPhase::Loading, cuda::memory_order_release);
        task_account(task_counter);
#if PK_PERF_COUNTERS
        if (perf_block) {
          ctrl_issue_cycles += clock64() - t0_perf;
          tasks_issued += 1;
        }
#endif
      }
      __syncwarp();
      seq++;
    }
  }

#if PK_PERF_COUNTERS
  if (lane == 0 && perf_block) {
    perf_block->ctrl_dequeue_cycles = ctrl_dequeue_cycles;
    perf_block->ctrl_scan_cycles = ctrl_scan_cycles;
    perf_block->ctrl_claim_cycles = ctrl_claim_cycles;
    perf_block->ctrl_wait_cycles = ctrl_wait_cycles;
    perf_block->ctrl_issue_cycles = ctrl_issue_cycles;

    perf_block->ctrl_scans_total = ctrl_scans_total;
    perf_block->ctrl_scan_len_sum = ctrl_scan_len_sum;
    perf_block->ctrl_scan_len_max = ctrl_scan_len_max;
    perf_block->ctrl_found_ready_count = ctrl_found_ready_count;
    perf_block->ctrl_deferred_count = ctrl_deferred_count;
    perf_block->ctrl_no_ready_rounds = ctrl_no_ready_rounds;

    perf_block->ctrl_claim_attempts = ctrl_claim_attempts;
    perf_block->ctrl_claim_success = ctrl_claim_success;
    perf_block->ctrl_claim_fail = ctrl_claim_fail;

    perf_block->pending_push_count = pending_push_count;
    perf_block->pending_pop_count = pending_pop_count;
    perf_block->pending_max_depth = pending_max_depth;
    perf_block->pending_full_count = pending_full_count;

    perf_block->tasks_issued = tasks_issued;
  }
#endif

  if (lane == 0) { // signal done to other warps

    __threadfence_block();
    *br.done = 1;
  }
}

__device__ void loader_loop(BlockRuntime& br, int lane, ProfileCounters* prof
#if PK_PERF_COUNTERS
                            ,
                            PerfBlock* perf_block
#endif
) {
  int seq = 0; // Loader tracks which sequence it's waiting for

#if PK_PERF_COUNTERS
  unsigned long long load_wait_slot_cycles = 0;
  unsigned long long load_work_cycles = 0;
  unsigned long long load_page_reuse_hits = 0;
  unsigned long long load_page_refill_count = 0;
#endif

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Spin until this slot has our sequence AND is in Loading phase
    unsigned long long t_wait_profile = 0;
#if PK_PERF_COUNTERS
    unsigned long long t_wait_perf = 0;
#endif
    while (true) {
      // Acquire semantics ensure we see controller's task initialization
      int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
      int phase = slot.phase.load(cuda::memory_order_acquire);

      if (*br.done && current_seq < seq) {
        break;
      }

      if (current_seq == seq && phase == (int)SlotPhase::Loading)
        break;

      if (lane == 0) {
        if (t_wait_profile == 0) {
          t_wait_profile = clock64();
        }
#if PK_PERF_COUNTERS
        if (perf_block && t_wait_perf == 0) {
          t_wait_perf = clock64();
        }
#endif
      }
      __nanosleep(50);
    }

    if (lane == 0 && t_wait_profile != 0) {
      profile_add(prof, ProfileRange::LoaderSlotWait, clock64() - t_wait_profile);
    }
#if PK_PERF_COUNTERS
    if (lane == 0 && perf_block && t_wait_perf != 0) {
      load_wait_slot_cycles += clock64() - t_wait_perf;
    }
#endif

    int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
    if (*br.done && current_seq < seq) {
      break;
    }

    unsigned long long t_load_profile = 0;
#if PK_PERF_COUNTERS
    unsigned long long t_load_perf = 0;
#endif
    if (lane == 0) {
      t_load_profile = clock64();
#if PK_PERF_COUNTERS
      if (perf_block) {
        t_load_perf = t_load_profile;
      }
#endif
    }
    dispatch_load(slot.task, br, slot_idx, lane
#if PK_PERF_COUNTERS
                  ,
                  perf_block ? &load_page_reuse_hits : nullptr,
                  perf_block ? &load_page_refill_count : nullptr
#else
                  ,
                  nullptr,
                  nullptr
#endif
    );
    __syncwarp();
    if (lane == 0) {
      profile_add(prof, ProfileRange::LoaderLoad, clock64() - t_load_profile);
    }
#if PK_PERF_COUNTERS
    if (lane == 0 && perf_block) {
      load_work_cycles += clock64() - t_load_perf;
    }
#endif

    if (lane == 0) {
      slot.phase.store((int)SlotPhase::Loaded, cuda::memory_order_release);
    }

    seq++;
  }

#if PK_PERF_COUNTERS
  if (lane == 0 && perf_block) {
    perf_block->load_wait_slot_cycles = load_wait_slot_cycles;
    perf_block->load_work_cycles = load_work_cycles;
    perf_block->load_page_reuse_hits = load_page_reuse_hits;
    perf_block->load_page_refill_count = load_page_refill_count;
  }
#endif
}

__device__ void compute_loop(BlockRuntime& br, int lane, int compute_warp_idx, int num_compute_warps,
                             ProfileCounters* prof
#if PK_PERF_COUNTERS
                             ,
                             unsigned long long* compute_wait_cycles,
                             unsigned long long* compute_work_cycles
#endif
) {
  int seq = 0;

#if PK_PERF_COUNTERS
  unsigned long long compute_wait_loaded_cycles_local = 0;
  unsigned long long compute_work_cycles_local = 0;
#endif

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Wait for our sequence to be Loaded or Computing
    int spins = 0;
    int current_seq;
    int phase;
    unsigned long long t_wait_profile = 0;
#if PK_PERF_COUNTERS
    unsigned long long t_wait_perf = 0;
#endif
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
        break;
      }

      if (current_seq == seq && (phase == (int)SlotPhase::Loaded || phase == (int)SlotPhase::Computing))
        break;

      if (lane == 0) {
        if (t_wait_profile == 0) {
          t_wait_profile = clock64();
        }
#if PK_PERF_COUNTERS
        if (compute_wait_cycles && t_wait_perf == 0) {
          t_wait_perf = clock64();
        }
#endif
      }
      __nanosleep(50);
      spins++;
    }
    if (compute_warp_idx == 0 && lane == 0 && t_wait_profile != 0) {
      profile_add(prof, ProfileRange::ComputeSlotWait, clock64() - t_wait_profile);
    }
#if PK_PERF_COUNTERS
    if (lane == 0 && compute_wait_cycles && t_wait_perf != 0) {
      compute_wait_loaded_cycles_local += clock64() - t_wait_perf;
    }
#endif

    if (*br.done && current_seq < seq) {
      break;
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

    unsigned long long t_math_profile = 0;
#if PK_PERF_COUNTERS
    unsigned long long t_math_perf = 0;
#endif
    if (compute_warp_idx == 0 && lane == 0) {
      t_math_profile = clock64();
    }
    if (lane == 0) {
#if PK_PERF_COUNTERS
      if (compute_work_cycles) {
        t_math_perf = clock64();
      }
#endif
    }
    dispatch_compute(slot.task, br, slot_idx, lane, compute_warp_idx, num_compute_warps);
    __syncwarp();
    if (compute_warp_idx == 0 && lane == 0) {
      profile_add(prof, ProfileRange::ComputeMath, clock64() - t_math_profile);
    }
#if PK_PERF_COUNTERS
    if (lane == 0 && compute_work_cycles) {
      compute_work_cycles_local += clock64() - t_math_perf;
    }
#endif

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

#if PK_PERF_COUNTERS
  if (lane == 0 && compute_wait_cycles && compute_work_cycles) {
    compute_wait_cycles[compute_warp_idx] = compute_wait_loaded_cycles_local;
    compute_work_cycles[compute_warp_idx] = compute_work_cycles_local;
  }
#endif
}

__device__ void storer_loop(BlockRuntime& br, int lane, ProfileCounters* prof
#if PK_PERF_COUNTERS
                            ,
                            PerfBlock* perf_block
#endif
#if PK_SEGMENT_GATE
                            ,
                            int* segment_completed,
                            int num_segments
#endif
) {
  int seq = 0;

#if PK_PERF_COUNTERS
  unsigned long long store_wait_computed_cycles = 0;
  unsigned long long store_work_cycles = 0;
  unsigned long long tasks_completed = 0;
#endif

  while (true) {
    int slot_idx = seq % br.num_slots;
    PipelineSlot& slot = br.slot(slot_idx);

    // Wait for our sequence to be Computing or Computed
    unsigned long long t_wait_profile = 0;
#if PK_PERF_COUNTERS
    unsigned long long t_wait_perf = 0;
#endif
    while (true) {
      // Acquire semantics ensure we see compute's results
      int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
      int phase = slot.phase.load(cuda::memory_order_acquire);

      if (*br.done && current_seq < seq) {
        break;
      }

      if (current_seq == seq &&
          (phase == (int)SlotPhase::Computing || phase == (int)SlotPhase::Computed))
        break;

      if (lane == 0) {
        if (t_wait_profile == 0) {
          t_wait_profile = clock64();
        }
#if PK_PERF_COUNTERS
        if (perf_block && t_wait_perf == 0) {
          t_wait_perf = clock64();
        }
#endif
      }
      __nanosleep(50);
    }

    if (lane == 0 && t_wait_profile != 0) {
      profile_add(prof, ProfileRange::StorerSlotWait, clock64() - t_wait_profile);
    }
#if PK_PERF_COUNTERS
    if (lane == 0 && perf_block && t_wait_perf != 0) {
      store_wait_computed_cycles += clock64() - t_wait_perf;
    }
#endif

    int current_seq = slot.task_seq.load(cuda::memory_order_acquire);
    if (*br.done && current_seq < seq) {
      break;
    }

#if PK_PERF_COUNTERS
    unsigned long long t_work_perf = 0;
    if (lane == 0 && perf_block) {
      t_work_perf = clock64();
    }
#endif
    wait_for_phase(&slot.phase, SlotPhase::Computed);

    dispatch_store(slot.task, br, slot_idx, lane);
    __syncwarp();

    if (lane == 0) {
      if (slot_idx == 0) PK_PRINTF("BLK%d STORER: task seq=%d opcode=%d buf_write=%d epoch=%d done\n",
                                   blockIdx.x, seq, (int)slot.task.header.opcode,
                                   slot.task.header.buffer_write_id, slot.task.header.write_epoch);

      // Publish data to global memory before marking dependency ready
      const unsigned long long t_fence0 = clock64();
#if !PK_RELAX_PUBLISH_FENCE
      __threadfence();  // Device-wide fence for global memory visibility
#endif
      profile_add(prof, ProfileRange::StorerThreadfence, clock64() - t_fence0);
      if (br.deps) {
        const unsigned long long t_deps0 = clock64();
        br.deps->mark_ready(slot.task.header.buffer_write_id,
                            static_cast<int>(slot.task.header.write_epoch));
        profile_add(prof, ProfileRange::StorerDepsMarkReady, clock64() - t_deps0);
      }

#if PK_SEGMENT_GATE
      if (segment_completed && num_segments > 0) {
        const int seg = static_cast<int>(slot.task.header.user_tag);
        if (seg >= 0 && seg < num_segments) {
          atomicAdd(&segment_completed[seg], 1);
        }
      }
#endif

      // Mark slot as empty with release semantics so controller sees it
      slot.task_seq.store(-1, cuda::memory_order_relaxed);
      slot.phase.store((int)SlotPhase::Empty, cuda::memory_order_release);

#if PK_PERF_COUNTERS
      if (perf_block) {
        tasks_completed += 1;
      }
#endif

    }

#if PK_PERF_COUNTERS
    if (lane == 0 && perf_block) {
      store_work_cycles += clock64() - t_work_perf;
    }
#endif

    seq++;
  }

#if PK_PERF_COUNTERS
  if (lane == 0 && perf_block) {
    perf_block->store_wait_computed_cycles = store_wait_computed_cycles;
    perf_block->store_work_cycles = store_work_cycles;
    perf_block->tasks_completed = tasks_completed;
  }
#endif
}

// Blocks spin, processing tasks from the global queue.
// Warps specializations: controller, loader, computer, storer
//
// Pipeline flow:
//   1. Controller: dequeues tasks from global queue -> fills slots
//   2. Loader: waits for filled slots -> loads data (Empty -> Loaded)
//   3. Compute: waits for loaded slots -> processes (Loaded -> Computed)
//   4. Storer: waits for computed slots -> stores results (Computed -> Empty)
__global__ void persistent_kernel(GlobalQueue queue, DependencyState* deps
#if PK_PROFILE_RANGES
                                  ,
                                  ProfileCounters* prof
#endif
#if PK_TASK_ACCOUNTING
                                  ,
                                  unsigned long long* task_counter
#endif
#if PK_PERF_COUNTERS
                                  ,
                                  PerfBlock* perf
#endif
#if PK_CLAIMED_READY_FIRST
                                  ,
                                  unsigned int* claimed,
                                  unsigned int* cursor,
                                  unsigned int* claimed_total
#endif
#if PK_SEGMENT_GATE
                                  ,
                                  SegmentDesc* segments,
                                  int num_segments,
                                  int* seg_lo,
                                  int* seg_hi,
                                  int* window_size,
                                  int* segment_completed
#endif
) {
  __shared__ PipelineSlot slots[Config::kPipelineStages];
  __shared__ int done_flag;
#if PK_CONTROLLER_SKIP_BLOCKED
  __shared__ int controller_pending[2048];
#endif
  // Allocate pages for SMEM staging
  __shared__ Page pages[Config::kPipelineStages * Config::kPagesPerSlot];
  __shared__ PageMeta page_meta[Config::kPipelineStages * Config::kPagesPerSlot];

#if PK_PERF_COUNTERS
  __shared__ PerfBlock perf_s;
  __shared__ unsigned long long compute_wait_cycles_s[Config::kNumComputeWarps];
  __shared__ unsigned long long compute_work_cycles_s[Config::kNumComputeWarps];
#endif

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
#if PK_PERF_COUNTERS
  if (tid == 0) {
    perf_s = PerfBlock{};
    for (int i = 0; i < Config::kNumComputeWarps; ++i) {
      compute_wait_cycles_s[i] = 0;
      compute_work_cycles_s[i] = 0;
    }
  }
#endif
  if (tid < br.num_slots) {
    br.slot(tid).reset();
  }
  if (tid < br.num_pages) {
    br.page_meta[tid].reset();
  }
  __syncthreads();

  const int wid = warp_id();
  const int lane = lane_id();

  ProfileCounters* prof_ptr = nullptr;
#if PK_PROFILE_RANGES
  prof_ptr = prof;
#endif

#if PK_TASK_ACCOUNTING
  unsigned long long* task_counter_ptr = task_counter;
#else
  unsigned long long* task_counter_ptr = nullptr;
#endif

#if PK_PERF_COUNTERS
  PerfBlock* perf_ptr = perf;
#endif

  if (is_controller(wid, br.roles)) {
    controller_loop(br, lane, prof_ptr
#if PK_CONTROLLER_SKIP_BLOCKED
                    ,
                    controller_pending
#endif
                    ,
                    task_counter_ptr
#if PK_PERF_COUNTERS
                    ,
                    perf_ptr ? &perf_s : nullptr
#endif
#if PK_CLAIMED_READY_FIRST
                    ,
                    claimed,
                    cursor,
                    claimed_total
#endif
#if PK_SEGMENT_GATE
                    ,
                    segments,
                    num_segments,
                    seg_lo,
                    seg_hi,
                    window_size,
                    segment_completed
#endif
    );
  } else if (is_loader(wid, br.roles)) {
    loader_loop(br, lane, prof_ptr
#if PK_PERF_COUNTERS
                ,
                perf_ptr ? &perf_s : nullptr
#endif
    );
  } else if (is_storer(wid, br.roles)) {
    storer_loop(br, lane, prof_ptr
#if PK_PERF_COUNTERS
                ,
                perf_ptr ? &perf_s : nullptr
#endif
#if PK_SEGMENT_GATE
                ,
                segment_completed,
                num_segments
#endif
    );
  } else if (is_compute(wid, br.roles)) {
    const int compute_idx = compute_warp_index(wid, br.roles);
    compute_loop(br, lane, compute_idx, br.roles.num_compute, prof_ptr
#if PK_PERF_COUNTERS
                 ,
                 perf_ptr ? compute_wait_cycles_s : nullptr,
                 perf_ptr ? compute_work_cycles_s : nullptr
#endif
    );
  }

#if PK_PERF_COUNTERS
  __syncthreads();
  if (perf_ptr && tid == 0) {
    unsigned long long wait_sum = 0;
    unsigned long long work_sum = 0;
    for (int i = 0; i < Config::kNumComputeWarps; ++i) {
      wait_sum += compute_wait_cycles_s[i];
      work_sum += compute_work_cycles_s[i];
    }
    perf_s.compute_wait_loaded_cycles = wait_sum;
    perf_s.compute_work_cycles = work_sum;
    perf_ptr[blockIdx.x] = perf_s;
  }
#endif
}

} // namespace pk
