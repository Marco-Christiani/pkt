#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda/atomic>

#include "config.cuh"
#include "dependency.cuh"
#include "pipeline.cuh"
#include "queue.cuh"

namespace pk {

// For SMEM paging
struct Page {
  alignas(16) std::byte data[Config::kPageBytes > 0 ? Config::kPageBytes : 16];

  template <typename T> __device__ inline T* as() {
    static_assert(alignof(T) <= 16, "T alignment too large for Page");
    return reinterpret_cast<T*>(data);
  }

  template <typename T> __device__ inline const T* as() const {
    static_assert(alignof(T) <= 16, "T alignment too large for Page");
    return reinterpret_cast<const T*>(data);
  }
};

enum class PageLease : int {
  Free = 0,
  Loader = 1,
  Ready = 2,
  Storer = 3,
};

struct PageHandle {
  int slot_idx{0};
  int page_idx{0};
};

struct PageMeta {
  cuda::atomic<int, cuda::thread_scope_block> lease;
  cuda::atomic<unsigned long long, cuda::thread_scope_block> tag;

  __device__ inline void reset() {
    lease.store((int)PageLease::Free, cuda::memory_order_relaxed);
    tag.store(0ull, cuda::memory_order_relaxed);
  }
};

__device__ inline unsigned long long mix_page_tag(unsigned long long a, unsigned long long b) {
  unsigned long long x = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}

__device__ inline unsigned long long make_page_tag(const void* ptr, int a, int b, int c) {
  unsigned long long x = static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(ptr));
  x = mix_page_tag(x, static_cast<unsigned long long>(static_cast<unsigned int>(a)));
  x = mix_page_tag(x, static_cast<unsigned long long>(static_cast<unsigned int>(b)));
  x = mix_page_tag(x, static_cast<unsigned long long>(static_cast<unsigned int>(c)));
  return x ? x : 1ull;
}

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
  PageMeta* page_meta{nullptr};

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

  __device__ inline PageHandle page_handle(int slot_idx, int page_idx) const {
    return PageHandle{slot_idx, page_idx};
  }

  __device__ inline int physical_page_index(PageHandle h) const {
    return h.slot_idx * Config::kPagesPerSlot + h.page_idx;
  }

  __device__ inline Page& page(PageHandle h) { return pages[physical_page_index(h)]; }

  __device__ inline const Page& page(PageHandle h) const { return pages[physical_page_index(h)]; }

  template <typename T> __device__ inline T* page_ptr(PageHandle h) { return page(h).as<T>(); }

  template <typename T> __device__ inline const T* page_ptr(PageHandle h) const {
    return page(h).as<T>();
  }

  __device__ inline PageMeta& meta(PageHandle h) { return page_meta[physical_page_index(h)]; }

  __device__ inline const PageMeta& meta(PageHandle h) const { return page_meta[physical_page_index(h)]; }

  __device__ inline bool page_try_acquire(PageHandle h) {
    int expected = (int)PageLease::Free;
    return meta(h).lease.compare_exchange_strong(expected, (int)PageLease::Loader,
                                                cuda::memory_order_acq_rel);
  }

  __device__ inline void page_wait_acquire(PageHandle h) {
    while (!page_try_acquire(h)) {
      __nanosleep(50);
    }
  }

  __device__ inline bool page_is_ready(PageHandle h) const {
    return meta(h).lease.load(cuda::memory_order_acquire) == (int)PageLease::Ready;
  }

  __device__ inline unsigned long long page_tag(PageHandle h) const {
    return meta(h).tag.load(cuda::memory_order_acquire);
  }

  __device__ inline bool page_is_ready_with_tag(PageHandle h, unsigned long long tag) const {
    return page_is_ready(h) && page_tag(h) == tag;
  }

  __device__ inline bool page_try_begin_overwrite(PageHandle h) {
    int expected = (int)PageLease::Free;
    if (meta(h).lease.compare_exchange_strong(expected, (int)PageLease::Loader,
                                             cuda::memory_order_acq_rel)) {
      return true;
    }
    expected = (int)PageLease::Ready;
    return meta(h).lease.compare_exchange_strong(expected, (int)PageLease::Loader,
                                                cuda::memory_order_acq_rel);
  }

  __device__ inline void page_wait_begin_overwrite(PageHandle h) {
    while (!page_try_begin_overwrite(h)) {
      __nanosleep(50);
    }
  }

  __device__ inline void page_publish(PageHandle h, unsigned long long tag) {
    meta(h).tag.store(tag, cuda::memory_order_release);
    meta(h).lease.store((int)PageLease::Ready, cuda::memory_order_release);
  }

  __device__ inline void page_publish_no_tag_change(PageHandle h) {
    meta(h).lease.store((int)PageLease::Ready, cuda::memory_order_release);
  }

  __device__ inline void page_release(PageHandle h) {
    meta(h).lease.store((int)PageLease::Free, cuda::memory_order_release);
  }
};

} // namespace pk
