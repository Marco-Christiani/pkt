#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "config.cuh"
#include "dependency.cuh"
#include "kernel.cuh"
#include "profile_ranges.cuh"
#include "queue.cuh"
#include "segment.cuh"
#include "task.cuh"

#if PK_PERF_COUNTERS
#include "perf_counters.cuh"
#endif

namespace pk {

class Runtime {
public:
  Runtime() = default;
  ~Runtime() { cleanup(); }

  void initialize(int max_tasks) {
    max_tasks_ = max_tasks;

    // Task mem and counters
    cudaMalloc(&d_tasks_, max_tasks * sizeof(Task));
    cudaMalloc(&d_next_index_, sizeof(int));
    cudaMemset(d_next_index_, 0, sizeof(int));
    cudaMalloc(&d_deps_, sizeof(DependencyState));
    reset_dependencies();

#if PK_PROFILE_RANGES
    cudaMalloc(&d_profile_, sizeof(ProfileCounters));
    cudaMemset(d_profile_, 0, sizeof(ProfileCounters));
#endif

#if PK_TASK_ACCOUNTING
    cudaMalloc(&d_task_counter_, sizeof(unsigned long long));
    cudaMemset(d_task_counter_, 0, sizeof(unsigned long long));
#endif

#if PK_CLAIMED_READY_FIRST
    if (claimed_capacity_ < max_tasks) {
      if (d_claimed_) {
        cudaFree(d_claimed_);
      }
      cudaMalloc(&d_claimed_, sizeof(unsigned int) * max_tasks);
      claimed_capacity_ = max_tasks;
    }
    cudaMalloc(&d_cursor_, sizeof(unsigned int));
    cudaMalloc(&d_claimed_total_, sizeof(unsigned int));
    cudaMemset(d_cursor_, 0, sizeof(unsigned int));
    cudaMemset(d_claimed_total_, 0, sizeof(unsigned int));
#endif

#if PK_SEGMENT_GATE
    if (segments_capacity_ < max_tasks) {
      if (d_segments_) {
        cudaFree(d_segments_);
      }
      cudaMalloc(&d_segments_, sizeof(SegmentDesc) * max_tasks);
      segments_capacity_ = max_tasks;
    }
    if (d_seg_lo_) {
      cudaFree(d_seg_lo_);
      d_seg_lo_ = nullptr;
    }
    if (d_seg_hi_) {
      cudaFree(d_seg_hi_);
      d_seg_hi_ = nullptr;
    }
    if (d_window_size_) {
      cudaFree(d_window_size_);
      d_window_size_ = nullptr;
    }
    if (d_segment_completed_) {
      cudaFree(d_segment_completed_);
      d_segment_completed_ = nullptr;
    }
    cudaMalloc(&d_seg_lo_, sizeof(int));
    cudaMalloc(&d_seg_hi_, sizeof(int));
    cudaMalloc(&d_window_size_, sizeof(int));
    cudaMalloc(&d_segment_completed_, sizeof(int) * max_tasks);
#endif
  }

  void submit_tasks(const std::vector<Task>& tasks) {
    if (tasks.empty())
      return;

    num_tasks_ = static_cast<int>(tasks.size());
    cudaMemcpy(d_tasks_, tasks.data(), num_tasks_ * sizeof(Task), cudaMemcpyHostToDevice);
    cudaMemset(d_next_index_, 0, sizeof(int)); // reset counter
    reset_dependencies();                      // fresh dependency counts per submission

#if PK_PROFILE_RANGES
    cudaMemset(d_profile_, 0, sizeof(ProfileCounters));
#endif

#if PK_TASK_ACCOUNTING
    cudaMemset(d_task_counter_, 0, sizeof(unsigned long long));
#endif

#if PK_CLAIMED_READY_FIRST
    if (d_claimed_) {
      cudaMemset(d_claimed_, 0, sizeof(unsigned int) * num_tasks_);
    }
    if (d_cursor_) {
      cudaMemset(d_cursor_, 0, sizeof(unsigned int));
    }
    if (d_claimed_total_) {
      cudaMemset(d_claimed_total_, 0, sizeof(unsigned int));
    }
#endif

#if PK_SEGMENT_GATE
    if (!host_segments_.empty()) {
      num_segments_ = static_cast<int>(host_segments_.size());
      cudaMemcpy(d_segments_, host_segments_.data(), sizeof(SegmentDesc) * host_segments_.size(),
                 cudaMemcpyHostToDevice);
      cudaMemset(d_segment_completed_, 0, sizeof(int) * host_segments_.size());
      int window_size = segment_window_size_;
      if (window_size < 1) {
        window_size = 1;
      }
      if (window_size > PK_SEGMENT_WINDOW_MAX) {
        window_size = PK_SEGMENT_WINDOW_MAX;
      }
      if (window_size > num_segments_) {
        window_size = num_segments_;
      }

      const int seg_lo = 0;
      const int seg_hi = window_size;
      cudaMemcpy(d_window_size_, &window_size, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_seg_lo_, &seg_lo, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_seg_hi_, &seg_hi, sizeof(int), cudaMemcpyHostToDevice);
#if PK_CLAIMED_READY_FIRST
      const unsigned int begin0 = static_cast<unsigned int>(host_segments_[0].begin);
      cudaMemcpy(d_cursor_, &begin0, sizeof(unsigned int), cudaMemcpyHostToDevice);
#endif
    } else {
      num_segments_ = 0;
      const int seg_lo = 0;
      const int seg_hi = 0;
      const int window_size = segment_window_size_;
      cudaMemcpy(d_window_size_, &window_size, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_seg_lo_, &seg_lo, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(d_seg_hi_, &seg_hi, sizeof(int), cudaMemcpyHostToDevice);
    }
#endif
  }

  void set_segments(const std::vector<SegmentDesc>& segments) {
#if PK_SEGMENT_GATE
    host_segments_ = segments;
#else
    (void)segments;
#endif
  }

  void set_segment_window_size(int window_size) {
#if PK_SEGMENT_GATE
    segment_window_size_ = window_size;
#else
    (void)window_size;
#endif
  }

  void launch(int num_blocks, cudaStream_t stream = nullptr) {
    GlobalQueue queue;
    queue.tasks = d_tasks_;
    queue.total = num_tasks_;
    queue.next_index = d_next_index_;

#if PK_PERF_COUNTERS
    PerfBlock* perf_ptr = nullptr;
    last_launch_blocks_ = num_blocks;
    if (perf_enabled_) {
      if (perf_blocks_capacity_ < num_blocks) {
        if (d_perf_blocks_) {
          cudaFree(d_perf_blocks_);
        }
        cudaMalloc(&d_perf_blocks_, sizeof(PerfBlock) * num_blocks);
        perf_blocks_capacity_ = num_blocks;
      }
      cudaMemset(d_perf_blocks_, 0, sizeof(PerfBlock) * num_blocks);
      perf_ptr = d_perf_blocks_;
    }
#else
    (void)num_blocks;
#endif

    persistent_kernel<<<num_blocks, Config::kThreadsPerBlock, 0, stream>>>(queue, d_deps_
#if PK_PROFILE_RANGES
                                                                            ,
                                                                            d_profile_
#endif
#if PK_TASK_ACCOUNTING
                                                                            ,
                                                                            task_accounting_enabled_ ? d_task_counter_ : nullptr
#endif
#if PK_PERF_COUNTERS
                                                                            ,
                                                                            perf_ptr
#endif
#if PK_CLAIMED_READY_FIRST
                                                                            ,
                                                                            d_claimed_,
                                                                            d_cursor_,
                                                                            d_claimed_total_
#endif
#if PK_SEGMENT_GATE
                                                                            ,
                                                                            d_segments_,
                                                                            num_segments_,
                                                                            d_seg_lo_,
                                                                            d_seg_hi_,
                                                                            d_window_size_,
                                                                            d_segment_completed_
#endif
    );
  }

  void synchronize() { cudaDeviceSynchronize(); }

  bool fetch_profile(ProfileCounters* out) {
#if PK_PROFILE_RANGES
    if (!out || !d_profile_) {
      return false;
    }
    cudaMemcpy(out, d_profile_, sizeof(ProfileCounters), cudaMemcpyDeviceToHost);
    return true;
#else
    (void)out;
    return false;
#endif
  }

  bool fetch_task_counter(unsigned long long* out) {
#if PK_TASK_ACCOUNTING
    if (!out || !d_task_counter_) {
      return false;
    }
    cudaMemcpy(out, d_task_counter_, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return true;
#else
    (void)out;
    return false;
#endif
  }

  void set_task_accounting_enabled(bool enabled) {
#if PK_TASK_ACCOUNTING
    task_accounting_enabled_ = enabled;
#else
    (void)enabled;
#endif
  }

  void set_perf_enabled(bool enabled) {
#if PK_PERF_COUNTERS
    perf_enabled_ = enabled;
#else
    (void)enabled;
#endif
  }

  int last_launch_blocks() const {
#if PK_PERF_COUNTERS
    return last_launch_blocks_;
#else
    return 0;
#endif
  }

#if PK_PERF_COUNTERS
  bool fetch_perf_blocks(std::vector<PerfBlock>& out) {
    if (!perf_enabled_ || !d_perf_blocks_ || last_launch_blocks_ <= 0) {
      return false;
    }
    out.resize(static_cast<size_t>(last_launch_blocks_));
    cudaMemcpy(out.data(), d_perf_blocks_, sizeof(PerfBlock) * last_launch_blocks_,
               cudaMemcpyDeviceToHost);
    return true;
  }
#endif

  bool fetch_segment_state(std::vector<int>& completed, int* seg_lo_out, int* seg_hi_out,
                           int* window_size_out) {
#if PK_SEGMENT_GATE
    if (!d_segments_ || !d_seg_lo_ || !d_seg_hi_ || !d_window_size_ || !d_segment_completed_ ||
        num_segments_ <= 0) {
      return false;
    }
    completed.resize(static_cast<size_t>(num_segments_));
    cudaMemcpy(completed.data(), d_segment_completed_, sizeof(int) * num_segments_,
               cudaMemcpyDeviceToHost);
    if (seg_lo_out) {
      cudaMemcpy(seg_lo_out, d_seg_lo_, sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (seg_hi_out) {
      cudaMemcpy(seg_hi_out, d_seg_hi_, sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (window_size_out) {
      cudaMemcpy(window_size_out, d_window_size_, sizeof(int), cudaMemcpyDeviceToHost);
    }
    return true;
#else
    (void)completed;
    (void)seg_lo_out;
    (void)seg_hi_out;
    (void)window_size_out;
    return false;
#endif
  }

  static int get_num_sms() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.multiProcessorCount;
  }

private:
  void cleanup() {
    if (d_tasks_) {
      cudaFree(d_tasks_);
      d_tasks_ = nullptr;
    }
    if (d_next_index_) {
      cudaFree(d_next_index_);
      d_next_index_ = nullptr;
    }
    if (d_deps_) {
      cudaFree(d_deps_);
      d_deps_ = nullptr;
    }
#if PK_PROFILE_RANGES
    if (d_profile_) {
      cudaFree(d_profile_);
      d_profile_ = nullptr;
    }
#endif

#if PK_TASK_ACCOUNTING
    if (d_task_counter_) {
      cudaFree(d_task_counter_);
      d_task_counter_ = nullptr;
    }
#endif

#if PK_PERF_COUNTERS
    if (d_perf_blocks_) {
      cudaFree(d_perf_blocks_);
      d_perf_blocks_ = nullptr;
    }
    perf_blocks_capacity_ = 0;
#endif

#if PK_CLAIMED_READY_FIRST
    if (d_claimed_) {
      cudaFree(d_claimed_);
      d_claimed_ = nullptr;
    }
    claimed_capacity_ = 0;
    if (d_cursor_) {
      cudaFree(d_cursor_);
      d_cursor_ = nullptr;
    }
    if (d_claimed_total_) {
      cudaFree(d_claimed_total_);
      d_claimed_total_ = nullptr;
    }
#endif

#if PK_SEGMENT_GATE
    if (d_segments_) {
      cudaFree(d_segments_);
      d_segments_ = nullptr;
    }
    segments_capacity_ = 0;
    if (d_seg_lo_) {
      cudaFree(d_seg_lo_);
      d_seg_lo_ = nullptr;
    }
    if (d_seg_hi_) {
      cudaFree(d_seg_hi_);
      d_seg_hi_ = nullptr;
    }
    if (d_window_size_) {
      cudaFree(d_window_size_);
      d_window_size_ = nullptr;
    }
    if (d_segment_completed_) {
      cudaFree(d_segment_completed_);
      d_segment_completed_ = nullptr;
    }
    num_segments_ = 0;
#endif
  }

  Task* d_tasks_{nullptr};
  int* d_next_index_{nullptr};
  DependencyState* d_deps_{nullptr};
#if PK_PROFILE_RANGES
  ProfileCounters* d_profile_{nullptr};
#endif
#if PK_TASK_ACCOUNTING
  unsigned long long* d_task_counter_{nullptr};
  bool task_accounting_enabled_{false};
#endif
#if PK_PERF_COUNTERS
  PerfBlock* d_perf_blocks_{nullptr};
  int perf_blocks_capacity_{0};
  int last_launch_blocks_{0};
  bool perf_enabled_{false};
#endif
#if PK_CLAIMED_READY_FIRST
  unsigned int* d_claimed_{nullptr};
  int claimed_capacity_{0};
  unsigned int* d_cursor_{nullptr};
  unsigned int* d_claimed_total_{nullptr};
#endif
#if PK_SEGMENT_GATE
  SegmentDesc* d_segments_{nullptr};
  int segments_capacity_{0};
  int num_segments_{0};
  int* d_seg_lo_{nullptr};
  int* d_seg_hi_{nullptr};
  int* d_window_size_{nullptr};
  int* d_segment_completed_{nullptr};
  std::vector<SegmentDesc> host_segments_;
  int segment_window_size_{PK_SEGMENT_WINDOW_SIZE};
#endif
  int max_tasks_{0};
  int num_tasks_{0};

  void reset_dependencies() {
    if (d_deps_) {
      cudaMemset(d_deps_, 0, sizeof(DependencyState));
    }
  }
};

} // namespace pk
