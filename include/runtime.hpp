#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "kernel.cuh"
#include "queue.cuh"
#include "task.cuh"

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
  }

  void submit_tasks(const std::vector<Task>& tasks) {
    if (tasks.empty())
      return;

    num_tasks_ = static_cast<int>(tasks.size());
    cudaMemcpy(d_tasks_, tasks.data(), num_tasks_ * sizeof(Task), cudaMemcpyHostToDevice);
    cudaMemset(d_next_index_, 0, sizeof(int)); // reset counter
  }

  void launch(int num_blocks, cudaStream_t stream = nullptr) {
    GlobalQueue queue;
    queue.tasks = d_tasks_;
    queue.total = num_tasks_;
    queue.next_index = d_next_index_;

    persistent_kernel<<<num_blocks, Config::kThreadsPerBlock, 0, stream>>>(queue);
  }

  void synchronize() { cudaDeviceSynchronize(); }

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
  }

  Task* d_tasks_{nullptr};
  int* d_next_index_{nullptr};
  int max_tasks_{0};
  int num_tasks_{0};
};

} // namespace mk
