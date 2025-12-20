#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "dependency.cuh"
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
    cudaMalloc(&d_deps_, sizeof(DependencyState));
    reset_dependencies();
  }

  void submit_tasks(const std::vector<Task>& tasks) {
    if (tasks.empty())
      return;

    num_tasks_ = static_cast<int>(tasks.size());
    cudaMemcpy(d_tasks_, tasks.data(), num_tasks_ * sizeof(Task), cudaMemcpyHostToDevice);
    cudaMemset(d_next_index_, 0, sizeof(int)); // reset counter
    reset_dependencies();                      // fresh dependency counts per submission
  }

  void launch(int num_blocks, cudaStream_t stream = nullptr) {
    GlobalQueue queue;
    queue.tasks = d_tasks_;
    queue.total = num_tasks_;
    queue.next_index = d_next_index_;

    persistent_kernel<<<num_blocks, Config::kThreadsPerBlock, 0, stream>>>(queue, d_deps_);
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
    if (d_deps_) {
      cudaFree(d_deps_);
      d_deps_ = nullptr;
    }
  }

  Task* d_tasks_{nullptr};
  int* d_next_index_{nullptr};
  DependencyState* d_deps_{nullptr};
  int max_tasks_{0};
  int num_tasks_{0};

  void reset_dependencies() {
    if (d_deps_) {
      cudaMemset(d_deps_, 0, sizeof(DependencyState));
    }
  }
};

} // namespace mk
