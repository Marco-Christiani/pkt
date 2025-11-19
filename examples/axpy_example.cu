#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../include/ops/axpy.cuh"
#include "../include/runtime.hpp"

void check_cuda_error(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error after " << msg << ": " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  constexpr int num_tasks = 1000;
  constexpr int elements_per_task = 1024;

  std::cout << "Megakernel AXPY Example\n";
  std::cout << "======================\n";
  std::cout << "Tasks: " << num_tasks << "\n";
  std::cout << "Elements per task: " << elements_per_task << "\n\n";

  pk::Runtime runtime;
  runtime.initialize(num_tasks);

  std::vector<float> h_x(num_tasks * elements_per_task);
  std::vector<float> h_y(num_tasks * elements_per_task);
  std::vector<float> h_y_ref(num_tasks * elements_per_task);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < num_tasks * elements_per_task; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_y_ref[i] = h_y[i];
  }

  float* d_x;
  float* d_y;
  cudaMalloc(&d_x, num_tasks * elements_per_task * sizeof(float));
  cudaMalloc(&d_y, num_tasks * elements_per_task * sizeof(float));
  check_cuda_error("malloc");

  cudaMemcpy(d_x, h_x.data(), num_tasks * elements_per_task * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), num_tasks * elements_per_task * sizeof(float),
             cudaMemcpyHostToDevice);
  check_cuda_error("memcpy H2D");

  // Create and submit tasks
  std::vector<pk::Task> tasks(num_tasks);
  const float alpha = 2.5f;
  for (int i = 0; i < num_tasks; ++i) {
    pk::AxpyArgs args;
    args.x = d_x + i * elements_per_task;
    args.y = d_y + i * elements_per_task;
    args.a = alpha;
    args.n = elements_per_task;

    pk::encode_args(tasks[i], pk::OpCode::Axpy, args);
  }
  runtime.submit_tasks(tasks);

  // 1 block / SM
  const int num_sms = pk::Runtime::get_num_sms();
  std::cout << "Launching on " << num_sms << " SMs\n";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  runtime.launch(num_sms);
  cudaEventRecord(stop);

  runtime.synchronize();
  check_cuda_error("kernel launch");

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaMemcpy(h_y.data(), d_y, num_tasks * elements_per_task * sizeof(float),
             cudaMemcpyDeviceToHost);
  check_cuda_error("memcpy D2H");

  // Verify
  bool correct = true;
  float max_error = 0.0f;

  for (int i = 0; i < num_tasks * elements_per_task; ++i) {
    float expected = alpha * h_x[i] + h_y_ref[i];
    float error = std::abs(h_y[i] - expected);
    max_error = std::max(max_error, error);

    if (error > 1e-5f) {
      correct = false;
      if (max_error == error) {
        std::cout << "Error at index " << i << ": "
                  << "expected " << expected << ", got " << h_y[i] << " (error: " << error << ")\n";
      }
    }
  }

  std::cout << "\nResults:\n";
  std::cout << "--------\n";
  std::cout << "Time: " << elapsed_ms << " ms\n";
  std::cout << "Throughput: "
            << (num_tasks * elements_per_task * sizeof(float) * 2 / elapsed_ms / 1e6) << " GB/s\n";
  std::cout << "Max error: " << max_error << "\n";
  std::cout << "Status: " << (correct ? "PASSED ✓" : "FAILED ✗") << "\n";

  cudaFree(d_x);
  cudaFree(d_y);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return correct ? 0 : 1;
}
