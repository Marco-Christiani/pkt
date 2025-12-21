// Test some runtime components: queue, memory management, multi-SM execution

#include <algorithm>
#include <random>
#include <vector>

#include "../include/ops/axpy.cuh"
#include "../include/queue.cuh"
#include "../include/runtime.hpp"
#include "test_utils.cuh"

using namespace pk;
using namespace pk::test;

bool test_runtime_init() {
  if (!cuda_is_available()) {
    return true;
  }
  Runtime runtime;
  runtime.initialize(100);

  return true;
}

bool test_empty_task_list() {
  if (!cuda_is_available()) {
    return true;
  }
  Runtime runtime;
  runtime.initialize(10);

  std::vector<Task> tasks; // empty
  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  return true;
}

bool test_get_num_sms() {
  if (!cuda_is_available()) {
    return true;
  }
  int num_sms = Runtime::get_num_sms();

  // Should be positive and reasonable (eg 1-128+ SMs)
  TEST_ASSERT(num_sms > 0, "num_sms should be positive");
  TEST_ASSERT(num_sms < 1000, "num_sms should be reasonable");

  return true;
}

bool test_multi_sm_execution() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int num_tasks = 256;
  constexpr int n = 512;

  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 2.0f;
  for (int i = 0; i < num_tasks * n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  float* d_x = alloc_device<float>(num_tasks * n);
  float* d_y = alloc_device<float>(num_tasks * n);
  copy_to_device(d_x, h_x.data(), num_tasks * n);
  copy_to_device(d_y, h_y.data(), num_tasks * n);

  Runtime runtime;
  runtime.initialize(num_tasks);

  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    AxpyArgs args;
    args.x = d_x + i * n;
    args.y = d_y + i * n;
    args.a = alpha;
    args.n = n;
    encode_args(tasks[i], OpCode::Axpy, args);
  }

  runtime.submit_tasks(tasks);

  // Use all SMs
  int num_sms = Runtime::get_num_sms();
  runtime.launch(num_sms);
  runtime.synchronize();

  std::vector<float> h_result(num_tasks * n);
  copy_to_host(h_result.data(), d_y, num_tasks * n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), num_tasks * n, 1e-5f, &max_error);

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

bool test_task_resubmission() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int n = 256;

  std::vector<float> h_x(n);
  std::vector<float> h_y(n);

  for (int i = 0; i < n; ++i) {
    h_x[i] = 1.0f;
    h_y[i] = 0.0f;
  }

  float* d_x = alloc_device<float>(n);
  float* d_y = alloc_device<float>(n);
  copy_to_device(d_x, h_x.data(), n);
  copy_to_device(d_y, h_y.data(), n);

  Runtime runtime;
  runtime.initialize(1);

  std::vector<Task> tasks(1);
  AxpyArgs args;
  args.x = d_x;
  args.y = d_y;
  args.a = 1.0f;
  args.n = n;
  encode_args(tasks[0], OpCode::Axpy, args);

  // First submission: y = 1*x + 0 = 1
  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  // Second submission: y = 1*x + 1 = 2
  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  // Third submission: y = 1*x + 2 = 3
  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  std::vector<float> h_result(n);
  copy_to_host(h_result.data(), d_y, n);

  // all should be 3.0
  for (int i = 0; i < n; ++i) {
    if (std::abs(h_result[i] - 3.0f) > 1e-5f) {
      fprintf(stderr, "Element %d: got %f, expected 3.0\n", i, h_result[i]);
      cudaFree(d_x);
      cudaFree(d_y);
      return false;
    }
  }

  cudaFree(d_x);
  cudaFree(d_y);
  return true;
}

bool test_varying_block_counts() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int num_tasks = 128;
  constexpr int n = 256;

  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y_orig(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(54321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 1.5f;
  for (int i = 0; i < num_tasks * n; ++i) {
    h_x[i] = dist(rng);
    h_y_orig[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y_orig[i];
  }

  float* d_x = alloc_device<float>(num_tasks * n);
  float* d_y = alloc_device<float>(num_tasks * n);
  copy_to_device(d_x, h_x.data(), num_tasks * n);

  int block_counts[] = {1, 2, 4, 8, 16};
  int num_block_configs = sizeof(block_counts) / sizeof(block_counts[0]);

  int max_sms = Runtime::get_num_sms();

  for (int bc = 0; bc < num_block_configs; ++bc) {
    int num_blocks = std::min(block_counts[bc], max_sms);

    // Reset y
    copy_to_device(d_y, h_y_orig.data(), num_tasks * n);

    Runtime runtime;
    runtime.initialize(num_tasks);

    std::vector<Task> tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i) {
      AxpyArgs args;
      args.x = d_x + i * n;
      args.y = d_y + i * n;
      args.a = alpha;
      args.n = n;
      encode_args(tasks[i], OpCode::Axpy, args);
    }

    runtime.submit_tasks(tasks);
    runtime.launch(num_blocks);
    runtime.synchronize();

    std::vector<float> h_result(num_tasks * n);
    copy_to_host(h_result.data(), d_y, num_tasks * n);

    float max_error;
    bool passed = verify_near(h_result.data(), h_expected.data(), num_tasks * n, 1e-5f, &max_error);

    if (!passed) {
      fprintf(stderr, "Failed with %d blocks (max error: %e)\n", num_blocks, max_error);
      cudaFree(d_x);
      cudaFree(d_y);
      return false;
    }
  }

  cudaFree(d_x);
  cudaFree(d_y);
  return true;
}

// Test queue chunking with task count not divisible by chunk size
bool test_non_aligned_task_count() {
  if (!cuda_is_available()) {
    return true;
  }
  // Config::kChunkSize is 32, test with 100 tasks
  constexpr int num_tasks = 100;
  constexpr int n = 128;

  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(99999);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 0.5f;
  for (int i = 0; i < num_tasks * n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  float* d_x = alloc_device<float>(num_tasks * n);
  float* d_y = alloc_device<float>(num_tasks * n);
  copy_to_device(d_x, h_x.data(), num_tasks * n);
  copy_to_device(d_y, h_y.data(), num_tasks * n);

  Runtime runtime;
  runtime.initialize(num_tasks);

  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    AxpyArgs args;
    args.x = d_x + i * n;
    args.y = d_y + i * n;
    args.a = alpha;
    args.n = n;
    encode_args(tasks[i], OpCode::Axpy, args);
  }

  runtime.submit_tasks(tasks);
  runtime.launch(Runtime::get_num_sms());
  runtime.synchronize();

  std::vector<float> h_result(num_tasks * n);
  copy_to_host(h_result.data(), d_y, num_tasks * n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), num_tasks * n, 1e-5f, &max_error);

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

bool test_large_task_count() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int num_tasks = 5000;
  constexpr int n = 64;

  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(11111);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 1.0f;
  for (int i = 0; i < num_tasks * n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  float* d_x = alloc_device<float>(num_tasks * n);
  float* d_y = alloc_device<float>(num_tasks * n);
  copy_to_device(d_x, h_x.data(), num_tasks * n);
  copy_to_device(d_y, h_y.data(), num_tasks * n);

  Runtime runtime;
  runtime.initialize(num_tasks);

  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    AxpyArgs args;
    args.x = d_x + i * n;
    args.y = d_y + i * n;
    args.a = alpha;
    args.n = n;
    encode_args(tasks[i], OpCode::Axpy, args);
  }

  runtime.submit_tasks(tasks);
  runtime.launch(Runtime::get_num_sms());
  runtime.synchronize();

  std::vector<float> h_result(num_tasks * n);
  copy_to_host(h_result.data(), d_y, num_tasks * n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), num_tasks * n, 1e-5f, &max_error);

  if (!passed) {
    fprintf(stderr, "Max error: %e\n", max_error);
  }

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

int main() {
  printf("Runtime Integration Tests\n");
  printf("GPU SMs available: %d\n\n", Runtime::get_num_sms());

  TestCase tests[] = {
      {.name = "runtime_init", .func = test_runtime_init},
      {.name = "empty_task_list", .func = test_empty_task_list},
      {.name = "get_num_sms", .func = test_get_num_sms},
      {.name = "multi_sm_execution", .func = test_multi_sm_execution},
      {.name = "task_resubmission", .func = test_task_resubmission},
      {.name = "varying_block_counts", .func = test_varying_block_counts},
      {.name = "non_aligned_task_count", .func = test_non_aligned_task_count},
      {.name = "large_task_count", .func = test_large_task_count},
  };

  return run_tests(tests, sizeof(tests) / sizeof(tests[0]));
}
