// end to end pipeline test

#include <random>
#include <vector>

#include "../include/ops/axpy.cuh"
#include "../include/runtime.hpp"
#include "test_utils.cuh"

using namespace pk;
using namespace pk::test;

bool test_single_axpy() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int n = 1024;

  // Host data
  std::vector<float> h_x(n);
  std::vector<float> h_y(n);
  std::vector<float> h_expected(n);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 2.5f;
  for (int i = 0; i < n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  // Device data
  float* d_x = alloc_device<float>(n);
  float* d_y = alloc_device<float>(n);
  copy_to_device(d_x, h_x.data(), n);
  copy_to_device(d_y, h_y.data(), n);

  // Setup runtime w a task
  Runtime runtime;
  runtime.initialize(1);

  std::vector<Task> tasks(1);
  AxpyArgs args;
  args.x = d_x;
  args.y = d_y;
  args.a = alpha;
  args.n = n;
  encode_args(tasks[0], OpCode::Axpy, args);

  // Execute
  runtime.submit_tasks(tasks);
  runtime.launch(1); // single block
  runtime.synchronize();

  // Verify
  std::vector<float> h_result(n);
  copy_to_host(h_result.data(), d_y, n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), n, 1e-5f, &max_error);

  if (!passed) {
    fprintf(stderr, "Max error: %e\n", max_error);
  }

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

bool test_multiple_axpy() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int num_tasks = 100;
  constexpr int n = 1024;

  // Host data
  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const float alpha = 3.14f;
  for (int i = 0; i < num_tasks * n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
  }

  // Device data
  float* d_x = alloc_device<float>(num_tasks * n);
  float* d_y = alloc_device<float>(num_tasks * n);
  copy_to_device(d_x, h_x.data(), num_tasks * n);
  copy_to_device(d_y, h_y.data(), num_tasks * n);

  // Create runtime and tasks
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

  // Execute on multiple SMs
  runtime.submit_tasks(tasks);
  int num_sms = Runtime::get_num_sms();
  runtime.launch(num_sms);
  runtime.synchronize();

  // Verify
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

bool test_varying_sizes() {
  if (!cuda_is_available()) {
    return true;
  }
  int sizes[] = {32, 128, 512, 1024, 2048};
  int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

  for (int s = 0; s < num_sizes; ++s) {
    int n = sizes[s];

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_expected(n);

    std::mt19937 rng(s);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float alpha = 1.5f;
    for (int i = 0; i < n; ++i) {
      h_x[i] = dist(rng);
      h_y[i] = dist(rng);
      h_expected[i] = alpha * h_x[i] + h_y[i];
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
    args.a = alpha;
    args.n = n;
    encode_args(tasks[0], OpCode::Axpy, args);

    runtime.submit_tasks(tasks);
    runtime.launch(1);
    runtime.synchronize();

    std::vector<float> h_result(n);
    copy_to_host(h_result.data(), d_y, n);

    float max_error;
    bool passed = verify_near(h_result.data(), h_expected.data(), n, 1e-5f, &max_error);

    cudaFree(d_x);
    cudaFree(d_y);

    if (!passed) {
      fprintf(stderr, "Failed for size %d (max error: %e)\n", n, max_error);
      return false;
    }
  }

  return true;
}

// Test with alpha = 0 (should just copy y)
bool test_alpha_zero() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int n = 256;

  std::vector<float> h_x(n, 1.0f);
  std::vector<float> h_y(n);
  std::vector<float> h_expected(n);

  for (int i = 0; i < n; ++i) {
    h_y[i] = static_cast<float>(i);
    h_expected[i] = h_y[i]; // alpha=0, so result = y
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
  args.a = 0.0f;
  args.n = n;
  encode_args(tasks[0], OpCode::Axpy, args);

  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  std::vector<float> h_result(n);
  copy_to_host(h_result.data(), d_y, n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), n, 1e-6f, &max_error);

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

// Test with negative alpha
bool test_negative_alpha() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int n = 512;

  std::vector<float> h_x(n);
  std::vector<float> h_y(n);
  std::vector<float> h_expected(n);

  std::mt19937 rng(999);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  const float alpha = -2.0f;
  for (int i = 0; i < n; ++i) {
    h_x[i] = dist(rng);
    h_y[i] = dist(rng);
    h_expected[i] = alpha * h_x[i] + h_y[i];
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
  args.a = alpha;
  args.n = n;
  encode_args(tasks[0], OpCode::Axpy, args);

  runtime.submit_tasks(tasks);
  runtime.launch(1);
  runtime.synchronize();

  std::vector<float> h_result(n);
  copy_to_host(h_result.data(), d_y, n);

  float max_error;
  bool passed = verify_near(h_result.data(), h_expected.data(), n, 1e-5f, &max_error);

  cudaFree(d_x);
  cudaFree(d_y);

  return passed;
}

// Test many small tasks to stress the queue
bool test_many_small_tasks() {
  if (!cuda_is_available()) {
    return true;
  }
  constexpr int num_tasks = 1000;
  constexpr int n = 32;

  std::vector<float> h_x(num_tasks * n);
  std::vector<float> h_y(num_tasks * n);
  std::vector<float> h_expected(num_tasks * n);

  std::mt19937 rng(777);
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
  int num_sms = Runtime::get_num_sms();
  runtime.launch(num_sms);
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
  printf("AXPY Integration Tests\n");
  printf("GPU SMs available: %d\n\n", Runtime::get_num_sms());

  TestCase tests[] = {
      {.name = "single_axpy", .func = test_single_axpy},
      {.name = "multiple_axpy", .func = test_multiple_axpy},
      {.name = "varying_sizes", .func = test_varying_sizes},
      {.name = "alpha_zero", .func = test_alpha_zero},
      {.name = "negative_alpha", .func = test_negative_alpha},
      {.name = "many_small_tasks", .func = test_many_small_tasks},
  };

  return run_tests(tests, sizeof(tests) / sizeof(tests[0]));
}
