#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace pk::test {

#define CUDA_CHECK(call)                                                                         \
  do {                                                                                           \
    cudaError_t err = (call);                                                                    \
    if (err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                   \
    }                                                                                            \
  } while (0)

#define TEST_ASSERT(cond, msg)                                                  \
  do {                                                                          \
    if (!(cond)) {                                                              \
      fprintf(stderr, "ASSERT FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
      return false;                                                             \
    }                                                                           \
  } while (0)

#define TEST_ASSERT_EQ(a, b, msg)                                                                  \
  do {                                                                                             \
    if ((a) != (b)) {                                                                              \
      fprintf(stderr, "ASSERT_EQ FAILED at %s:%d: %s (got %d, expected %d)\n", __FILE__, __LINE__, \
              msg, (int)(a), (int)(b));                                                            \
      return false;                                                                                \
    }                                                                                              \
  } while (0)

#define TEST_ASSERT_NEAR(a, b, eps, msg)                                                  \
  do {                                                                                    \
    float diff = std::abs((a) - (b));                                                     \
    if (diff > (eps)) {                                                                   \
      fprintf(stderr, "ASSERT_NEAR FAILED at %s:%d: %s (got %f, expected %f, diff %f)\n", \
              __FILE__, __LINE__, msg, (float)(a), (float)(b), diff);                     \
      return false;                                                                       \
    }                                                                                     \
  } while (0)

struct TestResult {
  const char* name;
  bool passed;
};

using TestFunc = bool (*)();

struct TestCase {
  const char* name;
  TestFunc func;
};

inline int run_tests(TestCase* tests, int num_tests) {
  int passed = 0;
  int failed = 0;

  printf("Running %d tests...\n", num_tests);
  printf("================================\n\n");

  for (int i = 0; i < num_tests; ++i) {
    printf("[%d/%d] %s... ", i + 1, num_tests, tests[i].name);
    fflush(stdout);

    bool result = tests[i].func();

    if (result) {
      printf("PASSED\n");
      passed++;
    } else {
      printf("FAILED\n");
      failed++;
    }
  }

  printf("\n================================\n");
  printf("Results: %d passed, %d failed\n", passed, failed);

  return (failed == 0) ? 0 : 1;
}

inline bool cuda_is_available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count <= 0) {
    cudaGetLastError();
    return false;
  }
  err = cudaFree(0);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

template <typename T> T* alloc_device(int count) {
  T* ptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  return ptr;
}

template <typename T> T* alloc_device_zero(int count) {
  T* ptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
  return ptr;
}

template <typename T> void copy_to_device(T* dst, const T* src, int count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> void copy_to_host(T* dst, const T* src, int count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

inline bool verify_near(const float* actual, const float* expected, int count, float eps,
                        float* max_error = nullptr) {
  float max_err = 0.0f;
  bool passed = true;

  for (int i = 0; i < count; ++i) {
    float err = std::abs(actual[i] - expected[i]);
    max_err = std::max(err, max_err);
    if (err > eps) {
      passed = false;
      if (i < 5) { // print first few errors
        fprintf(stderr, "  Mismatch at [%d]: got %f, expected %f (err: %f)\n", i, actual[i],
                expected[i], err);
      }
    }
  }

  if (max_error) {
    *max_error = max_err;
  }

  return passed;
}

} // namespace pk::test
