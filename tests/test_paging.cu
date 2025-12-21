// Tests for SMEM paging with LinearForward

#include <vector>

#include "../include/runtime.hpp"
#include "../include/ops/linear_fwd.cuh"
#include "test_utils.cuh"

using namespace pk;
using namespace pk::test;

bool test_linear_forward() {
  if (!cuda_is_available()) {
    return true;
  }
  const int m = 8;
  const int n = 8; // fits in one page
  const int batch = 2;
  const int tasks = 1;

  std::vector<float> h_W(m * n);
  std::vector<float> h_x(batch * n);
  std::vector<float> h_y(batch * m, 0.0f);

  // Simple data: W = identity on first m rows, x = 1..n per batch
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      h_W[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }
  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < n; ++j) {
      h_x[b * n + j] = static_cast<float>(j + 1);
    }
  }

  float *d_W, *d_x, *d_y;
  cudaMalloc(&d_W, m * n * sizeof(float));
  cudaMalloc(&d_x, batch * n * sizeof(float));
  cudaMalloc(&d_y, batch * m * sizeof(float));

  cudaMemcpy(d_W, h_W.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), batch * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_y, 0, batch * m * sizeof(float));

  Runtime rt;
  rt.initialize(tasks);

  std::vector<Task> tlist(tasks);
  LinearForwardArgs args{
      .W = d_W,
      .x = d_x,
      .y = d_y,
      .batch = batch,
      .m = m,
      .n = n,
      .tile_m_start = 0,
      .tile_m_count = batch,
      .tile_n_start = 0,
      .tile_n_count = m};
  encode_args(tlist[0], OpCode::LinearForward, args);

  rt.submit_tasks(tlist);
  rt.launch(1);
  rt.synchronize();

  cudaMemcpy(h_y.data(), d_y, batch * m * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_W);
  cudaFree(d_x);
  cudaFree(d_y);

  // Validate y matches input (due to identity W)
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < m; ++i) {
      float expected = h_x[b * n + i];
      TEST_ASSERT_NEAR(h_y[b * m + i], expected, 1e-5f, "matvec mismatch");
    }
  }

  return true;
}

int main() {
  TestCase tests[] = {{.name = "linear_forward", .func = test_linear_forward}};
  return run_tests(tests, sizeof(tests) / sizeof(tests[0]));
}
