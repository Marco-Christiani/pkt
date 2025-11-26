#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../include/ops/linear_bwd.cuh"
#include "../include/ops/linear_fwd.cuh"
#include "../include/ops/mse.cuh"
#include "../include/ops/sgd.cuh"
#include "../include/ops/zero.cuh"
#include "../include/runtime.hpp"
#include "include/op_traits.cuh"

void check_cuda(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at " << msg << ": " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

#include <cmath>
#include <random>
#include <vector>

struct TestData {
  std::vector<float> h_W;
  std::vector<float> h_x;
  std::vector<float> h_target;
};

// m outputs, n inputs
TestData make_linear_test_data(int m, int n) {
  std::mt19937 rng(42);

  // Xavier/Glorot variance: sqrt(2/(m+n))
  float scale = std::sqrt(2.0f / float(m + n));
  std::normal_distribution<float> wdist(0.0f, scale);
  std::normal_distribution<float> xdist(0.0f, 1.0f);
  std::normal_distribution<float> noise_dist(0.0f, 0.01f);

  std::vector<float> h_W(m * n);
  std::vector<float> h_x(n);
  std::vector<float> h_target(m);

  // W ~ N(0, scale^2)
  for (auto& w : h_W)
    w = wdist(rng);

  // x ~ N(0,1)
  for (auto& x : h_x)
    x = xdist(rng);

  // t = W x + noise
  for (int i = 0; i < m; i++) {
    float acc = 0.0f;
    const float* Wi = &h_W[i * n];
    for (int j = 0; j < n; j++)
      acc += Wi[j] * h_x[j];
    h_target[i] = acc + noise_dist(rng);
  }

  return {.h_W = h_W, .h_x = h_x, .h_target = h_target};
}

float host_mse(const std::vector<float>& y, const std::vector<float>& target) {
  // Mean squared error (no 0.5 factor) to match device loss scaling
  float acc = 0.f;
  for (size_t i = 0; i < y.size(); ++i) {
    float d = y[i] - target[i];
    acc += d * d;
  }
  return acc / static_cast<float>(y.size());
}

struct CPUTrainResult {
  std::vector<float> W;
  float final_loss;
  std::vector<float> loss_curve;
};

/*
  Trains:
      y = W x
      loss = 0.5 * ||y - target||^2
      SGD update: W -= lr * dW
  Single-sample loop.
*/
CPUTrainResult cpu_train_linear(std::vector<float> W,             // shape [m*n]
                                const std::vector<float>& x,      // shape [n]
                                const std::vector<float>& target, // shape [m]
                                int m, int n, int steps, float lr) {
  std::vector<float> y(m);
  std::vector<float> dy(m);
  std::vector<float> loss_curve;
  loss_curve.reserve(steps);

  for (int step = 0; step < steps; ++step) {
    // ----- Forward: y = W x -----
    for (int i = 0; i < m; ++i) {
      float acc = 0.f;
      const float* Wi = &W[i * n];
      for (int j = 0; j < n; ++j)
        acc += Wi[j] * x[j];
      y[i] = acc;
    }

    // ----- Loss + dy = y - target -----
    float loss = 0.f;
    for (int i = 0; i < m; ++i) {
      float diff = y[i] - target[i];
      dy[i] = 2.0f * diff / float(m);
      loss += diff * diff / float(m);
    }

    loss_curve.push_back(loss);

    // ----- Backward: dW = dy ⊗ x -----
    // SGD update: W -= lr * dW
    for (int i = 0; i < m; ++i) {
      float dyi = dy[i];
      float* Wi = &W[i * n];
      for (int j = 0; j < n; ++j) {
        Wi[j] -= lr * (dyi * x[j]);
      }
    }
  }

  return {.W = W, .final_loss = loss_curve.back(), .loss_curve = loss_curve};
}

int main() {
  constexpr int m = 64;
  constexpr int n = 128;
  constexpr int num_steps = 100;
  constexpr float lr = 0.001f;
  constexpr int tasks_per_step = 4 + 2; // +2 for zero mem

  std::cout << "Megakernel Training Demo\n";
  std::cout << "========================\n";
  std::cout << "Model: y = Wx, W:[" << m << "," << n << "]\n";
  std::cout << "Steps: " << num_steps << ", LR: " << lr << "\n";
  std::cout << "Tasks per step: " << tasks_per_step << "\n";
  std::cout << "Kernel launches: " << num_steps << "\n\n";

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Host data
  const auto td = make_linear_test_data(m, n);

  // Do a host-side forward to get baseline loss
  std::vector<float> h_y0(m);
  for (int i = 0; i < m; ++i) {
    float acc = 0.f;
    const float* Wi = &td.h_W[i * n];
    for (int j = 0; j < n; ++j)
      acc += Wi[j] * td.h_x[j];
    h_y0[i] = acc;
  }

  float initial_loss = host_mse(h_y0, td.h_target);
  std::cout << "Initial host loss: " << initial_loss << "\n";

  auto cpu_res = cpu_train_linear(td.h_W, td.h_x, td.h_target, m, n, num_steps, lr);

  std::cout << "CPU final loss: " << cpu_res.final_loss << "\n";
  std::cout << "CPU first step loss: " << cpu_res.loss_curve[0] << "\n";
  std::cout << "x[0:5] = ";
  for (int i = 0; i < 5; i++)
    std::cout << td.h_x[i] << " ";
  std::cout << "\n";
  std::cout << "target[0:5] = ";
  for (int i = 0; i < 5; i++)
    std::cout << td.h_target[i] << " ";
  std::cout << "\n";

  // Expected: y[0] = sum(W[0,j] * x[j]) for j in 0..n-1
  float expected_y0 = 0;
  for (int j = 0; j < n; j++)
    expected_y0 += td.h_W[j] * td.h_x[j];
  printf("Host expected y[0]=%f\n", expected_y0);

  // Device allocations
  float *d_W, *d_x, *d_y, *d_target, *d_dy, *d_dW, *d_loss;
  cudaMalloc(&d_W, m * n * sizeof(float));
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, m * sizeof(float));
  cudaMalloc(&d_target, m * sizeof(float));
  cudaMalloc(&d_dy, m * sizeof(float));
  cudaMalloc(&d_dW, m * n * sizeof(float));
  cudaMalloc(&d_loss, sizeof(float));
  check_cuda("malloc");

  cudaMemcpy(d_W, td.h_W.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, td.h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_target, td.h_target.data(), m * sizeof(float), cudaMemcpyHostToDevice);
  // dW and loss will be zeroed in the training loop
  check_cuda("memcpy H2D");

  // Initialize runtime with capacity for a single step worth of tasks
  pk::Runtime runtime;
  runtime.initialize(tasks_per_step);

  // === SINGLE KERNEL LAUNCH ===
  std::cout << "Launching megakernel with " << num_steps << " launches ("
            << tasks_per_step << " tasks per step)...\n";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int step = 0; step < num_steps; ++step) {
    // Debug first step
    if (step == 0) {
      float h_y_debug[5], h_dy_debug[5], h_dW_debug[5];
      cudaMemcpy(h_y_debug, d_y, 5 * sizeof(float), cudaMemcpyDeviceToHost);
      std::cout << "Step 0 - Before: y[0:5] = ";
      for (int i = 0; i < 5; i++)
        std::cout << h_y_debug[i] << " ";
      std::cout << "\n";
    }

    std::vector<pk::Task> tasks(tasks_per_step);

    // Task 0: Zero loss
    pk::ZeroMemoryArgs zero_loss{.ptr = d_loss, .size = 1};
    pk::encode_args(tasks[0], pk::OpCode::ZeroMemory, zero_loss);

    // Task 1: Forward
    pk::LinearForwardArgs fwd_args{.W = d_W, .x = d_x, .y = d_y, .m = m, .n = n};
    pk::encode_args(tasks[1], pk::OpCode::LinearForward, fwd_args);

    // Task 2: Loss + dy
    pk::MSELossArgs loss_args{.y = d_y, .target = d_target, .dy = d_dy, .loss = d_loss, .m = m};
    pk::encode_args(tasks[2], pk::OpCode::MSELoss, loss_args);

    // Task 3: Zero dW (must happen BEFORE backward accumulates into it)
    pk::ZeroMemoryArgs zero_dW{.ptr = d_dW, .size = m * n};
    pk::encode_args(tasks[3], pk::OpCode::ZeroMemory, zero_dW);

    // Task 4: Backward (dW += outer(dy, x))
    pk::LinearBackwardArgs bwd_args{.dy = d_dy, .x = d_x, .dW = d_dW, .m = m, .n = n};
    pk::encode_args(tasks[4], pk::OpCode::LinearBackward, bwd_args);

    // Task 5: SGD update (W -= lr * dW)
    pk::SGDUpdateArgs sgd_args{.W = d_W, .dW = d_dW, .lr = lr, .size = m * n};
    pk::encode_args(tasks[5], pk::OpCode::SGDUpdate, sgd_args);

    runtime.submit_tasks(tasks);
    runtime.launch(1); // single block processes tasks sequentially
    runtime.synchronize(); // enforce step ordering

    // Debug first and last steps
    if (step == 0 || step == num_steps - 1) {
      float h_loss_debug;
      float h_y_debug[5], h_dy_debug[5], h_dW_debug[5], h_W_debug[5];
      cudaMemcpy(&h_loss_debug, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_y_debug, d_y, 5 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_dy_debug, d_dy, 5 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_dW_debug, d_dW, 5 * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_W_debug, d_W, 5 * sizeof(float), cudaMemcpyDeviceToHost);
      std::cout << "Step " << step << " - After:\n";
      std::cout << "  loss = " << h_loss_debug << "\n";
      std::cout << "  y[0:5] = ";
      for (int i = 0; i < 5; i++)
        std::cout << h_y_debug[i] << " ";
      std::cout << "\n";
      std::cout << "  W[0:5] = ";
      for (int i = 0; i < 5; i++)
        std::cout << h_W_debug[i] << " ";
      std::cout << "\n";
      std::cout << "  dy[0:5] = ";
      for (int i = 0; i < 5; i++)
        std::cout << h_dy_debug[i] << " ";
      std::cout << "\n";
      std::cout << "  dW[0:5] = ";
      for (int i = 0; i < 5; i++)
        std::cout << h_dW_debug[i] << " ";
      std::cout << "\n";
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  // Read final loss
  float h_loss;
  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

  // Recompute loss on host using trained GPU weights for a stable comparison
  std::vector<float> h_W_out(m * n);
  cudaMemcpy(h_W_out.data(), d_W, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> h_y_final(m);
  for (int i = 0; i < m; ++i) {
    float acc = 0.f;
    const float* Wi = &h_W_out[i * n];
    for (int j = 0; j < n; ++j)
      acc += Wi[j] * td.h_x[j];
    h_y_final[i] = acc;
  }
  float host_recomputed_loss = host_mse(h_y_final, td.h_target);

  std::cout << "\nResults:\n";
  std::cout << "--------\n";
  std::cout << "Time: " << elapsed_ms << " ms\n";
  std::cout << "Time per step: " << (elapsed_ms / num_steps) << " ms\n";
  std::cout << "Final loss (device accumulation): " << h_loss << "\n";
  std::cout << "Final loss (host recompute): " << host_recomputed_loss << "\n";
  std::cout << "CPU final loss: " << cpu_res.final_loss << "\n";

  // Validation
  // Note: loss accumulates across all steps, so we check it's finite and reasonable
  const bool decreased = host_recomputed_loss < initial_loss && std::isfinite(host_recomputed_loss);
  if (decreased) {
    std::cout << "SUCCESS: loss decreased (" << initial_loss << " -> " << host_recomputed_loss
              << ")\n";
  } else {
    std::cout << "NOTE: loss did not decrease (" << initial_loss << " -> "
              << host_recomputed_loss
              << "). Try tweaking lr/steps if you want closer CPU parity.\n";
  }

  cudaFree(d_W);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_target);
  cudaFree(d_dy);
  cudaFree(d_dW);
  cudaFree(d_loss);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
