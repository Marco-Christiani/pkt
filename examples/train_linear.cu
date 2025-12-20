#include <cmath>
#include <cstdint>
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
  std::vector<float> h_x;      // [batch, n]
  std::vector<float> h_target; // [batch, m]
};

// m outputs, n inputs, batch samples
TestData make_linear_test_data(int m, int n, int batch) {
  std::mt19937 rng(42);

  // Xavier/Glorot variance: sqrt(2/(m+n))
  float scale = std::sqrt(2.0f / float(m + n));
  std::normal_distribution<float> wdist(0.0f, scale);
  std::normal_distribution<float> xdist(0.0f, 1.0f);
  std::normal_distribution<float> noise_dist(0.0f, 0.01f);

  std::vector<float> h_W(m * n);
  std::vector<float> h_x(batch * n);
  std::vector<float> h_target(batch * m);

  // W ~ N(0, scale^2)
  for (auto& w : h_W)
    w = wdist(rng);

  // x ~ N(0,1)
  for (auto& x : h_x)
    x = xdist(rng);

  // t = W x + noise, per sample
  for (int b = 0; b < batch; ++b) {
    const float* xb = &h_x[b * n];
    float* tb = &h_target[b * m];
    for (int i = 0; i < m; i++) {
      float acc = 0.0f;
      const float* Wi = &h_W[i * n];
      for (int j = 0; j < n; j++)
        acc += Wi[j] * xb[j];
      tb[i] = acc + noise_dist(rng);
    }
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
      y = W x (batched)
      loss = 0.5 * ||y - target||^2
      SGD update: W -= lr * dW
  Batched loop over "batch" samples per step.
*/
CPUTrainResult cpu_train_linear(std::vector<float> W,                  // shape [m*n]
                                const std::vector<float>& x,           // shape [batch, n]
                                const std::vector<float>& target,      // shape [batch, m]
                                int batch, int m, int n, int steps, float lr) {
  std::vector<float> y(batch * m);
  std::vector<float> dy(batch * m);
  std::vector<float> loss_curve;
  loss_curve.reserve(steps);

  for (int step = 0; step < steps; ++step) {
    // ----- Forward: y = W x -----
    for (int b = 0; b < batch; ++b) {
      const float* xb = &x[b * n];
      float* yb = &y[b * m];
      for (int i = 0; i < m; ++i) {
        float acc = 0.f;
        const float* Wi = &W[i * n];
        for (int j = 0; j < n; ++j)
          acc += Wi[j] * xb[j];
        yb[i] = acc;
      }
    }

    // ----- Loss + dy = y - target -----
    float loss = 0.f;
    const float scale = 2.0f / float(batch * m);
    for (int i = 0; i < batch * m; ++i) {
      float diff = y[i] - target[i];
      dy[i] = scale * diff;
      loss += diff * diff;
    }
    loss /= float(batch * m);

    loss_curve.push_back(loss);

    // ----- Backward: dW = dy ⊗ x -----
    // SGD update: W -= lr * dW
    for (int i = 0; i < m; ++i) {
      float* Wi = &W[i * n];
      for (int j = 0; j < n; ++j) {
        float grad = 0.f;
        for (int b = 0; b < batch; ++b) {
          grad += dy[b * m + i] * x[b * n + j];
        }
        Wi[j] -= lr * grad;
      }
    }
  }

  return {.W = W, .final_loss = loss_curve.back(), .loss_curve = loss_curve};
}

int main() {
  constexpr int m = 64;
  constexpr int n = 128;
  constexpr int batch = 16;
  constexpr int num_steps = 100;
  constexpr float lr = 0.001f;
  constexpr int tasks_per_step = 4 + 2; // +2 for zero mem

  // Dependency buffer ids (tokens only, not tied to physical buffers)
  constexpr uint16_t kBufY = 1;           // Forward output ready
  constexpr uint16_t kBufGradReady = 2;   // Loss + zero-dW prerequisites
  constexpr uint16_t kBufDW = 3;          // dW ready for SGD
  constexpr uint16_t kBufW = 4;           // Weight updates ready

  std::cout << "Megakernel Training Demo\n";
  std::cout << "========================\n";
  std::cout << "Model: y = Wx, W:[" << m << "," << n << "]\n";
  std::cout << "Steps: " << num_steps << ", LR: " << lr << "\n";
  std::cout << "Tasks per step: " << tasks_per_step << "\n";
  std::cout << "Kernel launches: 1\n\n";

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Host data
  const auto td = make_linear_test_data(m, n, batch);

  // Do a host-side forward to get baseline loss
  std::vector<float> h_y0(batch * m);
  for (int b = 0; b < batch; ++b) {
    float* yb = &h_y0[b * m];
    const float* xb = &td.h_x[b * n];
    for (int i = 0; i < m; ++i) {
      float acc = 0.f;
      const float* Wi = &td.h_W[i * n];
      for (int j = 0; j < n; ++j)
        acc += Wi[j] * xb[j];
      yb[i] = acc;
    }
  }

  float initial_loss = host_mse(h_y0, td.h_target);
  std::cout << "Initial host loss: " << initial_loss << "\n";

  auto cpu_res = cpu_train_linear(td.h_W, td.h_x, td.h_target, batch, m, n, num_steps, lr);

  std::cout << "CPU final loss: " << cpu_res.final_loss << "\n";
  std::cout << "CPU first step loss: " << cpu_res.loss_curve[0] << "\n";
  std::cout << "x[0:5] (sample 0) = ";
  for (int i = 0; i < 5; i++)
    std::cout << td.h_x[i] << " ";
  std::cout << "\n";
  std::cout << "target[0:5] (sample 0) = ";
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
  cudaMalloc(&d_x, batch * n * sizeof(float));
  cudaMalloc(&d_y, batch * m * sizeof(float));
  cudaMalloc(&d_target, batch * m * sizeof(float));
  cudaMalloc(&d_dy, batch * m * sizeof(float));
  cudaMalloc(&d_dW, m * n * sizeof(float));
  cudaMalloc(&d_loss, sizeof(float));
  check_cuda("malloc");

  cudaMemcpy(d_W, td.h_W.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, td.h_x.data(), batch * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_target, td.h_target.data(), batch * m * sizeof(float), cudaMemcpyHostToDevice);
  // dW and loss will be zeroed in the training loop
  check_cuda("memcpy H2D");

  // Initialize runtime with capacity for all steps worth of tasks
  pk::Runtime runtime;
  runtime.initialize(num_steps * tasks_per_step);
  const int num_blocks = pk::Runtime::get_num_sms();

  // === SINGLE KERNEL LAUNCH ===
  std::cout << "Launching megakernel with 1 launch (" << tasks_per_step << " tasks per step, "
            << num_blocks << " blocks)...\n";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<pk::Task> tasks;
  tasks.reserve(num_steps * tasks_per_step);

  // Track cumulative ready counts and epochs per buffer
  int ready_y = 0;
  int ready_grad = 0;
  int ready_dw = 0;
  int ready_w = 0;
  int epoch_y = 0;
  int epoch_grad = 0;
  int epoch_dw = 0;
  int epoch_w = 0;

  for (int step = 0; step < num_steps; ++step) {
    // Task 0: Zero loss
    pk::Task zero_loss_task{};
    pk::ZeroMemoryArgs zero_loss{.ptr = d_loss, .size = 1};
    pk::encode_args(zero_loss_task, pk::OpCode::ZeroMemory, zero_loss);
    zero_loss_task.header.buffer_read_id = 0;
    zero_loss_task.header.buffer_write_id = 0;
    zero_loss_task.header.wait_count = 0;
    tasks.push_back(zero_loss_task);

    // Task 1: Forward (reads W generation)
    pk::Task fwd_task{};
    pk::LinearForwardArgs fwd_args{.W = d_W, .x = d_x, .y = d_y, .batch = batch, .m = m, .n = n};
    pk::encode_args(fwd_task, pk::OpCode::LinearForward, fwd_args);
    fwd_task.header.buffer_read_id = kBufW;
    fwd_task.header.buffer_write_id = kBufY;
    fwd_task.header.wait_count = ready_w; // ensure latest W update applied
    fwd_task.header.read_epoch = epoch_w;
    fwd_task.header.write_epoch = epoch_y + 1;
    tasks.push_back(fwd_task);
    ready_y += 1;
    epoch_y += 1;

    // Task 2: Loss + dy (depends on forward output)
    pk::Task loss_task{};
    pk::MSELossArgs loss_args{
        .y = d_y, .target = d_target, .dy = d_dy, .loss = d_loss, .batch = batch, .m = m};
    pk::encode_args(loss_task, pk::OpCode::MSELoss, loss_args);
    loss_task.header.buffer_read_id = kBufY;
    loss_task.header.buffer_write_id = kBufGradReady; // part 1 of grad readiness
    loss_task.header.wait_count = ready_y;
    loss_task.header.read_epoch = epoch_y;
    loss_task.header.write_epoch = epoch_grad + 1;
    tasks.push_back(loss_task);
    ready_grad += 1;
    epoch_grad += 1;

    // Task 3: Zero dW (prereq for backward accumulation)
    pk::Task zero_dW_task{};
    pk::ZeroMemoryArgs zero_dW{.ptr = d_dW, .size = m * n};
    pk::encode_args(zero_dW_task, pk::OpCode::ZeroMemory, zero_dW);
    zero_dW_task.header.buffer_read_id = 0;
    zero_dW_task.header.buffer_write_id = kBufGradReady; // part 2 of grad readiness
    zero_dW_task.header.wait_count = 0;
    zero_dW_task.header.read_epoch = 0;
    zero_dW_task.header.write_epoch = epoch_grad + 1;
    tasks.push_back(zero_dW_task);
    ready_grad += 1;
    epoch_grad += 1;

    // Task 4: Backward (dW += outer(dy, x)) depends on grad readiness
    pk::Task bwd_task{};
    pk::LinearBackwardArgs bwd_args{.dy = d_dy, .x = d_x, .dW = d_dW, .batch = batch, .m = m,
                                    .n = n};
    pk::encode_args(bwd_task, pk::OpCode::LinearBackward, bwd_args);
    bwd_task.header.buffer_read_id = kBufGradReady;
    bwd_task.header.buffer_write_id = kBufDW;
    bwd_task.header.wait_count = ready_grad; // loss + zero_dW complete
    bwd_task.header.read_epoch = epoch_grad;
    bwd_task.header.write_epoch = epoch_dw + 1;
    tasks.push_back(bwd_task);
    ready_dw += 1;
    epoch_dw += 1;

    // Task 5: SGD update (W -= lr * dW) depends on dW for this step
    pk::Task sgd_task{};
    pk::SGDUpdateArgs sgd_args{.W = d_W, .dW = d_dW, .lr = lr, .size = m * n};
    pk::encode_args(sgd_task, pk::OpCode::SGDUpdate, sgd_args);
    sgd_task.header.buffer_read_id = kBufDW;
    sgd_task.header.buffer_write_id = kBufW;
    sgd_task.header.wait_count = ready_dw;
    sgd_task.header.read_epoch = epoch_dw;
    sgd_task.header.write_epoch = epoch_w + 1;
    tasks.push_back(sgd_task);
    ready_w += 1;
    epoch_w += 1;
  }

  cudaEventRecord(start);
  runtime.submit_tasks(tasks);
  runtime.launch(num_blocks);
  runtime.synchronize();
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
