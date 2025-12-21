#include <cmath>
#include <cstdint>
#include <cstring>
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

int main(int argc, char** argv) {
  int m = 1024;
  int n = 1024;
  int batch = 64;
  int num_steps = 20;
  float lr = 0.001f;
  int fwd_tile_m = 2;
  int fwd_tile_n = 128;
  int bwd_tile_m = 64;
  int bwd_tile_n = 128;
  int repeats = 1;
  bool skip_cpu = false;

  auto parse_int = [&](const char* s, int* out) {
    if (!s || !*s) {
      return false;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') {
      return false;
    }
    *out = static_cast<int>(v);
    return true;
  };

  auto parse_float = [&](const char* s, float* out) {
    if (!s || !*s) {
      return false;
    }
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (!end || *end != '\0') {
      return false;
    }
    *out = v;
    return true;
  };

  auto usage = [&]() {
    std::cout << "train_linear options\n";
    std::cout << "  --m <int>\n";
    std::cout << "  --n <int>\n";
    std::cout << "  --batch <int>\n";
    std::cout << "  --steps <int>\n";
    std::cout << "  --repeats <int>\n";
    std::cout << "  --lr <float>\n";
    std::cout << "  --fwd-tile-n <int>\n";
    std::cout << "  --fwd-tile-m <int>\n";
    std::cout << "  --bwd-tile-m <int>\n";
    std::cout << "  --bwd-tile-n <int>\n";
    std::cout << "  --skip-cpu\n";
  };

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        usage();
        exit(2);
      }
      return argv[++i];
    };

    if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
      usage();
      return 0;
    } else if (!std::strcmp(a, "--m")) {
      if (!parse_int(need(a), &m)) {
        std::cerr << "Bad value for --m\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--n")) {
      if (!parse_int(need(a), &n)) {
        std::cerr << "Bad value for --n\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--batch")) {
      if (!parse_int(need(a), &batch)) {
        std::cerr << "Bad value for --batch\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--steps")) {
      if (!parse_int(need(a), &num_steps)) {
        std::cerr << "Bad value for --steps\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--repeats")) {
      if (!parse_int(need(a), &repeats)) {
        std::cerr << "Bad value for --repeats\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--lr")) {
      if (!parse_float(need(a), &lr)) {
        std::cerr << "Bad value for --lr\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--fwd-tile-m")) {
      if (!parse_int(need(a), &fwd_tile_m)) {
        std::cerr << "Bad value for --fwd-tile-m\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--fwd-tile-n")) {
      if (!parse_int(need(a), &fwd_tile_n)) {
        std::cerr << "Bad value for --fwd-tile-n\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--bwd-tile-m")) {
      if (!parse_int(need(a), &bwd_tile_m)) {
        std::cerr << "Bad value for --bwd-tile-m\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--bwd-tile-n")) {
      if (!parse_int(need(a), &bwd_tile_n)) {
        std::cerr << "Bad value for --bwd-tile-n\n";
        return 2;
      }
    } else if (!std::strcmp(a, "--skip-cpu")) {
      skip_cpu = true;
    } else {
      std::cerr << "Unknown arg " << a << "\n";
      usage();
      return 2;
    }
  }

  if (m <= 0 || n <= 0 || batch <= 0 || num_steps <= 0 || repeats <= 0) {
    std::cerr << "m n batch steps must be positive\n";
    return 2;
  }
  if (fwd_tile_m <= 0 || fwd_tile_n <= 0 || bwd_tile_m <= 0 || bwd_tile_n <= 0) {
    std::cerr << "tile sizes must be positive\n";
    return 2;
  }

  const int fwd_tiles_m = (batch + fwd_tile_m - 1) / fwd_tile_m;
  const int fwd_tiles_n = (m + fwd_tile_n - 1) / fwd_tile_n;
  const int bwd_tiles_m = (m + bwd_tile_m - 1) / bwd_tile_m;
  const int bwd_tiles_n = (n + bwd_tile_n - 1) / bwd_tile_n;

  const int tasks_per_step =
      1 + (fwd_tiles_m * fwd_tiles_n) + 1 + 1 + (bwd_tiles_m * bwd_tiles_n) + 1;

  // Dependency buffer ids (tokens only, not tied to physical buffers)
  constexpr uint16_t kBufY = 1;           // Forward output ready
  constexpr uint16_t kBufGradReady = 2;   // Loss + zero-dW prerequisites
  constexpr uint16_t kBufDW = 3;          // dW ready for SGD
  constexpr uint16_t kBufW = 4;           // Weight updates ready

  std::cout << "Megakernel Training Demo\n";
  std::cout << "========================\n";
  std::cout << "Model: y = Wx, W:[" << m << "," << n << "]\n";
  std::cout << "Batch: " << batch << "\n";
  std::cout << "Steps: " << num_steps << ", LR: " << lr << "\n";
  std::cout << "Params: " << (static_cast<long long>(m) * static_cast<long long>(n)) << "\n";
  std::cout << "Tiles: fwd_m " << fwd_tile_m << " fwd_n " << fwd_tile_n << " bwd_m " << bwd_tile_m
            << " bwd_n " << bwd_tile_n << "\n";
  std::cout << "Tasks per step: " << tasks_per_step << " (including "
            << (fwd_tiles_m * fwd_tiles_n) << " forward tiles and "
            << (bwd_tiles_m * bwd_tiles_n) << " backward tiles)\n";
  std::cout << "Forward mode: LinearForward (task-decomposed)\n";
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

  CPUTrainResult cpu_res{.W = td.h_W, .final_loss = 0.0f, .loss_curve = {}};
  const long long cpu_work = static_cast<long long>(batch) * static_cast<long long>(m) *
                             static_cast<long long>(n);
  if (!skip_cpu && cpu_work <= 100000000LL) {
    cpu_res = cpu_train_linear(td.h_W, td.h_x, td.h_target, batch, m, n, num_steps, lr);
    std::cout << "CPU final loss: " << cpu_res.final_loss << "\n";
    if (!cpu_res.loss_curve.empty()) {
      std::cout << "CPU first step loss: " << cpu_res.loss_curve[0] << "\n";
    }
  } else {
    std::cout << "CPU baseline skipped\n";
  }
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
    for (int tm = 0; tm < fwd_tiles_m; ++tm) {
      int tile_m_start = tm * fwd_tile_m;
      int tile_m_count = (tile_m_start + fwd_tile_m <= batch) ? fwd_tile_m : (batch - tile_m_start);
      for (int tn = 0; tn < fwd_tiles_n; ++tn) {
        int tile_n_start = tn * fwd_tile_n;
        int tile_n_count = (tile_n_start + fwd_tile_n <= m) ? fwd_tile_n : (m - tile_n_start);

        pk::Task fwd_task{};
        pk::LinearForwardArgs fwd_args{.W = d_W,
                                       .x = d_x,
                                       .y = d_y,
                                       .batch = batch,
                                       .m = m,
                                       .n = n,
                                       .tile_m_start = tile_m_start,
                                       .tile_m_count = tile_m_count,
                                       .tile_n_start = tile_n_start,
                                       .tile_n_count = tile_n_count};
        pk::encode_args(fwd_task, pk::OpCode::LinearForward, fwd_args);
        fwd_task.header.buffer_read_id = kBufW;
        fwd_task.header.buffer_write_id = kBufY;
        fwd_task.header.wait_count = ready_w;
        fwd_task.header.read_epoch = epoch_w;
        fwd_task.header.write_epoch = epoch_y + 1;
        tasks.push_back(fwd_task);
        ready_y += 1;
        epoch_y += 1;
      }
    }

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

    // Task 4: Backward tiles (dW += outer(dy, x)) depends on grad readiness
    for (int tm = 0; tm < bwd_tiles_m; ++tm) {
      int tile_m_start = tm * bwd_tile_m;
      int tile_m_count = (tile_m_start + bwd_tile_m <= m) ? bwd_tile_m : (m - tile_m_start);
      for (int tn = 0; tn < bwd_tiles_n; ++tn) {
        int tile_n_start = tn * bwd_tile_n;
        int tile_n_count = (tile_n_start + bwd_tile_n <= n) ? bwd_tile_n : (n - tile_n_start);

        pk::Task bwd_task{};
        pk::LinearBackwardArgs bwd_args{.dy = d_dy,
                                        .x = d_x,
                                        .dW = d_dW,
                                        .batch = batch,
                                        .m = m,
                                        .n = n,
                                        .tile_m_start = tile_m_start,
                                        .tile_m_count = tile_m_count,
                                        .tile_n_start = tile_n_start,
                                        .tile_n_count = tile_n_count};
        pk::encode_args(bwd_task, pk::OpCode::LinearBackward, bwd_args);
        bwd_task.header.buffer_read_id = kBufGradReady;
        bwd_task.header.buffer_write_id = kBufDW;
        bwd_task.header.wait_count = ready_grad;
        bwd_task.header.read_epoch = epoch_grad;
        bwd_task.header.write_epoch = epoch_dw + 1;
        tasks.push_back(bwd_task);
        ready_dw += 1;
        epoch_dw += 1;
      }
    }

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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double mean_ms = 0.0;
  double m2_ms = 0.0;
  int count_ms = 0;

  float last_h_loss = 0.0f;
  float last_host_recomputed_loss = 0.0f;

  for (int rep = 0; rep < repeats; ++rep) {
    cudaMemcpy(d_W, td.h_W.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    runtime.submit_tasks(tasks);
    cudaEventRecord(start);
    runtime.launch(num_blocks);
    runtime.synchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    const float step_ms = elapsed_ms / static_cast<float>(num_steps);

    count_ms += 1;
    const double delta = step_ms - mean_ms;
    mean_ms += delta / static_cast<double>(count_ms);
    m2_ms += delta * (step_ms - mean_ms);

    cudaMemcpy(&last_h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

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
    last_host_recomputed_loss = host_mse(h_y_final, td.h_target);

    std::cout << "\nResults:\n";
    std::cout << "--------\n";
    std::cout << "Repeat: " << (rep + 1) << " of " << repeats << "\n";
    std::cout << "Time: " << elapsed_ms << " ms\n";
    std::cout << "Time per step: " << step_ms << " ms\n";
    std::cout << "Final loss (device accumulation): " << last_h_loss << "\n";
    std::cout << "Final loss (host recompute): " << last_host_recomputed_loss << "\n";

    const bool decreased =
        last_host_recomputed_loss < initial_loss && std::isfinite(last_host_recomputed_loss);
    if (decreased) {
      std::cout << "SUCCESS: loss decreased (" << initial_loss << " -> " << last_host_recomputed_loss
                << ")\n";
    } else {
      std::cout << "NOTE: loss did not decrease (" << initial_loss << " -> " << last_host_recomputed_loss
                << ")\n";
    }
  }

  const double stdev_ms = (count_ms > 1) ? std::sqrt(m2_ms / static_cast<double>(count_ms - 1))
                                        : 0.0;

  std::cout << "\nSummary:\n";
  std::cout << "--------\n";
  std::cout << "Repeats: " << repeats << "\n";
  std::cout << "Mean time per step: " << mean_ms << " ms\n";
  std::cout << "Stdev time per step: " << stdev_ms << " ms\n";

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
