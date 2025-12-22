#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "../include/ops/linear_bwd.cuh"
#include "../include/ops/linear_fwd.cuh"
#include "../include/ops/mse.cuh"
#include "../include/ops/sgd.cuh"
#include "../include/ops/zero.cuh"
#include "../include/profile_ranges.cuh"
#include "../include/runtime.hpp"
#include "../include/segment.cuh"
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
  int blocks_override = -1;
  bool print_profile = false;
  bool check_tasks = false;
  bool check_loss = false;
  bool perf = false;
  std::string perf_json_path;
#if PK_SEGMENT_GATE
  int segment_window_override = -1;
#endif

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
    std::cout << "  --blocks <int> (override grid blocks)\n";
    std::cout << "  --skip-cpu\n";
    std::cout << "  --print-profile (requires PK_PROFILE_RANGES=1 build)\n";
    std::cout << "  --check-tasks (requires PK_TASK_ACCOUNTING=1 build)\n";
    std::cout << "  --check-loss (assert device loss ~= host loss; expensive)\n";
    std::cout << "  --perf (requires PK_PERF_COUNTERS=1 build)\n";
    std::cout << "  --perf-json <path> (requires PK_PERF_COUNTERS=1 build)\n";
#if PK_SEGMENT_GATE
    std::cout << "  --seg-window <int> (override PK_SEGMENT_WINDOW_SIZE for segment gating)\n";
#endif
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
    } else if (!std::strcmp(a, "--print-profile")) {
      print_profile = true;
    } else if (!std::strcmp(a, "--check-tasks")) {
      check_tasks = true;
    } else if (!std::strcmp(a, "--check-loss")) {
      check_loss = true;
    } else if (!std::strcmp(a, "--perf")) {
      perf = true;
    } else if (!std::strcmp(a, "--perf-json")) {
      perf = true;
      perf_json_path = need(a);
    } else if (!std::strcmp(a, "--blocks")) {
      if (!parse_int(need(a), &blocks_override)) {
        std::cerr << "Bad value for --blocks\n";
        return 2;
      }
#if PK_SEGMENT_GATE
    } else if (!std::strcmp(a, "--seg-window")) {
      if (!parse_int(need(a), &segment_window_override)) {
        std::cerr << "Bad value for --seg-window\n";
        return 2;
      }
#endif
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
  if (blocks_override == 0 || blocks_override < -1) {
    std::cerr << "--blocks must be positive or omitted\n";
    return 2;
  }

  const int fwd_tiles_m = (batch + fwd_tile_m - 1) / fwd_tile_m;
  const int fwd_tiles_n = (m + fwd_tile_n - 1) / fwd_tile_n;
  const int bwd_tiles_m = (m + bwd_tile_m - 1) / bwd_tile_m;
  const int bwd_tiles_n = (n + bwd_tile_n - 1) / bwd_tile_n;

  const int tasks_per_step =
      1 + (fwd_tiles_m * fwd_tiles_n) + 1 + 1 + (bwd_tiles_m * bwd_tiles_n) + 1;
  const int eval_tasks = check_loss ? (1 + (fwd_tiles_m * fwd_tiles_n) + 1) : 0;

  // Dependency buffer ids (tokens only, not tied to physical buffers)
  // Rule: no implicit queue-order dependencies; all sequencing is expressed here.
  constexpr uint16_t kBufLossStart = 1;   // loss_reset + forward_done barrier for loss
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
  if (check_loss) {
    std::cout << "Extra tasks: " << eval_tasks << " (final eval loss)\n";
  }
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
  runtime.initialize(num_steps * tasks_per_step + eval_tasks);
  const int num_sms = pk::Runtime::get_num_sms();
  const int num_blocks = (blocks_override > 0) ? blocks_override : num_sms;
#if PK_TASK_ACCOUNTING
  runtime.set_task_accounting_enabled(check_tasks);
#endif
#if PK_SEGMENT_GATE
  if (segment_window_override > 0) {
    runtime.set_segment_window_size(segment_window_override);
  }
#endif
#if PK_PERF_COUNTERS
  runtime.set_perf_enabled(perf);
#else
  if (perf) {
    std::cerr << "Perf counters unavailable (build with PK_PERF_COUNTERS=1).\n";
    return 3;
  }
#endif
  // === SINGLE KERNEL LAUNCH ===
  std::cout << "Launching megakernel with 1 launch (" << tasks_per_step << " tasks per step, "
            << num_blocks << " blocks)...\n";

  std::vector<pk::Task> tasks;
  tasks.reserve(num_steps * tasks_per_step + eval_tasks);
  std::vector<pk::SegmentDesc> segments;
  segments.reserve(static_cast<size_t>(num_steps) * 6 + (check_loss ? 3 : 0));
  int segment_id = 0;

  auto finalize_segment = [&](std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
      tasks[i].header.user_tag = static_cast<std::uint32_t>(segment_id);
    }
    segments.push_back(pk::SegmentDesc{static_cast<int>(begin), static_cast<int>(end)});
    segment_id += 1;
  };

  // Track cumulative ready counts and epochs per buffer
  int ready_loss_start = 0;
  int ready_grad = 0;
  int ready_dw = 0;
  int ready_w = 0;
  int epoch_grad = 0;
  int epoch_dw = 0;
  int epoch_w = 0;

  for (int step = 0; step < num_steps; ++step) {
    const int step_epoch = step + 1;
    // Segment 0: loss reset
    const std::uint32_t loss_reset_begin = static_cast<std::uint32_t>(tasks.size());
    pk::Task zero_loss_task{};
    pk::ZeroMemoryArgs zero_loss{.ptr = d_loss, .size = 1};
    pk::encode_args(zero_loss_task, pk::OpCode::ZeroMemory, zero_loss);
    zero_loss_task.header.buffer_read_id = 0;
    zero_loss_task.header.buffer_write_id = kBufLossStart;
    zero_loss_task.header.wait_count = 0;
    zero_loss_task.header.write_epoch = step_epoch;
    tasks.push_back(zero_loss_task);
    ready_loss_start += 1;
    finalize_segment(loss_reset_begin, loss_reset_begin + 1);

    // Segment 1: forward tiles
    const std::uint32_t fwd_begin = static_cast<std::uint32_t>(tasks.size());
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
        fwd_task.header.buffer_write_id = kBufLossStart;
        fwd_task.header.wait_count = ready_w;
        fwd_task.header.read_epoch = epoch_w;
        fwd_task.header.write_epoch = step_epoch;
        tasks.push_back(fwd_task);
        ready_loss_start += 1;
      }
    }
    finalize_segment(fwd_begin, static_cast<std::uint32_t>(tasks.size()));

    // Segment 2: loss + dy (waits for all forward tiles + loss reset via kBufLossStart)
    const std::uint32_t loss_begin = static_cast<std::uint32_t>(tasks.size());
    pk::Task loss_task{};
    pk::MSELossArgs loss_args{
        .y = d_y, .target = d_target, .dy = d_dy, .loss = d_loss, .batch = batch, .m = m};
    pk::encode_args(loss_task, pk::OpCode::MSELoss, loss_args);
    loss_task.header.buffer_read_id = kBufLossStart;
    loss_task.header.buffer_write_id = kBufGradReady; // part 1 of grad readiness
    loss_task.header.wait_count = ready_loss_start;
    loss_task.header.read_epoch = step_epoch;
    loss_task.header.write_epoch = epoch_grad + 1;
    tasks.push_back(loss_task);
    ready_grad += 1;
    epoch_grad += 1;
    finalize_segment(loss_begin, loss_begin + 1);

    // Segment 3: zero dW (prereq for backward accumulation)
    const std::uint32_t zero_dw_begin = static_cast<std::uint32_t>(tasks.size());
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
    finalize_segment(zero_dw_begin, zero_dw_begin + 1);

    // Segment 4: backward tiles (dW += outer(dy, x)) depends on grad readiness
    const std::uint32_t bwd_begin = static_cast<std::uint32_t>(tasks.size());
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
    finalize_segment(bwd_begin, static_cast<std::uint32_t>(tasks.size()));

    // Segment 5: SGD update (W -= lr * dW)
    const std::uint32_t sgd_begin = static_cast<std::uint32_t>(tasks.size());
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
    finalize_segment(sgd_begin, sgd_begin + 1);
  }

  if (check_loss) {
    const int eval_epoch = num_steps + 1;

    const std::uint32_t eval_loss_reset_begin = static_cast<std::uint32_t>(tasks.size());
    pk::Task eval_zero_loss_task{};
    pk::ZeroMemoryArgs eval_zero_loss{.ptr = d_loss, .size = 1};
    pk::encode_args(eval_zero_loss_task, pk::OpCode::ZeroMemory, eval_zero_loss);
    eval_zero_loss_task.header.buffer_read_id = kBufW;
    eval_zero_loss_task.header.wait_count = ready_w;
    eval_zero_loss_task.header.read_epoch = epoch_w;
    eval_zero_loss_task.header.buffer_write_id = kBufLossStart;
    eval_zero_loss_task.header.write_epoch = eval_epoch;
    tasks.push_back(eval_zero_loss_task);
    ready_loss_start += 1;
    finalize_segment(eval_loss_reset_begin, eval_loss_reset_begin + 1);

    const std::uint32_t eval_fwd_begin = static_cast<std::uint32_t>(tasks.size());
    for (int tm = 0; tm < fwd_tiles_m; ++tm) {
      int tile_m_start = tm * fwd_tile_m;
      int tile_m_count =
          (tile_m_start + fwd_tile_m <= batch) ? fwd_tile_m : (batch - tile_m_start);
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
        fwd_task.header.buffer_write_id = kBufLossStart;
        fwd_task.header.wait_count = ready_w;
        fwd_task.header.read_epoch = epoch_w;
        fwd_task.header.write_epoch = eval_epoch;
        tasks.push_back(fwd_task);
        ready_loss_start += 1;
      }
    }
    finalize_segment(eval_fwd_begin, static_cast<std::uint32_t>(tasks.size()));

    const std::uint32_t eval_loss_begin = static_cast<std::uint32_t>(tasks.size());
    pk::Task eval_loss_task{};
    pk::MSELossArgs loss_args{
        .y = d_y, .target = d_target, .dy = d_dy, .loss = d_loss, .batch = batch, .m = m};
    pk::encode_args(eval_loss_task, pk::OpCode::MSELoss, loss_args);
    eval_loss_task.header.buffer_read_id = kBufLossStart;
    eval_loss_task.header.buffer_write_id = 0;
    eval_loss_task.header.wait_count = ready_loss_start;
    eval_loss_task.header.read_epoch = eval_epoch;
    tasks.push_back(eval_loss_task);
    finalize_segment(eval_loss_begin, eval_loss_begin + 1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);

  double mean_ms = 0.0;
  double m2_ms = 0.0;
  int count_ms = 0;

  float last_h_loss = 0.0f;
  float last_host_recomputed_loss = 0.0f;

#if PK_PERF_COUNTERS
  auto sum_u64 = [&](unsigned long long& dst, unsigned long long v) { dst += v; };
#endif

  for (int rep = 0; rep < repeats; ++rep) {
    cudaMemcpy(d_W, td.h_W.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    runtime.set_segments(segments);
    runtime.submit_tasks(tasks);
    cudaEventRecord(start);

    // NVTX capture range for Nsight Systems (host-side only).
    nvtxRangePushA("pk.launch");
    runtime.launch(num_blocks);
    runtime.synchronize();
    nvtxRangePop();

    if (check_tasks) {
      unsigned long long issued = 0;
      if (runtime.fetch_task_counter(&issued)) {
        if (issued != static_cast<unsigned long long>(tasks.size())) {
          std::cerr << "PK_TASK_ACCOUNTING mismatch: issued=" << issued
                    << " expected=" << tasks.size() << "\n";
          return 3;
        }
      } else {
        std::cerr << "Task accounting unavailable (build with PK_TASK_ACCOUNTING=1).\n";
        return 3;
      }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    const float step_ms = elapsed_ms / static_cast<float>(num_steps);

#if PK_PERF_COUNTERS
    if (perf) {
      std::vector<pk::PerfBlock> blocks;
      if (!runtime.fetch_perf_blocks(blocks) || blocks.empty()) {
        std::cerr << "Failed to fetch perf counters.\n";
        return 3;
      }

      pk::PerfBlock totals{};
      unsigned long long tasks_completed_min = ~0ull;
      unsigned long long tasks_completed_max = 0;
      unsigned long long pending_max_depth_max = 0;
      unsigned long long pending_full_sum = 0;
      unsigned long long ctrl_scan_len_max_max = 0;

      for (const pk::PerfBlock& b : blocks) {
        sum_u64(totals.ctrl_dequeue_cycles, b.ctrl_dequeue_cycles);
        sum_u64(totals.ctrl_scan_cycles, b.ctrl_scan_cycles);
        sum_u64(totals.ctrl_claim_cycles, b.ctrl_claim_cycles);
        sum_u64(totals.ctrl_wait_cycles, b.ctrl_wait_cycles);
        sum_u64(totals.ctrl_issue_cycles, b.ctrl_issue_cycles);

        sum_u64(totals.ctrl_scans_total, b.ctrl_scans_total);
        sum_u64(totals.ctrl_scan_len_sum, b.ctrl_scan_len_sum);
        if (b.ctrl_scan_len_max > ctrl_scan_len_max_max) {
          ctrl_scan_len_max_max = b.ctrl_scan_len_max;
        }
        sum_u64(totals.ctrl_found_ready_count, b.ctrl_found_ready_count);
        sum_u64(totals.ctrl_deferred_count, b.ctrl_deferred_count);
        sum_u64(totals.ctrl_no_ready_rounds, b.ctrl_no_ready_rounds);

        sum_u64(totals.ctrl_claim_attempts, b.ctrl_claim_attempts);
        sum_u64(totals.ctrl_claim_success, b.ctrl_claim_success);
        sum_u64(totals.ctrl_claim_fail, b.ctrl_claim_fail);

        sum_u64(totals.ctrl_window_occupancy_sum, b.ctrl_window_occupancy_sum);
        sum_u64(totals.ctrl_window_occupancy_samples, b.ctrl_window_occupancy_samples);

        sum_u64(totals.pending_push_count, b.pending_push_count);
        sum_u64(totals.pending_pop_count, b.pending_pop_count);
        if (b.pending_max_depth > pending_max_depth_max) {
          pending_max_depth_max = b.pending_max_depth;
        }
        pending_full_sum += b.pending_full_count;

        sum_u64(totals.load_wait_slot_cycles, b.load_wait_slot_cycles);
        sum_u64(totals.load_work_cycles, b.load_work_cycles);
        sum_u64(totals.load_page_reuse_hits, b.load_page_reuse_hits);
        sum_u64(totals.load_page_refill_count, b.load_page_refill_count);

        sum_u64(totals.compute_wait_loaded_cycles, b.compute_wait_loaded_cycles);
        sum_u64(totals.compute_work_cycles, b.compute_work_cycles);

        sum_u64(totals.store_wait_computed_cycles, b.store_wait_computed_cycles);
        sum_u64(totals.store_work_cycles, b.store_work_cycles);

        sum_u64(totals.tasks_issued, b.tasks_issued);
        sum_u64(totals.tasks_completed, b.tasks_completed);

        if (b.tasks_completed < tasks_completed_min) {
          tasks_completed_min = b.tasks_completed;
        }
        if (b.tasks_completed > tasks_completed_max) {
          tasks_completed_max = b.tasks_completed;
        }
      }

      const unsigned long long total_ctrl_cycles =
          totals.ctrl_dequeue_cycles + totals.ctrl_scan_cycles + totals.ctrl_claim_cycles +
          totals.ctrl_wait_cycles + totals.ctrl_issue_cycles;
      const double ctrl_scan_frac =
          total_ctrl_cycles ? double(totals.ctrl_scan_cycles) / double(total_ctrl_cycles) : 0.0;
      const double ctrl_wait_frac =
          total_ctrl_cycles ? double(totals.ctrl_wait_cycles) / double(total_ctrl_cycles) : 0.0;
      const double seg_window_occ_avg =
          totals.ctrl_window_occupancy_samples
              ? double(totals.ctrl_window_occupancy_sum) /
                    double(totals.ctrl_window_occupancy_samples)
              : 0.0;
      const unsigned long long total_compute_cycles =
          totals.compute_wait_loaded_cycles + totals.compute_work_cycles;
      const double compute_work_frac =
          total_compute_cycles ? double(totals.compute_work_cycles) / double(total_compute_cycles)
                               : 0.0;

      std::cout << "PERF blocks=" << num_blocks << " step_ms=" << step_ms
                << " ctrl(deq=" << totals.ctrl_dequeue_cycles << " scan=" << totals.ctrl_scan_cycles
                << " claim=" << totals.ctrl_claim_cycles << " wait=" << totals.ctrl_wait_cycles
                << " issue=" << totals.ctrl_issue_cycles << ") ctrl_scan%=" << (ctrl_scan_frac * 100.0)
                << " ctrl_wait%=" << (ctrl_wait_frac * 100.0) << " load(wait="
                << totals.load_wait_slot_cycles << " work=" << totals.load_work_cycles << ") compute(wait="
                << totals.compute_wait_loaded_cycles << " work=" << totals.compute_work_cycles
                << " work%=" << (compute_work_frac * 100.0) << ") store(wait="
                << totals.store_wait_computed_cycles << " work=" << totals.store_work_cycles
                << ") tasks(issued=" << totals.tasks_issued << " completed=" << totals.tasks_completed
                << " min=" << tasks_completed_min << " max=" << tasks_completed_max << ") pending(max_depth="
                << pending_max_depth_max << " full_sum=" << pending_full_sum << ") scan(len_max="
                << ctrl_scan_len_max_max << " no_ready_rounds=" << totals.ctrl_no_ready_rounds
                << ") seg_window(occ_avg=" << seg_window_occ_avg << ")\n";

#if PK_SEGMENT_GATE
      std::vector<int> seg_completed;
      int seg_lo = -1;
      int seg_hi = -1;
      int seg_win = -1;
      if (runtime.fetch_segment_state(seg_completed, &seg_lo, &seg_hi, &seg_win)) {
        long long sum = 0;
        int max_c = 0;
        for (int c : seg_completed) {
          sum += c;
          if (c > max_c) {
            max_c = c;
          }
        }
        std::cout << "SEG lo=" << seg_lo << " hi=" << seg_hi << " win=" << seg_win
                  << " occ=" << ((seg_lo >= 0 && seg_hi >= 0) ? (seg_hi - seg_lo) : 0)
                  << " num=" << seg_completed.size() << " completed_sum=" << sum
                  << " completed_max=" << max_c << "\n";
      }
#endif

      if (!perf_json_path.empty() && rep == 0) {
        std::ofstream out(perf_json_path);
        if (!out) {
          std::cerr << "Failed to open --perf-json path: " << perf_json_path << "\n";
          return 3;
        }
        out << "{\n";
        out << "  \"blocks\": " << num_blocks << ",\n";
        out << "  \"step_ms\": " << step_ms << ",\n";
        out << "  \"tasks_completed_min\": " << tasks_completed_min << ",\n";
        out << "  \"tasks_completed_max\": " << tasks_completed_max << ",\n";
        out << "  \"pending_max_depth_max\": " << pending_max_depth_max << ",\n";
        out << "  \"pending_full_sum\": " << pending_full_sum << ",\n";
        out << "  \"ctrl_scan_len_max_max\": " << ctrl_scan_len_max_max << ",\n";
        out << "  \"seg_window_occ_avg\": " << seg_window_occ_avg << ",\n";
        out << "  \"ratios\": {\n";
        out << "    \"ctrl_scan_frac\": " << ctrl_scan_frac << ",\n";
        out << "    \"ctrl_wait_frac\": " << ctrl_wait_frac << ",\n";
        out << "    \"compute_work_frac\": " << compute_work_frac << "\n";
        out << "  },\n";
        out << "  \"totals\": {\n";
        out << "    \"ctrl_dequeue_cycles\": " << totals.ctrl_dequeue_cycles << ",\n";
        out << "    \"ctrl_scan_cycles\": " << totals.ctrl_scan_cycles << ",\n";
        out << "    \"ctrl_claim_cycles\": " << totals.ctrl_claim_cycles << ",\n";
        out << "    \"ctrl_wait_cycles\": " << totals.ctrl_wait_cycles << ",\n";
        out << "    \"ctrl_issue_cycles\": " << totals.ctrl_issue_cycles << ",\n";
        out << "    \"ctrl_scans_total\": " << totals.ctrl_scans_total << ",\n";
        out << "    \"ctrl_scan_len_sum\": " << totals.ctrl_scan_len_sum << ",\n";
        out << "    \"ctrl_found_ready_count\": " << totals.ctrl_found_ready_count << ",\n";
        out << "    \"ctrl_deferred_count\": " << totals.ctrl_deferred_count << ",\n";
        out << "    \"ctrl_no_ready_rounds\": " << totals.ctrl_no_ready_rounds << ",\n";
        out << "    \"ctrl_claim_attempts\": " << totals.ctrl_claim_attempts << ",\n";
        out << "    \"ctrl_claim_success\": " << totals.ctrl_claim_success << ",\n";
        out << "    \"ctrl_claim_fail\": " << totals.ctrl_claim_fail << ",\n";
        out << "    \"ctrl_window_occupancy_sum\": " << totals.ctrl_window_occupancy_sum << ",\n";
        out << "    \"ctrl_window_occupancy_samples\": " << totals.ctrl_window_occupancy_samples << ",\n";
        out << "    \"pending_push_count\": " << totals.pending_push_count << ",\n";
        out << "    \"pending_pop_count\": " << totals.pending_pop_count << ",\n";
        out << "    \"load_wait_slot_cycles\": " << totals.load_wait_slot_cycles << ",\n";
        out << "    \"load_work_cycles\": " << totals.load_work_cycles << ",\n";
        out << "    \"load_page_reuse_hits\": " << totals.load_page_reuse_hits << ",\n";
        out << "    \"load_page_refill_count\": " << totals.load_page_refill_count << ",\n";
        out << "    \"compute_wait_loaded_cycles\": " << totals.compute_wait_loaded_cycles << ",\n";
        out << "    \"compute_work_cycles\": " << totals.compute_work_cycles << ",\n";
        out << "    \"store_wait_computed_cycles\": " << totals.store_wait_computed_cycles << ",\n";
        out << "    \"store_work_cycles\": " << totals.store_work_cycles << ",\n";
        out << "    \"tasks_issued\": " << totals.tasks_issued << ",\n";
        out << "    \"tasks_completed\": " << totals.tasks_completed << "\n";
        out << "  }\n";
        out << "}\n";
      }
    }
#endif

    if (print_profile) {
      pk::ProfileCounters counters{};
      if (runtime.fetch_profile(&counters)) {
        struct Row {
          const char* name;
          pk::ProfileRange range;
        };
        const Row rows[] = {
            {"controller.deps_wait", pk::ProfileRange::ControllerDepsWait},
            {"controller.queue_pop", pk::ProfileRange::ControllerQueuePop},
            {"loader.slot_wait", pk::ProfileRange::LoaderSlotWait},
            {"loader.load", pk::ProfileRange::LoaderLoad},
            {"compute.slot_wait", pk::ProfileRange::ComputeSlotWait},
            {"compute.math", pk::ProfileRange::ComputeMath},
            {"storer.slot_wait", pk::ProfileRange::StorerSlotWait},
            {"storer.threadfence", pk::ProfileRange::StorerThreadfence},
            {"storer.deps_mark_ready", pk::ProfileRange::StorerDepsMarkReady},
        };

        std::cout << "\nProfile ranges (device clock64; averaged per block; per step):\n";
        std::cout << "  GPU: " << prop.name << "  clockRate(kHz): " << prop.clockRate << "\n";
        std::cout << "  blocks: " << num_blocks << "  steps: " << num_steps
                  << "  step_ms: " << step_ms << "\n";
        for (const Row& row : rows) {
          const unsigned long long cyc = counters.cycles[static_cast<int>(row.range)];
          const double ms_per_step = static_cast<double>(cyc) /
                                     (static_cast<double>(prop.clockRate) *
                                      static_cast<double>(num_blocks) *
                                      static_cast<double>(num_steps));
          const double pct = step_ms > 0.0 ? (ms_per_step / static_cast<double>(step_ms) * 100.0)
                                           : 0.0;
          std::cout << "  " << row.name << ": " << ms_per_step << " ms (" << pct << "%)\n";
        }
      } else {
        std::cout << "\nProfile ranges unavailable (build with PK_PROFILE_RANGES=1).\n";
      }
    }

    count_ms += 1;
    const double delta = step_ms - mean_ms;
    mean_ms += delta / static_cast<double>(count_ms);
    m2_ms += delta * (step_ms - mean_ms);

    cudaMemcpy(&last_h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_W_out(m * n);
    cudaMemcpy(h_W_out.data(), d_W, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    if (check_loss) {
      const long long host_work = static_cast<long long>(batch) * static_cast<long long>(m) *
                                  static_cast<long long>(n);
      if (host_work > 200000000LL) {
        std::cerr << "WARNING: --check-loss host recompute is expensive (work=" << host_work << ")\n";
      }

      std::vector<float> h_y_final(batch * m);
      for (int b = 0; b < batch; ++b) {
        const float* xb = &td.h_x[b * n];
        float* yb = &h_y_final[b * m];
        for (int i = 0; i < m; ++i) {
          float acc = 0.f;
          const float* Wi = &h_W_out[i * n];
          for (int j = 0; j < n; ++j) {
            acc += Wi[j] * xb[j];
          }
          yb[i] = acc;
        }
      }
      last_host_recomputed_loss = host_mse(h_y_final, td.h_target);
    } else {
      std::vector<float> h_y_final(m);
      for (int i = 0; i < m; ++i) {
        float acc = 0.f;
        const float* Wi = &h_W_out[i * n];
        for (int j = 0; j < n; ++j)
          acc += Wi[j] * td.h_x[j];
        h_y_final[i] = acc;
      }
      last_host_recomputed_loss = host_mse(h_y_final, td.h_target);
    }

    std::cout << "\nResults:\n";
    std::cout << "--------\n";
    std::cout << "Repeat: " << (rep + 1) << " of " << repeats << "\n";
    std::cout << "Time: " << elapsed_ms << " ms\n";
    std::cout << "Time per step: " << step_ms << " ms\n";
    std::cout << "Final loss (device accumulation): " << last_h_loss << "\n";
    std::cout << "Final loss (host recompute): " << last_host_recomputed_loss << "\n";

    if (check_loss) {
      const float abs_err = std::fabs(last_h_loss - last_host_recomputed_loss);
      const float denom = (std::fabs(last_host_recomputed_loss) > 1e-12f)
                              ? std::fabs(last_host_recomputed_loss)
                              : 1.0f;
      const float rel_err = abs_err / denom;
      const float abs_tol = 5e-5f;
      const float rel_tol = 5e-2f;
      if (!(abs_err <= abs_tol || rel_err <= rel_tol)) {
        std::cerr << "CHECK_LOSS failed: device=" << last_h_loss
                  << " host=" << last_host_recomputed_loss << " abs_err=" << abs_err
                  << " rel_err=" << rel_err << "\n";
        return 4;
      }
    }

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
