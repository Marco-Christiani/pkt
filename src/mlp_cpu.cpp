#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "log.hpp"
#include "phasebits.hpp"

// Simple linear model: y = Wx + b
struct LinearModel {
  std::vector<float> W; // [out_dim, in_dim]
  std::vector<float> b; // [out_dim]
  std::vector<float> W_grad;
  std::vector<float> b_grad;
  int in_dim;
  int out_dim;
  float lr;

  LinearModel(size_t in_dim, size_t out_dim, float lr = 0.01F)
      : in_dim(in_dim), out_dim(out_dim), lr(lr) {
    // Random init
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0F, 0.1F);

    W.resize(out_dim * in_dim);
    b.resize(out_dim);
    W_grad.resize((out_dim * in_dim), 0.0F);
    b_grad.resize(out_dim, 0.0F);

    for (auto& w : W) {
      w = dist(rng);
    }
    for (auto& bi : b) {
      bi = dist(rng);
    }
  }

  // y = Wx + b
  void forward(const std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < out_dim; i++) {
      y[i] = b[i];
      for (int j = 0; j < in_dim; j++) {
        y[i] += W[(i * in_dim) + j] * x[j];
      }
    }
  }

  // Compute gradients: dL/dW, dL/db given dL/dy
  void backward(const std::vector<float>& x, const std::vector<float>& dy) {
    // dL/db = dy
    for (int i = 0; i < out_dim; i++) {
      b_grad[i] = dy[i];
    }

    // dL/dW = dy * x^T
    for (int i = 0; i < out_dim; i++) {
      for (int j = 0; j < in_dim; j++) {
        W_grad[(i * in_dim) + j] = dy[i] * x[j];
      }
    }
  }

  void step() {
    for (int i = 0; i < W.size(); i++) {
      W[i] -= lr * W_grad[i];
    }
    for (int i = 0; i < b.size(); i++) {
      b[i] -= lr * b_grad[i];
    }
  }
};

// Ring buffer for batches
struct Batch {
  std::vector<float> x;
  std::vector<float> y_target;

  Batch(int in_dim, int out_dim) : x(in_dim), y_target(out_dim) {}
};

constexpr int N_STAGES = 4;
constexpr int N_BATCHES = 10;
constexpr int IN_DIM = 8;
constexpr int OUT_DIM = 4;
constexpr int BATCH_SIZE = 1; // Single sample for simplicity

std::atomic<uint32_t> phasebits{0};
std::vector<Batch> ring_buffer;

// Generate synthetic data: y = 2*x + noise
void generate_batch(Batch& batch) {
  static thread_local std::mt19937 rng(std::random_device{}());
  std::normal_distribution<float> dist(0.0F, 0.1F);

  for (auto& xi : batch.x) {
    xi = dist(rng);
  }

  for (int i = 0; i < OUT_DIM; i++) {
    batch.y_target[i] = 0.0F;
    for (int j = 0; j < IN_DIM; j++) {
      batch.y_target[i] += 2.0F * batch.x[j]; // Simple target function
    }
    batch.y_target[i] += dist(rng) * 0.01F; // Small noise
  }
}

float mse_loss(const std::vector<float>& pred, const std::vector<float>& target,
               std::vector<float>& grad) {
  float loss = 0.0F;
  for (int i = 0; i < pred.size(); i++) {
    float diff = pred[i] - target[i];
    loss += diff * diff;
    grad[i] = 2.0F * diff; // dL/dy
  }
  return loss / pred.size();
}

void data_loader_thread() {
  int slot = 0;
  int expected = 0; // Data loader starts first, waits for slot to be empty (phase=0)

  for (int batch_idx = 0; batch_idx < N_BATCHES; batch_idx++) {
    Log(LogLevel::DEBUG) << "DataLoader: waiting for slot " << slot << " phase=" << expected;

    wait_for_phase(phasebits, slot, expected);

    Log(LogLevel::INFO) << "DataLoader: loading batch " << batch_idx << " into slot " << slot;

    generate_batch(ring_buffer[slot]);
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Simulate I/O

    arrive(phasebits, slot);
    expected ^= 1;
    slot = ring_advance(slot, N_STAGES);
  }

  Log(LogLevel::INFO) << "DataLoader: done";
}

void compute_thread() {
  LinearModel model(IN_DIM, OUT_DIM, 0.01F);

  int slot = 0;
  int expected = 1; // Compute waits for data to be ready (phase=1)

  std::vector<float> y_pred(OUT_DIM);
  std::vector<float> dy(OUT_DIM);

  for (int batch_idx = 0; batch_idx < N_BATCHES; batch_idx++) {
    Log(LogLevel::DEBUG) << "Compute: waiting for slot " << slot << " phase=" << expected;

    wait_for_phase(phasebits, slot, expected);

    Batch& batch = ring_buffer[slot];

    Log(LogLevel::INFO) << "Compute: processing batch " << batch_idx << " from slot " << slot;

    // Forward
    model.forward(batch.x, y_pred);

    // Loss & backward
    float loss = mse_loss(y_pred, batch.y_target, dy);
    model.backward(batch.x, dy);

    // Optimizer step
    model.step();

    Log(LogLevel::INFO) << "Compute: batch " << batch_idx << " loss=" << loss;

    std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate compute

    arrive(phasebits, slot);
    expected ^= 1;
    slot = ring_advance(slot, N_STAGES);
  }

  Log(LogLevel::INFO) << "Compute: done";
}

int main(int argc, char const* argv[]) {
  // Initialize ring buffer
  for (int i = 0; i < N_STAGES; i++) {
    ring_buffer.emplace_back(IN_DIM, OUT_DIM);
  }

  Log(LogLevel::INFO) << "Starting linear model training with phase bits";
  Log(LogLevel::INFO) << "Model: " << IN_DIM << " -> " << OUT_DIM;
  Log(LogLevel::INFO) << "Batches: " << N_BATCHES;
  Log(LogLevel::INFO) << "Pipeline stages: " << N_STAGES;

  std::thread t_data(data_loader_thread);
  std::thread t_compute(compute_thread);

  t_data.join();
  t_compute.join();

  Log(LogLevel::INFO) << "Training complete";

  return 0;
}
