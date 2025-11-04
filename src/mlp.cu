#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Matrix dimensions for testing
// constexpr int IN_DIM = 128;
// constexpr int OUT_DIM = 128;
// constexpr int BATCH_SIZE = 32;

// Small dimensions for testing on consumer GPUs
constexpr int IN_DIM = 64;
constexpr int OUT_DIM = 64;
constexpr int BATCH_SIZE = 16;

// Pipeline configuration
constexpr int N_STAGES = 2; // Double buffering

// Thread block configuration
constexpr int BLOCK_SIZE = 256; // 8 warps
constexpr int WARP_SIZE = 32;

// ============================================
// Phase Bit Synchronization Helpers
// ============================================
// Phase bits provide a lock-free synchronization mechanism between producer and consumer.
// Each slot in the ring buffer has a single bit (0 or 1) indicating its state:
//   - Phase 0: Slot is empty (ready for producer to write)
//   - Phase 1: Slot is full (ready for consumer to read)

// Spin-wait until the specified slot reaches the expected phase
__device__ inline void wait_for_phase(uint32_t* phasebits, int slot, int expected) {
  while (((atomicAdd(phasebits, 0) >> slot) & 1) != expected) {
    // Busy wait - could add __nanosleep(100) to reduce power consumption
  }
}

// Toggle the phase bit for the specified slot (0->1 or 1->0)
__device__ inline void arrive_phase(uint32_t* phasebits, int slot) {
  atomicXor(phasebits, 1u << slot);
}

// ============================================
// Matrix Multiplication Kernel
// ============================================
// Simple matrix multiply: C = A @ B^T
// A: [M, K] input activations in shared memory
// B: [N, K] weight matrix in shared memory
// C: [M, N] output in shared memory
// Each thread computes one or more elements of C
__device__ void naive_matmul_transpose(const float* A, // [M, K] in shared memory
                                       const float* B, // [N, K] in shared memory (weights)
                                       float* C,       // [M, N] in shared memory
                                       int M, int N, int K) {
  int tid = threadIdx.x;
  int total_elements = M * N;

  // Each thread handles multiple output elements via grid-stride loop
  for (int idx = tid; idx < total_elements; idx += BLOCK_SIZE) {
    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    // Compute dot product: A[row, :] · B[col, :]
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[col * K + k]; // B^T indexing
    }
    C[row * N + col] = sum;
  }
}

// ============================================
// Persistent Kernel with Producer-Consumer Pipeline
// ============================================
// This kernel implements a persistent thread block that processes multiple batches
// using a ring buffer to overlap data loading (producer) with computation (consumer).
//
// Architecture:
//   - 1 producer warp (warp 0): Loads input batches from global memory into ring buffer
//   - 7 consumer warps (warps 1-7): Compute matrix multiplication on buffered data
//
// Synchronization:
//   - Phase bits coordinate producer/consumer access to ring buffer slots
//   - Producer always waits for phase=0 (empty slot)
//   - Consumer always waits for phase=1 (full slot)
//   - __threadfence_block() ensures memory visibility between warps
//   - NO __syncthreads() in divergent paths (would cause deadlock)
__global__ void persistent_linear_kernel(const float* X_global, // [n_batches, BATCH_SIZE, IN_DIM]
                                         const float* W_global, // [OUT_DIM, IN_DIM]
                                         float* Y_global,       // [n_batches, BATCH_SIZE, OUT_DIM]
                                         uint32_t* phasebits, int n_batches) {
  // ============================================
  // Shared Memory Layout
  // ============================================
  __shared__ float x_ring[N_STAGES][BATCH_SIZE * IN_DIM]; // Ring buffer for input batches
  __shared__ float w_smem[OUT_DIM * IN_DIM];              // Weight matrix (loaded once)
  __shared__ float y_smem[BATCH_SIZE * OUT_DIM];          // Output buffer

  int warpid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;

  // ============================================
  // Initialize: Load weights into shared memory
  // ============================================
  int w_elements = OUT_DIM * IN_DIM;
  for (int i = threadIdx.x; i < w_elements; i += BLOCK_SIZE) {
    w_smem[i] = W_global[i];
  }
  __syncthreads(); // about to diverge

  if (warpid == 0) {
    // ============================================
    // PRODUCER WARP: Load data into ring buffer
    // ============================================
    int slot = 0;
    const int expected_phase = 0; // Always wait for empty slots (phase=0)

    for (int batch = 0; batch < n_batches; batch++) {
      // Wait for this slot to be empty (consumer finished processing it)
      wait_for_phase(phasebits, slot, expected_phase);

      // Copy batch from global memory to shared memory ring buffer
      int batch_elements = BATCH_SIZE * IN_DIM;
      const float* batch_src = X_global + batch * IN_DIM * BATCH_SIZE;
      float* batch_dst = x_ring[slot];

      for (int i = laneid; i < batch_elements; i += WARP_SIZE) {
        batch_dst[i] = batch_src[i];
      }
      __syncwarp(); // Ensure all lanes in producer warp finished writing

      // Make writes visible to consumer warps and signal slot is full
      __threadfence_block();
      if (laneid == 0) {
        arrive_phase(phasebits, slot); // Flip to phase=1 (full)
      }

      // advance
      slot = (slot + 1) % N_STAGES;
    }

  } else {
    // ============================================
    // CONSUMER WARPS: Compute Y = X @ W^T
    // ============================================
    int slot = 0;
    const int expected_phase = 1; // Always wait for full slots (phase=1)

    for (int batch = 0; batch < n_batches; batch++) {
      // Wait for this slot to be full (producer finished loading it)
      wait_for_phase(phasebits, slot, expected_phase);

      // Phase bit + __threadfence_block() ensures data is ready
      // NOTE: i cant sync here.. __syncthreads() would deadlock since producer warp never reaches
      // it

      // MM: y_smem = x_ring[slot] @ w_smem^T
      // All consumer warps participate
      naive_matmul_transpose(x_ring[slot], // [BATCH_SIZE, IN_DIM]
                             w_smem,       // [OUT_DIM, IN_DIM]
                             y_smem,       // [BATCH_SIZE, OUT_DIM]
                             BATCH_SIZE, OUT_DIM, IN_DIM);
      // sync before writeback
      __syncwarp();

      // single warp handles writeback to gbmem
      if (warpid == 1) {
        float* batch_out = Y_global + batch * BATCH_SIZE * OUT_DIM;
        for (int i = laneid; i < BATCH_SIZE * OUT_DIM; i += WARP_SIZE) {
          batch_out[i] = y_smem[i];
        }
        __syncwarp();

        // Make writes visible to producer warp and signal slot is empty
        __threadfence_block();
        if (laneid == 0) {
          arrive_phase(phasebits, slot); // Flip to phase=0 (empty)
        }
      }

      // advance
      slot = (slot + 1) % N_STAGES;
    }
  }
}

// ============================================
// Host Launch Function
// ============================================
void launch_persistent_linear(const float* X_dev, const float* W_dev, float* Y_dev,
                              uint32_t* phasebits_dev, int n_batches) {
  // maybe request more shared memory if using dyn alloc on my test gpu
  // cudaFuncSetAttribute(persistent_linear_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      163840);
  // cudaFuncSetAttribute(persistent_linear_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
  //                      100);

  // Initialize phase bits
  cudaMemset(phasebits_dev, 0, sizeof(uint32_t));

  // Launch single persistent thread block
  persistent_linear_kernel<<<1, BLOCK_SIZE>>>(X_dev, W_dev, Y_dev, phasebits_dev, n_batches);
  cudaDeviceSynchronize();
}

// ============================================
// Backward Pass + Optimizer Step
// ============================================
// Computes grads and updates weights in-place
// dL/dW = dL/dY^T @ X  (gradient of weights)
// W_new = W - lr * dL/dW  (SGD update)
//
// Args:
//   X: [M, K] input activations
//   dY: [M, N] gradient w.r.t. output
//   W: [N, K] weight matrix (updated in-place)
//   M, N, K: matrix dimensions
//   lr: learning rate
__device__ void backward_and_step(const float* X,  // [M, K] in shared memory
                                  const float* dY, // [M, N] in shared memory
                                  float* W,        // [N, K] in shared memory
                                  int M, int N, int K, float lr) {
  int tid = threadIdx.x;
  int total_elements = N * K;

  // each thread calc grad for one or more weight elements
  for (int idx = tid; idx < total_elements; idx += BLOCK_SIZE) {
    int n = idx / K; // Output dimension
    int k = idx % K; // Input dimension

    // dL/dW[n, k] = sum_m(dY[m, n] * X[m, k])
    float grad = 0.0f;
    for (int m = 0; m < M; m++) {
      grad += dY[m * N + n] * X[m * K + k];
    }

    // step
    W[n * K + k] -= lr * grad;
  }
}

__global__ void
persistent_linear_train_kernel(const float* X_global, // [n_batches, BATCH_SIZE, IN_DIM]
                               const float* Y_target, // [n_batches, BATCH_SIZE, OUT_DIM]
                               float* W_global,       // [OUT_DIM, IN_DIM] - mutable
                               float* loss_out,       // [n_batches] - output loss per batch
                               uint32_t* phasebits, int n_batches, float lr) {
  //  __shared__ float dy_ring[N_STAGES][BATCH_SIZE * OUT_DIM];  // Gradient ring buffer
  __shared__ float x_ring[N_STAGES][BATCH_SIZE * IN_DIM];
  __shared__ float w_smem[OUT_DIM * IN_DIM];
  __shared__ float y_smem[BATCH_SIZE * OUT_DIM];
  // __shared__ float y_target_smem[BATCH_SIZE * OUT_DIM];
  __shared__ float dy_smem[BATCH_SIZE * OUT_DIM];

  int warpid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;

  // Load initial weights
  int w_elements = OUT_DIM * IN_DIM;
  for (int i = threadIdx.x; i < w_elements; i += BLOCK_SIZE) {
    w_smem[i] = W_global[i];
  }
  __syncthreads();

  if (warpid == 0) {
    // PRODUCER: Load X and Y_target
    int slot = 0;
    const int expected_phase = 0;

    for (int batch = 0; batch < n_batches; batch++) {
      wait_for_phase(phasebits, slot, expected_phase);

      // Load input batch
      int batch_elements = BATCH_SIZE * IN_DIM;
      const float* batch_src = X_global + batch * BATCH_SIZE * IN_DIM;
      float* batch_dst = x_ring[slot];
      for (int i = laneid; i < batch_elements; i += WARP_SIZE) {
        batch_dst[i] = batch_src[i];
      }
      __syncwarp();

      __threadfence_block();
      if (laneid == 0) {
        arrive_phase(phasebits, slot);
      }

      slot = (slot + 1) % N_STAGES;
    }

  } else {
    // CONSUMER: Forward + Backward + Step
    int slot = 0;
    const int expected_phase = 1;

    for (int batch = 0; batch < n_batches; batch++) {
      wait_for_phase(phasebits, slot, expected_phase);

      // ===== Forward pass =====
      naive_matmul_transpose(x_ring[slot], w_smem, y_smem, BATCH_SIZE, OUT_DIM, IN_DIM);
      __syncwarp();

      // ===== Compute loss and gradient =====
      // All consumer warps participate
      int out_elements = BATCH_SIZE * OUT_DIM;
      float local_loss = 0.0f;

      for (int i = threadIdx.x; i < out_elements; i += BLOCK_SIZE) {
        // Load target
        float target = Y_target[batch * out_elements + i];
        float pred = y_smem[i];

        // MSE loss
        float diff = pred - target;
        local_loss += diff * diff;

        // Gradient: dL/dy = 2 * (pred - target) / batch_size
        dy_smem[i] = 2.0f * diff / (BATCH_SIZE * OUT_DIM);
      }
      __syncwarp();

      // TODO: Reduce local_loss across warps and write to loss_out[batch]

      // ===== Backward + Step =====
      backward_and_step(x_ring[slot], dy_smem, w_smem, BATCH_SIZE, OUT_DIM, IN_DIM, lr);
      __syncwarp();

      __threadfence_block();
      if (warpid == 1 && laneid == 0) {
        arrive_phase(phasebits, slot);
      }

      slot = (slot + 1) % N_STAGES;
    }
  }

  // Write weights back
  __syncthreads(); // warps reconverge after loop
  for (int i = threadIdx.x; i < w_elements; i += BLOCK_SIZE) {
    W_global[i] = w_smem[i];
  }
}

// ============================================
// Host Launch Function for Training
// ============================================
void launch_persistent_train(const float* X_dev, const float* Y_target_dev, float* W_dev,
                             float* loss_dev, uint32_t* phasebits_dev, int n_batches,
                             float learning_rate) {
  // Initialize phase bits to 0 (all slots empty)
  cudaMemset(phasebits_dev, 0, sizeof(uint32_t));

  // Launch single persistent thread block
  persistent_linear_train_kernel<<<1, BLOCK_SIZE>>>(X_dev, Y_target_dev, W_dev, loss_dev,
                                                    phasebits_dev, n_batches, learning_rate);
  cudaDeviceSynchronize();
}

// ============================================
// Test Harness
// ============================================
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

constexpr int N_BATCHES = 4;

int fwd_main() {
  using F = float;

  const int x_elems = N_BATCHES * BATCH_SIZE * IN_DIM;
  const int w_elems = OUT_DIM * IN_DIM;
  const int y_elems = N_BATCHES * BATCH_SIZE * OUT_DIM;

  // ----------------------------------------------------------
  // Host buffers
  // ----------------------------------------------------------
  std::vector<F> h_X(x_elems);
  std::vector<F> h_W(w_elems);
  std::vector<F> h_Y(y_elems, -1.0f);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& v : h_X)
    v = dist(rng);
  for (auto& w : h_W)
    w = dist(rng);

  printf("Y[0..7]: ");
  for (int i = 0; i < 8; ++i) {
    printf("%f ", h_Y[i]);
  }
  printf("\n");

  // ----------------------------------------------------------
  // Device allocation
  // ----------------------------------------------------------
  F* d_X = nullptr;
  F* d_W = nullptr;
  F* d_Y = nullptr;
  uint32_t* d_phasebits = nullptr;

  cudaMalloc(&d_X, x_elems * sizeof(F));
  cudaMalloc(&d_W, w_elems * sizeof(F));
  cudaMalloc(&d_Y, y_elems * sizeof(F));
  cudaMalloc(&d_phasebits, sizeof(uint32_t));

  cudaMemcpy(d_X, h_X.data(), x_elems * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), w_elems * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, h_Y.data(), y_elems * sizeof(F), cudaMemcpyHostToDevice);

  // ----------------------------------------------------------
  // Run persistent pipeline
  // ----------------------------------------------------------
  launch_persistent_linear(d_X, d_W, d_Y, d_phasebits, N_BATCHES);

  // ----------------------------------------------------------
  // Copy back results
  // ----------------------------------------------------------
  cudaMemcpy(h_Y.data(), d_Y, y_elems * sizeof(F), cudaMemcpyDeviceToHost);

  printf("Y[0..7]: ");
  for (int i = 0; i < 8; ++i) {
    printf("%f ", h_Y[i]);
  }
  printf("\n");

  cudaFree(d_X);
  cudaFree(d_W);
  cudaFree(d_Y);
  cudaFree(d_phasebits);
  return 0;
}

#include <cmath>
#include <iomanip>

int train_main() {
  using F = float;

  constexpr int N_EPOCHS = 100;
  constexpr int N_BATCHES_TRAIN = 8;
  constexpr float LEARNING_RATE = 0.01f;
  const int x_elems = N_BATCHES_TRAIN * BATCH_SIZE * IN_DIM;
  const int w_elems = OUT_DIM * IN_DIM;
  const int y_elems = N_BATCHES_TRAIN * BATCH_SIZE * OUT_DIM;

  // ----------------------------------------------------------
  // Generate training data
  // ----------------------------------------------------------
  std::vector<F> h_X(x_elems);
  std::vector<F> h_Y_target(y_elems);
  std::vector<F> h_W(w_elems);
  std::vector<F> h_loss(N_BATCHES_TRAIN);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::normal_distribution<float> weight_init(0.0f, 0.1f);

  // Random input data
  for (auto& v : h_X)
    v = dist(rng);

  // Initialize weights with small random values
  for (auto& w : h_W)
    w = weight_init(rng);

  // Generate targets: Y = X @ W_true^T + noise
  // random targets
  for (auto& y : h_Y_target)
    y = dist(rng);

  printf("Training configuration:\n");
  printf("  Input dim:    %d\n", IN_DIM);
  printf("  Output dim:   %d\n", OUT_DIM);
  printf("  Batch size:   %d\n", BATCH_SIZE);
  printf("  Num batches:  %d\n", N_BATCHES_TRAIN);
  printf("  Epochs:       %d\n", N_EPOCHS);
  printf("  Learning rate: %.4f\n\n", LEARNING_RATE);

  // ----------------------------------------------------------
  // Device allocation
  // ----------------------------------------------------------
  F* d_X = nullptr;
  F* d_Y_target = nullptr;
  F* d_W = nullptr;
  F* d_loss = nullptr;
  uint32_t* d_phasebits = nullptr;

  cudaMalloc(&d_X, x_elems * sizeof(F));
  cudaMalloc(&d_Y_target, y_elems * sizeof(F));
  cudaMalloc(&d_W, w_elems * sizeof(F));
  cudaMalloc(&d_loss, N_BATCHES_TRAIN * sizeof(F));
  cudaMalloc(&d_phasebits, sizeof(uint32_t));

  cudaMemcpy(d_X, h_X.data(), x_elems * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y_target, h_Y_target.data(), y_elems * sizeof(F), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W.data(), w_elems * sizeof(F), cudaMemcpyHostToDevice);

  // ----------------------------------------------------------
  // Training loop
  // ----------------------------------------------------------
  printf("Starting training...\n");
  printf("Epoch | Avg Loss\n");
  printf("------|----------\n");

  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
    launch_persistent_train(d_X, d_Y_target, d_W, d_loss, d_phasebits, N_BATCHES_TRAIN,
                            LEARNING_RATE);
    cudaMemcpy(h_loss.data(), d_loss, N_BATCHES_TRAIN * sizeof(F), cudaMemcpyDeviceToHost);

    float avg_loss = 0.0f;
    for (int i = 0; i < N_BATCHES_TRAIN; i++) {
      avg_loss += h_loss[i];
    }
    avg_loss /= N_BATCHES_TRAIN;

    if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
      printf("%5d | %.6f\n", epoch, avg_loss);
    }
  }

  // ----------------------------------------------------------
  // Copy back weights
  // ----------------------------------------------------------
  cudaMemcpy(h_W.data(), d_W, w_elems * sizeof(F), cudaMemcpyDeviceToHost);

  printf("\nTraining complete!\n");
  printf("Final weights[0..7]: ");
  for (int i = 0; i < 8; ++i) {
    printf("%.4f ", h_W[i]);
  }
  printf("\n");

  cudaFree(d_X);
  cudaFree(d_Y_target);
  cudaFree(d_W);
  cudaFree(d_loss);
  cudaFree(d_phasebits);
  return 0;
}

int main() {
  return train_main();
}
