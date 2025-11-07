#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

static __host__ __device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

static __host__ __device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

static __device__ __forceinline__ int warp_id() {
  return threadIdx.x / 32;
}

static __device__ __forceinline__ int lane_id() {
  return threadIdx.x % 32;
}

// ============================================================================
// Hand-coded 2-layer MLP training megakernel
// Architecture: Input[128] -> Hidden[256] (ReLU) -> Output[64]
// Phases: Load, Forward L0, ReLU, Forward L1, Backward L1, Backward L0, Accumulate Grads, Update
// ============================================================================

constexpr int TILE_M = 32;      // Batch tile size
constexpr int INPUT_DIM = 128;
constexpr int HIDDEN_DIM = 256;
constexpr int OUTPUT_DIM = 64;
constexpr int NUM_WARPS = 8;

extern "C" __global__ void
mlp_train_kernel(
    const __nv_bfloat16* __restrict__ X,        // [M, 128] input
    __nv_bfloat16* __restrict__ W0,             // [256, 128] layer 0 weights
    __nv_bfloat16* __restrict__ W1,             // [64, 256] layer 1 weights
    const __nv_bfloat16* __restrict__ Y_target, // [M, 64] target
    __nv_bfloat16* __restrict__ Y_out,          // [M, 64] output (optional)
    int M, float lr) {

  extern __shared__ char smem[];

  // Shared memory layout
  __nv_bfloat16* s_X = (__nv_bfloat16*)smem;                              // [32, 128]
  __nv_bfloat16* s_W0 = s_X + TILE_M * INPUT_DIM;                         // [256, 128]
  __nv_bfloat16* s_W1 = s_W0 + HIDDEN_DIM * INPUT_DIM;                    // [64, 256]
  float* s_H0 = (float*)(s_W1 + OUTPUT_DIM * HIDDEN_DIM);                 // [32, 256]
  float* s_Y = s_H0 + TILE_M * HIDDEN_DIM;                                // [32, 64]
  float* s_dY = s_Y + TILE_M * OUTPUT_DIM;                                // [32, 64]
  float* s_dH0 = s_dY + TILE_M * OUTPUT_DIM;                              // [32, 256]
  float* s_dW0 = s_dH0 + TILE_M * HIDDEN_DIM;                             // [256, 128] gradient accumulator
  float* s_dW1 = s_dW0 + HIDDEN_DIM * INPUT_DIM;                          // [64, 256] gradient accumulator

  const int wid = warp_id();
  const int lane = lane_id();

  // Load weights once (all warps collaborate)
  for (int idx = threadIdx.x; idx < HIDDEN_DIM * INPUT_DIM; idx += blockDim.x) {
    s_W0[idx] = W0[idx];
  }
  for (int idx = threadIdx.x; idx < OUTPUT_DIM * HIDDEN_DIM; idx += blockDim.x) {
    s_W1[idx] = W1[idx];
  }

  // Zero gradient accumulators
  for (int idx = threadIdx.x; idx < HIDDEN_DIM * INPUT_DIM; idx += blockDim.x) {
    s_dW0[idx] = 0.0f;
  }
  for (int idx = threadIdx.x; idx < OUTPUT_DIM * HIDDEN_DIM; idx += blockDim.x) {
    s_dW1[idx] = 0.0f;
  }
  __syncthreads();

  // Process each batch tile
  const int num_tiles = (M + TILE_M - 1) / TILE_M;

  for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    const int m_start = tile_idx * TILE_M;

    // ========== PHASE: Load X tile ==========
    if (wid < 2) {
      for (int idx = wid * 32 + lane; idx < TILE_M * INPUT_DIM; idx += 64) {
        const int local_m = idx / INPUT_DIM;
        const int local_i = idx % INPUT_DIM;
        const int global_m = m_start + local_m;
        s_X[idx] = (global_m < M) ? X[global_m * INPUT_DIM + local_i] : f32_to_bf16(0.0f);
      }
    }
    __syncthreads();

    // ========== PHASE: Forward Layer 0 (X @ W0^T + ReLU) ==========
    if (wid >= 2 && wid < 5) {
      const int compute_tid = (wid - 2) * 32 + lane;
      const int num_threads = 3 * 32;

      for (int idx = compute_tid; idx < TILE_M * HIDDEN_DIM; idx += num_threads) {
        const int m = idx / HIDDEN_DIM;
        const int h = idx % HIDDEN_DIM;

        float acc = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < INPUT_DIM; i++) {
          acc += bf16_to_f32(s_X[m * INPUT_DIM + i]) * bf16_to_f32(s_W0[h * INPUT_DIM + i]);
        }
        s_H0[idx] = fmaxf(0.0f, acc);  // ReLU
      }
    }
    __syncthreads();

    // ========== PHASE: Forward Layer 1 (H0 @ W1^T) ==========
    if (wid >= 5 && wid < 7) {
      const int compute_tid = (wid - 5) * 32 + lane;
      const int num_threads = 2 * 32;

      for (int idx = compute_tid; idx < TILE_M * OUTPUT_DIM; idx += num_threads) {
        const int m = idx / OUTPUT_DIM;
        const int o = idx % OUTPUT_DIM;

        float acc = 0.0f;
        #pragma unroll 8
        for (int h = 0; h < HIDDEN_DIM; h++) {
          acc += s_H0[m * HIDDEN_DIM + h] * bf16_to_f32(s_W1[o * HIDDEN_DIM + h]);
        }
        s_Y[idx] = acc;
      }
    }
    __syncthreads();

    // ========== PHASE: Compute dY (MSE loss gradient) ==========
    if (wid == 7) {
      for (int idx = lane; idx < TILE_M * OUTPUT_DIM; idx += 32) {
        const int m = idx / OUTPUT_DIM;
        const int o = idx % OUTPUT_DIM;
        const int global_m = m_start + m;

        float y_pred = s_Y[idx];
        float y_tgt = (global_m < M) ? bf16_to_f32(Y_target[global_m * OUTPUT_DIM + o]) : 0.0f;

        // Write output if needed
        if (Y_out && global_m < M) {
          Y_out[global_m * OUTPUT_DIM + o] = f32_to_bf16(y_pred);
        }

        s_dY[idx] = (y_pred - y_tgt) / float(M);
      }
    }
    __syncthreads();

    // ========== PHASE: Backward Layer 1 (compute dH0 = dY @ W1) ==========
    if (wid >= 2 && wid < 5) {
      const int compute_tid = (wid - 2) * 32 + lane;
      const int num_threads = 3 * 32;

      for (int idx = compute_tid; idx < TILE_M * HIDDEN_DIM; idx += num_threads) {
        const int m = idx / HIDDEN_DIM;
        const int h = idx % HIDDEN_DIM;

        float grad = 0.0f;
        #pragma unroll 4
        for (int o = 0; o < OUTPUT_DIM; o++) {
          grad += s_dY[m * OUTPUT_DIM + o] * bf16_to_f32(s_W1[o * HIDDEN_DIM + h]);
        }

        // Apply ReLU gradient (dH0 = dH0 * (H0 > 0))
        s_dH0[idx] = (s_H0[idx] > 0.0f) ? grad : 0.0f;
      }
    }
    __syncthreads();

    // ========== PHASE: Accumulate dW1 = dY^T @ H0 ==========
    if (wid >= 5 && wid < 7) {
      const int compute_tid = (wid - 5) * 32 + lane;
      const int num_threads = 2 * 32;

      for (int idx = compute_tid; idx < OUTPUT_DIM * HIDDEN_DIM; idx += num_threads) {
        const int o = idx / HIDDEN_DIM;
        const int h = idx % HIDDEN_DIM;

        float grad = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < TILE_M; m++) {
          const int global_m = m_start + m;
          if (global_m < M) {
            grad += s_dY[m * OUTPUT_DIM + o] * s_H0[m * HIDDEN_DIM + h];
          }
        }
        atomicAdd(&s_dW1[o * HIDDEN_DIM + h], grad);
      }
    }

    // ========== PHASE: Accumulate dW0 = dH0^T @ X ==========
    if (wid >= 2 && wid < 5) {
      const int compute_tid = (wid - 2) * 32 + lane;
      const int num_threads = 3 * 32;

      for (int idx = compute_tid; idx < HIDDEN_DIM * INPUT_DIM; idx += num_threads) {
        const int h = idx / INPUT_DIM;
        const int i = idx % INPUT_DIM;

        float grad = 0.0f;
        #pragma unroll 4
        for (int m = 0; m < TILE_M; m++) {
          const int global_m = m_start + m;
          if (global_m < M) {
            grad += s_dH0[m * HIDDEN_DIM + h] * bf16_to_f32(s_X[m * INPUT_DIM + i]);
          }
        }
        atomicAdd(&s_dW0[h * INPUT_DIM + i], grad);
      }
    }
    __syncthreads();
  }

  // ========== PHASE: Update weights W0 -= lr * dW0 ==========
  for (int idx = threadIdx.x; idx < HIDDEN_DIM * INPUT_DIM; idx += blockDim.x) {
    float w = bf16_to_f32(W0[idx]);
    float dw = s_dW0[idx];
    W0[idx] = f32_to_bf16(w - lr * dw);
  }

  // ========== PHASE: Update weights W1 -= lr * dW1 ==========
  for (int idx = threadIdx.x; idx < OUTPUT_DIM * HIDDEN_DIM; idx += blockDim.x) {
    float w = bf16_to_f32(W1[idx]);
    float dw = s_dW1[idx];
    W1[idx] = f32_to_bf16(w - lr * dw);
  }
}

// ============================================================================
// Host code
// ============================================================================

int main() {
  const int M = 256;
  const float lr = 0.01f;

  std::cout << "MLP Training Megakernel:\n";
  std::cout << "  Architecture: " << INPUT_DIM << " -> " << HIDDEN_DIM << " (ReLU) -> " << OUTPUT_DIM << "\n";
  std::cout << "  Batch size: " << M << ", Learning rate: " << lr << "\n\n";

  // Initialize data
  std::mt19937 rng(42);
  std::normal_distribution<float> weight_init(0.0f, 0.1f);
  std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

  std::vector<float> hX(M * INPUT_DIM);
  std::vector<float> hW0(HIDDEN_DIM * INPUT_DIM);
  std::vector<float> hW1(OUTPUT_DIM * HIDDEN_DIM);
  std::vector<float> hW0_true(HIDDEN_DIM * INPUT_DIM);
  std::vector<float> hW1_true(OUTPUT_DIM * HIDDEN_DIM);

  for (auto& v : hX) v = input_dist(rng);
  for (auto& v : hW0) v = weight_init(rng);
  for (auto& v : hW1) v = weight_init(rng);
  for (auto& v : hW0_true) v = weight_init(rng);
  for (auto& v : hW1_true) v = weight_init(rng);

  // Generate targets from true weights
  std::vector<float> hYt(M * OUTPUT_DIM, 0.0f);
  for (int m = 0; m < M; m++) {
    // H0_true = ReLU(X @ W0_true^T)
    std::vector<float> h0(HIDDEN_DIM, 0.0f);
    for (int h = 0; h < HIDDEN_DIM; h++) {
      float acc = 0.0f;
      for (int i = 0; i < INPUT_DIM; i++) {
        acc += hX[m * INPUT_DIM + i] * hW0_true[h * INPUT_DIM + i];
      }
      h0[h] = std::max(0.0f, acc);
    }

    // Y_true = H0_true @ W1_true^T
    for (int o = 0; o < OUTPUT_DIM; o++) {
      float acc = 0.0f;
      for (int h = 0; h < HIDDEN_DIM; h++) {
        acc += h0[h] * hW1_true[o * HIDDEN_DIM + h];
      }
      hYt[m * OUTPUT_DIM + o] = acc;
    }
  }

  // Add small noise
  std::normal_distribution<float> noise(0.0f, 0.01f);
  for (auto& y : hYt) y += noise(rng);

  // Convert to bf16
  std::vector<__nv_bfloat16> Xbf(M * INPUT_DIM);
  std::vector<__nv_bfloat16> W0bf(HIDDEN_DIM * INPUT_DIM);
  std::vector<__nv_bfloat16> W1bf(OUTPUT_DIM * HIDDEN_DIM);
  std::vector<__nv_bfloat16> Ytbf(M * OUTPUT_DIM);

  for (size_t i = 0; i < hX.size(); i++) Xbf[i] = f32_to_bf16(hX[i]);
  for (size_t i = 0; i < hW0.size(); i++) W0bf[i] = f32_to_bf16(hW0[i]);
  for (size_t i = 0; i < hW1.size(); i++) W1bf[i] = f32_to_bf16(hW1[i]);
  for (size_t i = 0; i < hYt.size(); i++) Ytbf[i] = f32_to_bf16(hYt[i]);

  // Allocate device memory
  __nv_bfloat16 *dX, *dW0, *dW1, *dYt, *dYout;
  cudaMalloc(&dX, M * INPUT_DIM * sizeof(__nv_bfloat16));
  cudaMalloc(&dW0, HIDDEN_DIM * INPUT_DIM * sizeof(__nv_bfloat16));
  cudaMalloc(&dW1, OUTPUT_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16));
  cudaMalloc(&dYt, M * OUTPUT_DIM * sizeof(__nv_bfloat16));
  cudaMalloc(&dYout, M * OUTPUT_DIM * sizeof(__nv_bfloat16));

  cudaMemcpy(dX, Xbf.data(), Xbf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dW0, W0bf.data(), W0bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dW1, W1bf.data(), W1bf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dYt, Ytbf.data(), Ytbf.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

  // Compute shared memory size
  size_t smem = 0;
  smem += TILE_M * INPUT_DIM * sizeof(__nv_bfloat16);        // s_X
  smem += HIDDEN_DIM * INPUT_DIM * sizeof(__nv_bfloat16);    // s_W0
  smem += OUTPUT_DIM * HIDDEN_DIM * sizeof(__nv_bfloat16);   // s_W1
  smem += TILE_M * HIDDEN_DIM * sizeof(float);               // s_H0
  smem += TILE_M * OUTPUT_DIM * sizeof(float);               // s_Y
  smem += TILE_M * OUTPUT_DIM * sizeof(float);               // s_dY
  smem += TILE_M * HIDDEN_DIM * sizeof(float);               // s_dH0
  smem += HIDDEN_DIM * INPUT_DIM * sizeof(float);            // s_dW0
  smem += OUTPUT_DIM * HIDDEN_DIM * sizeof(float);           // s_dW1

  std::cout << "Shared memory required: " << smem << " bytes\n";
  std::cout << "Launching with 1 block, " << (NUM_WARPS * 32) << " threads\n\n";

  // Training loop
  for (int iter = 0; iter < 100; iter++) {
    mlp_train_kernel<<<1, NUM_WARPS * 32, smem>>>(dX, dW0, dW1, dYt, dYout, M, lr);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
      break;
    }

    // Evaluate every 10 iterations
    if (iter % 10 == 0 || iter == 99) {
      std::vector<__nv_bfloat16> W0_new(HIDDEN_DIM * INPUT_DIM);
      std::vector<__nv_bfloat16> W1_new(OUTPUT_DIM * HIDDEN_DIM);
      std::vector<__nv_bfloat16> Y_pred(M * OUTPUT_DIM);

      cudaMemcpy(W0_new.data(), dW0, W0_new.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
      cudaMemcpy(W1_new.data(), dW1, W1_new.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
      cudaMemcpy(Y_pred.data(), dYout, Y_pred.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

      // Compute loss
      float loss = 0.0f;
      for (int i = 0; i < M * OUTPUT_DIM; i++) {
        float diff = bf16_to_f32(Y_pred[i]) - hYt[i];
        loss += diff * diff;
      }
      loss /= (M * OUTPUT_DIM);

      std::cout << "Iter " << iter << " | Loss: " << loss
                << " | Y[0:2]: [" << bf16_to_f32(Y_pred[0]) << ", " << bf16_to_f32(Y_pred[1]) << "]\n";
    }
  }

  cudaFree(dX);
  cudaFree(dW0);
  cudaFree(dW1);
  cudaFree(dYt);
  cudaFree(dYout);

  std::cout << "\nTraining complete!\n";
  return 0;
}
