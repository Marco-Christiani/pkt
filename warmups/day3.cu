// Add double buffering
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda/pipeline>
#include <iostream>
#include <random>
#include <vector>

#include "common.hpp"

constexpr long M = 1024;
constexpr long N = 1024;
constexpr long K = 1024;

constexpr int BLOCK = 32;

// Cooperative-tiled GEMM with double buffering - SYNC!
// Row-major: A[M,K], B[K,N], C[M,N]
// Each block computes one BLOCKxBLOCK tile of C.
// Each thread loads one element into shared memory from A and B tiles.
template <int BLOCK>
__global__ void gpu_gemm_sync(const float* __restrict__ A, // [M,K]
                              const float* __restrict__ B, // [K,N]
                              float* __restrict__ C,       // [M,N]
                              int M, int N, int K) {
  // Block tile coordinates in C
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  // Thread coordinates within the C tile
  const int ty = threadIdx.y; // row within tile
  const int tx = threadIdx.x; // col within tile

  // Global coords of the element this thread accumulates
  const int m = (blockRow * BLOCK) + ty; // row in C
  const int n = (blockCol * BLOCK) + tx; // col in C

  int stage = 0;
  __shared__ float As[2][BLOCK][BLOCK];
  __shared__ float Bs[2][BLOCK][BLOCK];

  float acc = 0.0f;

  const int num_k_tiles = (K + BLOCK - 1) / BLOCK;
  // load tile 0 stage 0
  if (m < M && tx < K) {
    As[0][ty][tx] = A[(m * K) + tx];
  }
  if (ty < K && n < N) {
    Bs[0][ty][tx] = B[(ty * N) + n];
  }
  __syncthreads();

  // Loop over Asub and Bsub tiles needed for this C tile
  for (int t = 0; t < num_k_tiles; ++t) {
    const int next_stage = stage ^ 1;
    if (t + 1 < num_k_tiles) {
      const int k1 = (t + 1) * BLOCK;
      // load next tile in next stage
      float next_a = 0.0f;
      float next_b = 0.0f;
      if (m < M && (k1 + tx) < K) {
        next_a = A[(m * K) + (k1 + tx)];
      }
      if ((k1 + ty) < K && n < N) {
        next_b = B[((k1 + ty) * N) + n];
      }
      As[next_stage][ty][tx] = next_a;
      Bs[next_stage][ty][tx] = next_b;
    }

// tile mac - sum over the shared block-sized k tile
#pragma unroll
    for (int e = 0; e < BLOCK; ++e) {
      acc += As[stage][ty][e] * Bs[stage][e][tx];
    }
    if (t + 1 < num_k_tiles) {
      __syncthreads();
    }
    stage = next_stage;
  }

  if (m < M && n < N) {
    C[(m * N) + n] = acc;
  }
}

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

template <int BLOCK>
__global__ void gpu_gemm(const float* __restrict__ A,  // [M, K], row-major
                         const float* __restrict__ BT, // [N, K], row-major (transposed B)
                         float* __restrict__ C,        // [M, N], row-major
                         int M, int N, int K) {
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  const int m_base = block_row * BLOCK; // starting row in C/A
  const int n_base = block_col * BLOCK; // starting col in C / row in BT

  // shared tiles: 2 stages
  __shared__ float As[2][BLOCK][BLOCK];
  __shared__ float Bs[2][BLOCK][BLOCK];

  // pipeline state (avoid ctor warning by manual storage)
  __shared__ alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>) unsigned char
      pipe_storage[sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>)];

  auto block = cg::this_thread_block();
  auto* pipe_state =
      reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, 2>*>(pipe_storage);
  auto pipe = cuda::make_pipeline(block, pipe_state);

  const int num_k_tiles = (K + BLOCK - 1) / BLOCK;
  float acc = 0.0f;
  int stage = 0;

  // small helper to zero a row tail cooperatively
  auto zero_tail = [&](float* row, int start) {
    for (int c = tx + start; c < BLOCK; c += blockDim.x) {
      row[c] = 0.0f;
    }
  };

  // preload tile 0 into stage 0
  if (num_k_tiles > 0) {
    const int k0 = 0;
    pipe.producer_acquire();
    for (int r = 0; r < BLOCK; ++r) {
      const int gm = m_base + r;
      const int gn = n_base + r;

      // A row
      {
        float* dst = &As[0][r][0];
        int remaining = K - k0;
        int copy_elems = remaining > BLOCK ? BLOCK : (remaining > 0 ? remaining : 0);
        if (gm < M && copy_elems > 0) {
          const float* src = &A[gm * K + k0];
          cuda::memcpy_async(block, dst, src, copy_elems * sizeof(float), pipe);
          if (copy_elems < BLOCK) {
            zero_tail(dst, copy_elems);
          }
        } else {
          zero_tail(dst, 0);
        }
      }

      // BT row (note: BT is [N, K], so row = n, contiguous in K)
      {
        float* dst = &Bs[0][r][0];
        int remaining = K - k0;
        int copy_elems = remaining > BLOCK ? BLOCK : (remaining > 0 ? remaining : 0);
        if (gn < N && copy_elems > 0) {
          const float* src = &BT[gn * K + k0];
          cuda::memcpy_async(block, dst, src, copy_elems * sizeof(float), pipe);
          if (copy_elems < BLOCK) {
            zero_tail(dst, copy_elems);
          }
        } else {
          zero_tail(dst, 0);
        }
      }
    }
    pipe.producer_commit();
  }

  for (int t = 0; t < num_k_tiles; ++t) {
    const int k0 = t * BLOCK;
    const int next_stage = stage ^ 1;

    // schedule next tile (t+1) into next_stage
    if (t + 1 < num_k_tiles) {
      const int k_next = (t + 1) * BLOCK;
      pipe.producer_acquire();
      for (int r = 0; r < BLOCK; ++r) {
        const int gm = m_base + r;
        const int gn = n_base + r;

        // A row
        {
          float* dst = &As[next_stage][r][0];
          int remaining = K - k_next;
          int copy_elems = remaining > BLOCK ? BLOCK : (remaining > 0 ? remaining : 0);
          if (gm < M && copy_elems > 0) {
            const float* src = &A[gm * K + k_next];
            cuda::memcpy_async(block, dst, src, copy_elems * sizeof(float), pipe);
            if (copy_elems < BLOCK) {
              zero_tail(dst, copy_elems);
            }
          } else {
            zero_tail(dst, 0);
          }
        }

        // BT row
        {
          float* dst = &Bs[next_stage][r][0];
          int remaining = K - k_next;
          int copy_elems = remaining > BLOCK ? BLOCK : (remaining > 0 ? remaining : 0);
          if (gn < N && copy_elems > 0) {
            const float* src = &BT[gn * K + k_next];
            cuda::memcpy_async(block, dst, src, copy_elems * sizeof(float), pipe);
            if (copy_elems < BLOCK) {
              zero_tail(dst, copy_elems);
            }
          } else {
            zero_tail(dst, 0);
          }
        }
      }
      pipe.producer_commit();
    }

    // wait for current tile to be ready
    pipe.consumer_wait();
    __syncthreads();

    // compute on current stage
#pragma unroll
    for (int e = 0; e < BLOCK; ++e) {
      const float a = As[stage][ty][e];
      const float b = Bs[stage][e][tx];
      acc += a * b;
    }

    pipe.consumer_release();
    __syncthreads();

    stage = next_stage;
  }

  // write back
  const int m = m_base + ty;
  const int n = n_base + tx;
  if (m < M && n < N) {
    C[m * N + n] = acc;
  }
}

void transpose_B_to_BT(const float* B, float* BT, int K, int N) {
  // B:  [K, N] row-major
  // BT: [N, K] row-major
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      BT[n * K + k] = B[k * N + n];
    }
  }
}

int main() {
  std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-8.f, 16.f);
  for (float& f : hA) {
    f = dist(rng);
  }
  for (float& f : hB) {
    f = dist(rng);
  }
  const auto t0_cpu = std::chrono::steady_clock::now();
  cpu_gemm(hA, hB, hRef, M, N, K);
  const auto t1_cpu = std::chrono::steady_clock::now();
  const auto dur_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(t1_cpu - t0_cpu);

  float *dA{}, *dB{}, *dC{};
  check(cudaMalloc(&dA, M * K * sizeof(float)));
  check(cudaMalloc(&dB, K * N * sizeof(float)));
  check(cudaMalloc(&dC, M * N * sizeof(float)));
  check(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  ///
  float* dBT{};
  std::vector<float> hBT(N * K);
  transpose_B_to_BT(hB.data(), hBT.data(), K, N);

  cudaMemcpy(dBT, hBT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice); // <-- BT upload
  ///

  // dim3 block(TILE_N, TILE_M);
  // dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  // int smem = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
  dim3 block(BLOCK, BLOCK);
  dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

  // shared: 2 stages * 2 tiles * BLOCK*BLOCK floats
  size_t shmem = 2 * 2 * BLOCK * BLOCK * sizeof(float);

  const auto t0_gpu = std::chrono::steady_clock::now();
  // gpu_gemm<BLOCK><<<grid, block>>>(dA, dB, dC, M, N, K);
  gpu_gemm<BLOCK><<<grid, block, shmem>>>(dA, dBT, dC, M, N, K);
  check(cudaDeviceSynchronize());
  const auto t1_gpu = std::chrono::steady_clock::now();
  const auto dur_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(t1_gpu - t0_gpu);

  check(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // verify
  float max_err = 0.;
  for (int i = 0; i < M * N; i++) {
    max_err = std::max(max_err, std::abs(hC[i] - hRef[i]));
  }
  std::cout << "cpu: " << dur_cpu << "\n";
  std::cout << "gpu: " << dur_gpu << "\n";
  std::cout << "max(error): " << max_err << "\n";

  // v2 ------------------------------------------------------------------------
  // constexpr int BLOCK = 32;
  // dim3 block2(BLOCK, BLOCK);
  // dim3 grid2((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
  //
  // const auto t0_gpu2 = std::chrono::steady_clock::now();
  // gpu_gemm2<BLOCK><<<grid2, block2>>>(dA, dB, dC, M, N, K);
  // check(cudaDeviceSynchronize());
  // ---------------------------------------------------------------------------

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
