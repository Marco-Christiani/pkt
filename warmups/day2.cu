#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>

#include "common.hpp"

constexpr long M = 1024;
constexpr long N = 1024;
constexpr long K = 1024;

// constexpr int TILE_M = 32;
// constexpr int TILE_N = 32;
// constexpr int TILE_K = 32;
// TODO: support non square tiles?
constexpr int BLOCK = 32;

// Cooperative-tiled GEMM
// Row-major: A[M,K], B[K,N], C[M,N]
// Each block computes one BLOCKxBLOCK tile of C.
// Each thread loads one element into shared memory from A and B tiles.
template <int BLOCK>
__global__ void gpu_gemm(const float* __restrict__ A, // [M,K]
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
  const int m = blockRow * BLOCK + ty; // row in C
  const int n = blockCol * BLOCK + tx; // col in C

  __shared__ float As[BLOCK][BLOCK];
  __shared__ float Bs[BLOCK][BLOCK];

  float acc = 0.0f;

  const int num_k_tiles = (K + BLOCK - 1) / BLOCK;

  // Loop over Asub and Bsub tiles needed for this C tile
  for (int t = 0; t < num_k_tiles; ++t) {
    const int k0 = t * BLOCK;

    // Cooperative load:
    // - As loads A[m, k0 + tx]  (row-major, contiguous across tx)
    // - Bs loads B[k0 + ty, n]
    float a_reg = 0.0f;
    float b_reg = 0.0f;

    if (m < M && (k0 + tx) < K) {
      a_reg = A[m * K + (k0 + tx)];
    }
    if ((k0 + ty) < K && n < N) {
      b_reg = B[(k0 + ty) * N + n];
    }

    As[ty][tx] = a_reg;
    Bs[ty][tx] = b_reg;

    __syncthreads();

// tile mac - sum over the shared block-sized k tile
#pragma unroll
    for (int e = 0; e < BLOCK; ++e) {
      acc += As[ty][e] * Bs[e][tx];
    }

    __syncthreads();
  }

  if (m < M && n < N) {
    C[m * N + n] = acc;
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

  // dim3 block(TILE_N, TILE_M);
  // dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  // int smem = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
  dim3 block(BLOCK, BLOCK);
  dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

  const auto t0_gpu = std::chrono::steady_clock::now();
  gpu_gemm<BLOCK><<<grid, block>>>(dA, dB, dC, M, N, K);
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
