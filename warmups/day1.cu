// naive GEMM on global memory only

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>

#include "common.hpp"
#include <nanobench.h>

constexpr long M = 512;
constexpr long N = 512;
constexpr long K = 512;

/* Naive global mem gemm.
 * Each thread computes 1 element
 * C[i, j] = sum_k A[i, k] * B[k, j]
 */
__global__ void gpu_gemm(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
  const int i = (blockIdx.y * blockDim.y) + threadIdx.y;
  const int j = (blockIdx.x * blockDim.x) + threadIdx.x;

  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  if (i >= M || i >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; k++) {
    // we need A[i,:] and B[:,j]
    acc += A[(i * lda) + k] * B[(k * ldb) + j];
  }
  C[(i * ldc) + j] = acc;
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

  dim3 block(16, 16);                      // 256 threads/block
  dim3 grid((N + 15) / 16, (M + 15) / 16); // enough blocks to cover matrix

  const auto t0_gpu = std::chrono::steady_clock::now();
  gpu_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
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

  ankerl::nanobench::Bench bench;
  bench.title("cpu/gpu naive gemm")
       .warmup(0)
       .batch(1)
       .epochs(1)
       .minEpochIterations(1)
       .unit("ns")
       .relative(false)
       .performanceCounters(false);

  auto r1 = ankerl::nanobench::Bench().run("cpu", [&] {
    cpu_gemm(hA, hB, hRef, M, N, K);
  });

  auto r2 = ankerl::nanobench::Bench().run("gpu", [&] {
      gpu_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
      check(cudaDeviceSynchronize());
  });
  // bench.results().back().
  std::cout << "t : " << r1.results().size() << "\n\n";
  std::cout << "cpu elapsed: " << r1.results().back().average(ankerl::nanobench::Result::Measure::elapsed) << "\n";
  std::cout << "gpu elapsed: " << r2.results().back().average(ankerl::nanobench::Result::Measure::elapsed) << "\n";

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
