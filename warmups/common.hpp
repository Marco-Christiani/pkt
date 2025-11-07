#include <cuda_runtime.h>

#include <iostream>
#include <vector>

static void check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";
    std::exit(1);
  }
}

/* Naive single threaded gemm */
void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
              int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += A[(i * K) + k] * B[(k * N) + j];
      }
      C[(i * N) + j] = acc;
    }
  }
}
