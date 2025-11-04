#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const int n = 1 << 20; // ~1 million elements
  const size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n);

  for (int i = 0; i < n; i++) {
    h_a[i] = float(i);
    h_b[i] = float(2 * i);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    printf("c[%d] = %.1f\n", i, h_c[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
