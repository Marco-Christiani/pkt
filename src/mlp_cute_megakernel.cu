// Trying to use CUTE with a basic DAG kind of pattern
// this seems like a much better direction that previous things i tried
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using OpBF16TN = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using AtomBF16TN = cute::MMA_Atom<OpBF16TN>;

// -------------------------------------------
// Generic tiled GEMM: C (f32) += A (bf16) @ B^T (bf16)
// Accumulates across K dimension
// -------------------------------------------
template <int TM, int TN, int TK>
struct TiledGemmBF16TN {
  __nv_bfloat16* A_base;
  int ldA;
  __nv_bfloat16* B_base;
  int ldB;
  float* C_base;
  int ldC;
  int K_full;

  __device__ inline void run_tile(int m0, int n0) const {
    auto mma = cute::make_tiled_mma(AtomBF16TN{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    auto gC = cute::make_tensor(
        cute::make_gmem_ptr(C_base + m0 * ldC + n0),
        cute::make_shape(cute::Int<TM>{}, cute::Int<TN>{}),
        cute::make_stride(ldC, cute::Int<1>{})
    );

    auto tCgC = thr_mma.partition_C(gC);
    auto fragC = thr_mma.make_fragment_C(tCgC);
    cute::clear(fragC);

    for (int k0 = 0; k0 < K_full; k0 += TK) {
      auto gA = cute::make_tensor(
          cute::make_gmem_ptr(A_base + m0 * ldA + k0),
          cute::make_shape(cute::Int<TM>{}, cute::Int<TK>{}),
          cute::make_stride(ldA, cute::Int<1>{})
      );

      auto gB = cute::make_tensor(
          cute::make_gmem_ptr(B_base + n0 * ldB + k0),
          cute::make_shape(cute::Int<TN>{}, cute::Int<TK>{}),
          cute::make_stride(ldB, cute::Int<1>{})
      );

      auto tAgA = thr_mma.partition_A(gA);
      auto tBgB = thr_mma.partition_B(gB);

      auto fragA = thr_mma.make_fragment_A(tAgA);
      auto fragB = thr_mma.make_fragment_B(tBgB);

      cute::copy(tAgA, fragA);
      cute::copy(tBgB, fragB);
      cute::gemm(mma, fragA, fragB, fragC);
    }

    cute::copy(fragC, tCgC);
  }
};

// -------------------------------------------
// Convert float tile to bf16
// -------------------------------------------
template <int TM, int TN>
struct TiledF32ToBF16 {
  float* src;
  int ld_src;
  __nv_bfloat16* dst;
  int ld_dst;

  __device__ inline void run_tile(int m0, int n0) const {
    for (int i = threadIdx.x; i < TM * TN; i += 32) {
      int r = i / TN;
      int c = i % TN;
      int row = m0 + r;
      int col = n0 + c;
      
      float val = src[row * ld_src + col];
      dst[row * ld_dst + col] = __float2bfloat16(val);
    }
  }
};

// -------------------------------------------
// Elementwise ReLU on a f32 tile
// -------------------------------------------
template <int TM, int TN>
struct TiledReLU {
  float* data;
  int ld;

  __device__ inline void run_tile(int m0, int n0) const {
    for (int i = threadIdx.x; i < TM * TN; i += 32) {
      int r = i / TN;
      int c = i % TN;
      int row = m0 + r;
      int col = n0 + c;
      
      float* ptr = data + row * ld + col;
      float val = *ptr;
      *ptr = val > 0.f ? val : 0.f;
    }
  }
};

// -------------------------------------------
// MLP Block: out = ReLU(in @ W^T)
// Processes ALL output tiles for given batch tile
// -------------------------------------------
template <int TM, int TN, int TK>
struct MLPBlock {
  int B;
  int in_dim;
  int out_dim;
  
  __nv_bfloat16* in_bf16;  int ld_in_bf16;
  __nv_bfloat16* W;        int ldW;
  float* out_f32;          int ld_out_f32;
  __nv_bfloat16* out_bf16; int ld_out_bf16;

  __device__ inline void run_all_tiles(int m0) const {
    // Process all output feature tiles for this batch tile
    for (int n0 = 0; n0 < out_dim; n0 += TN) {
      TiledGemmBF16TN<TM, TN, TK>{
          in_bf16, ld_in_bf16,
          W, ldW,
          out_f32, ld_out_f32,
          in_dim
      }.run_tile(m0, n0);
      
      __syncwarp();
      
      TiledReLU<TM, TN>{out_f32, ld_out_f32}.run_tile(m0, n0);
      
      __syncwarp();
      
      TiledF32ToBF16<TM, TN>{
          out_f32, ld_out_f32,
          out_bf16, ld_out_bf16
      }.run_tile(m0, n0);
      
      __syncwarp();
    }
  }
};

// -------------------------------------------
// Output Layer: out = in @ W^T (no activation)
// Processes ALL output tiles for given batch tile
// -------------------------------------------
template <int TM, int TN, int TK>
struct OutputLayer {
  int B;
  int in_dim;
  int out_dim;
  
  __nv_bfloat16* in_bf16;  int ld_in_bf16;
  __nv_bfloat16* W;        int ldW;
  float* out_f32;          int ld_out_f32;

  __device__ inline void run_all_tiles(int m0) const {
    for (int n0 = 0; n0 < out_dim; n0 += TN) {
      TiledGemmBF16TN<TM, TN, TK>{
          in_bf16, ld_in_bf16,
          W, ldW,
          out_f32, ld_out_f32,
          in_dim
      }.run_tile(m0, n0);
      
      __syncwarp();
    }
  }
};

// -------------------------------------------
// Full MLP Forward: X -> H1 -> H2 -> Y
// Each block processes one batch tile through entire pipeline
// -------------------------------------------
template <int TM, int TN, int TK>
struct MLPForward {
  int B, In, Hdim1, Hdim2, Out;
  
  __nv_bfloat16* X;   int ldX;
  
  __nv_bfloat16* W0;  int ldW0;
  float* H1_f32;      int ldH1_f32;
  __nv_bfloat16* H1_bf16; int ldH1_bf16;
  
  __nv_bfloat16* W1;  int ldW1;
  float* H2_f32;      int ldH2_f32;
  __nv_bfloat16* H2_bf16; int ldH2_bf16;
  
  __nv_bfloat16* W2;  int ldW2;
  float* Y;           int ldY;

  __device__ inline void run_pipeline(int m0) const {
    // Layer 1: H1 = ReLU(X @ W0^T)
    MLPBlock<TM, TN, TK>{
        B, In, Hdim1,
        X, ldX,
        W0, ldW0,
        H1_f32, ldH1_f32,
        H1_bf16, ldH1_bf16
    }.run_all_tiles(m0);
    
    // Layer 2: H2 = ReLU(H1 @ W1^T)
    MLPBlock<TM, TN, TK>{
        B, Hdim1, Hdim2,
        H1_bf16, ldH1_bf16,
        W1, ldW1,
        H2_f32, ldH2_f32,
        H2_bf16, ldH2_bf16
    }.run_all_tiles(m0);
    
    // Output Layer: Y = H2 @ W2^T
    OutputLayer<TM, TN, TK>{
        B, Hdim2, Out,
        H2_bf16, ldH2_bf16,
        W2, ldW2,
        Y, ldY
    }.run_all_tiles(m0);
  }
};

// -------------------------------------------
// Working kernel without CUTE that produces correct results
// -------------------------------------------
template <int TM, int TN, int TK>
__global__ void mlp_kernel(
    int B, int In, int Hdim1, int Hdim2, int Out,
    __nv_bfloat16* X, int ldX,
    __nv_bfloat16* W0, int ldW0,
    __nv_bfloat16* W1, int ldW1,
    __nv_bfloat16* W2, int ldW2,
    float* H1_f32, int ldH1_f32,
    __nv_bfloat16* H1_bf16, int ldH1_bf16,
    float* H2_f32, int ldH2_f32,
    __nv_bfloat16* H2_bf16, int ldH2_bf16,
    float* Y, int ldY
) {
  if (threadIdx.x >= 32) return;

  int m0 = blockIdx.x * TM;
  if (m0 >= B) return;

  // Process each batch element through the network
  // Layer 1: H1 = ReLU(X @ W0^T)
  for (int i = threadIdx.x; i < TM * Hdim1; i += 32) {
    int row = i / Hdim1;
    int col = i % Hdim1;
    if (m0 + row >= B) continue;

    float sum = 0.0f;
    for (int k = 0; k < In; k++) {
      float a = __bfloat162float(X[(m0 + row) * ldX + k]);
      float b = __bfloat162float(W0[col * ldW0 + k]);
      sum += a * b;
    }

    // ReLU activation
    sum = sum > 0.0f ? sum : 0.0f;
    H1_f32[(m0 + row) * ldH1_f32 + col] = sum;
    H1_bf16[(m0 + row) * ldH1_bf16 + col] = __float2bfloat16(sum);
  }
  __syncthreads();

  // Layer 2: H2 = ReLU(H1 @ W1^T)
  for (int i = threadIdx.x; i < TM * Hdim2; i += 32) {
    int row = i / Hdim2;
    int col = i % Hdim2;
    if (m0 + row >= B) continue;

    float sum = 0.0f;
    for (int k = 0; k < Hdim1; k++) {
      float a = __bfloat162float(H1_bf16[(m0 + row) * ldH1_bf16 + k]);
      float b = __bfloat162float(W1[col * ldW1 + k]);
      sum += a * b;
    }

    // ReLU activation
    sum = sum > 0.0f ? sum : 0.0f;
    H2_f32[(m0 + row) * ldH2_f32 + col] = sum;
    H2_bf16[(m0 + row) * ldH2_bf16 + col] = __float2bfloat16(sum);
  }
  __syncthreads();

  // Output layer: Y = H2 @ W2^T (no activation)
  for (int i = threadIdx.x; i < TM * Out; i += 32) {
    int row = i / Out;
    int col = i % Out;
    if (m0 + row >= B) continue;

    float sum = 0.0f;
    for (int k = 0; k < Hdim2; k++) {
      float a = __bfloat162float(H2_bf16[(m0 + row) * ldH2_bf16 + k]);
      float b = __bfloat162float(W2[col * ldW2 + k]);
      sum += a * b;
    }

    Y[(m0 + row) * ldY + col] = sum;
  }
}

// -------------------------------------------
// LAUNCHER
// -------------------------------------------
template <int TM, int TN, int TK>
void launch_mlp_kernel(
    dim3 grid, dim3 block,
    int B, int In, int Hdim1, int Hdim2, int Out,
    __nv_bfloat16* X, int ldX,
    __nv_bfloat16* W0, int ldW0,
    __nv_bfloat16* W1, int ldW1,
    __nv_bfloat16* W2, int ldW2,
    float* H1_f32, int ldH1_f32,
    __nv_bfloat16* H1_bf16, int ldH1_bf16,
    float* H2_f32, int ldH2_f32,
    __nv_bfloat16* H2_bf16, int ldH2_bf16,
    float* Y, int ldY,
    cudaStream_t stream = nullptr
) {
  static_assert(TM == 16 && TN == 8 && TK == 16,
                "Hard-coded for SM80 16x8x16 BF16 MMA");

  mlp_kernel<TM, TN, TK><<<grid, block, 0, stream>>>(
      B, In, Hdim1, Hdim2, Out,
      X, ldX, W0, ldW0, W1, ldW1, W2, ldW2,
      H1_f32, ldH1_f32, H1_bf16, ldH1_bf16,
      H2_f32, ldH2_f32, H2_bf16, ldH2_bf16,
      Y, ldY
  );
}

// -------------------------------------------
// MAIN
// -------------------------------------------
#include <cstdio>
#include <cstdlib>

static void cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    printf("CUDA ERROR at %s: %s\n", msg, cudaGetErrorString(err));
    std::abort();
  }
}

int main() {
  const int B     = 32;
  const int In    = 128;
  const int Hdim1 = 256;
  const int Hdim2 = 192;
  const int Out   = 64;

  constexpr int TM = 16;
  constexpr int TN = 8;
  constexpr int TK = 16;

  const int ldX  = In;
  const int ldW0 = In;
  const int ldW1 = Hdim1;
  const int ldW2 = Hdim2;
  const int ldH1_f32  = Hdim1;
  const int ldH1_bf16 = Hdim1;
  const int ldH2_f32  = Hdim2;
  const int ldH2_bf16 = Hdim2;
  const int ldY  = Out;

  size_t size_X   = B * In;
  size_t size_W0  = Hdim1 * In;
  size_t size_W1  = Hdim2 * Hdim1;
  size_t size_W2  = Out * Hdim2;
  size_t size_H1  = B * Hdim1;
  size_t size_H2  = B * Hdim2;
  size_t size_Y   = B * Out;

  __nv_bfloat16* hX  = new __nv_bfloat16[size_X];
  __nv_bfloat16* hW0 = new __nv_bfloat16[size_W0];
  __nv_bfloat16* hW1 = new __nv_bfloat16[size_W1];
  __nv_bfloat16* hW2 = new __nv_bfloat16[size_W2];
  float* hY  = new float[size_Y];

  // Random initialization with better range
  for (size_t i = 0; i < size_X;  ++i) hX[i]  = __float2bfloat16((float(rand())/RAND_MAX - 0.5f) * 0.1f);
  for (size_t i = 0; i < size_W0; ++i) hW0[i] = __float2bfloat16((float(rand())/RAND_MAX - 0.5f) * 0.1f);
  for (size_t i = 0; i < size_W1; ++i) hW1[i] = __float2bfloat16((float(rand())/RAND_MAX - 0.5f) * 0.1f);
  for (size_t i = 0; i < size_W2; ++i) hW2[i] = __float2bfloat16((float(rand())/RAND_MAX - 0.5f) * 0.1f);

  __nv_bfloat16 *dX, *dW0, *dW1, *dW2, *dH1_bf16, *dH2_bf16;
  float *dH1_f32, *dH2_f32, *dY;
  
  cuda_check(cudaMalloc(&dX,  size_X  * sizeof(__nv_bfloat16)), "malloc dX");
  cuda_check(cudaMalloc(&dW0, size_W0 * sizeof(__nv_bfloat16)), "malloc dW0");
  cuda_check(cudaMalloc(&dW1, size_W1 * sizeof(__nv_bfloat16)), "malloc dW1");
  cuda_check(cudaMalloc(&dW2, size_W2 * sizeof(__nv_bfloat16)), "malloc dW2");
  cuda_check(cudaMalloc(&dH1_f32,  size_H1 * sizeof(float)), "malloc dH1_f32");
  cuda_check(cudaMalloc(&dH1_bf16, size_H1 * sizeof(__nv_bfloat16)), "malloc dH1_bf16");
  cuda_check(cudaMalloc(&dH2_f32,  size_H2 * sizeof(float)), "malloc dH2_f32");
  cuda_check(cudaMalloc(&dH2_bf16, size_H2 * sizeof(__nv_bfloat16)), "malloc dH2_bf16");
  cuda_check(cudaMalloc(&dY,  size_Y  * sizeof(float)), "malloc dY");

  cuda_check(cudaMemcpy(dX,  hX,  size_X  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy X");
  cuda_check(cudaMemcpy(dW0, hW0, size_W0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy W0");
  cuda_check(cudaMemcpy(dW1, hW1, size_W1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy W1");
  cuda_check(cudaMemcpy(dW2, hW2, size_W2 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy W2");

  // Grid: one block per batch tile (only need to cover batch dimension)
  dim3 block(32);
  dim3 grid((B + TM - 1) / TM);

  printf("Launching kernel with grid=%d blocks, block=32 threads\n", grid.x);
  printf("Dimensions: B=%d In=%d Hdim1=%d Hdim2=%d Out=%d\n", B, In, Hdim1, Hdim2, Out);

  launch_mlp_kernel<TM, TN, TK>(
      grid, block,
      B, In, Hdim1, Hdim2, Out,
      dX, ldX,
      dW0, ldW0,
      dW1, ldW1,
      dW2, ldW2,
      dH1_f32, ldH1_f32,
      dH1_bf16, ldH1_bf16,
      dH2_f32, ldH2_f32,
      dH2_bf16, ldH2_bf16,
      dY, ldY
  );

  cuda_check(cudaDeviceSynchronize(), "kernel sync");
  
  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
  }

  // Check intermediate results
  float* hH1 = new float[size_H1];
  cuda_check(cudaMemcpy(hH1, dH1_f32, size_H1 * sizeof(float), cudaMemcpyDeviceToHost), "copy H1");
  printf("H1[0] = %f, H1[1] = %f, H1[255] = %f\n", hH1[0], hH1[1], hH1[255]);
  delete[] hH1;

  cuda_check(cudaMemcpy(hY, dY, size_Y * sizeof(float), cudaMemcpyDeviceToHost), "copy Y");

  printf("\nSuccess! Output Y:\n");
  printf("Y[0] = %f\n", hY[0]);
  for (int i = 0; i < 10 && i < size_Y; i++) {
    printf("Y[%d] = %f\n", i, hY[i]);
  }

  delete[] hX; delete[] hW0; delete[] hW1; delete[] hW2; delete[] hY;
  cudaFree(dX); cudaFree(dW0); cudaFree(dW1); cudaFree(dW2);
  cudaFree(dH1_f32); cudaFree(dH1_bf16);
  cudaFree(dH2_f32); cudaFree(dH2_bf16);
  cudaFree(dY);

  return 0;
}
