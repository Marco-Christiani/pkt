// Trying to use CUTE with a basic DAG kind of pattern
// this seems like a much better direction that previous things i tried
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
template <int TM, int TN, int TK> struct TiledGemmBF16TN {
  __nv_bfloat16* A_base;
  int ldA;
  __nv_bfloat16* B_base;
  int ldB;
  float* C_base;
  int ldC;
  int K_full; // Full K dimension to loop over

  __device__ inline void run_tile(int m0, int n0) const {
    auto mma = cute::make_tiled_mma(AtomBF16TN{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    // Create view of C tile
    auto gC = cute::make_tensor(cute::make_gmem_ptr(C_base + m0 * ldC + n0),
                                cute::make_shape(cute::Int<TM>{}, cute::Int<TN>{}),
                                cute::make_stride(ldC, cute::Int<1>{}));

    auto tCgC = thr_mma.partition_C(gC);
    auto fragC = thr_mma.make_fragment_C(tCgC);
    cute::clear(fragC); // Initialize accumulator

    // Loop over K dimension
    for (int k0 = 0; k0 < K_full; k0 += TK) {
      // Create views of A and B tiles for this K slice
      auto gA = cute::make_tensor(cute::make_gmem_ptr(A_base + m0 * ldA + k0),
                                  cute::make_shape(cute::Int<TM>{}, cute::Int<TK>{}),
                                  cute::make_stride(ldA, cute::Int<1>{}));

      auto gB = cute::make_tensor(cute::make_gmem_ptr(B_base + n0 * ldB + k0),
                                  cute::make_shape(cute::Int<TN>{}, cute::Int<TK>{}),
                                  cute::make_stride(ldB, cute::Int<1>{}));

      // Partition and load
      auto fragA = thr_mma.partition_fragment_A(gA);
      auto fragB = thr_mma.partition_fragment_B(gB);

      cute::copy(gA, fragA);
      cute::copy(gB, fragB);

      // Accumulate
      cute::gemm(mma, fragA, fragB, fragC);
    }

    // Store result
    cute::copy(fragC, tCgC);
  }
};

// -------------------------------------------
// Convert float tile to bf16
// -------------------------------------------
template <int TM, int TN> struct TiledF32ToBF16 {
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
template <int TM, int TN> struct TiledReLU {
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
// MLP Forward Pass for one tile
// H (f32) = ReLU(X (bf16) @ W0^T (bf16))
// H_bf16 = convert(H)
// Y (f32) = H_bf16 @ W1^T (bf16)
// -------------------------------------------
template <int TM, int TN, int TK> struct MLPForward {
  // Input dimensions
  int B, In, Hdim, Out;

  // Input data (bf16)
  __nv_bfloat16* X;
  int ldX; // [B, In]
  __nv_bfloat16* W0;
  int ldW0; // [Hdim, In]
  __nv_bfloat16* W1;
  int ldW1; // [Out, Hdim]

  // Scratch (f32 for accumulation, bf16 for second gemm input)
  float* H;
  int ldH; // [B, Hdim] f32
  __nv_bfloat16* H_bf16;
  int ldH_bf16; // [B, Hdim] bf16
  float* Y;
  int ldY; // [B, Out] f32

  __device__ inline void run_tile(int m0, int n0_max) const {
    // First GEMM: H = X @ W0^T
    for (int n0 = 0; n0 < Hdim; n0 += TN) {
      if (n0 >= n0_max)
        break;

      TiledGemmBF16TN<TM, TN, TK>{
          X, ldX, W0, ldW0, H, ldH,
          In // K dimension
      }
          .run_tile(m0, n0);

      __syncwarp();

      // ReLU in-place
      TiledReLU<TM, TN>{H, ldH}.run_tile(m0, n0);

      __syncwarp();

      // Convert to bf16 for next gemm
      TiledF32ToBF16<TM, TN>{H, ldH, H_bf16, ldH_bf16}.run_tile(m0, n0);

      __syncwarp();
    }

    // Second GEMM: Y = H_bf16 @ W1^T
    for (int n0 = 0; n0 < Out; n0 += TN) {
      if (n0 >= n0_max)
        break;

      TiledGemmBF16TN<TM, TN, TK>{
          H_bf16, ldH_bf16, W1, ldW1, Y, ldY,
          Hdim // K dimension
      }
          .run_tile(m0, n0);

      __syncwarp();
    }
  }
};

// -------------------------------------------
// KERNEL
// -------------------------------------------
template <int TM, int TN, int TK>
__global__ void mlp_kernel(int B, int In, int Hdim, int Out, __nv_bfloat16* X, int ldX,
                           __nv_bfloat16* W0, int ldW0, __nv_bfloat16* W1, int ldW1, float* H,
                           int ldH, __nv_bfloat16* H_bf16, int ldH_bf16, float* Y, int ldY) {
  if (threadIdx.x >= 32)
    return;

  int m0 = blockIdx.y * TM;
  int n0 = blockIdx.x * TN;

  if (m0 >= B)
    return;

  MLPForward<TM, TN, TK>{B,  In,   Hdim, Out, X,      ldX,      W0, ldW0,
                         W1, ldW1, H,    ldH, H_bf16, ldH_bf16, Y,  ldY}
      .run_tile(m0, n0);
}

// -------------------------------------------
// LAUNCHER
// -------------------------------------------
template <int TM, int TN, int TK>
void launch_mlp_kernel(dim3 grid, dim3 block, int B, int In, int Hdim, int Out, __nv_bfloat16* X,
                       int ldX, __nv_bfloat16* W0, int ldW0, __nv_bfloat16* W1, int ldW1, float* H,
                       int ldH, __nv_bfloat16* H_bf16, int ldH_bf16, float* Y, int ldY,
                       cudaStream_t stream = nullptr) {
  static_assert(TM == 16 && TN == 8 && TK == 16, "Hard-coded for SM80 16x8x16 BF16 MMA");

  mlp_kernel<TM, TN, TK><<<grid, block, 0, stream>>>(B, In, Hdim, Out, X, ldX, W0, ldW0, W1, ldW1,
                                                     H, ldH, H_bf16, ldH_bf16, Y, ldY);
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
  const int B = 32;
  const int In = 128;
  const int Hdim = 256;
  const int Out = 64;

  constexpr int TM = 16;
  constexpr int TN = 8;
  constexpr int TK = 16;

  const int ldX = In;
  const int ldW0 = In;
  const int ldW1 = Hdim;
  const int ldH = Hdim;
  const int ldH_bf16 = Hdim;
  const int ldY = Out;

  size_t size_X = B * In;
  size_t size_W0 = Hdim * In;
  size_t size_W1 = Out * Hdim;
  size_t size_H = B * Hdim;
  size_t size_Y = B * Out;

  __nv_bfloat16* hX = new __nv_bfloat16[size_X];
  __nv_bfloat16* hW0 = new __nv_bfloat16[size_W0];
  __nv_bfloat16* hW1 = new __nv_bfloat16[size_W1];
  float* hY = new float[size_Y];

  for (size_t i = 0; i < size_X; ++i)
    hX[i] = __float2bfloat16(float(rand()) / RAND_MAX - 0.5f);
  for (size_t i = 0; i < size_W0; ++i)
    hW0[i] = __float2bfloat16(float(rand()) / RAND_MAX - 0.5f);
  for (size_t i = 0; i < size_W1; ++i)
    hW1[i] = __float2bfloat16(float(rand()) / RAND_MAX - 0.5f);

  __nv_bfloat16 *dX, *dW0, *dW1, *dH_bf16;
  float *dH, *dY;

  cuda_check(cudaMalloc(&dX, size_X * sizeof(__nv_bfloat16)), "malloc dX");
  cuda_check(cudaMalloc(&dW0, size_W0 * sizeof(__nv_bfloat16)), "malloc dW0");
  cuda_check(cudaMalloc(&dW1, size_W1 * sizeof(__nv_bfloat16)), "malloc dW1");
  cuda_check(cudaMalloc(&dH, size_H * sizeof(float)), "malloc dH");
  cuda_check(cudaMalloc(&dH_bf16, size_H * sizeof(__nv_bfloat16)), "malloc dH_bf16");
  cuda_check(cudaMalloc(&dY, size_Y * sizeof(float)), "malloc dY");

  cuda_check(cudaMemcpy(dX, hX, size_X * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy X");
  cuda_check(cudaMemcpy(dW0, hW0, size_W0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
             "copy W0");
  cuda_check(cudaMemcpy(dW1, hW1, size_W1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
             "copy W1");

  dim3 block(32);
  int max_n = (Hdim > Out) ? Hdim : Out;
  dim3 grid((max_n + TN - 1) / TN, (B + TM - 1) / TM);

  launch_mlp_kernel<TM, TN, TK>(grid, block, B, In, Hdim, Out, dX, ldX, dW0, ldW0, dW1, ldW1, dH,
                                ldH, dH_bf16, ldH_bf16, dY, ldY);

  cuda_check(cudaDeviceSynchronize(), "kernel sync");
  cuda_check(cudaMemcpy(hY, dY, size_Y * sizeof(float), cudaMemcpyDeviceToHost), "copy Y");

  printf("Success! Y[0] = %f\n", hY[0]);
  for (int i = 0; i < 10 && i < size_Y; i++) {
    printf("Y[%d] = %f\n", i, hY[i]);
  }

  delete[] hX;
  delete[] hW0;
  delete[] hW1;
  delete[] hY;
  cudaFree(dX);
  cudaFree(dW0);
  cudaFree(dW1);
  cudaFree(dH);
  cudaFree(dH_bf16);
  cudaFree(dY);

  return 0;
}
