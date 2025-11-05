// Trying to use CUTE with a basic DAG kind of pattern
// this seems like a much better direction that previous things i tried
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

template <class T> __device__ __forceinline__ T clamp_to_zero(T x) {
  return x > T(0) ? x : T(0);
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

using OpBF16TN = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using AtomBF16TN = cute::MMA_Atom<OpBF16TN>;

// -------------------------------------------
// C += A * B^T, with
//   A: (TM, TK) bf16
//   B: (TN, TK) bf16
//   C: (TM, TN) f32
// -------------------------------------------
template <int TM, int TN, int TK> struct WarpGemmBF16TN {
  __device__ inline void operator()(__nv_bfloat16* A_ptr, int lda, __nv_bfloat16* B_ptr, int ldb,
                                    float* C_ptr, int ldc) const {
    auto mma = cute::make_tiled_mma(AtomBF16TN{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    // Full views
    auto gA = cute::make_tensor(cute::make_gmem_ptr(A_ptr),
                                cute::make_shape(cute::Int<TM>{}, cute::Int<TK>{}),
                                cute::make_stride(lda, cute::Int<1>{})); // (M,K)

    auto gB = cute::make_tensor(cute::make_gmem_ptr(B_ptr),
                                cute::make_shape(cute::Int<TN>{}, cute::Int<TK>{}),
                                cute::make_stride(ldb, cute::Int<1>{})); // (N,K)

    auto gC = cute::make_tensor(cute::make_gmem_ptr(C_ptr),
                                cute::make_shape(cute::Int<TM>{}, cute::Int<TN>{}),
                                cute::make_stride(ldc, cute::Int<1>{})); // (M,N)

    // Thread fragments
    auto fragA = thr_mma.partition_fragment_A(gA); // (MMA,mma_M,mma_K)
    auto fragB = thr_mma.partition_fragment_B(gB); // (MMA,mma_N,mma_K)
    auto tCgC = thr_mma.partition_C(gC);           // (MMA,mma_M,mma_N)
    auto fragC = thr_mma.make_fragment_C(tCgC);    // (MMA,mma_M,mma_N)

    // Load & compute (C = C + A*B^T)
    // C is used as is, init'd by caller aka 'future me'
    cute::copy(gA, fragA);
    cute::copy(gB, fragB);
    cute::gemm(mma, fragA, fragB, fragC);
    cute::copy(fragC, gC);
  }
};

// -------------------------------------------
// DAG ops
// -------------------------------------------

// Fwd gemm: H = X @ W0^T (bf16xbf16->f32), tile (TMxTN) from (BxH)
template <int TM, int TN, int TK> struct FwdGemm_X_W0T {
  __nv_bfloat16* X;
  int ldX; // [B, In]
  __nv_bfloat16* W0;
  int ldW0; // [H, In]  (row major HxIn)
  float* H;
  int ldH; // [B, H]   (f32 accum)

  __device__ inline void run_tile(int m0, int n0) const {
    // C_tile = A_tile * B_tile^T
    // A_tile: X(m0:m0+TM, 0:TK)
    // B_tile: W0(n0:n0+TN, 0:TK)
    WarpGemmBF16TN<TM, TN, TK>{}(X + m0 * ldX, ldX, W0 + n0 * ldW0, ldW0, H + m0 * ldH + n0, ldH);
  }
};

// ReLU in-place on H (f32)
struct ReLUInplace {
  float* H;
  int ldH;
  int B, Hdim;
  __device__ inline void run_tile(int m0, int n0, int TM, int TN) const {
    // elementwise on (TMxTN) tile
    for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) {
      int r = i / TN;     // [0..TM)
      int c = i - r * TN; // [0..TN)
      float* p = H + (m0 + r) * ldH + (n0 + c);
      float v = *p;
      *p = v > 0.f ? v : 0.f;
    }
  }
};

// Output gemm: Y = H @ W1^T  -> (BxOut), f32 accum
template <int TM, int TN, int TK> struct FwdGemm_H_W1T {
  float* H;
  int ldH; // [B, H], f32
  __nv_bfloat16* W1;
  int ldW1; // [Out, H], bf16
  float* Y;
  int ldY; // [B, Out], f32

  __device__ inline void run_tile(int m0, int n0) const {
    WarpGemmBF16TN<TM, TN, TK>{}(reinterpret_cast<__nv_bfloat16*>(H + m0 * ldH), /*lda*/ ldH,
                                 W1 + n0 * ldW1, /*ldb*/ ldW1, Y + m0 * ldY + n0, /*ldc*/ ldY);
  }
};

// -------------------------------------------
// KERNEL
// -------------------------------------------

template <int TM, int TN, int TK>
__global__ void mlp_kernel(
    // Dims and stuff
    int B, int In, int Hdim, int Out,

    // Data
    __nv_bfloat16* X, int ldX,   // [B, In]
    __nv_bfloat16* W0, int ldW0, // [H, In]
    __nv_bfloat16* W1, int ldW1, // [Out, H]
    __nv_bfloat16* Yb, int ldYb, // [B, Out] (bf16 out)
    float* T, int ldT,           // [B, Out] (target, f32)

    // Scratch / accumulators
    float* H, int ldH,     // [B, H]   (fwd activations, f32)
    float* Y, int ldY,     // [B, Out] (fwd output, f32)
    float* dY, int ldDY,   // [B, Out] (grad, f32)
    float* dH, int ldDH,   // [B, H]   (grad, f32)
    float* dW0, int lddW0, // [H, In]  (grad, f32)
    float* dW1, int lddW1, // [Out, H] (grad, f32)

    // Optim
    float lr) {
  // each block handles one output tile (n0) and one batch tile (m0)
  int warp_idx = threadIdx.x / 32;
  if (warp_idx != 0)
    return; // single-warp per block... stupid? idk, just work first.

  int m0 = blockIdx.y * TM; // batch tile offset
  int n0 = blockIdx.x * TN; // feature/out tile offset

  // Bounds guard (assumes multiples keep anyway)
  if (m0 >= B || n0 >= max(Hdim, Out))
    return;

  // Fwd: H = X W0^T
  {
    // zero H tile
    for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) {
      int r = i / TN;
      int c = i - r * TN;
      if (n0 + c < Hdim && m0 + r < B)
        *(H + (m0 + r) * ldH + (n0 + c)) = 0.f;
    }
    __syncwarp();

    // Sweep K dimension (In) in TK steps
    for (int k0 = 0; k0 < In; k0 += TK) {
      FwdGemm_X_W0T<TM, TN, TK>{X + m0 * ldX + k0, ldX, W0 + n0 * ldW0 + k0, ldW0,
                                H + m0 * ldH + n0, ldH}
          .run_tile(0, 0);
    }

    ReLUInplace{H, ldH, B, Hdim}.run_tile(m0, n0, TM, TN);
  }

  // Fwd: Y = H W1^T
  {
    // Init Y tile
    for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) {
      int r = i / TN;
      int c = i - r * TN;
      if (n0 + c < Out && m0 + r < B)
        *(Y + (m0 + r) * ldY + (n0 + c)) = 0.f;
    }
    __syncwarp();

    for (int k0 = 0; k0 < Hdim; k0 += TK) {
      FwdGemm_H_W1T<TM, TN, TK>{H + m0 * ldH + k0, ldH, W1 + n0 * ldW1 + k0, ldW1,
                                Y + m0 * ldY + n0, ldY}
          .run_tile(0, 0);
    }
  }

  // {
  //   //Y -> bf16
  //   StoreCastBF16{Y, ldY, Yb, ldYb, B, Out}.run_tile(m0, n0, TM, TN);
  // }
}

template <int TM, int TN, int TK>
void launch_mlp_kernel(dim3 grid, dim3 block, int B, int In, int Hdim, int Out,
                                     __nv_bfloat16* X, int ldX, __nv_bfloat16* W0, int ldW0,
                                     __nv_bfloat16* W1, int ldW1, __nv_bfloat16* Yb, int ldYb,
                                     float* T, int ldT, float* H, int ldH, float* Y, int ldY,
                                     float* dY, int ldDY, float* dH, int ldDH, float* dW0,
                                     int lddW0, float* dW1, int lddW1, float lr,
                                     cudaStream_t stream) {
  // compile-time tile constraints
  static_assert(TM == 16 && TN == 8 && TK == 16,
                "This example hard-codes SM80 16x8x16, edit stuff if you want different stuff.");
  // simple grid covering in (N x M) tiles
  mlp_kernel<TM, TN, TK>
      <<<grid, block, 0, stream>>>(B, In, Hdim, Out, X, ldX, W0, ldW0, W1, ldW1, Yb, ldYb, T, ldT,
                                   H, ldH, Y, ldY, dY, ldDY, dH, ldDH, dW0, lddW0, dW1, lddW1, lr);
}

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static void cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    printf("CUDA ERROR at %s: %s\n", msg, cudaGetErrorString(err));
    std::abort();
  }
}

int main() {
  // Must be multiples of tile (16x8x16)
  const int B   = 32;   // batch
  const int In  = 128;  // input dim
  const int Hdim= 256;  // hidden dim
  const int Out = 64;   // output dim

  // Tile sizes fixed by MMA_Atom (SM80_16x8x16 BF16)
  constexpr int TM = 16;
  constexpr int TN = 8;
  constexpr int TK = 16;

  // row-major...
  const int ldX  = In;
  const int ldW0 = In;     // W0[Hdim,In]
  const int ldW1 = Hdim;   // W1[Out,Hdim]
  const int ldYb = Out;
  const int ldT  = Out;
  const int ldH  = Hdim;
  const int ldY  = Out;
  const int ldDY = Out;
  const int ldDH = Hdim;
  const int lddW0= In;
  const int lddW1= Hdim;

  const float lr = 1e-2f;

  // Host allocs
  size_t size_X   = B * In;
  size_t size_W0  = Hdim * In;
  size_t size_W1  = Out * Hdim;
  size_t size_Yb  = B * Out;
  size_t size_T   = B * Out;
  size_t size_H   = B * Hdim;
  size_t size_Y   = B * Out;
  size_t size_dY  = B * Out;
  size_t size_dH  = B * Hdim;
  size_t size_dW0 = Hdim * In;
  size_t size_dW1 = Out * Hdim;

  __nv_bfloat16* hX   = new __nv_bfloat16[size_X];
  __nv_bfloat16* hW0  = new __nv_bfloat16[size_W0];
  __nv_bfloat16* hW1  = new __nv_bfloat16[size_W1];
  float*         hT   = new float[size_T];
  __nv_bfloat16* hYb  = new __nv_bfloat16[size_Yb];

  // Rand init
  for (size_t i = 0; i < size_X;  ++i) hX[i]  = __float2bfloat16(float(rand())/RAND_MAX - 0.5f);
  for (size_t i = 0; i < size_W0; ++i) hW0[i] = __float2bfloat16(float(rand())/RAND_MAX - 0.5f);
  for (size_t i = 0; i < size_W1; ++i) hW1[i] = __float2bfloat16(float(rand())/RAND_MAX - 0.5f);
  for (size_t i = 0; i < size_T;  ++i) hT[i]  = float(rand())/RAND_MAX - 0.5f;

  // Device allocs
  __nv_bfloat16 *dX, *dW0, *dW1, *dYb;
  float *dT, *dH, *dY, *dDY, *dDH, *dW0f, *dW1f;

  cuda_check(cudaMalloc(&dX,   size_X  * sizeof(__nv_bfloat16)), "malloc dX");
  cuda_check(cudaMalloc(&dW0,  size_W0 * sizeof(__nv_bfloat16)), "malloc dW0");
  cuda_check(cudaMalloc(&dW1,  size_W1 * sizeof(__nv_bfloat16)), "malloc dW1");
  cuda_check(cudaMalloc(&dYb,  size_Yb * sizeof(__nv_bfloat16)), "malloc dYb");

  cuda_check(cudaMalloc(&dT,   size_T  * sizeof(float)), "malloc dT");
  cuda_check(cudaMalloc(&dH,   size_H  * sizeof(float)), "malloc dH");
  cuda_check(cudaMalloc(&dY,   size_Y  * sizeof(float)), "malloc dY");
  cuda_check(cudaMalloc(&dDY,  size_dY * sizeof(float)), "malloc dDY");
  cuda_check(cudaMalloc(&dDH,  size_dH * sizeof(float)), "malloc dDH");
  cuda_check(cudaMalloc(&dW0f, size_dW0* sizeof(float)), "malloc dW0f");
  cuda_check(cudaMalloc(&dW1f, size_dW1* sizeof(float)), "malloc dW1f");

  // Memcpy initial data
  cuda_check(cudaMemcpy(dX,  hX,  size_X  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy X");
  cuda_check(cudaMemcpy(dW0, hW0, size_W0 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy W0");
  cuda_check(cudaMemcpy(dW1, hW1, size_W1 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy W1");
  cuda_check(cudaMemcpy(dT,  hT,  size_T  * sizeof(float),          cudaMemcpyHostToDevice), "copy T");

  // ------------------------------------------
  // Kernel launch: tile over (batch x output)
  // grid dims: X direction = output features / TN
  //            Y direction = batch / TM
  // ------------------------------------------
  dim3 block(32);  // 1 warp
  dim3 grid((Out + TN - 1) / TN, (B + TM - 1) / TM);

  launch_mlp_kernel<
      TM, TN, TK>(
      grid, block,
      B, In, Hdim, Out,
      dX, ldX,
      dW0, ldW0,
      dW1, ldW1,
      dYb, ldYb,
      dT, ldT,
      dH, ldH,
      dY, ldY,
      dDY, ldDY,
      dDH, ldDH,
      dW0f, lddW0,
      dW1f, lddW1,
      lr,
      nullptr);

  cuda_check(cudaDeviceSynchronize(), "kernel sync");

  cuda_check(cudaMemcpy(hYb, dYb, size_Yb*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "copy Yb");

  printf("Success. Yb[0] = %f\n", __bfloat162float(hYb[0]));
  for(int i = 0; i < size_Yb; i++) {
    if (i%128 == 0) printf("Yb[%d] = %f\n", i, __bfloat162float(hYb[0]));
  }

  delete[] hX; delete[] hW0; delete[] hW1; delete[] hT; delete[] hYb;
  cudaFree(dX); cudaFree(dW0); cudaFree(dW1); cudaFree(dYb);
  cudaFree(dT); cudaFree(dH); cudaFree(dY); cudaFree(dDY);
  cudaFree(dDH); cudaFree(dW0f); cudaFree(dW1f);

  return 0;
}
