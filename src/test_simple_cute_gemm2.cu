#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using OpBF16TN = cute::SM80_16x8x16_F32BF16BF16F32_TN; // (M,N,K) = (16,8,16)
using AtomBF16TN = cute::MMA_Atom<OpBF16TN>;

// -------------------------------------------
// C (f32) += A (bf16) @ B^T (bf16), TN path
// TM, TN must be multiples of (16,8) respectively, TK multiple of 16
// -------------------------------------------
template <int TM, int TN, int TK> struct TiledGemmBF16TN {
  __nv_bfloat16* A_base;
  int ldA; // A: [M,K], row-major, ldA=K
  __nv_bfloat16* B_base;
  int ldB; // B: [N,K], row-major, ldB=K (we use B^T)
  float* C_base;
  int ldC;    // C: [M,N], row-major, ldC=N
  int K_full; // total K

  __device__ inline void run_tile(int m0, int n0, int M_actual, int N_actual) const {
    // one warp kernel, blockDim.x == 32
    auto mma = cute::make_tiled_mma(AtomBF16TN{}); // 16x8x16 per warp
    auto thr = mma.get_slice(threadIdx.x);

    // actual tile dims for boundaries
    int m_tile = cute::min(TM, M_actual - m0);
    int n_tile = cute::min(TN, N_actual - n0);

    // mma smem tiles and register frags have static shape
    auto gC_static = cute::make_tensor(cute::make_gmem_ptr(C_base + m0 * ldC + n0),
                                       cute::make_shape(cute::Int<TM>{}, cute::Int<TN>{}),
                                       cute::make_stride(ldC, cute::Int<1>{}));
    auto tCgC = thr.partition_C(gC_static);
    auto fragC = thr.make_fragment_C(tCgC);
    cute::clear(fragC);

    // smem staging for A and B (req for ldmatrix?)
    extern __shared__ __nv_bfloat16 smem[]; // dynamic smem
    __nv_bfloat16* sA = smem;               // TM x TK
    __nv_bfloat16* sB = sA + TM * TK;       // TN x TK

    // smem tiles
    auto sA_t = cute::make_tensor(cute::make_smem_ptr(sA),
                                  cute::make_shape(cute::Int<TM>{}, cute::Int<TK>{}),
                                  cute::make_stride(cute::Int<TK>{}, cute::Int<1>{}));
    auto sB_t = cute::make_tensor(cute::make_smem_ptr(sB),
                                  cute::make_shape(cute::Int<TN>{}, cute::Int<TK>{}),
                                  cute::make_stride(cute::Int<TK>{}, cute::Int<1>{}));

    // thread views on smem
    auto tAsA = thr.partition_A(sA_t);
    auto tBsB = thr.partition_B(sB_t);

    auto fragA = thr.make_fragment_A(tAsA);

    // register frags
    auto fragB = thr.make_fragment_B(tBsB);

    // K loop: GMEM -> SMEM, then SMEM -> REG, then MMA
    for (int k0 = 0; k0 < K_full; k0 += TK) {
      int k_tile = cute::min(TK, K_full - k0);

      // Clear smem tiles for zero-padding partial tiles
      for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) {
        sA[i] = __float2bfloat16(0.0f);
      }
      for (int i = threadIdx.x; i < TN * TK; i += blockDim.x) {
        sB[i] = __float2bfloat16(0.0f);
      }
      __syncthreads();

      // cooperative copy w boundary predicates
      for (int tid = threadIdx.x; tid < m_tile * k_tile; tid += blockDim.x) {
        int m = tid / k_tile;
        int k = tid % k_tile;
        sA[m * TK + k] = A_base[(m0 + m) * ldA + (k0 + k)];
      }

      for (int tid = threadIdx.x; tid < n_tile * k_tile; tid += blockDim.x) {
        int n = tid / k_tile;
        int k = tid % k_tile;
        sB[n * TK + k] = B_base[(n0 + n) * ldB + (k0 + k)];
      }

      __syncthreads(); // wait for smem tiles to be ready

      // shared -> reg fragments (ldmatrix path)
      cute::copy(tAsA, fragA);
      cute::copy(tBsB, fragB);

      // tensor core mma
      cute::gemm(mma, fragA, fragB, fragC);

      __syncthreads(); // free smem for next K-slice
    }

    // Store accumulator to C
    // use cute's predicated store by creating runtime-shaped target
    // copy fragc to smem then cooperatively store to gmem with bounds

    // or... directly store with per-thread boundary checks
    // tCgC partition tells us which elements each thread owns
    // we store only if within bounds

    // simple: use cooperative store w/ smem staging
    float* sC = reinterpret_cast<float*>(smem); // reuse smem for C staging
    auto sC_t = cute::make_tensor(cute::make_smem_ptr(sC),
                                  cute::make_shape(cute::Int<TM>{}, cute::Int<TN>{}),
                                  cute::make_stride(cute::Int<TN>{}, cute::Int<1>{}));
    auto tCsC = thr.partition_C(sC_t);

    // reg -> smem
    cute::copy(fragC, tCsC);
    __syncthreads();

    // cooperative smem -> gmem w/ boundary predicates
    for (int tid = threadIdx.x; tid < m_tile * n_tile; tid += blockDim.x) {
      int m = tid / n_tile;
      int n = tid % n_tile;
      C_base[(m0 + m) * ldC + (n0 + n)] = sC[m * TN + n];
    }
  }
};

// one warp computes one (TM x TN) tile
template <int TM, int TN, int TK>
__global__ void gemm_bf16_tn_kernel(int M, int N, int K, const __nv_bfloat16* __restrict__ A,
                                    int ldA, const __nv_bfloat16* __restrict__ B, int ldB,
                                    float* __restrict__ C, int ldC) {
  static_assert(TM % 16 == 0 && TN % 8 == 0 && TK % 16 == 0, "Tile multiples");
  const int m0 = blockIdx.y * TM;
  const int n0 = blockIdx.x * TN;
  if (m0 >= M || n0 >= N)
    return;

  TiledGemmBF16TN<TM, TN, TK> op{const_cast<__nv_bfloat16*>(A), ldA, const_cast<__nv_bfloat16*>(B), ldB, C, ldC, K};
  op.run_tile(m0, n0, M, N);
}

int main() {
    // try various arbitrary sizes that are not multiples of tile dimensions
    // M=50: not multiple of TM=16 (needs 4 blocks, last has 2 rows)
    // N=20: not multiple of TN=8 (needs 3 blocks, last has 4 cols)
    // K=33: not multiple of TK=16 (needs 3 K-iterations, last has 1 element)
    const int M = 50, N = 20, K = 33;

    __nv_bfloat16 *hA = new __nv_bfloat16[M*K];
    __nv_bfloat16 *hB = new __nv_bfloat16[N*K];
    float *hC = new float[M*N];

    for (int i = 0; i < M*K; i++) hA[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < N*K; i++) hB[i] = __float2bfloat16(1.0f);

    __nv_bfloat16 *dA, *dB;
    float *dC;

    cudaMalloc(&dA, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&dB, N*K*sizeof(__nv_bfloat16));
    cudaMalloc(&dC, M*N*sizeof(float));

    cudaMemcpy(dA, hA, M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(float));

    printf("Launching GEMM with arbitrary sizes: M=%d, N=%d, K=%d\n", M, N, K);

    // 2D grid: each block handles one (TM x TN) tile w/ one warp
    constexpr int TM = 16, TN = 8, TK = 16;
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    constexpr size_t smem_size = (TM*TK + TN*TK) * sizeof(__nv_bfloat16); // 768 bytes

    printf("Grid: (%d, %d) blocks, each with 32 threads (1 warp)\n", grid.x, grid.y);
    gemm_bf16_tn_kernel<TM, TN, TK><<<grid, 32, smem_size>>>(M, N, K, dA, K, dB, K, dC, N);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nResult (should be %.1f for all elements):\n", (float)K);

    bool correct = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (float)K;
            float actual = hC[i*N + j];
            if (fabs(actual - expected) > 0.1f) {
                printf("ERROR at C[%d,%d]: expected %.1f, got %.1f\n", i, j, expected, actual);
                correct = false;
            }
        }
    }

    if (correct) {
        printf("All elements correct.\n");
    }

    // look at corners and boundaries
    printf("\nSample output:\n");
    printf("Top-left corner:\n");
    for (int i = 0; i < cute::min(4, M); i++) {
        for (int j = 0; j < cute::min(4, N); j++) {
            printf("%.1f ", hC[i*N + j]);
        }
        printf("\n");
    }
    printf("\nBottom-right corner:\n");
    for (int i = cute::max(0, M-4); i < M; i++) {
        for (int j = cute::max(0, N-4); j < N; j++) {
            printf("%.1f ", hC[i*N + j]);
        }
        printf("\n");
    }

    delete[] hA; delete[] hB; delete[] hC;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

