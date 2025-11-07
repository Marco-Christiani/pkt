// ============================================================================
// MLP: X -> W0 -> ReLU -> W1 -> Y
// bf16 weights, fp32 accum, persistent warp specialization
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cute/arch/mma_sm80.hpp>
#include <cute/tensor.hpp>
#include <iostream>
#include <random>
#include <vector>

using namespace cute;

// bf16 helpers
__host__ __device__ inline __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}
__host__ __device__ inline float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// ============================================================================
// Tensor wrapper
// ============================================================================
template <typename T> struct Tensor {
  T* data;
  int M, N;
  __device__ inline int stride() const { return N; } // row-major
};

// ============================================================================
// Op: MatMul tile using CUTE MMA (bf16 x bf16 -> fp32)
// ============================================================================
template <int WarpId, int TM, int TN, int TK> struct MatMulOp {
  Tensor<__nv_bfloat16> A; // [M,K]
  Tensor<__nv_bfloat16> B; // [N,K]
  Tensor<float> C;         // [M,N] accumulation

  using Atom = MMA_Atom<SM80_F16BF16F32M16N8K16_TN>; // 16x8x16

  __device__ void run(int wid, int batch_row) {
    if (wid != WarpId)
      return;

    int lane = threadIdx.x % 32;

    // Tile views into fragments
    Tensor tA = {A.data + batch_row * A.stride(), TK, TM};
    Tensor tB = {B.data, TK, TN};
    Tensor tC = {C.data + batch_row * C.stride(), TM, TN};

    auto sA = make_tensor(tA.data, make_shape(Int<TM>{}, Int<TK>{}), LayoutRight{});
    auto sB = make_tensor(tB.data, make_shape(Int<TN>{}, Int<TK>{}), LayoutRight{});
    auto sC = make_tensor(tC.data, make_shape(Int<TM>{}, Int<TN>{}), LayoutRight{});

    Tensor fragA = make_fragment_like<Atom::A>(sA);
    Tensor fragB = make_fragment_like<Atom::B>(sB);
    Tensor fragC = make_fragment_like<Atom::C>(sC);

    copy(sA, fragA);
    copy(sB, fragB);
    Atom{}(fragC, fragA, fragB, fragC);
    copy(fragC, sC);
  }
};

// ============================================================================
// Op: ReLU activation (fp32)
// ============================================================================
template <int WarpId> struct ReLUOp {
  Tensor<float> H;
  __device__ void run(int wid, int batch_row) {
    if (wid != WarpId)
      return;
    int idx = threadIdx.x;
    if (idx < H.M * H.N)
      H.data[idx] = fmaxf(H.data[idx], 0.f);
  }
};

// ============================================================================
// Op: Store result (convert fp32 -> bf16)
// ============================================================================
template <int WarpId> struct StoreOp {
  Tensor<float> src;
  Tensor<__nv_bfloat16> dst;
  __device__ void run(int wid, int batch_row) {
    if (wid != WarpId)
      return;
    int idx = threadIdx.x;
    if (idx < src.M * src.N)
      dst.data[idx] = __float2bfloat16(src.data[idx]);
  }
};

// ============================================================================
// Compile-time DAG (tuple<Ops...>)
// ============================================================================
template <typename... Ops> struct DAG {
  std::tuple<Ops...> ops;
  __device__ void run_all(int wid, int batch_row) {
    std::apply([&](auto&... op) { (op.run(wid, batch_row), ...); }, ops);
  }
};

// ============================================================================
// Persistent megakernel
// ============================================================================
template <typename DAG> __global__ void mlp_persistent(DAG dag, int batch_size, int tile) {
  int wid = threadIdx.x / 32; // which warp
  for (int row = blockIdx.x * tile; row < batch_size; row += gridDim.x * tile) {
    dag.run_all(wid, row);
    __syncthreads();
  }
}

// ============================================================================
// Host side test harness
// ============================================================================
int main() {
  const int B = 32; // batch size (one tile)
  const int In = 128;
  const int H = 256;
  const int Out = 64;

  // Host buffers
  std::vector<float> hX(B * In), hW0(H * In), hW1(Out * H);
  std::vector<float> hH(B * H), hY(B * Out);

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  for (auto& v : hX)
    v = dist(rng);
  for (auto& v : hW0)
    v = dist(rng);
  for (auto& v : hW1)
    v = dist(rng);

  // CPU reference
  for (int m = 0; m < B; ++m) {
    for (int h = 0; h < H; ++h) {
      float acc = 0;
      for (int i = 0; i < In; ++i)
        acc += hX[m * In + i] * hW0[h * In + i];
      hH[m * H + h] = fmaxf(acc, 0); // ReLU
    }
  }
  for (int m = 0; m < B; ++m)
    for (int o = 0; o < Out; ++o) {
      float acc = 0;
      for (int h = 0; h < H; ++h)
        acc += hH[m * H + h] * hW1[o * H + h];
      hY[m * Out + o] = acc;
    }

  // GPU memory
  __nv_bfloat16 *dX, *dW0, *dW1, *dY;
  float *dH0, *dYf;
  cudaMalloc(&dX, B * In * sizeof(__nv_bfloat16));
  cudaMalloc(&dW0, H * In * sizeof(__nv_bfloat16));
  cudaMalloc(&dW1, Out * H * sizeof(__nv_bfloat16));
  cudaMalloc(&dH0, B * H * sizeof(float));
  cudaMalloc(&dYf, B * Out * sizeof(float));
  cudaMalloc(&dY, B * Out * sizeof(__nv_bfloat16));

  std::vector<__nv_bfloat16> Xbf(B * In), W0bf(H * In), W1bf(Out * H);
  for (int i = 0; i < B * In; i++)
    Xbf[i] = f32_to_bf16(hX[i]);
  for (int i = 0; i < H * In; i++)
    W0bf[i] = f32_to_bf16(hW0[i]);
  for (int i = 0; i < Out * H; i++)
    W1bf[i] = f32_to_bf16(hW1[i]);

  cudaMemcpy(dX, Xbf.data(), B * In * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dW0, W0bf.data(), H * In * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(dW1, W1bf.data(), Out * H * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

  // Build DAG at compile time
  using DAG_t = DAG<MatMulOp<0, B, H, In>,  // warp0: X*W0^T
                    ReLUOp<1>,              // warp1: activation
                    MatMulOp<2, B, Out, H>, // warp2: H*W1^T
                    StoreOp<3>              // warp3: fp32 -> bf16 copy
                    >;

  DAG_t dag = {MatMulOp<0, B, H, In>{Tensor<__nv_bfloat16>{dX, B, In},
                                     Tensor<__nv_bfloat16>{dW0, H, In}, Tensor<float>{dH0, B, H}},
               ReLUOp<1>{Tensor<float>{dH0, B, H}},
               MatMulOp<2, B, Out, H>{Tensor<__nv_bfloat16>{dH0, B, H},
                                      Tensor<__nv_bfloat16>{dW1, Out, H},
                                      Tensor<float>{dYf, B, Out}},
               StoreOp<3>{Tensor<float>{dYf, B, Out}, Tensor<__nv_bfloat16>{dY, B, Out}}};

  mlp_persistent<<<1, 128>>>(dag, B, B);
  cudaDeviceSynchronize();

  // Validate
  std::vector<__nv_bfloat16> Ybf(B * Out);
  cudaMemcpy(Ybf.data(), dY, B * Out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  float max_err = 0;
  for (int i = 0; i < B * Out; ++i)
    max_err = fmaxf(max_err, fabsf(bf16_to_f32(Ybf[i]) - hY[i]));

  std::cout << "Max absolute error = " << max_err << "\n";
  assert(max_err < 0.5f);
  std::cout << "✅ PASSED\n";
}
