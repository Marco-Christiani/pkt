#pragma once

#include <cuda_fp16.h>
#include <mma.h>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"
#include "../tile.cuh"

namespace pk {

// Forward as GEMM shaped math
// Y[M, N] = X[M, K] * W[N, K]^T
// M is batch
// N is output size m
// K is input size n
struct LinearForwardArgs {
  const float* W; // [m, n]
  const float* x; // [batch, n]
  float* y;       // [batch, m]
  int batch;
  int m;
  int n;
  int tile_m_start;
  int tile_m_count;
  int tile_n_start;
  int tile_n_count;
};

template <> struct OpTraits<OpCode::LinearForward> {
  using Args = LinearForwardArgs;

  template <bool UseTensorCores, int TileM, int NFrag, int KChunk>
  __device__ static inline void accumulate_kchunk(RTile2D<float, TileM, NFrag>& acc, const Args& args,
                                                  STile2D<const __half, TileM, (Config::kPageBytes / sizeof(__half)) / TileM> x_st,
                                                  int out_col_base, int valid_cols, int k0,
                                                  int k_valid) {
    if constexpr (UseTensorCores) {
      if (k_valid != KChunk) {
        accumulate_kchunk<false, TileM, NFrag, KChunk>(acc, args, x_st, out_col_base, valid_cols, k0,
                                                       k_valid);
        return;
      }
      if (args.tile_n_count != 128) {
        accumulate_kchunk<false, TileM, NFrag, KChunk>(acc, args, x_st, out_col_base, valid_cols, k0,
                                                       k_valid);
        return;
      }
      if (args.tile_n_start + 128 > args.m) {
        accumulate_kchunk<false, TileM, NFrag, KChunk>(acc, args, x_st, out_col_base, valid_cols, k0,
                                                       k_valid);
        return;
      }
      if (valid_cols < NFrag) {
        accumulate_kchunk<false, TileM, NFrag, KChunk>(acc, args, x_st, out_col_base, valid_cols, k0,
                                                       k_valid);
        return;
      }

      constexpr int kWmmaM = 8;
      constexpr int kWmmaN = 32;
      constexpr int kWmmaK = 16;

      const int lane = lane_id();
      const int wid = warp_id();
      const int smem_wid = wid - Config::kFirstComputeWarp;
      const int lane_frag = lane & 7;

      __shared__ __align__(16) __half wmma_a_smem[Config::kNumComputeWarps][kWmmaM * kWmmaK];
      __shared__ __align__(16) __half wmma_b_smem[Config::kNumComputeWarps][kWmmaK * kWmmaN];
      __shared__ __align__(16) float wmma_c_smem[Config::kNumComputeWarps][kWmmaM * kWmmaN];

      __half* a_smem = &wmma_a_smem[smem_wid][0];
      __half* b_smem = &wmma_b_smem[smem_wid][0];
      float* c_smem = &wmma_c_smem[smem_wid][0];

      namespace wmma = nvcuda::wmma;

      const int bn = smem_wid;
      const int lane_col_in_block = lane_frag * NFrag;

      for (int linear = lane; linear < kWmmaM * kWmmaK; linear += Config::kWarpSize) {
        a_smem[linear] = __float2half(0.0f);
      }
      if (lane < kWmmaK) {
        a_smem[0 * kWmmaK + lane] = x_st(0, k0 + lane);
        a_smem[1 * kWmmaK + lane] = x_st(1, k0 + lane);
      }

      __syncwarp();

      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, a_smem, kWmmaK);

      if (bn < (128 / kWmmaN)) {
        constexpr int kKHalf2 = kWmmaK / 2;
        for (int linear2 = lane; linear2 < kWmmaN * kKHalf2; linear2 += Config::kWarpSize) {
          const int col = linear2 / kKHalf2;
          const int row2 = linear2 - col * kKHalf2;
          const int row = row2 * 2;
          const int out_col = args.tile_n_start + bn * kWmmaN + col;

          float2 w = make_float2(0.0f, 0.0f);
          if (out_col < args.m) {
            w = reinterpret_cast<const float2*>(args.W + out_col * args.n + (k0 + row))[0];
          }
          reinterpret_cast<__half2*>(b_smem + col * kWmmaK + row)[0] =
              __floats2half2_rn(w.x, w.y);
        }

        __syncwarp();

        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;

        wmma::load_matrix_sync(b_frag, b_smem, kWmmaK);
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(c_smem, c_frag, kWmmaN, wmma::mem_row_major);

        __syncwarp();

        if (lane < 8) {
#pragma unroll
          for (int c = 0; c < NFrag; ++c) {
            const int col = lane_col_in_block + c;
            acc(0, c) += c_smem[0 * kWmmaN + col];
            acc(1, c) += c_smem[1 * kWmmaN + col];
          }
        }
      }
    } else {
      const __half* x0 = &x_st(0, 0);
      const __half* x1 = &x_st(1, 0);

      if (k_valid == KChunk) {
        const __half* x0_ptr = x0 + k0;
        const __half* x1_ptr = x1 + k0;

#pragma unroll
        for (int kk = 0; kk < KChunk; kk += 4) {
          const __half2 hx0a = reinterpret_cast<const __half2*>(x0_ptr + kk)[0];
          const __half2 hx0b = reinterpret_cast<const __half2*>(x0_ptr + kk)[1];
          const __half2 hx1a = reinterpret_cast<const __half2*>(x1_ptr + kk)[0];
          const __half2 hx1b = reinterpret_cast<const __half2*>(x1_ptr + kk)[1];

          const float2 fx0a = __half22float2(hx0a);
          const float2 fx0b = __half22float2(hx0b);
          const float2 fx1a = __half22float2(hx1a);
          const float2 fx1b = __half22float2(hx1b);

          const float x00 = fx0a.x;
          const float x01 = fx0a.y;
          const float x02 = fx0b.x;
          const float x03 = fx0b.y;
          const float x10 = fx1a.x;
          const float x11 = fx1a.y;
          const float x12 = fx1b.x;
          const float x13 = fx1b.y;

#pragma unroll
          for (int c = 0; c < NFrag; ++c) {
            if (c >= valid_cols) {
              continue;
            }
            const int out_col = out_col_base + c;
            const float* Wrow = args.W + out_col * args.n + k0;
            const float4 wv = reinterpret_cast<const float4*>(Wrow)[kk / 4];

            acc(0, c) += wv.x * x00 + wv.y * x01 + wv.z * x02 + wv.w * x03;
            acc(1, c) += wv.x * x10 + wv.y * x11 + wv.z * x12 + wv.w * x13;
          }
        }
      } else {
        for (int kk = 0; kk < k_valid; ++kk) {
          const float x0f = __half2float(x_st(0, k0 + kk));
          const float x1f = __half2float(x_st(1, k0 + kk));
#pragma unroll
          for (int c = 0; c < NFrag; ++c) {
            if (c >= valid_cols) {
              continue;
            }
            const int out_col = out_col_base + c;
            const float w = args.W[out_col * args.n + (k0 + kk)];
            acc(0, c) += w * x0f;
            acc(1, c) += w * x1f;
          }
        }
      }
    }
  }

  __device__ static void load(const Args& args, BlockRuntime& br, int slot_idx, int lane) {
    constexpr int kTileM = 2;
    constexpr int kKMax = (Config::kPageBytes / sizeof(__half)) / kTileM;

    (void)kTileM;
#if !defined(NDEBUG)
    if (lane == 0) {
      if (args.tile_m_count != kTileM) {
        printf("LinearForward expects tile_m_count equal to %d\n", kTileM);
        asm volatile("trap;");
      }
      if (args.n > kKMax) {
        printf("LinearForward expects n at most %d\n", kKMax);
        asm volatile("trap;");
      }
    }
#endif

    PageHandle x_page = br.page_handle(slot_idx, 0);
    const unsigned long long tag = make_page_tag(args.x, args.tile_m_start, args.tile_m_count,
                                                 br.slot(slot_idx).task.header.read_epoch);
    if (br.page_is_ready_with_tag(x_page, tag)) {
      return;
    }

    br.page_wait_begin_overwrite(x_page);

    __half* smem_x = br.page_ptr<__half>(x_page);

    for (int linear = lane; linear < kTileM * kKMax; linear += Config::kWarpSize) {
      smem_x[linear] = __float2half(0.0f);
    }

    for (int b_local = 0; b_local < kTileM; ++b_local) {
      int b = args.tile_m_start + b_local;
      if (b >= args.batch || b_local >= args.tile_m_count) {
        continue;
      }
      const float* x_row = args.x + b * args.n;
      for (int k = lane; k < args.n && k < kKMax; k += Config::kWarpSize) {
        smem_x[b_local * kKMax + k] = __float2half(x_row[k]);
      }
    }

    br.page_publish(x_page, tag);
  }

  __device__ static void compute(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                                 int compute_warp_idx, int num_compute_warps) {
    constexpr int kTileM = 2;
    constexpr int kKMax = (Config::kPageBytes / sizeof(__half)) / kTileM;
    constexpr int kKChunk = Config::kGemmKChunk;
    constexpr int kNFrag = Config::kLinearFwdThreadNFrag;

    const __half* smem_x = br.page_ptr<__half>(br.page_handle(slot_idx, 0));
    STile2D<const __half, kTileM, kKMax> x_st{smem_x};

    if constexpr (Config::kUseTensorCores) {
      constexpr int kColsPerWarp = 32;
      constexpr int kFragsPerWarp = kColsPerWarp / kNFrag;

      const int slice_start = compute_warp_idx * kColsPerWarp;
      if (slice_start >= args.tile_n_count) {
        return;
      }

      const int lane_frag = lane & (kFragsPerWarp - 1);
      const int n_local_base = slice_start + lane_frag * kNFrag;
      const int out_col_base = args.tile_n_start + n_local_base;
      if (out_col_base >= args.m) {
        return;
      }

      const int max_cols_from_tile = args.tile_n_count - n_local_base;
      const int max_cols_from_tensor = args.m - out_col_base;
      const int valid_cols = (max_cols_from_tile < max_cols_from_tensor) ? max_cols_from_tile
                                                                         : max_cols_from_tensor;
      if (valid_cols <= 0) {
        return;
      }

      RTile2D<float, kTileM, kNFrag> acc;
      acc.clear(0.0f);

      const int n = args.n;
      for (int k0 = 0; k0 < n; k0 += kKChunk) {
        const int k_rem = n - k0;
        const int k_valid = (k_rem < kKChunk) ? k_rem : kKChunk;

        accumulate_kchunk<Config::kUseTensorCores, kTileM, kNFrag, kKChunk>(
            acc, args, x_st, out_col_base, valid_cols, k0, k_valid);
      }

      const int b0 = args.tile_m_start + 0;
      const int valid_rows = (args.tile_m_count < kTileM) ? args.tile_m_count : kTileM;

      if (lane < kFragsPerWarp && b0 < args.batch && valid_rows > 0) {
        store_rtile_fragment<float, kTileM, kNFrag>(acc, args.y, args.m, b0, out_col_base,
                                                    valid_rows, valid_cols);
      }
    } else {
      int tid = compute_warp_idx * 32 + lane;
      int total_threads = num_compute_warps * 32;

      for (int n_local_base = tid * kNFrag; n_local_base < args.tile_n_count;
           n_local_base += total_threads * kNFrag) {
        const int out_col_base = args.tile_n_start + n_local_base;
        if (out_col_base >= args.m) {
          continue;
        }

        const int max_cols_from_tile = args.tile_n_count - n_local_base;
        const int max_cols_from_tensor = args.m - out_col_base;
        const int valid_cols = (max_cols_from_tile < max_cols_from_tensor) ? max_cols_from_tile
                                                                           : max_cols_from_tensor;
        if (valid_cols <= 0) {
          continue;
        }

        RTile2D<float, kTileM, kNFrag> acc;
        acc.clear(0.0f);

        const int n = args.n;
        for (int k0 = 0; k0 < n; k0 += kKChunk) {
          const int k_rem = n - k0;
          const int k_valid = (k_rem < kKChunk) ? k_rem : kKChunk;

          accumulate_kchunk<Config::kUseTensorCores, kTileM, kNFrag, kKChunk>(
              acc, args, x_st, out_col_base, valid_cols, k0, k_valid);
        }

        const int b0 = args.tile_m_start + 0;
        const int valid_rows = (args.tile_m_count < kTileM) ? args.tile_m_count : kTileM;

        if (b0 < args.batch && valid_rows > 0) {
          store_rtile_fragment<float, kTileM, kNFrag>(acc, args.y, args.m, b0, out_col_base,
                                                      valid_rows, valid_cols);
        }
      }
    }
  }

  __device__ static void store(const Args& args, BlockRuntime& br, int slot_idx, int lane) {
    (void)args;
    (void)br;
    (void)slot_idx;
    (void)lane;
  }
};

} // namespace pk
