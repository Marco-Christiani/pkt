#pragma once

#include <cuda_fp16.h>

#include "../block_runtime.cuh"
#include "../config.cuh"
#include "../op_traits.cuh"
#include "../tile.cuh"

namespace pk {

// Computes weight gradient for a row range: dW = sum_b dy_b @ x_b^T.
// dy: [batch, m]
// x: [batch, n]
// dW: [m, n] row-major (accumulated)
struct LinearBackwardArgs {
  const float* dy;
  const float* x;
  float* dW;
  int batch;
  int m;
  int n;
  int tile_m_start;
  int tile_m_count;
  int tile_n_start;
  int tile_n_count;
};

template <> struct OpTraits<OpCode::LinearBackward> {
  using Args = LinearBackwardArgs;

  __device__ static void load(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                              unsigned long long* page_reuse_hits,
                              unsigned long long* page_refill_count) {
    constexpr int kRowsPerPage = 32;
    constexpr int kBatchMax = (Config::kPageBytes / sizeof(__half)) / kRowsPerPage;
    constexpr int kMaxTileM = 2 * kRowsPerPage;
#if !defined(NDEBUG)
    if (lane == 0) {
      if (args.batch > kBatchMax) {
        printf("LinearBackward expects batch at most %d\n", kBatchMax);
        asm volatile("trap;");
      }
      if (args.tile_m_count > kMaxTileM) {
        printf("LinearBackward expects tile_m_count at most %d\n", kMaxTileM);
        asm volatile("trap;");
      }
    }
#endif

    PageHandle dy_page0 = br.page_handle(slot_idx, 0);
    PageHandle dy_page1 = br.page_handle(slot_idx, 1);
    const int epoch = br.slot(slot_idx).task.header.read_epoch;
    const unsigned long long tag = make_page_tag(args.dy, args.tile_m_start, args.tile_m_count, epoch);

    if (br.page_is_ready_with_tag(dy_page0, tag) && br.page_is_ready_with_tag(dy_page1, tag)) {
      if (lane == 0 && page_reuse_hits) {
        *page_reuse_hits += 1;
      }
      return;
    }
    if (lane == 0 && page_refill_count) {
      *page_refill_count += 2;
    }

    br.page_wait_begin_overwrite(dy_page0);
    br.page_wait_begin_overwrite(dy_page1);

    auto dy0 = make_stile_from_page<__half, kRowsPerPage, kBatchMax>(br, dy_page0);
    auto dy1 = make_stile_from_page<__half, kRowsPerPage, kBatchMax>(br, dy_page1);

    const int rows = args.tile_m_count;
    const int batch = args.batch;

    for (int linear = lane; linear < rows * batch; linear += Config::kWarpSize) {
      int r = linear / batch;
      int b = linear - r * batch;
      int row = args.tile_m_start + r;
      __half val = __float2half(0.0f);
      if (row < args.m) {
        val = __float2half(args.dy[b * args.m + row]);
      }
      if (r < kRowsPerPage) {
        dy0(r, b) = val;
      } else {
        dy1(r - kRowsPerPage, b) = val;
      }
    }

    br.page_publish(dy_page0, tag);
    br.page_publish(dy_page1, tag);
  }

  __device__ static void compute(const Args& args, BlockRuntime& br, int slot_idx, int lane,
                                 int compute_warp_idx, int num_compute_warps) {
    constexpr int kRowsPerPage = 32;
    constexpr int kBatchMax = (Config::kPageBytes / sizeof(__half)) / kRowsPerPage;
    constexpr int kKChunk = Config::kGemmKChunk;
    constexpr int kNFrag = Config::kLinearBwdThreadNFrag;

    PageHandle dy_page0 = br.page_handle(slot_idx, 0);
    PageHandle dy_page1 = br.page_handle(slot_idx, 1);
    STile2D<const __half, kRowsPerPage, kBatchMax> dy0{br.page_ptr<const __half>(dy_page0)};
    STile2D<const __half, kRowsPerPage, kBatchMax> dy1{br.page_ptr<const __half>(dy_page1)};

    int tid = compute_warp_idx * 32 + lane;
    int total_threads = num_compute_warps * 32;

    const int m_tile = args.tile_m_count;
    const int n_tile = args.tile_n_count;
    const int n_frag_tiles = (n_tile + kNFrag - 1) / kNFrag;
    const int tile_elems = m_tile * n_frag_tiles;

    for (int idx = tid; idx < tile_elems; idx += total_threads) {
      int r = idx / n_frag_tiles;
      int c_tile = idx - r * n_frag_tiles;
      int row = args.tile_m_start + r;
      int col_base = args.tile_n_start + c_tile * kNFrag;

      if (row >= args.m || col_base >= args.n) {
        continue;
      }

      const int max_cols_from_tile = n_tile - c_tile * kNFrag;
      const int max_cols_from_tensor = args.n - col_base;
      const int valid_cols = (max_cols_from_tile < max_cols_from_tensor) ? max_cols_from_tile
                                                                         : max_cols_from_tensor;
      if (valid_cols <= 0) {
        continue;
      }

      RTile2D<float, 1, kNFrag> grad;
      grad.clear(0.0f);

      for (int b0 = 0; b0 < args.batch; b0 += kKChunk) {
        const int b_rem = args.batch - b0;
        const int b_valid = (b_rem < kKChunk) ? b_rem : kKChunk;
        const int dy_r = (r < kRowsPerPage) ? r : (r - kRowsPerPage);

        for (int kk = 0; kk < b_valid; ++kk) {
          const __half dyh = (r < kRowsPerPage) ? dy0(dy_r, b0 + kk) : dy1(dy_r, b0 + kk);
          const float dyf = __half2float(dyh);
          const float* x_row = args.x + (b0 + kk) * args.n + col_base;

#pragma unroll
          for (int c = 0; c < kNFrag; ++c) {
            if (c >= valid_cols) {
              continue;
            }
            grad(0, c) += dyf * x_row[c];
          }
        }
      }

      store_rtile_fragment<float, 1, kNFrag>(grad, args.dW, args.n, row, col_base, 1, valid_cols);
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
