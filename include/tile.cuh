#pragma once

#include <cstdint>

#include "block_runtime.cuh"

namespace pk {

struct Tile2DDesc {
  int rows{0};
  int cols{0};
  int ld{0};
};

template <typename T> struct TileView {
  const T* __restrict__ data{nullptr};
  int rows{0};
  int cols{0};
  int stride{0};

  __device__ inline const T& operator()(int r, int c) const { return data[r * stride + c]; }

  __device__ inline const T& load(int r, int c) const { return data[r * stride + c]; }
};

// Mutable view for output tiles
template <typename T> struct MutableTileView {
  T* __restrict__ data{nullptr};
  int rows{0};
  int cols{0};
  int stride{0};

  __device__ inline T& operator()(int r, int c) const { return data[r * stride + c]; }

  __device__ inline T& ref(int r, int c) const { return data[r * stride + c]; }

  __device__ inline void store(int r, int c, const T& value) const { data[r * stride + c] = value; }
};

// Create immutable tile view from pointer + descriptor
template <typename T>
__device__ __host__ inline TileView<T> make_tile_view(const T* data, const Tile2DDesc& desc) {
  return TileView<T>{data, desc.rows, desc.cols, desc.ld};
}

// Create mutable tile view from pointer + descriptor
template <typename T>
__device__ __host__ inline MutableTileView<T> make_mutable_tile_view(T* data,
                                                                     const Tile2DDesc& desc) {
  return MutableTileView<T>{data, desc.rows, desc.cols, desc.ld};
}

template <typename T, int Rows, int Cols> struct STile2D {
  // Shared memory tile view
  // Data lives in SMEM and is shared across the block
  // Layout is row major with contiguous columns
  static constexpr int kRank = 2;
  static constexpr int kRows = Rows;
  static constexpr int kCols = Cols;

  T* __restrict__ data{nullptr};

  __device__ inline T& operator()(int r, int c) { return data[r * Cols + c]; }
  __device__ inline const T& operator()(int r, int c) const { return data[r * Cols + c]; }
};

template <typename T, int Rows, int Cols> struct RTile2D {
  // Register tile fragment
  // Data lives in registers and is owned by the calling thread
  // Higher level code assigns fragments to warps or subsets of lanes
  static constexpr int kRank = 2;
  static constexpr int kRows = Rows;
  static constexpr int kCols = Cols;

  T v[Rows * Cols];

  __device__ inline void clear(T value = T(0)) {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      v[i] = value;
    }
  }

  __device__ inline T& operator()(int r, int c) { return v[r * Cols + c]; }
  __device__ inline const T& operator()(int r, int c) const { return v[r * Cols + c]; }
};

template <typename T, int Rows, int Cols>
__device__ inline STile2D<T, Rows, Cols> make_stile_from_page(BlockRuntime& br, PageHandle h) {
  return STile2D<T, Rows, Cols>{br.page_ptr<T>(h)};
}

template <typename T, int Rows, int Cols>
__device__ inline void store_rtile_fragment(const RTile2D<T, Rows, Cols>& tile, T* dst, int ld,
                                            int row0, int col0, int valid_rows,
                                            int valid_cols) {
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    if (r >= valid_rows) {
      continue;
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      if (c >= valid_cols) {
        continue;
      }
      dst[(row0 + r) * ld + (col0 + c)] = tile(r, c);
    }
  }
}

} // namespace pk
