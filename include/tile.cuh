#pragma once

#include <cstdint>

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

// Create immutable tile view from pointer + descripter
template <typename T>
__device__ __host__ inline TileView<T> make_tile_view(const T* data, const Tile2DDesc& desc) {
  return TileView<T>{data, desc.rows, desc.cols, desc.ld};
}

// Create mutable tile view from pointer + descripter
template <typename T>
__device__ __host__ inline MutableTileView<T> make_mutable_tile_view(T* data,
                                                                     const Tile2DDesc& desc) {
  return MutableTileView<T>{data, desc.rows, desc.cols, desc.ld};
}

} // namespace mk
