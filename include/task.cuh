#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "config.cuh"

namespace pk {

enum class OpCode : std::uint16_t {
  Invalid = 0,
  ZeroMemory = 1,
  Axpy = 2,
  Gemm = 3,
  LinearForward = 10,
  MSELoss = 11,
  LinearBackward = 12,
  SGDUpdate = 13,
};

struct TaskHeader {
  OpCode opcode{OpCode::Invalid}; // 2 bytes, offset 0
  std::uint16_t arg_bytes{0};     // 2 bytes, offset 2
  std::uint32_t user_tag{0};      // 4 bytes, offset 4 (aligned)
  // Dependency metadata
  std::uint16_t buffer_read_id{0};
  std::uint16_t buffer_write_id{0};
  std::uint32_t wait_count{0};
  std::uint16_t read_epoch{0};
  std::uint16_t write_epoch{0};
  // Total: 20 bytes, so args[] starts at 20-byte boundary
};

// Task structure = header + opaque argument payload
struct alignas(16) Task {
  TaskHeader header;
  std::uint8_t args[Config::kArgBytes];
};

// Zero-copy argument decoder
template <typename Args> __host__ __device__ inline const Args& decode_args(const Task& t) {
  static_assert(std::is_trivially_copyable_v<Args>, "Args must be trivially copyable");
  static_assert(sizeof(Args) <= Config::kArgBytes, "Args size exceeds task payload capacity");
  return *reinterpret_cast<const Args*>(t.args);
}

// Argument encoder for host-side task creation
template <typename Args> inline void encode_args(Task& t, OpCode opcode, const Args& args) {
  static_assert(std::is_trivially_copyable_v<Args>, "Args must be trivially copyable");
  static_assert(sizeof(Args) <= Config::kArgBytes, "Args size exceeds task payload capacity");

  t.header.opcode = opcode;
  t.header.arg_bytes = static_cast<std::uint16_t>(sizeof(Args));
  t.header.buffer_read_id = 0;
  t.header.buffer_write_id = 0;
  t.header.wait_count = 0;
  t.header.read_epoch = 0;
  t.header.write_epoch = 0;

  std::memset(t.args, 0, Config::kArgBytes);
  std::memcpy(t.args, &args, sizeof(Args)); // copy args into payload
}

} // namespace pk
