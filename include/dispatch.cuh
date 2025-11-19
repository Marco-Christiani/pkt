#pragma once

#include "block_runtime.cuh"
#include "op_traits.cuh"
#include "task.cuh"

namespace pk {

// Maps opcodes to OpTraits implementations
// The thinking here is to use a separate dispatch functions (per pipeline phase)
//  for better instruction cache utilization compared to a single switch.
// For example with 20+ operations:
//   Single dispatch: 1 function * 300 bytes = I-cache thrashing
//   Three dispatches: 3 functions * 100 bytes = better locality

// Load phase dispatch
// Called by loader warp to prefetch data
__device__ inline void dispatch_load(const Task& task, BlockRuntime& br, int slot_idx, int lane) {
  switch (task.header.opcode) {
    case OpCode::Axpy: {
      const auto& args = decode_args<OpTraits<OpCode::Axpy>::Args>(task);
      OpTraits<OpCode::Axpy>::load(args, br, slot_idx, lane);
      break;
    }

    case OpCode::Gemm: {
      // const auto& args = decode_args<OpTraits<OpCode::Gemm>::Args>(task);
      // OpTraits<OpCode::Gemm>::load(args, br, slot_idx, lane);
      break;
    }

    default:
      // Unknown opcode - trap in debug builds?
      break;
  }
}

// Compute phase dispatch
// Called by compute warps to process data
__device__ inline void dispatch_compute(const Task& task, BlockRuntime& br, int slot_idx, int lane,
                                        int compute_warp_idx, int num_compute_warps) {
  switch (task.header.opcode) {
    case OpCode::Axpy: {
      const auto& args = decode_args<OpTraits<OpCode::Axpy>::Args>(task);
      OpTraits<OpCode::Axpy>::compute(args, br, slot_idx, lane, compute_warp_idx, num_compute_warps);
      break;
    }

    case OpCode::Gemm: {
      // const auto& args = decode_args<OpTraits<OpCode::Gemm>::Args>(task);
      // OpTraits<OpCode::Gemm>::compute(args, br, slot_idx, lane,
      //                                 compute_warp_idx, num_compute_warps);
      break;
    }

    default:
      break;
  }
}

// Store phase dispatch
// Called by storer warp to write results back
__device__ inline void dispatch_store(const Task& task, BlockRuntime& br, int slot_idx, int lane) {
  switch (task.header.opcode) {
    case OpCode::Axpy: {
      const auto& args = decode_args<OpTraits<OpCode::Axpy>::Args>(task);
      OpTraits<OpCode::Axpy>::store(args, br, slot_idx, lane);
      break;
    }

    case OpCode::Gemm: {
      // const auto& args = decode_args<OpTraits<OpCode::Gemm>::Args>(task);
      // OpTraits<OpCode::Gemm>::store(args, br, slot_idx, lane);
      break;
    }

    default:
      break;
  }
}

} // namespace mk
