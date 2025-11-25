#pragma once

#include "block_runtime.cuh"
#include "op_traits.cuh"
#include "ops/axpy.cuh"
#include "ops/linear_bwd.cuh"
#include "ops/linear_fwd.cuh"
#include "ops/mse.cuh"
#include "ops/sgd.cuh"
#include "ops/zero.cuh"
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
    case OpCode::ZeroMemory: {
      const auto& args = decode_args<OpTraits<OpCode::ZeroMemory>::Args>(task);
      OpTraits<OpCode::ZeroMemory>::load(args, br, slot_idx, lane);
      break;
    }
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
    case OpCode::LinearForward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearForward>::Args>(task);
      OpTraits<OpCode::LinearForward>::load(args, br, slot_idx, lane);
      break;
    }
    case OpCode::MSELoss: {
      const auto& args = decode_args<OpTraits<OpCode::MSELoss>::Args>(task);
      OpTraits<OpCode::MSELoss>::load(args, br, slot_idx, lane);
      break;
    }
    case OpCode::LinearBackward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearBackward>::Args>(task);
      OpTraits<OpCode::LinearBackward>::load(args, br, slot_idx, lane);
      break;
    }
    case OpCode::SGDUpdate: {
      const auto& args = decode_args<OpTraits<OpCode::SGDUpdate>::Args>(task);
      OpTraits<OpCode::SGDUpdate>::load(args, br, slot_idx, lane);
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
    case OpCode::ZeroMemory: {
      const auto& args = decode_args<OpTraits<OpCode::ZeroMemory>::Args>(task);
      OpTraits<OpCode::ZeroMemory>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                            num_compute_warps);
      break;
    }
    case OpCode::Axpy: {
      const auto& args = decode_args<OpTraits<OpCode::Axpy>::Args>(task);
      OpTraits<OpCode::Axpy>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                      num_compute_warps);
      break;
    }

    case OpCode::Gemm: {
      // const auto& args = decode_args<OpTraits<OpCode::Gemm>::Args>(task);
      // OpTraits<OpCode::Gemm>::compute(args, br, slot_idx, lane,
      //                                 compute_warp_idx, num_compute_warps);
      break;
    }
    case OpCode::LinearForward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearForward>::Args>(task);
      OpTraits<OpCode::LinearForward>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                               num_compute_warps);
      break;
    }
    case OpCode::MSELoss: {
      const auto& args = decode_args<OpTraits<OpCode::MSELoss>::Args>(task);
      OpTraits<OpCode::MSELoss>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                         num_compute_warps);
      break;
    }
    case OpCode::LinearBackward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearBackward>::Args>(task);
      OpTraits<OpCode::LinearBackward>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                                num_compute_warps);
      break;
    }
    case OpCode::SGDUpdate: {
      const auto& args = decode_args<OpTraits<OpCode::SGDUpdate>::Args>(task);
      OpTraits<OpCode::SGDUpdate>::compute(args, br, slot_idx, lane, compute_warp_idx,
                                           num_compute_warps);
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
    case OpCode::ZeroMemory: {
      const auto& args = decode_args<OpTraits<OpCode::ZeroMemory>::Args>(task);
      OpTraits<OpCode::ZeroMemory>::store(args, br, slot_idx, lane);
      break;
    }
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
    case OpCode::LinearForward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearForward>::Args>(task);
      OpTraits<OpCode::LinearForward>::store(args, br, slot_idx, lane);
      break;
    }
    case OpCode::MSELoss: {
      const auto& args = decode_args<OpTraits<OpCode::MSELoss>::Args>(task);
      OpTraits<OpCode::MSELoss>::store(args, br, slot_idx, lane);
      break;
    }
    case OpCode::LinearBackward: {
      const auto& args = decode_args<OpTraits<OpCode::LinearBackward>::Args>(task);
      OpTraits<OpCode::LinearBackward>::store(args, br, slot_idx, lane);
      break;
    }
    case OpCode::SGDUpdate: {
      const auto& args = decode_args<OpTraits<OpCode::SGDUpdate>::Args>(task);
      OpTraits<OpCode::SGDUpdate>::store(args, br, slot_idx, lane);
      break;
    }

    default:
      break;
  }
}

} // namespace pk
