#pragma once

#include "block_runtime.cuh"
#include "task.cuh"

namespace pk {

// Each operation specializes this template to define:
//   - Args: typed argument structure
//   - load(args, br, slot_idx, lane): prefetch / stage any inputs (may be a no-op)
//   - compute(args, br, slot_idx, lane, compute_warp_idx, num_compute_warps): do the work
//   - store(args, br, slot_idx, lane): write results back (may be a no-op)
//
// The runtime dispatches to these phases in pipeline order.
template <OpCode Code> struct OpTraits;

} // namespace pk
