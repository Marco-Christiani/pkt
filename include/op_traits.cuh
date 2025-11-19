#pragma once

#include "block_runtime.cuh"
#include "task.cuh"

namespace pk {

// Each operation specializes this template to define:
//   - Args: typed argument structure
//   - load(): data loading phase (future - prefetch to SMEM)
//   - compute(): main computation
//   - store(): result writeback (if compute writes directly)
//
// The runtime dispatches to these three phases in pipeline order
template <OpCode Code> struct OpTraits;

} // namespace mk
