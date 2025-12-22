#pragma once

#include <cstdint>

namespace pk {

enum class ProfileRange : int {
  ControllerDepsWait = 0,
  ControllerQueuePop = 1,
  LoaderSlotWait = 2,
  LoaderLoad = 3,
  ComputeSlotWait = 4,
  ComputeMath = 5,
  StorerSlotWait = 6,
  StorerThreadfence = 7,
  StorerDepsMarkReady = 8,
  Count = 9,
};

struct ProfileCounters {
  // Total cycles per range across all blocks/warps that report it.
  unsigned long long cycles[static_cast<int>(ProfileRange::Count)];
};

} // namespace pk

