#pragma once

#include <cstdint>

namespace pk {

struct SegmentDesc {
  int begin;
  int end; // [begin, end)
};

} // namespace pk
