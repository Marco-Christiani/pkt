#pragma once

#include <cstdint>

namespace pk {

// Per-block counters; written once per block at kernel end.
struct PerfBlock {
  // Controller (lane0)
  unsigned long long ctrl_dequeue_cycles{0};
  unsigned long long ctrl_scan_cycles{0};
  unsigned long long ctrl_claim_cycles{0};
  unsigned long long ctrl_wait_cycles{0};
  unsigned long long ctrl_issue_cycles{0};

  unsigned long long ctrl_scans_total{0};
  unsigned long long ctrl_scan_len_sum{0};
  unsigned long long ctrl_scan_len_max{0};
  unsigned long long ctrl_found_ready_count{0};
  unsigned long long ctrl_deferred_count{0};
  unsigned long long ctrl_no_ready_rounds{0};

  unsigned long long ctrl_claim_attempts{0};
  unsigned long long ctrl_claim_success{0};
  unsigned long long ctrl_claim_fail{0};

  // Segment gate window occupancy sampling (controller lane0)
  unsigned long long ctrl_window_occupancy_sum{0};
  unsigned long long ctrl_window_occupancy_samples{0};

  unsigned long long pending_push_count{0};
  unsigned long long pending_pop_count{0};
  unsigned long long pending_max_depth{0};
  unsigned long long pending_full_count{0};

  // Loader (lane0 of loader role)
  unsigned long long load_wait_slot_cycles{0};
  unsigned long long load_work_cycles{0};
  unsigned long long load_page_reuse_hits{0};
  unsigned long long load_page_refill_count{0};

  // Compute (lane0 of compute warps; summed per block)
  unsigned long long compute_wait_loaded_cycles{0};
  unsigned long long compute_work_cycles{0};

  // Storer (lane0 of storer role)
  unsigned long long store_wait_computed_cycles{0};
  unsigned long long store_work_cycles{0};

  // Global progress / balance (per block)
  unsigned long long tasks_issued{0};
  unsigned long long tasks_completed{0};
};

} // namespace pk
