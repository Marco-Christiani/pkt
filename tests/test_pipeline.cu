// Tests PipelineSlot, SlotPhase, and ring buffer

#include "../include/config.cuh"
#include "../include/pipeline.cuh"
#include "test_utils.cuh"

using namespace pk;
using namespace pk::test;

bool test_pipeline_slot_alignment() {
  // should be 128-byte aligned to prevent false sharing
  TEST_ASSERT_EQ(alignof(PipelineSlot), 128, "PipelineSlot should be 128-byte aligned");

  return true;
}

bool test_slot_phase_values() {
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Empty), 0, "Empty should be 0");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Loading), 1, "Loading should be 1");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Loaded), 2, "Loaded should be 2");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Computing), 3, "Computing should be 3");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Computed), 4, "Computed should be 4");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Storing), 5, "Storing should be 5");
  TEST_ASSERT_EQ(static_cast<int>(SlotPhase::Done), 6, "Done should be 6");

  return true;
}

bool test_ring_advance_power_of_2() {
  // Verify kPipelineStages is a power of 2 for the optimized bit-mask path
  static_assert((Config::kPipelineStages & (Config::kPipelineStages - 1)) == 0,
                "Test assumes kPipelineStages is a power of 2");
  static_assert(Config::kPipelineStages > 0, "kPipelineStages must be positive");

  // Test wrap-around behavior on CPU
  int index = 0;
  int iterations = Config::kPipelineStages * 3; // a few wrap-arounds
  for (int i = 0; i < iterations; ++i) {
    int expected = (i + 1) % Config::kPipelineStages;
    index = (index + 1) & (Config::kPipelineStages - 1); // same as ring_advance for power of 2

    if (index != expected) {
      fprintf(stderr, "ring_advance mismatch at iteration %d: got %d, expected %d\n", i, index,
              expected);
      return false;
    }
  }

  return true;
}

bool test_config_const_consistency() {
  // total warps = controller + loader + storer + compute warps
  int expected_total = 1 + 1 + 1 + Config::kNumComputeWarps;
  TEST_ASSERT_EQ(Config::kTotalWarps, expected_total, "Total warps mismatch");

  // threads per block = warps * warp size
  TEST_ASSERT_EQ(Config::kThreadsPerBlock, Config::kTotalWarps * Config::kWarpSize,
                 "Threads per block mismatch");

  // max logical pages = pipeline stages * pages per slot
  TEST_ASSERT_EQ(Config::kMaxLogicalPages, Config::kPipelineStages * Config::kPagesPerSlot,
                 "Max logical pages mismatch");

  return true;
}

bool test_warp_roles_defaults() {
  WarpRoles roles;

  TEST_ASSERT_EQ(roles.controller, Config::kControllerWarp, "Controller warp mismatch");
  TEST_ASSERT_EQ(roles.loader, Config::kLoaderWarp, "Loader warp mismatch");
  TEST_ASSERT_EQ(roles.storer, Config::kStorerWarp, "Storer warp mismatch");
  TEST_ASSERT_EQ(roles.first_compute, Config::kFirstComputeWarp, "First compute warp mismatch");
  TEST_ASSERT_EQ(roles.num_compute, Config::kNumComputeWarps, "Num compute warps mismatch");

  return true;
}

__global__ void test_ring_advance_kernel(int* results, int num_iterations) {
  if (threadIdx.x == 0) {
    int index = 0;
    for (int i = 0; i < num_iterations; ++i) {
      index = ring_advance(index);
      results[i] = index;
    }
  }
}

bool test_ring_advance_device() {
  const int num_iterations = 12;
  int* d_results = alloc_device<int>(num_iterations);

  test_ring_advance_kernel<<<1, 32>>>(d_results, num_iterations);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_results[num_iterations];
  copy_to_host(h_results, d_results, num_iterations);
  cudaFree(d_results);

  // Expect: 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0
  for (int i = 0; i < num_iterations; ++i) {
    int expected = (i + 1) % Config::kPipelineStages;
    if (h_results[i] != expected) {
      fprintf(stderr, "Device ring_advance mismatch at %d: got %d, expected %d\n", i, h_results[i],
              expected);
      return false;
    }
  }

  return true;
}

__global__ void test_warp_roles_kernel(int* results) {
  WarpRoles roles;
  int wid = threadIdx.x / Config::kWarpSize;

  if (threadIdx.x % Config::kWarpSize == 0) {
    results[wid * 4 + 0] = is_controller(wid, roles) ? 1 : 0;
    results[wid * 4 + 1] = is_loader(wid, roles) ? 1 : 0;
    results[wid * 4 + 2] = is_storer(wid, roles) ? 1 : 0;
    results[wid * 4 + 3] = is_compute(wid, roles) ? 1 : 0;
  }
}

bool test_warp_role_identification() {
  int* d_results = alloc_device_zero<int>(Config::kTotalWarps * 4);

  test_warp_roles_kernel<<<1, Config::kThreadsPerBlock>>>(d_results);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_results[Config::kTotalWarps * 4];
  copy_to_host(h_results, d_results, Config::kTotalWarps * 4);
  cudaFree(d_results);

  // Each warp should have exactly one role (except controller/loader/storer are exclusive,
  // compute is the remaining warps)
  for (int wid = 0; wid < Config::kTotalWarps; ++wid) {
    int is_ctrl = h_results[wid * 4 + 0];
    int is_load = h_results[wid * 4 + 1];
    int is_store = h_results[wid * 4 + 2];
    int is_comp = h_results[wid * 4 + 3];

    int total_roles = is_ctrl + is_load + is_store + is_comp;
    if (total_roles != 1) {
      fprintf(stderr, "Warp %d has %d roles (expected 1)\n", wid, total_roles);
      return false;
    }

    // Check specialized
    if (wid == Config::kControllerWarp && !is_ctrl) {
      fprintf(stderr, "Warp %d should be controller\n", wid);
      return false;
    }
    if (wid == Config::kLoaderWarp && !is_load) {
      fprintf(stderr, "Warp %d should be loader\n", wid);
      return false;
    }
    if (wid == Config::kStorerWarp && !is_store) {
      fprintf(stderr, "Warp %d should be storer\n", wid);
      return false;
    }
    if (wid >= Config::kFirstComputeWarp &&
        wid < Config::kFirstComputeWarp + Config::kNumComputeWarps && !is_comp) {
      fprintf(stderr, "Warp %d should be compute\n", wid);
      return false;
    }
  }

  return true;
}

// Kernel to test lane_id and warp_id
__global__ void test_thread_ids_kernel(int* lane_ids, int* warp_ids) {
  int tid = threadIdx.x;
  lane_ids[tid] = lane_id();
  warp_ids[tid] = warp_id();
}

// Test lane_id and warp_id helpers
bool test_thread_id_helpers() {
  int* d_lane_ids = alloc_device<int>(Config::kThreadsPerBlock);
  int* d_warp_ids = alloc_device<int>(Config::kThreadsPerBlock);

  test_thread_ids_kernel<<<1, Config::kThreadsPerBlock>>>(d_lane_ids, d_warp_ids);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_lane_ids[Config::kThreadsPerBlock];
  int h_warp_ids[Config::kThreadsPerBlock];
  copy_to_host(h_lane_ids, d_lane_ids, Config::kThreadsPerBlock);
  copy_to_host(h_warp_ids, d_warp_ids, Config::kThreadsPerBlock);

  cudaFree(d_lane_ids);
  cudaFree(d_warp_ids);

  // Make sure they match
  for (int tid = 0; tid < Config::kThreadsPerBlock; ++tid) {
    int expected_lane = tid % Config::kWarpSize;
    int expected_warp = tid / Config::kWarpSize;

    if (h_lane_ids[tid] != expected_lane) {
      fprintf(stderr, "Thread %d: lane_id mismatch (got %d, expected %d)\n", tid, h_lane_ids[tid],
              expected_lane);
      return false;
    }
    if (h_warp_ids[tid] != expected_warp) {
      fprintf(stderr, "Thread %d: warp_id mismatch (got %d, expected %d)\n", tid, h_warp_ids[tid],
              expected_warp);
      return false;
    }
  }

  return true;
}

int main() {
  TestCase tests[] = {
      {.name = "pipeline_slot_alignment", .func = test_pipeline_slot_alignment},
      {.name = "slot_phase_values", .func = test_slot_phase_values},
      {.name = "ring_advance_power_of_2", .func = test_ring_advance_power_of_2},
      {.name = "config_const_consistency", .func = test_config_const_consistency},
      {.name = "warp_roles_defaults", .func = test_warp_roles_defaults},
      {.name = "ring_advance_device", .func = test_ring_advance_device},
      {.name = "warp_role_identification", .func = test_warp_role_identification},
      {.name = "thread_id_helpers", .func = test_thread_id_helpers},
  };

  return run_tests(tests, sizeof(tests) / sizeof(tests[0]));
}
