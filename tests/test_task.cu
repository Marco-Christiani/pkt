// Test task args encoding/decoding

#include <cstring>

#include "../include/ops/axpy.cuh"
#include "../include/task.cuh"
#include "test_utils.cuh"

using namespace pk;
using namespace pk::test;

bool test_task_size_and_alignment() {
  TEST_ASSERT_EQ(alignof(Task), 16, "Task alignment should be 16 bytes");
  TEST_ASSERT_EQ(sizeof(TaskHeader), 8, "TaskHeader should be 8 bytes");

  // total = header + args (+ padding)
  size_t expected_size = sizeof(TaskHeader) + Config::kArgBytes;
  // with alignment padding
  TEST_ASSERT(sizeof(Task) >= expected_size, "Task size too small");

  return true;
}

// TaskHeader field layout
bool test_task_header_layout() {
  TaskHeader header;
  header.opcode = OpCode::Axpy;
  header.arg_bytes = 24;
  header.user_tag = 0x12345678;
  TEST_ASSERT_EQ(static_cast<int>(header.opcode), 1, "Axpy opcode should be 1");
  TEST_ASSERT_EQ(header.arg_bytes, 24, "arg_bytes mismatch");
  TEST_ASSERT_EQ(header.user_tag, 0x12345678u, "user_tag mismatch");
  return true;
}

// encoding and decoding AxpyArgs
bool test_axpy_args_encode_decode() {
  Task task;

  // some args
  float dummy_x[10];
  float dummy_y[10];
  const float* ptr_x = dummy_x;
  float* ptr_y = dummy_y;

  AxpyArgs args_in;
  args_in.x = ptr_x;
  args_in.y = ptr_y;
  args_in.a = 2.5f;
  args_in.n = 1024;

  encode_args(task, OpCode::Axpy, args_in);

  // Check header
  TEST_ASSERT_EQ(static_cast<int>(task.header.opcode), static_cast<int>(OpCode::Axpy),
                 "opcode mismatch after encode");
  TEST_ASSERT_EQ(task.header.arg_bytes, sizeof(AxpyArgs), "arg_bytes mismatch after encode");

  // Decode
  const AxpyArgs& args_out = decode_args<AxpyArgs>(task);

  TEST_ASSERT(args_out.x == ptr_x, "x pointer mismatch");
  TEST_ASSERT(args_out.y == ptr_y, "y pointer mismatch");
  TEST_ASSERT_NEAR(args_out.a, 2.5f, 1e-6f, "alpha value mismatch");
  TEST_ASSERT_EQ(args_out.n, 1024, "n value mismatch");

  return true;
}

// Test args are zero-filled before copy
bool test_args_zero_fill() {
  Task task;

  // fill w garbage
  std::memset(&task, 0xFF, sizeof(Task));

  // small args
  AxpyArgs args;
  args.x = nullptr;
  args.y = nullptr;
  args.a = 1.0f;
  args.n = 100;

  encode_args(task, OpCode::Axpy, args);

  // make sure its zero
  const uint8_t* payload = task.args;
  for (size_t i = sizeof(AxpyArgs); i < Config::kArgBytes; ++i) {
    if (payload[i] != 0) {
      fprintf(stderr, "Non-zero byte at offset %zu: 0x%02x\n", i, payload[i]);
      return false;
    }
  }

  return true;
}

// Test multiple encodes overwrite
bool test_multiple_encodes() {
  Task task;

  // encode 1
  AxpyArgs args1;
  args1.x = reinterpret_cast<const float*>(0x1000);
  args1.y = reinterpret_cast<float*>(0x2000);
  args1.a = 1.0f;
  args1.n = 100;
  encode_args(task, OpCode::Axpy, args1);

  // encode 2
  AxpyArgs args2;
  args2.x = reinterpret_cast<const float*>(0x3000);
  args2.y = reinterpret_cast<float*>(0x4000);
  args2.a = 2.0f;
  args2.n = 200;
  encode_args(task, OpCode::Axpy, args2);

  // Decode and check second values
  const AxpyArgs& decoded = decode_args<AxpyArgs>(task);

  TEST_ASSERT(decoded.x == reinterpret_cast<const float*>(0x3000), "x pointer not overwritten");
  TEST_ASSERT(decoded.y == reinterpret_cast<float*>(0x4000), "y pointer not overwritten");
  TEST_ASSERT_NEAR(decoded.a, 2.0f, 1e-6f, "alpha not overwritten");
  TEST_ASSERT_EQ(decoded.n, 200, "n not overwritten");

  return true;
}

bool test_opcode_values() {
  TEST_ASSERT_EQ(static_cast<int>(OpCode::Invalid), 0, "Invalid opcode should be 0");
  TEST_ASSERT_EQ(static_cast<int>(OpCode::Axpy), 1, "Axpy opcode should be 1");
  TEST_ASSERT_EQ(static_cast<int>(OpCode::Gemm), 2, "Gemm opcode should be 2");
  return true;
}

// Test AxpyArgs size
bool test_axpy_args_size() {
  // AxpyArgs: x(8) + y(8) + a(4) + n(4) = 24 bytes
  TEST_ASSERT_EQ(sizeof(AxpyArgs), 24, "AxpyArgs should be 24 bytes");

  // Must fit in kArgBytes
  TEST_ASSERT(sizeof(AxpyArgs) <= Config::kArgBytes, "AxpyArgs exceeds kArgBytes");

  return true;
}

int main() {
  TestCase tests[] = {
      {.name = "task_size_and_alignment", .func = test_task_size_and_alignment},
      {.name = "task_header_layout", .func = test_task_header_layout},
      {.name = "axpy_args_encode_decode", .func = test_axpy_args_encode_decode},
      {.name = "args_zero_fill", .func = test_args_zero_fill},
      {.name = "multiple_encodes", .func = test_multiple_encodes},
      {.name = "opcode_values", .func = test_opcode_values},
      {.name = "axpy_args_size", .func = test_axpy_args_size},
  };

  return run_tests(tests, sizeof(tests) / sizeof(tests[0]));
}
