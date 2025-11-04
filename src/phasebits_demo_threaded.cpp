#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>

#include "log.hpp"
#include "phasebits.hpp"

void random_sleep() {
  static thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(1, 10);
  std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
}

constexpr int N_STAGES = 4;
constexpr int ROUNDS = 3;
constexpr int TOTAL_STEPS = N_STAGES * ROUNDS;

std::atomic<uint32_t> phasebits{0};

void producer(int id) {
  int slot = 0;
  int expected = 0;

  for (int i = 0; i < TOTAL_STEPS; i++) {
    Log(LogLevel::DEBUG) << "producer waiting: slot=" << slot << " expected=" << expected;

    wait_for_phase(phasebits, slot, expected);

    Log(LogLevel::INFO) << "producer resumed -> producing: slot=" << slot;
    random_sleep();

    arrive(phasebits, slot);
    expected ^= 1;
    slot = ring_advance(slot, N_STAGES);
  }

  Log(LogLevel::INFO) << "producer done";
}

void consumer(int id) {
  int slot = 0;
  int expected = 1;

  for (int i = 0; i < TOTAL_STEPS; i++) {
    Log(LogLevel::DEBUG) << "consumer waiting: slot=" << slot << " expected=" << expected;

    wait_for_phase(phasebits, slot, expected);

    Log(LogLevel::INFO) << "consumer resumed -> consuming: slot=" << slot;
    random_sleep();

    arrive(phasebits, slot);
    expected ^= 1;
    slot = ring_advance(slot, N_STAGES);
  }
  Log(LogLevel::INFO) << "consumer done";
}

int main(int argc, char const* argv[]) {
  // uint32_t phasebits = 0;
  int curr = 0;

  printf("testing phasebits...\n");
  for (int i = 0; i < N_STAGES * ROUNDS; i++) {
    printf("[%d.%d] : ", i, curr);

    print_bitfield(phasebits, N_STAGES);
    // get_phasebit(phasebits, curr);

    // advance phase
    update_phasebit(phasebits, curr);
    // advance ring
    curr = ring_advance(curr, N_STAGES);
    if (curr % N_STAGES == 0) {
      printf("\n");
    }
  }
  std::thread t_p(producer, 0);
  std::thread t_c(consumer, 1);
  t_p.join();
  t_c.join();
  return 0;
}
