#include <cstdint>
#include <cstdio>

int get_phasebit(uint32_t bitfield, int slot) {
  return (bitfield >> slot) & 1;
}

void update_phasebit(uint32_t& bitfield, int slot) {
  bitfield ^= (1U << slot);
}

// Advance to next slot
int ring_advance(int curr, int stages) {
  return (curr + 1) % stages;
}

void print_bitfield(uint32_t bitfield, int num_slots) {
  printf("0b");
  for (int i = num_slots - 1; i >= 0; i--) {
    // print i'th bit, aka phase bit i
    printf("%d", get_phasebit(bitfield, i));
  }
  printf("\n");
}

int main(int argc, char const* argv[]) {
  constexpr int N_STAGES = 4;
  constexpr int ROUNDS = 3;

  uint32_t phasebits = 0;
  int curr = 0;

  for (int i = 0; i < N_STAGES * ROUNDS; i++) {
    printf("[%d.%d] : ", i, curr);
    print_bitfield(phasebits, N_STAGES);
    // advance phase
    update_phasebit(phasebits, curr);
    // advance ring
    curr = ring_advance(curr, N_STAGES);
    if (curr % N_STAGES == 0) {
      printf("\n");
    }
  }
  return 0;
}
