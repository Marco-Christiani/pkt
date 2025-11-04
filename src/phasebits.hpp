#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <type_traits>

template <typename T> struct is_atomic : std::false_type {};

template <typename U> struct is_atomic<std::atomic<U>> : std::true_type {};

template <typename T> inline constexpr bool is_atomic_v = is_atomic<T>::value;

template <typename T>
[[nodiscard]] constexpr std::conditional_t<is_atomic_v<T>, typename T::value_type, T>
unwrap(const T& x) {
  using Child = std::conditional_t<is_atomic_v<T>, typename T::value_type, T>;
  if constexpr (is_atomic_v<T>) {
    return x.load(std::memory_order_acquire);
  } else {
    return static_cast<Child>(x);
  }
}

// Zig >> ...
// template <typename T> int get_phasebit(T const& bitfield, int slot) {
//   using Child = std::conditional_t<is_atomic_v<T>, typename T::value_type, T>;
//   Child val = static_cast<Child>(bitfield);
//   return (val >> slot) & 1;
// }

template <typename T> int get_phasebit(std::atomic<T>const & bitfield, int slot) {
  const T val = bitfield.load(std::memory_order_acquire);
  return (val >> slot) & 1;
}

// update the phasebit for slot
template <typename T> void update_phasebit(std::atomic<T>& bitfield, int slot) {
  const T mask = (T(1) << slot);
  bitfield.fetch_xor(mask, std::memory_order_acq_rel);
}

// Advance to next slot
int ring_advance(int curr, int stages) {
  return (curr + 1) % stages;
}

template <typename T> void print_bitfield(std::atomic<T>const & bitfield, int num_slots) {
  printf("0b");
  for (int i = num_slots - 1; i >= 0; i--) {
    // print i'th bit, aka phase bit i
    printf("%d", get_phasebit(bitfield, i));
  }
  printf("\n");
}

// Wait until phasebits[slot] == expected_phase
template <typename T> void wait_for_phase(std::atomic<T> const& phasebits, int slot, int expected_phase) {
    while (get_phasebit(phasebits, slot) != expected_phase){}
}

// Flip the phase bit for this slot (signals the next stage)
template <typename T> void arrive(std::atomic<T>& phasebits, int slot) {
    update_phasebit(phasebits, slot);
}
