// Shared host helpers for the tests/misc end-to-end programs (fork-only). Owner: T2.
//
// Each program builds sm_90a kernels with inline PTX (translated by tools/nvcc_emu), runs
// them and compares every result with a CPU expectation derived from the PTX ISA text (and
// include/wgmma_emu/cvt.h, itself checked against tests/misc/ref_cvt.py). Output follows
// PLAN section 11: one "WGEMU-TEST name=<id> cases=<n> mismatches=<m>" line per item, at
// most kMaxMismatchLines "WGEMU-MISMATCH name=<id> ..." lines per item, "WGEMU-TEST-END".
#pragma once
#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define MISC_CUDA(call)                                                    \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, \
              cudaGetErrorString(err_));                                   \
      exit(3);                                                             \
    }                                                                      \
  } while (0)

// Launch check: launch errors and errors raised while the kernel ran (trap, bad address).
#define MISC_SYNC()                     \
  do {                                  \
    MISC_CUDA(cudaGetLastError());      \
    MISC_CUDA(cudaDeviceSynchronize()); \
  } while (0)

namespace misc_e2e {

constexpr int kMaxMismatchLines = 8;

inline long& total_mismatches() {
  static long n = 0;
  return n;
}

// One result line. Use check() per case, then the destructor prints the WGEMU-TEST line.
class Item {
 public:
  explicit Item(const char* name) : name_(name) {}
  Item(const Item&) = delete;
  Item& operator=(const Item&) = delete;
  ~Item() {
    printf("WGEMU-TEST name=%s cases=%ld mismatches=%ld\n", name_, cases_, mism_);
    fflush(stdout);
    total_mismatches() += mism_;
  }
  // Counts one case; on failure prints the formatted detail (bounded).
  __attribute__((format(printf, 3, 4))) bool check(bool ok, const char* fmt, ...) {
    ++cases_;
    if (ok) return true;
    if (mism_ < kMaxMismatchLines) {
      printf("WGEMU-MISMATCH name=%s ", name_);
      va_list ap;
      va_start(ap, fmt);
      vprintf(fmt, ap);
      va_end(ap);
      printf("\n");
    }
    ++mism_;
    return false;
  }
  long cases() const { return cases_; }

 private:
  const char* name_;
  long cases_ = 0, mism_ = 0;
};

inline int finish() {
  printf("WGEMU-TEST-END\n");
  fflush(stdout);
  return total_mismatches() == 0 ? 0 : 1;
}

// Device buffer mirroring a host vector.
template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  Dev() = default;
  explicit Dev(size_t count) : n(count) {
    MISC_CUDA(cudaMalloc(&p, (count ? count : 1) * sizeof(T)));
    MISC_CUDA(cudaMemset(p, 0, (count ? count : 1) * sizeof(T)));
  }
  explicit Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
  ~Dev() {
    if (p) cudaFree(p);
  }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void put(const std::vector<T>& h) {
    MISC_CUDA(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    MISC_CUDA(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

inline int blocks_for(size_t n, int threads = 256) { return (int)((n + threads - 1) / threads); }

// splitmix64: deterministic inputs.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t u32() { return (uint32_t)next(); }
  uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};

inline uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
inline float u2f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline uint64_t d2u(double d) {
  uint64_t u;
  memcpy(&u, &d, 8);
  return u;
}
inline double u2d(uint64_t u) {
  double d;
  memcpy(&d, &u, 8);
  return d;
}

inline bool nan32(uint32_t x) { return (x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu); }
inline bool nan64(uint64_t x) {
  return (x & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (x & 0xFFFFFFFFFFFFFull);
}
inline bool nan_f16(uint32_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }
inline bool nan_bf16(uint32_t h) { return (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu); }

// Equal bits, or both NaN when PTX leaves the NaN encoding open (hardware-native paths).
inline bool same32(uint32_t got, uint32_t want, bool nan_class) {
  return got == want || (nan_class && nan32(got) && nan32(want));
}
inline bool same64(uint64_t got, uint64_t want, bool nan_class) {
  return got == want || (nan_class && nan64(got) && nan64(want));
}
inline bool same_f16(uint32_t got, uint32_t want, bool nan_class) {
  got &= 0xFFFFu;
  want &= 0xFFFFu;
  return got == want || (nan_class && nan_f16(got) && nan_f16(want));
}
inline bool same_bf16(uint32_t got, uint32_t want, bool nan_class) {
  got &= 0xFFFFu;
  want &= 0xFFFFu;
  return got == want || (nan_class && nan_bf16(got) && nan_bf16(want));
}

// f32 inputs: special encodings, values near every power of two, seeded random words.
inline std::vector<uint32_t> f32_inputs(int n_random, uint64_t seed) {
  std::vector<uint32_t> v = {0x00000000u, 0x80000000u, 0x00000001u, 0x007FFFFFu, 0x00800000u,
                             0x3F800000u, 0xBF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u,
                             0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x7FFFFFFFu,
                             0x43E00000u, 0x43F00000u, 0x47800000u, 0x477FE000u, 0x477FF000u};
  for (uint32_t e = 1; e < 255; ++e) {
    for (uint32_t m : {0u, 1u, 0x7FFFFFu, 0x400000u, 0x080000u, 0x0FFFFFu, 0x100000u, 0x8000u,
                       0x10000u, 0x7FFFu}) {
      v.push_back((e << 23) | m);
      v.push_back(0x80000000u | (e << 23) | m);
    }
  }
  Rng r(seed);
  for (int i = 0; i < n_random; ++i) {
    v.push_back(r.u32());
    v.push_back((r.u32() & 0x807FFFFFu) | ((97u + r.below(50)) << 23));
  }
  return v;
}

}  // namespace misc_e2e
