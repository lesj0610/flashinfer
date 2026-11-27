// Helpers for the miscellaneous sm_89/sm_90 instructions emulated on SM80
// (fork-only). Owner: T2. Used by runtime/lib_misc.cu and tests/misc.
//
// Fragment locations of the mma.sync forms SM80 lacks (fp8 dense m16n8k16/k32,
// fp8 sparse m16n8k64, f64 m16n8k4/k8/k16). The 8-bit layouts are the PTX
// layouts of .s8/.u8, which .e4m3/.e5m2 share, so tests/misc/diag_sm80.cu checks
// them against SM80's native s8 mma.sync / mma.sp. Notation: lane l, g = l / 4,
// q = l % 4. Host+device (WGEMU_HD):
//   a8_loc(m, k, &lane, &reg, &byte)    A 16xK 8-bit (K = 16, 32; sparse: compressed col < 32)
//   b8_loc(k, n, &lane, &reg, &byte)    B Kx8 8-bit (K = 16, 32, 64)
//   cd_loc(m, n, &lane, &idx)           C/D 16x8 element index (f32: register; f16: half)
//   sp8_meta_loc(m, c, &lane, &bit)     sparse 8-bit metadata, row m, 4-column chunk c
//   a64_loc(m, k, &lane, &reg), b64_loc(k, n, &lane, &reg)    f64 m16n8k{4,8,16}
// Device only (all lanes of the warp must call them, full-mask shuffles):
//   gather_a8<K>(a[4], rows[2][K])      rows g and g + 8 of A
//   gather_b8<K>(b[4], cols[2][K])      columns 2q and 2q + 1 of B
//   gather_meta8(e, meta[2][16])        metadata nibbles of rows g and g + 8
//   gather_a64<K>(a[8], rows[2][K]), gather_b64<K>(b[4], cols[2][K])
// The gathers index the register arrays with compile-time indices only (the
// lane part is a shuffle source), so they do not force the arrays to local memory.
//
// Constants shared with the translator (tools/ptxemu/rules_misc.py) and with the
// async owner (OOB-NaN fill):
//   kOobNan16 / kOobNan32   OOB-NaN encodings ([inference], see below)
//   kCbar*                  software cluster-barrier state layout
//   kB128LockCount          size of the b128 atom lock table (u32 words)
#pragma once
#include <stdint.h>

#include "wgmma_emu/types.h"

namespace wgemu {
namespace misc {

// OOB-NaN encodings tested by fma.oob. [inference]: PTX names the value "a special
// NaN called OOB-NaN" (5.5.3.3) without giving its bits; the async/TMA OOB-NaN
// fill must write these same values.
constexpr uint32_t kOobNan32 = 0x7FFFFFFFu;
constexpr uint32_t kOobNan16 = 0x7FFFu;

// Software barrier.cluster state (module .shared, kCbarBytes bytes, 8-aligned):
//   word 0: cnt (bits 0..10) | expected (bits 11..21) | generation (bits 22..31)
//   byte 8 + linear_tid: 0 = never arrived, else 1 + parity of the phase arrived at
constexpr int kCbarBytes = 8 + 1024;
constexpr uint32_t kCbarCntMask = 0x7FFu;
constexpr int kCbarExpShift = 11;
constexpr int kCbarGenShift = 22;

constexpr int kB128LockCount = 1024;

WGEMU_HD void a8_loc(int m, int k, int* lane, int* reg, int* byte) {
  *reg = (m >= 8 ? 1 : 0) + (k >= 16 ? 2 : 0);
  *lane = 4 * (m % 8) + (k % 16) / 4;
  *byte = k % 4;
}

WGEMU_HD void b8_loc(int k, int n, int* lane, int* reg, int* byte) {
  *reg = k / 16;
  *lane = 4 * n + (k % 16) / 4;
  *byte = k % 4;
}

WGEMU_HD void cd_loc(int m, int n, int* lane, int* idx) {
  *lane = 4 * (m % 8) + n / 2;
  *idx = 2 * (m >= 8 ? 1 : 0) + n % 2;
}

WGEMU_HD void sp8_meta_loc(int m, int chunk, int* lane, int* bit) {
  *lane = 4 * (m % 8) + m / 8 + 2 * (chunk >= 8 ? 1 : 0);
  *bit = 4 * (chunk % 8);
}

WGEMU_HD void a64_loc(int m, int k, int* lane, int* reg) {
  *reg = 2 * (k / 4) + (m >= 8 ? 1 : 0);
  *lane = 4 * (m % 8) + k % 4;
}

WGEMU_HD void b64_loc(int k, int n, int* lane, int* reg) {
  *reg = k / 4;
  *lane = 4 * n + k % 4;
}

#if defined(__CUDACC__)

WGEMU_DEV uint32_t lane_id() {
  uint32_t l;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
  return l;
}

// rows[ri][k] = A[g + 8 * ri][k] (8-bit, low byte). Register index ri + 2 * (k / 16) matches
// a8_loc; the lane is 4 * g + (k % 16) / 4.
template <int K>
WGEMU_DEV void gather_a8(const uint32_t (&a)[4], uint32_t (&rows)[2][K]) {
  const int g = (int)lane_id() / 4;
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
#pragma unroll
    for (int k4 = 0; k4 < K / 4; ++k4) {
      const uint32_t v = __shfl_sync(0xFFFFFFFFu, a[ri + 2 * (k4 / 4)], 4 * g + k4 % 4);
#pragma unroll
      for (int by = 0; by < 4; ++by) rows[ri][4 * k4 + by] = (v >> (8 * by)) & 0xFFu;
    }
  }
}

// cols[ci][k] = B[k][2q + ci]. Register index k / 16 and lane 4 * n + (k % 16) / 4 (b8_loc).
template <int K>
WGEMU_DEV void gather_b8(const uint32_t (&b)[4], uint32_t (&cols)[2][K]) {
  const int q = (int)lane_id() % 4;
#pragma unroll
  for (int ci = 0; ci < 2; ++ci) {
#pragma unroll
    for (int k4 = 0; k4 < K / 4; ++k4) {
      const uint32_t v = __shfl_sync(0xFFFFFFFFu, b[k4 / 4], 4 * (2 * q + ci) + k4 % 4);
#pragma unroll
      for (int by = 0; by < 4; ++by) cols[ci][4 * k4 + by] = (v >> (8 * by)) & 0xFFu;
    }
  }
}

// meta[ri][c]: 4-bit metadata of row g + 8 * ri, chunk c (sp8_meta_loc).
WGEMU_DEV void gather_meta8(uint32_t e, uint32_t (&meta)[2][16]) {
  const int g = (int)lane_id() / 4;
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t v = __shfl_sync(0xFFFFFFFFu, e, 4 * g + ri + 2 * h);
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) meta[ri][8 * h + c8] = (v >> (4 * c8)) & 0xFu;
    }
  }
}

WGEMU_DEV uint64_t shfl_u64(uint64_t x, int src) {
  const uint32_t lo = __shfl_sync(0xFFFFFFFFu, (uint32_t)x, src);
  const uint32_t hi = __shfl_sync(0xFFFFFFFFu, (uint32_t)(x >> 32), src);
  return ((uint64_t)hi << 32) | lo;
}

// rows[ri][k] = A[g + 8 * ri][k] (f64 bits), a64_loc.
template <int K>
WGEMU_DEV void gather_a64(const uint64_t (&a)[8], uint64_t (&rows)[2][K]) {
  const int g = (int)lane_id() / 4;
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
#pragma unroll
    for (int k = 0; k < K; ++k) rows[ri][k] = shfl_u64(a[2 * (k / 4) + ri], 4 * g + k % 4);
  }
}

// cols[ci][k] = B[k][2q + ci] (f64 bits), b64_loc.
template <int K>
WGEMU_DEV void gather_b64(const uint64_t (&b)[4], uint64_t (&cols)[2][K]) {
  const int q = (int)lane_id() % 4;
#pragma unroll
  for (int ci = 0; ci < 2; ++ci) {
#pragma unroll
    for (int k = 0; k < K; ++k) cols[ci][k] = shfl_u64(b[k / 4], 4 * (2 * q + ci) + k % 4);
  }
}

#endif  // __CUDACC__

}  // namespace misc
}  // namespace wgemu
