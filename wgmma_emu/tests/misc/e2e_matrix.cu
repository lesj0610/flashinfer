// End-to-end check of stmatrix.m8n8 .b16 and ldmatrix.m8n16 .s8.s4 (sm_90a) rewritten by
// tools/ptxemu/rules_misc.py into lane shuffles plus shared stores / loads (fork-only).
// Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Layout (ptx.txt 9.7.16.5.15 / 9.7.16.5.16): thread 8 i + r supplies the row address of row r
// of matrix i; lane 4 g + q holds, for matrix i, row g at columns 2 q, 2 q + 1 (b16) or
// 4 q .. 4 q + 3 (s4 expanded to s8). .trans swaps rows and columns. Rows sit at a permuted
// slot of a padded shared buffer so that each row address is checked on its own; every byte
// outside the written rows must keep its sentinel.
// [inference] The .s4 source row is eight packed bytes and the lower nibble of each byte holds
// the lower column; the PTX text gives no figure for the nibble order.
#include <algorithm>

#include "misc_e2e.h"

using namespace misc_e2e;

constexpr int kWarps = 4;
constexpr int kSlots = 32;       // row slots of 32 bytes per warp
constexpr int kSlotHalves = 16;  // b16 per slot
constexpr int kSlotWords = 8;    // u32 per slot
constexpr uint16_t kSentinel = 0xFFFF;

// Element value lane holds in half k of register i (unique per lane, register and half).
__host__ __device__ inline uint16_t st_val(int lane, int i, int k) {
  return (uint16_t)(0x4000 | (i << 8) | (lane << 1) | k);
}

#define ST_ASM_1(INS, A, AC) asm volatile(INS " [%0], {%1};" ::AC(A), "r"(r[0]) : "memory")
#define ST_ASM_2(INS, A, AC) \
  asm volatile(INS " [%0], {%1, %2};" ::AC(A), "r"(r[0]), "r"(r[1]) : "memory")
#define ST_ASM_4(INS, A, AC)                                                                      \
  asm volatile(INS " [%0], {%1, %2, %3, %4};" ::AC(A), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) \
               : "memory")

// SHARED: 1 passes a 32-bit .shared address, 0 a generic 64-bit address.
#define ST_KERNEL(NAME, INS, NUM, SHARED)                                                         \
  __global__ void NAME(const int* perm, uint16_t* out) {                                          \
    __shared__ __align__(16) uint16_t s[kSlots * kSlotHalves];                                    \
    const int lane = threadIdx.x, w = blockIdx.x;                                                 \
    for (int i = lane; i < kSlots * kSlotHalves; i += 32) s[i] = kSentinel;                       \
    __syncthreads();                                                                              \
    uint16_t* row = s + perm[32 * w + lane] * kSlotHalves;                                        \
    uint32_t r[4];                                                                                \
    for (int i = 0; i < 4; ++i) r[i] = st_val(lane, i, 0) | ((uint32_t)st_val(lane, i, 1) << 16); \
    if (SHARED) {                                                                                 \
      const uint32_t a = (uint32_t)__cvta_generic_to_shared(row);                                 \
      ST_ASM_##NUM(INS, a, "r");                                                                  \
    } else {                                                                                      \
      ST_ASM_##NUM(INS, row, "l");                                                                \
    }                                                                                             \
    __syncthreads();                                                                              \
    for (int i = lane; i < kSlots * kSlotHalves; i += 32)                                         \
      out[w * kSlots * kSlotHalves + i] = s[i];                                                   \
  }

ST_KERNEL(k_st_x1, "stmatrix.sync.aligned.m8n8.x1.shared.b16", 1, 1)
ST_KERNEL(k_st_x2, "stmatrix.sync.aligned.m8n8.x2.shared::cta.b16", 2, 1)
ST_KERNEL(k_st_x4, "stmatrix.sync.aligned.m8n8.x4.shared.b16", 4, 1)
ST_KERNEL(k_st_x4_generic, "stmatrix.sync.aligned.m8n8.x4.b16", 4, 0)
ST_KERNEL(k_st_x1_trans, "stmatrix.sync.aligned.m8n8.x1.trans.shared.b16", 1, 1)
ST_KERNEL(k_st_x2_trans, "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16", 2, 1)
ST_KERNEL(k_st_x4_trans, "stmatrix.sync.aligned.m8n8.x4.trans.b16", 4, 0)

#define LD_ASM_1(INS, A, AC) asm volatile(INS " {%0}, [%1];" : "=r"(d[0]) : AC(A) : "memory")
#define LD_ASM_2(INS, A, AC) \
  asm volatile(INS " {%0, %1}, [%2];" : "=r"(d[0]), "=r"(d[1]) : AC(A) : "memory")
#define LD_ASM_4(INS, A, AC)                                    \
  asm volatile(INS " {%0, %1, %2, %3}, [%4];"                   \
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) \
               : AC(A)                                          \
               : "memory")

#define LD_KERNEL(NAME, INS, NUM, SHARED)                                                         \
  __global__ void NAME(const int* perm, const uint32_t* src, uint32_t* out) {                     \
    __shared__ __align__(16) uint32_t s[kSlots * kSlotWords];                                     \
    const int lane = threadIdx.x, w = blockIdx.x;                                                 \
    for (int i = lane; i < kSlots * kSlotWords; i += 32) s[i] = src[w * kSlots * kSlotWords + i]; \
    __syncthreads();                                                                              \
    const uint32_t* row = s + perm[32 * w + lane] * kSlotWords;                                   \
    uint32_t d[4] = {0, 0, 0, 0};                                                                 \
    if (SHARED) {                                                                                 \
      const uint32_t a = (uint32_t)__cvta_generic_to_shared(row);                                 \
      LD_ASM_##NUM(INS, a, "r");                                                                  \
    } else {                                                                                      \
      LD_ASM_##NUM(INS, row, "l");                                                                \
    }                                                                                             \
    for (int i = 0; i < 4; ++i) out[4 * (32 * w + lane) + i] = d[i];                              \
  }

LD_KERNEL(k_ld_x1, "ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4", 1, 1)
LD_KERNEL(k_ld_x2, "ldmatrix.sync.aligned.m8n16.x2.shared.s8.s4", 2, 1)
LD_KERNEL(k_ld_x4, "ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4", 4, 1)
LD_KERNEL(k_ld_x4_generic, "ldmatrix.sync.aligned.m8n16.x4.s8.s4", 4, 0)

// ---- host side ---------------------------------------------------------------------------------

// One permutation of the 32 slots per warp.
static std::vector<int> make_perm(uint64_t seed) {
  Rng r(seed);
  std::vector<int> p(32 * kWarps);
  for (int w = 0; w < kWarps; ++w) {
    for (int t = 0; t < 32; ++t) p[32 * w + t] = t;
    for (int t = 31; t > 0; --t) std::swap(p[32 * w + t], p[32 * w + (int)r.below(t + 1)]);
  }
  return p;
}

typedef void (*KSt)(const int*, uint16_t*);
typedef void (*KLd)(const int*, const uint32_t*, uint32_t*);

static void run_st(const char* name, KSt k, int num, bool trans, uint64_t seed) {
  const std::vector<int> perm = make_perm(seed);
  Dev<int> dp(perm);
  Dev<uint16_t> dout(kWarps * kSlots * kSlotHalves);
  k<<<kWarps, 32>>>(dp.p, dout.p);
  MISC_SYNC();
  const std::vector<uint16_t> out = dout.get();
  Item it(name);
  for (int w = 0; w < kWarps; ++w) {
    for (int t = 0; t < 32; ++t) {  // thread t supplied the address of row t % 8 of matrix t / 8
      const int slot = perm[32 * w + t], i = t / 8, rr = t % 8;
      const uint16_t* got = &out[(w * kSlots + slot) * kSlotHalves];
      for (int c = 0; c < kSlotHalves; ++c) {
        uint16_t want = kSentinel;
        if (i < num && c < 8) {
          // Non-trans: element (rr, c) came from lane 4 rr + c / 2, half c % 2.
          // Trans: element (rr, c) of the stored matrix is element (c, rr) of the fragment,
          // held by lane 4 c + rr / 2, half rr % 2.
          want = trans ? st_val(4 * c + rr / 2, i, rr % 2) : st_val(4 * rr + c / 2, i, c % 2);
        }
        it.check(got[c] == want, "warp=%d matrix=%d row=%d col=%d got=%04x want=%04x", w, i, rr, c,
                 got[c], want);
      }
    }
  }
}

static void run_ld(const char* name, KLd k, int num, uint64_t seed) {
  const std::vector<int> perm = make_perm(seed);
  Rng r(seed ^ 0x5A5A);
  std::vector<uint32_t> src(kWarps * kSlots * kSlotWords);
  for (uint32_t& x : src) x = r.u32();
  Dev<int> dp(perm);
  Dev<uint32_t> dsrc(src), dout(4 * 32 * kWarps);
  k<<<kWarps, 32>>>(dp.p, dsrc.p, dout.p);
  MISC_SYNC();
  const std::vector<uint32_t> out = dout.get();
  Item it(name);
  for (int w = 0; w < kWarps; ++w) {
    for (int lane = 0; lane < 32; ++lane) {
      const int g = lane >> 2, q = lane & 3;
      for (int i = 0; i < num; ++i) {
        const int slot = perm[32 * w + 8 * i + g];
        const uint8_t* row = (const uint8_t*)&src[(w * kSlots + slot) * kSlotWords];
        uint32_t want = 0;
        for (int e = 0; e < 4; ++e) {  // column 4 q + e
          const int col = 4 * q + e;
          const int nib = (row[col / 2] >> (4 * (col % 2))) & 0xF;
          const int v = nib >= 8 ? nib - 16 : nib;
          want |= (uint32_t)(uint8_t)(int8_t)v << (8 * e);
        }
        const uint32_t got = out[4 * (32 * w + lane) + i];
        it.check(got == want, "warp=%d lane=%d matrix=%d got=%08x want=%08x", w, lane, i, got,
                 want);
      }
    }
  }
}

int main() {
  run_st("misc.e2e.stmatrix.m8n8.x1.shared.b16", k_st_x1, 1, false, 0x5101);
  run_st("misc.e2e.stmatrix.m8n8.x2.shared::cta.b16", k_st_x2, 2, false, 0x5102);
  run_st("misc.e2e.stmatrix.m8n8.x4.shared.b16", k_st_x4, 4, false, 0x5104);
  run_st("misc.e2e.stmatrix.m8n8.x4.generic.b16", k_st_x4_generic, 4, false, 0x5105);
  run_st("misc.e2e.stmatrix.m8n8.x1.trans.shared.b16", k_st_x1_trans, 1, true, 0x5111);
  run_st("misc.e2e.stmatrix.m8n8.x2.trans.shared::cta.b16", k_st_x2_trans, 2, true, 0x5112);
  run_st("misc.e2e.stmatrix.m8n8.x4.trans.generic.b16", k_st_x4_trans, 4, true, 0x5114);
  run_ld("misc.e2e.ldmatrix.m8n16.x1.shared::cta.s8.s4", k_ld_x1, 1, 0x1D01);
  run_ld("misc.e2e.ldmatrix.m8n16.x2.shared.s8.s4", k_ld_x2, 2, 0x1D02);
  run_ld("misc.e2e.ldmatrix.m8n16.x4.shared::cta.s8.s4", k_ld_x4, 4, 0x1D04);
  run_ld("misc.e2e.ldmatrix.m8n16.x4.generic.s8.s4", k_ld_x4_generic, 4, 0x1D05);
  return finish();
}
