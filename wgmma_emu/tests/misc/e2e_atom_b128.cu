// End-to-end check of atom .cas.b128 / .exch.b128 (sm_90) rewritten by tools/ptxemu/rules_misc.py
// into lib:wgemu_atom_b128 (fork-only). Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Needs Ctx.global_var from T1 (module-scope lock table); kept in its own program so that a
// missing T1 piece fails only this test.
//
// cas: every thread increments a 128-bit counter (lo + 1, hi + 3) with a CAS loop. The value a
// successful CAS returns must be a partial count (i, 3 i): each i exactly once per counter, and
// hi == 3 lo proves that the two halves were updated as one atomic access. exch: every thread
// swaps in its own value; the returned values plus the final value must be exactly the
// contributed values plus the initial value.
#include <algorithm>
#include <utility>

#include "misc_e2e.h"

using namespace misc_e2e;

typedef unsigned long long u64;

constexpr int kNC = 64, kPer = 64, kThreads = 256, kNT = kNC * kPer, kBlocks = kNT / kThreads;

__device__ __forceinline__ void cas128(void* p, u64 c0, u64 c1, u64 s0, u64 s1, u64& o0, u64& o1) {
  asm volatile(
      "{\n\t.reg .b128 cmp, sw, old;\n\tmov.b128 cmp, {%2, %3};\n\tmov.b128 sw, {%4, %5};\n\t"
      "atom.global.cas.b128 old, [%6], cmp, sw;\n\tmov.b128 {%0, %1}, old;\n\t}"
      : "=l"(o0), "=l"(o1)
      : "l"(c0), "l"(c1), "l"(s0), "l"(s1), "l"(p)
      : "memory");
}

__global__ void k_cas_global(u64* c, u64* won) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  u64* p = c + 2 * (t % kNC);
  u64 lo = p[0], hi = p[1];  // any start value works: a stale one just fails the first CAS
  for (;;) {
    u64 o0, o1;
    cas128(p, lo, hi, lo + 1, hi + 3, o0, o1);
    if (o0 == lo && o1 == hi) break;
    lo = o0;
    hi = o1;
  }
  won[2 * t] = lo;
  won[2 * t + 1] = hi;
}

// Same through a .shared counter (one per block) and a generic address.
__global__ void k_cas_shared(u64* fin, u64* won) {
  __shared__ __align__(16) u64 s[2];
  if (threadIdx.x == 0) s[0] = s[1] = 0;
  __syncthreads();
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
  u64 lo = 0, hi = 0;
  for (;;) {
    u64 o0, o1;
    asm volatile(
        "{\n\t.reg .b128 cmp, sw, old;\n\tmov.b128 cmp, {%2, %3};\n\tmov.b128 sw, {%4, %5};\n\t"
        "atom.shared.cas.b128 old, [%6], cmp, sw;\n\tmov.b128 {%0, %1}, old;\n\t}"
        : "=l"(o0), "=l"(o1)
        : "l"(lo), "l"(hi), "l"(lo + 1), "l"(hi + 3), "r"(a)
        : "memory");
    if (o0 == lo && o1 == hi) break;
    lo = o0;
    hi = o1;
  }
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  won[2 * t] = lo;
  won[2 * t + 1] = hi;
  __syncthreads();
  if (threadIdx.x == 0) {
    fin[2 * blockIdx.x] = s[0];
    fin[2 * blockIdx.x + 1] = s[1];
  }
}

__global__ void k_exch_generic(u64* c, u64* old) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const u64 v0 = 0x1000000000ull + (u64)t, v1 = ~(u64)t;
  u64 o0, o1;
  asm volatile(
      "{\n\t.reg .b128 v, old;\n\tmov.b128 v, {%2, %3};\n\t"
      "atom.exch.b128 old, [%4], v;\n\tmov.b128 {%0, %1}, old;\n\t}"
      : "=l"(o0), "=l"(o1)
      : "l"(v0), "l"(v1), "l"(c + 2 * (t % kNC))
      : "memory");
  old[2 * t] = o0;
  old[2 * t + 1] = o1;
}

// won: pairs (lo, hi) of nt threads; thread t used counter t % nc. Checks the partial counts.
static void check_won(Item& it, const std::vector<u64>& won, int nc, int per) {
  const int nt = (int)(won.size() / 2);
  for (int k = 0; k < nc; ++k) {
    std::vector<int> seen(per, 0);
    bool ok = true;
    for (int t = k; t < nt; t += nc) {
      const u64 lo = won[2 * t], hi = won[2 * t + 1];
      if (hi != 3 * lo || lo >= (u64)per || seen[lo]++) ok = false;
    }
    it.check(ok, "counter=%d", k);
  }
}

int main() {
  {
    Dev<u64> c(2 * kNC), won(2 * kNT);
    k_cas_global<<<kBlocks, kThreads>>>(c.p, won.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.global.cas.b128");
    check_won(it, won.get(), kNC, kPer);
    const std::vector<u64> cv = c.get();
    for (int k = 0; k < kNC; ++k) {
      it.check(cv[2 * k] == (u64)kPer && cv[2 * k + 1] == 3ull * kPer, "counter=%d got=%llx,%llx",
               k, cv[2 * k], cv[2 * k + 1]);
    }
  }
  {
    const int blocks = 16;
    Dev<u64> fin(2 * blocks), won(2 * blocks * kThreads);
    k_cas_shared<<<blocks, kThreads>>>(fin.p, won.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.shared.cas.b128");
    const std::vector<u64> wv = won.get(), fv = fin.get();
    for (int b = 0; b < blocks; ++b) {
      const std::vector<u64> blk(wv.begin() + 2 * b * kThreads,
                                 wv.begin() + 2 * (b + 1) * kThreads);
      check_won(it, blk, 1, kThreads);
      it.check(fv[2 * b] == (u64)kThreads && fv[2 * b + 1] == 3ull * kThreads,
               "block=%d got=%llx,%llx", b, fv[2 * b], fv[2 * b + 1]);
    }
  }
  {
    std::vector<u64> init(2 * kNC);
    for (int k = 0; k < kNC; ++k) {
      init[2 * k] = 0xABC0000000ull + (u64)k;
      init[2 * k + 1] = 0x5A5A5A5A5A5A5A5Aull;
    }
    Dev<u64> c(init), old(2 * kNT);
    k_exch_generic<<<kBlocks, kThreads>>>(c.p, old.p);
    MISC_SYNC();
    Item it("misc.e2e.atom.exch.b128");
    const std::vector<u64> ov = old.get(), cv = c.get();
    for (int k = 0; k < kNC; ++k) {
      // returned values + final == contributed values + initial (as multisets of pairs)
      std::vector<std::pair<u64, u64>> got, want;
      for (int t = k; t < kNT; t += kNC) {
        got.emplace_back(ov[2 * t], ov[2 * t + 1]);
        want.emplace_back(0x1000000000ull + (u64)t, ~(u64)t);
      }
      got.emplace_back(cv[2 * k], cv[2 * k + 1]);
      want.emplace_back(init[2 * k], init[2 * k + 1]);
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      it.check(got == want, "counter=%d", k);
    }
  }
  return finish();
}
