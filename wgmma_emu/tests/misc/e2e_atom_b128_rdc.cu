// End-to-end check that two -rdc translation units share one atom .b128 lock table (sm_90,
// fork-only). Built for sm_90a by tools/nvcc_emu with -rdc=true and e2e_atom_b128_rdc_tu2.cu as
// a second input (manifest "area:" argument).
//
// rules_misc rewrites atom .b128 into lib:wgemu_atom_b128 under a lock table that every
// translated module declares as .weak .global (Ctx.global_var); the device linker merges the
// declarations. Half of the warps run the CAS loop of this unit, the other half the one of
// tu2, on the same 64 counters (lo + 1, hi + 3 per thread, as in e2e_atom_b128.cu). Two
// tables would let both units succeed with the same old value: a repeated partial count, a
// torn (lo, hi) pair or a short final count.
// Premise, not tested: modules loaded separately (two programs) do not share the table, so
// atomicity holds only within one linked program; .sys scope is rejected by the translator.
#include "misc_e2e.h"

using namespace misc_e2e;

typedef unsigned long long u64;

constexpr int kNC = 64, kPer = 64, kThreads = 256, kNT = kNC * kPer, kBlocks = kNT / kThreads;

extern __device__ void tu2_cas128(void* p, u64 c0, u64 c1, u64 s0, u64 s1, u64* o);

__device__ __forceinline__ void tu1_cas128(void* p, u64 c0, u64 c1, u64 s0, u64 s1, u64* o) {
  asm volatile(
      "{\n\t.reg .b128 cmp, sw, old;\n\tmov.b128 cmp, {%2, %3};\n\tmov.b128 sw, {%4, %5};\n\t"
      "atom.global.cas.b128 old, [%6], cmp, sw;\n\tmov.b128 {%0, %1}, old;\n\t}"
      : "=l"(o[0]), "=l"(o[1])
      : "l"(c0), "l"(c1), "l"(s0), "l"(s1), "l"(p)
      : "memory");
}

__global__ void k_cas_two_units(u64* c, u64* won) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  u64* p = c + 2 * (t % kNC);
  const bool tu2 = (t / kNC) & 1;  // per counter: kPer / 2 contributors from each unit
  u64 lo = p[0], hi = p[1];
  for (;;) {
    u64 o[2];
    if (tu2)
      tu2_cas128(p, lo, hi, lo + 1, hi + 3, o);
    else
      tu1_cas128(p, lo, hi, lo + 1, hi + 3, o);
    if (o[0] == lo && o[1] == hi) break;
    lo = o[0];
    hi = o[1];
  }
  won[2 * t] = lo;
  won[2 * t + 1] = hi;
}

int main() {
  Dev<u64> c(2 * kNC), won(2 * kNT);
  k_cas_two_units<<<kBlocks, kThreads>>>(c.p, won.p);
  MISC_SYNC();
  Item it("misc.e2e.atom.global.cas.b128.two_rdc_units");
  const std::vector<u64> wv = won.get(), cv = c.get();
  for (int k = 0; k < kNC; ++k) {
    std::vector<int> seen(kPer, 0);
    bool ok = true;
    for (int t = k; t < kNT; t += kNC) {
      const u64 lo = wv[2 * t], hi = wv[2 * t + 1];
      if (hi != 3 * lo || lo >= (u64)kPer || seen[lo]++) ok = false;
    }
    it.check(ok, "counter=%d partial counts", k);
    it.check(cv[2 * k] == (u64)kPer && cv[2 * k + 1] == 3ull * kPer, "counter=%d got=%llx,%llx",
             k, cv[2 * k], cv[2 * k + 1]);
  }
  return finish();
}
