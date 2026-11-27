// End-to-end check of .attribute(.unified(...)) variables and the ld address operand qualifier
// .unified (sm_90, ptx.txt 3115 / 13175). Fork-only. Owner: T2. Built for sm_90a by
// tools/nvcc_emu.
//
// WGEMU-COVERS-FEATURE: ld_unified_operand
//
// The core (T1) removes .attribute(.unified(...)) from the declaration; rules_misc drops the
// .unified suffix of the ld address operand. A .unified variable is read-only and must be
// loaded with that qualifier, so the test only loads its initial value.
// [inference] CUDA has no source attribute for .unified, so the variable is declared inside the
// inline asm block (function scope); kept in its own program so that a front-end or ptxas
// rejection fails only this test.
#include "misc_e2e.h"

using namespace misc_e2e;

__global__ void k_unified(uint32_t* out) {
  uint32_t v, w;
  asm volatile(
      "{\n\t.global .attribute(.unified(0x1, 0x2)) .u32 wgemu_u = 0x2468ACE1;\n\t"
      ".reg .u64 a;\n\t"
      "ld.global.u32 %0, [wgemu_u].unified;\n\t"
      "mov.u64 a, wgemu_u;\n\t"
      "ld.global.u32 %1, [a].unified;\n\t}"
      : "=r"(v), "=r"(w));
  out[2 * threadIdx.x] = v;
  out[2 * threadIdx.x + 1] = w;
}

int main() {
  Dev<uint32_t> out(2 * 64);
  k_unified<<<1, 64>>>(out.p);
  MISC_SYNC();
  const std::vector<uint32_t> o = out.get();
  Item it("misc.e2e.ld.unified");
  for (int t = 0; t < 64; ++t) {
    it.check(o[2 * t] == 0x2468ACE1u && o[2 * t + 1] == 0x2468ACE1u, "thread=%d got=%08x,%08x", t,
             o[2 * t], o[2 * t + 1]);
  }
  return finish();
}
