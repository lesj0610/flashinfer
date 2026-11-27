// Second translation unit of misc.atom_b128_rdc; see e2e_atom_b128_rdc.cu. Built with -rdc=true
// as an extra nvcc input (manifest "area:" argument). Fork-only.
typedef unsigned long long u64;

// Not inlined into the other unit (no LTO): its atom .b128 is translated in this module, with
// this module's .weak lock table declaration.
__device__ void tu2_cas128(void* p, u64 c0, u64 c1, u64 s0, u64 s1, u64* o) {
  asm volatile(
      "{\n\t.reg .b128 cmp, sw, old;\n\tmov.b128 cmp, {%2, %3};\n\tmov.b128 sw, {%4, %5};\n\t"
      "atom.global.cas.b128 old, [%6], cmp, sw;\n\tmov.b128 {%0, %1}, old;\n\t}"
      : "=l"(o[0]), "=l"(o[1])
      : "l"(c0), "l"(c1), "l"(s0), "l"(s1), "l"(p)
      : "memory");
}
