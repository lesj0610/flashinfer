// End-to-end check of the sm_89 / sm_90 cvt forms rewritten by tools/ptxemu/rules_misc.py
// (fork-only). Owner: T2. Built for sm_90a by tools/nvcc_emu; run on SM80.
//
// CPU expectations: include/wgmma_emu/cvt.h (checked bit for bit against the exact reference
// tests/misc/ref_cvt.py by tests/misc/test_cvt_cpu.py) and, for the bf16-source forms, exact
// double arithmetic. Library-backed results are compared bit for bit; results the rules
// produce with a native SM80 instruction accept any NaN where PTX does not fix the encoding.
//
// Integer destinations narrower than the register are sign- / zero-extended (ptx.txt 14687-14690),
// so all 16 bits of an s8/u8 result are compared. tf32 results of a native conversion are
// compared on the upper 19 bits only: the tf32 layout is implementation defined (ptx.txt 2404)
// and mma reads those bits.
//
// [inference] cvt .sat to bf16: NaN and any negative value (including -0) give +0 (same rule as
// runtime/lib_misc.cu bf16_sat). [inference] .relu on a -0 input: PTX says "clamps the result
// to 0 if negative" without saying whether -0 counts, so native .relu results accept either zero
// where +0 is expected.
#include <math.h>

#include "misc_e2e.h"
#include "wgmma_emu/cvt.h"

using namespace misc_e2e;
namespace c = wgemu::cvt;

typedef unsigned short u16;
typedef unsigned long long u64;

#define K1(NAME, TD, CD, TA, CA, INS)                                         \
  __global__ void NAME(const TA* __restrict__ a, TD* __restrict__ d, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                      \
    if (i < n) {                                                              \
      TD r;                                                                   \
      asm volatile(INS " %0, %1;" : "=" CD(r) : CA(a[i]));                    \
      d[i] = r;                                                               \
    }                                                                         \
  }

#define K2(NAME, TD, CD, TA, CA, INS)                                                          \
  __global__ void NAME(const TA* __restrict__ a, const TA* __restrict__ b, TD* __restrict__ d, \
                       int n) {                                                                \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                                       \
    if (i < n) {                                                                               \
      TD r;                                                                                    \
      asm volatile(INS " %0, %1, %2;" : "=" CD(r) : CA(a[i]), CA(b[i]));                       \
      d[i] = r;                                                                                \
    }                                                                                          \
  }

// ---- fp8 (lib:wgemu_cvt_*) -------------------------------------------------------------------
K2(k_e4m3x2_f32, u16, "h", uint32_t, "r", "cvt.rn.satfinite.e4m3x2.f32")
K2(k_e5m2x2_f32, u16, "h", uint32_t, "r", "cvt.rn.satfinite.e5m2x2.f32")
K2(k_e4m3x2_f32_relu, u16, "h", uint32_t, "r", "cvt.rn.satfinite.relu.e4m3x2.f32")
K2(k_e5m2x2_f32_relu, u16, "h", uint32_t, "r", "cvt.rn.satfinite.relu.e5m2x2.f32")
K1(k_e4m3x2_f16x2, u16, "h", uint32_t, "r", "cvt.rn.satfinite.e4m3x2.f16x2")
K1(k_e5m2x2_f16x2, u16, "h", uint32_t, "r", "cvt.rn.satfinite.e5m2x2.f16x2")
K1(k_e4m3x2_f16x2_relu, u16, "h", uint32_t, "r", "cvt.rn.satfinite.relu.e4m3x2.f16x2")
K1(k_e5m2x2_f16x2_relu, u16, "h", uint32_t, "r", "cvt.rn.satfinite.relu.e5m2x2.f16x2")
K1(k_f16x2_e4m3x2, uint32_t, "r", u16, "h", "cvt.rn.f16x2.e4m3x2")
K1(k_f16x2_e5m2x2, uint32_t, "r", u16, "h", "cvt.rn.f16x2.e5m2x2")
K1(k_f16x2_e4m3x2_relu, uint32_t, "r", u16, "h", "cvt.rn.relu.f16x2.e4m3x2")
K1(k_f16x2_e5m2x2_relu, uint32_t, "r", u16, "h", "cvt.rn.relu.f16x2.e5m2x2")

// ---- tf32 .rn / .rz (lib:wgemu_cvt_f32_to_tf32) --------------------------------------------
K1(k_tf32_rn, uint32_t, "r", uint32_t, "r", "cvt.rn.tf32.f32")
K1(k_tf32_rz, uint32_t, "r", uint32_t, "r", "cvt.rz.tf32.f32")
K1(k_tf32_rn_relu, uint32_t, "r", uint32_t, "r", "cvt.rn.relu.tf32.f32")
K1(k_tf32_rz_relu, uint32_t, "r", uint32_t, "r", "cvt.rz.relu.tf32.f32")

// ---- bf16 destinations (lib:wgemu_cvt_to_bf16) ---------------------------------------------
K1(k_bf16_f16_rn, u16, "h", u16, "h", "cvt.rn.bf16.f16")
K1(k_bf16_f16_rz, u16, "h", u16, "h", "cvt.rz.bf16.f16")
K1(k_bf16_f16_rm, u16, "h", u16, "h", "cvt.rm.bf16.f16")
K1(k_bf16_f16_rp, u16, "h", u16, "h", "cvt.rp.bf16.f16")
K1(k_bf16_f16_rn_sat, u16, "h", u16, "h", "cvt.rn.sat.bf16.f16")
K1(k_bf16_f64_rn, u16, "h", u64, "l", "cvt.rn.bf16.f64")
K1(k_bf16_f64_rz, u16, "h", u64, "l", "cvt.rz.bf16.f64")
K1(k_bf16_f64_rm, u16, "h", u64, "l", "cvt.rm.bf16.f64")
K1(k_bf16_f64_rp, u16, "h", u64, "l", "cvt.rp.bf16.f64")
K1(k_bf16_u8_rn, u16, "h", u16, "h", "cvt.rn.bf16.u8")
K1(k_bf16_s8_rz, u16, "h", u16, "h", "cvt.rz.bf16.s8")
K1(k_bf16_u16_rn, u16, "h", u16, "h", "cvt.rn.bf16.u16")
K1(k_bf16_s16_rm, u16, "h", u16, "h", "cvt.rm.bf16.s16")
K1(k_bf16_u32_rp, u16, "h", uint32_t, "r", "cvt.rp.bf16.u32")
K1(k_bf16_s32_rn, u16, "h", uint32_t, "r", "cvt.rn.bf16.s32")
K1(k_bf16_u64_rz, u16, "h", u64, "l", "cvt.rz.bf16.u64")
K1(k_bf16_s64_rn, u16, "h", u64, "l", "cvt.rn.bf16.s64")
K1(k_bf16_f32_rm, u16, "h", uint32_t, "r", "cvt.rm.bf16.f32")
K1(k_bf16_f32_rp, u16, "h", uint32_t, "r", "cvt.rp.bf16.f32")
K1(k_bf16_f32_rm_ftz, u16, "h", uint32_t, "r", "cvt.rm.ftz.bf16.f32")

// ---- .satfinite with f32 sources (native conversion + inf fix) -----------------------------
K1(k_f16_f32_rn_satfin, u16, "h", uint32_t, "r", "cvt.rn.satfinite.f16.f32")
K1(k_bf16_f32_rz_satfin, u16, "h", uint32_t, "r", "cvt.rz.satfinite.bf16.f32")
K2(k_bf16x2_f32_rn_relu_satfin, uint32_t, "r", uint32_t, "r", "cvt.rn.relu.satfinite.bf16x2.f32")
K2(k_f16x2_f32_rn_satfin, uint32_t, "r", uint32_t, "r", "cvt.rn.satfinite.f16x2.f32")
K1(k_tf32_rna_satfin, uint32_t, "r", uint32_t, "r", "cvt.rna.satfinite.tf32.f32")

// ---- bf16 sources (f32 round trip + native conversion) -------------------------------------
K1(k_bf16_bf16_rni, u16, "h", u16, "h", "cvt.rni.bf16.bf16")
K1(k_bf16_bf16_rzi, u16, "h", u16, "h", "cvt.rzi.bf16.bf16")
K1(k_bf16_bf16_rmi, u16, "h", u16, "h", "cvt.rmi.bf16.bf16")
K1(k_bf16_bf16_rpi, u16, "h", u16, "h", "cvt.rpi.bf16.bf16")
K1(k_s32_bf16_rzi, uint32_t, "r", u16, "h", "cvt.rzi.s32.bf16")
K1(k_u32_bf16_rni, uint32_t, "r", u16, "h", "cvt.rni.u32.bf16")
K1(k_s64_bf16_rmi, u64, "l", u16, "h", "cvt.rmi.s64.bf16")
K1(k_u64_bf16_rpi, u64, "l", u16, "h", "cvt.rpi.u64.bf16")
K1(k_s16_bf16_rni, u16, "h", u16, "h", "cvt.rni.s16.bf16")
K1(k_u16_bf16_rzi, u16, "h", u16, "h", "cvt.rzi.u16.bf16")
K1(k_s8_bf16_rni, u16, "h", u16, "h", "cvt.rni.s8.bf16")
K1(k_u8_bf16_rpi, u16, "h", u16, "h", "cvt.rpi.u8.bf16")
K1(k_f16_bf16_rn, u16, "h", u16, "h", "cvt.rn.f16.bf16")
K1(k_f16_bf16_rz, u16, "h", u16, "h", "cvt.rz.f16.bf16")
K1(k_f16_bf16_rm, u16, "h", u16, "h", "cvt.rm.f16.bf16")
K1(k_f64_bf16, u64, "l", u16, "h", "cvt.f64.bf16")
K1(k_f32_bf16_ftz, uint32_t, "r", u16, "h", "cvt.ftz.f32.bf16")

// ---------------------------------------------------------------------------------------------

template <class TD, class TA>
std::vector<TD> run1(void (*k)(const TA*, TD*, int), const std::vector<TA>& a) {
  Dev<TA> da(a);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}

template <class TD, class TA>
std::vector<TD> run2(void (*k)(const TA*, const TA*, TD*, int), const std::vector<TA>& a,
                     const std::vector<TA>& b) {
  Dev<TA> da(a), db(b);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, db.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}

// Compare helpers: name, inputs (printed as hex), results, expectation, equality.
template <class TD, class TA, class Want, class Eq>
void check1(const char* name, void (*k)(const TA*, TD*, int), const std::vector<TA>& a, Want want,
            Eq eq) {
  const std::vector<TD> got = run1(k, a);
  Item it(name);
  for (size_t i = 0; i < a.size(); ++i) {
    const u64 w = (u64)want(a[i]);
    it.check(eq((u64)got[i], w), "in=%llx got=%llx want=%llx", (u64)a[i], (u64)got[i], w);
  }
}

template <class TD, class TA, class Want, class Eq>
void check2(const char* name, void (*k)(const TA*, const TA*, TD*, int), const std::vector<TA>& a,
            const std::vector<TA>& b, Want want, Eq eq) {
  const std::vector<TD> got = run2(k, a, b);
  Item it(name);
  for (size_t i = 0; i < a.size(); ++i) {
    const u64 w = (u64)want(a[i], b[i]);
    it.check(eq((u64)got[i], w), "in=%llx,%llx got=%llx want=%llx", (u64)a[i], (u64)b[i],
             (u64)got[i], w);
  }
}

static bool eq_exact(u64 g, u64 w) { return g == w; }
static bool eq16(u64 g, u64 w) { return (g & 0xFFFF) == (w & 0xFFFF); }
static bool eq_f16_nan(u64 g, u64 w) { return same_f16((uint32_t)g, (uint32_t)w, true); }
static bool eq_bf16_nan(u64 g, u64 w) { return same_bf16((uint32_t)g, (uint32_t)w, true); }
static bool eq_f32_nan(u64 g, u64 w) { return same32((uint32_t)g, (uint32_t)w, true); }
static bool eq_f64_nan(u64 g, u64 w) { return same64(g, w, true); }
static bool eq_x2_nan16(u64 g, u64 w, bool bf) {
  const auto s = bf ? same_bf16 : same_f16;
  return s((uint32_t)g, (uint32_t)w, true) && s((uint32_t)(g >> 16), (uint32_t)(w >> 16), true);
}
static bool eq_bf16x2_nan(u64 g, u64 w) { return eq_x2_nan16(g, w, true); }
// Native .relu: a half expected as +0 may come back as -0 (see the header comment).
static bool eq_bf16x2_relu(u64 g, u64 w) {
  for (int s = 0; s < 32; s += 16) {
    const uint32_t gh = (uint32_t)(g >> s) & 0xFFFFu, wh = (uint32_t)(w >> s) & 0xFFFFu;
    if (!(same_bf16(gh, wh, true) || (wh == 0u && gh == 0x8000u))) return false;
  }
  return true;
}
// Native tf32: NaN class, else the upper 19 bits (sign, exponent, 10 mantissa bits).
static bool eq_tf32_native(u64 g, u64 w) {
  const uint32_t gu = (uint32_t)g, wu = (uint32_t)w;
  if (nan32(gu) || nan32(wu)) return nan32(gu) && nan32(wu);
  return (gu & 0xFFFFE000u) == (wu & 0xFFFFE000u);
}
static bool eq_f16x2_nan(u64 g, u64 w) { return eq_x2_nan16(g, w, false); }

// cvt .sat to a bf16 destination: clamp to [0, 1], NaN and negatives (incl. -0) -> +0.
static uint32_t sat_bf16(uint32_t h) {
  h &= 0xFFFFu;
  if (nan_bf16(h) || (h & 0x8000u)) return 0u;
  return h > 0x3F80u ? 0x3F80u : h;
}

// Exact value of a bf16 as double.
static double bf16_val(uint32_t b) { return (double)u2f((b & 0xFFFFu) << 16); }

// PTX integer rounding of an exact value: 0 rni (ties to even), 1 rzi, 2 rmi, 3 rpi.
static double round_int(double x, int mode) {
  switch (mode) {
    case 0:
      return nearbyint(x);  // default FE_TONEAREST
    case 1:
      return trunc(x);
    case 2:
      return floor(x);
    default:
      return ceil(x);
  }
}

// float-to-integer cvt: saturate to the destination range. NaN (ptx.txt 14596-14599): 0 unless
// the destination is .s64 / .u64 (or the source .f64), then 1 << (bits - 1).
static u64 to_int(uint32_t b, int mode, int bits, bool is_signed) {
  if (nan_bf16(b)) return bits == 64 ? 1ull << 63 : 0;
  const double y = round_int(bf16_val(b), mode);
  const double lim = ldexp(1.0, is_signed ? bits - 1 : bits);  // exclusive upper bound
  if (is_signed) {
    if (y >= lim) return (1ull << (bits - 1)) - 1ull;
    if (y <= -lim) return 0ull - (1ull << (bits - 1));
    return (u64)(long long)y;
  }
  if (y <= 0) return 0;
  if (y >= lim) return bits == 64 ? ~0ull : (1ull << bits) - 1ull;
  return (u64)y;
}

static uint32_t bf16_round_int(uint32_t b, int mode) {
  if (nan_bf16(b)) return 0x7FFFu;  // compared as a NaN class
  const double y = round_int(bf16_val(b), mode);
  return c::f32_to_bf16(f2u((float)y), c::RN);  // exact: y is a bf16 value
}

int main() {
  // ---------------------------------------------------------------- inputs
  const std::vector<uint32_t> f32 = f32_inputs(8192, 0xC0FFEE);
  std::vector<uint32_t> f32b(f32.size());
  for (size_t i = 0; i < f32.size(); ++i) f32b[i] = f32[(i * 7919 + 13) % f32.size()];
  std::vector<u16> h16(1 << 16);
  for (int i = 0; i < (1 << 16); ++i) h16[i] = (u16)i;
  std::vector<uint32_t> h16x2(1 << 16);
  for (uint32_t i = 0; i < (1u << 16); ++i) h16x2[i] = i | (((i * 40503u) & 0xFFFFu) << 16);
  std::vector<u64> f64;
  {
    Rng r(0xF64);
    for (u64 b : {0ull, 1ull << 63, 1ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
                  0x7FF8000000000000ull, 0x7FEFFFFFFFFFFFFFull, 0x3FF0000000000000ull,
                  0x47EFFFFFE0000000ull, 0x47EFFFFFFFFFFFFFull, 0x380FFFFFFFFFFFFFull})
      f64.push_back(b);
    for (uint32_t b = 0; b < 0x10000u; b += 3) {  // bf16 values, midpoints and neighbours
      const u64 v = d2u(bf16_val(b));
      const u64 mid = v + (1ull << 44);  // + half a bf16 quantum (normal range)
      f64.push_back(v);
      f64.push_back(mid);
      f64.push_back(mid - 1);
      f64.push_back(mid + 1);
    }
    for (int i = 0; i < 8192; ++i) f64.push_back(r.next());
  }
  std::vector<uint32_t> i32;
  std::vector<u64> i64;
  {
    Rng r(0x1234);
    for (int k = 0; k < 64; ++k) {
      for (long long d : {-1ll, 0ll, 1ll}) {
        i64.push_back((1ull << k) + d);
        i64.push_back(0ull - ((1ull << k) + d));
        if (k >= 8) i64.push_back((1ull << k) + (1ull << (k - 8)) + d);  // bf16 ties
      }
    }
    for (int i = 0; i < 8192; ++i) i64.push_back(r.next() >> r.below(64));
    for (u64 v : i64) i32.push_back((uint32_t)v);
  }

  // ---------------------------------------------------------------- fp8
  struct Fp8F32 {
    const char* name;
    void (*k)(const uint32_t*, const uint32_t*, u16*, int);
    bool e5, relu;
  };
  for (const Fp8F32& t : {Fp8F32{"misc.e2e.cvt.e4m3x2.f32", k_e4m3x2_f32, false, false},
                          Fp8F32{"misc.e2e.cvt.e5m2x2.f32", k_e5m2x2_f32, true, false},
                          Fp8F32{"misc.e2e.cvt.relu.e4m3x2.f32", k_e4m3x2_f32_relu, false, true},
                          Fp8F32{"misc.e2e.cvt.relu.e5m2x2.f32", k_e5m2x2_f32_relu, true, true}}) {
    check2(
        t.name, t.k, f32, f32b,
        [&](uint32_t a, uint32_t b) { return c::f32x2_to_fp8x2(a, b, t.e5, t.relu, true); }, eq16);
  }
  struct Fp8F16 {
    const char* name;
    void (*k)(const uint32_t*, u16*, int);
    bool e5, relu;
  };
  for (const Fp8F16& t :
       {Fp8F16{"misc.e2e.cvt.e4m3x2.f16x2", k_e4m3x2_f16x2, false, false},
        Fp8F16{"misc.e2e.cvt.e5m2x2.f16x2", k_e5m2x2_f16x2, true, false},
        Fp8F16{"misc.e2e.cvt.relu.e4m3x2.f16x2", k_e4m3x2_f16x2_relu, false, true},
        Fp8F16{"misc.e2e.cvt.relu.e5m2x2.f16x2", k_e5m2x2_f16x2_relu, true, true}}) {
    check1(
        t.name, t.k, h16x2, [&](uint32_t a) { return c::f16x2_to_fp8x2(a, t.e5, t.relu, true); },
        eq16);
  }
  struct F16Fp8 {
    const char* name;
    void (*k)(const u16*, uint32_t*, int);
    bool e5, relu;
  };
  for (const F16Fp8& t :
       {F16Fp8{"misc.e2e.cvt.f16x2.e4m3x2", k_f16x2_e4m3x2, false, false},
        F16Fp8{"misc.e2e.cvt.f16x2.e5m2x2", k_f16x2_e5m2x2, true, false},
        F16Fp8{"misc.e2e.cvt.relu.f16x2.e4m3x2", k_f16x2_e4m3x2_relu, false, true},
        F16Fp8{"misc.e2e.cvt.relu.f16x2.e5m2x2", k_f16x2_e5m2x2_relu, true, true}}) {
    check1(t.name, t.k, h16, [&](u16 a) { return c::fp8x2_to_f16x2(a, t.e5, t.relu); }, eq_exact);
  }

  // ---------------------------------------------------------------- tf32
  struct Tf32 {
    const char* name;
    void (*k)(const uint32_t*, uint32_t*, int);
    c::Rnd r;
    bool relu;
  };
  for (const Tf32& t : {Tf32{"misc.e2e.cvt.rn.tf32.f32", k_tf32_rn, c::RN, false},
                        Tf32{"misc.e2e.cvt.rz.tf32.f32", k_tf32_rz, c::RZ, false},
                        Tf32{"misc.e2e.cvt.rn.relu.tf32.f32", k_tf32_rn_relu, c::RN, true},
                        Tf32{"misc.e2e.cvt.rz.relu.tf32.f32", k_tf32_rz_relu, c::RZ, true}}) {
    check1(
        t.name, t.k, f32, [&](uint32_t x) { return c::f32_to_tf32(x, t.r, t.relu, false); },
        eq_exact);
  }

  // ---------------------------------------------------------------- bf16 destinations
  struct F16Src {
    const char* name;
    void (*k)(const u16*, u16*, int);
    c::Rnd r;
  };
  for (const F16Src& t : {F16Src{"misc.e2e.cvt.rn.bf16.f16", k_bf16_f16_rn, c::RN},
                          F16Src{"misc.e2e.cvt.rz.bf16.f16", k_bf16_f16_rz, c::RZ},
                          F16Src{"misc.e2e.cvt.rm.bf16.f16", k_bf16_f16_rm, c::RM},
                          F16Src{"misc.e2e.cvt.rp.bf16.f16", k_bf16_f16_rp, c::RP}}) {
    check1(t.name, t.k, h16, [&](u16 h) { return c::f16_to_bf16(h, t.r); }, eq16);
  }
  check1(
      "misc.e2e.cvt.rn.sat.bf16.f16", k_bf16_f16_rn_sat, h16,
      [](u16 h) { return sat_bf16(c::f16_to_bf16(h, c::RN)); }, eq16);
  struct F64Src {
    const char* name;
    void (*k)(const u64*, u16*, int);
    c::Rnd r;
  };
  for (const F64Src& t : {F64Src{"misc.e2e.cvt.rn.bf16.f64", k_bf16_f64_rn, c::RN},
                          F64Src{"misc.e2e.cvt.rz.bf16.f64", k_bf16_f64_rz, c::RZ},
                          F64Src{"misc.e2e.cvt.rm.bf16.f64", k_bf16_f64_rm, c::RM},
                          F64Src{"misc.e2e.cvt.rp.bf16.f64", k_bf16_f64_rp, c::RP}}) {
    check1(t.name, t.k, f64, [&](u64 d) { return c::f64_to_bf16(d, t.r); }, eq16);
  }
  check1(
      "misc.e2e.cvt.rn.bf16.u8", k_bf16_u8_rn, h16,
      [](u16 v) { return c::int_to_bf16(v, false, 8, c::RN); }, eq16);
  check1(
      "misc.e2e.cvt.rz.bf16.s8", k_bf16_s8_rz, h16,
      [](u16 v) { return c::int_to_bf16(v, true, 8, c::RZ); }, eq16);
  check1(
      "misc.e2e.cvt.rn.bf16.u16", k_bf16_u16_rn, h16,
      [](u16 v) { return c::int_to_bf16(v, false, 16, c::RN); }, eq16);
  check1(
      "misc.e2e.cvt.rm.bf16.s16", k_bf16_s16_rm, h16,
      [](u16 v) { return c::int_to_bf16(v, true, 16, c::RM); }, eq16);
  check1(
      "misc.e2e.cvt.rp.bf16.u32", k_bf16_u32_rp, i32,
      [](uint32_t v) { return c::int_to_bf16(v, false, 32, c::RP); }, eq16);
  check1(
      "misc.e2e.cvt.rn.bf16.s32", k_bf16_s32_rn, i32,
      [](uint32_t v) { return c::int_to_bf16(v, true, 32, c::RN); }, eq16);
  check1(
      "misc.e2e.cvt.rz.bf16.u64", k_bf16_u64_rz, i64,
      [](u64 v) { return c::int_to_bf16(v, false, 64, c::RZ); }, eq16);
  check1(
      "misc.e2e.cvt.rn.bf16.s64", k_bf16_s64_rn, i64,
      [](u64 v) { return c::int_to_bf16(v, true, 64, c::RN); }, eq16);
  check1(
      "misc.e2e.cvt.rm.bf16.f32", k_bf16_f32_rm, f32,
      [](uint32_t x) { return c::f32_to_bf16(x, c::RM); }, eq_bf16_nan);
  check1(
      "misc.e2e.cvt.rp.bf16.f32", k_bf16_f32_rp, f32,
      [](uint32_t x) { return c::f32_to_bf16(x, c::RP); }, eq_bf16_nan);
  check1(
      "misc.e2e.cvt.rm.ftz.bf16.f32", k_bf16_f32_rm_ftz, f32,
      [](uint32_t x) { return c::f32_to_bf16(c::f32_ftz(x), c::RM); }, eq_bf16_nan);

  // ---------------------------------------------------------------- .satfinite
  check1(
      "misc.e2e.cvt.rn.satfinite.f16.f32", k_f16_f32_rn_satfin, f32,
      [](uint32_t x) { return c::convert_dec(c::decode_f32(x), c::fmt_f16(), c::RN, false, true); },
      eq_f16_nan);
  check1(
      "misc.e2e.cvt.rz.satfinite.bf16.f32", k_bf16_f32_rz_satfin, f32,
      [](uint32_t x) {
        return c::convert_dec(c::decode_f32(x), c::fmt_bf16(), c::RZ, false, true);
      },
      eq_bf16_nan);
  check2(
      "misc.e2e.cvt.rn.relu.satfinite.bf16x2.f32", k_bf16x2_f32_rn_relu_satfin, f32, f32b,
      [](uint32_t a, uint32_t b) {
        const auto one = [](uint32_t x) {
          return c::convert_dec(c::decode_f32(x), c::fmt_bf16(), c::RN, true, true);
        };
        return (one(a) << 16) | one(b);  // a goes to the upper half
      },
      eq_bf16x2_relu);
  check2(
      "misc.e2e.cvt.rn.satfinite.f16x2.f32", k_f16x2_f32_rn_satfin, f32, f32b,
      [](uint32_t a, uint32_t b) {
        const auto one = [](uint32_t x) {
          return c::convert_dec(c::decode_f32(x), c::fmt_f16(), c::RN, false, true);
        };
        return (one(a) << 16) | one(b);
      },
      eq_f16x2_nan);
  check1(
      "misc.e2e.cvt.rna.satfinite.tf32.f32", k_tf32_rna_satfin, f32,
      [](uint32_t x) { return c::f32_to_tf32(x, c::RNA, false, true); }, eq_tf32_native);

  // ---------------------------------------------------------------- bf16 sources
  struct RoundBf16 {
    const char* name;
    void (*k)(const u16*, u16*, int);
    int mode;
  };
  for (const RoundBf16& t : {RoundBf16{"misc.e2e.cvt.rni.bf16.bf16", k_bf16_bf16_rni, 0},
                             RoundBf16{"misc.e2e.cvt.rzi.bf16.bf16", k_bf16_bf16_rzi, 1},
                             RoundBf16{"misc.e2e.cvt.rmi.bf16.bf16", k_bf16_bf16_rmi, 2},
                             RoundBf16{"misc.e2e.cvt.rpi.bf16.bf16", k_bf16_bf16_rpi, 3}}) {
    check1(t.name, t.k, h16, [&](u16 b) { return bf16_round_int(b, t.mode); }, eq_bf16_nan);
  }
  check1(
      "misc.e2e.cvt.rzi.s32.bf16", k_s32_bf16_rzi, h16,
      [](u16 b) { return (uint32_t)to_int(b, 1, 32, true); }, eq_exact);
  check1(
      "misc.e2e.cvt.rni.u32.bf16", k_u32_bf16_rni, h16,
      [](u16 b) { return (uint32_t)to_int(b, 0, 32, false); }, eq_exact);
  check1(
      "misc.e2e.cvt.rmi.s64.bf16", k_s64_bf16_rmi, h16,
      [](u16 b) { return to_int(b, 2, 64, true); }, eq_exact);
  check1(
      "misc.e2e.cvt.rpi.u64.bf16", k_u64_bf16_rpi, h16,
      [](u16 b) { return to_int(b, 3, 64, false); }, eq_exact);
  check1(
      "misc.e2e.cvt.rni.s16.bf16", k_s16_bf16_rni, h16,
      [](u16 b) { return to_int(b, 0, 16, true); }, eq16);
  check1(
      "misc.e2e.cvt.rzi.u16.bf16", k_u16_bf16_rzi, h16,
      [](u16 b) { return to_int(b, 1, 16, false); }, eq16);
  check1(
      "misc.e2e.cvt.rni.s8.bf16", k_s8_bf16_rni, h16, [](u16 b) { return to_int(b, 0, 8, true); },
      eq16);
  check1(
      "misc.e2e.cvt.rpi.u8.bf16", k_u8_bf16_rpi, h16, [](u16 b) { return to_int(b, 3, 8, false); },
      eq16);
  check1(
      "misc.e2e.cvt.rn.f16.bf16", k_f16_bf16_rn, h16,
      [](u16 b) { return c::bf16_to_f16(b, c::RN); }, eq_f16_nan);
  check1(
      "misc.e2e.cvt.rz.f16.bf16", k_f16_bf16_rz, h16,
      [](u16 b) { return c::bf16_to_f16(b, c::RZ); }, eq_f16_nan);
  check1(
      "misc.e2e.cvt.rm.f16.bf16", k_f16_bf16_rm, h16,
      [](u16 b) { return c::bf16_to_f16(b, c::RM); }, eq_f16_nan);
  check1(
      "misc.e2e.cvt.f64.bf16", k_f64_bf16, h16,
      [](u16 b) { return c::f32_to_f64(c::bf16_to_f32(b)); }, eq_f64_nan);
  check1(
      "misc.e2e.cvt.ftz.f32.bf16", k_f32_bf16_ftz, h16,
      [](u16 b) { return c::f32_ftz(c::bf16_to_f32(b)); }, eq_f32_nan);
  return finish();
}
