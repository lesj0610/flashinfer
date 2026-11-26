// Bit-exact software floating-point conversions for the SM80 wgmma emulation
// layer (fork-only). Owner: T2. Host and device (no libm, no libdevice).
//
// Used by runtime/lib_misc.cu for the sm_89/sm_90 cvt forms that SM80 lacks and
// by tests/misc as the CPU side of the e2e checks. tests/misc/ref_cvt.py is an
// independent fractions-based reference for every function below.
//
// Entry points (namespace wgemu::cvt, all WGEMU_HD):
//   enum Rnd { RN, RZ, RM, RP, RNA }
//   struct Fmt; fmt_f32(), fmt_f16(), fmt_bf16(), fmt_tf32(), fmt_e4m3(), fmt_e5m2()
//   uint32_t round_to_fmt(Fmt f, bool neg, uint64_t mag, int exp2, Rnd r, bool satfinite)
//       value = (-1)^neg * mag * 2^exp2 (exact), rounded once into format f.
//   uint32_t f32_to_fp8(uint32_t x, bool e5m2, bool relu, bool satfinite)
//   uint32_t f16_to_fp8(uint32_t h, bool e5m2, bool relu, bool satfinite)
//   uint32_t fp8_to_f16(uint32_t v, bool e5m2, bool relu)
//   uint32_t f32x2_to_fp8x2(uint32_t a, uint32_t b, bool e5m2, bool relu, bool satfinite)
//       = cvt(a) << 8 | cvt(b)              (PTX: a goes to the upper byte)
//   uint32_t f16x2_to_fp8x2(uint32_t a, bool e5m2, bool relu, bool satfinite)
//       = cvt(a[31:16]) << 8 | cvt(a[15:0])
//   uint32_t fp8x2_to_f16x2(uint32_t a, bool e5m2, bool relu)
//       = cvt(a[15:8]) << 16 | cvt(a[7:0])
//   uint32_t f32_to_tf32(uint32_t x, Rnd r /* RN, RZ, RNA */, bool relu, bool satfinite)
//   uint32_t f32_to_bf16(uint32_t x, Rnd r), f16_to_bf16(uint32_t h, Rnd r),
//            f64_to_bf16(uint64_t d, Rnd r), int_to_bf16(uint64_t v, bool is_signed, int bits, Rnd
//            r)
//   uint32_t f32_to_f16(uint32_t x, Rnd r), bf16_to_f16(uint32_t b, Rnd r)
//   uint32_t f16_to_f32(uint32_t h), bf16_to_f32(uint32_t b), f32_ftz(uint32_t x)
//   uint64_t f32_to_f64(uint32_t x)
//
// NaN results are canonical: f32/tf32 0x7FFFFFFF, f16/bf16 0x7FFF, fp8 0x7F
// (sign dropped). For fp8 this is a [inference]: PTX only says "NaN in the
// destination format"; cuda_fp8.hpp's host fallback keeps the sign instead.
// .relu: NaN -> canonical NaN, any result with the sign bit set (including -0)
// -> +0. The -0 case is a [inference]; tests/misc/diag_sm80.cu checks the same
// rule against SM80's native cvt.rn.relu.f16x2.f32.
#pragma once
#include <stdint.h>

#include "wgmma_emu/types.h"

namespace wgemu {
namespace cvt {

enum Rnd : int { RN = 0, RZ = 1, RM = 2, RP = 3, RNA = 4 };

struct Fmt {
  int ebits;         // exponent field width
  int mbits;         // stored mantissa bits
  int bias;          // exponent bias
  int shift;         // left shift of the packed magnitude inside the storage word (tf32: 13)
  bool has_inf;      // IEEE-like: all-ones exponent is inf/NaN
  uint32_t maxnorm;  // largest finite magnitude, packed (before shift)
  uint32_t inf;      // packed infinity (if has_inf)
  uint32_t nan;      // canonical NaN, full storage word
  uint32_t sign;     // storage sign bit
};

// Packed magnitude = exponent_field << mbits | mantissa. Formats are returned by
// value from functions (not constexpr globals) so device code can use them
// without --expt-relaxed-constexpr.
WGEMU_HD Fmt fmt_f32() {
  return Fmt{8, 23, 127, 0, true, 0x7F7FFFFFu, 0x7F800000u, 0x7FFFFFFFu, 0x80000000u};
}
WGEMU_HD Fmt fmt_tf32() {
  return Fmt{8, 10, 127, 13, true, 0x3FBFFu, 0x3FC00u, 0x7FFFFFFFu, 0x80000000u};
}
WGEMU_HD Fmt fmt_f16() { return Fmt{5, 10, 15, 0, true, 0x7BFFu, 0x7C00u, 0x7FFFu, 0x8000u}; }
WGEMU_HD Fmt fmt_bf16() { return Fmt{8, 7, 127, 0, true, 0x7F7Fu, 0x7F80u, 0x7FFFu, 0x8000u}; }
WGEMU_HD Fmt fmt_e4m3() { return Fmt{4, 3, 7, 0, false, 0x7Eu, 0u, 0x7Fu, 0x80u}; }
WGEMU_HD Fmt fmt_e5m2() { return Fmt{5, 2, 15, 0, true, 0x7Bu, 0x7Cu, 0x7Fu, 0x80u}; }

WGEMU_HD int clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
  return __clzll((long long)x);
#else
  return x == 0 ? 64 : __builtin_clzll(x);
#endif
}

// Rounds (-1)^neg * mag * 2^exp2 into f once. NaN/inf inputs are handled by the
// callers; mag == 0 gives a signed zero.
WGEMU_HD uint32_t round_to_fmt(Fmt f, bool neg, uint64_t mag, int exp2, Rnd r, bool satfinite) {
  const uint32_t sgn = neg ? f.sign : 0u;
  if (mag == 0) return sgn;
  const int p = 63 - clz64(mag);
  const int e_lead = p + exp2;
  const int emin = 1 - f.bias;
  const int q0 = (e_lead < emin ? emin : e_lead) - f.mbits;  // quantum exponent
  const int sh = q0 - exp2;
  uint64_t rr;
  bool up = false;
  if (sh <= 0) {
    rr = mag << (-sh);
  } else {
    uint64_t rem;
    int cmp;  // rem vs half: -1 below, 0 equal, 1 above
    if (sh > 64) {
      rr = 0;
      rem = mag;
      cmp = -1;  // mag < 2^63 < half
    } else if (sh == 64) {
      rr = 0;
      rem = mag;
      const uint64_t half = 1ull << 63;
      cmp = rem > half ? 1 : (rem == half ? 0 : -1);
    } else {
      rr = mag >> sh;
      rem = mag & ((1ull << sh) - 1ull);
      const uint64_t half = 1ull << (sh - 1);
      cmp = rem > half ? 1 : (rem == half ? 0 : -1);
    }
    switch (r) {
      case RN:
        up = cmp > 0 || (cmp == 0 && (rr & 1ull));
        break;
      case RNA:
        up = cmp >= 0;
        break;
      case RZ:
        up = false;
        break;
      case RM:
        up = neg && rem != 0;
        break;
      case RP:
        up = !neg && rem != 0;
        break;
    }
  }
  rr += up ? 1ull : 0ull;
  int q = q0;
  if (rr == (2ull << f.mbits)) {
    rr >>= 1;
    q += 1;
  }
  if (rr == 0) return sgn;
  uint64_t expf, mant;
  if (rr < (1ull << f.mbits)) {
    expf = 0;
    mant = rr;
  } else {
    expf = (uint64_t)(q + f.mbits + f.bias);
    mant = rr - (1ull << f.mbits);
  }
  const bool ovf = expf >= (1ull << f.ebits) || ((expf << f.mbits) | mant) > f.maxnorm;
  if (ovf) {
    if (satfinite) return sgn | (f.maxnorm << f.shift);
    if (!f.has_inf) return f.nan;
    bool to_inf;
    switch (r) {
      case RZ:
        to_inf = false;
        break;
      case RM:
        to_inf = neg;
        break;
      case RP:
        to_inf = !neg;
        break;
      default:
        to_inf = true;
        break;
    }
    return sgn | ((to_inf ? f.inf : f.maxnorm) << f.shift);
  }
  return sgn | ((uint32_t)((expf << f.mbits) | mant) << f.shift);
}

// Decoded IEEE-style value. kind: 0 finite (incl. zero), 1 inf, 2 NaN.
struct Dec {
  int kind;
  bool neg;
  uint64_t mag;
  int exp2;
};

WGEMU_HD Dec decode_ieee(uint64_t bits, int ebits, int mbits, int bias) {
  Dec d;
  const uint64_t mmask = (1ull << mbits) - 1ull;
  const uint64_t emask = (1ull << ebits) - 1ull;
  d.neg = ((bits >> (ebits + mbits)) & 1ull) != 0;
  const uint64_t e = (bits >> mbits) & emask;
  const uint64_t m = bits & mmask;
  if (e == emask) {
    d.kind = m ? 2 : 1;
    d.mag = 0;
    d.exp2 = 0;
    return d;
  }
  d.kind = 0;
  if (e == 0) {
    d.mag = m;
    d.exp2 = 1 - bias - mbits;
  } else {
    d.mag = m | (1ull << mbits);
    d.exp2 = (int)e - bias - mbits;
  }
  return d;
}

WGEMU_HD Dec decode_f32(uint32_t x) { return decode_ieee(x, 8, 23, 127); }
WGEMU_HD Dec decode_f16(uint32_t h) { return decode_ieee(h & 0xFFFFu, 5, 10, 15); }
WGEMU_HD Dec decode_bf16(uint32_t b) { return decode_ieee(b & 0xFFFFu, 8, 7, 127); }
WGEMU_HD Dec decode_f64(uint64_t d) { return decode_ieee(d, 11, 52, 1023); }

// Generic conversion of a decoded value. relu and satfinite as in PTX cvt.
WGEMU_HD uint32_t convert_dec(const Dec& d, Fmt f, Rnd r, bool relu, bool satfinite) {
  if (d.kind == 2) return f.nan;
  if (relu && d.neg) return 0u;
  if (d.kind == 1) {
    if (satfinite) return (d.neg ? f.sign : 0u) | (f.maxnorm << f.shift);
    if (!f.has_inf) return f.nan;
    return (d.neg ? f.sign : 0u) | (f.inf << f.shift);
  }
  return round_to_fmt(f, d.neg, d.mag, d.exp2, r, satfinite);
}

// ---- fp8 ------------------------------------------------------------------

WGEMU_HD uint32_t f32_to_fp8(uint32_t x, bool e5m2, bool relu, bool satfinite) {
  return convert_dec(decode_f32(x), e5m2 ? fmt_e5m2() : fmt_e4m3(), RN, relu, satfinite);
}

WGEMU_HD uint32_t f16_to_fp8(uint32_t h, bool e5m2, bool relu, bool satfinite) {
  return convert_dec(decode_f16(h), e5m2 ? fmt_e5m2() : fmt_e4m3(), RN, relu, satfinite);
}

// fp8 -> f16 is exact. NaN -> 0x7FFF (canonical).
WGEMU_HD uint32_t fp8_to_f16(uint32_t v, bool e5m2, bool relu) {
  v &= 0xFFu;
  const bool neg = (v & 0x80u) != 0;
  Dec d;
  if (e5m2) {
    d = decode_ieee(v, 5, 2, 15);
  } else {
    const uint32_t e = (v >> 3) & 0xFu, m = v & 7u;
    d.neg = neg;
    if (e == 0xFu && m == 7u) {
      d.kind = 2;
      d.mag = 0;
      d.exp2 = 0;
    } else {
      d.kind = 0;
      d.mag = e == 0 ? m : (m | 8u);
      d.exp2 = e == 0 ? -9 : (int)e - 10;
    }
  }
  return convert_dec(d, fmt_f16(), RN, relu, false);
}

WGEMU_HD uint32_t f32x2_to_fp8x2(uint32_t a, uint32_t b, bool e5m2, bool relu, bool satfinite) {
  return (f32_to_fp8(a, e5m2, relu, satfinite) << 8) | f32_to_fp8(b, e5m2, relu, satfinite);
}

WGEMU_HD uint32_t f16x2_to_fp8x2(uint32_t a, bool e5m2, bool relu, bool satfinite) {
  return (f16_to_fp8(a >> 16, e5m2, relu, satfinite) << 8) |
         f16_to_fp8(a & 0xFFFFu, e5m2, relu, satfinite);
}

WGEMU_HD uint32_t fp8x2_to_f16x2(uint32_t a, bool e5m2, bool relu) {
  return (fp8_to_f16((a >> 8) & 0xFFu, e5m2, relu) << 16) | fp8_to_f16(a & 0xFFu, e5m2, relu);
}

// ---- tf32 / bf16 / f16 ----------------------------------------------------

WGEMU_HD uint32_t f32_to_tf32(uint32_t x, Rnd r, bool relu, bool satfinite) {
  return convert_dec(decode_f32(x), fmt_tf32(), r, relu, satfinite);
}

WGEMU_HD uint32_t f32_to_bf16(uint32_t x, Rnd r) {
  return convert_dec(decode_f32(x), fmt_bf16(), r, false, false);
}

WGEMU_HD uint32_t f16_to_bf16(uint32_t h, Rnd r) {
  return convert_dec(decode_f16(h), fmt_bf16(), r, false, false);
}

WGEMU_HD uint32_t f64_to_bf16(uint64_t d, Rnd r) {
  return convert_dec(decode_f64(d), fmt_bf16(), r, false, false);
}

// v holds a `bits`-wide integer in its low bits (sign-extended or not).
WGEMU_HD uint32_t int_to_bf16(uint64_t v, bool is_signed, int bits, Rnd r) {
  if (bits < 64) {
    const uint64_t mask = (1ull << bits) - 1ull;
    v &= mask;
    if (is_signed && ((v >> (bits - 1)) & 1ull)) v |= ~mask;  // sign-extend
  }
  bool neg = false;
  uint64_t mag = v;
  if (is_signed && (int64_t)v < 0) {
    neg = true;
    mag = (uint64_t)0 - v;  // |INT64_MIN| = 2^63 fits
  }
  return round_to_fmt(fmt_bf16(), neg, mag, 0, r, false);
}

WGEMU_HD uint32_t f32_to_f16(uint32_t x, Rnd r) {
  return convert_dec(decode_f32(x), fmt_f16(), r, false, false);
}

WGEMU_HD uint32_t bf16_to_f16(uint32_t b, Rnd r) {
  return convert_dec(decode_bf16(b), fmt_f16(), r, false, false);
}

WGEMU_HD uint32_t f16_to_f32(uint32_t h) {
  return convert_dec(decode_f16(h), fmt_f32(), RN, false, false);
}

// bf16 -> f32 is a 16-bit shift; NaN stays NaN with its payload (PTX leaves the
// NaN unspecified, SM80 hardware shifts).
WGEMU_HD uint32_t bf16_to_f32(uint32_t b) { return (b & 0xFFFFu) << 16; }

// Flush an f32 subnormal to a sign-preserving zero.
WGEMU_HD uint32_t f32_ftz(uint32_t x) { return (x & 0x7F800000u) == 0 ? (x & 0x80000000u) : x; }

WGEMU_HD uint64_t f32_to_f64(uint32_t x) {
  const Dec d = decode_f32(x);
  const uint64_t s = d.neg ? (1ull << 63) : 0ull;
  if (d.kind == 2) return 0x7FFFFFFFFFFFFFFFull;
  if (d.kind == 1) return s | 0x7FF0000000000000ull;
  if (d.mag == 0) return s;
  const int p = 63 - clz64(d.mag);
  const uint64_t e = (uint64_t)(p + d.exp2 + 1023);
  const uint64_t m = (d.mag << (52 - p)) & ((1ull << 52) - 1ull);
  return s | (e << 52) | m;
}

}  // namespace cvt
}  // namespace wgemu
