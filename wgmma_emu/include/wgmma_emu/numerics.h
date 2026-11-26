// Bit-exact tensor-core dot-product numerics for the SM80 wgmma emulation layer (fork-only).
// Owner: N. Integer-only port of MMA-Sim b953d74 (MIT):
//   arithmetic/fdpa.py  frexp_and_normalize, ldexp_and_normalize, truncated_fused_sum, t_fdpa,
//                       MMA_T_FDPA.dpa
//   arithmetic/fma.py   MMA_FMA.dpa (sequential IEEE fma, f64 mma)
//   nv_ptx/sim.py       MMA (Ampere, Ada Lovelace, Hopper) and WGMMA (Hopper) parameter tables,
//                       including the Ada Lovelace fp8 special case (F=13, RZ-E8M13)
// Out of scope: MMABlockScale, tcgen05, Volta/Turing, RTX Blackwell, TPU/AMD models.
//
// API (namespace wgemu, header-only, no library entry points):
//   enum class Rho, Arch, ParamSource; struct FdpaParams
//   FdpaParams  wgmma_params(EType a, EType b, bool d_f16)             Hopper wgmma.mma_async
//   FdpaParams  mma_params(Arch arch, EType a, EType b, bool d_f16)    mma.sync
//   ParamSource wgmma_params_source(EType a, EType b, bool d_f16)
//   ParamSource mma_params_source(Arch arch, EType a, EType b, bool d_f16)
//   uint32_t    tfdpa(a, b, n, at, bt, c, d_f16, p)          one T-FDPA block, 1 <= n <= L_max
//   uint32_t    dot_float(a, b, K, at, bt, c, d_f16, p)      L_max blocks accumulated in order
//   int32_t     dot_int(a, b, K, a_signed, b_signed, c, satfinite)
//   int32_t     dot_b1_and_popc(a_words, b_words, nwords, c)
//   uint64_t    fma_f64_rn(a, b, c)                          IEEE binary64 fma, RNE
//   uint64_t    dot_f64_fma(a, b, K, c)                      MMA-Sim MMA_FMA (f64 mma.sync)
//
// Operand encoding: f16/bf16 in the low 16 bits, tf32 as f32 bits (low 13 bits are ignored, as
// MMA-Sim's `x >> 13 << 13`), fp8 in the low 8 bits, s8/u8 in the low 8 bits. c and the result are
// f16 in the low 16 bits when d_f16, f32 bits otherwise.
//
// Arithmetic: integers only (no float/double operations, no libm), usable on host and device.
// Unsupported inputs are explicit errors: abort() with a WGEMU-NUMERICS-FATAL message on the host,
// `trap` on the device.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"

#if defined(__CUDA_ARCH__)
#define WGEMU_NUM_FAIL(msg) asm volatile("trap;")
#else
#define WGEMU_NUM_FAIL(msg) (fprintf(stderr, "WGEMU-NUMERICS-FATAL %s\n", (msg)), abort())
#endif

namespace wgemu {

// Output rounding (MMA-Sim rho): RZ-FP32, RZ-E8M13, RNE-FP16.
enum class Rho : uint8_t { RZ_FP32, RZ_E8M13, RNE_FP16 };
enum class Arch : uint8_t { Ampere, Ada, Hopper };
// F: fraction bits kept below the largest term, L_max: products per block (elements),
// e_zero: exponent given to zero terms.
struct FdpaParams {
  int F;
  int L_max;
  int e_zero;
  Rho rho;
};
// Where a parameter set comes from. MmaSim: the combination is in MMA-Sim's instruction list for
// the architecture (hardware-validated by MMA-Sim). Inferred: MMA-Sim's per-architecture table
// applied to a combination its list does not contain (Hopper mma.sync fp8).
enum class ParamSource : uint8_t { Unsupported = 0, MmaSim = 1, Inferred = 2 };

namespace detail {

WGEMU_HD int clz64(uint64_t x) {  // x != 0
#if defined(__CUDA_ARCH__)
  return __clzll(static_cast<long long>(x));
#else
  return __builtin_clzll(x);
#endif
}

WGEMU_HD int popc32(uint32_t x) {
#if defined(__CUDA_ARCH__)
  return __popc(x);
#else
  return __builtin_popcount(x);
#endif
}

// Two's-complement reinterpretation without implementation-defined conversions.
WGEMU_HD int32_t to_int32(uint32_t u) {
  return u >= 0x80000000u ? static_cast<int32_t>(static_cast<int64_t>(u) - 0x100000000LL)
                          : static_cast<int32_t>(u);
}

enum : uint8_t { kFinite = 0, kInf = 1, kNaN = 2 };

// MMA-Sim frexp_and_normalize in integers: |x| = sig * 2^(exp - frac) with exp the normalized
// exponent e (1 <= s < 2 for normals) and e = e_subnormal (1 - bias) for subnormals and zeros.
struct Unpacked {
  uint64_t sig;  // significand integer, implicit bit included for normals
  int exp;
  int frac;
  uint8_t cls;
  bool neg;
};

WGEMU_HD Unpacked unpack_fp(uint64_t bits, int ebits, int mbits, int bias, bool ieee_special) {
  Unpacked u;
  const uint64_t mmask = (uint64_t(1) << mbits) - 1;
  const uint32_t emask = (1u << ebits) - 1;
  const uint32_t ef = static_cast<uint32_t>(bits >> mbits) & emask;
  const uint64_t mf = bits & mmask;
  u.neg = ((bits >> (ebits + mbits)) & 1) != 0;
  u.frac = mbits;
  u.cls = kFinite;
  if (ef == emask) {
    if (ieee_special) {
      u.cls = mf == 0 ? kInf : kNaN;
    } else if (mf == mmask) {
      u.cls = kNaN;  // e4m3 (torch float8_e4m3fn / PTX .e4m3): only S.1111.111 is NaN, no inf
    }
  }
  if (ef == 0) {
    u.sig = mf;
    u.exp = 1 - bias;
  } else {
    u.sig = mf | (uint64_t(1) << mbits);
    u.exp = static_cast<int>(ef) - bias;
  }
  return u;
}

// Subnormal exponents match MMA-Sim helper.dtype_subnormal_exponent: f16 -14, bf16 -126,
// f32/tf32 -126, e5m2 -14, e4m3 -6.
WGEMU_HD Unpacked unpack_input(EType t, uint32_t bits) {
  switch (t) {
    case EType::F16:
      return unpack_fp(bits & 0xFFFFu, 5, 10, 15, true);
    case EType::BF16:
      return unpack_fp(bits & 0xFFFFu, 8, 7, 127, true);
    case EType::TF32:  // MMA-Sim truncate_fp32_to_tf32: x >> 13 << 13 on the int32 view
      return unpack_fp(bits & 0xFFFFE000u, 8, 23, 127, true);
    case EType::E4M3:
      return unpack_fp(bits & 0xFFu, 4, 3, 7, false);
    case EType::E5M2:
      return unpack_fp(bits & 0xFFu, 5, 2, 15, true);
    default:
      break;
  }
  WGEMU_NUM_FAIL("unpack_input: not a floating-point tensor-core input type");
  return unpack_fp(0, 8, 23, 127, true);
}

WGEMU_HD Unpacked unpack_acc(uint32_t c, bool d_f16) {
  return d_f16 ? unpack_fp(c & 0xFFFFu, 5, 10, 15, true) : unpack_fp(c, 8, 23, 127, true);
}

// torch itemsize of the MMA-Sim operand dtype (tf32 is held as torch.float32).
WGEMU_HD int input_itemsize(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 2;
    case EType::TF32:
      return 4;
    case EType::E4M3:
    case EType::E5M2:
      return 1;
    default:
      break;
  }
  WGEMU_NUM_FAIL("input_itemsize: not a floating-point tensor-core input type");
  return 1;
}

// One term of truncated_fused_sum: trunc(s * 2^(e - e_max + F)) with s = sig * 2^-frac, as a
// magnitude. e <= e_max and sig < 2^(frac + 2) give a result below 2^(F + 2).
WGEMU_HD uint64_t trunc_term(uint64_t sig, int e, int frac, int e_max, int F) {
  const int sh = e - e_max + F - frac;
  if (sh >= 0) return sig << sh;  // sh <= F - frac <= 36 (F <= 40, frac >= 4)
  if (sh <= -64) return 0;
  return sig >> (-sh);
}

// m * 2^sh truncated toward zero. Left shifts are only used where the result is known small.
WGEMU_HD uint64_t shift_rz(uint64_t m, int sh) {
  if (sh >= 0) return m << sh;
  if (sh <= -64) return 0;
  return m >> (-sh);
}

// m * 2^sh rounded to nearest, ties to even (torch.round). m < 2^62 here, so a right shift by 64 or
// more rounds to 0.
WGEMU_HD uint64_t shift_rne(uint64_t m, int sh) {
  if (sh >= 0) return m << sh;
  const int k = -sh;
  if (k >= 64) return 0;
  uint64_t q = m >> k;
  const uint64_t rem = m & ((uint64_t(1) << k) - 1);
  const uint64_t half = uint64_t(1) << (k - 1);
  if (rem > half || (rem == half && (q & 1))) ++q;
  return q;
}

// ldexp_and_normalize(S * 2^-F, e_max, rho) + 0.0, with E0 = e_max - F, so value = S * 2^E0.
// |S| <= (L_max + 1) * 2^(F + 2) < 2^55 (params_ok: L_max <= 4096, F <= 40): no int64 overflow.
WGEMU_HD uint32_t round_out(int64_t S, int E0, Rho rho) {
  if (S == 0) return 0;  // exact zero; "+ 0.0" turns -0 into +0
  const bool neg = S < 0;
  const uint64_t mag = neg ? uint64_t(0) - static_cast<uint64_t>(S) : static_cast<uint64_t>(S);
  const int msb = 63 - clz64(mag);
  int ev = msb + E0;  // value in [2^ev, 2^(ev+1))
  if (rho == Rho::RNE_FP16) {
    uint32_t bits;
    if (ev >= -14) {
      uint64_t r = shift_rne(mag, 10 - msb);  // round(s * 2^10), s in [1, 2): r in [1024, 2048]
      if (r == 2048) {
        r = 1024;
        ++ev;
      }
      if (ev > 15) {
        bits = 0x7C00u;  // f64 -> f16 conversion of >= 2^16 gives inf
      } else {
        bits = (static_cast<uint32_t>(ev + 15) << 10) | static_cast<uint32_t>(r & 0x3FFu);
      }
    } else {
      const uint64_t r = shift_rne(mag, E0 + 24);  // subnormal grid 2^-24, r <= 1024
      if (r == 0) return 0;                        // rounded to -0/+0, then "+ 0.0"
      bits = static_cast<uint32_t>(r);             // r == 1024 encodes the min normal 2^-14
    }
    return (neg ? 0x8000u : 0u) | bits;
  }
  const int keep = rho == Rho::RZ_FP32 ? 23 : 13;  // RZ-FP32 / RZ-E8M13 fraction bits
  const uint32_t sign = neg ? 0x80000000u : 0u;
  if (ev >= 128) return sign | 0x7F800000u;  // truncated value >= 2^128 converts to inf
  uint32_t bits;
  if (ev >= -126) {
    const uint64_t m = shift_rz(mag, keep - msb);  // keep+1 bits with the leading one
    bits = (static_cast<uint32_t>(ev + 127) << 23) |
           (static_cast<uint32_t>(m & ((uint64_t(1) << keep) - 1)) << (23 - keep));
  } else {
    const uint64_t m = shift_rz(mag, E0 + 126 + keep);  // subnormal grid 2^-(126+keep)
    if (m == 0) return 0;
    bits = static_cast<uint32_t>(m) << (23 - keep);
  }
  return sign | bits;
}

WGEMU_HD bool params_ok(const FdpaParams& p, bool d_f16) {
  return p.F >= 1 && p.F <= 40 && p.L_max >= 1 && p.L_max <= 4096 &&
         ((p.rho == Rho::RNE_FP16) == d_f16);
}

}  // namespace detail

WGEMU_HD ParamSource wgmma_params_source(EType a, EType b, bool d_f16) {
  // nv_ptx/isa_wgmma.py hopper_wgmma_shape_and_type
  if (a == EType::F16 && b == EType::F16) return ParamSource::MmaSim;
  if (a == EType::BF16 && b == EType::BF16 && !d_f16) return ParamSource::MmaSim;
  if (a == EType::TF32 && b == EType::TF32 && !d_f16) return ParamSource::MmaSim;
  if (etype_is_fp8(a) && etype_is_fp8(b)) return ParamSource::MmaSim;
  return ParamSource::Unsupported;
}

// nv_ptx/sim.py WGMMA.__init__.
WGEMU_HD FdpaParams wgmma_params(EType a, EType b, bool d_f16) {
  FdpaParams p;
  if (wgmma_params_source(a, b, d_f16) == ParamSource::Unsupported) {
    WGEMU_NUM_FAIL("wgmma_params: unsupported (atype, btype, dtype)");
    p.F = 0;
    p.L_max = 0;
    p.e_zero = 0;
    p.rho = Rho::RZ_FP32;
    return p;
  }
  if (etype_is_fp8(a)) {
    p.F = 13;
    p.rho = d_f16 ? Rho::RNE_FP16 : Rho::RZ_E8M13;
  } else {
    p.F = 25;
    p.rho = d_f16 ? Rho::RNE_FP16 : Rho::RZ_FP32;
  }
  p.L_max = 32 / detail::input_itemsize(a);
  p.e_zero = d_f16 ? -21 : -133;
  return p;
}

WGEMU_HD ParamSource mma_params_source(Arch arch, EType a, EType b, bool d_f16) {
  // nv_ptx/isa_mma.py: sm_75/sm_80 lists (all three architectures), sm_89 fp8 list (Ada only).
  if (a == EType::F16 && b == EType::F16) return ParamSource::MmaSim;
  if ((a == EType::BF16 && b == EType::BF16) || (a == EType::TF32 && b == EType::TF32)) {
    return d_f16 ? ParamSource::Unsupported : ParamSource::MmaSim;
  }
  if (etype_is_fp8(a) && etype_is_fp8(b)) {
    if (arch == Arch::Ada) return ParamSource::MmaSim;
    if (arch == Arch::Hopper) return ParamSource::Inferred;  // not in MMA-Sim's Hopper list
  }
  return ParamSource::Unsupported;
}

// nv_ptx/sim.py MMA.__init__ (_mma_F_bits, _mma_L_max_bytes, _mma_e_zero_for_fp16/fp32 and the
// "Ada Lovelace" fp8 special case). Hopper fp8 (ParamSource::Inferred) takes the generic Hopper
// branch of the same code: F=25, L_max=32, RZ-FP32 / RNE-FP16.
WGEMU_HD FdpaParams mma_params(Arch arch, EType a, EType b, bool d_f16) {
  FdpaParams p;
  if (mma_params_source(arch, a, b, d_f16) == ParamSource::Unsupported) {
    WGEMU_NUM_FAIL("mma_params: unsupported (arch, atype, btype, dtype)");
    p.F = 0;
    p.L_max = 0;
    p.e_zero = 0;
    p.rho = Rho::RZ_FP32;
    return p;
  }
  const bool hopper = arch == Arch::Hopper;
  p.F = hopper ? 25 : 24;
  p.L_max = (hopper ? 32 : 16) / detail::input_itemsize(a);
  p.e_zero = d_f16 ? (hopper ? -21 : -20) : (hopper ? -133 : -132);
  p.rho = d_f16 ? Rho::RNE_FP16 : Rho::RZ_FP32;
  if (arch == Arch::Ada && etype_is_fp8(a)) {
    p.F = 13;
    p.rho = d_f16 ? Rho::RNE_FP16 : Rho::RZ_E8M13;
  }
  return p;
}

// One MMA-Sim t_fdpa block: c + sum_i a[i] * b[i] over n <= L_max products.
WGEMU_HD uint32_t tfdpa(const uint32_t* a, const uint32_t* b, int n, EType at, EType bt, uint32_t c,
                        bool d_f16, const FdpaParams& p) {
  if (!detail::params_ok(p, d_f16) || n < 1 || n > p.L_max || !etype_is_float_input(at) ||
      !etype_is_float_input(bt)) {
    WGEMU_NUM_FAIL("tfdpa: invalid parameters, block length or operand type");
    return 0;
  }
  using detail::kFinite;
  using detail::kInf;
  using detail::kNaN;
  // Pass 1: inf/NaN classification and e_max. A NaN operand, inf * 0, or +inf with -inf makes the
  // double sum NaN; otherwise any inf term makes it inf (finite terms stay finite: s < 4 and
  // pow2(delta_e + F) <= 2^F). torch.frexp gives exponent 0 for inf/NaN, so inf terms never turn
  // into inf * 0.
  const detail::Unpacked uc = detail::unpack_acc(c, d_f16);
  bool nan = uc.cls == kNaN;
  bool pinf = uc.cls == kInf && !uc.neg;
  bool ninf = uc.cls == kInf && uc.neg;
  int e_max = (uc.cls == kFinite && uc.sig != 0) ? uc.exp : p.e_zero;
  for (int i = 0; i < n; ++i) {
    const detail::Unpacked ua = detail::unpack_input(at, a[i]);
    const detail::Unpacked ub = detail::unpack_input(bt, b[i]);
    if (ua.cls == kNaN || ub.cls == kNaN) {
      nan = true;
      continue;
    }
    if (ua.cls == kInf || ub.cls == kInf) {
      const bool other_zero =
          (ua.cls == kFinite && ua.sig == 0) || (ub.cls == kFinite && ub.sig == 0);
      if (other_zero) {
        nan = true;
      } else if (ua.neg != ub.neg) {
        ninf = true;
      } else {
        pinf = true;
      }
      continue;
    }
    const int e = (ua.sig == 0 || ub.sig == 0) ? p.e_zero : ua.exp + ub.exp;
    if (e > e_max) e_max = e;
  }
  if (nan || (pinf && ninf)) return d_f16 ? 0x7FFFu : 0x7FFFFFFFu;  // canonical NaN
  if (pinf) return d_f16 ? 0x7C00u : 0x7F800000u;
  if (ninf) return d_f16 ? 0xFC00u : 0xFF800000u;
  // Pass 2: truncated fused sum, exact in int64 (each term < 2^(F + 2)).
  int64_t sum = 0;
  if (uc.sig != 0) {
    const int64_t t = static_cast<int64_t>(detail::trunc_term(uc.sig, uc.exp, uc.frac, e_max, p.F));
    sum += uc.neg ? -t : t;
  }
  for (int i = 0; i < n; ++i) {
    const detail::Unpacked ua = detail::unpack_input(at, a[i]);
    const detail::Unpacked ub = detail::unpack_input(bt, b[i]);
    if (ua.sig == 0 || ub.sig == 0) continue;  // zero term: only its e_zero matters (pass 1)
    // sig_a * sig_b < 2^24 * 2^24 (tf32 held as f32): exact in uint64. s_a * s_b is exact in the
    // torch float32 product as well (at most 22 significant bits).
    const uint64_t sig = ua.sig * ub.sig;
    const int64_t t = static_cast<int64_t>(
        detail::trunc_term(sig, ua.exp + ub.exp, ua.frac + ub.frac, e_max, p.F));
    sum += (ua.neg != ub.neg) ? -t : t;
  }
  return detail::round_out(sum, e_max - p.F, p.rho);
}

// MMA_T_FDPA.dpa: L = min(K, L_max) blocks, each fed the previous block's rounded result as c.
WGEMU_HD uint32_t dot_float(const uint32_t* a, const uint32_t* b, int K, EType at, EType bt,
                            uint32_t c, bool d_f16, const FdpaParams& p) {
  if (K < 1 || !detail::params_ok(p, d_f16)) {
    WGEMU_NUM_FAIL("dot_float: K < 1 or invalid parameters");
    return 0;
  }
  const int L = K < p.L_max ? K : p.L_max;
  uint32_t acc = c;
  for (int i = 0; i < K; i += L) {
    const int n = (K - i) < L ? (K - i) : L;
    acc = tfdpa(a + i, b + i, n, at, bt, acc, d_f16, p);
  }
  return acc;
}

// Integer mma/wgmma: exact sum of c and all products, then saturate (.satfinite) or wrap mod 2^32.
// |a_k * b_k| <= 255 * 255 < 2^16 and |c| <= 2^31, so |sum| < 2^31 + K * 2^16 < 2^48 for any int K.
WGEMU_HD int32_t dot_int(const uint32_t* a, const uint32_t* b, int K, bool a_signed, bool b_signed,
                         int32_t c, bool satfinite) {
  int64_t s = c;
  for (int k = 0; k < K; ++k) {
    const int32_t ab = static_cast<int32_t>(a[k] & 0xFFu);
    const int32_t bb = static_cast<int32_t>(b[k] & 0xFFu);
    const int32_t x = (a_signed && ab >= 128) ? ab - 256 : ab;
    const int32_t y = (b_signed && bb >= 128) ? bb - 256 : bb;
    s += static_cast<int64_t>(x) * y;
  }
  if (satfinite) {
    if (s > 2147483647LL) return 2147483647;
    if (s < -2147483648LL) return static_cast<int32_t>(-2147483647 - 1);
    return static_cast<int32_t>(s);
  }
  return detail::to_int32(static_cast<uint32_t>(static_cast<uint64_t>(s)));
}

// b1 .and.popc: c + sum popc(a & b), wrapped mod 2^32.
WGEMU_HD int32_t dot_b1_and_popc(const uint32_t* a_words, const uint32_t* b_words, int nwords,
                                 int32_t c) {
  uint32_t s = static_cast<uint32_t>(c);
  for (int w = 0; w < nwords; ++w)
    s += static_cast<uint32_t>(detail::popc32(a_words[w] & b_words[w]));
  return detail::to_int32(s);
}

namespace detail {

struct U128 {
  uint64_t hi, lo;
};

WGEMU_HD U128 u128_mul64(uint64_t a, uint64_t b) {
  const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);  // < 2^34
  U128 r;
  r.lo = (mid << 32) | (p00 & 0xFFFFFFFFu);
  r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  return r;
}

WGEMU_HD bool u128_is_zero(U128 x) { return (x.hi | x.lo) == 0; }

WGEMU_HD int u128_clz(U128 x) { return x.hi ? clz64(x.hi) : 64 + clz64(x.lo); }  // x != 0

WGEMU_HD U128 u128_shl(U128 x, int s) {  // 0 <= s < 128
  U128 r;
  if (s == 0) return x;
  if (s >= 64) {
    r.hi = x.lo << (s - 64);
    r.lo = 0;
  } else {
    r.hi = (x.hi << s) | (x.lo >> (64 - s));
    r.lo = x.lo << s;
  }
  return r;
}

WGEMU_HD U128 u128_shr(U128 x, int s) {  // 0 <= s, plain
  U128 r;
  if (s == 0) return x;
  if (s >= 128) {
    r.hi = 0;
    r.lo = 0;
  } else if (s >= 64) {
    r.hi = 0;
    r.lo = x.hi >> (s - 64);
  } else {
    r.hi = x.hi >> s;
    r.lo = (x.lo >> s) | (x.hi << (64 - s));
  }
  return r;
}

// True when any of bits [0, n) of x is set.
WGEMU_HD bool u128_low_nonzero(U128 x, int n) {
  if (n <= 0) return false;
  if (n >= 128) return !u128_is_zero(x);
  if (n >= 64) return x.lo != 0 || (n > 64 && (x.hi & ((uint64_t(1) << (n - 64)) - 1)) != 0);
  return (x.lo & ((uint64_t(1) << n) - 1)) != 0;
}

WGEMU_HD bool u128_bit(U128 x, int i) {  // 0 <= i < 128
  return ((i >= 64 ? (x.hi >> (i - 64)) : (x.lo >> i)) & 1) != 0;
}

// Right shift with the shifted-out bits ORed into bit 0 (sticky).
WGEMU_HD U128 u128_shr_sticky(U128 x, int s) {
  if (s == 0) return x;
  const bool lost = u128_low_nonzero(x, s);
  U128 r = u128_shr(x, s);
  r.lo |= lost ? 1u : 0u;
  return r;
}

WGEMU_HD U128 u128_add(U128 a, U128 b) {
  U128 r;
  r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + (r.lo < a.lo ? 1u : 0u);
  return r;
}

WGEMU_HD U128 u128_sub(U128 a, U128 b) {  // a >= b
  U128 r;
  r.lo = a.lo - b.lo;
  r.hi = a.hi - b.hi - (a.lo < b.lo ? 1u : 0u);
  return r;
}

WGEMU_HD bool u128_ge(U128 a, U128 b) { return a.hi != b.hi ? a.hi > b.hi : a.lo >= b.lo; }

// x * 2^-k rounded to nearest even, k >= 1, result known to fit in 64 bits.
WGEMU_HD uint64_t u128_rne_shr(U128 x, int k) {
  if (k > 128) return 0;  // x < 2^128: x * 2^-k < 1/2
  const bool half = u128_bit(x, k - 1);
  const bool rest = u128_low_nonzero(x, k - 1);
  uint64_t q = k >= 128 ? 0 : u128_shr(x, k).lo;
  if (half && (rest || (q & 1))) ++q;
  return q;
}

// Round R * 2^ER (R != 0) to binary64, round to nearest even, with gradual underflow.
WGEMU_HD uint64_t f64_round_pack(bool neg, U128 R, int ER) {
  const uint64_t sign = neg ? 0x8000000000000000ull : 0;
  const int msb = 127 - u128_clz(R);
  int ev = msb + ER;
  if (ev >= -1022) {
    const int drop = msb - 52;
    uint64_t m = drop <= 0 ? (R.lo << (-drop)) : u128_rne_shr(R, drop);  // R < 2^53 when drop <= 0
    if (m == (uint64_t(1) << 53)) {
      m >>= 1;
      ++ev;
    }
    if (ev > 1023) return sign | 0x7FF0000000000000ull;
    return sign | (static_cast<uint64_t>(ev + 1023) << 52) | (m & 0x000FFFFFFFFFFFFFull);
  }
  const int sh = ER + 1074;  // subnormal grid 2^-1074; R * 2^sh < 2^52 here
  const uint64_t m = sh >= 0 ? (R.lo << sh) : u128_rne_shr(R, -sh);
  return sign | m;  // m == 2^52 encodes the min normal, m == 0 a signed zero
}

WGEMU_HD bool f64_is_nan(uint64_t x) { return (x & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull; }

}  // namespace detail

// IEEE 754 binary64 fused multiply-add, round to nearest even (std::fma in MMA-Sim fma_cpu.cpp).
// NaN results: a NaN operand is returned quieted (first of a, b, c); invalid operations return
// 0x7FFFFFFFFFFFFFFF. The NaN payload is not part of the model: MMA-Sim's equivalence tests compare
// f64 mma with allow_different_nan (tests/equivalence/test_nv_*.py).
WGEMU_HD uint64_t fma_f64_rn(uint64_t a, uint64_t b, uint64_t c) {
  using namespace detail;
  const uint64_t kSign = 0x8000000000000000ull, kExp = 0x7FF0000000000000ull;
  const uint64_t kMant = 0x000FFFFFFFFFFFFFull, kQuiet = 0x0008000000000000ull;
  const uint64_t kDefaultNaN = 0x7FFFFFFFFFFFFFFFull;
  if (f64_is_nan(a)) return a | kQuiet;
  if (f64_is_nan(b)) return b | kQuiet;
  if (f64_is_nan(c)) return c | kQuiet;
  const bool sa = (a & kSign) != 0, sb = (b & kSign) != 0, sc = (c & kSign) != 0;
  const bool sp = sa != sb;
  const bool a_inf = (a & ~kSign) == kExp, b_inf = (b & ~kSign) == kExp,
             c_inf = (c & ~kSign) == kExp;
  const bool a_zero = (a & ~kSign) == 0, b_zero = (b & ~kSign) == 0, c_zero = (c & ~kSign) == 0;
  if (a_inf || b_inf) {
    if (a_zero || b_zero) return kDefaultNaN;   // inf * 0
    if (c_inf && sc != sp) return kDefaultNaN;  // inf - inf
    return (sp ? kSign : 0) | kExp;
  }
  if (c_inf) return c;
  if (a_zero || b_zero) {
    if (c_zero) return (sp && sc) ? kSign : 0;  // (+-0) + (+-0) under round to nearest
    return c;                                   // exact
  }
  // |x| = m * 2^(field - 1075) for normals (implicit bit set), field 1 for subnormals.
  const int fa = static_cast<int>((a >> 52) & 0x7FF), fb = static_cast<int>((b >> 52) & 0x7FF);
  const uint64_t ma = (a & kMant) | (fa ? (uint64_t(1) << 52) : 0);
  const uint64_t mb = (b & kMant) | (fb ? (uint64_t(1) << 52) : 0);
  U128 P = u128_mul64(ma, mb);  // exact, < 2^106
  int EP = (fa ? fa : 1) - 1075 + (fb ? fb : 1) - 1075;
  if (c_zero) return f64_round_pack(sp, P, EP);
  const int fc = static_cast<int>((c >> 52) & 0x7FF);
  U128 C;
  C.hi = 0;
  C.lo = (c & kMant) | (fc ? (uint64_t(1) << 52) : 0);
  int EC = (fc ? fc : 1) - 1075;
  // Put both leading bits at bit 125 (two bits of headroom), then align the smaller one with a
  // sticky bit. Shifts up to 20 lose no bits (P has at most 106, C at most 53 significant bits), so
  // an inexact alignment needs an exponent difference above 20, where cancellation removes at most
  // one leading bit and the sticky bit (bit 0) stays far below the round bit (bit >= 71).
  const int sP = 125 - (127 - u128_clz(P));
  P = u128_shl(P, sP);
  EP -= sP;
  const int sC = 125 - (127 - u128_clz(C));
  C = u128_shl(C, sC);
  EC -= sC;
  U128 X, Y;
  bool sx, sy;
  int EX;
  if (EP >= EC) {
    X = P;
    sx = sp;
    Y = u128_shr_sticky(C, EP - EC);
    sy = sc;
    EX = EP;
  } else {
    X = C;
    sx = sc;
    Y = u128_shr_sticky(P, EC - EP);
    sy = sp;
    EX = EC;
  }
  U128 R;
  bool sr;
  if (sx == sy) {
    R = u128_add(X, Y);  // < 2^127
    sr = sx;
  } else if (u128_ge(X, Y)) {
    R = u128_sub(X, Y);
    sr = sx;
  } else {
    R = u128_sub(Y, X);
    sr = sy;
  }
  if (u128_is_zero(R)) return 0;  // exact cancellation gives +0 under round to nearest
  return f64_round_pack(sr, R, EX);
}

// MMA-Sim MMA_FMA.dpa: c = fma(a[k], b[k], c) for k = 0 .. K-1 (f64 mma.sync on all three arches).
WGEMU_HD uint64_t dot_f64_fma(const uint64_t* a, const uint64_t* b, int K, uint64_t c) {
  if (K < 1) {
    WGEMU_NUM_FAIL("dot_f64_fma: K < 1");
    return 0;
  }
  for (int k = 0; k < K; ++k) c = fma_f64_rn(a[k], b[k], c);
  return c;
}

}  // namespace wgemu
