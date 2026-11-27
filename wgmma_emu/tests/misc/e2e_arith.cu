// End-to-end check of the sm_86 / sm_90 arithmetic and comparison forms rewritten by
// tools/ptxemu/rules_misc.py (fork-only). Owner: T2. Built for sm_90a by tools/nvcc_emu.
//
// Covered: add/sub/mul{.rn}.bf16{x2}, fma.rn.oob{.relu}.{f16,f16x2,bf16,bf16x2},
// min/max{.ftz}{.NaN}.xorsign.abs.{f32,f16,f16x2,bf16,bf16x2}, tanh.approx.bf16{x2},
// ex2.approx.ftz.bf16{x2}, set/setp with bf16 / bf16x2 (including BoolOp and .bf16 destinations
// with other source types), add.{u16x2,s16x2}, min/max{.relu}.{u16x2,s16x2,s32}.
//
// Expectations come from the PTX text: exact a*b+c rounded once (cvt.h round_to_fmt, checked
// against the exact reference tests/misc/ref_cvt.py), the xorsign pseudo-code (ptx.txt
// 9306-9326), the error bounds of tanh / ex2 (ptx.txt 10741-10742, 10786-10835) and the set /
// setp semantics (ptx.txt 11701-11790). The rewrites run native SM80 instructions, so NaN
// results are compared by class only.
//
// [inference] OOB-NaN is 0x7FFF in both 16-bit formats (PTX names the value without giving the
// encoding). [inference] .relu on an exact -0 result: either zero is accepted where +0 is
// expected. [inference] set with a .bf16x2 source writes 1.0 per half for a .bf16x2 destination
// and 0xffff per half for .u32 / .s32 (the description text; the pseudo-code at ptx.txt 11759
// tests stype where dtype is meant).
#include <math.h>

#include "misc_e2e.h"
#include "wgmma_emu/cvt.h"

using namespace misc_e2e;
namespace c = wgemu::cvt;

typedef unsigned short u16;
typedef unsigned long long u64;

// ---- kernel shapes ---------------------------------------------------------------------------
#define K1(NAME, T, C, INS)                                                 \
  __global__ void NAME(const T* __restrict__ a, T* __restrict__ d, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                    \
    if (i < n) {                                                            \
      T r;                                                                  \
      asm volatile(INS " %0, %1;" : "=" C(r) : C(a[i]));                    \
      d[i] = r;                                                             \
    }                                                                       \
  }
#define K2(NAME, T, C, INS)                                                                 \
  __global__ void NAME(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ d, \
                       int n) {                                                             \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (i < n) {                                                                            \
      T r;                                                                                  \
      asm volatile(INS " %0, %1, %2;" : "=" C(r) : C(a[i]), C(b[i]));                       \
      d[i] = r;                                                                             \
    }                                                                                       \
  }
#define K3(NAME, T, C, INS)                                                                        \
  __global__ void NAME(const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ cc, \
                       T* __restrict__ d, int n) {                                                 \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                                           \
    if (i < n) {                                                                                   \
      T r;                                                                                         \
      asm volatile(INS " %0, %1, %2, %3;" : "=" C(r) : C(a[i]), C(b[i]), C(cc[i]));                \
      d[i] = r;                                                                                    \
    }                                                                                              \
  }

// bf16 add / sub / mul
K2(k_add_bf16, u16, "h", "add.bf16")
K2(k_add_rn_bf16, u16, "h", "add.rn.bf16")
K2(k_sub_rn_bf16, u16, "h", "sub.rn.bf16")
K2(k_sub_bf16, u16, "h", "sub.bf16")
K2(k_mul_rn_bf16, u16, "h", "mul.rn.bf16")
K2(k_mul_bf16, u16, "h", "mul.bf16")
K2(k_add_rn_bf16x2, uint32_t, "r", "add.rn.bf16x2")
K2(k_sub_bf16x2, uint32_t, "r", "sub.bf16x2")
K2(k_mul_rn_bf16x2, uint32_t, "r", "mul.rn.bf16x2")

// fma.oob
K3(k_fma_oob_f16, u16, "h", "fma.rn.oob.f16")
K3(k_fma_oob_relu_f16, u16, "h", "fma.rn.oob.relu.f16")
K3(k_fma_oob_bf16, u16, "h", "fma.rn.oob.bf16")
K3(k_fma_oob_relu_bf16, u16, "h", "fma.rn.oob.relu.bf16")
K3(k_fma_oob_f16x2, uint32_t, "r", "fma.rn.oob.f16x2")
K3(k_fma_oob_relu_f16x2, uint32_t, "r", "fma.rn.oob.relu.f16x2")
K3(k_fma_oob_bf16x2, uint32_t, "r", "fma.rn.oob.bf16x2")
K3(k_fma_oob_relu_bf16x2, uint32_t, "r", "fma.rn.oob.relu.bf16x2")

// xorsign
K2(k_min_xs_f32, uint32_t, "r", "min.xorsign.abs.f32")
K2(k_max_xs_f32, uint32_t, "r", "max.xorsign.abs.f32")
K2(k_min_nan_xs_f32, uint32_t, "r", "min.NaN.xorsign.abs.f32")
K2(k_max_ftz_xs_f32, uint32_t, "r", "max.ftz.xorsign.abs.f32")
K2(k_min_ftz_nan_xs_f32, uint32_t, "r", "min.ftz.NaN.xorsign.abs.f32")
K2(k_min_xs_f16, u16, "h", "min.xorsign.abs.f16")
K2(k_max_ftz_xs_f16, u16, "h", "max.ftz.xorsign.abs.f16")
K2(k_max_nan_xs_f16x2, uint32_t, "r", "max.NaN.xorsign.abs.f16x2")
K2(k_max_xs_bf16, u16, "h", "max.xorsign.abs.bf16")
K2(k_min_nan_xs_bf16, u16, "h", "min.NaN.xorsign.abs.bf16")
K2(k_min_xs_bf16x2, uint32_t, "r", "min.xorsign.abs.bf16x2")

// tanh / ex2
K1(k_tanh_bf16, u16, "h", "tanh.approx.bf16")
K1(k_tanh_bf16x2, uint32_t, "r", "tanh.approx.bf16x2")
K1(k_ex2_bf16, u16, "h", "ex2.approx.ftz.bf16")
K1(k_ex2_bf16x2, uint32_t, "r", "ex2.approx.ftz.bf16x2")

// integer SIMD
K2(k_add_u16x2, uint32_t, "r", "add.u16x2")
K2(k_add_s16x2, uint32_t, "r", "add.s16x2")
K2(k_min_u16x2, uint32_t, "r", "min.u16x2")
K2(k_max_u16x2, uint32_t, "r", "max.u16x2")
K2(k_min_s16x2, uint32_t, "r", "min.s16x2")
K2(k_max_s16x2, uint32_t, "r", "max.s16x2")
K2(k_min_relu_s16x2, uint32_t, "r", "min.relu.s16x2")
K2(k_max_relu_s16x2, uint32_t, "r", "max.relu.s16x2")
K2(k_min_relu_s32, uint32_t, "r", "min.relu.s32")
K2(k_max_relu_s32, uint32_t, "r", "max.relu.s32")

// setp.<cmp>.bf16 p, a, b -> d = p
#define KSETP(NAME, CMP)                                                     \
  __global__ void NAME(const u16* __restrict__ a, const u16* __restrict__ b, \
                       uint32_t* __restrict__ d, int n) {                    \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i < n) {                                                             \
      uint32_t r;                                                            \
      asm volatile("{\n\t.reg .pred p;\n\tsetp." CMP                         \
                   ".bf16 p, %1, %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"          \
                   : "=r"(r)                                                 \
                   : "h"(a[i]), "h"(b[i]));                                  \
      d[i] = r;                                                              \
    }                                                                        \
  }
KSETP(k_setp_eq, "eq")
KSETP(k_setp_ne, "ne")
KSETP(k_setp_lt, "lt")
KSETP(k_setp_le, "le")
KSETP(k_setp_gt, "gt")
KSETP(k_setp_ge, "ge")
KSETP(k_setp_equ, "equ")
KSETP(k_setp_neu, "neu")
KSETP(k_setp_ltu, "ltu")
KSETP(k_setp_leu, "leu")
KSETP(k_setp_gtu, "gtu")
KSETP(k_setp_geu, "geu")
KSETP(k_setp_num, "num")
KSETP(k_setp_nan, "nan")

// setp.lt.bf16x2 p|q, a, b -> d = p | q << 1
__global__ void k_setp_lt_x2(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b,
                             uint32_t* __restrict__ d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t rp, rq;
    asm volatile(
        "{\n\t.reg .pred p, q;\n\tsetp.lt.bf16x2 p|q, %2, %3;\n\tselp.u32 %0, 1, 0, p;\n\t"
        "selp.u32 %1, 1, 0, q;\n\t}"
        : "=r"(rp), "=r"(rq)
        : "r"(a[i]), "r"(b[i]));
    d[i] = rp | (rq << 1);
  }
}

// setp.ge.and.bf16 p, a, b, c with c = (f != 0)
__global__ void k_setp_ge_and(const u16* __restrict__ a, const u16* __restrict__ b,
                              const uint32_t* __restrict__ f, uint32_t* __restrict__ d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t r;
    asm volatile(
        "{\n\t.reg .pred p, c;\n\tsetp.ne.u32 c, %3, 0;\n\tsetp.ge.and.bf16 p, %1, %2, c;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(r)
        : "h"(a[i]), "h"(b[i]), "r"(f[i]));
    d[i] = r;
  }
}

// setp.ltu.or.bf16x2 p|q, a, b, !c -> d = p | q << 1
__global__ void k_setp_ltu_or_x2(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b,
                                 const uint32_t* __restrict__ f, uint32_t* __restrict__ d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t rp, rq;
    asm volatile(
        "{\n\t.reg .pred p, q, c;\n\tsetp.ne.u32 c, %4, 0;\n\tsetp.ltu.or.bf16x2 p|q, %2, %3, "
        "!c;\n\t"
        "selp.u32 %0, 1, 0, p;\n\tselp.u32 %1, 1, 0, q;\n\t}"
        : "=r"(rp), "=r"(rq)
        : "r"(a[i]), "r"(b[i]), "r"(f[i]));
    d[i] = rp | (rq << 1);
  }
}

// set.<cmp>.<dtype>.<stype> d, a, b
#define KSET(NAME, TD, CD, TS, CS, INS)                                                        \
  __global__ void NAME(const TS* __restrict__ a, const TS* __restrict__ b, TD* __restrict__ d, \
                       int n) {                                                                \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                                       \
    if (i < n) {                                                                               \
      TD r;                                                                                    \
      asm volatile(INS " %0, %1, %2;" : "=" CD(r) : CS(a[i]), CS(b[i]));                       \
      d[i] = r;                                                                                \
    }                                                                                          \
  }
KSET(k_set_eq_u32_bf16, uint32_t, "r", u16, "h", "set.eq.u32.bf16")
KSET(k_set_lt_s16_bf16, u16, "h", u16, "h", "set.lt.s16.bf16")
KSET(k_set_equ_u16_bf16, u16, "h", u16, "h", "set.equ.u16.bf16")
KSET(k_set_gtu_bf16_bf16, u16, "h", u16, "h", "set.gtu.bf16.bf16")
KSET(k_set_le_bf16_f16, u16, "h", u16, "h", "set.le.bf16.f16")
KSET(k_set_ne_bf16_f32, u16, "h", uint32_t, "r", "set.ne.bf16.f32")
KSET(k_set_num_u32_bf16x2, uint32_t, "r", uint32_t, "r", "set.num.u32.bf16x2")
KSET(k_set_ge_bf16x2_bf16x2, uint32_t, "r", uint32_t, "r", "set.ge.bf16x2.bf16x2")

// set.lt.xor.s32.bf16 d, a, b, c with c = (f != 0)
__global__ void k_set_lt_xor_s32(const u16* __restrict__ a, const u16* __restrict__ b,
                                 const uint32_t* __restrict__ f, uint32_t* __restrict__ d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t r;
    asm volatile(
        "{\n\t.reg .pred c;\n\tsetp.ne.u32 c, %3, 0;\n\tset.lt.xor.s32.bf16 %0, %1, %2, c;\n\t}"
        : "=r"(r)
        : "h"(a[i]), "h"(b[i]), "r"(f[i]));
    d[i] = r;
  }
}

// ---- launch helpers ------------------------------------------------------------------------

template <class TD, class TA>
std::vector<TD> run(void (*k)(const TA*, TD*, int), const std::vector<TA>& a) {
  Dev<TA> da(a);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}
template <class TD, class TA>
std::vector<TD> run(void (*k)(const TA*, const TA*, TD*, int), const std::vector<TA>& a,
                    const std::vector<TA>& b) {
  Dev<TA> da(a), db(b);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, db.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}
template <class TD, class TA>
std::vector<TD> run(void (*k)(const TA*, const TA*, const TA*, TD*, int), const std::vector<TA>& a,
                    const std::vector<TA>& b, const std::vector<TA>& cc) {
  Dev<TA> da(a), db(b), dc(cc);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, db.p, dc.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}
template <class TD, class TA>
std::vector<TD> run_flag(void (*k)(const TA*, const TA*, const uint32_t*, TD*, int),
                    const std::vector<TA>& a, const std::vector<TA>& b,
                    const std::vector<uint32_t>& f) {
  Dev<TA> da(a), db(b);
  Dev<uint32_t> df(f);
  Dev<TD> dd(a.size());
  k<<<blocks_for(a.size()), 256>>>(da.p, db.p, df.p, dd.p, (int)a.size());
  MISC_SYNC();
  return dd.get();
}

// ---- 16-bit formats --------------------------------------------------------------------------

struct F16Kind {
  bool bf;  // bf16 when true, else f16
  uint32_t sign() const { return 0x8000u; }
  bool nan(uint32_t h) const { return bf ? nan_bf16(h) : nan_f16(h); }
  bool inf(uint32_t h) const { return (h & 0x7FFFu) == (bf ? 0x7F80u : 0x7C00u); }
  c::Dec dec(uint32_t h) const { return bf ? c::decode_bf16(h) : c::decode_f16(h); }
  c::Fmt fmt() const { return bf ? c::fmt_bf16() : c::fmt_f16(); }
  uint32_t exp_mask() const { return bf ? 0x7F80u : 0x7C00u; }
  double val(uint32_t h) const {
    return (double)u2f(bf ? c::bf16_to_f32(h) : c::f16_to_f32(h & 0xFFFFu));
  }
  bool same(uint32_t g, uint32_t w) const {
    return bf ? same_bf16(g, w, true) : same_f16(g, w, true);
  }
};
static const F16Kind kBf{true}, kHf{false};
static const uint32_t kQNaN16 = 0x7FFFu;  // canonical NaN of both formats
static const uint32_t kOobNaN16 = 0x7FFFu;

// Exact a * b + c rounded once to nearest even (IEEE 754 fusedMultiplyAdd, no flush).
static uint32_t fma16_ref(const F16Kind& k, uint32_t a, uint32_t b, uint32_t cc) {
  const c::Dec da = k.dec(a), db = k.dec(b), dc = k.dec(cc);
  if (da.kind == 2 || db.kind == 2 || dc.kind == 2) return kQNaN16;
  const bool pneg = da.neg != db.neg;
  const bool azero = da.kind == 0 && da.mag == 0, bzero = db.kind == 0 && db.mag == 0;
  if (da.kind == 1 || db.kind == 1) {
    if (azero || bzero) return kQNaN16;                  // inf * 0
    if (dc.kind == 1 && dc.neg != pneg) return kQNaN16;  // inf - inf
    return (pneg ? 0x8000u : 0u) | k.exp_mask();
  }
  if (dc.kind == 1) return (dc.neg ? 0x8000u : 0u) | k.exp_mask();
  uint64_t pm = da.mag * db.mag;
  int pe = da.exp2 + db.exp2;
  uint64_t cm = dc.mag;
  int ce = dc.exp2;
  if (pm == 0 && cm == 0) return (pneg && dc.neg) ? 0x8000u : 0u;
  if (pm == 0) return c::round_to_fmt(k.fmt(), dc.neg, cm, ce, c::RN, false);
  if (cm == 0) return c::round_to_fmt(k.fmt(), pneg, pm, pe, c::RN, false);
  // A term more than 100 bits below the other only contributes its sign as a sticky bit.
  if (ce < pe - 100) {
    cm = 1;
    ce = pe - 100;
  } else if (pe < ce - 100) {
    pm = 1;
    pe = ce - 100;
  }
  const int e0 = pe < ce ? pe : ce;
  const unsigned __int128 P = (unsigned __int128)pm << (pe - e0);
  const unsigned __int128 C = (unsigned __int128)cm << (ce - e0);
  bool neg;
  unsigned __int128 m;
  if (pneg == dc.neg) {
    neg = pneg;
    m = P + C;
  } else if (P >= C) {
    neg = pneg;
    m = P - C;
  } else {
    neg = dc.neg;
    m = C - P;
  }
  if (m == 0) return 0u;  // exact cancellation: +0 under .rn
  int e = e0;
  while (m >> 64) {
    m = (m >> 1) | (m & 1u);
    ++e;
  }
  return c::round_to_fmt(k.fmt(), neg, (uint64_t)m, e, c::RN, false);
}

// fma.rn.oob{.relu}: OOB-NaN operand -> +0; .relu: negative -> +0, NaN -> canonical NaN.
static uint32_t fma_oob_ref(const F16Kind& k, uint32_t a, uint32_t b, uint32_t cc, bool relu) {
  if (a == kOobNaN16 || b == kOobNaN16 || cc == kOobNaN16) return 0u;
  const uint32_t r = fma16_ref(k, a, b, cc);
  if (relu && !k.nan(r) && (r & 0x8000u)) return 0u;
  return r;
}

static bool eq16_relu(const F16Kind& k, uint32_t g, uint32_t w) {
  return k.same(g, w) || (w == 0u && g == 0x8000u);
}

// min / max with .xorsign.abs (ptx.txt 9306-9326) on one element.
struct XsFmt {
  uint32_t sbit, emask;
  bool (*nan)(uint32_t);
};
static const XsFmt kXs32{0x80000000u, 0x7F800000u, nan32};
static const XsFmt kXsBf{0x8000u, 0x7F80u, nan_bf16};
static const XsFmt kXsHf{0x8000u, 0x7C00u, nan_f16};

// Sets *nan when the result is NaN (then compared by class only).
static uint32_t xorsign_ref(const XsFmt& f, uint32_t a, uint32_t b, bool is_max, bool nan_mode,
                            bool ftz, bool* nan) {
  uint32_t ma = a & (f.sbit - 1), mb = b & (f.sbit - 1);
  if (ftz) {
    if ((ma & f.emask) == 0) ma = 0;
    if ((mb & f.emask) == 0) mb = 0;
  }
  const bool na = f.nan(ma), nb = f.nan(mb);
  *nan = (na && nb) || ((na || nb) && nan_mode);
  if (*nan) return f.sbit - 1;
  uint32_t d;
  if (na) {
    d = mb;
  } else if (nb) {
    d = ma;
  } else {  // non-NaN magnitudes order like their encodings
    d = is_max ? (ma > mb ? ma : mb) : (ma < mb ? ma : mb);
  }
  return d | ((a ^ b) & f.sbit);
}

// tanh.approx.bf16: corner cases exact, otherwise absolute error <= 2^-8 (ptx.txt 10741-10742).
static bool tanh_ok(uint32_t a, uint32_t g) {
  a &= 0xFFFFu;
  g &= 0xFFFFu;
  if (nan_bf16(a)) return nan_bf16(g);
  if ((a & 0x7FFFu) == 0) return g == a;
  if ((a & 0x7FFFu) == 0x7F80u) return g == ((a & 0x8000u) | 0x3F80u);
  if (nan_bf16(g) || (g & 0x7FFFu) >= 0x7F80u) return false;
  return fabs(kBf.val(g) - tanh(kBf.val(a))) <= ldexp(1.0, -8);
}

// ex2.approx.ftz.bf16: corner cases exact (ptx.txt 10788-10800), subnormal inputs act as 0,
// subnormal results flush to +0, otherwise relative error <= 2^-7.
static bool ex2_ok(uint32_t a, uint32_t g) {
  a &= 0xFFFFu;
  g &= 0xFFFFu;
  if (nan_bf16(a)) return nan_bf16(g);
  if (a == 0xFF80u) return g == 0u;
  if (a == 0x7F80u) return g == 0x7F80u;
  if ((a & 0x7F80u) == 0) return g == 0x3F80u;
  if (nan_bf16(g)) return false;
  const double e = exp2(kBf.val(a));
  const double tiny = ldexp(1.0, -126);
  if (e < tiny) return g == 0u;
  if (g == 0u) return e < tiny * (1.0 + ldexp(1.0, -7));  // approximation crossed the flush
  if (e >= ldexp(1.0, 128)) return g == 0x7F80u;
  if (g & 0x8000u) return false;
  return fabs(kBf.val(g) - e) <= ldexp(e, -7);
}

// Comparison operators of set / setp (ptx.txt 11768-11775).
enum Cmp { EQ, NE, LT, LE, GT, GE, EQU, NEU, LTU, LEU, GTU, GEU, NUM, NAN_ };
static bool cmp_ref(int op, double x, double y, bool nx, bool ny) {
  const bool u = nx || ny;
  switch (op) {
    case EQ:
      return !u && x == y;
    case NE:
      return !u && x != y;
    case LT:
      return !u && x < y;
    case LE:
      return !u && x <= y;
    case GT:
      return !u && x > y;
    case GE:
      return !u && x >= y;
    case EQU:
      return u || x == y;
    case NEU:
      return u || x != y;
    case LTU:
      return u || x < y;
    case LEU:
      return u || x <= y;
    case GTU:
      return u || x > y;
    case GEU:
      return u || x >= y;
    case NUM:
      return !u;
    default:
      return u;
  }
}
static bool cmp_bf(int op, uint32_t a, uint32_t b) {
  return cmp_ref(op, kBf.val(a), kBf.val(b), nan_bf16(a & 0xFFFFu), nan_bf16(b & 0xFFFFu));
}

// ---- inputs ----------------------------------------------------------------------------------

static const std::vector<uint32_t> kBfSpecial = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x3F81,
    0xBF81, 0x3F7F, 0x4000, 0x3F00, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0,
    0x7F81, 0x7FFF, 0xFFFF, 0x4040, 0xC040, 0x0100, 0x3C00, 0x4300, 0x4380};
static const std::vector<uint32_t> kHfSpecial = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x8400, 0x3C00, 0xBC00, 0x3C01,
    0xBC01, 0x3BFF, 0x4000, 0x3800, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00,
    0x7C01, 0x7FFF, 0xFFFF, 0x4200, 0xC200, 0x0800, 0x6800, 0x1400, 0x0200};

// A value a few hundred encodings away from x (same binade region), random sign.
static uint32_t near16(Rng& r, uint32_t x, int mbits) {
  const int span = 1 << (mbits + 2);
  int m = (int)(x & 0x7FFFu) + (int)r.below(2 * span) - span;
  m = m < 0 ? 0 : (m > 0x7FFF ? 0x7FFF : m);
  return (uint32_t)m | (r.below(2) ? 0x8000u : 0u);
}

struct Pairs {
  std::vector<u16> a, b;
};
// special x special, equal and sign-flipped pairs, random pairs and near pairs.
static Pairs make_pairs(const std::vector<uint32_t>& sp, int mbits, int n, uint64_t seed) {
  Pairs p;
  auto push = [&](uint32_t x, uint32_t y) {
    p.a.push_back((u16)x);
    p.b.push_back((u16)y);
  };
  for (uint32_t x : sp)
    for (uint32_t y : sp) push(x, y);
  Rng r(seed);
  for (int i = 0; i < n; ++i) {
    const uint32_t x = r.u32() & 0xFFFFu;
    push(x, r.u32() & 0xFFFFu);
    push(x, near16(r, x, mbits));
    if (i % 8 == 0) {
      push(x, x);
      push(x, x ^ 0x8000u);
    }
  }
  return p;
}

struct Triples {
  std::vector<u16> a, b, c;
};
// special^3, random triples, and c close to -(a*b) to exercise cancellation.
static Triples make_triples(const F16Kind& k, const std::vector<uint32_t>& sp, int mbits, int n,
                            uint64_t seed) {
  Triples t;
  auto push = [&](uint32_t x, uint32_t y, uint32_t z) {
    t.a.push_back((u16)x);
    t.b.push_back((u16)y);
    t.c.push_back((u16)z);
  };
  for (uint32_t x : sp)
    for (uint32_t y : sp)
      for (uint32_t z : sp) push(x, y, z);
  Rng r(seed);
  for (int i = 0; i < n; ++i) {
    const uint32_t x = r.u32() & 0xFFFFu, y = r.u32() & 0xFFFFu;
    push(x, y, r.u32() & 0xFFFFu);
    push(x, y, near16(r, x, mbits));
    const uint32_t prod = fma16_ref(k, x, y, 0u);
    if (!k.nan(prod) && !k.inf(prod)) {
      const int dm = (int)r.below(5) - 2;
      int m = (int)(prod & 0x7FFFu) + dm;
      m = m < 0 ? 0 : (m > 0x7FFF ? 0x7FFF : m);
      push(x, y, (uint32_t)m | ((prod ^ 0x8000u) & 0x8000u));
    }
  }
  return t;
}

// Packs element i and a partner element into one x2 word (low half = element i).
static std::vector<uint32_t> pack_x2(const std::vector<u16>& v) {
  std::vector<uint32_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    o[i] = v[i] | ((uint32_t)v[(i * 7919 + 13) % v.size()] << 16);
  return o;
}

static uint32_t hi16(uint32_t x) { return x >> 16; }
static uint32_t lo16(uint32_t x) { return x & 0xFFFFu; }

struct In3 {
  u64 a, b, c;
};

// One item: got[i] against want(i) with eq(got, want, i); inputs printed through in(i).
template <class T, class Want, class Eq, class In>
void verify(const char* name, const std::vector<T>& got, Want want, Eq eq, In in) {
  Item it(name);
  for (size_t i = 0; i < got.size(); ++i) {
    const u64 g = (u64)got[i], w = (u64)want(i);
    const In3 x = in(i);
    it.check(eq(g, w, i), "a=%llx b=%llx c=%llx got=%llx want=%llx", x.a, x.b, x.c, g, w);
  }
}

int main() {
  const Pairs bp = make_pairs(kBfSpecial, 7, 65536, 0xB16);
  const Pairs hp = make_pairs(kHfSpecial, 10, 65536, 0xF16);
  const std::vector<uint32_t> bpa2 = pack_x2(bp.a), bpb2 = pack_x2(bp.b);
  const std::vector<uint32_t> hpa2 = pack_x2(hp.a), hpb2 = pack_x2(hp.b);
  auto in_bp = [&](size_t i) { return In3{bp.a[i], bp.b[i], 0}; };
  auto in_bp2 = [&](size_t i) { return In3{bpa2[i], bpb2[i], 0}; };
  auto in_hp = [&](size_t i) { return In3{hp.a[i], hp.b[i], 0}; };
  auto in_hp2 = [&](size_t i) { return In3{hpa2[i], hpb2[i], 0}; };

  // ------------------------------------------------------------ bf16 add / sub / mul
  {
    auto ref = [](int op, uint32_t a, uint32_t b) {
      if (op == 0) return fma16_ref(kBf, a, 0x3F80u, b);  // a + b
      if (op == 1) return fma16_ref(kBf, b, 0xBF80u, a);  // a - b
      return fma16_ref(kBf, a, b, 0x8000u);               // a * b
    };
    auto eqb = [](u64 g, u64 w, size_t) { return kBf.same((uint32_t)g, (uint32_t)w); };
    auto eqb2 = [](u64 g, u64 w, size_t) {
      return kBf.same(lo16((uint32_t)g), lo16((uint32_t)w)) &&
             kBf.same(hi16((uint32_t)g), hi16((uint32_t)w));
    };
    struct S {
      const char* name;
      void (*k)(const u16*, const u16*, u16*, int);
      int op;
    };
    for (const S& s : {S{"misc.e2e.arith.add.bf16", k_add_bf16, 0},
                       S{"misc.e2e.arith.add.rn.bf16", k_add_rn_bf16, 0},
                       S{"misc.e2e.arith.sub.rn.bf16", k_sub_rn_bf16, 1},
                       S{"misc.e2e.arith.sub.bf16", k_sub_bf16, 1},
                       S{"misc.e2e.arith.mul.rn.bf16", k_mul_rn_bf16, 2},
                       S{"misc.e2e.arith.mul.bf16", k_mul_bf16, 2}}) {
      verify(
          s.name, run(s.k, bp.a, bp.b), [&](size_t i) { return ref(s.op, bp.a[i], bp.b[i]); }, eqb,
          in_bp);
    }
    struct S2 {
      const char* name;
      void (*k)(const uint32_t*, const uint32_t*, uint32_t*, int);
      int op;
    };
    for (const S2& s : {S2{"misc.e2e.arith.add.rn.bf16x2", k_add_rn_bf16x2, 0},
                        S2{"misc.e2e.arith.sub.bf16x2", k_sub_bf16x2, 1},
                        S2{"misc.e2e.arith.mul.rn.bf16x2", k_mul_rn_bf16x2, 2}}) {
      verify(
          s.name, run(s.k, bpa2, bpb2),
          [&](size_t i) {
            return ref(s.op, lo16(bpa2[i]), lo16(bpb2[i])) |
                   (ref(s.op, hi16(bpa2[i]), hi16(bpb2[i])) << 16);
          },
          eqb2, in_bp2);
    }
  }

  // ------------------------------------------------------------ fma.rn.oob{.relu}
  for (int bf = 0; bf < 2; ++bf) {
    const F16Kind& k = bf ? kBf : kHf;
    const Triples t =
        make_triples(k, bf ? kBfSpecial : kHfSpecial, bf ? 7 : 10, 65536, bf ? 0xFAB : 0xFA6);
    const std::vector<uint32_t> a2 = pack_x2(t.a), b2 = pack_x2(t.b), c2 = pack_x2(t.c);
    auto in1 = [&](size_t i) { return In3{t.a[i], t.b[i], t.c[i]}; };
    auto in2 = [&](size_t i) { return In3{a2[i], b2[i], c2[i]}; };
    for (int relu = 0; relu < 2; ++relu) {
      auto eq1 = [&](u64 g, u64 w, size_t) {
        return relu ? eq16_relu(k, (uint32_t)g, (uint32_t)w) : k.same((uint32_t)g, (uint32_t)w);
      };
      auto eq2 = [&](u64 g, u64 w, size_t i) {
        return eq1(lo16((uint32_t)g), lo16((uint32_t)w), i) &&
               eq1(hi16((uint32_t)g), hi16((uint32_t)w), i);
      };
      void (*k1)(const u16*, const u16*, const u16*, u16*, int) =
          bf ? (relu ? k_fma_oob_relu_bf16 : k_fma_oob_bf16)
             : (relu ? k_fma_oob_relu_f16 : k_fma_oob_f16);
      void (*k2)(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, int) =
          bf ? (relu ? k_fma_oob_relu_bf16x2 : k_fma_oob_bf16x2)
             : (relu ? k_fma_oob_relu_f16x2 : k_fma_oob_f16x2);
      const char* n1 =
          bf ? (relu ? "misc.e2e.arith.fma.rn.oob.relu.bf16" : "misc.e2e.arith.fma.rn.oob.bf16")
             : (relu ? "misc.e2e.arith.fma.rn.oob.relu.f16" : "misc.e2e.arith.fma.rn.oob.f16");
      const char* n2 =
          bf ? (relu ? "misc.e2e.arith.fma.rn.oob.relu.bf16x2" : "misc.e2e.arith.fma.rn.oob.bf16x2")
             : (relu ? "misc.e2e.arith.fma.rn.oob.relu.f16x2" : "misc.e2e.arith.fma.rn.oob.f16x2");
      verify(
          n1, run(k1, t.a, t.b, t.c),
          [&](size_t i) { return fma_oob_ref(k, t.a[i], t.b[i], t.c[i], relu); }, eq1, in1);
      verify(
          n2, run(k2, a2, b2, c2),
          [&](size_t i) {
            return fma_oob_ref(k, lo16(a2[i]), lo16(b2[i]), lo16(c2[i]), relu) |
                   (fma_oob_ref(k, hi16(a2[i]), hi16(b2[i]), hi16(c2[i]), relu) << 16);
          },
          eq2, in2);
    }
  }

  // ------------------------------------------------------------ xorsign
  {
    const std::vector<uint32_t> fa = f32_inputs(8192, 0x5165);
    std::vector<uint32_t> xa, xb;
    for (size_t i = 0; i < fa.size(); ++i) {
      const uint32_t y = fa[(i * 7919 + 13) % fa.size()];
      xa.insert(xa.end(), {fa[i], fa[i], fa[i]});
      xb.insert(xb.end(), {y, fa[i] ^ 0x80000000u, fa[i] ^ 1u});
    }
    auto in32 = [&](size_t i) { return In3{xa[i], xb[i], 0}; };
    struct X32 {
      const char* name;
      void (*k)(const uint32_t*, const uint32_t*, uint32_t*, int);
      bool is_max, nan_mode, ftz;
    };
    for (const X32& s :
         {X32{"misc.e2e.arith.min.xorsign.abs.f32", k_min_xs_f32, false, false, false},
          X32{"misc.e2e.arith.max.xorsign.abs.f32", k_max_xs_f32, true, false, false},
          X32{"misc.e2e.arith.min.NaN.xorsign.abs.f32", k_min_nan_xs_f32, false, true, false},
          X32{"misc.e2e.arith.max.ftz.xorsign.abs.f32", k_max_ftz_xs_f32, true, false, true},
          X32{"misc.e2e.arith.min.ftz.NaN.xorsign.abs.f32", k_min_ftz_nan_xs_f32, false, true,
              true}}) {
      verify(
          s.name, run(s.k, xa, xb),
          [&](size_t i) {
            bool nan;
            return xorsign_ref(kXs32, xa[i], xb[i], s.is_max, s.nan_mode, s.ftz, &nan);
          },
          [](u64 g, u64 w, size_t) { return same32((uint32_t)g, (uint32_t)w, true); }, in32);
    }
    struct X16 {
      const char* name;
      void (*k)(const u16*, const u16*, u16*, int);
      bool bf, is_max, nan_mode, ftz;
    };
    for (const X16& s :
         {X16{"misc.e2e.arith.min.xorsign.abs.f16", k_min_xs_f16, false, false, false, false},
          X16{"misc.e2e.arith.max.ftz.xorsign.abs.f16", k_max_ftz_xs_f16, false, true, false, true},
          X16{"misc.e2e.arith.max.xorsign.abs.bf16", k_max_xs_bf16, true, true, false, false},
          X16{"misc.e2e.arith.min.NaN.xorsign.abs.bf16", k_min_nan_xs_bf16, true, false, true,
              false}}) {
      const Pairs& p = s.bf ? bp : hp;
      const F16Kind& k = s.bf ? kBf : kHf;
      verify(
          s.name, run(s.k, p.a, p.b),
          [&](size_t i) {
            bool nan;
            return xorsign_ref(s.bf ? kXsBf : kXsHf, p.a[i], p.b[i], s.is_max, s.nan_mode, s.ftz,
                               &nan);
          },
          [&](u64 g, u64 w, size_t) { return k.same((uint32_t)g, (uint32_t)w); },
          [&](size_t i) { return s.bf ? in_bp(i) : in_hp(i); });
    }
    struct X16x2 {
      const char* name;
      void (*k)(const uint32_t*, const uint32_t*, uint32_t*, int);
      bool bf, is_max, nan_mode;
    };
    for (const X16x2& s :
         {X16x2{"misc.e2e.arith.max.NaN.xorsign.abs.f16x2", k_max_nan_xs_f16x2, false, true, true},
          X16x2{"misc.e2e.arith.min.xorsign.abs.bf16x2", k_min_xs_bf16x2, true, false, false}}) {
      const std::vector<uint32_t>& a2 = s.bf ? bpa2 : hpa2;
      const std::vector<uint32_t>& b2 = s.bf ? bpb2 : hpb2;
      const F16Kind& k = s.bf ? kBf : kHf;
      const XsFmt& f = s.bf ? kXsBf : kXsHf;
      verify(
          s.name, run(s.k, a2, b2),
          [&](size_t i) {
            bool nan;
            return xorsign_ref(f, lo16(a2[i]), lo16(b2[i]), s.is_max, s.nan_mode, false, &nan) |
                   (xorsign_ref(f, hi16(a2[i]), hi16(b2[i]), s.is_max, s.nan_mode, false, &nan)
                    << 16);
          },
          [&](u64 g, u64 w, size_t) {
            return k.same(lo16((uint32_t)g), lo16((uint32_t)w)) &&
                   k.same(hi16((uint32_t)g), hi16((uint32_t)w));
          },
          [&](size_t i) { return s.bf ? in_bp2(i) : in_hp2(i); });
    }
  }

  // ------------------------------------------------------------ tanh / ex2
  {
    std::vector<u16> all(1 << 16);
    for (int i = 0; i < (1 << 16); ++i) all[i] = (u16)i;
    const std::vector<uint32_t> all2 = pack_x2(all);
    auto in1 = [&](size_t i) { return In3{all[i], 0, 0}; };
    auto in2 = [&](size_t i) { return In3{all2[i], 0, 0}; };
    // want: exact value rounded to bf16, for the mismatch lines only.
    auto shown = [](bool is_tanh, uint32_t a) {
      if (nan_bf16(a)) return 0x7FFFu;
      const double x = kBf.val(a);
      return c::f32_to_bf16(f2u((float)(is_tanh ? tanh(x) : exp2(x))), c::RN);
    };
    for (int is_tanh = 0; is_tanh < 2; ++is_tanh) {
      auto ok = is_tanh ? tanh_ok : ex2_ok;
      verify(
          is_tanh ? "misc.e2e.arith.tanh.approx.bf16" : "misc.e2e.arith.ex2.approx.ftz.bf16",
          run(is_tanh ? k_tanh_bf16 : k_ex2_bf16, all),
          [&](size_t i) { return shown(is_tanh, all[i]); },
          [&](u64 g, u64, size_t i) { return ok(all[i], (uint32_t)g); }, in1);
      verify(
          is_tanh ? "misc.e2e.arith.tanh.approx.bf16x2" : "misc.e2e.arith.ex2.approx.ftz.bf16x2",
          run(is_tanh ? k_tanh_bf16x2 : k_ex2_bf16x2, all2),
          [&](size_t i) {
            return shown(is_tanh, lo16(all2[i])) | (shown(is_tanh, hi16(all2[i])) << 16);
          },
          [&](u64 g, u64, size_t i) {
            return ok(lo16(all2[i]), lo16((uint32_t)g)) && ok(hi16(all2[i]), hi16((uint32_t)g));
          },
          in2);
    }
  }

  // ------------------------------------------------------------ setp / set
  {
    auto eq = [](u64 g, u64 w, size_t) { return g == w; };
    struct SP {
      const char* name;
      void (*k)(const u16*, const u16*, uint32_t*, int);
      int op;
    };
    for (const SP& s : {SP{"misc.e2e.arith.setp.eq.bf16", k_setp_eq, EQ},
                        SP{"misc.e2e.arith.setp.ne.bf16", k_setp_ne, NE},
                        SP{"misc.e2e.arith.setp.lt.bf16", k_setp_lt, LT},
                        SP{"misc.e2e.arith.setp.le.bf16", k_setp_le, LE},
                        SP{"misc.e2e.arith.setp.gt.bf16", k_setp_gt, GT},
                        SP{"misc.e2e.arith.setp.ge.bf16", k_setp_ge, GE},
                        SP{"misc.e2e.arith.setp.equ.bf16", k_setp_equ, EQU},
                        SP{"misc.e2e.arith.setp.neu.bf16", k_setp_neu, NEU},
                        SP{"misc.e2e.arith.setp.ltu.bf16", k_setp_ltu, LTU},
                        SP{"misc.e2e.arith.setp.leu.bf16", k_setp_leu, LEU},
                        SP{"misc.e2e.arith.setp.gtu.bf16", k_setp_gtu, GTU},
                        SP{"misc.e2e.arith.setp.geu.bf16", k_setp_geu, GEU},
                        SP{"misc.e2e.arith.setp.num.bf16", k_setp_num, NUM},
                        SP{"misc.e2e.arith.setp.nan.bf16", k_setp_nan, NAN_}}) {
      verify(
          s.name, run(s.k, bp.a, bp.b),
          [&](size_t i) { return (uint32_t)cmp_bf(s.op, bp.a[i], bp.b[i]); }, eq, in_bp);
    }
    verify(
        "misc.e2e.arith.setp.lt.bf16x2", run(k_setp_lt_x2, bpa2, bpb2),
        [&](size_t i) {
          return (uint32_t)cmp_bf(LT, lo16(bpa2[i]), lo16(bpb2[i])) |
                 ((uint32_t)cmp_bf(LT, hi16(bpa2[i]), hi16(bpb2[i])) << 1);
        },
        eq, in_bp2);
    std::vector<uint32_t> flag(bp.a.size());
    {
      Rng r(0xF1A6);
      for (auto& f : flag) f = r.below(2);
    }
    verify(
        "misc.e2e.arith.setp.ge.and.bf16", run_flag(k_setp_ge_and, bp.a, bp.b, flag),
        [&](size_t i) { return (uint32_t)(cmp_bf(GE, bp.a[i], bp.b[i]) && flag[i]); }, eq,
        [&](size_t i) { return In3{bp.a[i], bp.b[i], flag[i]}; });
    verify(
        "misc.e2e.arith.setp.ltu.or.bf16x2", run_flag(k_setp_ltu_or_x2, bpa2, bpb2, flag),
        [&](size_t i) {
          const bool nc = !flag[i];
          return (uint32_t)(cmp_bf(LTU, lo16(bpa2[i]), lo16(bpb2[i])) || nc) |
                 ((uint32_t)(cmp_bf(LTU, hi16(bpa2[i]), hi16(bpb2[i])) || nc) << 1);
        },
        eq, [&](size_t i) { return In3{bpa2[i], bpb2[i], flag[i]}; });
    verify(
        "misc.e2e.arith.set.lt.xor.s32.bf16", run_flag(k_set_lt_xor_s32, bp.a, bp.b, flag),
        [&](size_t i) { return (cmp_bf(LT, bp.a[i], bp.b[i]) != (flag[i] != 0)) ? ~0u : 0u; }, eq,
        [&](size_t i) { return In3{bp.a[i], bp.b[i], flag[i]}; });

    verify(
        "misc.e2e.arith.set.eq.u32.bf16", run(k_set_eq_u32_bf16, bp.a, bp.b),
        [&](size_t i) { return cmp_bf(EQ, bp.a[i], bp.b[i]) ? 0xFFFFFFFFu : 0u; }, eq, in_bp);
    verify(
        "misc.e2e.arith.set.lt.s16.bf16", run(k_set_lt_s16_bf16, bp.a, bp.b),
        [&](size_t i) { return cmp_bf(LT, bp.a[i], bp.b[i]) ? 0xFFFFu : 0u; }, eq, in_bp);
    verify(
        "misc.e2e.arith.set.equ.u16.bf16", run(k_set_equ_u16_bf16, bp.a, bp.b),
        [&](size_t i) { return cmp_bf(EQU, bp.a[i], bp.b[i]) ? 0xFFFFu : 0u; }, eq, in_bp);
    verify(
        "misc.e2e.arith.set.gtu.bf16.bf16", run(k_set_gtu_bf16_bf16, bp.a, bp.b),
        [&](size_t i) { return cmp_bf(GTU, bp.a[i], bp.b[i]) ? 0x3F80u : 0u; }, eq, in_bp);
    verify(
        "misc.e2e.arith.set.le.bf16.f16", run(k_set_le_bf16_f16, hp.a, hp.b),
        [&](size_t i) {
          return cmp_ref(LE, kHf.val(hp.a[i]), kHf.val(hp.b[i]), nan_f16(hp.a[i]), nan_f16(hp.b[i]))
                     ? 0x3F80u
                     : 0u;
        },
        eq, in_hp);
    {
      const std::vector<uint32_t> fa = f32_inputs(4096, 0x5E7);
      std::vector<uint32_t> xa, xb;
      for (size_t i = 0; i < fa.size(); ++i) {
        xa.insert(xa.end(), {fa[i], fa[i], fa[i]});
        xb.insert(xb.end(), {fa[(i * 7919 + 13) % fa.size()], fa[i], fa[i] ^ 0x80000000u});
      }
      verify(
          "misc.e2e.arith.set.ne.bf16.f32", run(k_set_ne_bf16_f32, xa, xb),
          [&](size_t i) {
            return cmp_ref(NE, (double)u2f(xa[i]), (double)u2f(xb[i]), nan32(xa[i]), nan32(xb[i]))
                       ? 0x3F80u
                       : 0u;
          },
          eq, [&](size_t i) { return In3{xa[i], xb[i], 0}; });
    }
    verify(
        "misc.e2e.arith.set.num.u32.bf16x2", run(k_set_num_u32_bf16x2, bpa2, bpb2),
        [&](size_t i) {
          return (cmp_bf(NUM, lo16(bpa2[i]), lo16(bpb2[i])) ? 0xFFFFu : 0u) |
                 (cmp_bf(NUM, hi16(bpa2[i]), hi16(bpb2[i])) ? 0xFFFF0000u : 0u);
        },
        eq, in_bp2);
    verify(
        "misc.e2e.arith.set.ge.bf16x2.bf16x2", run(k_set_ge_bf16x2_bf16x2, bpa2, bpb2),
        [&](size_t i) {
          return (cmp_bf(GE, lo16(bpa2[i]), lo16(bpb2[i])) ? 0x3F80u : 0u) |
                 (cmp_bf(GE, hi16(bpa2[i]), hi16(bpb2[i])) ? 0x3F800000u : 0u);
        },
        eq, in_bp2);
  }

  // ------------------------------------------------------------ integer SIMD
  {
    const uint32_t h[] = {0x0000u, 0x0001u, 0x00FFu, 0x7FFFu, 0x8000u, 0x8001u, 0xFFFFu};
    std::vector<uint32_t> w;
    for (uint32_t x : h)
      for (uint32_t y : h) w.push_back(x | (y << 16));
    std::vector<uint32_t> ia, ib;
    for (uint32_t x : w)
      for (uint32_t y : w) {
        ia.push_back(x);
        ib.push_back(y);
      }
    Rng r(0x51D);
    for (int i = 0; i < 65536; ++i) {
      ia.push_back(r.u32());
      ib.push_back(r.u32());
    }
    auto in = [&](size_t i) { return In3{ia[i], ib[i], 0}; };
    auto eq = [](u64 g, u64 w2, size_t) { return g == w2; };
    // op: 0 add, 1 min, 2 max; sgn: signed halves; relu: clamp at 0.
    auto simd = [](uint32_t a, uint32_t b, int op, bool sgn, bool relu) {
      uint32_t o = 0;
      for (int s = 0; s < 32; s += 16) {
        const uint32_t x = (a >> s) & 0xFFFFu, y = (b >> s) & 0xFFFFu;
        uint32_t r16;
        if (op == 0) {
          r16 = (x + y) & 0xFFFFu;
        } else if (sgn) {
          const int sx = (int16_t)x, sy = (int16_t)y;
          int v = op == 1 ? (sx < sy ? sx : sy) : (sx > sy ? sx : sy);
          if (relu && v < 0) v = 0;
          r16 = (uint32_t)v & 0xFFFFu;
        } else {
          r16 = op == 1 ? (x < y ? x : y) : (x > y ? x : y);
        }
        o |= r16 << s;
      }
      return o;
    };
    struct SI {
      const char* name;
      void (*k)(const uint32_t*, const uint32_t*, uint32_t*, int);
      int op;
      bool sgn, relu;
    };
    for (const SI& s : {SI{"misc.e2e.arith.add.u16x2", k_add_u16x2, 0, false, false},
                        SI{"misc.e2e.arith.add.s16x2", k_add_s16x2, 0, true, false},
                        SI{"misc.e2e.arith.min.u16x2", k_min_u16x2, 1, false, false},
                        SI{"misc.e2e.arith.max.u16x2", k_max_u16x2, 2, false, false},
                        SI{"misc.e2e.arith.min.s16x2", k_min_s16x2, 1, true, false},
                        SI{"misc.e2e.arith.max.s16x2", k_max_s16x2, 2, true, false},
                        SI{"misc.e2e.arith.min.relu.s16x2", k_min_relu_s16x2, 1, true, true},
                        SI{"misc.e2e.arith.max.relu.s16x2", k_max_relu_s16x2, 2, true, true}}) {
      verify(
          s.name, run(s.k, ia, ib),
          [&](size_t i) { return simd(ia[i], ib[i], s.op, s.sgn, s.relu); }, eq, in);
    }
    for (int is_max = 0; is_max < 2; ++is_max) {
      verify(
          is_max ? "misc.e2e.arith.max.relu.s32" : "misc.e2e.arith.min.relu.s32",
          run(is_max ? k_max_relu_s32 : k_min_relu_s32, ia, ib),
          [&](size_t i) {
            const int x = (int)ia[i], y = (int)ib[i];
            const int v = is_max ? (x > y ? x : y) : (x < y ? x : y);
            return (uint32_t)(v < 0 ? 0 : v);
          },
          eq, in);
    }
  }
  return finish();
}
