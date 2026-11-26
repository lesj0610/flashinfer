// Host reference model, data generators and shared-memory image builder for the wgmma tests.
// Owner: W. Host code only; used by tests/wgmma/test_direct.cu, the generated
// tests/wgmma/e2e/*.cu programs (through e2e_common.h) and tests/wgmma/cute_gmma.cu.
//
// Independence: the register-fragment and sparse-metadata layouts below are transcribed from
// PLAN section 7 and do not call spec.h, so a wrong spec.h (or a wrong use of it in the device
// code) shows up as a GPU mismatch; tests/wgmma/test_direct.cu also compares these literals with
// spec.h directly (wgmma.ref.literal_vs_spec). Shared-memory images are laid out with
// spec::smem_addr as PLAN requires; base_off != 0 images are made by shifting a base_off = 0
// image by 128 * base_off bytes, which does not use spec's base_off arithmetic.
//
// Numerics:
//   exact: one numerics::dot_float / dot_int / dot_b1_and_popc per D element with
//          wgmma_params(atype, btype, d_f16). Dense: the K products in k order. Sparse: the K/2
//          stored A values (compressed order) times the B values their metadata selects.
//   fast:  the decomposition documented in include/wgmma_emu/wgmma_device.cuh (fast mode),
//          evaluated with mma_params(Arch::Ampere, ...): fp8 -> f16, sparse -> dense with +0 at
//          pruned positions, sub-mma chain in s order, mma column order inside a sub-mma (fp8
//          uses the documented column -> k mapping). Integer and b1 results equal exact.
#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "wgmma_emu/numerics.h"
#include "wgmma_emu/spec.h"
#include "wgmma_emu/types.h"

namespace wgref {

using wgemu::EType;

[[noreturn]] inline void die(const char* what) {
  fprintf(stderr, "WGEMU-REF-FATAL %s\n", what);
  fflush(stdout);
  abort();
}

// ---------------------------------------------------------------------------------------------
// Random numbers (splitmix64)
// ---------------------------------------------------------------------------------------------
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  uint32_t u32() { return uint32_t(next() >> 32); }
  int below(int n) { return int(next() % uint64_t(n)); }
};

// ---------------------------------------------------------------------------------------------
// Shapes (PLAN section 7, written independently of spec.h)
// ---------------------------------------------------------------------------------------------
inline bool is_int_type(EType t) { return t == EType::S8 || t == EType::U8 || t == EType::B1; }
inline bool is_fp8(EType t) { return t == EType::E4M3 || t == EType::E5M2; }
inline bool is_float_in(EType t) {
  return t == EType::F16 || t == EType::BF16 || t == EType::TF32 || is_fp8(t);
}
inline int type_bits(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 16;
    case EType::TF32:
    case EType::F32:
    case EType::S32:
      return 32;
    case EType::B1:
      return 1;
    default:
      return 8;
  }
}
// Logical K of one wgmma.
inline int kdim(EType at, bool sparse) {
  switch (type_bits(at)) {
    case 16:
      return sparse ? 32 : 16;
    case 32:
      return sparse ? 16 : 8;
    case 1:
      return 256;
    default:
      return sparse ? 64 : 32;
  }
}
// Valid N (PLAN section 7; integer superset with n240/n256).
inline bool n_valid(EType at, int n) {
  if (n < 8 || n > 256 || n % 8) return false;
  if (is_int_type(at)) return n <= 32 || n % 16 == 0;
  return true;
}
inline std::vector<int> all_n(EType at) {
  std::vector<int> r;
  for (int n = 8; n <= 256; n += 8)
    if (n_valid(at, n)) r.push_back(n);
  return r;
}
inline const char* type_name(EType t) {
  switch (t) {
    case EType::F16:
      return "f16";
    case EType::BF16:
      return "bf16";
    case EType::TF32:
      return "tf32";
    case EType::E4M3:
      return "e4m3";
    case EType::E5M2:
      return "e5m2";
    case EType::S8:
      return "s8";
    case EType::U8:
      return "u8";
    case EType::B1:
      return "b1";
    case EType::F32:
      return "f32";
    case EType::S32:
      return "s32";
  }
  return "?";
}

// One wgmma, logical description (operand sources and layouts are not part of it).
struct Op {
  int n = 8;
  EType at = EType::F16, bt = EType::F16, dt = EType::F32;
  bool sparse = false, neg_a = false, neg_b = false, satfinite = false;
  int sp_sel = 0;
  bool scale_d = true;
};
inline int op_k(const Op& o) { return kdim(o.at, o.sparse); }
// Stored A values per row: dense K, sparse K/2 (compressed), b1 8 words.
inline int op_ka(const Op& o) { return o.at == EType::B1 ? 8 : (o.sparse ? op_k(o) / 2 : op_k(o)); }
// Stored B values per column: K, b1 8 words.
inline int op_kb(const Op& o) { return o.at == EType::B1 ? 8 : op_k(o); }
inline int op_nregs(const Op& o) { return o.dt == EType::F16 ? o.n / 4 : o.n / 2; }
// Bits per stored value in smem (b1 as 32-bit words).
inline int store_bits(EType t) { return t == EType::B1 ? 32 : type_bits(t); }
// Metadata chunks per row: f16/bf16 32/4, tf32 16/2, 8-bit 64/4.
inline int meta_chunks(EType at) { return at == EType::TF32 ? 8 : (type_bits(at) == 8 ? 16 : 8); }

// ---------------------------------------------------------------------------------------------
// Register layouts (PLAN section 7 literals): t = tid % 128, w = t/32, l = lane, g = l/4,
// q = l%4, r = 16w + g.
// ---------------------------------------------------------------------------------------------
inline void d_coord(int t, int i, int* m, int* n) {
  const int w = (t & 127) >> 5, l = t & 31, g = l >> 2, q = l & 3, r = 16 * w + g;
  *m = r + 8 * ((i % 4) / 2);
  *n = 8 * (i / 4) + 2 * q + (i % 2);
}
// Elements per A register: 16-bit 2, tf32 1, 8-bit 4, b1 32 (bits).
inline int a_per_reg(EType at) { return 32 / type_bits(at); }
// RS A register j, element e: dense -> logical k (b1: bit index), sparse -> compressed column.
inline void a_coord(EType at, bool sparse, int t, int j, int e, int* m, int* k) {
  const int w = (t & 127) >> 5, l = t & 31, g = l >> 2, q = l & 3, r = 16 * w + g;
  const int K = sparse ? kdim(at, true) / 2 : kdim(at, false);
  *m = r + 8 * (j & 1);
  *k = (j >> 1) * (K / 2) + q * (K / 8) + e;
}
// Metadata nibble of row m (0..63), chunk c: lane of warp m/16 and bit offset.
inline void meta_loc(EType at, int sel, int m, int c, int* lane, int* bit) {
  const int rr = m % 16, g = rr % 8, hi = rr / 8;
  if (type_bits(at) == 8) {
    *lane = 4 * g + hi + 2 * (c >= 8 ? 1 : 0);
    *bit = 4 * (c % 8);
  } else {
    *lane = 4 * g + 2 * sel + (c >= 4 ? 1 : 0);
    *bit = 16 * hi + 4 * (c % 4);
  }
}

// ---------------------------------------------------------------------------------------------
// fp8 -> f16 (independent: through double)
// ---------------------------------------------------------------------------------------------
inline double fp8_value(uint32_t x, EType t, bool* nan, bool* inf) {
  x &= 0xFFu;
  const int s = (x >> 7) & 1;
  int e, m, ebits = t == EType::E4M3 ? 4 : 5, mbits = t == EType::E4M3 ? 3 : 2;
  const int bias = t == EType::E4M3 ? 7 : 15;
  e = int((x >> mbits) & ((1u << ebits) - 1));
  m = int(x & ((1u << mbits) - 1));
  *nan = *inf = false;
  if (t == EType::E4M3 && e == 15 && m == 7) *nan = true;
  if (t == EType::E5M2 && e == 31) {
    if (m)
      *nan = true;
    else
      *inf = true;
  }
  double v = (e == 0) ? ldexp(double(m), 1 - bias - mbits)
                      : ldexp(double(m + (1 << mbits)), e - bias - mbits);
  return s ? -v : v;
}
// Exact f16 bits of a value representable in f16 (finite).
inline uint32_t f16_of(double v) {
  uint32_t s = 0;
  if (signbit(v)) {
    s = 0x8000u;
    v = -v;
  }
  if (v == 0.0) return s;
  int e;
  const double f = frexp(v, &e);  // v = f * 2^e, f in [0.5, 1)
  const int E = e - 1 + 15;
  if (E >= 1) {
    const double mant = (f * 2.0 - 1.0) * 1024.0;
    if (mant != floor(mant) || E > 30) die("f16_of: not representable");
    return s | (uint32_t(E) << 10) | uint32_t(mant);
  }
  const double mant = ldexp(v, 24);
  if (mant != floor(mant) || mant >= 1024.0) die("f16_of: not representable (subnormal)");
  return s | uint32_t(mant);
}
// NaN -> sign | 0x7E00 (e4m3) or the e5m2 payload shifted into f16 (e5m2 is the top byte of f16).
inline uint32_t fp8_to_f16_ref(uint32_t x, EType t) {
  bool nan, inf;
  const double v = fp8_value(x, t, &nan, &inf);
  const uint32_t s = (x & 0x80u) << 8;
  if (nan) return t == EType::E4M3 ? (s | 0x7E00u) : ((x & 0xFFu) << 8);
  if (inf) return s | 0x7C00u;
  return f16_of(v) | ((v == 0.0) ? s : 0u);
}

// ---------------------------------------------------------------------------------------------
// Value generators (raw bits as numerics.h expects)
// ---------------------------------------------------------------------------------------------
inline uint32_t gen_input(EType t, Rng& r) {
  const int roll = r.below(1000);
  switch (t) {
    case EType::F16: {
      static const uint16_t fin[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF,
                                     0x83FF, 0x0400, 0x8400, 0x7BFF, 0xFBFF};
      static const uint16_t nonfin[] = {0x7C00, 0xFC00, 0x7E00, 0x7D55, 0xFE01};
      if (roll < 3) return nonfin[r.below(5)];
      if (roll < 40) return fin[r.below(10)];
      return (r.u32() & 0x8000u) | (uint32_t(9 + r.below(10)) << 10) | (r.u32() & 0x3FFu);
    }
    case EType::BF16: {
      static const uint16_t fin[] = {0x0000, 0x8000, 0x0001, 0x8001,
                                     0x007F, 0x0080, 0x7F7F, 0xFF7F};
      static const uint16_t nonfin[] = {0x7F80, 0xFF80, 0x7FC0, 0x7F81};
      if (roll < 3) return nonfin[r.below(4)];
      if (roll < 40) return fin[r.below(8)];
      return (r.u32() & 0x8000u) | (uint32_t(121 + r.below(10)) << 7) | (r.u32() & 0x7Fu);
    }
    case EType::TF32: {
      // Low 13 bits are random everywhere: the tf32 operand ignores them.
      static const uint32_t fin[] = {0x00000000u, 0x80000000u, 0x00002000u, 0x807FE000u,
                                     0x00800000u, 0x7F7FE000u, 0xFF7FE000u};
      static const uint32_t nonfin[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F802000u,
                                        0x7F800001u};
      const uint32_t low = r.u32() & 0x1FFFu;
      if (roll < 3) return nonfin[r.below(5)] | (roll == 0 ? low : 0u);
      if (roll < 40) return fin[r.below(7)] | low;
      return (r.u32() & 0x80000000u) | (uint32_t(121 + r.below(10)) << 23) | (r.u32() & 0x7FFFFFu);
    }
    case EType::E4M3: {
      if (roll < 3) return r.below(2) ? 0x7Fu : 0xFFu;
      uint32_t x = r.u32() & 0xFFu;
      if ((x & 0x7Fu) == 0x7Fu) x ^= 1u;
      return x;
    }
    case EType::E5M2: {
      static const uint8_t nonfin[] = {0x7C, 0xFC, 0x7E, 0xFD};
      if (roll < 3) return nonfin[r.below(4)];
      uint32_t x = r.u32() & 0xFFu;
      if ((x & 0x7Cu) == 0x7Cu) x &= ~0x40u;
      return x;
    }
    case EType::S8:
    case EType::U8:
      return r.u32() & 0xFFu;
    case EType::B1:
      return r.u32();
    default:
      die("gen_input: type");
  }
}
inline uint32_t gen_acc(EType dt, Rng& r) {
  const int roll = r.below(1000);
  switch (dt) {
    case EType::F16:
      return gen_input(EType::F16, r);
    case EType::F32: {
      static const uint32_t fin[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu,
                                     0x7F7FFFFFu, 0xFF7FFFFFu, 0x00800000u};
      static const uint32_t nonfin[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7FA00001u};
      if (roll < 3) return nonfin[r.below(4)];
      if (roll < 40) return fin[r.below(7)];
      return (r.u32() & 0x80000000u) | (uint32_t(119 + r.below(16)) << 23) | (r.u32() & 0x7FFFFFu);
    }
    case EType::S32:
      if (roll < 30) return 0x7FFFFFFFu - uint32_t(r.below(1 << 17));
      if (roll < 60) return 0x80000000u + uint32_t(r.below(1 << 17));
      return uint32_t(int32_t(r.below(1 << 21)) - (1 << 20));
    default:
      die("gen_acc: type");
  }
}

// ---------------------------------------------------------------------------------------------
// Problem: logical matrices of one wgmma
// ---------------------------------------------------------------------------------------------
struct Problem {
  Op op;
  std::vector<uint32_t> A;    // 64 x ka (dense: logical k; sparse: compressed; b1: words)
  std::vector<uint8_t> F;     // sparse: 64 x meta_chunks metadata nibbles
  std::vector<uint8_t> Aidx;  // sparse: 64 x ka logical k of each stored A value
  std::vector<uint32_t> B;    // n x kb (b1: words)
  std::vector<uint32_t> C;    // 64 x n (f16 D: low 16 bits)
};

// Valid 2:4 nibbles: all ordered pairs (i0, i1), i0 != i1 (PLAN section 7: 0b0000, 0b0101,
// 0b1010, 0b1111 are undefined). tf32: 0b0100 (element 0), 0b1110 (element 1).
inline uint32_t meta_nibble(EType at, int idx) {
  if (at == EType::TF32) return (idx & 1) ? 0xEu : 0x4u;
  static const uint8_t pairs[12][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
                                       {1, 0}, {2, 0}, {3, 0}, {2, 1}, {3, 1}, {3, 2}};
  const int p = idx % 12;
  return uint32_t(pairs[p][0] | (pairs[p][1] << 2));
}
inline void fill_aidx(Problem& P) {
  const Op& o = P.op;
  const int ka = op_ka(o), nch = meta_chunks(o.at);
  P.Aidx.assign(size_t(64) * ka, 0);
  for (int m = 0; m < 64; ++m) {
    for (int c = 0; c < nch; ++c) {
      const uint32_t f = P.F[size_t(m) * nch + c];
      if (o.at == EType::TF32) {
        P.Aidx[size_t(m) * ka + c] = uint8_t(2 * c + int((f & 3u) >> 1));
      } else {
        P.Aidx[size_t(m) * ka + 2 * c] = uint8_t(4 * c + int(f & 3u));
        P.Aidx[size_t(m) * ka + 2 * c + 1] = uint8_t(4 * c + int((f >> 2) & 3u));
      }
    }
  }
}

inline Problem make_problem(const Op& op, uint64_t seed) {
  Problem P;
  P.op = op;
  Rng r(seed);
  const int ka = op_ka(op), kb = op_kb(op), n = op.n;
  P.A.resize(size_t(64) * ka);
  P.B.resize(size_t(n) * kb);
  P.C.resize(size_t(64) * n);
  for (auto& x : P.A) x = gen_input(op.at, r);
  for (auto& x : P.B) x = gen_input(op.bt, r);
  for (auto& x : P.C) x = gen_acc(op.dt, r);
  if (is_fp8(op.at)) {
    // Every fp8 code appears: A rows 0..7 and B columns 0..256/kb-1 hold permutations of 0..255.
    std::vector<uint32_t> perm(256);
    for (int i = 0; i < 256; ++i) perm[i] = uint32_t(i);
    for (int i = 255; i > 0; --i) std::swap(perm[i], perm[r.below(i + 1)]);
    for (int i = 0; i < 256; ++i) P.A[size_t(i / ka) * ka + i % ka] = perm[i];
    for (int i = 255; i > 0; --i) std::swap(perm[i], perm[r.below(i + 1)]);
    for (int i = 0; i < 256 && i / kb < n; ++i) P.B[size_t(i / kb) * kb + i % kb] = perm[i];
  }
  if (is_float_in(op.at) && !op.sparse) {
    // Exact cancellation: rows m % 5 == 2 pair (a, -a), columns n % 3 == 1 pair (b, b).
    const uint32_t sa = op.at == EType::TF32 ? 0x80000000u : (is_fp8(op.at) ? 0x80u : 0x8000u);
    for (int m = 8; m < 64; ++m)
      if (m % 5 == 2)
        for (int k = 0; k + 1 < ka; k += 2)
          P.A[size_t(m) * ka + k + 1] = P.A[size_t(m) * ka + k] ^ sa;
    for (int j = 8; j < n; ++j)
      if (j % 3 == 1)
        for (int k = 0; k + 1 < kb; k += 2) P.B[size_t(j) * kb + k + 1] = P.B[size_t(j) * kb + k];
  }
  if (op.at == EType::S8 || op.at == EType::U8) {
    // Range ends: rows of 0x80 / 0x7F / 0xFF, columns of 0x80 / 0xFF.
    for (int m = 0; m < 64; ++m) {
      const uint32_t v =
          m % 13 == 5 ? 0x80u : (m % 13 == 6 ? 0x7Fu : (m % 13 == 7 ? 0xFFu : 0x100u));
      if (v != 0x100u)
        for (int k = 0; k < ka; ++k) P.A[size_t(m) * ka + k] = v;
    }
    for (int j = 0; j < n; ++j) {
      const uint32_t v = j % 7 == 3 ? 0x80u : (j % 7 == 4 ? 0xFFu : 0x100u);
      if (v != 0x100u)
        for (int k = 0; k < kb; ++k) P.B[size_t(j) * kb + k] = v;
    }
  }
  if (op.sparse) {
    const int nch = meta_chunks(op.at);
    P.F.resize(size_t(64) * nch);
    const int off = int(seed % 12);
    for (int m = 0; m < 64; ++m)
      for (int c = 0; c < nch; ++c)
        P.F[size_t(m) * nch + c] = uint8_t(meta_nibble(op.at, m * nch + c + off));
    fill_aidx(P);
  }
  return P;
}

// ---------------------------------------------------------------------------------------------
// Reference results (64 x n, f16 D in the low 16 bits)
// ---------------------------------------------------------------------------------------------
inline uint32_t elem_sign(EType t) {
  switch (t) {
    case EType::F16:
    case EType::BF16:
      return 0x8000u;
    case EType::TF32:
      return 0x80000000u;
    case EType::E4M3:
    case EType::E5M2:
      return 0x80u;
    default:
      return 0u;
  }
}

inline std::vector<uint32_t> ref_exact(const Problem& P, const std::vector<uint32_t>& C) {
  const Op& o = P.op;
  const int ka = op_ka(o), kb = op_kb(o), n = o.n;
  const bool d16 = o.dt == EType::F16;
  const uint32_t sa = o.neg_a ? elem_sign(o.at) : 0u, sb = o.neg_b ? elem_sign(o.bt) : 0u;
  wgemu::FdpaParams p = {};
  if (is_float_in(o.at)) p = wgemu::wgmma_params(o.at, o.bt, d16);
  std::vector<uint32_t> D(size_t(64) * n);
  std::vector<uint32_t> a(ka), b(ka);
  for (int m = 0; m < 64; ++m) {
    for (int k = 0; k < ka; ++k) a[k] = P.A[size_t(m) * ka + k] ^ sa;
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < ka; ++k) {
        const int kk = o.sparse ? P.Aidx[size_t(m) * ka + k] : k;
        b[k] = P.B[size_t(j) * kb + kk] ^ sb;
      }
      const uint32_t c = o.scale_d ? C[size_t(m) * n + j] : 0u;
      uint32_t r;
      if (o.at == EType::B1) {
        r = uint32_t(wgemu::dot_b1_and_popc(a.data(), b.data(), ka, int32_t(c)));
      } else if (o.at == EType::S8 || o.at == EType::U8) {
        r = uint32_t(wgemu::dot_int(a.data(), b.data(), ka, o.at == EType::S8, o.bt == EType::S8,
                                    int32_t(c), o.satfinite));
      } else {
        r = wgemu::dot_float(a.data(), b.data(), ka, o.at, o.bt, c, d16, p);
      }
      D[size_t(m) * n + j] = r;
    }
  }
  return D;
}
inline std::vector<uint32_t> ref_exact(const Problem& P) { return ref_exact(P, P.C); }

// Column c of fast-mode sub-mma s -> logical k (fp8 mapping from wgmma_device.cuh).
inline int fast_kmap(EType at, int s, int c) {
  if (is_fp8(at))
    return c < 8 ? 16 * s + 4 * (c / 2) + (c % 2) : 16 * s + 4 * ((c - 8) / 2) + 2 + (c % 2);
  return (at == EType::TF32 ? 8 : 16) * s + c;
}

inline std::vector<uint32_t> ref_fast(const Problem& P, const std::vector<uint32_t>& C) {
  const Op& o = P.op;
  if (!is_float_in(o.at)) return ref_exact(P, C);
  const int K = op_k(o), ka = op_ka(o), kb = op_kb(o), n = o.n;
  const bool d16 = o.dt == EType::F16;
  const EType mt = is_fp8(o.at) ? EType::F16 : o.at;
  const int kmma = o.at == EType::TF32 ? 8 : 16, S = K / kmma;
  const uint32_t sa = o.neg_a ? elem_sign(o.at) : 0u, sb = o.neg_b ? elem_sign(o.bt) : 0u;
  const wgemu::FdpaParams p = wgemu::mma_params(wgemu::Arch::Ampere, mt, mt, d16);
  auto conv = [&](uint32_t x, EType t) -> uint32_t {
    if (is_fp8(t)) return fp8_to_f16_ref(x, t);
    if (t == EType::TF32) return x & 0xFFFFE000u;
    return x;
  };
  std::vector<uint32_t> dA(size_t(64) * K, 0u), dB(size_t(n) * K);
  for (int m = 0; m < 64; ++m)
    for (int k = 0; k < ka; ++k) {
      const int kk = o.sparse ? P.Aidx[size_t(m) * ka + k] : k;
      dA[size_t(m) * K + kk] = conv(P.A[size_t(m) * ka + k] ^ sa, o.at);
    }
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < kb; ++k) dB[size_t(j) * K + k] = conv(P.B[size_t(j) * kb + k] ^ sb, o.bt);
  std::vector<uint32_t> D(size_t(64) * n);
  uint32_t a[16], b[16];
  for (int m = 0; m < 64; ++m)
    for (int j = 0; j < n; ++j) {
      uint32_t c = o.scale_d ? C[size_t(m) * n + j] : 0u;
      for (int s = 0; s < S; ++s) {
        for (int cc = 0; cc < kmma; ++cc) {
          const int k = fast_kmap(o.at, s, cc);
          a[cc] = dA[size_t(m) * K + k];
          b[cc] = dB[size_t(j) * K + k];
        }
        c = wgemu::dot_float(a, b, kmma, mt, mt, c, d16, p);
      }
      D[size_t(m) * n + j] = c;
    }
  return D;
}
inline std::vector<uint32_t> ref_fast(const Problem& P) { return ref_fast(P, P.C); }

inline std::vector<uint32_t> ref_mode(const Problem& P, const std::vector<uint32_t>& C, bool fast) {
  return fast ? ref_fast(P, C) : ref_exact(P, C);
}

// ---------------------------------------------------------------------------------------------
// Register packing (PLAN section 7 literals)
// ---------------------------------------------------------------------------------------------
inline void pack_c(const Op& o, const std::vector<uint32_t>& C, int t, uint32_t* regs) {
  const int nr = op_nregs(o);
  for (int i = 0; i < nr; ++i) regs[i] = 0u;
  for (int i = 0; i < o.n / 2; ++i) {
    int m, j;
    d_coord(t, i, &m, &j);
    const uint32_t v = C[size_t(m) * o.n + j];
    if (o.dt == EType::F16)
      regs[i >> 1] |= (v & 0xFFFFu) << (16 * (i & 1));
    else
      regs[i] = v;
  }
}
inline void unpack_d(const Op& o, int t, const uint32_t* regs, std::vector<uint32_t>& D) {
  for (int i = 0; i < o.n / 2; ++i) {
    int m, j;
    d_coord(t, i, &m, &j);
    D[size_t(m) * o.n + j] =
        o.dt == EType::F16 ? ((regs[i >> 1] >> (16 * (i & 1))) & 0xFFFFu) : regs[i];
  }
}
inline void pack_a(const Problem& P, int t, uint32_t a[4]) {
  const Op& o = P.op;
  const int ka = op_ka(o);
  const int per = a_per_reg(o.at), bits = type_bits(o.at);
  for (int j = 0; j < 4; ++j) {
    a[j] = 0u;
    int m, k0;
    a_coord(o.at, o.sparse, t, j, 0, &m, &k0);
    if (o.at == EType::B1) {
      if (k0 % 32) die("pack_a: b1 register not word aligned");
      a[j] = P.A[size_t(m) * ka + k0 / 32];
      continue;
    }
    for (int e = 0; e < per; ++e) {
      int mm, k;
      a_coord(o.at, o.sparse, t, j, e, &mm, &k);
      const uint32_t mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u);
      a[j] |= (P.A[size_t(mm) * ka + k] & mask) << (e * bits);
    }
  }
}
// sp-meta register of thread t; positions no row of this thread's warp uses keep `fill`.
inline uint32_t pack_meta(const Problem& P, int t, uint32_t fill) {
  const Op& o = P.op;
  const int nch = meta_chunks(o.at), w = (t & 127) >> 5, l = t & 31;
  uint32_t reg = fill;
  for (int m = 16 * w; m < 16 * w + 16; ++m)
    for (int c = 0; c < nch; ++c) {
      int lane, bit;
      meta_loc(o.at, o.sp_sel, m, c, &lane, &bit);
      if (lane != l) continue;
      reg = (reg & ~(0xFu << bit)) | (uint32_t(P.F[size_t(m) * nch + c] & 0xFu) << bit);
    }
  return reg;
}

// ---------------------------------------------------------------------------------------------
// Shared-memory image builder (canonical GMMA layouts, spec::smem_addr)
// ---------------------------------------------------------------------------------------------
struct LayoutSel {
  bool mn_major = false;
  int sw = 0;     // swizzle bytes: 0, 32, 64, 128
  int order = 0;  // 0: the k-direction stride is the inner one, 1: the mn-direction stride
  int kofs = 0;   // K-major swizzled: start offset in units of the row's K bytes (< W / kbytes)
  int gap = 0;    // slack selector 0..3 (extra space between core matrices / atoms)
};

struct Image {
  std::vector<uint8_t> bytes;  // region-relative image (random filler + operands)
  std::vector<uint8_t> owner;  // 1 where an operand element lives
  uint32_t cap = 0;
  uint64_t injective_checked = 0;  // elements whose placement was verified
  uint64_t injective_bad = 0;      // overlaps, misalignment or out-of-range placements
};

inline void image_init(Image& im, uint32_t cap, Rng& r) {
  im.cap = cap;
  im.bytes.resize(cap);
  for (auto& b : im.bytes) b = uint8_t(r.u32());
  im.owner.assign(cap, 0);
}

// Lays out rows x cols stored values (bits 8/16/32; mn = row, k = column) at region offset
// `base` (1024-aligned). Returns false and sets *why when the layout cannot be built (not a
// canonical layout or out of space); the caller reports that, it never drops a case silently.
inline bool place_operand(Image& im, uint32_t base, const LayoutSel& L, int bits, int rows,
                          int cols, const uint32_t* vals, wgemu::spec::Desc* out, uint32_t* end,
                          std::string* why) {
  const uint32_t eb = uint32_t(bits / 8), kbytes = uint32_t(cols) * eb;
  const uint32_t W = uint32_t(L.sw);
  const uint32_t g1 = uint32_t(L.gap & 1), g2 = uint32_t((L.gap >> 1) & 1);
  wgemu::spec::Desc d;
  d.base_off = 0;
  d.swizzle_bytes = L.sw;
  d.start = base;
  if (base % 1024u) {
    *why = "base not 1024-aligned";
    return false;
  }
  if (L.mn_major && bits != 16) {
    *why = "MN-major exists only for 16-bit types";
    return false;
  }
  if (!L.mn_major) {
    const uint32_t nr1 = uint32_t(rows) / 8u;
    if (W == 0) {
      const uint32_t ncolk = kbytes / 16u;
      if (L.order == 0) {
        d.lbo = 128u + 16u * (g1 + 2u * g2);
        d.sbo = d.lbo * ncolk + 16u * (1u + g1);
      } else {
        d.sbo = 128u + 16u * (g1 + 2u * g2);
        d.lbo = d.sbo * nr1 + 16u * (1u + g1);
      }
    } else {
      if (kbytes > W) {
        *why = "K-major swizzled row wider than the swizzle span (spec: [inferred], not tested)";
        return false;
      }
      const uint32_t nk = W / kbytes;
      d.start = base + (uint32_t(L.kofs) % nk) * kbytes;
      d.sbo = 8u * W * (1u + g1 + g2);
      d.lbo = 16u * (1u + 5u * g1 + 3u * g2);  // unused by K-major swizzled layouts
    }
  } else {
    const uint32_t mbytes = uint32_t(rows) * eb, nk1 = uint32_t(cols) / 8u;
    if (cols % 8) {
      *why = "MN-major K not a multiple of 8";
      return false;
    }
    if (W == 0) {
      const uint32_t nchunk = (mbytes + 15u) / 16u;
      if (L.order == 0) {
        d.lbo = 128u + 16u * (g1 + 2u * g2);
        d.sbo = d.lbo * nk1 + 16u * (1u + g1);
      } else {
        d.sbo = 128u + 16u * (g1 + 2u * g2);
        d.lbo = d.sbo * nchunk + 16u * (1u + g1);
      }
    } else {
      const uint32_t nblk = (mbytes + W - 1u) / W;
      if (L.order == 0) {
        d.sbo = 8u * W * (1u + g1);
        d.lbo = d.sbo * nk1 + 8u * W * g2;
      } else {
        d.lbo = 8u * W * (1u + g1);
        d.sbo = d.lbo * nblk + 8u * W * g2;
      }
    }
  }
  if (d.lbo >= (1u << 18) || d.sbo >= (1u << 18) || d.start >= (1u << 18)) {
    *why = "descriptor field overflow";
    return false;
  }
  uint32_t mx = 0;
  std::vector<uint32_t> addr(size_t(rows) * cols);
  for (int mn = 0; mn < rows; ++mn)
    for (int k = 0; k < cols; ++k) {
      const uint32_t a = wgemu::spec::smem_addr(d, L.mn_major, bits, mn, k);
      addr[size_t(mn) * cols + k] = a;
      mx = std::max(mx, a + eb);
    }
  if (mx > im.cap) {
    *why = "operand does not fit the region";
    return false;
  }
  for (int mn = 0; mn < rows; ++mn)
    for (int k = 0; k < cols; ++k) {
      const uint32_t a = addr[size_t(mn) * cols + k];
      ++im.injective_checked;
      bool bad = (a % eb) != 0;
      for (uint32_t i = 0; i < eb && !bad; ++i) bad = im.owner[a + i] != 0;
      if (bad) {
        ++im.injective_bad;
        continue;
      }
      const uint32_t v = vals[size_t(mn) * cols + k];
      for (uint32_t i = 0; i < eb; ++i) {
        im.owner[a + i] = 1;
        im.bytes[a + i] = uint8_t(v >> (8 * i));
      }
    }
  *out = d;
  *end = mx;
  return true;
}

inline uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1u) / a * a; }

// A (64 x ka) and B (n x kb) of P into one region image. A is skipped for RS. base_off > 0
// shifts the finished image by 128 * base_off bytes and sets the descriptors' base_off field.
struct Placement {
  uint64_t adesc = 0, bdesc = 0;  // region-relative start
  uint32_t bytes = 0;             // image bytes to copy (multiple of 16)
  uint32_t b_first_addr = 0;      // region-relative address of B element (0, 0)
};
inline bool place_problem(const Problem& P, bool rs, const LayoutSel& la, const LayoutSel& lb,
                          int base_off, uint32_t cap, uint64_t seed, std::vector<uint8_t>& out,
                          Placement* pl, Image* stats, std::string* why) {
  Rng r(seed ^ 0x5EEDull);
  Image im;
  const uint32_t shift = 128u * uint32_t(base_off);
  if (shift + 2048u > cap) {
    *why = "region too small";
    return false;
  }
  image_init(im, cap - shift, r);
  const Op& o = P.op;
  uint32_t a_end = 0;
  wgemu::spec::Desc da = {}, db = {};
  if (!rs) {
    if (!place_operand(im, 0u, la, store_bits(o.at), 64, op_ka(o), P.A.data(), &da, &a_end, why))
      return false;
  }
  const uint32_t bbase = align_up(a_end, 1024u) + (lb.gap & 2 ? 1024u : 0u);
  uint32_t b_end = 0;
  if (!place_operand(im, bbase, lb, store_bits(o.bt), o.n, op_kb(o), P.B.data(), &db, &b_end, why))
    return false;
  pl->b_first_addr = wgemu::spec::smem_addr(db, lb.mn_major, store_bits(o.bt), 0, 0) + shift;
  const uint32_t used = align_up(std::max(a_end, b_end), 16u);
  out.assign(shift + used, 0);
  for (uint32_t i = 0; i < shift; ++i) out[i] = uint8_t(r.u32());
  memcpy(out.data() + shift, im.bytes.data(), used);
  if (shift) {
    da.start += shift;
    db.start += shift;
    da.base_off = db.base_off = uint32_t(base_off);
  }
  pl->adesc = rs ? 0u : wgemu::spec::encode_desc(da);
  pl->bdesc = wgemu::spec::encode_desc(db);
  pl->bytes = uint32_t(out.size());
  if (stats) {
    stats->injective_checked += im.injective_checked;
    stats->injective_bad += im.injective_bad;
  }
  if (im.injective_bad) {
    *why = "operand placement not injective / misaligned";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Result bookkeeping (PLAN section 11 output)
// ---------------------------------------------------------------------------------------------
struct Stat {
  uint64_t cases = 0, mismatches = 0;
  int printed = 0;
};

// Compare 64 x n results; prints the first few mismatches of `name`.
inline void compare(const char* name, const std::string& what, const Op& o,
                    const std::vector<uint32_t>& got, const std::vector<uint32_t>& want, Stat& st) {
  for (int m = 0; m < 64; ++m)
    for (int j = 0; j < o.n; ++j) {
      ++st.cases;
      const uint32_t g = got[size_t(m) * o.n + j], w = want[size_t(m) * o.n + j];
      if (g == w) continue;
      ++st.mismatches;
      if (st.printed < 8) {
        ++st.printed;
        printf("WGEMU-MISMATCH name=%s %s m=%d n=%d got=0x%08x want=0x%08x\n", name, what.c_str(),
               m, j, g, w);
      }
    }
}

}  // namespace wgref
