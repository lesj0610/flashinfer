// End-to-end check of the sm_89 / sm_90 mma.sync forms rewritten by tools/ptxemu/rules_misc.py
// into lib:wgemu_mma_fp8 / wgemu_mma_sp_fp8 / wgemu_mma_f64 (fork-only). Owner: T2.
// Built for sm_90a by tools/nvcc_emu.
//
// Covered: mma m16n8k16 / m16n8k32 with e4m3 / e5m2 A and B, f32 and f16 C/D;
// mma.sp and mma.sp::ordered_metadata m16n8k64 (8-bit, f32); mma m16n8k4 / k8 / k16 .f64.
//
// Inputs are small values that every format holds exactly and whose dot products are exact in
// the accumulator, so the expectation (a plain double matmul) does not depend on the internal
// accumulation order; the check is about fragment layout (PTX section 9.7.14.5) and metadata.
// Fragment coordinates, with g = lane / 4 and q = lane % 4:
//   fp8 A reg i byte j: row g + 8 (i % 2), col 4 q + j + 16 (i / 2)
//   fp8 B reg i byte j: k 4 q + j + 16 i, n g
//   f64 A reg i: row g + 8 (i % 2), col 4 (i / 2) + q;  f64 B reg i: k q + 4 i, n g
//   C/D f32 / f64 element i: row g + 8 (i / 2), col 2 q + i % 2;  f16 C/D reg i: row g + 8 i,
//   cols 2 q (low half) and 2 q + 1 (high half)
//
// Sparse m16n8k64: A holds the 16 x 32 compressed matrix in the dense k32 layout; compressed
// column c belongs to group c / 2 of four logical columns, at the position given by 2-bit
// index (c % 2) of that group's metadata nibble (bits 1:0 first). [inference] The B and
// metadata register layouts are drawn only as figures, so the test uses one metadata nibble for
// every group and B values that depend only on (k % 4, n): the expectation then holds for any
// B layout in which byte j of a register holds a k with k % 4 == j for column n = lane / 4, and
// for any nibble-to-group mapping, while the six valid nibbles still give six different results.
#include <math.h>

#include "misc_e2e.h"

using namespace misc_e2e;

typedef unsigned short u16;
typedef unsigned long long u64;

constexpr int kWarps = 16;  // independent random cases per configuration
constexpr int kLanes = 32 * kWarps;

// ---- kernels: every lane reads 4 A, 4 B, 4 C words and writes 4 D words (unused ones 0) ------
#define LANE() (blockIdx.x * blockDim.x + threadIdx.x)

#define MMA_FP8_F32(NAME, INS, AN, BN)                                                         \
  __global__ void NAME(const uint32_t* A, const uint32_t* B, const uint32_t* C, uint32_t* D) { \
    const int l = LANE();                                                                      \
    const uint32_t *a = A + 4 * l, *b = B + 4 * l, *c = C + 4 * l;                             \
    float d0, d1, d2, d3;                                                                      \
    MMA_FP8_F32_ASM_##AN##_##BN(INS);                                                          \
    D[4 * l] = __float_as_uint(d0);                                                            \
    D[4 * l + 1] = __float_as_uint(d1);                                                        \
    D[4 * l + 2] = __float_as_uint(d2);                                                        \
    D[4 * l + 3] = __float_as_uint(d3);                                                        \
  }
#define F32C                                                                          \
  "f"(__uint_as_float(c[0])), "f"(__uint_as_float(c[1])), "f"(__uint_as_float(c[2])), \
      "f"(__uint_as_float(c[3]))
#define MMA_FP8_F32_ASM_2_1(INS)                                           \
  asm volatile(INS " {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};" \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                    \
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), F32C)
#define MMA_FP8_F32_ASM_4_2(INS)                                                          \
  asm volatile(INS " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                                   \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), F32C)

#define MMA_FP8_F16(NAME, INS, AN, BN)                                                         \
  __global__ void NAME(const uint32_t* A, const uint32_t* B, const uint32_t* C, uint32_t* D) { \
    const int l = LANE();                                                                      \
    const uint32_t *a = A + 4 * l, *b = B + 4 * l, *c = C + 4 * l;                             \
    uint32_t d0, d1;                                                                           \
    MMA_FP8_F16_ASM_##AN##_##BN(INS);                                                          \
    D[4 * l] = d0;                                                                             \
    D[4 * l + 1] = d1;                                                                         \
    D[4 * l + 2] = 0;                                                                          \
    D[4 * l + 3] = 0;                                                                          \
  }
#define MMA_FP8_F16_ASM_2_1(INS)                          \
  asm volatile(INS " {%0, %1}, {%2, %3}, {%4}, {%5, %6};" \
               : "=r"(d0), "=r"(d1)                       \
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]))
#define MMA_FP8_F16_ASM_4_2(INS)                                                              \
  asm volatile(INS " {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"                         \
               : "=r"(d0), "=r"(d1)                                                           \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), \
                 "r"(c[1]))

MMA_FP8_F32(k_k16_e4m3_e4m3_f32, "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32", 2, 1)
MMA_FP8_F16(k_k16_e5m2_e4m3_f16, "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16", 2, 1)
MMA_FP8_F32(k_k32_e4m3_e5m2_f32, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32", 4, 2)
MMA_FP8_F16(k_k32_e5m2_e5m2_f16, "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16", 4, 2)

#define MMA_SP(NAME, INS)                                                                       \
  __global__ void NAME(const uint32_t* A, const uint32_t* B, const uint32_t* C, uint32_t* D,    \
                       uint32_t meta) {                                                         \
    const int l = LANE();                                                                       \
    const uint32_t *a = A + 4 * l, *b = B + 4 * l, *c = C + 4 * l;                              \
    float d0, d1, d2, d3;                                                                       \
    asm volatile(INS                                                                            \
                 " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "                    \
                 "{%12, %13, %14, %15}, %16, 0x0;"                                              \
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                                       \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), \
                   "r"(b[3]), F32C, "r"(meta));                                                 \
    D[4 * l] = __float_as_uint(d0);                                                             \
    D[4 * l + 1] = __float_as_uint(d1);                                                         \
    D[4 * l + 2] = __float_as_uint(d2);                                                         \
    D[4 * l + 3] = __float_as_uint(d3);                                                         \
  }
MMA_SP(k_sp_ordered_e4m3_e4m3,
       "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32")
MMA_SP(k_sp_e5m2_e4m3, "mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32")

#define MMA_F64(NAME, INS, AN, BN)                                                     \
  __global__ void NAME(const double* A, const double* B, const double* C, double* D) { \
    const int l = LANE();                                                              \
    const double *a = A + 8 * l, *b = B + 4 * l, *c = C + 4 * l;                       \
    double d0, d1, d2, d3;                                                             \
    MMA_F64_ASM_##AN##_##BN(INS);                                                      \
    D[4 * l] = d0;                                                                     \
    D[4 * l + 1] = d1;                                                                 \
    D[4 * l + 2] = d2;                                                                 \
    D[4 * l + 3] = d3;                                                                 \
  }
#define F64C "d"(c[0]), "d"(c[1]), "d"(c[2]), "d"(c[3])
#define MMA_F64_ASM_2_1(INS)                                               \
  asm volatile(INS " {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};" \
               : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)                    \
               : "d"(a[0]), "d"(a[1]), "d"(b[0]), F64C)
#define MMA_F64_ASM_4_2(INS)                                                              \
  asm volatile(INS " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" \
               : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)                                   \
               : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(b[0]), "d"(b[1]), F64C)
#define MMA_F64_ASM_8_4(INS)                                                                   \
  asm volatile(INS                                                                             \
               " {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, " \
               "{%16, %17, %18, %19};"                                                         \
               : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)                                        \
               : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]), "d"(a[4]), "d"(a[5]), "d"(a[6]),  \
                 "d"(a[7]), "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]), F64C)

MMA_F64(k_f64_k4, "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64", 2, 1)
MMA_F64(k_f64_k8_rn, "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64.rn", 4, 2)
MMA_F64(k_f64_k16, "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64", 8, 4)

// ---- host side ---------------------------------------------------------------------------------

static double fp8_val(uint32_t v, bool e5) {
  v &= 0xFFu;
  const int eb = e5 ? 5 : 4, mb = e5 ? 2 : 3, bias = e5 ? 15 : 7;
  const int e = (int)(v >> mb) & ((1 << eb) - 1), m = (int)v & ((1 << mb) - 1);
  const double mag =
      e == 0 ? ldexp((double)m, 1 - bias - mb) : ldexp((double)(m | (1 << mb)), e - bias - mb);
  return (v & 0x80u) ? -mag : mag;  // only finite small values are used
}
static uint32_t fp8_code(double x, bool e5) {
  for (uint32_t v = 0; v < 256; ++v)
    if (fp8_val(v, e5) == x && !(x == 0 && v)) return v;
  fprintf(stderr, "value %g has no exact fp8 encoding\n", x);
  exit(3);
}
static double f16_val(uint32_t h) {
  h &= 0xFFFFu;
  const int e = (int)(h >> 10) & 0x1F, m = (int)h & 0x3FF;
  const double mag = e == 0 ? ldexp((double)m, -24) : ldexp((double)(m | 0x400), e - 25);
  return (h & 0x8000u) ? -mag : mag;
}
static uint32_t f16_code(double x) {
  for (uint32_t v = 0; v < 0x7C00u; ++v) {
    if (f16_val(v) == x) return v;
    if (f16_val(v | 0x8000u) == x && x != 0) return v | 0x8000u;
  }
  fprintf(stderr, "value %g has no exact f16 encoding\n", x);
  exit(3);
}

static const double kFp8Vals[] = {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3};

struct Mats {
  int K;
  std::vector<double> A, B, C;  // A: 16 x K row-major, B: K x 8 (k-major), C: 16 x 8
};

static Mats random_mats(Rng& r, int K, const double* vals, int nvals, int cmax) {
  Mats m{K, std::vector<double>(16 * K), std::vector<double>(K * 8), std::vector<double>(16 * 8)};
  for (double& x : m.A) x = vals[r.below(nvals)];
  for (double& x : m.B) x = vals[r.below(nvals)];
  for (double& x : m.C) x = (double)((int)r.below(2 * cmax + 1) - cmax);
  return m;
}

static double dot_ref(const Mats& m, int row, int col) {
  double s = m.C[row * 8 + col];
  for (int k = 0; k < m.K; ++k) s += m.A[row * m.K + k] * m.B[k * 8 + col];
  return s;
}

// Packs one warp's fp8 A (16 x K) and B (K x 8) and its C into lane registers.
static void pack_fp8(const Mats& m, bool ae5, bool be5, bool cf16, int warp,
                     std::vector<uint32_t>& A, std::vector<uint32_t>& B, std::vector<uint32_t>& C) {
  const int na = m.K / 8, nb = m.K / 16;  // k16: 2 / 1, k32: 4 / 2
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane >> 2, q = lane & 3, l = 32 * warp + lane;
    for (int i = 0; i < na; ++i) {
      uint32_t w = 0;
      for (int j = 0; j < 4; ++j) {
        const int row = g + 8 * (i % 2), col = 4 * q + j + 16 * (i / 2);
        w |= fp8_code(m.A[row * m.K + col], ae5) << (8 * j);
      }
      A[4 * l + i] = w;
    }
    for (int i = 0; i < nb; ++i) {
      uint32_t w = 0;
      for (int j = 0; j < 4; ++j) w |= fp8_code(m.B[(4 * q + j + 16 * i) * 8 + g], be5) << (8 * j);
      B[4 * l + i] = w;
    }
    if (cf16) {
      for (int i = 0; i < 2; ++i) {
        const int row = g + 8 * i;
        C[4 * l + i] = f16_code(m.C[row * 8 + 2 * q]) | (f16_code(m.C[row * 8 + 2 * q + 1]) << 16);
      }
    } else {
      for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i / 2), col = 2 * q + i % 2;
        C[4 * l + i] = f2u((float)m.C[row * 8 + col]);
      }
    }
  }
}

// Compares one warp's D registers with the reference D.
static void check_d(Item& it, const std::vector<uint32_t>& D, int warp, bool f16,
                    const std::vector<double>& want) {
  for (int lane = 0; lane < 32; ++lane) {
    const int g = lane >> 2, q = lane & 3, l = 32 * warp + lane;
    for (int i = 0; i < 4; ++i) {
      // Same (row, col) for both: f16 reg i / 2 holds row g + 8 (i / 2), half i % 2 col 2q + i % 2.
      const int row = g + 8 * (i / 2), col = 2 * q + i % 2;
      const double got =
          f16 ? f16_val(D[4 * l + i / 2] >> (16 * (i % 2))) : (double)u2f(D[4 * l + i]);
      const double w = want[row * 8 + col];
      it.check(got == w, "warp=%d lane=%d row=%d col=%d got=%g want=%g", warp, lane, row, col, got,
               w);
    }
  }
}

typedef void (*KMma)(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*);
typedef void (*KSp)(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, uint32_t);
typedef void (*KF64)(const double*, const double*, const double*, double*);

static void run_dense(const char* name, KMma k, int K, bool ae5, bool be5, bool f16,
                      uint64_t seed) {
  Rng r(seed);
  std::vector<uint32_t> A(4 * kLanes), B(4 * kLanes), C(4 * kLanes);
  std::vector<Mats> ms;
  for (int w = 0; w < kWarps; ++w) {
    ms.push_back(random_mats(r, K, kFp8Vals, 10, 8));
    pack_fp8(ms.back(), ae5, be5, f16, w, A, B, C);
  }
  Dev<uint32_t> dA(A), dB(B), dC(C), dD(4 * kLanes);
  k<<<kWarps, 32>>>(dA.p, dB.p, dC.p, dD.p);
  MISC_SYNC();
  const std::vector<uint32_t> D = dD.get();
  Item it(name);
  for (int w = 0; w < kWarps; ++w) {
    std::vector<double> want(16 * 8);
    for (int row = 0; row < 16; ++row)
      for (int col = 0; col < 8; ++col) want[row * 8 + col] = dot_ref(ms[w], row, col);
    check_d(it, D, w, f16, want);
  }
}

// Valid 8-bit metadata nibbles (ptx.txt, sparse mma metadata) and the weights of k % 4.
static const uint32_t kMetaNibbles[] = {0x4, 0x8, 0x9, 0xC, 0xD, 0xE};
static const double kKWeight[4] = {1, 2, -1, 0.5};

static void run_sparse(const char* name, KSp k, bool ae5, bool be5, uint64_t seed) {
  Item it(name);
  Rng r(seed);
  for (uint32_t nib : kMetaNibbles) {
    const uint32_t meta = nib * 0x11111111u;
    const int idx[2] = {(int)(nib & 3u), (int)((nib >> 2) & 3u)};
    std::vector<uint32_t> A(4 * kLanes), B(4 * kLanes, 0), C(4 * kLanes);
    std::vector<Mats> ms;
    for (int w = 0; w < kWarps; ++w) {
      // Compressed A (16 x 32) uses the dense k32 A layout; B is 64 x 8 with
      // B[k][n] = kKWeight[k % 4] * beta[n].
      Mats m = random_mats(r, 32, kFp8Vals, 10, 8);
      double beta[8];
      for (double& x : beta) x = kFp8Vals[5 + r.below(4)];  // 0.5, 1, 1.5, 2
      m.B.assign(64 * 8, 0);
      for (int kk = 0; kk < 64; ++kk)
        for (int n = 0; n < 8; ++n) m.B[kk * 8 + n] = kKWeight[kk % 4] * beta[n];
      ms.push_back(m);
      std::vector<uint32_t> Bd(4 * kLanes);
      Mats ma = m;
      ma.B.assign(32 * 8, 0);  // pack A and C only through the dense k32 path
      pack_fp8(ma, ae5, be5, false, w, A, Bd, C);
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, q = lane & 3, l = 32 * w + lane;
        for (int i = 0; i < 4; ++i) {
          uint32_t word = 0;
          for (int j = 0; j < 4; ++j)
            word |= fp8_code(m.B[(4 * q + j + 16 * i) * 8 + g], be5) << (8 * j);
          B[4 * l + i] = word;
        }
      }
    }
    Dev<uint32_t> dA(A), dB(B), dC(C), dD(4 * kLanes);
    k<<<kWarps, 32>>>(dA.p, dB.p, dC.p, dD.p, meta);
    MISC_SYNC();
    const std::vector<uint32_t> D = dD.get();
    for (int w = 0; w < kWarps; ++w) {
      const Mats& m = ms[w];
      std::vector<double> want(16 * 8);
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 8; ++col) {
          double s = m.C[row * 8 + col];
          for (int c = 0; c < 32; ++c) {
            const int kk = 4 * (c / 2) + idx[c % 2];
            s += m.A[row * 32 + c] * m.B[kk * 8 + col];
          }
          want[row * 8 + col] = s;
        }
      }
      check_d(it, D, w, false, want);
    }
  }
}

static void run_f64(const char* name, KF64 k, int K, uint64_t seed) {
  static const double vals[] = {-4, -3, -2.5, -1, -0.5, 0, 0.25, 1, 2, 3.5};
  Rng r(seed);
  std::vector<double> A(8 * kLanes), B(4 * kLanes), C(4 * kLanes);
  std::vector<Mats> ms;
  for (int w = 0; w < kWarps; ++w) {
    ms.push_back(random_mats(r, K, vals, 10, 16));
    const Mats& m = ms.back();
    for (int lane = 0; lane < 32; ++lane) {
      const int g = lane >> 2, q = lane & 3, l = 32 * w + lane;
      for (int i = 0; i < K / 2; ++i) A[8 * l + i] = m.A[(g + 8 * (i % 2)) * K + 4 * (i / 2) + q];
      for (int i = 0; i < K / 4; ++i) B[4 * l + i] = m.B[(q + 4 * i) * 8 + g];
      for (int i = 0; i < 4; ++i) C[4 * l + i] = m.C[(g + 8 * (i / 2)) * 8 + 2 * q + i % 2];
    }
  }
  Dev<double> dA(A), dB(B), dC(C), dD(4 * kLanes);
  k<<<kWarps, 32>>>(dA.p, dB.p, dC.p, dD.p);
  MISC_SYNC();
  const std::vector<double> D = dD.get();
  Item it(name);
  for (int w = 0; w < kWarps; ++w) {
    for (int lane = 0; lane < 32; ++lane) {
      const int g = lane >> 2, q = lane & 3, l = 32 * w + lane;
      for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i / 2), col = 2 * q + i % 2;
        const double want = dot_ref(ms[w], row, col), got = D[4 * l + i];
        it.check(got == want, "warp=%d lane=%d row=%d col=%d got=%g want=%g", w, lane, row, col,
                 got, want);
      }
    }
  }
}

int main() {
  run_dense("misc.e2e.mma.m16n8k16.f32.e4m3.e4m3.f32", k_k16_e4m3_e4m3_f32, 16, false, false, false,
            0x8816);
  run_dense("misc.e2e.mma.m16n8k16.f16.e5m2.e4m3.f16", k_k16_e5m2_e4m3_f16, 16, true, false, true,
            0x8817);
  run_dense("misc.e2e.mma.m16n8k32.f32.e4m3.e5m2.f32", k_k32_e4m3_e5m2_f32, 32, false, true, false,
            0x8832);
  run_dense("misc.e2e.mma.m16n8k32.f16.e5m2.e5m2.f16", k_k32_e5m2_e5m2_f16, 32, true, true, true,
            0x8833);
  run_sparse("misc.e2e.mma.sp::ordered_metadata.m16n8k64.f32.e4m3.e4m3.f32", k_sp_ordered_e4m3_e4m3,
             false, false, 0x5964);
  run_sparse("misc.e2e.mma.sp.m16n8k64.f32.e5m2.e4m3.f32", k_sp_e5m2_e4m3, true, false, 0x5965);
  run_f64("misc.e2e.mma.m16n8k4.f64", k_f64_k4, 4, 0xF644);
  run_f64("misc.e2e.mma.m16n8k8.f64.rn", k_f64_k8_rn, 8, 0xF648);
  run_f64("misc.e2e.mma.m16n8k16.f64", k_f64_k16, 16, 0xF6416);
  return finish();
}
