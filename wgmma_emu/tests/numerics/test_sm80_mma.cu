// Acceptance criterion 2(b) (PLAN section 1): SM80 mma.sync against numerics.h (Arch::Ampere).
//
// Each warp runs one m16n8kK tile; every tile yields 128 dot products, D[m][n] =
// C[m][n] + sum_k A[m][k] * B[n][k], computed by the tensor core and by numerics.h on the host
// (dot_float with mma_params(Arch::Ampere, ...), dot_int, dot_b1_and_popc), then compared bit for
// bit. Fragment layouts follow the PTX ISA "Matrix Fragments for mma.m16n8k*" sections
// (9.7.16.5.6-9.7.16.5.13). The m16n8k256 .b1 A fragment uses col = threadID_in_group * 32 +
// (i & 0x1F) (+ 128 for i >= 64): the printed "+ i for i < 64" would run past the 128 columns a
// register pair covers and contradicts the m16n8k128 pattern [inferred].
//
// Inputs come from a per-tile splitmix64 stream (seed, combination, tile), so results do not depend
// on --threads or --batch-tiles. Every tile draws one category:
//   float  uniform bits, narrow / wide exponents, specials (+-0, subnormals, inf, NaN payloads),
//          cancellation, c-dominant, F-boundary terms per L_max block, overflow, e_zero region
//   int    uniform, extreme values, sums that cross INT32 limits (mixed-sign orderings tell
//          "clamp after the exact sum" from saturating partial sums)
//   b1     uniform, bit densities 0..1, accumulators that wrap
// D is pre-filled with 0xA5A5A5A5 so a kernel that did not write a lane shows up as mismatches.
//
// Build: nvcc -std=c++17 -O2 -arch=sm_80 -I include tests/numerics/test_sm80_mma.cu \
//          -Xcompiler -pthread -o test_sm80_mma
// Run:   ./test_sm80_mma [--dots N] [--seed S] [--threads N] [--only SUBSTRING] [--max-print N]
//          [--batch-tiles N] [--allow-non-sm80]
// Output: per combination "WGEMU-TEST name=<id> cases=<dots> mismatches=<m>", WGEMU-MISMATCH
// details for the first mismatches, then WGEMU-SUMMARY and WGEMU-TEST-END. Exit 0 only if all
// combinations ran with zero mismatches.
#include <cuda_runtime.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "wgmma_emu/numerics.h"

#define WGEMU_CUDA_CHECK(x)                                                         \
  do {                                                                              \
    const cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                                      \
      fprintf(stderr, "WGEMU-ERROR cuda %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
              cudaGetErrorString(err_));                                            \
      exit(2);                                                                      \
    }                                                                               \
  } while (0)

namespace {

using wgemu::Arch;
using wgemu::EType;
using wgemu::FdpaParams;

// ---------------------------------------------------------------------------------------------
// Combinations. Integer ops are kIntBase + (k16 ? 8 : 0) + (a_signed ? 4 : 0) + (b_signed ? 2 : 0)
// + (satfinite ? 1 : 0).
enum Op : int {
  kF16F32K16,
  kF16F32K8,
  kF16F16K16,
  kF16F16K8,
  kBF16F32K16,
  kBF16F32K8,
  kTF32F32K8,
  kTF32F32K4,
  kIntBase,
  kB1K256 = kIntBase + 16,
  kNumOps
};

__host__ __device__ constexpr bool op_is_int(int op) { return op >= kIntBase && op < kB1K256; }
__host__ __device__ constexpr bool op_is_16bit(int op) { return op <= kBF16F32K8; }
__host__ __device__ constexpr bool op_d_f16(int op) { return op == kF16F16K16 || op == kF16F16K8; }
__host__ __device__ constexpr int op_k(int op) {
  return op == kF16F32K16 || op == kF16F16K16 || op == kBF16F32K16                    ? 16
         : op == kF16F32K8 || op == kF16F16K8 || op == kBF16F32K8 || op == kTF32F32K8 ? 8
         : op == kTF32F32K4                                                           ? 4
         : op == kB1K256                                                              ? 256
         : ((op - kIntBase) & 8)                                                      ? 16
                                                                                      : 32;
}
// Stored units per row: elements, or 32-bit words for b1.
__host__ __device__ constexpr int op_units(int op) { return op == kB1K256 ? 8 : op_k(op); }

// ---------------------------------------------------------------------------------------------
// Device side.
#define WGEMU_MMA_F32_A4B2(INSTR)                                                              \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"            \
               : "=f"(fd[0]), "=f"(fd[1]), "=f"(fd[2]), "=f"(fd[3])                            \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(fc[0]), \
                 "f"(fc[1]), "f"(fc[2]), "f"(fc[3]))
#define WGEMU_MMA_F32_A2B1(INSTR)                                       \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n" \
               : "=f"(fd[0]), "=f"(fd[1]), "=f"(fd[2]), "=f"(fd[3])     \
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "f"(fc[0]), "f"(fc[1]), "f"(fc[2]), "f"(fc[3]))
#define WGEMU_MMA_F16_A4B2(INSTR)                                                             \
  asm volatile(INSTR " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"                           \
               : "=r"(d[0]), "=r"(d[1])                                                       \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), \
                 "r"(c[1]))
#define WGEMU_MMA_F16_A2B1(INSTR)                          \
  asm volatile(INSTR " {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n" \
               : "=r"(d[0]), "=r"(d[1])                    \
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]))
#define WGEMU_MMA_S32_A4B2(INSTR)                                                             \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"           \
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])                               \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), \
                 "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define WGEMU_MMA_S32_A2B1(INSTR)                                       \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n" \
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])         \
               : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]))
#define WGEMU_MMA_INT(SHAPE, SAT, AT, BT) \
  "mma.sync.aligned." SHAPE ".row.col" SAT ".s32." AT "." BT ".s32"

// d/c hold f32 bits, or two packed f16 in d[0..1]/c[0..1] for f16 accumulators.
template <int OP>
__device__ __forceinline__ void mma_op(uint32_t (&d)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2], const uint32_t (&c)[4]) {
  float fd[4];
  float fc[4];
  for (int i = 0; i < 4; ++i) fc[i] = __uint_as_float(c[i]);
  if constexpr (OP == kF16F32K16) {
    WGEMU_MMA_F32_A4B2("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32");
  } else if constexpr (OP == kF16F32K8) {
    WGEMU_MMA_F32_A2B1("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32");
  } else if constexpr (OP == kF16F16K16) {
    WGEMU_MMA_F16_A4B2("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16");
  } else if constexpr (OP == kF16F16K8) {
    WGEMU_MMA_F16_A2B1("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16");
  } else if constexpr (OP == kBF16F32K16) {
    WGEMU_MMA_F32_A4B2("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32");
  } else if constexpr (OP == kBF16F32K8) {
    WGEMU_MMA_F32_A2B1("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32");
  } else if constexpr (OP == kTF32F32K8) {
    WGEMU_MMA_F32_A4B2("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32");
  } else if constexpr (OP == kTF32F32K4) {
    WGEMU_MMA_F32_A2B1("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32");
  } else if constexpr (OP == kB1K256) {
    WGEMU_MMA_S32_A4B2("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc");
  } else if constexpr (op_is_int(OP)) {
    constexpr int v = OP - kIntBase;
    constexpr bool k16 = (v & 8) != 0, as = (v & 4) != 0, bs = (v & 2) != 0, sat = (v & 1) != 0;
#define WGEMU_INT_CASE(AS, BS, SAT, AT, BT, SATS)                  \
  if constexpr (as == AS && bs == BS && sat == SAT) {              \
    if constexpr (k16) {                                           \
      WGEMU_MMA_S32_A2B1(WGEMU_MMA_INT("m16n8k16", SATS, AT, BT)); \
    } else {                                                       \
      WGEMU_MMA_S32_A4B2(WGEMU_MMA_INT("m16n8k32", SATS, AT, BT)); \
    }                                                              \
  }
    WGEMU_INT_CASE(false, false, false, "u8", "u8", "")
    WGEMU_INT_CASE(false, false, true, "u8", "u8", ".satfinite")
    WGEMU_INT_CASE(false, true, false, "u8", "s8", "")
    WGEMU_INT_CASE(false, true, true, "u8", "s8", ".satfinite")
    WGEMU_INT_CASE(true, false, false, "s8", "u8", "")
    WGEMU_INT_CASE(true, false, true, "s8", "u8", ".satfinite")
    WGEMU_INT_CASE(true, true, false, "s8", "s8", "")
    WGEMU_INT_CASE(true, true, true, "s8", "s8", ".satfinite")
#undef WGEMU_INT_CASE
  }
  if constexpr (!op_d_f16(OP) && !op_is_int(OP) && OP != kB1K256) {
    for (int i = 0; i < 4; ++i) d[i] = __float_as_uint(fd[i]);
  }
}

__device__ __forceinline__ uint32_t pack_h2(uint32_t lo, uint32_t hi) {
  return (lo & 0xFFFFu) | (hi << 16);
}
__device__ __forceinline__ uint32_t pack_b4(const uint32_t* p) {
  return (p[0] & 0xFFu) | ((p[1] & 0xFFu) << 8) | ((p[2] & 0xFFu) << 16) | (p[3] << 24);
}

// A: [tile][16][units], B: [tile][8][units] (n-major, k contiguous), C/D: [tile][16][8]; one
// uint32 per element (low 16 / 8 bits for 16-bit / 8-bit types, one word per 32 b1 elements).
template <int OP>
__global__ void __launch_bounds__(256)
    mma_kernel(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B,
               const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int tiles) {
  constexpr int K = op_k(OP);
  constexpr int U = op_units(OP);
  const int tile = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  if (tile >= tiles) return;  // uniform per warp: mma.sync.aligned sees full warps only
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const uint32_t* a = A + static_cast<size_t>(tile) * 16 * U;
  const uint32_t* b = B + static_cast<size_t>(tile) * 8 * U;
  const uint32_t* c = C + static_cast<size_t>(tile) * 128;
  uint32_t* dst = D + static_cast<size_t>(tile) * 128;
  const int crow[4] = {g, g, g + 8, g + 8};
  const int ccol[4] = {2 * q, 2 * q + 1, 2 * q, 2 * q + 1};
  uint32_t ra[4] = {0, 0, 0, 0}, rb[2] = {0, 0}, rc[4], rd[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) rc[i] = c[crow[i] * 8 + ccol[i]];
  if constexpr (op_is_16bit(OP) && K == 16) {  // PTX 9.7.16.5.8
    for (int j = 0; j < 4; ++j) {
      const int row = g + 8 * (j & 1), k = 2 * q + 8 * (j >> 1);
      ra[j] = pack_h2(a[row * U + k], a[row * U + k + 1]);
    }
    for (int j = 0; j < 2; ++j) {
      const int k = 2 * q + 8 * j;
      rb[j] = pack_h2(b[g * U + k], b[g * U + k + 1]);
    }
  } else if constexpr (op_is_16bit(OP)) {  // K == 8, PTX 9.7.16.5.7
    for (int j = 0; j < 2; ++j) {
      const int row = g + 8 * j;
      ra[j] = pack_h2(a[row * U + 2 * q], a[row * U + 2 * q + 1]);
    }
    rb[0] = pack_h2(b[g * U + 2 * q], b[g * U + 2 * q + 1]);
  } else if constexpr (OP == kTF32F32K8) {  // PTX 9.7.16.5.7
    ra[0] = a[g * U + q];
    ra[1] = a[(g + 8) * U + q];
    ra[2] = a[g * U + q + 4];
    ra[3] = a[(g + 8) * U + q + 4];
    rb[0] = b[g * U + q];
    rb[1] = b[g * U + q + 4];
  } else if constexpr (OP == kTF32F32K4) {  // PTX 9.7.16.5.6
    ra[0] = a[g * U + q];
    ra[1] = a[(g + 8) * U + q];
    rb[0] = b[g * U + q];
  } else if constexpr (op_is_int(OP) && K == 32) {  // PTX 9.7.16.5.10
    for (int j = 0; j < 4; ++j) {
      const int row = g + 8 * (j & 1), k = 4 * q + 16 * (j >> 1);
      ra[j] = pack_b4(a + row * U + k);
    }
    for (int j = 0; j < 2; ++j) rb[j] = pack_b4(b + g * U + 4 * q + 16 * j);
  } else if constexpr (op_is_int(OP)) {  // K == 16, PTX 9.7.16.5.9
    for (int j = 0; j < 2; ++j) ra[j] = pack_b4(a + (g + 8 * j) * U + 4 * q);
    rb[0] = pack_b4(b + g * U + 4 * q);
  } else {  // b1 m16n8k256, PTX 9.7.16.5.13 (word w holds columns 32w .. 32w + 31)
    for (int j = 0; j < 4; ++j) ra[j] = a[(g + 8 * (j & 1)) * U + q + 4 * (j >> 1)];
    for (int j = 0; j < 2; ++j) rb[j] = b[g * U + q + 4 * j];
  }
  if constexpr (op_d_f16(OP)) {
    const uint32_t c0 = pack_h2(rc[0], rc[1]), c1 = pack_h2(rc[2], rc[3]);
    rc[0] = c0;
    rc[1] = c1;
    rc[2] = rc[3] = 0;
  }
  mma_op<OP>(rd, ra, rb, rc);
  if constexpr (op_d_f16(OP)) {
    dst[crow[0] * 8 + ccol[0]] = rd[0] & 0xFFFFu;
    dst[crow[1] * 8 + ccol[1]] = rd[0] >> 16;
    dst[crow[2] * 8 + ccol[2]] = rd[1] & 0xFFFFu;
    dst[crow[3] * 8 + ccol[3]] = rd[1] >> 16;
  } else {
    for (int i = 0; i < 4; ++i) dst[crow[i] * 8 + ccol[i]] = rd[i];
  }
}

using LaunchFn = void (*)(const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*, int);

template <int OP>
void launch(const uint32_t* A, const uint32_t* B, const uint32_t* C, uint32_t* D, int tiles) {
  mma_kernel<OP><<<(tiles + 7) / 8, 256>>>(A, B, C, D, tiles);
}

template <int... I>
constexpr std::array<LaunchFn, sizeof...(I)> make_launchers(std::integer_sequence<int, I...>) {
  return {{&launch<I>...}};
}
const std::array<LaunchFn, kNumOps> kLaunch =
    make_launchers(std::make_integer_sequence<int, kNumOps>{});

// ---------------------------------------------------------------------------------------------
// Host side: combinations, input generation, reference.
struct Combo {
  std::string name;
  int op, K, U;
  int kind;  // 0 float, 1 int, 2 b1
  EType at, bt;
  bool d_f16, as, bs, sat;
  FdpaParams p;
};

Combo make_combo(const std::string& name, int op, int kind, EType at, EType bt, bool as, bool bs,
                 bool sat) {
  Combo c;
  c.name = name;
  c.op = op;
  c.K = op_k(op);
  c.U = op_units(op);
  c.kind = kind;
  c.at = at;
  c.bt = bt;
  c.d_f16 = op_d_f16(op);
  c.as = as;
  c.bs = bs;
  c.sat = sat;
  c.p = kind == 0 ? wgemu::mma_params(Arch::Ampere, at, bt, c.d_f16)
                  : FdpaParams{0, 0, 0, wgemu::Rho::RZ_FP32};
  return c;
}

std::vector<Combo> make_combos() {
  std::vector<Combo> out;
  struct F {
    int op;
    const char* st;
    EType t;
  };
  const F fl[] = {{kF16F32K16, "m16n8k16.f32.f16.f16.f32", EType::F16},
                  {kF16F32K8, "m16n8k8.f32.f16.f16.f32", EType::F16},
                  {kF16F16K16, "m16n8k16.f16.f16.f16.f16", EType::F16},
                  {kF16F16K8, "m16n8k8.f16.f16.f16.f16", EType::F16},
                  {kBF16F32K16, "m16n8k16.f32.bf16.bf16.f32", EType::BF16},
                  {kBF16F32K8, "m16n8k8.f32.bf16.bf16.f32", EType::BF16},
                  {kTF32F32K8, "m16n8k8.f32.tf32.tf32.f32", EType::TF32},
                  {kTF32F32K4, "m16n8k4.f32.tf32.tf32.f32", EType::TF32}};
  for (const F& f : fl)
    out.push_back(
        make_combo(std::string("sm80.mma.") + f.st, f.op, 0, f.t, f.t, false, false, false));
  for (int v = 0; v < 16; ++v) {
    const bool k16 = v & 8, as = v & 4, bs = v & 2, sat = v & 1;
    const std::string name = std::string("sm80.mma.m16n8k") + (k16 ? "16" : "32") +
                             (sat ? ".satfinite" : "") + ".s32." + (as ? "s8" : "u8") + "." +
                             (bs ? "s8" : "u8") + ".s32";
    out.push_back(make_combo(name, kIntBase + v, 1, as ? EType::S8 : EType::U8,
                             bs ? EType::S8 : EType::U8, as, bs, sat));
  }
  out.push_back(make_combo("sm80.mma.m16n8k256.s32.b1.b1.s32.and.popc", kB1K256, 2, EType::B1,
                           EType::B1, false, false, false));
  return out;
}

uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

struct Rng {
  uint64_t s;
  Rng(uint64_t seed, uint64_t combo, uint64_t tile) : s(seed) {
    s ^= splitmix(s) + combo * 0xD1B54A32D192ED03ull;
    s ^= splitmix(s) + tile * 0x8CB92BA72F3D8DD7ull;
    splitmix(s);
  }
  uint64_t next() { return splitmix(s); }
  uint32_t u32() { return static_cast<uint32_t>(next() >> 32); }
  int range(int lo, int hi) {  // inclusive
    return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1));
  }
  double unif() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
  bool coin() { return (next() >> 63) != 0; }
};

struct FmtD {
  int nbits, ebits, mbits, bias;
  int top, emin, emax, esub;
  uint32_t sign;
};

FmtD make_fmt(int nbits, int ebits, int mbits, int bias) {  // IEEE-style specials
  FmtD f;
  f.nbits = nbits;
  f.ebits = ebits;
  f.mbits = mbits;
  f.bias = bias;
  f.top = (1 << ebits) - 1;
  f.emin = 1 - bias;
  f.emax = f.top - 1 - bias;
  f.esub = f.emin - mbits;
  f.sign = 1u << (ebits + mbits);
  return f;
}

const FmtD kF16 = make_fmt(16, 5, 10, 15);
const FmtD kBF16 = make_fmt(16, 8, 7, 127);
const FmtD kF32 = make_fmt(32, 8, 23, 127);  // also tf32 operands: random low 13 bits

const FmtD& fmt_of(EType t) { return t == EType::F16 ? kF16 : t == EType::BF16 ? kBF16 : kF32; }

uint32_t rbits(Rng& r, const FmtD& f) {
  return f.nbits == 32 ? r.u32() : r.u32() & ((1u << f.nbits) - 1);
}

// Finite value whose leading bit has exponent e (zero below the subnormal range).
uint32_t encode(Rng& r, const FmtD& f, bool neg, int e) {
  e = std::min(std::max(e, f.esub - 1), f.emax);
  const uint32_t rnd = static_cast<uint32_t>(r.next() & ((1ull << f.mbits) - 1));
  uint32_t expf = 0, mant = 0;
  if (e >= f.emin) {
    expf = static_cast<uint32_t>(e + f.bias);
    mant = rnd;
  } else if (e >= f.esub) {
    const int p = e - f.esub;
    mant = (1u << p) | (rnd & ((1u << p) - 1));
  }
  return (expf << f.mbits) | mant | (neg ? f.sign : 0u);
}

uint32_t special(Rng& r, const FmtD& f) {
  const uint32_t mmask = (1u << f.mbits) - 1, top = static_cast<uint32_t>(f.top) << f.mbits;
  const uint32_t vals[] = {0,
                           1,
                           mmask,
                           1u << f.mbits,
                           ((static_cast<uint32_t>(f.top) - 1) << f.mbits) | mmask,
                           static_cast<uint32_t>(f.bias) << f.mbits,
                           top,
                           top | (1u << (f.mbits - 1)),
                           top | 1u,
                           top | mmask};
  return vals[r.range(0, 9)] | (r.coin() ? f.sign : 0u);
}

double decode(uint32_t bits, const FmtD& f, bool tf32) {
  if (tf32) bits &= 0xFFFFE000u;
  const bool neg = (bits & f.sign) != 0;
  const int ef = static_cast<int>((bits >> f.mbits) & static_cast<uint32_t>(f.top));
  const uint32_t m = bits & ((1u << f.mbits) - 1);
  double v;
  if (ef == f.top)
    v = m ? NAN : INFINITY;
  else if (ef == 0)
    v = std::ldexp(static_cast<double>(m), f.esub);
  else
    v = std::ldexp(static_cast<double>(m | (1u << f.mbits)), ef - f.bias - f.mbits);
  return neg ? -v : v;
}

// Bits of v in format f if v is finite and exactly representable.
bool exact_bits(double v, const FmtD& f, uint32_t* out) {
  if (!std::isfinite(v)) return false;
  const uint32_t sign = std::signbit(v) ? f.sign : 0u;
  v = std::fabs(v);
  if (v == 0) {
    *out = sign;
    return true;
  }
  int ex;
  std::frexp(v, &ex);
  const int e = ex - 1;
  if (e > f.emax) return false;
  const int scale = e >= f.emin ? f.mbits - e : f.mbits - f.emin;
  const double s = std::ldexp(v, scale);
  if (s != std::floor(s)) return false;
  const uint32_t si = static_cast<uint32_t>(s);
  const uint32_t normal = (static_cast<uint32_t>(e + f.bias) << f.mbits) | (si - (1u << f.mbits));
  *out = sign | (e >= f.emin ? normal : si);
  return true;
}

const char* const kFloatCats[] = {"uniform", "narrow", "wide",     "specials", "cancel",
                                  "cdom",    "fbound", "overflow", "ezero"};
const int kFloatW[] = {30, 14, 10, 10, 10, 8, 8, 4, 6};
const char* const kIntCats[] = {"uniform", "extremes", "boundary"};
const int kIntW[] = {50, 20, 30};
const char* const kB1Cats[] = {"uniform", "density", "boundary"};
const int kB1W[] = {50, 25, 25};

int pick(Rng& r, const int* w, int n) {
  int total = 0;
  for (int i = 0; i < n; ++i) total += w[i];
  int x = r.range(0, total - 1);
  for (int i = 0; i < n; ++i) {
    if (x < w[i]) return i;
    x -= w[i];
  }
  return n - 1;
}

void gen_float_tile(Rng& r, int cat, const Combo& cb, uint32_t* A, uint32_t* B, uint32_t* C) {
  const int K = cb.K, L = cb.p.L_max, F = cb.p.F, ez = cb.p.e_zero;
  const FmtD& fa = fmt_of(cb.at);
  const FmtD& fb = fmt_of(cb.bt);
  const FmtD& fc = cb.d_f16 ? kF16 : kF32;
  const bool tf32 = cb.at == EType::TF32;
  int ca[16], cbn[8];
  for (int m = 0; m < 16; ++m) ca[m] = r.range(fa.esub, fa.emax);
  for (int n = 0; n < 8; ++n) cbn[n] = r.range(fb.esub, fb.emax);
  auto narrow = [&] {
    for (int m = 0; m < 16; ++m)
      for (int k = 0; k < K; ++k) A[m * K + k] = encode(r, fa, r.coin(), ca[m] + r.range(-2, 2));
    for (int n = 0; n < 8; ++n)
      for (int k = 0; k < K; ++k) B[n * K + k] = encode(r, fb, r.coin(), cbn[n] + r.range(-2, 2));
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < 8; ++n)
        C[m * 8 + n] = encode(r, fc, r.coin(), ca[m] + cbn[n] + r.range(-4, 4));
  };
  switch (cat) {
    case 0:  // uniform
      for (int i = 0; i < 16 * K; ++i) A[i] = rbits(r, fa);
      for (int i = 0; i < 8 * K; ++i) B[i] = rbits(r, fb);
      for (int i = 0; i < 128; ++i) C[i] = rbits(r, fc);
      break;
    case 1:  // narrow
      narrow();
      break;
    case 2:  // wide
      for (int i = 0; i < 16 * K; ++i) A[i] = encode(r, fa, r.coin(), r.range(fa.esub, fa.emax));
      for (int i = 0; i < 8 * K; ++i) B[i] = encode(r, fb, r.coin(), r.range(fb.esub, fb.emax));
      for (int i = 0; i < 128; ++i) C[i] = encode(r, fc, r.coin(), r.range(fc.esub, fc.emax));
      break;
    case 3:  // specials
      narrow();
      for (int i = 0; i < 16 * K; ++i)
        if (r.unif() < 0.12) A[i] = special(r, fa);
      for (int i = 0; i < 8 * K; ++i)
        if (r.unif() < 0.12) B[i] = special(r, fb);
      for (int i = 0; i < 128; ++i)
        if (r.unif() < 0.25) C[i] = special(r, fc);
      break;
    case 4:  // cancel: a_j b_j + a_j (-b_j) pairs, optionally one ulp off (tf32: an ignored bit)
      narrow();
      for (int j = 1; j + 1 < K; j += 2) {
        for (int m = 0; m < 16; ++m) A[m * K + j + 1] = A[m * K + j];
        for (int n = 0; n < 8; ++n) {
          B[n * K + j + 1] = B[n * K + j] ^ fb.sign;
          if (r.unif() < 0.4) B[n * K + j + 1] ^= 1u;
        }
      }
      for (int m = 0; m < 16; ++m)
        if (r.coin()) A[m * K + K - 1] = 0;
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
          const double v = -(decode(A[m * K], fa, tf32) * decode(B[n * K], fb, tf32));
          uint32_t bits;
          if (r.unif() < 0.6 && exact_bits(v, fc, &bits)) C[m * 8 + n] = bits;
        }
      break;
    case 5: {  // cdom
      narrow();
      const int far[] = {F - 2, F - 1, F, F + 1, F + 2, F + 3, 40, 80};
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
          const int d = r.coin() ? far[r.range(0, 7)] : -r.range(F - 3, F + 3);
          C[m * 8 + n] = encode(r, fc, r.coin(), ca[m] + cbn[n] + d);
        }
      break;
    }
    case 6: {  // fbound: block-leading term dominant, the rest F +- 3 binades below it
      int ta[16];
      const int lo = std::min(fa.esub + F + 3, fa.emax);
      for (int m = 0; m < 16; ++m) ta[m] = r.range(lo, fa.emax);
      for (int m = 0; m < 16; ++m)
        for (int k = 0; k < K; ++k)
          A[m * K + k] = encode(r, fa, r.coin(), k % L == 0 ? ta[m] : ta[m] - F + r.range(-3, 3));
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < K; ++k) B[n * K + k] = encode(r, fb, r.coin(), cbn[n]);
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n)
          C[m * 8 + n] = r.coin() ? (r.coin() ? fc.sign : 0u)
                                  : encode(r, fc, r.coin(), ta[m] + cbn[n] - F + r.range(-3, 3));
      break;
    }
    case 7: {  // overflow: same-sign products and c at the largest exponents
      bool neg[16];
      for (int m = 0; m < 16; ++m) neg[m] = r.coin();
      for (int m = 0; m < 16; ++m)
        for (int k = 0; k < K; ++k) A[m * K + k] = encode(r, fa, neg[m], fa.emax - r.range(0, 1));
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < K; ++k) B[n * K + k] = encode(r, fb, false, fb.emax - r.range(0, 1));
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n)
          C[m * 8 + n] = r.coin() ? encode(r, fc, neg[m], fc.emax - r.range(0, 1))
                                  : encode(r, fc, r.coin(), r.range(fc.esub, fc.emax));
      break;
    }
    default: {  // ezero: c = +-0, products around e_zero - F - 4 .. e_zero + 4 where the
                // operand ranges reach (otherwise the smallest subnormals)
      const int ha = ez / 2;
      for (int m = 0; m < 16; ++m) {
        const int xa = ha + r.range(-2, 2);
        for (int k = 0; k < K; ++k)
          A[m * K + k] = encode(r, fa, r.coin(), std::max(fa.esub, xa + r.range(-1, 1)));
      }
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < K; ++k)
          B[n * K + k] = encode(r, fb, r.coin(), std::max(fb.esub, ez - ha + r.range(-F - 4, 4)));
      for (int i = 0; i < 128; ++i) C[i] = r.coin() ? fc.sign : 0u;
      break;
    }
  }
}

void gen_int_tile(Rng& r, int cat, const Combo& cb, uint32_t* A, uint32_t* B, uint32_t* C) {
  const int K = cb.K;
  static const uint32_t ext_s[] = {0x80, 0x7F, 0xFF, 0x00, 0x01};  // -128 127 -1 0 1
  static const uint32_t ext_u[] = {0x00, 0xFF, 0x80, 0x7F, 0x01};
  if (cat == 0) {
    for (int i = 0; i < 16 * K; ++i) A[i] = r.u32() & 0xFFu;
    for (int i = 0; i < 8 * K; ++i) B[i] = r.u32() & 0xFFu;
    for (int i = 0; i < 128; ++i) C[i] = r.u32();
  } else if (cat == 1) {
    for (int i = 0; i < 16 * K; ++i) A[i] = (cb.as ? ext_s : ext_u)[r.range(0, 4)];
    for (int i = 0; i < 8 * K; ++i) B[i] = (cb.bs ? ext_s : ext_u)[r.range(0, 4)];
    for (int i = 0; i < 128; ++i) C[i] = r.u32();
  } else {
    // Large products whose sign follows a per-tile k pattern (the signed operand carries it; two
    // unsigned operands give positive sums only) and c within 2^19 of INT32_MAX / INT32_MIN.
    const int pattern = r.range(0, 3);
    bool pos[32];
    for (int k = 0; k < K; ++k)
      pos[k] = pattern == 0   ? k < K / 2
               : pattern == 1 ? k >= K / 2
               : pattern == 2 ? (k & 1) == 0
                              : r.coin();
    auto big = [&](bool is_signed, bool positive) -> uint32_t {
      if (!is_signed) return static_cast<uint32_t>(r.range(228, 255));
      return positive ? static_cast<uint32_t>(r.range(100, 127))
                      : static_cast<uint32_t>(256 - r.range(100, 128));
    };
    for (int m = 0; m < 16; ++m)
      for (int k = 0; k < K; ++k) A[m * K + k] = big(cb.as, pos[k]);
    for (int n = 0; n < 8; ++n)
      for (int k = 0; k < K; ++k) B[n * K + k] = big(cb.bs, cb.as ? true : pos[k]);
    for (int i = 0; i < 128; ++i) {
      const int32_t off = r.range(0, (1 << 19) - 1);
      C[i] = static_cast<uint32_t>(r.coin() ? INT32_MAX - off : INT32_MIN + off);
    }
  }
}

void gen_b1_tile(Rng& r, int cat, uint32_t* A, uint32_t* B, uint32_t* C) {
  if (cat == 1) {
    static const double dens[] = {0.0, 0.05, 0.5, 0.95, 1.0};
    auto fill = [&](uint32_t* row) {
      const double d = dens[r.range(0, 4)];
      for (int w = 0; w < 8; ++w) {
        uint32_t x = 0;
        for (int t = 0; t < 32; ++t) x |= (r.unif() < d ? 1u : 0u) << t;
        row[w] = x;
      }
    };
    for (int m = 0; m < 16; ++m) fill(A + m * 8);
    for (int n = 0; n < 8; ++n) fill(B + n * 8);
  } else {
    for (int i = 0; i < 128; ++i) A[i] = r.u32();
    for (int i = 0; i < 64; ++i) B[i] = r.u32();
  }
  for (int i = 0; i < 128; ++i) {
    if (cat == 2) {
      const int32_t off = r.range(0, 511);  // popc sum <= 256: some of these wrap
      C[i] = static_cast<uint32_t>(r.coin() ? INT32_MAX - off : INT32_MIN + off);
    } else {
      C[i] = r.u32();
    }
  }
}

int gen_tile(uint64_t seed, int combo_idx, uint64_t tile, const Combo& cb, uint32_t* A, uint32_t* B,
             uint32_t* C) {
  Rng r(seed, static_cast<uint64_t>(combo_idx), tile);
  int cat;
  if (cb.kind == 0) {
    cat = pick(r, kFloatW, 9);
    gen_float_tile(r, cat, cb, A, B, C);
  } else if (cb.kind == 1) {
    cat = pick(r, kIntW, 3);
    gen_int_tile(r, cat, cb, A, B, C);
  } else {
    cat = pick(r, kB1W, 3);
    gen_b1_tile(r, cat, A, B, C);
  }
  return cat;
}

uint32_t reference(const Combo& cb, const uint32_t* a, const uint32_t* b, uint32_t c) {
  if (cb.kind == 0) return wgemu::dot_float(a, b, cb.K, cb.at, cb.bt, c, cb.d_f16, cb.p);
  if (cb.kind == 1)
    return static_cast<uint32_t>(
        wgemu::dot_int(a, b, cb.K, cb.as, cb.bs, static_cast<int32_t>(c), cb.sat));
  return static_cast<uint32_t>(wgemu::dot_b1_and_popc(a, b, cb.U, static_cast<int32_t>(c)));
}

template <typename Fn>
void parallel_for(int threads, uint64_t n, Fn fn) {
  const int nt = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(threads), n));
  std::vector<std::thread> pool;
  for (int t = 0; t < nt; ++t) pool.emplace_back([=] { fn(t, n * t / nt, n * (t + 1) / nt); });
  for (std::thread& th : pool) th.join();
}

std::string hex_row(const uint32_t* p, int n, int digits) {
  std::string s;
  char buf[16];
  for (int i = 0; i < n; ++i) {
    snprintf(buf, sizeof(buf), "%s%0*x", i ? "," : "", digits, p[i]);
    s += buf;
  }
  return s;
}

int affinity_cpus() {
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) return CPU_COUNT(&set);
  const unsigned n = std::thread::hardware_concurrency();
  return n ? static_cast<int>(n) : 1;
}

struct Options {
  uint64_t dots = 10000000;
  uint64_t seed = 20261125;
  int threads = 1;
  int max_print = 5;
  int batch_tiles = 32768;
  bool allow_non_sm80 = false;
  std::string only;
};

struct Miss {
  uint64_t tile;
  int m, n;
  uint32_t hw, model;
};

// Returns true if the combination passed.
bool run_combo(const Options& o, int ci, const Combo& cb) {
  const uint64_t tiles = (o.dots + 127) / 128;
  const int U = cb.U;
  const size_t bt = static_cast<size_t>(std::min<uint64_t>(tiles, uint64_t(o.batch_tiles)));
  std::vector<uint32_t> hA(bt * 16 * U), hB(bt * 8 * U), hC(bt * 128), hD(bt * 128);
  std::vector<uint8_t> cats(bt);
  uint32_t *dA, *dB, *dC, *dD;
  WGEMU_CUDA_CHECK(cudaMalloc(&dA, hA.size() * 4));
  WGEMU_CUDA_CHECK(cudaMalloc(&dB, hB.size() * 4));
  WGEMU_CUDA_CHECK(cudaMalloc(&dC, hC.size() * 4));
  WGEMU_CUDA_CHECK(cudaMalloc(&dD, hD.size() * 4));
  const int ncat = cb.kind == 0 ? 9 : 3;
  const char* const* cat_names = cb.kind == 0 ? kFloatCats : cb.kind == 1 ? kIntCats : kB1Cats;
  std::vector<uint64_t> cat_dots(ncat, 0);
  uint64_t cases = 0, mismatches = 0;
  int printed = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t t0_tile = 0; t0_tile < tiles; t0_tile += bt) {
    const uint64_t nb = std::min<uint64_t>(bt, tiles - t0_tile);
    parallel_for(o.threads, nb, [&](int, uint64_t lo, uint64_t hi) {
      for (uint64_t t = lo; t < hi; ++t)
        cats[t] = static_cast<uint8_t>(
            gen_tile(o.seed, ci, t0_tile + t, cb, &hA[t * 16 * U], &hB[t * 8 * U], &hC[t * 128]));
    });
    WGEMU_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nb * 16 * U * 4, cudaMemcpyHostToDevice));
    WGEMU_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nb * 8 * U * 4, cudaMemcpyHostToDevice));
    WGEMU_CUDA_CHECK(cudaMemcpy(dC, hC.data(), nb * 128 * 4, cudaMemcpyHostToDevice));
    WGEMU_CUDA_CHECK(cudaMemset(dD, 0xA5, nb * 128 * 4));
    kLaunch[cb.op](dA, dB, dC, dD, static_cast<int>(nb));
    WGEMU_CUDA_CHECK(cudaGetLastError());
    WGEMU_CUDA_CHECK(cudaDeviceSynchronize());
    WGEMU_CUDA_CHECK(cudaMemcpy(hD.data(), dD, nb * 128 * 4, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> part_miss(o.threads, 0);
    std::vector<std::vector<Miss>> part_first(o.threads);
    parallel_for(o.threads, nb, [&](int ti, uint64_t lo, uint64_t hi) {
      for (uint64_t t = lo; t < hi; ++t)
        for (int m = 0; m < 16; ++m)
          for (int n = 0; n < 8; ++n) {
            const uint32_t hw = hD[t * 128 + m * 8 + n];
            const uint32_t model =
                reference(cb, &hA[(t * 16 + m) * U], &hB[(t * 8 + n) * U], hC[t * 128 + m * 8 + n]);
            if (hw == model) continue;
            ++part_miss[ti];
            if (part_first[ti].size() < static_cast<size_t>(o.max_print))
              part_first[ti].push_back({t, m, n, hw, model});
          }
    });
    std::vector<Miss> first;
    for (int ti = 0; ti < o.threads; ++ti) {
      mismatches += part_miss[ti];
      first.insert(first.end(), part_first[ti].begin(), part_first[ti].end());
    }
    std::sort(first.begin(), first.end(), [](const Miss& x, const Miss& y) {
      return x.tile != y.tile ? x.tile < y.tile : x.m != y.m ? x.m < y.m : x.n < y.n;
    });
    const int digits = cb.kind == 0 ? (cb.at == EType::TF32 ? 8 : 4) : cb.kind == 1 ? 2 : 8;
    const int cdig = cb.d_f16 ? 4 : 8;
    for (const Miss& x : first) {
      if (printed >= o.max_print) break;
      ++printed;
      const uint64_t t = x.tile;
      printf(
          "WGEMU-MISMATCH name=%s tile=%llu m=%d n=%d category=%s K=%d a=[%s] b=[%s] c=%0*x "
          "hw=%0*x model=%0*x\n",
          cb.name.c_str(), static_cast<unsigned long long>(t0_tile + t), x.m, x.n,
          cat_names[cats[t]], cb.K, hex_row(&hA[(t * 16 + x.m) * U], U, digits).c_str(),
          hex_row(&hB[(t * 8 + x.n) * U], U, digits).c_str(), cdig, hC[t * 128 + x.m * 8 + x.n],
          cdig, x.hw, cdig, x.model);
    }
    for (uint64_t t = 0; t < nb; ++t) cat_dots[cats[t]] += 128;
    cases += nb * 128;
  }
  WGEMU_CUDA_CHECK(cudaFree(dA));
  WGEMU_CUDA_CHECK(cudaFree(dB));
  WGEMU_CUDA_CHECK(cudaFree(dC));
  WGEMU_CUDA_CHECK(cudaFree(dD));
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::string cat_str;
  for (int i = 0; i < ncat; ++i)
    cat_str += std::string(i ? "," : "") + cat_names[i] + ":" + std::to_string(cat_dots[i]);
  if (cb.kind == 0)
    printf(
        "WGEMU-INFO name=%s params=(F=%d,L_max=%d,e_zero=%d,rho=%u) categories=%s seconds=%.2f\n",
        cb.name.c_str(), cb.p.F, cb.p.L_max, cb.p.e_zero, static_cast<unsigned>(cb.p.rho),
        cat_str.c_str(), sec);
  else
    printf("WGEMU-INFO name=%s categories=%s seconds=%.2f\n", cb.name.c_str(), cat_str.c_str(),
           sec);
  printf("WGEMU-TEST name=%s cases=%llu mismatches=%llu\n", cb.name.c_str(),
         static_cast<unsigned long long>(cases), static_cast<unsigned long long>(mismatches));
  fflush(stdout);
  return cases > 0 && mismatches == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  o.threads = affinity_cpus();
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--dots" && has) {
      o.dots = strtoull(argv[++i], nullptr, 10);
    } else if (a == "--seed" && has) {
      o.seed = strtoull(argv[++i], nullptr, 10);
    } else if (a == "--threads" && has) {
      o.threads = atoi(argv[++i]);
    } else if (a == "--max-print" && has) {
      o.max_print = atoi(argv[++i]);
    } else if (a == "--batch-tiles" && has) {
      o.batch_tiles = atoi(argv[++i]);
    } else if (a == "--only" && has) {
      o.only = argv[++i];
    } else if (a == "--allow-non-sm80") {
      o.allow_non_sm80 = true;
    } else {
      fprintf(stderr,
              "usage: %s [--dots N] [--seed S] [--threads N] [--only SUBSTRING] [--max-print N] "
              "[--batch-tiles N] [--allow-non-sm80]\n",
              argv[0]);
      return 2;
    }
  }
  if (o.dots < 1 || o.threads < 1 || o.max_print < 0 || o.batch_tiles < 1) {
    fprintf(stderr, "test_sm80_mma: --dots, --threads, --batch-tiles must be >= 1\n");
    return 2;
  }
  int dev = 0;
  cudaDeviceProp prop;
  WGEMU_CUDA_CHECK(cudaGetDevice(&dev));
  WGEMU_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const bool sm80 = prop.major == 8 && prop.minor == 0;
  printf("WGEMU-INFO device=%d cc=%d.%d sms=%d threads=%d dots_per_combo>=%llu seed=%llu\n", dev,
         prop.major, prop.minor, prop.multiProcessorCount, o.threads,
         static_cast<unsigned long long>(o.dots), static_cast<unsigned long long>(o.seed));
  if (!sm80 && !(o.allow_non_sm80 && prop.major * 10 + prop.minor >= 80)) {
    fprintf(stderr,
            "test_sm80_mma: device cc %d.%d is not 8.0 (use --allow-non-sm80 for cc >= 8.0)\n",
            prop.major, prop.minor);
    printf("WGEMU-TEST name=device cases=0 mismatches=0\nWGEMU-TEST-END\n");
    return 1;
  }
  const std::vector<Combo> combos = make_combos();
  int ran = 0, failed = 0;
  for (size_t i = 0; i < combos.size(); ++i) {
    if (!o.only.empty() && combos[i].name.find(o.only) == std::string::npos) continue;
    ++ran;
    if (!run_combo(o, static_cast<int>(i), combos[i])) ++failed;
  }
  if (ran == 0) {
    printf("WGEMU-TEST name=selection:%s cases=0 mismatches=0\n", o.only.c_str());
    failed = 1;
  }
  printf("WGEMU-SUMMARY combinations=%d failed=%d result=%s\n", ran, failed,
         failed ? "FAIL" : "PASS");
  printf("WGEMU-TEST-END\n");
  return failed ? 1 : 0;
}
