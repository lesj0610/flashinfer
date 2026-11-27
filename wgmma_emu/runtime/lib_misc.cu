// Library entry points of the miscellaneous sm_89/sm_90 rules (PLAN section 6). Owner: T2.
// Included by runtime/emu_lib.cu (library PTX) and directly by tests/misc. The translator side
// is tools/ptxemu/rules_misc.py; the flag words below must match its LIB_* / *_SHIFT constants.
//
//   u32  wgemu_cvt_f32x2_to_fp8x2(u32 a, u32 b, u32 flags)     cvt.rn.satfinite{.relu}.f8x2.f32
//   u32  wgemu_cvt_f16x2_to_fp8x2(u32 a, u32 flags)            cvt.rn.satfinite{.relu}.f8x2.f16x2
//   u32  wgemu_cvt_fp8x2_to_f16x2(u32 a, u32 flags)            cvt.rn{.relu}.f16x2.f8x2 (a: low 16)
//        flags: bit0 e5m2 (else e4m3), bit1 relu, bit2 satfinite
//   u32  wgemu_cvt_f32_to_tf32(u32 a, u32 ctrl)                cvt.{rn,rz}{.relu}.tf32.f32
//        ctrl: bits 0..1 rounding (0 rn, 1 rz), bit2 relu
//   u32  wgemu_cvt_to_bf16(u64 a, u32 ctrl)                    cvt.frnd{.ftz}{.sat}.bf16.atype
//        ctrl: bits 0..3 source (0 f32, 1 f16, 2 f64, 3..6 u8/u16/u32/u64, 7..10 s8/s16/s32/s64),
//        bits 4..6 rounding (0 rn, 1 rz, 2 rm, 3 rp), bit8 ftz (f32 source), bit9 sat ([0, 1])
//   void wgemu_mma_fp8(u32 a0..a3, u32 b0, b1, u32 c0..c3, u64 out, u32 cfg)
//        mma.sync.aligned.m16n8k{16,32}.row.col.{f16,f32}.f8.f8.{f16,f32}; unused registers 0
//        cfg: bit0 A e5m2, bit1 B e5m2, bit2 f16 C/D, bit3 k32, bit4 Ada numerics (else Hopper)
//        out: generic pointer to 4 (f32) or 2 (f16) u32 D registers
//   void wgemu_mma_sp_fp8(u32 a0..a3, u32 b0..b3, u32 c0..c3, u32 e, u64 out, u32 cfg)
//        mma.sp{::ordered_metadata}.sync.aligned.m16n8k64.row.col.f32.f8.f8.f32, selector 0
//        cfg: as wgemu_mma_fp8 (bit2, bit3 must be 0)
//   void wgemu_mma_f64(u64 a0..a7, u64 b0..b3, u64 c0..c3, u64 out, u32 cfg)
//        mma.sync.aligned.m16n8k{4,8,16}.row.col{.rnd}.f64.f64.f64.f64; unused registers 0
//        cfg: bits 0..1 K (0: 4, 1: 8, 2: 16), bits 2..3 rounding (0 rn, 1 rz, 2 rm, 3 rp)
//   void wgemu_atom_b128(u64 addr, u64 locks, u32 cfg, u64 cmp_lo, cmp_hi, new_lo, new_hi, u64 out)
//        atom{.sem}{.scope}{.space}.{cas,exch}.b128: addr generic, locks generic pointer to the
//        module's kB128LockCount-word lock table; cfg bit0 cas (else exch), bit1 .sys scope
//        out: generic pointer to the 16-byte old value
//
// Every mma entry point must be called by all 32 lanes of the warp (mma.sync is .aligned).
// Numerics: fp8 mma uses MMA-Sim's T-FDPA model (include/wgmma_emu/numerics.h, mma_params);
// f64 mma is a sequential fma chain k = 0 .. K-1 rounded with the instruction's mode.
#include "wgmma_emu/cvt.h"
#include "wgmma_emu/misc_device.cuh"
#include "wgmma_emu/numerics.h"

namespace wgemu {
namespace misc {

constexpr uint32_t kFp8E5m2 = 1u, kFp8Relu = 2u, kFp8Sat = 4u;
constexpr uint32_t kTf32Relu = 4u;
constexpr uint32_t kBf16Ftz = 1u << 8, kBf16Sat = 1u << 9;
constexpr uint32_t kMmaAE5m2 = 1u, kMmaBE5m2 = 2u, kMmaDF16 = 4u, kMmaK32 = 8u, kMmaAda = 16u;
constexpr uint32_t kB128Cas = 1u, kB128Sys = 2u;

WGEMU_DEV cvt::Rnd rnd_of(uint32_t code) {
  switch (code & 3u) {
    case 0:
      return cvt::RN;
    case 1:
      return cvt::RZ;
    case 2:
      return cvt::RM;
    default:
      return cvt::RP;
  }
}

// cvt.sat with a floating-point destination clamps to [0.0, 1.0]; NaN -> +0 ([inference]).
WGEMU_DEV uint32_t bf16_sat(uint32_t h) {
  h &= 0xFFFFu;
  if ((h & 0x7FFFu) > 0x7F80u) return 0u;  // NaN
  if (h & 0x8000u) return 0u;              // negative, -0
  return h > 0x3F80u ? 0x3F80u : h;
}

WGEMU_DEV EType fp8_type(bool e5m2) { return e5m2 ? EType::E5M2 : EType::E4M3; }

WGEMU_DEV Arch mma_arch(uint32_t cfg) { return (cfg & kMmaAda) ? Arch::Ada : Arch::Hopper; }

// D/C element (ri, ci) of this lane: row g + 8 * ri, column 2q + ci (cd_loc).
template <int K>
WGEMU_DEV void mma_fp8_dense(const uint32_t (&a)[4], const uint32_t (&b)[4], const uint32_t (&c)[4],
                             uint32_t* out, uint32_t cfg) {
  uint32_t rows[2][K], cols[2][K];
  gather_a8<K>(a, rows);
  gather_b8<K>(b, cols);
  const EType at = fp8_type(cfg & kMmaAE5m2), bt = fp8_type(cfg & kMmaBE5m2);
  const bool d_f16 = (cfg & kMmaDF16) != 0;
  const FdpaParams p = mma_params(mma_arch(cfg), at, bt, d_f16);
  uint32_t d[2][2];
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
#pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
      // f32: register 2 * ri + ci; f16: half ci of register ri.
      const uint32_t cv = d_f16 ? (c[ri] >> (16 * ci)) & 0xFFFFu : c[2 * ri + ci];
      d[ri][ci] = dot_float(rows[ri], cols[ci], K, at, bt, cv, d_f16, p);
    }
  }
  if (d_f16) {
    out[0] = (d[0][0] & 0xFFFFu) | (d[0][1] << 16);
    out[1] = (d[1][0] & 0xFFFFu) | (d[1][1] << 16);
  } else {
    out[0] = d[0][0];
    out[1] = d[0][1];
    out[2] = d[1][0];
    out[3] = d[1][1];
  }
}

template <int K>
WGEMU_DEV void mma_f64(const uint64_t (&a)[8], const uint64_t (&b)[4], const uint64_t (&c)[4],
                       uint64_t* out, uint32_t rnd) {
  uint64_t rows[2][K], cols[2][K];
  gather_a64<K>(a, rows);
  gather_b64<K>(b, cols);
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
#pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
      double acc = __longlong_as_double((long long)c[2 * ri + ci]);
#pragma unroll
      for (int k = 0; k < K; ++k) {
        const double x = __longlong_as_double((long long)rows[ri][k]);
        const double y = __longlong_as_double((long long)cols[ci][k]);
        switch (rnd) {
          case 0:
            acc = __fma_rn(x, y, acc);
            break;
          case 1:
            acc = __fma_rz(x, y, acc);
            break;
          case 2:
            acc = __fma_rd(x, y, acc);
            break;
          default:
            acc = __fma_ru(x, y, acc);
            break;
        }
      }
      out[2 * ri + ci] = (uint64_t)__double_as_longlong(acc);
    }
  }
}

}  // namespace misc
}  // namespace wgemu

extern "C" {

__device__ __noinline__ uint32_t wgemu_cvt_f32x2_to_fp8x2(uint32_t a, uint32_t b, uint32_t flags) {
  using namespace wgemu::misc;
  return wgemu::cvt::f32x2_to_fp8x2(a, b, flags & kFp8E5m2, flags & kFp8Relu, flags & kFp8Sat);
}

__device__ __noinline__ uint32_t wgemu_cvt_f16x2_to_fp8x2(uint32_t a, uint32_t flags) {
  using namespace wgemu::misc;
  return wgemu::cvt::f16x2_to_fp8x2(a, flags & kFp8E5m2, flags & kFp8Relu, flags & kFp8Sat);
}

__device__ __noinline__ uint32_t wgemu_cvt_fp8x2_to_f16x2(uint32_t a, uint32_t flags) {
  using namespace wgemu::misc;
  return wgemu::cvt::fp8x2_to_f16x2(a & 0xFFFFu, flags & kFp8E5m2, flags & kFp8Relu);
}

__device__ __noinline__ uint32_t wgemu_cvt_f32_to_tf32(uint32_t a, uint32_t ctrl) {
  using namespace wgemu::misc;
  const wgemu::cvt::Rnd r = (ctrl & 3u) == 1u ? wgemu::cvt::RZ : wgemu::cvt::RN;
  return wgemu::cvt::f32_to_tf32(a, r, (ctrl & kTf32Relu) != 0, false);
}

__device__ __noinline__ uint32_t wgemu_cvt_to_bf16(uint64_t a, uint32_t ctrl) {
  using namespace wgemu::misc;
  namespace c = wgemu::cvt;
  const c::Rnd r = rnd_of(ctrl >> 4);
  const uint32_t kind = ctrl & 15u;
  uint32_t h;
  switch (kind) {
    case 0: {
      uint32_t x = (uint32_t)a;
      if (ctrl & kBf16Ftz) x = c::f32_ftz(x);
      h = c::f32_to_bf16(x, r);
      break;
    }
    case 1:
      h = c::f16_to_bf16((uint32_t)a & 0xFFFFu, r);
      break;
    case 2:
      h = c::f64_to_bf16(a, r);
      break;
    case 3:
    case 4:
    case 5:
    case 6:
      h = c::int_to_bf16(a, false, 8 << (kind - 3), r);
      break;
    case 7:
    case 8:
    case 9:
    case 10:
      h = c::int_to_bf16(a, true, 8 << (kind - 7), r);
      break;
    default:
      asm volatile("trap;");
      return 0;
  }
  if (ctrl & kBf16Sat) h = bf16_sat(h);
  return h & 0xFFFFu;
}

__device__ __noinline__ void wgemu_mma_fp8(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, uint32_t c0, uint32_t c1,
                                           uint32_t c2, uint32_t c3, uint64_t out, uint32_t cfg) {
  using namespace wgemu::misc;
  const uint32_t a[4] = {a0, a1, a2, a3}, b[4] = {b0, b1, 0u, 0u}, c[4] = {c0, c1, c2, c3};
  uint32_t* o = reinterpret_cast<uint32_t*>(out);
  if (cfg & kMmaK32) {
    mma_fp8_dense<32>(a, b, c, o, cfg);
  } else {
    mma_fp8_dense<16>(a, b, c, o, cfg);
  }
}

__device__ __noinline__ void wgemu_mma_sp_fp8(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
                                              uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                              uint32_t e, uint64_t out, uint32_t cfg) {
  using namespace wgemu;
  using namespace wgemu::misc;
  if (cfg & (kMmaDF16 | kMmaK32)) asm volatile("trap;");
  const uint32_t a[4] = {a0, a1, a2, a3}, b[4] = {b0, b1, b2, b3}, c[4] = {c0, c1, c2, c3};
  // A is the compressed 16x32 operand (dense m16n8k32 layout); B is the full 64x8 operand.
  uint32_t rows[2][32], cols[2][64], meta[2][16];
  gather_a8<32>(a, rows);
  gather_b8<64>(b, cols);
  gather_meta8(e, meta);
  const EType at = fp8_type(cfg & kMmaAE5m2), bt = fp8_type(cfg & kMmaBE5m2);
  const FdpaParams p = mma_params(mma_arch(cfg), at, bt, false);
  uint32_t* o = reinterpret_cast<uint32_t*>(out);
#pragma unroll
  for (int ri = 0; ri < 2; ++ri) {
    // Compressed column kc holds the element of chunk kc / 2 at the 2-bit index in bits
    // 2 * (kc % 2) of the chunk's metadata nibble ([inference]: products in kc order).
    uint32_t bsel[2][32];
#pragma unroll
    for (int kc = 0; kc < 32; ++kc) {
      const uint32_t idx = (meta[ri][kc / 2] >> (2 * (kc % 2))) & 3u;
      const int k = 4 * (kc / 2) + (int)idx;
#pragma unroll
      for (int ci = 0; ci < 2; ++ci) bsel[ci][kc] = cols[ci][k];
    }
#pragma unroll
    for (int ci = 0; ci < 2; ++ci) {
      o[2 * ri + ci] = dot_float(rows[ri], bsel[ci], 32, at, bt, c[2 * ri + ci], false, p);
    }
  }
}

__device__ __noinline__ void wgemu_mma_f64(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                           uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
                                           uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3,
                                           uint64_t c0, uint64_t c1, uint64_t c2, uint64_t c3,
                                           uint64_t out, uint32_t cfg) {
  using namespace wgemu::misc;
  const uint64_t a[8] = {a0, a1, a2, a3, a4, a5, a6, a7}, b[4] = {b0, b1, b2, b3};
  const uint64_t c[4] = {c0, c1, c2, c3};
  uint64_t* o = reinterpret_cast<uint64_t*>(out);
  const uint32_t rnd = (cfg >> 2) & 3u;
  switch (cfg & 3u) {
    case 0:
      mma_f64<4>(a, b, c, o, rnd);
      break;
    case 1:
      mma_f64<8>(a, b, c, o, rnd);
      break;
    case 2:
      mma_f64<16>(a, b, c, o, rnd);
      break;
    default:
      asm volatile("trap;");
  }
}

__device__ __noinline__ void wgemu_atom_b128(uint64_t addr, uint64_t locks, uint32_t cfg,
                                             uint64_t cmp_lo, uint64_t cmp_hi, uint64_t new_lo,
                                             uint64_t new_hi, uint64_t out) {
  using namespace wgemu::misc;
  if (addr & 15u) asm volatile("trap;");  // .b128 atom needs 16-byte alignment
  // Lock striping by 16-byte line. Every atom .b128 of the module goes through this table, so
  // the read-modify-write is atomic with respect to the other .b128 atoms ([inference]: mixing
  // with narrower atoms on the same bytes is not covered, as on hardware where atomicity is
  // only per operation of the same size).
  uint32_t* lk = reinterpret_cast<uint32_t*>(locks);
  const uint64_t line = addr >> 4;
  const uint32_t slot = (uint32_t)((line ^ (line >> 10) ^ (line >> 20)) % kB128LockCount);
  while (atomicCAS(&lk[slot], 0u, 1u) != 0u) __nanosleep(32);
  if (cfg & kB128Sys) {
    __threadfence_system();
  } else {
    __threadfence();
  }
  volatile uint64_t* p = reinterpret_cast<volatile uint64_t*>(addr);
  const uint64_t old_lo = p[0], old_hi = p[1];
  const bool store = !(cfg & kB128Cas) || (old_lo == cmp_lo && old_hi == cmp_hi);
  if (store) {
    p[0] = new_lo;
    p[1] = new_hi;
  }
  if (cfg & kB128Sys) {
    __threadfence_system();
  } else {
    __threadfence();
  }
  atomicExch(&lk[slot], 0u);
  uint64_t* o = reinterpret_cast<uint64_t*>(out);
  o[0] = old_lo;
  o[1] = old_hi;
}

}  // extern "C"
