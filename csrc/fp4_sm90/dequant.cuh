/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// In-register dequantization of one k16 step of the wgmma A operand (FP4 weights).
//
// Input word W8: the 8 E2M1 nibbles of one thread for one k16 step. Output j (0..3) takes nibble j
// (low element) and nibble j + 4 (high element); j -> A register:
//   j0 -> a[0] (row g,     k 2q, 2q+1)      j1 -> a[2] (row g,     k 2q+8, 2q+9)
//   j2 -> a[1] (row g + 8, k 2q, 2q+1)      j3 -> a[3] (row g + 8, k 2q+8, 2q+9)
// Scale pair: low byte = scale of row g, high byte = scale of row g + 8.
//
// Scale contract (T1 reference, plan F2 / G7 / G8):
//   E4M3   0x00-0x7E as float8_e4m3fn, 0x7F NaN; 0x80-0xFF signed like the FI reference
//          (0x80 = -0.0, 0xFF NaN). The sign never leaks into the exponent.
//   UE8M0  1-254 = 2^(e-127), 0 = 2^-127, 255 = NaN. Exact for every e (no exponent-field add).
//   S0E5M3 FP16 with bits (byte << 7): unsigned, E5M3 incl. subnormals, 0xF8 inf, 0xF9-0xFF NaN.
// The result is value * 2^-fold with fold = dq_fold_log2<...>() (power-of-two shift, exact); the
// epilogue multiplies the FP32 accumulators by 2^fold (EpiCfg::kAccScale, mainloop.cuh).
//
// Nibble placement: s e1 e0 m at bits 15:12 (and 31:28) moves to the sign, the two low exponent
// bits and the top mantissa bit: value = E2M1 * 2^-126 (BF16) or E2M1 * 2^-14 (FP16), exactly
// (e = 0 lands on the subnormal 0.m * 2^min). mul.rn.{bf16x2,f16x2} keep subnormals (no .ftz).
#pragma once

#include <cstdint>

#include "fp4_sm90/params.cuh"
#include "fp4_sm90/wgmma.cuh"

namespace flashinfer {
namespace fp4_sm90 {
namespace dq {

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// IEEE FP32 multiply with subnormals (independent of -ftz / -use_fast_math).
__device__ __forceinline__ float mul_f32(float a, float b) {
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}

// Exact E4M3 pair -> FP16 pair (sign and NaN kept).
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t v) {
  uint32_t d;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(uint16_t(v)));
  return d;
}

// (lo, hi) FP32 -> FP16 pair, round to nearest even.
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

constexpr uint32_t kBf16x2_2p126 = 0x7E807E80u;
constexpr uint32_t kBf16x2_2p116 = 0x79807980u;
constexpr uint32_t kBf16x2_2p115 = 0x79007900u;
constexpr uint32_t kBf16x2_2p112 = 0x77807780u;
constexpr uint32_t kBf16x2_2p111 = 0x77007700u;
constexpr uint32_t kF16x2_2p14 = 0x74007400u;
constexpr uint32_t kF16x2_2p7 = 0x58005800u;
constexpr uint32_t kF16x2_2pm6 = 0x24002400u;
constexpr uint32_t kF32_2p126 = 0x7E800000u;

// Nibbles at bits 15:12 and 31:28 -> E2M1 * 2^-126 (BF16) / E2M1 * 2^-14 (FP16) pair.
template <bool F16>
__device__ __forceinline__ uint32_t place2(uint32_t x) {
  if constexpr (F16) {
    return (x & 0x80008000u) | ((x >> 3) & 0x0E000E00u);
  } else {
    return (x & 0x80008000u) | ((x >> 6) & 0x01C001C0u);
  }
}

// Byte pair (lo, hi) of a 16-bit value -> one byte per 16-bit half.
__device__ __forceinline__ uint32_t spread2(uint32_t v16) { return ptx::prmt(v16, 0u, 0x4140u); }

// a[] = place(W8 nibbles) * scale pair, optionally pre-multiplied by `unit` (exact power of two).
// s: per 16-bit half the multiplier of row g (low half) and row g + 8 (high half).
template <bool F16, bool kUnit>
__device__ __forceinline__ void apply_pair(uint32_t w8, uint32_t s, uint32_t unit,
                                           uint32_t (&a)[4]) {
  const uint32_t s0 = ptx::prmt(s, 0u, 0x1010u);  // row g
  const uint32_t s1 = ptx::prmt(s, 0u, 0x3232u);  // row g + 8
  uint32_t v[4];
  v[0] = place2<F16>(w8 << 12);
  v[1] = place2<F16>(w8 << 8);
  v[2] = place2<F16>(w8 << 4);
  v[3] = place2<F16>(w8);
  if constexpr (kUnit) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = F16 ? mul_f16x2(v[j], unit) : mul_bf16x2(v[j], unit);
  }
  if constexpr (F16) {
    a[0] = mul_f16x2(v[0], s0);
    a[2] = mul_f16x2(v[1], s0);
    a[1] = mul_f16x2(v[2], s1);
    a[3] = mul_f16x2(v[3], s1);
  } else {
    a[0] = mul_bf16x2(v[0], s0);
    a[2] = mul_bf16x2(v[1], s0);
    a[1] = mul_bf16x2(v[2], s1);
    a[3] = mul_bf16x2(v[3], s1);
  }
}

// E4M3 pair -> BF16 pair of scale * 2^-112 with the scale's sign; NaN halves become 0xFFFF.
// FP16 bits >> 3 re-bias the exponent by 112 for every finite E4M3 (incl. subnormals); the mask
// drops the sign (and the other half's bits) that the shift moves into bits 12-15.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2_m112(uint32_t sf2) {
  const uint32_t z = cvt_e4m3x2_f16x2(sf2);
  const uint32_t nan = ptx::prmt((z & 0x7FFF7FFFu) + 0x04000400u, 0u, 0xBB99u);
  return ((z >> 3) & 0x0FFF0FFFu) | (z & 0x80008000u) | nan;
}

// UE8M0 pair -> BF16 pair 2^(e-127): e << 7 for 1..254; e = 0 -> 0x0040 (2^-127), e = 255 ->
// 0x7FC0 (NaN). Bit 6 is set exactly for e in {0, 255}.
__device__ __forceinline__ uint32_t ue8m0x2_to_bf16x2(uint32_t e8x2) {
  const uint32_t x = spread2(e8x2);
  const uint32_t t = (x + 0x00010001u) & 0x00FE00FEu;         // 0 iff e in {255, 0}
  const uint32_t u = ((t + 0x00FF00FFu) >> 2) & 0x00400040u;  // bit 6 iff t != 0
  return (x << 7) | (u ^ 0x00400040u);
}

// UE8M0 byte -> FP32 2^(e-127) (e = 0: subnormal 2^-127, e = 255: NaN).
__device__ __forceinline__ float ue8m0_to_f32(uint32_t e) {
  const uint32_t special = (((e + 1u) & 0xFEu) == 0u) ? 0x00400000u : 0u;
  return __uint_as_float((e << 23) | special);
}

// E2M1 nibble -> FP32 E2M1 * 2^-126.
__device__ __forceinline__ float place_f32(uint32_t n) {
  return __uint_as_float(((n & 8u) << 28) | ((n & 7u) << 22));
}

}  // namespace dq

// log2 of the factor the GEMM applies to its accumulators for each dequantization variant.
template <SfKind K, bool F16, int DQ>
__host__ __device__ constexpr int dq_fold_log2() {
  return K == SfKind::kE4M3 ? (DQ == 0 ? 7 : 0) : K == SfKind::kUE8M0 ? 0 : (F16 ? 6 : 15);
}

// NVFP4 (E4M3 scale per 16 values).
//   DQ 0 ("fold7"): a = E2M1 * scale * 2^-7, one multiply per register after the scale pair.
//     BF16: scale * 2^119 (448 * 2^119 < BF16 max); FP16: scale * 2^7 (448 * 2^7 < FP16 max).
//     Exact: every E2M1 x E4M3 product has <= 6 significant bits and its least significant bit
//     is >= 2^-10, so the 2^-7 copy is a normal BF16 or an FP16 >= 2^-17.
//   DQ 1 ("exact"): a = E2M1 * scale (two multiplies per register).
template <int DQ, bool F16>
__device__ __forceinline__ void dq_nvfp4(uint32_t w8, uint16_t sf2, uint32_t (&a)[4]) {
  static_assert(DQ == 0 || DQ == 1, "DQ is 0 (fold7) or 1 (exact)");
  if constexpr (F16) {
    const uint32_t z = dq::cvt_e4m3x2_f16x2(sf2);
    if constexpr (DQ == 0) {
      dq::apply_pair<true, false>(w8, dq::mul_f16x2(z, dq::kF16x2_2p7), 0u, a);
    } else {
      dq::apply_pair<true, true>(w8, z, dq::kF16x2_2p14, a);
    }
  } else {
    const uint32_t m = dq::e4m3x2_to_bf16x2_m112(sf2);
    if constexpr (DQ == 0) {
      const uint32_t s = dq::mul_bf16x2(dq::mul_bf16x2(m, dq::kBf16x2_2p116), dq::kBf16x2_2p115);
      dq::apply_pair<false, false>(w8, s, 0u, a);
    } else {
      dq::apply_pair<false, true>(w8, dq::mul_bf16x2(m, dq::kBf16x2_2p112), dq::kBf16x2_2p126, a);
    }
  }
}

// MXFP4 (UE8M0 scale per 32 values), exact (fold 0).
//   BF16: a = (place * 2^126) * 2^(e-127); the first product is E2M1 exactly, the second rounds
//         once (overflow to inf exactly when the FP32 product overflows BF16).
//   FP16: E2M1 * 2^(e-127) is formed in FP32 (exact, or inf) and rounded once to FP16, i.e. the
//         same value as the FP32 reference cast to FP16 (inf / subnormal / 0 outside FP16 range).
template <bool F16>
__device__ __forceinline__ void dq_mxfp4(uint32_t w8, uint16_t e8x2, uint32_t (&a)[4]) {
  if constexpr (F16) {
    const float s_lo = dq::ue8m0_to_f32(uint32_t(e8x2) & 0xFFu);
    const float s_hi = dq::ue8m0_to_f32(uint32_t(e8x2) >> 8);
    const float unit = __uint_as_float(dq::kF32_2p126);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float s = j < 2 ? s_lo : s_hi;
      const float lo = dq::mul_f32(dq::mul_f32(dq::place_f32((w8 >> (4 * j)) & 0xFu), unit), s);
      const float hi =
          dq::mul_f32(dq::mul_f32(dq::place_f32((w8 >> (4 * j + 16)) & 0xFu), unit), s);
      a[((j & 1) << 1) | (j >> 1)] = dq::pack_f16x2(lo, hi);  // j0 a0, j1 a2, j2 a1, j3 a3
    }
  } else {
    dq::apply_pair<false, true>(w8, dq::ue8m0x2_to_bf16x2(e8x2), dq::kBf16x2_2p126, a);
  }
}

// S0E5M3 (dense cute-dsl SM12x prep, scale per 16 values).
//   BF16 (fold 15): s = scale * 2^111 (61440 * 2^111 < BF16 max); byte << 4 is the BF16 of
//         scale * 2^-112 for every finite byte (subnormals included); 0xF8 -> inf, 0xF9+ -> NaN.
//   FP16 (fold 6): a = E2M1 * (scale * 2^-6); scale * 2^-6 is exact (>= 2^-23) and
//         6 * 61440 * 2^-6 < FP16 max; products are multiples of 2^-24.
template <bool F16>
__device__ __forceinline__ void dq_s0e5m3(uint32_t w8, uint16_t sf2, uint32_t (&a)[4]) {
  const uint32_t x = dq::spread2(sf2);
  if constexpr (F16) {
    dq::apply_pair<true, true>(w8, dq::mul_f16x2(x << 7, dq::kF16x2_2pm6), dq::kF16x2_2p14, a);
  } else {
    uint32_t s = dq::mul_bf16x2(dq::mul_bf16x2(x << 4, dq::kBf16x2_2p112), dq::kBf16x2_2p111);
    const uint32_t special = ptx::prmt((x + 0x00080008u) << 7, 0u, 0xBB99u);  // byte >= 0xF8
    const uint32_t inf_nan = 0x7F807F80u | ((x << 4) & 0x00700070u);
    s = (s & ~special) | (inf_nan & special);
    dq::apply_pair<false, false>(w8, s, 0u, a);
  }
}

// One k16 step for weight format AF: `sf2` is the scale pair of the step (UE8M0: of its 32-block).
template <AFmt AF, bool F16, int DQ = 0>
__device__ __forceinline__ void dq_step(uint32_t w8, uint32_t sf2, uint32_t (&a)[4]) {
  constexpr SfKind kKind = sf_kind(AF);
  if constexpr (kKind == SfKind::kE4M3) {
    dq_nvfp4<DQ, F16>(w8, uint16_t(sf2), a);
  } else if constexpr (kKind == SfKind::kUE8M0) {
    dq_mxfp4<F16>(w8, uint16_t(sf2), a);
  } else {
    dq_s0e5m3<F16>(w8, uint16_t(sf2), a);
  }
}

template <AFmt AF, bool F16, int DQ = 0>
__host__ __device__ constexpr int dq_fold_log2_fmt() {
  return dq_fold_log2<sf_kind(AF), F16, (sf_kind(AF) == SfKind::kE4M3 ? DQ : 0)>();
}

}  // namespace fp4_sm90
}  // namespace flashinfer
