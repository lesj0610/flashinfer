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
// Weight (A) side of the SM90 FP4 GEMM: kernel row -> native physical row, scale and code byte
// offsets of every AFmt, the A part of a pipeline stage, the producer loads and the consumer
// fragment reads.
//
// Kernel rows. Non-gated: kernel row = logical row q. Gated GEMM1: in each 16-row group b, row t
// (t < 8) is gate[i] and row t + 8 is up[i] with i = 8 * b + t, so one thread's rows g and g + 8
// of a wgmma fragment are the gate / up pair of the same output column.
//
// Stage A layout (kernel rows r in [0, CTA_M) of the tile starting at kernel row m0):
//   codes  CTA_M rows x 64 bytes (K 128), 128-byte swizzle (TMA CU_TENSOR_MAP_SWIZZLE_128B):
//          16-byte chunk j of tile row rho sits at (rho >> 1) * 128 + ((((rho & 1) << 2) | j) ^
//          ((rho >> 1) & 7)) << 4. Tile row rho = a_smem_row(r) is the TMA fill order (TRTLLM rows
//          are loaded in logical order through a 5D view of the row permutation).
//          kDenseCuteSm12x: 8 segments (k16 blocks) of CTA_M * 8 bytes, the packed int32 order.
//   scales CTA_M * 8 bytes. Row r's scales of this K tile at r * 8 (E4M3: 8 bytes, UE8M0: 4).
//          kDenseCuteSm12x: S0E5M3 [8][CTA_M] (k16 block, row), the checkpoint's transposed order.
//   (n2)   the producer re-stages the above into 64-row blocks of 4608 bytes (plan 3-3 layout):
//          [0, 4096) codes: warp w at w * 1024, plane p at + p * 512, lane l at + l * 16, word i
//          = W8 of k16 step 4p + i; [4096, 4608) scales: warp w at + w * 128, group g at + g * 16,
//          bytes 2s / 2s + 1 = scale of rows g / g + 8 for k16 step s (UE8M0: 32-block s).
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include "fp4_sm90/params.cuh"
#include "fp4_sm90/wgmma.cuh"
#include "tvm_ffi_utils.h"

#ifndef FP4_SM90_HD
#define FP4_SM90_HD __host__ __device__ __forceinline__
#endif

namespace flashinfer {
namespace fp4_sm90 {

// ---------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------
FP4_SM90_HD constexpr bool afmt_trtllm(AFmt f) {
  return f == AFmt::kNvfp4TrtllmShuffled || f == AFmt::kMxfp4TrtllmShuffled;
}

FP4_SM90_HD constexpr bool afmt_sf128x4(AFmt f) {
  return f == AFmt::kNvfp4TrtllmShuffled || f == AFmt::kNvfp4Swz128x4 ||
         f == AFmt::kMxfp4TrtllmShuffled || f == AFmt::kMxfp4Swz128x4;
}

// TRTLLM row permutation (physical row of logical row q) and its inverse.
FP4_SM90_HD int trtllm_perm(int q) { return (q / 32) * 32 + 8 * (q % 4) + (q % 32) / 4; }
FP4_SM90_HD int trtllm_perm_inv(int p) { return (p / 32) * 32 + 4 * (p % 8) + (p % 32) / 8; }

// Gated kernel row -> (h, i): h = 1 gate (rows 0-7 of a 16-row group), 0 up (rows 8-15).
FP4_SM90_HD int gated_half(int r) { return (r & 15) < 8 ? 1 : 0; }
FP4_SM90_HD int gated_index(int r) { return 8 * (r >> 4) + (r & 7); }

// Physical row of logical gated row (h, i) under `pair`.
FP4_SM90_HD int pair_phys_row(Pair pair, int half_rows, int pair_g, int h, int i) {
  switch (pair) {
    case Pair::kUpGateHalves:
      return h * half_rows + i;
    case Pair::kTrtllmW31:
      return trtllm_perm(2 * i + h);
    case Pair::kGroupInterleave:
      return (i / pair_g) * 2 * pair_g + h * pair_g + i % pair_g;
    default:
      return -1;
  }
}

// Physical row of absolute kernel row r, -1 when it has none (past the logical rows).
FP4_SM90_HD int a_phys_row(const ASrc& a, AFmt f, bool gated, int r) {
  if (r < 0 || r >= a.rows) return -1;
  int p;
  if (gated) {
    const int i = gated_index(r);
    if (i >= a.half_rows) return -1;
    p = pair_phys_row(a.pair, a.half_rows, a.pair_g, gated_half(r), i);
  } else {
    p = afmt_trtllm(f) ? trtllm_perm(r) : r;
  }
  return p < a.rows_phys ? p : -1;
}

// Tile row (0 .. cta_m) that holds kernel row r of a tile (TMA / fallback fill order).
FP4_SM90_HD int a_smem_row(Pair pair, int pair_g, bool gated, int cta_m, int r) {
  if (!gated) return r;
  const int h = gated_half(r), i = gated_index(r);
  switch (pair) {
    case Pair::kTrtllmW31:
      return 2 * i + h;  // logical order q = 2i + h
    case Pair::kUpGateHalves:
      return h * (cta_m / 2) + i;  // up box, then gate box
    case Pair::kGroupInterleave:
      return (i / pair_g) * 2 * pair_g + h * pair_g + i % pair_g;
    default:
      return r;
  }
}

// ---------------------------------------------------------------------------------------------
// Scale and code byte offsets (per expert; add e * sf_expert_stride / e * codes_expert_stride).
// ---------------------------------------------------------------------------------------------
// Physical scale columns (layout pitch) and the columns that hold data of logical K.
FP4_SM90_HD int a_sf_cols(const ASrc& a, AFmt f) {
  return (a.k_phys + sf_block(f) - 1) / sf_block(f);
}
FP4_SM90_HD int a_sf_cols_valid(const ASrc& a, AFmt f) {
  return (a.k + sf_block(f) - 1) / sf_block(f);
}

FP4_SM90_HD int64_t sf128x4_offset(int r, int c, int cols) {
  return int64_t(r / 128) * ((cols + 3) / 4) * 512 + int64_t(c / 4) * 512 + (r % 32) * 16 +
         ((r % 128) / 32) * 4 + c % 4;
}

FP4_SM90_HD int64_t sf128x4_expert_bytes(int rows, int cols) {
  return int64_t((rows + 127) / 128 * 128) * ((cols + 3) / 4 * 4);
}

FP4_SM90_HD int64_t a_sf_offset(const ASrc& a, AFmt f, int p, int c) {
  if (afmt_sf128x4(f)) return sf128x4_offset(p, c, a_sf_cols(a, f));
  if (f == AFmt::kDenseCuteSm12x) return int64_t(c) * a.rows_phys + p;  // (K / 16, N)
  return int64_t(p) * a_sf_cols(a, f) + c;                              // linear [N, K / 16]
}

// kDenseCuteSm12x code byte of weight row n, byte k2 (values 2 k2, 2 k2 + 1): inverse of
// `_cute_dsl_pack_fp4_weight` (16K x 64N blocks of 128 int32, row-major over (K / 16, N / 64)).
FP4_SM90_HD int64_t sm12x_code_offset(int n, int k2, int rows) {
  const int kt16 = k2 >> 3, kh = k2 & 7, nt = n >> 6, nn = n & 63;
  const int b = (kh >> 2) + 2 * ((nn >> 4) & 1);
  const int lane = 4 * (nn & 7) + (kh & 3);
  const int pos = 64 * ((nn & 15) >> 3) + 2 * lane + (nn >> 5);
  return int64_t(kt16) * (rows / 64) * 512 + int64_t(nt) * 512 + 4 * pos + b;
}

FP4_SM90_HD int64_t a_code_offset(const ASrc& a, AFmt f, int p, int k2) {
  if (f == AFmt::kDenseCuteSm12x) return sm12x_code_offset(p, k2, a.rows_phys);
  return int64_t(p) * (a.k_phys / 2) + k2;
}

// ---------------------------------------------------------------------------------------------
// Load modes (a function of ASrc only, so host and kernel agree).
// ---------------------------------------------------------------------------------------------
enum class ACodeLoad : uint8_t {
  kTma = 0,        // 3D map {row bytes, rows, expert}, one CTA_M-row box
  kTmaHalves = 1,  // Pair::kUpGateHalves: up map + gate map, two CTA_M / 2-row boxes
  kTmaTrtllm = 2,  // 5D view {bytes, q % 4, (q % 32) / 4, q / 32, expert} of the row permutation
  kCpAsync = 3,    // row bytes not a multiple of 16 or unaligned base: producer cp.async 8 B
  kSm12x = 4,      // packed int32 codes: producer cp.async 16 B
};

enum class ASfLoad : uint8_t {
  kCp4 = 0,    // producer cp.async 4 B (128x4 words, aligned linear rows, S0E5M3 rows)
  kBytes = 1,  // linear SF with a pitch not a multiple of 4: byte loads + st.shared + arrive
};

FP4_SM90_HD ACodeLoad a_code_load(const ASrc& a, AFmt f, bool gated) {
  if (f == AFmt::kDenseCuteSm12x) return ACodeLoad::kSm12x;
  const int64_t rb = a.k_phys / 2;
  const bool aligned = rb % 16 == 0 && (reinterpret_cast<uintptr_t>(a.codes) & 15) == 0 &&
                       a.codes_expert_stride % 16 == 0;
  if (!aligned) return ACodeLoad::kCpAsync;
  if (gated && a.pair == Pair::kUpGateHalves) return ACodeLoad::kTmaHalves;
  if (afmt_trtllm(f)) return ACodeLoad::kTmaTrtllm;
  return ACodeLoad::kTma;
}

FP4_SM90_HD ASfLoad a_sf_load(const ASrc& a, AFmt f) {
  if (f == AFmt::kNvfp4LinearSf &&
      (a_sf_cols(a, f) % 4 != 0 || (reinterpret_cast<uintptr_t>(a.sf) & 3) != 0 ||
       a.sf_expert_stride % 4 != 0)) {
    return ASfLoad::kBytes;
  }
  return ASfLoad::kCp4;
}

FP4_SM90_HD bool a_code_load_is_tma(ACodeLoad m) {
  return m == ACodeLoad::kTma || m == ACodeLoad::kTmaHalves || m == ACodeLoad::kTmaTrtllm;
}

// ---------------------------------------------------------------------------------------------
// Stage A layout
// ---------------------------------------------------------------------------------------------
template <class Cfg>
struct ASmem {
  static constexpr int kCodeBytes = Cfg::kCtaM * 64;
  static constexpr int kSfOff = kCodeBytes;
  static constexpr int kSfBytes = Cfg::kCtaM * 8;
  static constexpr int kNativeBytes = kCodeBytes + kSfBytes;
  static constexpr int kBlockBytes = 4608;  // (n2) 64 rows x K 128
  static constexpr int kPrepackOff = kNativeBytes;
  static constexpr int kPrepackBytes = Cfg::kCtaM / 64 * kBlockBytes;
  static constexpr int kBytes = kNativeBytes + (Cfg::kAPath == APath::kN2 ? kPrepackBytes : 0);
  static_assert(kNativeBytes % 1024 == 0 && kBytes % 1024 == 0, "stage A keeps 1024-byte steps");
};

// Byte offset of 16-byte chunk j of tile row rho (128-byte swizzle, tile base 1024-aligned).
FP4_SM90_HD uint32_t a_code_chunk(int rho, int j) {
  return uint32_t(rho >> 1) * 128u +
         ((uint32_t(((rho & 1) << 2) | j) ^ uint32_t((rho >> 1) & 7)) << 4);
}

// ---------------------------------------------------------------------------------------------
// Producer loads of one stage (expert e, kernel rows [m0, m0 + CTA_M), K tile kt)
// ---------------------------------------------------------------------------------------------
// TMA modes of a_code_load(): kTmaHalves when `halves`, otherwise kTmaTrtllm or kTma by format
// (only the runtime bit is passed so the producer keeps no mode register across its loop).
template <class Cfg>
__device__ __forceinline__ void a_issue_tma(bool halves, const CUtensorMap* map0,
                                            const CUtensorMap* map1, uint32_t dst, int e, int m0,
                                            int kt, uint32_t bar) {
  if (halves) {
    ptx::tma_3d(dst, map0, kt * 64, m0 / 2, e, bar);
    ptx::tma_3d(dst + uint32_t(Cfg::kCtaM / 2) * 64u, map1, kt * 64, m0 / 2, e, bar);
  } else if constexpr (afmt_trtllm(Cfg::kAFmt)) {
    ptx::tma_5d(dst, map0, kt * 64, 0, 0, m0 / 32, e, bar);
  } else {
    ptx::tma_3d(dst, map0, kt * 64, m0, e, bar);
  }
}

// Codes through cp.async (kCpAsync: 8-byte pieces into the swizzled tile; kSm12x: 16-byte
// chunks of the packed segments). Missing rows / K are zero-filled.
template <class Cfg>
__device__ __forceinline__ void a_codes_cp_async(const ASrc& a, uint32_t dst, int e, int m0, int kt,
                                                 int tid) {
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  if constexpr (Cfg::kAFmt == AFmt::kDenseCuteSm12x) {
    constexpr int kChunksPerSeg = Cfg::kCtaM / 2;  // CTA_M * 8 bytes / 16
    const int row_blocks = a.rows_phys / 64, kt16_end = a.k_phys / 16;
#pragma unroll 1
    for (int it = tid; it < 8 * kChunksPerSeg; it += 128) {
      const int s = it / kChunksPerSeg, c = it % kChunksPerSeg;
      const int kt16 = kt * 8 + s, nt = m0 / 64 + c / 32;
      const bool ok = kt16 < kt16_end && nt < row_blocks;
      const uint8_t* src =
          ok ? a.codes + (int64_t(kt16) * row_blocks + m0 / 64) * 512 + 16 * c : a.codes;
      ptx::cp_async_16(dst + uint32_t(s * Cfg::kCtaM * 8 + 16 * c), src, ok ? 16u : 0u);
    }
    return;
  }
  const int64_t rb = a.k_phys / 2;
  const uint8_t* base = a.codes + int64_t(e) * a.codes_expert_stride;
#pragma unroll 1
  for (int it = tid; it < Cfg::kCtaM * 8; it += 128) {
    const int r = it >> 3, h8 = it & 7;
    const int p = a_phys_row(a, Cfg::kAFmt, kGated, m0 + r);
    const int64_t k2 = int64_t(kt) * 64 + 8 * h8;
    const bool ok = p >= 0 && k2 < rb;
    const int rho = a_smem_row(a.pair, a.pair_g, kGated, Cfg::kCtaM, r);
    ptx::cp_async_8(dst + a_code_chunk(rho, h8 >> 1) + uint32_t(h8 & 1) * 8u,
                    ok ? base + int64_t(p) * rb + k2 : a.codes, ok ? 8u : 0u);
  }
}

// Scales of the stage in kernel-row order (or [8][CTA_M] for S0E5M3). kBytes mode returns after
// its st.shared; the caller then arrives on the stage barrier.
template <class Cfg>
__device__ __forceinline__ void a_sf_load_stage(const ASrc& a, ASfLoad mode, uint32_t dst, int e,
                                                int m0, int kt, int tid) {
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  constexpr AFmt kF = Cfg::kAFmt;
  const int cols_valid = a_sf_cols_valid(a, kF);
  if constexpr (kF == AFmt::kDenseCuteSm12x) {
    constexpr int kWordsPerSeg = Cfg::kCtaM / 4;
#pragma unroll 1
    for (int it = tid; it < 8 * kWordsPerSeg; it += 128) {
      const int s = it / kWordsPerSeg, wd = it % kWordsPerSeg;
      const int c = kt * 8 + s, r = m0 + 4 * wd;
      // words past a.rows read as scale 0, like a_phys_row() < 0 in the other formats (a word
      // that starts below a.rows is loaded whole; its padding rows are never stored)
      const bool ok = c < cols_valid && r < a.rows;
      ptx::cp_async_4(dst + uint32_t(s * Cfg::kCtaM + 4 * wd),
                      ok ? a.sf + int64_t(c) * a.rows_phys + r : a.sf, ok ? 4u : 0u);
    }
    return;
  }
  constexpr int kWords = sf_block(kF) == 32 ? 1 : 2;  // 4-byte words per row and K tile
  constexpr int kColsPerTile = 128 / sf_block(kF);
  const uint8_t* base = a.sf + int64_t(e) * a.sf_expert_stride;
#pragma unroll 1
  for (int it = tid; it < Cfg::kCtaM * kWords; it += 128) {
    const int r = it / kWords, j = it % kWords;
    const int p = a_phys_row(a, kF, kGated, m0 + r);
    const int c0 = kt * kColsPerTile + 4 * j;
    int n = p >= 0 ? cols_valid - c0 : 0;
    n = n < 0 ? 0 : (n > 4 ? 4 : n);
    const uint32_t d = dst + uint32_t(r * 8 + 4 * j);
    if (mode == ASfLoad::kCp4) {
      ptx::cp_async_4(d, n > 0 ? base + a_sf_offset(a, kF, p, c0) : a.sf, uint32_t(n));
    } else {
      uint32_t word = 0;
      for (int b = 0; b < n; ++b) {
        word |= uint32_t(__ldg(base + a_sf_offset(a, kF, p, c0 + b))) << (8 * b);
      }
      ptx::sts32(d, word);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Consumer reads
// ---------------------------------------------------------------------------------------------
// W8 of k16 step 2 * (chunk index) + (i / 2) from 16-byte chunks of rows g (c0) and g + 8 (c1):
// byte q of words i, i + 1 of both rows, then nibbles 1 <-> 4 and 3 <-> 6 swapped.
FP4_SM90_HD uint32_t w8_from_bytes(uint32_t p) {
  const uint32_t t = ((p >> 12) ^ p) & 0x0000F0F0u;
  return p ^ t ^ (t << 12);
}

FP4_SM90_HD uint32_t sel_q(int q) {
  const uint32_t u = uint32_t(q & 3);
  return u | ((u + 4) << 4) | (u << 8) | ((u + 4) << 12);
}

__device__ __forceinline__ uint32_t u4_word(const uint4& v, int i) {
  return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

__device__ __forceinline__ uint32_t w8_from_rows(const uint4& c0, const uint4& c1, int i,
                                                 uint32_t selq) {
  const uint32_t x = ptx::prmt(u4_word(c0, i), u4_word(c1, i), selq);          // [B0 B1 . .]
  const uint32_t y = ptx::prmt(u4_word(c0, i + 1), u4_word(c1, i + 1), selq);  // [B2 B3 . .]
  return w8_from_bytes(ptx::prmt(x, y, 0x5410u));
}

// kDenseCuteSm12x: W8 of step s from the packed segment (byte offset x of this thread, see
// a_frag_init): bytes (row g, q) | (row g, q + 4) at x, the row g + 8 pair at x + 256.
__device__ __forceinline__ uint32_t w8_sm12x(uint32_t seg, uint32_t x) {
  return w8_from_bytes(ptx::prmt(ptx::lds_u16(seg + x), ptx::lds_u16(seg + x + 256u), 0x5140u));
}

// Scale pairs of one K tile: sf[k] holds the 16-bit pairs (row g byte, row g + 8 byte) of
// index 2k (low half) and 2k + 1; the index is the k16 step (E4M3, S0E5M3) or 32-block (UE8M0).
template <AFmt AF>
__device__ __forceinline__ uint32_t sf_pair(const uint32_t (&sf)[4], int s) {
  const int idx = sf_kind(AF) == SfKind::kUE8M0 ? (s >> 1) : s;
  return (sf[idx >> 1] >> (16 * (idx & 1))) & 0xFFFFu;
}

// Per-thread offsets for one m64 block (blk = consumer warpgroup * CTA_M / 128 + mi), warp w.
struct AFragRows {
  uint32_t c0, c1;    // native: tile row base of rows g / g + 8; sm12x: packed byte; (n2): lane
  uint32_t x0, x1;    // native: swizzle term of the two rows
  uint32_t sf0, sf1;  // scale byte offsets of the two rows (or (n2) group offset in sf0)
};

template <class Cfg>
__device__ __forceinline__ AFragRows a_frag_init(Pair pair, int pair_g, int blk, int w, int lane) {
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  const int g = lane >> 2, q = lane & 3;
  const int r0 = 64 * blk + 16 * w + g, r1 = r0 + 8;
  AFragRows f;
  if constexpr (Cfg::kAPath == APath::kN2) {
    const uint32_t b = uint32_t(ASmem<Cfg>::kPrepackOff + blk * ASmem<Cfg>::kBlockBytes);
    f.c0 = b + uint32_t(w * 1024 + lane * 16);
    f.c1 = f.c0 + 512u;
    f.x0 = f.x1 = 0;
    f.sf0 = b + uint32_t(4096 + w * 128 + g * 16);
    f.sf1 = 0;
  } else if constexpr (Cfg::kAFmt == AFmt::kDenseCuteSm12x) {
    f.c0 = uint32_t(blk * 512 + 32 * g + 8 * q + 4 * (w >> 1) + 2 * (w & 1));
    f.c1 = 0;
    f.x0 = f.x1 = 0;
    f.sf0 = uint32_t(ASmem<Cfg>::kSfOff + r0);
    f.sf1 = uint32_t(ASmem<Cfg>::kSfOff + r1);
  } else {
    const int rho0 = a_smem_row(pair, pair_g, kGated, Cfg::kCtaM, r0);
    const int rho1 = a_smem_row(pair, pair_g, kGated, Cfg::kCtaM, r1);
    f.c0 = uint32_t(rho0 >> 1) * 128u;
    f.c1 = uint32_t(rho1 >> 1) * 128u;
    f.x0 = uint32_t(((rho0 & 1) << 2) ^ ((rho0 >> 1) & 7));
    f.x1 = uint32_t(((rho1 & 1) << 2) ^ ((rho1 >> 1) & 7));
    f.sf0 = uint32_t(ASmem<Cfg>::kSfOff + r0 * 8);
    f.sf1 = uint32_t(ASmem<Cfg>::kSfOff + r1 * 8);
  }
  return f;
}

// Scale words of a K tile for one m64 block (stage base `st`).
template <class Cfg>
__device__ __forceinline__ void a_load_sf(uint32_t st, const AFragRows& f, uint32_t (&sf)[4]) {
  constexpr AFmt kF = Cfg::kAFmt;
  if constexpr (Cfg::kAPath == APath::kN2) {
    const uint4 v = ptx::lds128(st + f.sf0);
    sf[0] = v.x;
    sf[1] = v.y;
    sf[2] = v.z;
    sf[3] = v.w;
  } else if constexpr (kF == AFmt::kDenseCuteSm12x) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const uint32_t s0 = st + uint32_t(2 * k * Cfg::kCtaM), s1 = s0 + uint32_t(Cfg::kCtaM);
      sf[k] = ptx::lds_u8(s0 + f.sf0) | (ptx::lds_u8(s0 + f.sf1) << 8) |
              (ptx::lds_u8(s1 + f.sf0) << 16) | (ptx::lds_u8(s1 + f.sf1) << 24);
    }
  } else if constexpr (sf_kind(kF) == SfKind::kUE8M0) {
    const uint32_t u = ptx::lds32(st + f.sf0), v = ptx::lds32(st + f.sf1);
    sf[0] = ptx::prmt(u, v, 0x5140u);
    sf[1] = ptx::prmt(u, v, 0x7362u);
    sf[2] = sf[3] = 0;
  } else {
    const uint2 u = ptx::lds64(st + f.sf0), v = ptx::lds64(st + f.sf1);
    sf[0] = ptx::prmt(u.x, v.x, 0x5140u);
    sf[1] = ptx::prmt(u.x, v.x, 0x7362u);
    sf[2] = ptx::prmt(u.y, v.y, 0x5140u);
    sf[3] = ptx::prmt(u.y, v.y, 0x7362u);
  }
}

// Native code chunk j (steps 2j, 2j + 1) of row g (r = 0) or g + 8 (r = 1): stage byte offset.
FP4_SM90_HD uint32_t a_chunk_offset(const AFragRows& f, int r, int j) {
  return r == 0 ? f.c0 + ((f.x0 ^ uint32_t(j)) << 4) : f.c1 + ((f.x1 ^ uint32_t(j)) << 4);
}

__device__ __forceinline__ void a_load_chunk(uint32_t st, const AFragRows& f, int j, uint4& c0,
                                             uint4& c1) {
  c0 = ptx::lds128(st + a_chunk_offset(f, 0, j));
  c1 = ptx::lds128(st + a_chunk_offset(f, 1, j));
}

// W8 of k16 step s of one m64 block read without the mainloop's prefetching (tests, same words).
template <class Cfg>
__device__ __forceinline__ uint32_t a_read_w8(uint32_t st, const AFragRows& f, int s,
                                              uint32_t selq) {
  if constexpr (Cfg::kAPath == APath::kN2) {
    const uint4 v = ptx::lds128(st + (s < 4 ? f.c0 : f.c1));
    return u4_word(v, s & 3);
  } else if constexpr (Cfg::kAFmt == AFmt::kDenseCuteSm12x) {
    return w8_sm12x(st + uint32_t(s * Cfg::kCtaM * 8), f.c0);
  } else {
    uint4 c0, c1;
    a_load_chunk(st, f, s >> 1, c0, c1);
    return w8_from_rows(c0, c1, (s & 1) * 2, selq);
  }
}

// ---------------------------------------------------------------------------------------------
// (n2) producer re-staging of one stage: native (stage base st) -> 64-row blocks at kPrepackOff.
// Item (blk, w, g, j) writes steps 2j, 2j + 1 of lanes 4g .. 4g + 3 and their scale pairs.
// ---------------------------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void a_prepack(uint32_t st, Pair pair, int pair_g, int tid) {
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  constexpr AFmt kF = Cfg::kAFmt;
  constexpr int kItems = Cfg::kCtaM / 64 * 128;
  const uint32_t sf_native = st + uint32_t(ASmem<Cfg>::kSfOff);
#pragma unroll 1
  for (int it = tid; it < kItems; it += 128) {
    const int j = it & 3, g = (it >> 2) & 7, w = (it >> 5) & 3, blk = it >> 7;
    const int r0 = 64 * blk + 16 * w + g, r1 = r0 + 8;
    const uint32_t ob = st + uint32_t(ASmem<Cfg>::kPrepackOff + blk * ASmem<Cfg>::kBlockBytes);
    const uint32_t code_dst = ob + uint32_t(w * 1024 + (j >> 1) * 512 + (j & 1) * 8);
    if constexpr (kF == AFmt::kDenseCuteSm12x) {
      const uint32_t seg0 = st + uint32_t(2 * j * Cfg::kCtaM * 8);
      const uint32_t seg1 = seg0 + uint32_t(Cfg::kCtaM * 8);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t x = uint32_t(blk * 512 + 32 * g + 8 * q + 4 * (w >> 1) + 2 * (w & 1));
        ptx::sts64(code_dst + uint32_t((4 * g + q) * 16), w8_sm12x(seg0, x), w8_sm12x(seg1, x));
      }
    } else {
      const int rho0 = a_smem_row(pair, pair_g, kGated, Cfg::kCtaM, r0);
      const int rho1 = a_smem_row(pair, pair_g, kGated, Cfg::kCtaM, r1);
      const uint4 c0 = ptx::lds128(st + a_code_chunk(rho0, j));
      const uint4 c1 = ptx::lds128(st + a_code_chunk(rho1, j));
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t sq = sel_q(q);
        ptx::sts64(code_dst + uint32_t((4 * g + q) * 16), w8_from_rows(c0, c1, 0, sq),
                   w8_from_rows(c0, c1, 2, sq));
      }
    }
    const uint32_t sf_dst = ob + uint32_t(4096 + w * 128 + g * 16);
    if constexpr (sf_kind(kF) == SfKind::kUE8M0) {
      ptx::sts16(sf_dst + uint32_t(2 * j),
                 ptx::lds_u8(sf_native + uint32_t(r0 * 8 + j)) |
                     (ptx::lds_u8(sf_native + uint32_t(r1 * 8 + j)) << 8));
    } else if constexpr (kF == AFmt::kDenseCuteSm12x) {
      const uint32_t s0 = sf_native + uint32_t(2 * j * Cfg::kCtaM), s1 = s0 + uint32_t(Cfg::kCtaM);
      ptx::sts32(sf_dst + uint32_t(4 * j), ptx::lds_u8(s0 + uint32_t(r0)) |
                                               (ptx::lds_u8(s0 + uint32_t(r1)) << 8) |
                                               (ptx::lds_u8(s1 + uint32_t(r0)) << 16) |
                                               (ptx::lds_u8(s1 + uint32_t(r1)) << 24));
    } else {
      const uint32_t u = ptx::lds_u16(sf_native + uint32_t(r0 * 8 + 2 * j));
      const uint32_t v = ptx::lds_u16(sf_native + uint32_t(r1 * 8 + 2 * j));
      ptx::sts32(sf_dst + uint32_t(4 * j), ptx::prmt(u, v, 0x5140u));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Host: tensor maps of the A codes (map0, map1; zeroed when unused).
// ---------------------------------------------------------------------------------------------
inline PFN_cuTensorMapEncodeTiled_v12000 fp4_sm90_encode_tiled() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault,
                                     &status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &status);
#endif
    TVM_FFI_CHECK(status == cudaDriverEntryPointSuccess && ptr != nullptr, RuntimeError)
        << "cuTensorMapEncodeTiled is unavailable from the CUDA driver";
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

inline void encode_a_map(CUtensorMap* map, const void* base, uint32_t rank, const cuuint64_t* dims,
                         const cuuint64_t* strides, const cuuint32_t* box, const char* what) {
  const cuuint32_t elem[5] = {1, 1, 1, 1, 1};
  const CUresult r = fp4_sm90_encode_tiled()(
      map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides, box, elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled failed for the SM90 FP4 " << what << " map, CUresult "
      << static_cast<int>(r);
}

inline void make_a_tmaps(const ASrc& a, AFmt f, bool gated, int cta_m, CUtensorMap* map0,
                         CUtensorMap* map1) {
  std::memset(map0, 0, sizeof(CUtensorMap));
  std::memset(map1, 0, sizeof(CUtensorMap));
  const ACodeLoad mode = a_code_load(a, f, gated);
  if (!a_code_load_is_tma(mode)) return;
  const cuuint64_t rb = cuuint64_t(a.k_phys / 2);
  const bool moe = a.codes_expert_stride != 0;
  const cuuint64_t experts = moe ? cuuint64_t(a.num_experts) : 1;
  const cuuint64_t estride = moe ? cuuint64_t(a.codes_expert_stride) : rb * cuuint64_t(a.rows_phys);
  if (mode == ACodeLoad::kTma) {
    const cuuint64_t dims[3] = {rb, cuuint64_t(a.rows_phys), experts};
    const cuuint64_t strides[2] = {rb, estride};
    const cuuint32_t box[3] = {64, cuuint32_t(cta_m), 1};
    encode_a_map(map0, a.codes, 3, dims, strides, box, "weight code");
  } else if (mode == ACodeLoad::kTmaHalves) {
    const cuuint64_t dims[3] = {rb, cuuint64_t(a.half_rows), experts};
    const cuuint64_t strides[2] = {rb, estride};
    const cuuint32_t box[3] = {64, cuuint32_t(cta_m / 2), 1};
    encode_a_map(map0, a.codes, 3, dims, strides, box, "weight code (up half)");
    encode_a_map(map1, a.codes + int64_t(a.half_rows) * int64_t(rb), 3, dims, strides, box,
                 "weight code (gate half)");
  } else {
    // Box element (b, c, a) is physical row 8b + c + 32a = P(q) for q = 32a + 4c + b, so the
    // box lands in shared memory in logical row order.
    const cuuint64_t dims[5] = {rb, 4, 8, cuuint64_t(a.rows_phys / 32), experts};
    const cuuint64_t strides[4] = {8 * rb, rb, 32 * rb, estride};
    const cuuint32_t box[5] = {64, 4, 8, cuuint32_t(cta_m / 32), 1};
    encode_a_map(map0, a.codes, 5, dims, strides, box, "weight code (TRTLLM rows)");
  }
}

}  // namespace fp4_sm90
}  // namespace flashinfer
