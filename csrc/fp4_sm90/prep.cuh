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
// Prep of the SM90 FP4 GEMM (PrepArgs in params.cuh): routing sort, n-tile list and the GEMM1
// B tiles.
//
// Expanded rows p are sorted by local expert and, inside an expert, by token * top_k + slot, so
// the order is deterministic. The B tiles hold BF16/FP16 values: the input as is, or for NVFP4
// (quantized here with the recipe, or given as FP4) the exact dequantized value code * E4M3 SF.
// An MXFP4 input gives code * 2^(e-127) in BF16 (exact; inf past the BF16 range).
// The global scale stays out of the tiles (the epilogue alpha holds it), the per-token scale goes
// to tok_scale. Quantization calls the FI quantizer (cvt_warp_fp16_to_fp4) and gives the bits of
// fp4_quantize / nvfp4_quantize(per_token_activation=True) of the same recipe.
//
// x_global:
//   per_token = 0, quantize = 1: the global (encode) scale of fp4_quantize, one or one per
//                                local expert (x_global_per_expert); null = 1.
//   per_token = 1, quantize = 1: the inverse global scale of the per-token recipe
//                                (a_global_sf of nvfp4_quantize); null = 1 / (E4M3_MAX * 6).
// An FP4 input keeps its per-token scale outside prep (per_token needs quantize).
//
// Dense with num_tiles null: the GEMM walks max_tiles n-tiles, so prep writes all of them (zero
// past the tokens).
//
// Build: FLASHINFER_FP4_SW_CVT and ENABLE_BF16/FP8/FP4. The quantized bits are those of the "90"
// quantization module when built as it is (-use_fast_math, no FLASHINFER_FP4_SW_CVT_FFMA): the
// fast-math products of the recipes flush subnormals. FP4_SM90_RECIPE (kRecipe* bits) and
// FLASHINFER_FP4_SM90_EXACT (0/1), when defined, compile one quantizer instead of all of them;
// otherwise the recipe is dispatched at run time and FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH is
// read at launch, as in the SM100 quantizers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <type_traits>

#include "flashinfer/exception.h"
#include "moe_layout.h"
#include "params.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

#if !defined(FLASHINFER_FP4_SW_CVT)
#error \
    "fp4_sm90 prep needs FLASHINFER_FP4_SW_CVT (cvt_warp_fp16_to_fp4 returns 0 below SM100 without it)"
#endif

namespace flashinfer {
namespace fp4_sm90 {

namespace prep_detail {

namespace tk = tensorrt_llm::kernels;

constexpr int kSortThreads = 256;
constexpr int kSortWarps = kSortThreads / 32;
constexpr int kChunk = moe_layout::kPrepChunk;  // routing entries per sort CTA
constexpr int kScanThreads = 1024;
constexpr int kTileThreads = 128;  // one B row per CTA iteration
constexpr int kMaxTileCtas = 4096;

enum class TileMode : int { kCopy = 0, kQuant = 1, kFp4 = 2 };

struct TileParams {
  int dense_tiles;       // dense: n-tiles of the B rows written (no tile list is read)
  float gs_inv_default;  // per-token inverse global scale of the recipe (x_global null)
};

__device__ __forceinline__ int local_expert(const PrepArgs& a, int flat) {
  if (a.topk_ids == nullptr) return 0;
  int const e = a.topk_ids[flat] - a.local_expert_offset;
  return (e >= 0 && e < a.num_local_experts) ? e : -1;
}

// Per chunk of kChunk entries: rows per local expert -> workspace[chunk][expert].
__global__ void __launch_bounds__(kSortThreads) prep_count(PrepArgs a, int total) {
  extern __shared__ int hist[];
  int const E = a.num_local_experts;
  for (int e = threadIdx.x; e < E; e += kSortThreads) hist[e] = 0;
  __syncthreads();
  int const end = min(total, (blockIdx.x + 1) * kChunk);
  for (int i = blockIdx.x * kChunk + threadIdx.x; i < end; i += kSortThreads) {
    int const e = local_expert(a, i);
    if (e >= 0) atomicAdd(&hist[e], 1);
  }
  __syncthreads();
  int* const counts = static_cast<int*>(a.workspace) + int64_t(blockIdx.x) * E;
  for (int e = threadIdx.x; e < E; e += kSortThreads) counts[e] = hist[e];
}

// One CTA: expert_offset, expert_tile_base, num_tiles, the tile list, and the chunk counts turned
// into the first expanded row of each (chunk, expert). Dense: one expert of `total` rows.
__global__ void __launch_bounds__(kScanThreads) prep_scan(PrepArgs a, int total, int num_chunks) {
  using Scan = cub::BlockScan<int, kScanThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ int carry[2];
  bool const dense = a.topk_ids == nullptr;
  int* const counts = static_cast<int*>(a.workspace);
  int const E = a.num_local_experts, n = a.n;
  if (threadIdx.x == 0) carry[0] = carry[1] = 0;
  __syncthreads();
  for (int e0 = 0; e0 < E; e0 += kScanThreads) {
    int const e = e0 + threadIdx.x;
    int cnt = 0;
    if (e < E) {
      if (dense) {
        cnt = total;
      } else {
        for (int c = 0; c < num_chunks; ++c) cnt += counts[int64_t(c) * E + e];
      }
    }
    int row0, rows, tile0, tiles;
    Scan(scan_tmp).ExclusiveSum(cnt, row0, rows);
    __syncthreads();
    Scan(scan_tmp).ExclusiveSum((cnt + n - 1) / n, tile0, tiles);
    row0 += carry[0];
    tile0 += carry[1];
    if (e < E) {
      if (a.expert_offset) a.expert_offset[e] = row0;
      if (a.expert_tile_base) a.expert_tile_base[e] = tile0;
      if (!dense) {
        for (int c = 0, run = row0; c < num_chunks; ++c) {
          int64_t const i = int64_t(c) * E + e;
          int const v = counts[i];
          counts[i] = run;
          run += v;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry[0] += rows;
      carry[1] += tiles;
    }
    __syncthreads();
  }
  int const num_tiles = carry[1];
  if (threadIdx.x == 0) {
    if (a.expert_offset) a.expert_offset[E] = carry[0];
    if (a.expert_tile_base) a.expert_tile_base[E] = num_tiles;
    if (a.num_tiles) *a.num_tiles = num_tiles;
  }
  if (dense) {
    for (int i = threadIdx.x; i < total; i += kScanThreads) {
      if (a.permuted_src) a.permuted_src[i] = i;
      if (a.src_permuted) a.src_permuted[i] = i;
    }
  }
  __syncthreads();  // expert_offset / expert_tile_base above are read back below
  for (int t = threadIdx.x; t < num_tiles && t < a.max_tiles; t += kScanThreads) {
    int e = 0;
    if (!dense) {  // last expert with expert_tile_base[e] <= t
      int hi = E - 1;
      while (e < hi) {
        int const mid = (e + hi + 1) / 2;
        if (a.expert_tile_base[mid] <= t) {
          e = mid;
        } else {
          hi = mid - 1;
        }
      }
    }
    int const base = dense ? 0 : a.expert_tile_base[e];
    int const row0 = dense ? 0 : a.expert_offset[e];
    if (a.tile_expert) a.tile_expert[t] = e;
    if (a.tile_n0) a.tile_n0[t] = row0 + (t - base) * n;
  }
}

// GEMM2's n-tile list when its n (a.n2) differs from n: the same experts and rows (expert_offset
// from prep_scan), tiles of n2 rows. One CTA, as prep_scan.
__global__ void __launch_bounds__(kScanThreads) prep_scan2(PrepArgs a) {
  using Scan = cub::BlockScan<int, kScanThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ int carry;
  int const E = a.num_local_experts, n2 = a.n2;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int e0 = 0; e0 < E; e0 += kScanThreads) {
    int const e = e0 + threadIdx.x;
    int const cnt = e < E ? a.expert_offset[e + 1] - a.expert_offset[e] : 0;
    int tile0, tiles;
    Scan(scan_tmp).ExclusiveSum((cnt + n2 - 1) / n2, tile0, tiles);
    if (e < E) a.expert_tile_base2[e] = tile0 + carry;
    __syncthreads();
    if (threadIdx.x == 0) carry += tiles;
    __syncthreads();
  }
  int const num_tiles = carry;
  if (threadIdx.x == 0) {
    a.expert_tile_base2[E] = num_tiles;
    *a.num_tiles2 = num_tiles;
  }
  __syncthreads();  // expert_tile_base2 is read back below
  for (int t = threadIdx.x; t < num_tiles && t < a.max_tiles2; t += kScanThreads) {
    int e = 0;
    int hi = E - 1;
    while (e < hi) {  // last expert with expert_tile_base2[e] <= t
      int const mid = (e + hi + 1) / 2;
      if (a.expert_tile_base2[mid] <= t) {
        e = mid;
      } else {
        hi = mid - 1;
      }
    }
    a.tile_expert2[t] = e;
    a.tile_n02[t] = a.expert_offset[e] + (t - a.expert_tile_base2[e]) * n2;
  }
}

// Per tensor, n2 != n: GEMM1 writes GEMM2's B tiles (act_tiles) row by row and only for the rows
// it has. This kernel is the only writer of the others: the rows past an expert's count in its
// last n2-tile, over every K tile and half. (GEMM1 zeroes the K padding of the rows it writes.)
__global__ void __launch_bounds__(kSortThreads) prep_pad(PrepArgs a) {
  int const n2 = a.n2;
  int const per_row = ((a.act_k + 127) / 128) * 2 * 8;  // 16-byte chunks of a padded row
  char* const tiles = static_cast<char*>(a.act_tiles);
  for (int e = blockIdx.x; e < a.num_local_experts; e += gridDim.x) {
    int const cnt = a.expert_offset[e + 1] - a.expert_offset[e];
    int const rem = cnt % n2;
    if (rem == 0) continue;
    int const t = a.expert_tile_base2[e] + cnt / n2;  // the partly filled tile
    int const k_tiles = per_row / 16;
    for (int i = threadIdx.x; i < (n2 - rem) * per_row; i += kSortThreads) {
      int const r = rem + i / per_row, j = i % per_row;
      int const kt = j >> 4, kh = (j >> 3) & 1, c = (j & 7) * 8;
      *reinterpret_cast<uint4*>(tiles + b_tile_byte(t, k_tiles, kt, kh, n2, r, c)) =
          make_uint4(0, 0, 0, 0);
    }
  }
}

// Stable scatter: entries of a chunk in flat order, a round of kSortThreads at a time; the rank
// inside a round comes from the warp match and the counts of the warps before.
__global__ void __launch_bounds__(kSortThreads) prep_scatter(PrepArgs a, int total) {
  extern __shared__ int smem[];
  int const E = a.num_local_experts;
  int* const run = smem;           // [E] next expanded row of each expert
  int* const warp_cnt = smem + E;  // [kSortWarps][E]
  int const* const bases = static_cast<int const*>(a.workspace) + int64_t(blockIdx.x) * E;
  for (int e = threadIdx.x; e < E; e += kSortThreads) run[e] = bases[e];
  int const warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int round0 = blockIdx.x * kChunk; round0 < min(total, (blockIdx.x + 1) * kChunk);
       round0 += kSortThreads) {
    for (int i = threadIdx.x; i < kSortWarps * E; i += kSortThreads) warp_cnt[i] = 0;
    __syncthreads();
    int const flat = round0 + threadIdx.x;
    int const e = flat < total ? local_expert(a, flat) : -1;
    unsigned const peers = __match_any_sync(0xffffffffu, e);
    int const rank = __popc(peers & ((1u << lane) - 1u));
    if (e >= 0 && rank == 0) warp_cnt[warp * E + e] = __popc(peers);
    __syncthreads();
    int p = -1;
    if (e >= 0) {
      p = run[e] + rank;
      for (int w = 0; w < warp; ++w) p += warp_cnt[w * E + e];
      a.permuted_src[p] = flat;
    }
    if (flat < total) a.src_permuted[flat] = p;
    __syncthreads();
    for (int ee = threadIdx.x; ee < E; ee += kSortThreads) {
      int s = 0;
      for (int w = 0; w < kSortWarps; ++w) s += warp_cnt[w * E + ee];
      run[ee] += s;
    }
    __syncthreads();
  }
}

// B tile store of 16 values (chunk j along the padded K): two 16-byte rows of the SW128 layout.
__device__ __forceinline__ void store16(char* tiles, int t, int k_tiles, int n, int r, int j,
                                        uint4 lo, uint4 hi) {
  int const kt = j >> 3, kh = (j >> 2) & 1, c = (j & 3) * 16;
  *reinterpret_cast<uint4*>(tiles + b_tile_byte(t, k_tiles, kt, kh, n, r, c)) = lo;
  *reinterpret_cast<uint4*>(tiles + b_tile_byte(t, k_tiles, kt, kh, n, r, c + 8)) = hi;
}

__device__ __forceinline__ uint32_t pack2(bool f16, float x, float y) {
  uint32_t w;
  if (f16) {
    __half2 const h = __floats2half2_rn(x, y);
    memcpy(&w, &h, 4);
  } else {
    __nv_bfloat162 const b = __floats2bfloat162_rn(x, y);
    memcpy(&w, &b, 4);
  }
  return w;
}

// code * SF of 16 FP4 values (value i in bits 4i..4i+3). The product is exact in FP32 and in
// BF16/FP16, so the B value is the exact dequantized value.
__device__ __forceinline__ void dequant16(uint64_t codes, uint8_t sf, bool f16, uint4& lo,
                                          uint4& hi) {
  float const s = tk::e4m3_to_fp32_sw(sf);
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float const x = __fmul_rn(tk::e2m1_to_fp32_sw(uint32_t(codes >> (8 * i))), s);
    float const y = __fmul_rn(tk::e2m1_to_fp32_sw(uint32_t(codes >> (8 * i + 4))), s);
    w[i] = pack2(f16, x, y);
  }
  lo = make_uint4(w[0], w[1], w[2], w[3]);
  hi = make_uint4(w[4], w[5], w[6], w[7]);
}

// UE8M0 byte -> BF16 2^(e-127): e = 0 -> 2^-127 (subnormal), e = 255 -> NaN (dequant.cuh contract).
__device__ __forceinline__ __nv_bfloat16 ue8m0_to_bf16(uint32_t e) {
  uint16_t const b = e == 255u ? 0x7FC0u : e == 0u ? 0x0040u : uint16_t(e << 7);
  return __ushort_as_bfloat16(b);
}

// MXFP4: code * 2^(e-127) of 16 FP4 values, multiplied in BF16. BF16 math keeps the BF16
// subnormals (FP32 ones would flush under -ftz), so each value is exact or inf on overflow.
__device__ __forceinline__ void dequant16_mx(uint64_t codes, uint8_t sf, uint4& lo, uint4& hi) {
  __nv_bfloat162 const s = __bfloat162bfloat162(ue8m0_to_bf16(sf));
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_bfloat162 const v =
        __hmul2(__floats2bfloat162_rn(tk::e2m1_to_fp32_sw(uint32_t(codes >> (8 * i))),
                                      tk::e2m1_to_fp32_sw(uint32_t(codes >> (8 * i + 4)))),
                s);
    memcpy(&w[i], &v, 4);
  }
  lo = make_uint4(w[0], w[1], w[2], w[3]);
  hi = make_uint4(w[4], w[5], w[6], w[7]);
}

template <class TIn>
__device__ __forceinline__ void load16(tk::PackedVec<TIn, 16>& v, TIn const* src) {
  uint4 raw[2];
  raw[0] = reinterpret_cast<uint4 const*>(src)[0];
  raw[1] = reinterpret_cast<uint4 const*>(src)[1];
  memcpy(&v, raw, sizeof(raw));
}

// Per-token scales of nvfp4QuantAndPerTokenScaleKernel: the encode scale given to the quantizer
// and the per-token (decode) scale written to tok_scale.
template <class Cfg4o6, bool kExact>
__device__ __forceinline__ void token_scales(float row_amax, float gs_inv, float& enc, float& pts) {
  if constexpr (tensorrt_llm::common::IsNVFP44Over6Config<Cfg4o6>::value && kExact) {
    float const gs = __fdiv_rn(1.0f, gs_inv);
    enc = 0.0f;
    if (row_amax != 0.0f) {
      enc = fminf(__fdiv_rn(gs, row_amax), FLT_MAX);
      if (enc == 0.0f) enc = 1.0f;
    }
    pts = enc != 0.0f ? __fdiv_rn(1.0f, enc) : 0.0f;
  } else if constexpr (kExact) {
    float const gs = __fdiv_rn(1.0f, gs_inv);
    float const row_enc = row_amax != 0.0f ? fminf(__fdiv_rn(gs, row_amax), FLT_MAX) : FLT_MAX;
    enc = row_enc != 0.0f ? row_enc : 1.0f;
    pts = __fdiv_rn(1.0f, enc);
  } else {
    pts = tk::nvfp4_quantize_mul(row_amax, gs_inv);  // the fast-math product of the FI kernel
    enc = pts != 0.0f ? tk::reciprocal_approximate_ftz(pts) : 0.0f;
  }
}

// B rows of GEMM1, a CTA per row (grid-stride); rows past the expert's count and values past K
// are written as 0.
template <TileMode kMode, class TIn, class Cfg4o6, bool kExact>
__global__ void __launch_bounds__(kTileThreads) prep_tiles(PrepArgs a, TileParams tp) {
  bool const dense = a.topk_ids == nullptr;
  int const n = a.n, k = a.k;
  int const k_tiles = (k + 127) / 128;
  int const chunks = k_tiles * 8;
  int const rows = (dense ? tp.dense_tiles : *a.num_tiles) * n;
  char* const tiles = static_cast<char*>(a.x_tiles);
  uint4 const zero = make_uint4(0u, 0u, 0u, 0u);
  for (int g = blockIdx.x; g < rows; g += gridDim.x) {
    int const t = g / n, r = g % n;
    int p = g, token = -1, e = 0;
    if (dense) {
      if (p < a.tokens) token = p;
    } else {
      e = a.tile_expert[t];
      p = a.tile_n0[t] + r;
      if (p < a.expert_offset[e + 1]) token = a.permuted_src[p] / a.top_k;
    }
    if (token < 0) {  // uniform over the CTA
      for (int j = threadIdx.x; j < chunks; j += kTileThreads) {
        store16(tiles, t, k_tiles, n, r, j, zero, zero);
      }
      continue;
    }
    if constexpr (kMode == TileMode::kCopy) {
      uint16_t const* const xrow = static_cast<uint16_t const*>(a.x) + int64_t(token) * a.x_ld;
      for (int j = threadIdx.x; j < chunks; j += kTileThreads) {
        int const v = j * 16;  // k % 8 == 0: a half is all in or all out
        uint4 const lo = v < k ? *reinterpret_cast<uint4 const*>(xrow + v) : zero;
        uint4 const hi = v + 8 < k ? *reinterpret_cast<uint4 const*>(xrow + v + 8) : zero;
        store16(tiles, t, k_tiles, n, r, j, lo, hi);
      }
    } else if constexpr (kMode == TileMode::kFp4) {
      uint8_t const* const xrow = static_cast<uint8_t const*>(a.x) + int64_t(token) * a.x_ld;
      int const k16 = k / 16;  // 16-value groups of the logical K
      for (int j = threadIdx.x; j < chunks; j += kTileThreads) {
        uint4 lo = zero, hi = zero;
        if (j < k16) {
          uint64_t const codes = *reinterpret_cast<uint64_t const*>(xrow + j * 8);
          if (a.x_fmt == XFmt::kMxfp4Swz128x4) {  // one UE8M0 SF per 32 values
            dequant16_mx(codes, a.x_sf[tk::get_sf_out_offset_128x4(token, j / 2, a.x_sf_cols)], lo,
                         hi);
          } else {
            int64_t const sf_off = a.x_fmt == XFmt::kNvfp4Linear ? int64_t(token) * a.x_sf_cols + j
                                   : a.x_fmt == XFmt::kNvfp4Swz128x4
                                       ? tk::get_sf_out_offset_128x4(token, j, a.x_sf_cols)
                                       : tk::get_sf_out_offset_8x4(token, j, a.x_sf_cols);
            dequant16(codes, a.x_sf[sf_off], a.f16, lo, hi);
          }
        }
        store16(tiles, t, k_tiles, n, r, j, lo, hi);
      }
    } else {
      using Vec = tk::PackedVec<TIn, 16>;
      using Reduce = cub::BlockReduce<float, kTileThreads>;
      __shared__ typename Reduce::TempStorage reduce_tmp;
      __shared__ float row_amax_smem;
      TIn const* const xrow = static_cast<TIn const*>(a.x) + int64_t(token) * a.x_ld;
      float row_amax = 0.0f, enc;
      if (a.per_token) {
        float local = 0.0f;
        for (int j = threadIdx.x; j < k / 16; j += kTileThreads) {
          Vec v;
          load16(v, xrow + j * 16);
          std::remove_reference_t<decltype(v.elts[0])> m(0.f, 0.f);
#pragma unroll
          for (int i = 0; i < 8; ++i) m = __hmax2(m, __habs2(v.elts[i]));
          local = fmaxf(local, static_cast<float>(__hmax(m.x, m.y)));
        }
        float const amax = Reduce(reduce_tmp).Reduce(local, cuda::maximum<>{});
        if (threadIdx.x == 0) row_amax_smem = amax;
        __syncthreads();
        row_amax = row_amax_smem;
        float pts;
        token_scales<Cfg4o6, kExact>(row_amax, a.x_global ? *a.x_global : tp.gs_inv_default, enc,
                                     pts);
        if (threadIdx.x == 0) a.tok_scale[p] = pts;
      } else {
        enc = a.x_global ? a.x_global[a.x_global_per_expert ? e : 0] : 1.0f;
      }
      for (int j = threadIdx.x; j < chunks; j += kTileThreads) {
        uint4 lo = zero, hi = zero;
        if (j * 16 < k) {
          Vec v;
          load16(v, xrow + j * 16);
          uint8_t sf;
          uint64_t const codes =
              tk::cvt_warp_fp16_to_fp4<TIn, 16, 16, false, kExact, Cfg4o6>(v, enc, &sf, row_amax);
          dequant16(codes, sf, a.f16, lo, hi);
        }
        store16(tiles, t, k_tiles, n, r, j, lo, hi);
      }
      if (a.per_token) __syncthreads();  // reduce_tmp and row_amax_smem serve the next row
    }
  }
}

inline void check_launch(char const* what) {
  cudaError_t const err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, "fp4_sm90 prep: ", what,
                   " launch failed: ", cudaGetErrorString(err));
}

template <class Kernel>
inline void set_smem(Kernel kernel, size_t bytes) {
  if (bytes > 48 * 1024) {
    cudaError_t const err =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    FLASHINFER_CHECK(err == cudaSuccess, "fp4_sm90 prep: ", bytes,
                     " bytes of shared memory for the experts: ", cudaGetErrorString(err));
  }
}

// kRecipe* bits -> std::false_type (standard NVFP4) or NVFP44Over6Config.
template <class F>
inline void dispatch_recipe(uint8_t recipe, F&& f) {
  using tensorrt_llm::common::NVFP44Over6Config;
  using Err = tensorrt_llm::common::NVFP44Over6ErrMode;
  FLASHINFER_CHECK((recipe & ~0xFu) == 0, "fp4_sm90 prep: unknown recipe bits ", int(recipe));
#if defined(FP4_SM90_RECIPE)
  FLASHINFER_CHECK(recipe == FP4_SM90_RECIPE, "fp4_sm90 prep: built for recipe ", FP4_SM90_RECIPE,
                   ", got ", int(recipe));
#endif
  if (!(recipe & kRecipe4over6)) {
    FLASHINFER_CHECK(recipe == 0, "fp4_sm90 prep: recipe bits ", int(recipe),
                     " need the 4over6 bit");
#if !defined(FP4_SM90_RECIPE) || FP4_SM90_RECIPE == 0
    f(std::false_type{});
#endif
    return;
  }
  bool const e256 = recipe & kRecipeE4m3Max256, mse = recipe & kRecipeErrMse,
             fast = recipe & kRecipeFastMath;
#if defined(FP4_SM90_RECIPE)
  if constexpr ((FP4_SM90_RECIPE & kRecipe4over6) != 0) {
    (void)e256, (void)mse, (void)fast;
    f(NVFP44Over6Config < (FP4_SM90_RECIPE & kRecipeE4m3Max256) ? 256 : 448,
      (FP4_SM90_RECIPE & kRecipeErrMse) ? Err::MSE : Err::MAE,
      (FP4_SM90_RECIPE & kRecipeFastMath) != 0 > {});
  }
#else
  auto const by_err = [&](auto e4m3) {
    constexpr int kE4m3 = decltype(e4m3)::value;
    if (mse) {
      fast ? f(NVFP44Over6Config<kE4m3, Err::MSE, true>{})
           : f(NVFP44Over6Config<kE4m3, Err::MSE, false>{});
    } else {
      fast ? f(NVFP44Over6Config<kE4m3, Err::MAE, true>{})
           : f(NVFP44Over6Config<kE4m3, Err::MAE, false>{});
    }
  };
  e256 ? by_err(std::integral_constant<int, 256>{}) : by_err(std::integral_constant<int, 448>{});
#endif
}

template <TileMode kMode, class TIn, class Cfg4o6, bool kExact>
inline void launch_tiles(const PrepArgs& a, const TileParams& tp, int grid, cudaStream_t stream) {
  prep_tiles<kMode, TIn, Cfg4o6, kExact><<<grid, kTileThreads, 0, stream>>>(a, tp);
  check_launch("tile");
}

template <class TIn>
inline void launch_quant_tiles(const PrepArgs& a, const TileParams& tp, int grid,
                               cudaStream_t stream) {
#if defined(FLASHINFER_FP4_SM90_EXACT)
  constexpr bool kExact = FLASHINFER_FP4_SM90_EXACT != 0;
  dispatch_recipe(a.recipe, [&](auto cfg) {
    launch_tiles<TileMode::kQuant, TIn, decltype(cfg), kExact>(a, tp, grid, stream);
  });
#else
  bool const exact = tensorrt_llm::common::getEnvDisableFP4QuantFastMath();
  dispatch_recipe(a.recipe, [&](auto cfg) {
    exact ? launch_tiles<TileMode::kQuant, TIn, decltype(cfg), true>(a, tp, grid, stream)
          : launch_tiles<TileMode::kQuant, TIn, decltype(cfg), false>(a, tp, grid, stream);
  });
#endif
}

inline bool aligned(const void* p, uintptr_t bytes) {
  return reinterpret_cast<uintptr_t>(p) % bytes == 0;
}

}  // namespace prep_detail

// Bytes of PrepArgs::workspace for a routed prep: int32 per (sort chunk, local expert). Sized from
// the shape alone, since the MoE layout is made before topk_ids is bound; a dense prep uses none.
inline size_t fp4_sm90_prep_workspace_bytes(const PrepArgs& a) {
  return moe_layout::prep_workspace_bytes(a.tokens, a.top_k, a.num_local_experts);
}

// Routing sort, n-tile list and GEMM1 B tiles (x_tiles null: sort and list only). Dense
// (topk_ids null) may leave every schedule output null; MoE needs them all.
inline void launch_fp4_sm90_prep(const PrepArgs& a, cudaStream_t stream) {
  using namespace prep_detail;
  bool const dense = a.topk_ids == nullptr;
  int const n = a.n, E = a.num_local_experts;
  FLASHINFER_CHECK(a.tokens >= 0 && a.k > 0, "fp4_sm90 prep: tokens ", a.tokens, ", k ", a.k);
  FLASHINFER_CHECK(n >= 8 && n <= 256 && (n & (n - 1)) == 0,
                   "fp4_sm90 prep: n must be a power of two in [8, 256], got ", n);
  if (dense) {
    FLASHINFER_CHECK(a.top_k == 1 && E == 1 && a.local_expert_offset == 0,
                     "fp4_sm90 prep: dense takes top_k 1 and one expert");
  } else {
    FLASHINFER_CHECK(a.top_k >= 1 && E >= 1 && a.local_expert_offset >= 0 &&
                         a.local_expert_offset + E <= a.num_experts,
                     "fp4_sm90 prep: top_k ", a.top_k, ", local experts [", a.local_expert_offset,
                     ", +", E, ") of ", a.num_experts);
    FLASHINFER_CHECK(a.expert_offset && a.expert_tile_base && a.permuted_src && a.src_permuted &&
                         a.tile_expert && a.tile_n0 && a.num_tiles,
                     "fp4_sm90 prep: MoE needs every schedule output");
  }
  // n2 != 0: GEMM2 has its own n-tile list (MoE, n2 != n).
  bool const cross = a.n2 != 0;
  if (cross) {
    FLASHINFER_CHECK(!dense && a.n2 != n && a.n2 >= 8 && a.n2 <= 256 && (a.n2 & (a.n2 - 1)) == 0,
                     "fp4_sm90 prep: n2 must be another power of two in [8, 256], got ", a.n2);
    FLASHINFER_CHECK(a.expert_tile_base2 && a.tile_expert2 && a.tile_n02 && a.num_tiles2,
                     "fp4_sm90 prep: n2 needs its schedule outputs");
    FLASHINFER_CHECK(a.act_tiles == nullptr || (aligned(a.act_tiles, 16) && a.act_k > 0),
                     "fp4_sm90 prep: act_tiles / act_k");
  }
  int64_t const total64 = int64_t(a.tokens) * a.top_k;
  FLASHINFER_CHECK(total64 <= INT_MAX / 2, "fp4_sm90 prep: ", total64, " expanded rows");
  int const total = int(total64);
  int64_t const busy = std::min<int64_t>(E, total);  // nonempty experts at most
  int64_t const need_tiles = (total64 + busy * (n - 1)) / n;
  FLASHINFER_CHECK(a.max_tiles >= need_tiles, "fp4_sm90 prep: max_tiles ", a.max_tiles,
                   " is below the routing bound ", need_tiles);
  if (cross) {
    int64_t const need_tiles2 = (total64 + busy * (a.n2 - 1)) / a.n2;
    FLASHINFER_CHECK(a.max_tiles2 >= need_tiles2, "fp4_sm90 prep: max_tiles2 ", a.max_tiles2,
                     " is below the routing bound ", need_tiles2);
  }

  // B tile source checks (before any launch: an error leaves no partial output).
  bool const mx_in = a.x_fmt == XFmt::kMxfp4Swz128x4;
  bool const fp4_in = a.x_fmt == XFmt::kNvfp4Linear || a.x_fmt == XFmt::kNvfp4Swz128x4 ||
                      a.x_fmt == XFmt::kNvfp4Swz8x4 || mx_in;
  bool const f16_in = a.x_fmt == XFmt::kF16;
  TileParams tp{dense && a.num_tiles == nullptr ? a.max_tiles : int((total64 + n - 1) / n), 0.0f};
  if (a.x_tiles == nullptr) {
    FLASHINFER_CHECK(!a.quantize && !a.per_token,
                     "fp4_sm90 prep: quantize / per_token need x_tiles");
  } else {
    FLASHINFER_CHECK(a.x != nullptr && aligned(a.x_tiles, 16), "fp4_sm90 prep: x / x_tiles");
    FLASHINFER_CHECK(!a.per_token || a.tok_scale != nullptr,
                     "fp4_sm90 prep: per_token needs tok_scale");
    if (fp4_in) {
      FLASHINFER_CHECK(!a.quantize && !a.per_token,
                       "fp4_sm90 prep: an FP4 input is neither quantized again nor given a "
                       "per-token scale here");
      FLASHINFER_CHECK(
          a.x_sf != nullptr && a.k % (mx_in ? 32 : 16) == 0 && a.x_ld % 8 == 0 && aligned(a.x, 8) &&
              a.x_sf_cols >= a.k / (mx_in ? 32 : 16),
          mx_in ? "fp4_sm90 prep: FP4 input needs x_sf, k % 32 == 0, 8-byte rows and k / 32 scale "
                  "columns"
                : "fp4_sm90 prep: FP4 input needs x_sf, k % 16 == 0, 8-byte rows and k / 16 scale "
                  "columns");
      FLASHINFER_CHECK(!mx_in || !a.f16,
                       "fp4_sm90 prep: an MXFP4 input needs BF16 B (FP16 cannot hold 2^(e-127))");
    } else {
      FLASHINFER_CHECK(a.x_fmt == XFmt::kBf16 || f16_in, "fp4_sm90 prep: x_fmt ", int(a.x_fmt));
      FLASHINFER_CHECK(a.x_ld % 8 == 0 && aligned(a.x, 16) && a.k % (a.quantize ? 16 : 8) == 0,
                       "fp4_sm90 prep: 16-bit input needs 16-byte rows and k % ",
                       a.quantize ? 16 : 8, " == 0");
      if (a.quantize) {
        bool const e256 = (a.recipe & kRecipe4over6) && (a.recipe & kRecipeE4m3Max256);
        tp.gs_inv_default = 1.f / ((e256 ? 256.f : 448.f) * 6.f);  // NVFP4RecipeSpec
      } else {
        FLASHINFER_CHECK(!a.per_token, "fp4_sm90 prep: per_token needs quantize");
        FLASHINFER_CHECK(bool(a.f16) == f16_in,
                         "fp4_sm90 prep: a 16-bit input is copied as is, so its dtype must be "
                         "the B dtype");
      }
    }
  }
  if (a.quantize) dispatch_recipe(a.recipe, [](auto) {});  // recipe check before any launch

  if (!dense) {
    int const chunks = (total + kChunk - 1) / kChunk;
    FLASHINFER_CHECK(chunks == 0 || a.workspace != nullptr, "fp4_sm90 prep: workspace");
    size_t const count_smem = size_t(E) * sizeof(int);
    size_t const scatter_smem = size_t(E) * (1 + kSortWarps) * sizeof(int);
    if (chunks > 0) {
      set_smem(prep_count, count_smem);
      prep_count<<<chunks, kSortThreads, count_smem, stream>>>(a, total);
      check_launch("count");
    }
    prep_scan<<<1, kScanThreads, 0, stream>>>(a, total, chunks);
    check_launch("scan");
    if (chunks > 0) {
      set_smem(prep_scatter, scatter_smem);
      prep_scatter<<<chunks, kSortThreads, scatter_smem, stream>>>(a, total);
      check_launch("scatter");
    }
    if (cross) {
      prep_scan2<<<1, kScanThreads, 0, stream>>>(a);
      check_launch("scan2");
      if (a.act_tiles != nullptr) {
        prep_pad<<<std::min(E, kMaxTileCtas), kSortThreads, 0, stream>>>(a);
        check_launch("pad");
      }
    }
  } else if (a.expert_offset || a.expert_tile_base || a.permuted_src || a.src_permuted ||
             a.tile_expert || a.tile_n0 || a.num_tiles) {
    prep_scan<<<1, kScanThreads, 0, stream>>>(a, total, 0);
    check_launch("scan");
  }

  if (a.x_tiles == nullptr || (dense ? tp.dense_tiles : total) == 0) return;
  int64_t const rows = int64_t(dense ? tp.dense_tiles : a.max_tiles) * n;
  int const grid = int(std::min<int64_t>(rows, kMaxTileCtas));
  if (fp4_in) {
    launch_tiles<TileMode::kFp4, __nv_bfloat16, std::false_type, false>(a, tp, grid, stream);
  } else if (!a.quantize) {
    launch_tiles<TileMode::kCopy, __nv_bfloat16, std::false_type, false>(a, tp, grid, stream);
  } else if (f16_in) {
    launch_quant_tiles<half>(a, tp, grid, stream);
  } else {
    launch_quant_tiles<__nv_bfloat16>(a, tp, grid, stream);
  }
}

}  // namespace fp4_sm90
}  // namespace flashinfer
