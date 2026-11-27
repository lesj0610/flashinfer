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
// Token-wise NVFP4 quantization of the GEMM2 input (per-token recipes): one CTA per expanded row,
// the SM100+ nvfp4_quant_and_per_token_scale numerics with the inverse global scale gsi (the
// caller's, else the recipe's 1 / (e4m3Max x 6)):
//   fast:  s = amax x gsi (FTZ), encode = rcp.approx.ftz(s) (0 for s = 0)
//   exact (DISABLE_FP4_QUANT_FAST_MATH): encode = min(gs / amax, FLT_MAX) with 1 for an
//          underflow, amax = 0 -> 0 (4over6) or FLT_MAX; s = 1 / encode (gs = 1 / gsi)
// then cvt_warp_fp16_to_fp4(x, encode, amax) per 16 values. The GEMM2 B tile gets the exact
// dequantized value code x SF; tok_scale[p] = s goes to GEMM2's epilogue. Rows of the n-tile past
// the expert's count and K past cols are written as 0. Expanded rows past the local experts get
// tok_scale 0 and no tile row. With fq_codes (fc2_input_out) row p's codes, scales and s also go
// to the caller's row fq_src[p], (token, slot) order, zero past cols; CTA p zeroes the caller's
// row p if no local expert takes it (fq_dst[p] < 0).
#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "epilogue.cuh"
#include "flashinfer/exception.h"
#include "tensorrt_llm/common/envUtils.h"

namespace flashinfer {
namespace fp4_sm90 {
namespace rq {

namespace tk = tensorrt_llm::kernels;
constexpr int kThreads = 128;

__device__ __forceinline__ float div_rn_ftz(float a, float b) {
  float r;
  asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

template <class T>
__device__ __forceinline__ float vmax_abs(uint4 w, float m) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t u = (&w.x)[i];
    const float2 f =
        epi::to_f2(*reinterpret_cast<
                   const std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>*>(&u));
    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  return m;
}

// Zeroes blocks [b0, fq_cols / 16) of the caller's row q.
__device__ __forceinline__ void fq_zero(const RowQuantArgs& a, int64_t q, int b0) {
  const int fb = a.fq_cols / 16;
  for (int b = b0 + static_cast<int>(threadIdx.x); b < fb; b += kThreads) {
    reinterpret_cast<uint64_t*>(a.fq_codes)[q * fb + b] = 0;
    a.fq_sf[q * fb + b] = 0;
  }
}

template <class TI, class TO, uint8_t R, bool EXACT>
__global__ void __launch_bounds__(kThreads) row_quant_kernel(RowQuantArgs a) {
  __shared__ int s_loc[3];  // tile, row in tile, rows of the tile (0: not local)
  __shared__ float s_red[kThreads / 32];
  const int p = blockIdx.x, tid = threadIdx.x;
  const int n = a.n, kp = (a.cols + 127) / 128 * 128, k_tiles = kp / 128;
  const int blocks = a.cols / 16;

  if (tid == 0) {
    int t = 0, r = 0, cnt = 0;
    if (a.expert_offset == nullptr) {
      t = p / n;
      r = p % n;
      cnt = min(n, a.rows - t * n);
    } else {
      // Local expert of p. expert_tile_base[E_local] == *num_tiles ends the walk (prep).
      const int nt = *a.num_tiles;
      for (int e = 0;; ++e) {
        const int base = a.expert_tile_base[e];
        if (base >= nt) break;  // past the local experts
        const int o0 = a.expert_offset[e], o1 = a.expert_offset[e + 1];
        if (p < o1) {
          const int off = p - o0;
          t = base + off / n;
          r = off % n;
          cnt = min(n, o1 - (o0 + off / n * n));
          break;
        }
      }
    }
    s_loc[0] = t;
    s_loc[1] = r;
    s_loc[2] = cnt;
  }
  __syncthreads();
  const int t = s_loc[0], r = s_loc[1], cnt = s_loc[2];
  const bool fq = a.fq_codes != nullptr;
  if (fq && a.fq_dst[p] < 0) {  // the caller's row p belongs to another rank
    fq_zero(a, p, 0);
    if (tid == 0) a.fq_tok[p] = 0.f;
  }
  if (cnt == 0) {
    if (tid == 0) a.tok_scale[p] = 0.f;
    return;
  }
  const int64_t q = fq ? a.fq_src[p] : 0, fb = a.fq_cols / 16;

  const TI* row = static_cast<const TI*>(a.in) + static_cast<int64_t>(p) * a.in_ld;
  float m = 0.f;
  for (int b = tid; b < blocks; b += kThreads) {
    const uint4* src = reinterpret_cast<const uint4*>(row + 16 * b);
    m = vmax_abs<TI>(src[1], vmax_abs<TI>(src[0], m));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) s_red[tid >> 5] = m;
  __syncthreads();
  m = s_red[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; ++w) m = fmaxf(m, s_red[w]);

  constexpr bool k4over6 = (R & kRecipe4over6) != 0;
  const float gsi = a.gsi != nullptr ? *a.gsi : 1.f / (epi::recipe_e4m3_max(R) * 6.f);
  float s, enc;
  if constexpr (EXACT) {
    const float gs = div_rn_ftz(1.f, gsi);
    if (k4over6) {
      enc = m != 0.f ? fminf(div_rn_ftz(gs, m), FLT_MAX) : 0.f;
      if (m != 0.f && enc == 0.f) enc = 1.f;
      s = enc != 0.f ? div_rn_ftz(1.f, enc) : 0.f;
    } else {
      enc = m != 0.f ? fminf(div_rn_ftz(gs, m), FLT_MAX) : FLT_MAX;
      if (enc == 0.f) enc = 1.f;
      s = div_rn_ftz(1.f, enc);
    }
  } else {
    s = epi::mul_ftz(m, gsi);
    enc = s != 0.f ? tk::reciprocal_approximate_ftz(s) : 0.f;
  }
  if (tid == 0) {
    a.tok_scale[p] = s;
    if (fq) a.fq_tok[q] = s;
  }

  char* const tiles = static_cast<char*>(a.out_tiles);
  for (int b = tid; b < kp / 16; b += kThreads) {
    const int c0 = 16 * b;
    uint4 o[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    if (b < blocks) {
      const uint4* src = reinterpret_cast<const uint4*>(row + c0);
      auto v = epi::packed16<TI>(src[0], src[1]);
      uint8_t sf;
      const uint64_t codes = epi::nvfp4_block<R, EXACT, TI, TO>(v, enc, m, o, sf);
      if (fq) {
        reinterpret_cast<uint64_t*>(a.fq_codes)[q * fb + b] = codes;
        a.fq_sf[q * fb + b] = sf;
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int c = c0 + 8 * u;
      *reinterpret_cast<uint4*>(tiles +
                                b_tile_byte(t, k_tiles, c >> 7, (c >> 6) & 1, n, r, c & 63)) = o[u];
    }
  }
  if (fq) fq_zero(a, q, blocks);

  // The tile's last row zeroes the rows past the expert's count (whole 128-byte rows).
  if (r == cnt - 1 && cnt < n) {
    const int per_half = (n - cnt) * 8;
    for (int i = tid; i < (kp / 64) * per_half; i += kThreads) {
      const int hf = i / per_half, j = i % per_half;
      // 16-byte chunk j % 8 of row cnt + j / 8 (value column 8 (j % 8), swizzled per chunk)
      *reinterpret_cast<uint4*>(tiles + b_tile_byte(t, k_tiles, hf >> 1, hf & 1, n, cnt + j / 8,
                                                    8 * (j % 8))) = make_uint4(0, 0, 0, 0);
    }
  }
}

// FP4_SM90_RECIPE / FLASHINFER_FP4_SM90_EXACT (module defines, as prep.cuh) compile one quantizer;
// otherwise the recipe is dispatched and FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH read at launch,
// as in the SM100 quantizers.
template <class TI, class TO, uint8_t R>
void launch(const RowQuantArgs& a, cudaStream_t stream) {
#if defined(FP4_SM90_RECIPE)
  FLASHINFER_CHECK(R == (FP4_SM90_RECIPE), "row_quant: this module is built for recipe ",
                   int(FP4_SM90_RECIPE), ", got ", int(R));
  if constexpr (R == (FP4_SM90_RECIPE)) {
#else
  {
#endif
#if defined(FLASHINFER_FP4_SM90_EXACT)
    row_quant_kernel<TI, TO, R, (FLASHINFER_FP4_SM90_EXACT != 0)>
        <<<a.rows, kThreads, 0, stream>>>(a);
#else
    if (tensorrt_llm::common::getEnvDisableFP4QuantFastMath())
      row_quant_kernel<TI, TO, R, true><<<a.rows, kThreads, 0, stream>>>(a);
    else
      row_quant_kernel<TI, TO, R, false><<<a.rows, kThreads, 0, stream>>>(a);
#endif
  }
}

template <class TI, class TO>
void dispatch_recipe(const RowQuantArgs& a, cudaStream_t stream) {
  switch (a.recipe) {
    case 0:
      return launch<TI, TO, 0>(a, stream);
    case 1:
      return launch<TI, TO, 1>(a, stream);
    case 3:
      return launch<TI, TO, 3>(a, stream);
    case 5:
      return launch<TI, TO, 5>(a, stream);
    case 7:
      return launch<TI, TO, 7>(a, stream);
    case 9:
      return launch<TI, TO, 9>(a, stream);
    case 11:
      return launch<TI, TO, 11>(a, stream);
    case 13:
      return launch<TI, TO, 13>(a, stream);
    case 15:
      return launch<TI, TO, 15>(a, stream);
    default:
      FLASHINFER_CHECK(false, "row_quant: recipe bits ", int(a.recipe),
                       " are not a recipe (bits 1-3 need kRecipe4over6)");
  }
}

}  // namespace rq

// GEMM2 B tiles (params.cuh layout) and per-row scales from GEMM1's BF16/FP16 rows.
inline void launch_fp4_sm90_row_quant(const RowQuantArgs& a, cudaStream_t stream) {
  if (a.rows == 0) return;
  FLASHINFER_CHECK(a.rows > 0 && a.cols > 0 && a.cols % 16 == 0,
                   "row_quant: cols must be a positive multiple of 16, got ", a.cols);
  FLASHINFER_CHECK(a.n >= 8 && a.n <= 256 && (a.n & (a.n - 1)) == 0,
                   "row_quant: n must be a power of two in [8, 256], got ", a.n);
  FLASHINFER_CHECK(a.in != nullptr && a.out_tiles != nullptr && a.tok_scale != nullptr,
                   "row_quant: in, out_tiles and tok_scale are required");
  FLASHINFER_CHECK(a.in_ld >= a.cols && a.in_ld % 8 == 0 &&
                       reinterpret_cast<uintptr_t>(a.in) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(a.out_tiles) % 16 == 0,
                   "row_quant: in and out_tiles must be 16-byte aligned with in_ld % 8 == 0");
  const bool fq = a.fq_codes != nullptr;
  FLASHINFER_CHECK(fq == (a.fq_sf != nullptr) && fq == (a.fq_tok != nullptr) &&
                       fq == (a.fq_src != nullptr) && fq == (a.fq_dst != nullptr),
                   "row_quant: fq_codes, fq_sf, fq_tok, fq_src and fq_dst go together");
  FLASHINFER_CHECK(!fq || (a.expert_offset != nullptr && a.fq_cols >= a.cols &&
                           a.fq_cols % 16 == 0 && reinterpret_cast<uintptr_t>(a.fq_codes) % 8 == 0),
                   "row_quant: the caller's copy needs the routing, fq_cols >= cols (a multiple of "
                   "16) and 8-byte aligned codes");
  FLASHINFER_CHECK((a.expert_offset == nullptr) == (a.expert_tile_base == nullptr) &&
                       (a.expert_offset == nullptr) == (a.num_tiles == nullptr),
                   "row_quant: expert_offset, expert_tile_base and num_tiles go together");
  FLASHINFER_CHECK((a.recipe & ~0xF) == 0, "row_quant: unknown recipe bits ", int(a.recipe));
  if (a.in_f16) {
    if (a.out_f16)
      rq::dispatch_recipe<half, half>(a, stream);
    else
      rq::dispatch_recipe<half, __nv_bfloat16>(a, stream);
  } else {
    if (a.out_f16)
      rq::dispatch_recipe<__nv_bfloat16, half>(a, stream);
    else
      rq::dispatch_recipe<__nv_bfloat16, __nv_bfloat16>(a, stream);
  }
  const cudaError_t err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, "row_quant launch: ", cudaGetErrorString(err));
}

}  // namespace fp4_sm90
}  // namespace flashinfer
