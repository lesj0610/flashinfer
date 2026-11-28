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
// Ordered top-k sum of the MoE GEMM2 rows (D6):
//   out[t, c] = OT(sum over slots k = 0 .. top_k - 1 in order of w[t, k] y[src_permuted[t, k], c])
// in FP32 from 0, one FMA per slot; slots without a local row (src_permuted -1) are skipped and
// a null topk_weights is weight 1. Deterministic: the order depends on neither the routing nor
// the launch. keep_unrouted: a token without a local row keeps its out row (all-to-all).
// W4A16 modules (FP4_SM90_W4A16) follow the split-kernel reference's combine instead: the sum
// starts at +0 and takes physical slot 0's term as mul.rn(term, w) only if slot 0 is local,
// every later local slot as fma.rn(term, w, sum), explicit non-FTZ roundings (no contraction, no
// flush: -use_fast_math's fmaf is fma.rn.ftz), so a skipped slot 0 leaves +0 and a -0 term
// added to it gives +0 as in the reference. Their fused finalize (FinalizeArgs::fused: GEMM2
// already scaled each row by alpha x its weight and rounded it to BF16) sums the local rows in
// slot order from +0, each add rounded to BF16 without flushing: one of the orders of the
// add.noftz.bf16 reduction into a zeroed out of SM100's fused finalize, and deterministic.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "flashinfer/exception.h"
#include "params.cuh"

namespace flashinfer {
namespace fp4_sm90 {
namespace fin {

constexpr int kThreads = 256;

__device__ __forceinline__ float ld1(const float* p) { return *p; }
__device__ __forceinline__ float ld1(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float ld1(const half* p) { return __half2float(*p); }

__device__ __forceinline__ void ld8(const float* p, float (&f)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}
template <class T>
__device__ __forceinline__ void ld8(const T* p, float (&f)[8]) {
  const uint4 w = *reinterpret_cast<const uint4*>(p);
  const T* e = reinterpret_cast<const T*>(&w);
#pragma unroll
  for (int i = 0; i < 8; ++i) f[i] = ld1(e + i);
}

__device__ __forceinline__ void st1(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void st1(half* p, float v) { *p = __float2half_rn(v); }
template <class T>
__device__ __forceinline__ void st8(T* p, const float (&f)[8]) {
  uint4 w;
  T* e = reinterpret_cast<T*>(&w);
#pragma unroll
  for (int i = 0; i < 8; ++i) st1(e + i, f[i]);
  *reinterpret_cast<uint4*>(p) = w;
}

#if defined(FP4_SM90_W4A16)
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float fma_rn(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
// BF16 RN of a + b for BF16 a and b, no flush: the FP32 sum's second rounding is innocuous
// (24 >= 2 x 8 + 2 bits), so this is the BF16 add (add.noftz.bf16) of the fused reduction.
__device__ __forceinline__ float add_bf16(float a, float b) {
  float s;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(b));
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(s));
  return __uint_as_float(static_cast<uint32_t>(h) << 16);
}
#endif

template <class YT, class WT, class OT, int V>
__global__ void __launch_bounds__(kThreads) finalize_kernel(FinalizeArgs a) {
  const YT* y = static_cast<const YT*>(a.y);
  const WT* w = static_cast<const WT*>(a.topk_weights);
  OT* out = static_cast<OT*>(a.out);
  const int hv = (a.hidden + V - 1) / V;
  const int64_t total = static_cast<int64_t>(a.tokens) * hv;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * kThreads) {
    const int t = static_cast<int>(i / hv), c0 = static_cast<int>(i % hv) * V;
    const int64_t slot0 = static_cast<int64_t>(t) * a.top_k;
    float acc[V];
#pragma unroll
    for (int v = 0; v < V; ++v) acc[v] = 0.f;
    bool routed = false;
    for (int k = 0; k < a.top_k; ++k) {
      const int row = a.src_permuted[slot0 + k];
      if (row < 0) continue;
      routed = true;
      const float wk = w != nullptr ? ld1(w + slot0 + k) : 1.f;
      const YT* src = y + static_cast<int64_t>(row) * a.y_ld + c0;
      float f[V];
      if constexpr (V == 8) {
        ld8(src, f);
      } else {
        f[0] = ld1(src);
      }
#if defined(FP4_SM90_W4A16)
      if (a.fused) {
#pragma unroll
        for (int v = 0; v < V; ++v) acc[v] = add_bf16(acc[v], f[v]);
        continue;
      }
#pragma unroll
      for (int v = 0; v < V; ++v) acc[v] = k == 0 ? mul_rn(f[v], wk) : fma_rn(f[v], wk, acc[v]);
#else
#pragma unroll
      for (int v = 0; v < V; ++v) acc[v] = fmaf(wk, f[v], acc[v]);
#endif
    }
    if (!routed && a.keep_unrouted) continue;
    OT* dst = out + static_cast<int64_t>(t) * a.out_ld + c0;
    if constexpr (V == 8) {
      st8(dst, acc);
    } else {
      st1(dst, acc[0]);
    }
  }
}

template <class YT, class WT, class OT>
void launch_v(const FinalizeArgs& a, bool vec, cudaStream_t stream) {
  const int v = vec ? 8 : 1;
  const int64_t total = static_cast<int64_t>(a.tokens) * ((a.hidden + v - 1) / v);
  const int grid = static_cast<int>(std::min<int64_t>((total + kThreads - 1) / kThreads, 1 << 20));
  if (vec)
    finalize_kernel<YT, WT, OT, 8><<<grid, kThreads, 0, stream>>>(a);
  else
    finalize_kernel<YT, WT, OT, 1><<<grid, kThreads, 0, stream>>>(a);
}

template <class YT, class WT>
void launch_o(const FinalizeArgs& a, bool vec, cudaStream_t stream) {
  if (a.out_f16)
    launch_v<YT, WT, half>(a, vec, stream);
  else
    launch_v<YT, WT, __nv_bfloat16>(a, vec, stream);
}

template <class YT>
void launch_w(const FinalizeArgs& a, bool vec, cudaStream_t stream) {
  switch (a.w_dtype) {
    case 0:
      return launch_o<YT, float>(a, vec, stream);
    case 1:
      return launch_o<YT, __nv_bfloat16>(a, vec, stream);
    default:
      return launch_o<YT, half>(a, vec, stream);
  }
}

}  // namespace fin

inline void launch_fp4_sm90_finalize(const FinalizeArgs& a, cudaStream_t stream) {
  FLASHINFER_CHECK(a.tokens >= 0 && a.top_k >= 1 && a.hidden >= 1, "finalize: bad shape tokens ",
                   a.tokens, " top_k ", a.top_k, " hidden ", a.hidden);
  if (a.tokens == 0) return;
  FLASHINFER_CHECK(a.y != nullptr && a.out != nullptr && a.src_permuted != nullptr,
                   "finalize: y, out and src_permuted are required");
  FLASHINFER_CHECK(a.y_dtype <= 2 && a.w_dtype <= 2, "finalize: y_dtype / w_dtype out of range");
  FLASHINFER_CHECK(a.y_ld >= a.hidden && a.out_ld >= a.hidden,
                   "finalize: y_ld and out_ld must be >= hidden");
  const bool vec = a.hidden % 8 == 0 && a.y_ld % 8 == 0 && a.out_ld % 8 == 0 &&
                   reinterpret_cast<uintptr_t>(a.y) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(a.out) % 16 == 0;
  switch (a.y_dtype) {
    case 0:
      fin::launch_w<float>(a, vec, stream);
      break;
    case 1:
      fin::launch_w<__nv_bfloat16>(a, vec, stream);
      break;
    default:
      fin::launch_w<half>(a, vec, stream);
  }
  const cudaError_t err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, "finalize launch: ", cudaGetErrorString(err));
}

}  // namespace fp4_sm90
}  // namespace flashinfer
