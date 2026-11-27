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
// Epilogue of the SM90 (wgmma) FP4 GEMM: alpha, per-token scale, gate/up activation, bias and the
// optional NVFP4 quantization of the result, written row-major or as GEMM2 B tiles (params.cuh).
//
// Numerics follow the SM100+ path each entry point runs:
//  - NVFP4 output quantization (per-tensor recipes, cutlass doActivation): g = T(acc alpha), plus
//    the bias in FP32, the activation in FP32, T(y), then cvt_warp_fp16_to_fp4 under the recipe
//    with fq_global (one or the expert's) as the encode scale. The stored value is code x E4M3 SF
//    (exact in BF16/FP16); 1 / fq_global belongs to GEMM2's alpha.
//  - Per-token scales (tok_scale): T(acc x (alpha x s)) with FTZ products, the bias after the
//    rounding; Relu2 propagates NaN. GEMM2: T(acc x (alpha x s) + bias).
//  - Otherwise (W4A16, GEMM2, dense): T(act(acc alpha + bias)), no rounding before the activation.
//
// Accumulators arrive with the dequantization fold already removed (dequant.cuh), unless the Cfg
// states Cfg::kAccScale.
//
// Build: -DENABLE_BF16 -DENABLE_FP8 -DFLASHINFER_FP4_SW_CVT (FlashInfer's NVFP4 software path
// below SM100). FLASHINFER_FP4_SM90_EXACT=0/1 is the quantizer's DISABLE_FP4_QUANT_FAST_MATH
// (FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH read by the module builder); the NVFP4 epilogue needs
// it. FP4_SM90_ACT=<ActivationType>, when defined, fixes the GEMM1 activation at compile time
// (EpiArgs::act must match).
#pragma once

#if !defined(ENABLE_BF16) || !defined(ENABLE_FP8)
#error "fp4_sm90 needs -DENABLE_BF16 -DENABLE_FP8"
#endif
#if !defined(FLASHINFER_FP4_SW_CVT)
#error "fp4_sm90 needs -DFLASHINFER_FP4_SW_CVT: cvt_warp_fp16_to_fp4 has no SM90 body without it"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

#include "a_layout.cuh"
#include "params.cuh"
#include "tensorrt_llm/kernels/cutlass_kernels/include/common.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

namespace flashinfer {
namespace fp4_sm90 {
namespace epi {

namespace tk = tensorrt_llm::kernels;
using Act = tensorrt_llm::kernels::cutlass_kernels::ActivationType;

#if defined(FLASHINFER_FP4_SM90_EXACT)
constexpr bool kExactFixed = true;
constexpr bool kExact = FLASHINFER_FP4_SM90_EXACT != 0;
#else
constexpr bool kExactFixed = false;
constexpr bool kExact = false;
#endif
#if defined(FP4_SM90_ACT)
constexpr int kActFixed = FP4_SM90_ACT;
#else
constexpr int kActFixed = -1;
#endif

// kRecipe* bits -> the quantizer's compile-time recipe (std::false_type: standard NVFP4).
template <uint8_t R>
using RecipeCfg =
    std::conditional_t<(R & kRecipe4over6) != 0,
                       tensorrt_llm::common::NVFP44Over6Config<
                           (R & kRecipeE4m3Max256) != 0 ? 256 : 448,
                           (R & kRecipeErrMse) != 0 ? tensorrt_llm::common::NVFP44Over6ErrMode::MSE
                                                    : tensorrt_llm::common::NVFP44Over6ErrMode::MAE,
                           (R & kRecipeFastMath) != 0>,
                       std::false_type>;

// E4M3 maximum of a recipe (NVFP4RecipeSpec::e4m3Max).
__host__ __device__ constexpr float recipe_e4m3_max(uint8_t r) {
  return (r & kRecipe4over6) != 0 && (r & kRecipeE4m3Max256) != 0 ? 256.f : 448.f;
}

// 16-bit output types.
template <class T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f<half>(half v) {
  return __half2float(v);
}
template <class T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ half from_f<half>(float v) {
  return __float2half_rn(v);
}
__device__ __forceinline__ uint16_t bits(__nv_bfloat16 v) { return __bfloat16_as_ushort(v); }
__device__ __forceinline__ uint16_t bits(half v) { return __half_as_ushort(v); }
// bits(from_f<T>(lo)) | bits(from_f<T>(hi)) << 16 in one cvt; the asm keeps the pair in one
// register instead of letting the compiler split it back into two 16-bit values.
template <class T>
__device__ __forceinline__ uint32_t bits2(float lo, float hi);
template <>
__device__ __forceinline__ uint32_t bits2<__nv_bfloat16>(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
template <>
__device__ __forceinline__ uint32_t bits2<half>(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
template <class T>
__device__ __forceinline__ float round_x(float v) {
  return to_f<T>(from_f<T>(v));
}
__device__ __forceinline__ float2 to_f2(__nv_bfloat162 v) { return __bfloat1622float2(v); }
__device__ __forceinline__ float2 to_f2(half2 v) { return __half22float2(v); }

// FP32 product with FTZ, as the -use_fast_math SM100+ modules compute it.
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
// max(x, 0) that keeps NaN (the per-token Relu2 reference).
__device__ __forceinline__ float max0_nan(float x) {
  float r;
  asm("max.NaN.f32 %0, %1, 0f00000000;" : "=f"(r) : "f"(x));
  return r;
}
// Activation functions of the cutlass_fused_moe adaptors (CUTLASS Sigmoid / SiLu / GELU /
// GELU_taylor<float>; expf as __expf, the -use_fast_math lowering).
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sigm(x); }
__device__ __forceinline__ float gelu(float x) {
  return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
}
__device__ __forceinline__ float gelu_taylor(float z) {
  constexpr float k0 = 0.7978845608028654f;
  constexpr float k1 = k0 * 0.044715f;
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(z * fmaf(k1 * z, z, k0)));
  return 0.5f * fmaf(z, t, z);
}
// tanh z as sigmoid(2 z) 2 - 1 (SituAdaptor).
__device__ __forceinline__ float tanh_sg(float z) { return sigm(z * 2.f) * 2.f + -1.f; }

// Per-expert activation parameters (ActivationParams of cutlass_fused_moe): x, y and z are
// FP32 [E_local] each; a null one keeps the adaptor default:
//   SwigluBias: alpha x 1, beta y 0, limit z inf     SwigluStep: limit z 7
//   PowLU: m x 2.5, limit z inf                      ClampedRelu2: limit z (required)
//   SiTU: beta x 4, linear beta y 25, limit z inf (TRT-LLM's: the gate from above, the linear
//   both ways)
// Swiglu, Geglu, GegluTanh and the plain activations take none (GLUAdaptor / IdentityAdaptor).
struct ActQ {
  float x, y, z;
};
__device__ __forceinline__ ActQ act_params(int act, const float* px, const float* py,
                                           const float* pz, int e) {
  ActQ q{1.f, 0.f, __int_as_float(0x7f800000)};
  if (act == int(Act::SwigluStep)) q.z = 7.f;
  if (act == int(Act::PowLU)) q.x = 2.5f;
  if (act == int(Act::Situ)) {
    q.x = 4.f;
    q.y = 25.f;
  }
  if (px != nullptr) q.x = px[e];
  if (py != nullptr) q.y = py[e];
  if (pz != nullptr) q.z = pz[e];
  return q;
}

// Gated activation of gate g and linear u. ACT < 0: the activation is read at run time.
template <int ACT>
__device__ __forceinline__ float act_gated(int act_rt, float g, float u, ActQ q) {
  const int act = ACT >= 0 ? ACT : act_rt;
  switch (act) {
    case int(Act::Swiglu):
      return silu(g) * u;
    case int(Act::Geglu):
      return gelu(g) * u;
    case int(Act::GegluTanh):
      return gelu_taylor(g) * u;
    case int(Act::SwigluBias): {
      const float gc = fminf(g, q.z);
      return gc * sigm(gc * q.x) * (fmaxf(fminf(u, q.z), -q.z) + q.y);
    }
    case int(Act::SwigluStep):
      return fminf(silu(g), q.z) * fmaxf(fminf(u, q.z), -q.z);
    case int(Act::PowLU): {
      const float s = 1.f / (1.f + __expf(-g));
      const float a = g > 0.f ? __powf(g, q.x / (sqrtf(g) + 1.f)) * s : g * s;
      return fminf(a, q.z) * fmaxf(fminf(u, q.z), -q.z);
    }
    case int(Act::Situ):
      if (q.z != __int_as_float(0x7f800000)) {
        g = fminf(g, q.z);
        u = fmaxf(fminf(u, q.z), -q.z);
      }
      return (tanh_sg(g * (1.f / q.x)) * sigm(g) * q.x) * (tanh_sg(u * (1.f / q.y)) * q.y);
    default:  // not a gated activation: the host rejects it
      return __int_as_float(0x7fffffff);
  }
}

// Non-gated activation; pt: the per-token reference (Relu2 keeps NaN).
template <int ACT>
__device__ __forceinline__ float act_plain(int act_rt, float g, ActQ q, bool pt) {
  const int act = ACT >= 0 ? ACT : act_rt;
  switch (act) {
    case int(Act::Gelu):
      return gelu(g);
    case int(Act::Relu):
      return fmaxf(g, 0.f);
    case int(Act::Silu):
      return silu(g);
    case int(Act::Identity):
      return g;
    case int(Act::Relu2): {
      const float r = pt ? max0_nan(g) : fmaxf(g, 0.f);
      return mul_ftz(r, r);
    }
    case int(Act::ClampedRelu2): {
      const float t = tanhf(fmaxf(g, 0.f) * (1.f / q.z)) * q.z;
      return t * t;
    }
    default:  // gated or invalid activation: the host rejects it
      return __int_as_float(0x7fffffff);
  }
}

template <class T>
__device__ __forceinline__ tk::PackedVec<T, 16> packed16(uint4 a, uint4 b) {
  tk::PackedVec<T, 16> v;
  uint32_t* w = reinterpret_cast<uint32_t*>(&v);
  w[0] = a.x;
  w[1] = a.y;
  w[2] = a.z;
  w[3] = a.w;
  w[4] = b.x;
  w[5] = b.y;
  w[6] = b.z;
  w[7] = b.w;
  return v;
}

// One 16-value NVFP4 block in one thread (cvt_warp_fp16_to_fp4 with 16 values a thread: no
// shuffles, and below SM100 bit-identical to the 8-value lane pairs). enc: encode scale;
// row_amax: the row's maximum |x| (per token) or 0. Writes code x SF as TO to o and the SF to sf.
template <uint8_t R, bool EXACT, class TI, class TO>
__device__ __forceinline__ uint64_t nvfp4_block(tk::PackedVec<TI, 16>& v, float enc, float row_amax,
                                                uint4 (&o)[2], uint8_t& sf) {
  sf = 0;
  const uint64_t c =
      tk::cvt_warp_fp16_to_fp4<TI, 16, 16, false, EXACT, RecipeCfg<R>>(v, enc, &sf, row_amax);
  const float s = tk::e4m3_to_fp32_sw(sf);
  uint32_t w[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const TO lo = from_f<TO>(tk::e2m1_to_fp32_sw(static_cast<uint32_t>(c >> (8 * i))) * s);
    const TO hi = from_f<TO>(tk::e2m1_to_fp32_sw(static_cast<uint32_t>(c >> (8 * i + 4))) * s);
    w[i] = bits(lo) | static_cast<uint32_t>(bits(hi)) << 16;
  }
  o[0] = make_uint4(w[0], w[1], w[2], w[3]);
  o[1] = make_uint4(w[4], w[5], w[6], w[7]);
  return c;
}

__device__ __forceinline__ void wg_bar(int id) {
  asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}
__device__ __forceinline__ void sts16(uint32_t addr, uint16_t v) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(v) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr)
               : "memory");
  return v;
}

// Factor still on the accumulators when they reach the epilogue. dequant.cuh: the GEMM multiplies
// them by 2^fold (dq_fold_log2_fmt) first, so 1; a Cfg that leaves the fold to the epilogue
// states it as kAccScale (folded into alpha, exact: a power of two).
template <class C, class = void>
struct AccScale {
  static constexpr float value = 1.f;
};
template <class C>
struct AccScale<C, std::void_t<decltype(C::kAccScale)>> {
  static constexpr float value = C::kAccScale;
};

// D7 staging chunk Nc in {32, 16, 8} (at most N): the largest whose CTA shared memory
// (1024 alignment + stages x (A 9216 (CTA_M / 128) + B 256 N + barrier 16) + CTA_M Nc 2) fits
// kSmemLimit at Cfg::kStages; Cfg::kStagingN overrides it. 0: none fits.
template <class C, class = void>
struct StagingN {
  static constexpr int bytes(int nc) {
    return 1024 + C::kStages * (9216 * (C::kCtaM / 128) + 256 * C::kN + 16) + C::kCtaM * nc * 2;
  }
  static constexpr int pick() {
    for (int nc = 32; nc >= 8; nc /= 2) {
      const int m = nc < C::kN ? nc : C::kN;
      if (bytes(m) <= kSmemLimit) return m;
    }
    return 0;
  }
  static constexpr int value = pick();
};
template <class C>
struct StagingN<C, std::void_t<decltype(C::kStagingN)>> {
  static constexpr int value = C::kStagingN;
};

}  // namespace epi

// Epilogue of one (m-tile, n-tile) for consumer warp group wg (warp in [0, 4), lane). acc is the
// wgmma m64nNk16 F32 fragment (params.cuh / K3 contract):
//   acc[(mi (kN / 8) + j) 4 + v], v0 = (row lane / 4, column 8 j + 2 (lane % 4)), v1 = column + 1,
//   v2 = row + 8, v3 = both; kernel row m0 + wg kCtaM / 2 + 64 mi + 16 warp + row, expanded row
//   (token) p = n0 + column, valid for p < n0 + n_valid. Gated GEMM1: rows lane / 4 and
//   lane / 4 + 8 of a 16-row group b are gate[i] and up[i], i = 8 b + lane / 4.
// Output column: gated i (m0 / 2 based), else the kernel row; rows past ASrc::rows are not
// written. out_tiled: GEMM2 B tile `tile` of out_ld K values (the output columns, or a localized
// GEMM1's full K with out at its first K tile), padding written as 0; the last m-tile also zeroes
// the K range past its columns. Otherwise row-major [p, column] (out_ld).
// Synchronization: named barrier 1 + wg (128 threads) only, for the NVFP4 staging.
template <class Cfg>
struct Epilogue {
  static constexpr Fq kFq = epi_fq(Cfg::kEpi);
  static constexpr uint8_t kRecipe = epi_recipe(Cfg::kEpi);
  static constexpr bool kGated = epi_gated(Cfg::kEpi);
  static constexpr bool kGemm2 = epi_gemm2(Cfg::kEpi);
  static constexpr bool kFqNvfp4 = kFq == Fq::kNvfp4;
  static_assert(kFq == Fq::kNone || kFqNvfp4, "MXFP8 / FP8 epilogue quantization is M6");
  static_assert(!kGemm2 || (!kGated && kFq == Fq::kNone), "GEMM2 writes plain rows");
  static_assert(!kFqNvfp4 || epi::kExactFixed,
                "the NVFP4 epilogue needs FLASHINFER_FP4_SM90_EXACT=0/1 (module builder)");

  static constexpr int kCtaM = Cfg::kCtaM;
  static constexpr int kN = Cfg::kN;
  static constexpr int kM64 = Cfg::kM64PerWarpGroup;
  static constexpr int kJ = kN / 8;
  static constexpr float kAccScale = epi::AccScale<Cfg>::value;
  // NVFP4 staging: per warp group kNc tokens x kWgCols output columns (16-bit), 16-byte units
  // XOR-swizzled by token.
  static constexpr int kWgCols = kGated ? kCtaM / 4 : kCtaM / 2;
  static constexpr int kUnits = kWgCols / 8;
  static constexpr int kBlocks = kWgCols / 16;
  static constexpr int kNc = kFqNvfp4 ? epi::StagingN<Cfg>::value : 0;
  static_assert(!kFqNvfp4 || (kNc >= 8 && kN % kNc == 0), "no D7 staging chunk fits");
  // CTA staging bytes (both warp groups), 16-byte aligned by the caller.
  static constexpr int kStagingBytes = kFqNvfp4 ? 2 * kWgCols * kNc * 2 : 0;

  __device__ static void apply(const float (&acc)[Cfg::kAccPerThread], const GemmArgs& args,
                               int expert, int m0, int tile, int n0, int n_valid, int wg, int warp,
                               int lane, uint32_t staging_smem) {
    if (args.epi.out_f16)
      run<half>(acc, args, expert, m0, tile, n0, n_valid, wg, warp, lane, staging_smem);
    else
      run<__nv_bfloat16>(acc, args, expert, m0, tile, n0, n_valid, wg, warp, lane, staging_smem);
  }

 private:
  template <class T>
  __device__ static void run(const float (&acc)[Cfg::kAccPerThread], const GemmArgs& args,
                             int expert, int m0, int tile, int n0, int n_valid, int wg, int warp,
                             int lane, uint32_t staging_smem) {
    const EpiArgs& ep = args.epi;
    // The host checks ep.act against the module's FP4_SM90_ACT; the NVFP4 GEMM1 output
    // quantization runs per tensor only (per-token recipes quantize in row_quant), so
    // tok_scale is null there. No __trap() in the consumer: it caps its registers at 168.
    const int rows = args.a.rows;
    const int ncols = kGated ? rows / 2 : rows;
    const int k_tiles = (static_cast<int>(ep.out_ld) + 127) / 128;  // out_tiled
    const int kp = (ncols + 127) / 128 * 128;
    const float alpha = ep.alpha != nullptr ? ep.alpha[expert] : 1.f;
    const float a = alpha * kAccScale;
    const bool pt = ep.tok_scale != nullptr;
    const int act = ep.act;
    const epi::ActQ q = kGemm2 ? epi::ActQ{1.f, 0.f, 0.f}
                               : epi::act_params(act, ep.act_x, ep.act_y, ep.act_z, expert);
    const void* const bias = ep.bias;
    const int64_t bias0 = static_cast<int64_t>(expert) * rows;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int p_end = n0 + n_valid;
    // Output column of mi = 0 (gated) or mi = 0, row lane / 4 (plain).
    const int col0 = kGated ? (m0 >> 1) + wg * (kCtaM / 4) + 8 * warp + lr
                            : m0 + wg * (kCtaM / 2) + 16 * warp + lr;
    char* const outb = static_cast<char*>(ep.out);

    // Bias of [mi][0] gate, [mi][1] up (gated) or [mi][half] for columns + 8 half (plain). BF16
    // or FP16 (ep.bias_f16): the logical rows, gated [up; gate] halves. TRT-LLM's
    // (ep.bias_trtllm): FP32 on the weight's rows (w3/w1 interleave, shuffle) in accumulator
    // units, so x alpha.
    float bb[kM64][2];
#pragma unroll
    for (int mi = 0; mi < kM64; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = kGated ? col0 + 32 * mi : col0 + 64 * mi + 8 * h;
        if (bias == nullptr || c >= ncols) {
          bb[mi][h] = 0.f;
        } else if (ep.bias_trtllm) {
          const int q = kGated ? 2 * c + (h == 0 ? 1 : 0) : c;
          bb[mi][h] = static_cast<const float*>(bias)[bias0 + trtllm_perm(q)] * alpha;
        } else {
          const int64_t o = bias0 + (kGated && h == 0 ? int64_t(ncols) + c : int64_t(c));
          bb[mi][h] = ep.bias_f16
                          ? epi::to_f<half>(static_cast<const half*>(bias)[o])
                          : epi::to_f<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(bias)[o]);
        }
      }

    // Value before the activation (rs: alpha x s of the row, per token).
    auto pre = [&](float v, float rs) {
      if constexpr (kGemm2) {
        return pt ? epi::mul_ftz(v * kAccScale, rs) : v * a;
      } else {
        if (pt) return epi::round_x<T>(epi::mul_ftz(v * kAccScale, rs));
        return kFqNvfp4 ? epi::round_x<T>(v * a) : v * a;
      }
    };
    auto add_bias = [&](float x, float b) { return bias != nullptr ? x + b : x; };
    auto gated_value = [&](float gv, float uv, float rs, int mi) {
      return epi::act_gated<epi::kActFixed>(act, add_bias(pre(gv, rs), bb[mi][0]),
                                            add_bias(pre(uv, rs), bb[mi][1]), q);
    };
    auto plain_value = [&](float v, float rs, int mi, int h) {
      const float x = add_bias(pre(v, rs), bb[mi][h]);
      if constexpr (kGemm2) return x;
      return epi::act_plain<epi::kActFixed>(act, x, q, pt);
    };
    // One 16-bit output at (column c, n-tile row r), times GEMM1's out_scale of the expert.
    const float os = !kGemm2 && ep.out_scale != nullptr ? __ldg(ep.out_scale + expert) : 1.f;
    auto put = [&](int c, int r, bool ok, float y) {
      if (ep.out_tiled) {
        if (c < kp)
          *reinterpret_cast<T*>(outb + b_tile_byte(tile, k_tiles, c >> 7, (c >> 6) & 1, kN, r,
                                                   c & 63)) = epi::from_f<T>(ok ? y * os : 0.f);
      } else if (ok) {
        static_cast<T*>(ep.out)[static_cast<int64_t>(n0 + r) * ep.out_ld + c] =
            epi::from_f<T>(y * os);
      }
    };

    if constexpr (!kFqNvfp4) {
#pragma unroll
      for (int j = 0; j < kJ; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = 8 * j + lc + h;
          const bool pv = n0 + r < p_end;
          const float rs = pt && pv ? epi::mul_ftz(alpha, ep.tok_scale[n0 + r]) : 0.f;
#pragma unroll
          for (int mi = 0; mi < kM64; ++mi) {
            if constexpr (kGated) {
              const int c = col0 + 32 * mi;
              const bool ok = pv && c < ncols;
              const float y = ok ? gated_value(acc[(mi * kJ + j) * 4 + h],
                                               acc[(mi * kJ + j) * 4 + 2 + h], rs, mi)
                                 : 0.f;
              put(c, r, ok, y);
            } else {
#pragma unroll
              for (int hh = 0; hh < 2; ++hh) {
                const int c = col0 + 64 * mi + 8 * hh;
                const bool ok = pv && c < ncols;
                const float y =
                    ok ? plain_value(acc[(mi * kJ + j) * 4 + 2 * hh + h], rs, mi, hh) : 0.f;
                put(c, r, ok, y);
              }
            }
          }
        }
    } else {
      // Stage kNc tokens of T(y) per chunk, then one thread per (token, 16-column block).
      const int wcol0 = kGated ? (m0 >> 1) + wg * (kCtaM / 4) : m0 + wg * (kCtaM / 2);
      const uint32_t st = staging_smem + wg * (kUnits * 16 * kNc);
      const int tid = warp * 32 + lane;
      auto put16 = [&](int c0, int r, bool ok, const uint4(&o)[2]) {
        if (ep.out_tiled) {
          if (c0 < kp) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              const int c = c0 + 8 * u;
              *reinterpret_cast<uint4*>(
                  outb + b_tile_byte(tile, k_tiles, c >> 7, (c >> 6) & 1, kN, r, c & 63)) = o[u];
            }
          }
        } else if (ok) {
          T* dst = static_cast<T*>(ep.out) + static_cast<int64_t>(n0 + r) * ep.out_ld + c0;
          if (c0 + 16 <= ncols && (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
            reinterpret_cast<uint4*>(dst)[0] = o[0];
            reinterpret_cast<uint4*>(dst)[1] = o[1];
          } else {
            const uint32_t w[8] = {o[0].x, o[0].y, o[0].z, o[0].w, o[1].x, o[1].y, o[1].z, o[1].w};
#pragma unroll
            for (int i = 0; i < 16; ++i)
              if (c0 + i < ncols) {
                const uint16_t b = static_cast<uint16_t>(w[i / 2] >> (16 * (i % 2)));
                reinterpret_cast<uint16_t*>(dst)[i] = b;
              }
          }
        }
      };
      // T(y) of every output this thread owns, tokens h = 0 / 1 packed in one register: made
      // before the chunk loop, so the fp32 accumulators are dead when the first chunk quantizes
      // and the staged values hold half their registers.
      uint32_t yp[kGated ? kM64 * kJ : kM64 * kJ * 2];
#pragma unroll
      for (int j = 0; j < kJ; ++j) {
        const bool pv0 = n0 + 8 * j + lc < p_end, pv1 = n0 + 8 * j + lc + 1 < p_end;
#pragma unroll
        for (int mi = 0; mi < kM64; ++mi) {
          const int a = (mi * kJ + j) * 4;
          if constexpr (kGated) {
            const bool ok = wcol0 + 32 * mi + 8 * warp + lr < ncols;
            yp[mi * kJ + j] =
                epi::bits2<T>(ok && pv0 ? gated_value(acc[a], acc[a + 2], 0.f, mi) : 0.f,
                              ok && pv1 ? gated_value(acc[a + 1], acc[a + 3], 0.f, mi) : 0.f);
          } else {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const bool ok = wcol0 + 64 * mi + 16 * warp + lr + 8 * hh < ncols;
              yp[(mi * kJ + j) * 2 + hh] =
                  epi::bits2<T>(ok && pv0 ? plain_value(acc[a + 2 * hh], 0.f, mi, hh) : 0.f,
                                ok && pv1 ? plain_value(acc[a + 2 * hh + 1], 0.f, mi, hh) : 0.f);
            }
          }
        }
      }
#pragma unroll
      for (int ch = 0; ch < kN / kNc; ++ch) {
        epi::wg_bar(1 + wg);  // the previous chunk (or tile) has been read
#pragma unroll
        for (int jj = 0; jj < kNc / 8; ++jj) {
          const int j = ch * (kNc / 8) + jj;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tl = 8 * jj + lc + h;
            const uint32_t row = st + tl * (kUnits * 16);
#pragma unroll
            for (int mi = 0; mi < kM64; ++mi) {
#pragma unroll
              for (int hh = 0; hh < (kGated ? 1 : 2); ++hh) {
                const int lcol =
                    kGated ? 32 * mi + 8 * warp + lr : 64 * mi + 16 * warp + lr + 8 * hh;
                const int i = kGated ? mi * kJ + j : (mi * kJ + j) * 2 + hh;
                epi::sts16(row + (((lcol >> 3) ^ (tl % kUnits)) << 4) + (lcol & 7) * 2,
                           static_cast<uint16_t>(yp[i] >> (16 * h)));
              }
            }
          }
        }
        epi::wg_bar(1 + wg);
        const float fq_global =
            ep.fq_global != nullptr ? __ldg(ep.fq_global + (ep.fq_per_expert ? expert : 0)) : 1.f;
        for (int it = tid; it < kNc * kBlocks; it += 128) {
          const int tl = it / kBlocks, blk = it % kBlocks;
          const int r = ch * kNc + tl, c0 = wcol0 + 16 * blk;
          const bool ok = n0 + r < p_end && c0 < ncols;
          uint4 o[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
          if (ok) {
            const uint32_t row = st + tl * (kUnits * 16);
            const int x = tl % kUnits;
            auto v = epi::packed16<T>(epi::lds128(row + (((2 * blk) ^ x) << 4)),
                                      epi::lds128(row + (((2 * blk + 1) ^ x) << 4)));
            uint8_t sf;
            epi::nvfp4_block<kRecipe, epi::kExact, T, T>(v, fq_global, 0.f, o, sf);
          }
          put16(c0, r, ok, o);
        }
      }
    }

    // The last m-tile zeroes GEMM2's K padding past the columns any m-tile covers (whole 64-value
    // halves: kN x 128 contiguous bytes each). Both warp groups share it.
    if (ep.out_tiled) {
      const int cover_end = kGated ? (m0 + kCtaM) >> 1 : m0 + kCtaM;
      if (m0 + kCtaM >= rows && cover_end < kp) {
        const int t256 = wg * 128 + warp * 32 + lane;
        for (int hf = cover_end >> 6; hf < kp >> 6; ++hf) {
          uint4* dst = reinterpret_cast<uint4*>(
              outb + b_tile_byte(tile, k_tiles, hf >> 1, hf & 1, kN, 0, 0));
          for (int i = t256; i < kN * 8; i += 256) dst[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }
  }
};

}  // namespace fp4_sm90
}  // namespace flashinfer
