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
// Shared interface of the SM90 (wgmma) FP4 GEMM used by the FP4 MoE and dense entry points.
//
// SM90 has no public weight format of its own: every weight layout below is one that an
// SM100+ entry point already accepts, read as is. Swap-AB: weights are the wgmma A operand
// (M = weight rows, dequantized in registers), tokens are the B operand (N, BF16/FP16 in smem).
//
// The integer values of every enum here are mirrored in flashinfer/fused_moe/sm90/spec.py.
#pragma once

#include <cstdint>

namespace flashinfer {
namespace fp4_sm90 {

// SM90 opt-in dynamic shared memory per block: the configuration limit of every kernel here.
constexpr int kSmemLimit = 232448;

// ---------------------------------------------------------------------------------------------
// Weight (A) byte layouts. Per expert e, row r (physical), FP4 element k:
//   code byte  = codes + e * codes_expert_stride + r * (k_total / 2) + k / 2
//                (even k = low nibble, odd k = high nibble)
//   SF128x4(r, c; R, C) = (r / 128) * ceil(C / 4) * 512 + (c / 4) * 512 + (r % 32) * 16
//                         + ((r % 128) / 32) * 4 + c % 4
//   with C = ceil(k_total / sf_block). The 128-row tile pitch is ceil(C / 4) * 512 bytes, i.e.
//   it uses the column count padded to 4, and the expert stride is pad128(R) * pad4(C).
// Which physical row holds logical row q is given by Pair (below).
// ---------------------------------------------------------------------------------------------
enum class AFmt : uint8_t {
  // TRTLLM `trtllm_fp4_block_scale_moe`: rows permuted by P(q) = (q/32)*32 + 8*(q%4) + (q%32)/4,
  // E4M3 SF (16 values) 128x4-swizzled on the permuted rows.
  kNvfp4TrtllmShuffled = 0,
  // Rows in logical (or Pair-interleaved) order, E4M3 SF (16 values) 128x4-swizzled.
  // cutlass_fused_moe; CuTe DSL MoE (its "MMA layout" SF is a view of these same bytes);
  // dense cutlass / cudnn / cute-dsl SM10x (`mm_fp4`, `mm_bf16_fp4` cute-dsl SM10x prep).
  kNvfp4Swz128x4 = 1,
  // codes [N, K/2], E4M3 SF linear [N, K/16] (dense `mm_bf16_fp4` cudnn, blackwell-native).
  kNvfp4LinearSf = 2,
  // MXFP4 (E8M0 SF, 32 values) with the TRTLLM row permutation. Not a 4over6 layout (M6).
  kMxfp4TrtllmShuffled = 3,
  // MXFP4 (E8M0 SF, 32 values) codes in row order, SF 128x4-swizzled (dense cudnn / cute-dsl /
  // cutedsl_low_latency / b12x MXFP4).
  kMxfp4Swz128x4 = 4,
  // Dense cute-dsl SM12x prep (also blackwell-tiled): int32 codes (K/16, 2N) and
  // S0E5M3 SF (K/16, N). See flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py `_prepare_cute_dsl_sm12x`.
  kDenseCuteSm12x = 5,
};

enum class SfKind : uint8_t { kE4M3 = 0, kUE8M0 = 1, kS0E5M3 = 2 };

__host__ __device__ constexpr SfKind sf_kind(AFmt f) {
  return (f == AFmt::kMxfp4TrtllmShuffled || f == AFmt::kMxfp4Swz128x4) ? SfKind::kUE8M0
         : f == AFmt::kDenseCuteSm12x                                   ? SfKind::kS0E5M3
                                                                        : SfKind::kE4M3;
}

// Number of FP4 values that share one scale byte.
__host__ __device__ constexpr int sf_block(AFmt f) {
  return sf_kind(f) == SfKind::kUE8M0 ? 32 : 16;
}

// ---------------------------------------------------------------------------------------------
// Gate/up pairing of the gated GEMM1 weight (logical row = (h, i), h = 0 up, h = 1 gate,
// i in [0, half_rows)). Non-gated weights and GEMM2 use kNone (logical row = physical row,
// before the AFmt row permutation).
// Inside the kernel a warp's 16-row group holds gate[i] at row g and up[i] at row g + 8
// (i = 8 * group + g); the A loader maps from the native rows below to that order.
// ---------------------------------------------------------------------------------------------
enum class Pair : uint8_t {
  kNone = 0,
  // [up; gate] halves: physical row = h * half_rows + i (cutlass_fused_moe).
  kUpGateHalves = 1,
  // TRTLLM w3/w1 interleave: q = 2 * i + h, then the AFmt row permutation P(q).
  // Shared by kNvfp4TrtllmShuffled and kMxfp4TrtllmShuffled.
  kTrtllmW31 = 2,
  // CuTe DSL group interleave of [up; gate] with group size g (= ASrc::pair_g; 64 by default,
  // 16 for FI swap_ab): s = (r % 2g) / g (0 up, 1 gate), i = (r / 2g) * g + r % g.
  kGroupInterleave = 3,
};

// Activation (B) source formats accepted by the prep kernel. The GEMM itself always reads
// BF16/FP16 values (exact dequantized values for FP4 inputs).
enum class XFmt : uint8_t {
  kBf16 = 0,
  kF16 = 1,
  kNvfp4Linear = 2,    // codes [T, K/2], E4M3 SF linear [T, K/16]
  kNvfp4Swz128x4 = 3,  // codes [T, K/2], E4M3 SF 128x4-swizzled
  kNvfp4Swz8x4 = 4,    // codes [T, K/2], E4M3 SF 8x4-swizzled
  kMxfp4Swz128x4 = 5,  // codes [T, K/2], UE8M0 SF (32 values) 128x4-swizzled
};

// ---------------------------------------------------------------------------------------------
// Epilogue code (Cfg::kEpi). Compile-time part of the epilogue; runtime values are in EpiArgs.
//   bits 0-1  Fq     output quantization of the GEMM result
//   bits 2-5  recipe NVFP4 recipe of that quantization (kRecipe* bits)
//   bit  6    gated  GEMM1 with a gate/up activation
//   bit  7    gemm2  GEMM2 (writes per (token, slot) rows for the separate finalize)
// ---------------------------------------------------------------------------------------------
enum class Fq : uint8_t {
  kNone = 0,   // write BF16/FP16 (token-wise recipes quantize later in row_quant)
  kNvfp4 = 1,  // quantize to NVFP4 (16 values, E4M3 SF) and write the exact dequantized value
  kMxfp8 = 2,  // M6
  kFp8 = 3,    // M6
};

// Recipe bits, mapping to tensorrt_llm::common::NVFP44Over6Config<E4M3_MAX, ERR_MODE, FAST_MATH>.
constexpr uint8_t kRecipe4over6 = 1u << 0;      // 0: standard NVFP4 quantization
constexpr uint8_t kRecipeE4m3Max256 = 1u << 1;  // E4M3_MAX 256 (else 448)
constexpr uint8_t kRecipeErrMse = 1u << 2;      // NVFP44Over6ErrMode::MSE (else MAE)
constexpr uint8_t kRecipeFastMath = 1u << 3;    // ERR_USE_FAST_MATH
#if defined(FP4_SM90_W4A16)
// EpiArgs::w4a16_flags (the host's w4a16_flags).
constexpr uint8_t kW4a16SwigluParams = 1u << 0;  // SwiGLU with alpha / beta / limit
constexpr uint8_t kW4a16SituLinear = 1u << 1;    // SiTU with a linear beta
// use_fused_finalize: GEMM2 rows scaled by alpha x their routing weight, summed in BF16
constexpr uint8_t kW4a16FusedFinalize = 1u << 2;
#endif

__host__ __device__ constexpr int epi_code(Fq fq, uint8_t recipe, bool gated, bool gemm2) {
  return int(fq) | (int(recipe & 0xF) << 2) | (gated ? 1 << 6 : 0) | (gemm2 ? 1 << 7 : 0);
}
__host__ __device__ constexpr Fq epi_fq(int epi) { return Fq(epi & 3); }
__host__ __device__ constexpr uint8_t epi_recipe(int epi) { return uint8_t((epi >> 2) & 0xF); }
__host__ __device__ constexpr bool epi_gated(int epi) { return (epi >> 6) & 1; }
__host__ __device__ constexpr bool epi_gemm2(int epi) { return (epi >> 7) & 1; }

// ---------------------------------------------------------------------------------------------
// Runtime arguments.
// ---------------------------------------------------------------------------------------------
struct ASrc {
  const uint8_t* codes;         // FP4 codes (int32-packed for kDenseCuteSm12x, same bytes)
  const uint8_t* sf;            // scale bytes (E4M3 / UE8M0 / S0E5M3)
  int64_t codes_expert_stride;  // bytes; 0 for dense
  int64_t sf_expert_stride;     // bytes; 0 for dense. 128x4: pad128(rows_phys) * pad4(sf_cols)
  int rows;                     // logical rows M (gated: 2 * half_rows)
  int rows_phys;                // physical rows per expert (padding included)
  int k;                        // logical K (FP4 values per row)
  int k_phys;                   // physical K per row (padding included); row bytes = k_phys / 2
  int half_rows;                // gated: rows of one half (physical, padding included); else 0
  int pair_g;                   // Pair::kGroupInterleave group size; else 0
  int num_experts;              // experts addressed by the codes / sf pointers (dense: 1)
  AFmt fmt;
  Pair pair;
};

struct BSrc {
  const void* x;  // BF16/FP16 activations: [tokens, k] (tiled == 0) or B tiles
  int tokens;     // expanded rows (MoE: tokens * top_k after routing) or dense M
  int k;
  int64_t x_ld;   // elements between rows of x (tiled == 0)
  uint8_t tiled;  // 1: x is written by prep / row_quant in expert order as SW128 tiles
};

struct EpiArgs {
  const float* alpha;      // per expert (MoE) or scalar (dense) fp32: weight global x input global
  const float* tok_scale;  // per expanded row fp32 (token-wise recipes); null = none
  // Activation parameters, FP32 [E_local] each (cutlass_fused_moe's SwiGLU alpha / beta /
  // limit, SiTU beta / linear beta); null = the activation's default
  const float* act_x;
  const float* act_y;
  const float* act_z;
  const void* bias;        // per expert [rows], BF16 or FP16 (bias_f16) or FP32 (bias_trtllm);
                           // null = none
  void* out;               // output
  int64_t out_ld;          // elements between output rows
  const float* out_scale;  // per expert fp32 factor of 16-bit output rows (TRT-LLM's per-token
                           // FC1 rows carry output1_scale_scalar, as SM100); null = 1
  const float* fq_global;  // device fp32 scalar or per expert (fq_per_expert): global scale of
                           // the epilogue quantization (Fq != kNone); null = 1
  // TRT-LLM's gemm1_lora_delta, read by the gated GEMM1 with the NVFP4 quantization only: BF16
  // rows of lora_ld values by token * top_k + slot (lora_row[output row]), [up; gate] lora_half
  // apart, added to the activation's inputs; null = none. act_out: the activation (before the
  // quantization) as rows of act_out_ld values by output row; null = none.
  const void* lora;
  const int32_t* lora_row;
  int64_t lora_ld;
  int64_t lora_half;
  void* act_out;
  int64_t act_out_ld;
  uint8_t act;            // flashinfer.tllm_enums.ActivationType value
  uint8_t fq;             // Fq (same as epi_fq(Cfg::kEpi); kept for runtime checks)
  uint8_t recipe;         // kRecipe* bits (same as epi_recipe(Cfg::kEpi))
  uint8_t out_f16;        // 1: FP16 output, 0: BF16 output
  uint8_t out_tiled;      // 1: write GEMM2 input tiles (expert order, SW128); 0: row-major
  uint8_t bias_f16;       // 1: FP16 bias, 0: BF16 bias
  uint8_t bias_trtllm;    // 1: TRT-LLM's FP32 bias on the weight's rows, accumulator units
  uint8_t fq_per_expert;  // 1: fq_global is [E_local] (TRT-LLM's per-expert scalars)
#if defined(FP4_SM90_W4A16)
  // W4A16 modules: the activation constants of the host's w4a16_consts, by value (SwiGLU:
  // -alpha log2(e), beta, limit; SiTU: beta, 1 / beta, linear beta, 1 / linear beta), and
  // w4a16_flags (kW4a16SwigluParams, kW4a16SituLinear, kW4a16FusedFinalize).
  float w4a16_c[4];
  uint8_t w4a16_flags;
#endif
};

// Persistent schedule written by prep. Expanded rows p are sorted by local expert; expert e owns
// p in [expert_offset[e], expert_offset[e + 1]). Its rows are cut into n-tiles of Cfg::kN rows;
// n-tile t = expert_tile_base[e] + (p - expert_offset[e]) / kN, listed in (tile_expert, tile_n0).
// The GEMM walks t * m_tiles + m_tile with m_tiles = ceil(ASrc::rows / kCtaM). Both GEMMs of one
// MoE call share kN and this list, unless their n differ: then each GEMM gets the list of its own
// n (PrepArgs::n2 for GEMM2) over the same expert_offset.
struct Sched {
  const int32_t* tile_expert;       // [max_tiles] local expert of n-tile t; null: dense (expert 0)
  const int32_t* tile_n0;           // [max_tiles] first expanded row of n-tile t; null: t * kN
  const int32_t* num_tiles;         // device scalar: number of n-tiles; null: max_tiles
  const int32_t* expert_offset;     // [E_local + 1]; null: dense (rows [0, BSrc::tokens))
  const int32_t* expert_tile_base;  // [E_local + 1]; null: dense (t = p / kN)
  int max_tiles;
};

// B tile layout (BSrc::tiled == 1, EpiArgs::out_tiled == 1, prep and row_quant outputs):
// n-tile t, K tile kt (128 values), half kh (64 values), row r in [0, kN), value c in [0, 64):
//   byte = ((t * k_tiles + kt) * 2 + kh) * kN * 128 + r * 128 + (((c / 8) ^ (r % 8)) * 16) + (c %
//   8) * 2
// i.e. one stage of B is contiguous (kN * 256 bytes) and each 64-value half is the wgmma K-major
// 128B-swizzle layout. Rows past the expert's count and values past K are written as 0.
__host__ __device__ constexpr int64_t b_tile_byte(int t, int k_tiles, int kt, int kh, int n, int r,
                                                  int c) {
  return ((int64_t(t) * k_tiles + kt) * 2 + kh) * n * 128 + r * 128 + (((c / 8) ^ (r % 8)) * 16) +
         (c % 8) * 2;
}

// GEMM1 of a MoE call whose GEMMs differ in n (Cfg::kCrossN): the epilogue writes GEMM2's B tiles,
// n2 = 1 << n2_shift rows each, directly. Output row p of expert e is row off = p -
// expert_offset[e] of the expert: row off % n2 of n2-tile expert_tile_base2[e] + off / n2.
struct CrossN {
  const int32_t* expert_tile_base2;  // [E_local + 1] first n2-tile of each expert
  int n2_shift;                      // log2(n2)
};

struct GemmArgs {
  ASrc a;
  BSrc b;
  EpiArgs epi;
  Sched sched;
  // Last, so that no other member and no later kernel parameter (the tensor maps) moves.
  CrossN cross;
};
static_assert(sizeof(GemmArgs) <= 320, "GemmArgs must keep the offset of the kernels' tensor maps");

// Routing sort, tile list, activation quantization / dequantization (K1 prep.cuh).
// Dense: topk_ids == null, top_k == 1, one expert.
struct PrepArgs {
  const int32_t* topk_ids;  // [tokens, top_k]
  int tokens;
  int top_k;
  int num_experts;
  int local_expert_offset;
  int num_local_experts;
  // activation input
  const void* x;        // [tokens, k] in x_fmt (FP4: packed codes [tokens, k / 2])
  const uint8_t* x_sf;  // FP4 x_fmt only
  int x_sf_cols;        // FP4 x_fmt only: scale columns of an x row (pitch, >= k / block)
  int64_t x_ld;         // elements (BF16/FP16) or bytes (FP4) between rows
  int k;
  XFmt x_fmt;
  const float* x_global;  // input global scale used by in-prep quantization; null = none
  uint8_t quantize;       // 1: quantize BF16/FP16 x with `recipe`, keep the dequantized value
  uint8_t per_token;      // 1: per-token global scale (written to tok_scale)
  uint8_t recipe;         // kRecipe* bits
  uint8_t f16;            // B element type FP16 (else BF16)
  int n;                  // kN of the GEMMs (n-tile size)
  int max_tiles;
  uint8_t x_global_per_expert;  // 1: x_global per local expert (per tensor, MoE only)
  // outputs
  int32_t* expert_offset;     // [num_local_experts + 1]
  int32_t* expert_tile_base;  // [num_local_experts + 1]
  int32_t* permuted_src;      // [tokens * top_k] expanded row -> token * top_k + slot
  int32_t* src_permuted;  // [tokens * top_k] token * top_k + slot -> expanded row (-1: not local)
  int32_t* tile_expert;   // [max_tiles]
  int32_t* tile_n0;       // [max_tiles]
  int32_t* num_tiles;     // device scalar
  void* x_tiles;          // B tiles of GEMM1 (layout above); null when GEMM1 gathers x directly
  float* tok_scale;       // [tokens * top_k] by expanded row (per_token); null = none
  void* workspace;        // scratch owned by prep
  // GEMM2's own n-tile list when its n (n2) differs from n (MoE only); n2 = 0: GEMM2 shares n and
  // the list above, and prep does no more.
  int n2;
  int max_tiles2;
  int32_t* expert_tile_base2;  // [num_local_experts + 1]
  int32_t* tile_expert2;       // [max_tiles2]
  int32_t* tile_n02;           // [max_tiles2]
  int32_t* num_tiles2;         // device scalar
  // Per tensor: GEMM2's B tiles (n2 rows each, act_k values per row), which GEMM1 writes row by
  // row; prep zeroes their padding rows, the rows past an expert's count in its last n2-tile.
  // Null: none (per token, row_quant writes those tiles whole).
  void* act_tiles;
  int act_k;
};

// Token-wise quantization of the GEMM2 input after the GEMM1 epilogue (K3 row_quant.cuh).
struct RowQuantArgs {
  const void* in;  // GEMM1 activation output [expanded rows, cols], BF16/FP16 row-major
  int64_t in_ld;
  int rows;
  int cols;
  uint8_t in_f16;
  uint8_t recipe;  // kRecipe* bits
  uint8_t out_f16;
  int n;             // kN (tile size of the GEMM2 B tiles)
  const float* gsi;  // inverse global scale (device scalar); null = the recipe's 1 / (6 e4m3Max)
  const int32_t* expert_offset;
  const int32_t* expert_tile_base;
  const int32_t* num_tiles;
  void* out_tiles;   // GEMM2 B tiles (layout above)
  float* tok_scale;  // [rows] per expanded row
  // Optional caller copy of the quantized input in (token, slot) order (fc2_input_out, MoE only;
  // null = none): codes [rows, fq_cols / 2], linear E4M3 scales [rows, fq_cols / 16] and token
  // scales [rows], fq_cols >= cols with the columns past cols zero; rows of other ranks' experts
  // all zero.
  uint8_t* fq_codes;
  uint8_t* fq_sf;
  float* fq_tok;
  const int32_t* fq_src;  // [rows] expanded row -> token * top_k + slot (PrepArgs::permuted_src)
  const int32_t* fq_dst;  // [rows] token * top_k + slot -> expanded row, -1: not local
  int fq_cols;
};

// Ordered top-k sum (D6): out[token] = sum_{slot = 0..top_k-1} w[token, slot] * y[row(token,
// slot)].
struct FinalizeArgs {
  const void* y;  // GEMM2 output [expanded rows, hidden]
  int64_t y_ld;
  uint8_t y_dtype;  // 0 fp32, 1 bf16, 2 fp16
  const int32_t* src_permuted;
  const void* topk_weights;  // [tokens, top_k]; null = 1
  uint8_t w_dtype;           // 0 fp32, 1 bf16, 2 fp16
  void* out;                 // [tokens, hidden]
  int64_t out_ld;
  uint8_t out_f16;
  int tokens;
  int top_k;
  int hidden;
  uint8_t keep_unrouted;  // 1: tokens without a local row keep their out row
#if defined(FP4_SM90_W4A16)
  uint8_t fused;  // kW4a16FusedFinalize: y rows carry their weights, summed in BF16
#endif
};

// ---------------------------------------------------------------------------------------------
// Compile-time configuration. D7: CTA_M in {128, 256}, N in {8, ..., 256},
// accumulators per consumer thread (CTA_M / 128) * N / 2 <= 128, STAGES >= 2.
// APath selects the internal A path compared at the first build (D1): kN1 consumers fragment
// the native stage directly, kN2 the producer re-stages it into the 64-row x K128 block layout.
// F16 is the wgmma operand type (FP16 instead of BF16); the output dtype is EpiArgs::out_f16.
// Named barriers 1 and 2 are reserved for the epilogue of consumer warp groups 0 and 1.
// ---------------------------------------------------------------------------------------------
enum class APath : uint8_t { kN1 = 0, kN2 = 1 };

// CROSS_N_: GEMM1 writes GEMM2's B tiles of another n (CrossN); a separate instance, so that a
// call whose GEMMs share n runs the kernels without it.
template <int CTA_M_, int N_, int STAGES_, AFmt AF_, bool F16_, int EPI_, APath AP_ = APath::kN1,
          bool CROSS_N_ = false>
struct Cfg {
  static constexpr int kCtaM = CTA_M_;
  static constexpr int kN = N_;
  static constexpr int kStages = STAGES_;
  static constexpr AFmt kAFmt = AF_;
  static constexpr bool kF16 = F16_;
  static constexpr int kEpi = EPI_;
  static constexpr APath kAPath = AP_;
  static constexpr bool kCrossN = CROSS_N_;

  static constexpr int kTileK = 128;  // 8 x k16 per stage
  static constexpr int kConsumerWarpGroups = 2;
  static constexpr int kThreads = 128 * (1 + kConsumerWarpGroups);  // 384
  static constexpr int kM64PerWarpGroup = kCtaM / 128;
  static constexpr int kAccPerThread = kM64PerWarpGroup * kN / 2;
  static constexpr int kSfBlock = sf_block(AF_);
  static constexpr int kProducerRegs = 40;
  static constexpr int kConsumerRegs = 232;

  static_assert(kCtaM == 128 || kCtaM == 256, "CTA_M must be 128 or 256");
  static_assert(kN >= 8 && kN <= 256 && (kN & (kN - 1)) == 0,
                "N must be a power of two in [8, 256]");
  static_assert(kAccPerThread <= 128, "accumulators per consumer thread exceed 128 (D7)");
  static_assert(kStages >= 2, "at least two stages");
  static_assert(kProducerRegs * 128 + kConsumerRegs * 256 <= 65536, "register file");
};

}  // namespace fp4_sm90
}  // namespace flashinfer
