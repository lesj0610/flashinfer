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
// SM90 (wgmma) FP4 MoE: prep -> GEMM1 -> [row_quant, token-wise recipes] -> GEMM2 -> finalize.
//
// One module per (recipe, exact, activation, f16), see gen_fp4_moe_sm90_module. The GEMM kernel
// table (D7) is split over kNumParts translation units: a part (FP4_MOE_SM90_PART defined)
// only instantiates run_gemm<variant, APath>; the main unit holds prep, row_quant, finalize and
// the TVM-FFI binding.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "fp4_sm90/kernel.cuh"
#include "fp4_sm90/params.cuh"

#if !defined(FP4_SM90_RECIPE) || !defined(FP4_SM90_ACT) || !defined(FP4_SM90_GATED) || \
    !defined(FP4_SM90_F16)
#error "fp4_moe_sm90.cu is built by gen_fp4_moe_sm90_module (module defines missing)"
#endif

namespace flashinfer {
namespace fp4_sm90 {
namespace moe {

constexpr uint8_t kModuleRecipe = FP4_SM90_RECIPE;
constexpr int kModuleAct = FP4_SM90_ACT;
constexpr bool kModuleGated = FP4_SM90_GATED != 0;
constexpr bool kModuleF16 = FP4_SM90_F16 != 0;
constexpr int kActIdentity = 9;  // ActivationType.Identity
constexpr int kMaxStages = 32;

// GEMM variants: compile-time weight format and epilogue.
enum Variant : int {
  kG1TokenTrtllm = 0,   // GEMM1 -> BF16 rows (token-wise row_quant next), TRTLLM-shuffled weights
  kG1TokenSwz = 1,      // same, 128x4 weights (CuTe DSL)
  kG1TensorSwz = 2,     // GEMM1 -> NVFP4 epilogue quantization into GEMM2 B tiles (cutlass)
  kG2Trtllm = 3,        // GEMM2, TRTLLM-shuffled weights
  kG2Swz = 4,           // GEMM2, 128x4 weights
  kG1TensorTrtllm = 5,  // GEMM1 -> NVFP4 epilogue quantization, TRTLLM-shuffled weights (trtllm)
  kNumVariants = 6,
};
constexpr int kNumParts = kNumVariants * 2;  // x APath; FP4_MOE_SM90_PARTS in Python

template <int V>
struct VariantOf {
  static constexpr AFmt kA = (V == kG1TokenTrtllm || V == kG2Trtllm || V == kG1TensorTrtllm)
                                 ? AFmt::kNvfp4TrtllmShuffled
                                 : AFmt::kNvfp4Swz128x4;
  static constexpr bool kGemm2 = V == kG2Trtllm || V == kG2Swz;
  static constexpr Fq kFq = (V == kG1TensorSwz || V == kG1TensorTrtllm) ? Fq::kNvfp4 : Fq::kNone;
  // TRT-LLM's per-tensor path has no 4over6 recipe (as SM100): no kernels in recipe modules.
  static constexpr bool kBuilt = !(V == kG1TensorTrtllm && kModuleRecipe != 0);
  static constexpr int kEpi =
      epi_code(kFq, kFq == Fq::kNone ? 0 : kModuleRecipe, !kGemm2 && kModuleGated, kGemm2);
};

// D7 shapes: CTA_M in {128, 256} x N in {8, ..., 256}, (CTA_M / 128) * N / 2 <= 128.
struct Shape {
  int cta_m, n;
};
constexpr Shape kShapes[] = {{128, 8}, {128, 16}, {128, 32}, {128, 64}, {128, 128}, {128, 256},
                             {256, 8}, {256, 16}, {256, 32}, {256, 64}, {256, 128}};
constexpr int kNumShapes = sizeof(kShapes) / sizeof(kShapes[0]);

// Largest stage count whose shared memory fits; 0 when fewer than 2 fit (shape dropped, D7).
template <int CM, int N, int V, APath AP, int S = kMaxStages>
constexpr int max_stages() {
  if constexpr (S < 2 || !VariantOf<V>::kBuilt) {
    return 0;
  } else if constexpr (fp4_sm90_gemm_smem_bytes<
                           Cfg<CM, N, S, VariantOf<V>::kA, kModuleF16, VariantOf<V>::kEpi, AP>>() <=
                       size_t(kSmemLimit)) {
    return S;
  } else {
    return max_stages<CM, N, V, AP, S - 1>();
  }
}

template <int V, APath AP, int I = 0>
void stage_table(int* out) {
  if constexpr (I < kNumShapes) {
    out[I] = max_stages<kShapes[I].cta_m, kShapes[I].n, V, AP>();
    stage_table<V, AP, I + 1>(out);
  }
}

template <int V, APath AP, int I = 0>
int run_gemm_shape(const GemmArgs& args, int cta_m, int n, int num_sms, cudaStream_t stream) {
  if constexpr (I == kNumShapes) {
    return 0;
  } else {
    constexpr int CM = kShapes[I].cta_m, N = kShapes[I].n;
    constexpr int S = max_stages<CM, N, V, AP>();
    if (cta_m == CM && n == N) {
      if constexpr (S >= 2) {
        launch_fp4_sm90_gemm<Cfg<CM, N, S, VariantOf<V>::kA, kModuleF16, VariantOf<V>::kEpi, AP>>(
            args, num_sms, stream);
      }
      return S;
    }
    return run_gemm_shape<V, AP, I + 1>(args, cta_m, n, num_sms, stream);
  }
}

// Launches the (cta_m, n) kernel of a variant; returns its stage count (0: not instantiated).
template <int V, APath AP>
int run_gemm(const GemmArgs& args, int cta_m, int n, int num_sms, cudaStream_t stream) {
  return run_gemm_shape<V, AP>(args, cta_m, n, num_sms, stream);
}

}  // namespace moe
}  // namespace fp4_sm90
}  // namespace flashinfer

#if defined(FP4_MOE_SM90_PART)

static_assert(FP4_MOE_SM90_PART >= 0 && FP4_MOE_SM90_PART < flashinfer::fp4_sm90::moe::kNumParts,
              "FP4_MOE_SM90_PART out of range");
template int flashinfer::fp4_sm90::moe::run_gemm<
    (FP4_MOE_SM90_PART) / 2, static_cast<flashinfer::fp4_sm90::APath>((FP4_MOE_SM90_PART) % 2)>(
    const flashinfer::fp4_sm90::GemmArgs&, int, int, int, cudaStream_t);

#else  // main unit

#include "fp4_sm90/finalize.cuh"
#include "fp4_sm90/prep.cuh"
#include "fp4_sm90/row_quant.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace flashinfer {
namespace fp4_sm90 {
namespace moe {

#if defined(FP4_MOE_SM90_SPLIT) && FP4_MOE_SM90_SPLIT
#define FP4_MOE_SM90_EXTERN(V, AP) \
  extern template int run_gemm<V, AP>(const GemmArgs&, int, int, int, cudaStream_t);
#define FP4_MOE_SM90_EXTERN_V(V)     \
  FP4_MOE_SM90_EXTERN(V, APath::kN1) \
  FP4_MOE_SM90_EXTERN(V, APath::kN2)
FP4_MOE_SM90_EXTERN_V(0)
FP4_MOE_SM90_EXTERN_V(1)
FP4_MOE_SM90_EXTERN_V(2)
FP4_MOE_SM90_EXTERN_V(3)
FP4_MOE_SM90_EXTERN_V(4)
FP4_MOE_SM90_EXTERN_V(5)
#undef FP4_MOE_SM90_EXTERN_V
#undef FP4_MOE_SM90_EXTERN
#endif

int dispatch_gemm(int variant, APath ap, const GemmArgs& args, int cta_m, int n, int num_sms,
                  cudaStream_t stream) {
#define FP4_MOE_SM90_CASE(V)                                                           \
  case V:                                                                              \
    return ap == APath::kN1 ? run_gemm<V, APath::kN1>(args, cta_m, n, num_sms, stream) \
                            : run_gemm<V, APath::kN2>(args, cta_m, n, num_sms, stream);
  switch (variant) {
    FP4_MOE_SM90_CASE(0)
    FP4_MOE_SM90_CASE(1)
    FP4_MOE_SM90_CASE(2)
    FP4_MOE_SM90_CASE(3)
    FP4_MOE_SM90_CASE(4)
    FP4_MOE_SM90_CASE(5)
    default:
      return 0;
  }
#undef FP4_MOE_SM90_CASE
}

void stages_of(int variant, APath ap, int* out) {
#define FP4_MOE_SM90_CASE(V) \
  case V:                    \
    return ap == APath::kN1 ? stage_table<V, APath::kN1>(out) : stage_table<V, APath::kN2>(out);
  switch (variant) {
    FP4_MOE_SM90_CASE(0)
    FP4_MOE_SM90_CASE(1)
    FP4_MOE_SM90_CASE(2)
    FP4_MOE_SM90_CASE(3)
    FP4_MOE_SM90_CASE(4)
    FP4_MOE_SM90_CASE(5)
    default:
      for (int i = 0; i < kNumShapes; ++i) out[i] = 0;
  }
#undef FP4_MOE_SM90_CASE
}

int gemm1_variant(bool per_token, AFmt w1_fmt) {
  if (w1_fmt == AFmt::kNvfp4TrtllmShuffled) return per_token ? kG1TokenTrtllm : kG1TensorTrtllm;
  return per_token ? kG1TokenSwz : kG1TensorSwz;
}

int gemm2_variant(AFmt w2_fmt) { return w2_fmt == AFmt::kNvfp4TrtllmShuffled ? kG2Trtllm : kG2Swz; }

// The caller's per-token global scales by expanded row (NVFP4 per-token activations).
__global__ void gather_token_scale(const float* __restrict__ token_scale,
                                   const int32_t* __restrict__ permuted_src,
                                   const int32_t* __restrict__ count, int top_k, int rows,
                                   float* __restrict__ out) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < rows && p < *count) out[p] = token_scale[permuted_src[p] / top_k];
}

constexpr size_t kAlign = 256;
inline size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

inline int max_tiles_of(int rows, int n, int num_local_experts) {
  const int bound = (rows + n - 1) / n + num_local_experts;
  return bound < rows ? bound : rows;
}

// B tile bytes of one GEMM: max_tiles n-tiles x ceil(K / 128) K tiles x n rows x 256 B.
inline size_t b_tiles_bytes(int max_tiles, int k, int n) {
  return size_t(max_tiles) * size_t((k + 127) / 128) * size_t(n) * 256;
}

struct Layout {
  size_t expert_offset, expert_tile_base, permuted_src, tile_expert, tile_n0, num_tiles;
  size_t tok1, tok2, x_tiles, act, x2_tiles, prep, total;
};

Layout make_layout(const PrepArgs& prep, int inter, bool per_token) {
  const size_t rows = size_t(prep.tokens) * prep.top_k;
  const size_t experts = size_t(prep.num_local_experts) + 1;
  Layout l{};
  size_t at = 0;
  auto take = [&at](size_t bytes) {
    const size_t off = at;
    at += align_up(bytes);
    return off;
  };
  l.expert_offset = take(experts * 4);
  l.expert_tile_base = take(experts * 4);
  l.permuted_src = take(rows * 4);
  l.tile_expert = take(size_t(prep.max_tiles) * 4);
  l.tile_n0 = take(size_t(prep.max_tiles) * 4);
  l.num_tiles = take(4);
  l.tok1 = take(per_token ? rows * 4 : 0);
  l.tok2 = take(per_token ? rows * 4 : 0);
  l.x_tiles = take(b_tiles_bytes(prep.max_tiles, prep.k, prep.n));
  // per token: BF16 GEMM1 rows for row_quant; per tensor: GEMM2 B tiles written by GEMM1
  l.act = take(per_token ? rows * size_t(inter) * 2 : b_tiles_bytes(prep.max_tiles, inter, prep.n));
  l.x2_tiles = take(per_token ? b_tiles_bytes(prep.max_tiles, inter, prep.n) : 0);
  l.prep = take(fp4_sm90_prep_workspace_bytes(prep));
  l.total = at;
  return l;
}

PrepArgs prep_shape(int64_t tokens, int64_t top_k, int64_t num_experts, int64_t local_expert_offset,
                    int64_t num_local_experts, int64_t hidden, int64_t x_fmt, int64_t quantize,
                    int64_t n) {
  PrepArgs p{};
  p.tokens = static_cast<int>(tokens);
  p.top_k = static_cast<int>(top_k);
  p.num_experts = static_cast<int>(num_experts);
  p.local_expert_offset = static_cast<int>(local_expert_offset);
  p.num_local_experts = static_cast<int>(num_local_experts);
  p.k = static_cast<int>(hidden);
  p.x_fmt = static_cast<XFmt>(x_fmt);
  p.quantize = static_cast<uint8_t>(quantize);
  p.per_token = 0;
  p.recipe = kModuleRecipe;
  p.f16 = kModuleF16;
  p.n = static_cast<int>(n);
  p.max_tiles = max_tiles_of(p.tokens * p.top_k, p.n, p.num_local_experts);
  return p;
}

bool valid_n(int64_t n) { return n >= 8 && n <= 256 && (n & (n - 1)) == 0; }

// geom: fmt, pair, rows, rows_phys, k, k_phys, half_rows, pair_g, codes stride, sf stride (bytes),
// num_experts
ASrc weight_of(TensorView codes, TensorView sf, const Array<int64_t>& geom, const char* name) {
  TVM_FFI_ICHECK_EQ(geom.size(), 11) << name << ": weight geometry must hold 11 values";
  CHECK_CUDA(codes);
  CHECK_CUDA(sf);
  TVM_FFI_ICHECK(codes.dtype() == dl_uint8 && sf.dtype() == dl_uint8)
      << name << ": codes and block scales must be uint8 views";
  ASrc a{};
  a.codes = static_cast<const uint8_t*>(codes.data_ptr());
  a.sf = static_cast<const uint8_t*>(sf.data_ptr());
  a.fmt = static_cast<AFmt>(geom[0]);
  a.pair = static_cast<Pair>(geom[1]);
  a.rows = static_cast<int>(geom[2]);
  a.rows_phys = static_cast<int>(geom[3]);
  a.k = static_cast<int>(geom[4]);
  a.k_phys = static_cast<int>(geom[5]);
  a.half_rows = static_cast<int>(geom[6]);
  a.pair_g = static_cast<int>(geom[7]);
  a.codes_expert_stride = geom[8];
  a.sf_expert_stride = geom[9];
  a.num_experts = static_cast<int>(geom[10]);
  TVM_FFI_ICHECK(a.fmt == AFmt::kNvfp4TrtllmShuffled || a.fmt == AFmt::kNvfp4Swz128x4)
      << name << ": the SM90 FP4 MoE takes NVFP4 TRTLLM-shuffled or 128x4 weights";
  return a;
}

// A localized launch's locality domains (SM100's LocalizedMoE::kNumDomains).
constexpr int kNumDomains = 2;

// SM100's DomainForkJoin (trtllm_fused_moe_runner.cu): fork makes the domain streams wait for the
// caller's stream, join makes it wait for them. Per-call events; under graph capture each record
// and wait is a graph edge, so the domain streams join the capture and merge back before it ends.
class DomainForkJoin {
 public:
  DomainForkJoin(cudaStream_t main, const cudaStream_t (&streams)[kNumDomains]) : main_(main) {
    for (int d = 0; d < kNumDomains; ++d) streams_[d] = streams[d];
    // A throwing constructor skips the destructor, so a failed create frees the events made so far.
    cudaError_t err = cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming);
    for (int d = 0; d < kNumDomains && err == cudaSuccess; ++d) {
      err = cudaEventCreateWithFlags(&done_[d], cudaEventDisableTiming);
    }
    if (err != cudaSuccess) destroy();
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaEventCreate";
  }
  ~DomainForkJoin() { destroy(); }
  DomainForkJoin(const DomainForkJoin&) = delete;
  DomainForkJoin& operator=(const DomainForkJoin&) = delete;

  void fork() {
    TVM_FFI_ICHECK_EQ(cudaEventRecord(ready_, main_), cudaSuccess) << "cudaEventRecord";
    for (cudaStream_t s : streams_) {
      TVM_FFI_ICHECK_EQ(cudaStreamWaitEvent(s, ready_, 0), cudaSuccess) << "cudaStreamWaitEvent";
    }
  }
  void join() {
    for (int d = 0; d < kNumDomains; ++d) {
      TVM_FFI_ICHECK_EQ(cudaEventRecord(done_[d], streams_[d]), cudaSuccess) << "cudaEventRecord";
      TVM_FFI_ICHECK_EQ(cudaStreamWaitEvent(main_, done_[d], 0), cudaSuccess)
          << "cudaStreamWaitEvent";
    }
  }
  // join() for error paths: never throws, so the original error propagates.
  void join_no_throw() noexcept {
    for (int d = 0; d < kNumDomains; ++d) {
      if (cudaEventRecord(done_[d], streams_[d]) == cudaSuccess) {
        cudaStreamWaitEvent(main_, done_[d], 0);
      }
    }
  }

 private:
  cudaStream_t main_;
  cudaStream_t streams_[kNumDomains];
  cudaEvent_t ready_{};
  cudaEvent_t done_[kNumDomains]{};

  void destroy() noexcept {
    if (ready_ != nullptr) cudaEventDestroy(ready_);
    for (cudaEvent_t done : done_) {
      if (done != nullptr) cudaEventDestroy(done);
    }
  }
};

}  // namespace moe
}  // namespace fp4_sm90
}  // namespace flashinfer

using namespace flashinfer::fp4_sm90;
using namespace flashinfer::fp4_sm90::moe;

int64_t fp4_moe_sm90_workspace_bytes(int64_t tokens, int64_t top_k, int64_t num_experts,
                                     int64_t local_expert_offset, int64_t num_local_experts,
                                     int64_t hidden, int64_t inter, int64_t x_fmt, int64_t quantize,
                                     bool per_token, int64_t n) {
  TVM_FFI_ICHECK(valid_n(n)) << "n must be a power of two in [8, 256]";
  const PrepArgs p = prep_shape(tokens, top_k, num_experts, local_expert_offset, num_local_experts,
                                hidden, x_fmt, quantize, n);
  return static_cast<int64_t>(make_layout(p, static_cast<int>(inter), per_token).total);
}

// Tactics (cta_m, n, stages of GEMM1) whose GEMM1 and GEMM2 kernels both hold 2 or more stages.
Array<int64_t> fp4_moe_sm90_tactics(bool per_token, int64_t w1_fmt, int64_t w2_fmt,
                                    int64_t a_path) {
  const APath ap = a_path ? APath::kN2 : APath::kN1;
  int s1[kNumShapes], s2[kNumShapes];
  stages_of(gemm1_variant(per_token, static_cast<AFmt>(w1_fmt)), ap, s1);
  stages_of(gemm2_variant(static_cast<AFmt>(w2_fmt)), ap, s2);
  std::vector<int64_t> out;
  for (int i = 0; i < kNumShapes; ++i) {
    if (s1[i] >= 2 && s2[i] >= 2) {
      out.push_back(kShapes[i].cta_m);
      out.push_back(kShapes[i].n);
      out.push_back(s1[i]);
    }
  }
  return Array<int64_t>(out);
}

void fp4_moe_sm90_run(
    // activations: BF16/FP16 [T, H] or NVFP4 codes [T, H / 2] with block scales
    TensorView x, Optional<TensorView> x_sf, int64_t x_fmt, Optional<TensorView> x_global,
    int64_t quantize, Optional<TensorView> token_scale,
    // routing
    TensorView topk_ids, Optional<TensorView> topk_weights, int64_t num_experts,
    int64_t local_expert_offset, int64_t num_local_experts,
    // weights (native SM100+ layouts, uint8 views)
    TensorView w1_codes, TensorView w1_sf, Array<int64_t> w1_geom, TensorView w2_codes,
    TensorView w2_sf, Array<int64_t> w2_geom,
    // epilogues
    TensorView alpha1, TensorView alpha2, Optional<TensorView> act_x, Optional<TensorView> act_y,
    Optional<TensorView> act_z, int64_t act, bool per_token, Optional<TensorView> fq_global,
    Optional<TensorView> bias1, Optional<TensorView> bias2,
    // per token: FP32 [num_local_experts] factor of GEMM1's 16-bit rows (null = 1)
    Optional<TensorView> out_scale1,
    // scratch and outputs
    TensorView workspace, TensorView src_permuted, TensorView y, Optional<TensorView> out,
    // fc2_input_out (per token): GEMM2's quantized input in (token, slot) order, codes
    // [>= T, top_k, I / 2], linear E4M3 scales [>= T, top_k, I / 16] and token scales
    // [>= T, top_k]
    Optional<TensorView> fc2_codes, Optional<TensorView> fc2_sf, Optional<TensorView> fc2_tok,
    // gemm1_lora_delta (gated, per tensor): BF16 [T, top_k, 2 * I] by (token, slot), [up; gate]
    // with I >= the GEMM2 K, added before the activation in output units; act_out receives
    // the activation by
    // expanded row, GEMM1's output dtype [>= T * top_k, >= the GEMM2 K]
    Optional<TensorView> lora, Optional<TensorView> act_out,
    // tactic
    int64_t cta_m, int64_t n, int64_t stages, int64_t a_path,
    // localization (SM100's localized_*): per domain its shards' w1 codes, w1 scales, w2 codes and
    // w2 scales (uint8 views) and their geometries (w1's, then w2's), the domain streams and the
    // SM count of each (0: off)
    Array<Tensor> loc_weights, Array<int64_t> loc_geom, Array<int64_t> loc_streams,
    int64_t loc_sm_count) {
  CHECK_CUDA(x);
  CHECK_CONTIGUOUS(x);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_DIM(2, topk_ids);
  CHECK_INPUT_AND_TYPE(alpha1, dl_float32);
  CHECK_INPUT_AND_TYPE(alpha2, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_INPUT_AND_TYPE(src_permuted, dl_int32);
  CHECK_INPUT(y);
  // Every consumed tensor on x's device (the launches run there).
  for (const TensorView& t :
       {topk_ids, w1_codes, w1_sf, w2_codes, w2_sf, alpha1, alpha2, workspace, src_permuted, y}) {
    CHECK_DEVICE(t, x);
  }
  if (x_sf.has_value()) {
    CHECK_CUDA(x_sf.value());
    CHECK_DEVICE(x_sf.value(), x);
  }
  TVM_FFI_ICHECK(y.dtype() == dl_bfloat16 || y.dtype() == dl_float16) << "y must be BF16 or FP16";
  TVM_FFI_ICHECK_EQ(act, kModuleAct) << "activation " << act << " is not this module's";
  TVM_FFI_ICHECK(valid_n(n)) << "n must be a power of two in [8, 256]";
  TVM_FFI_ICHECK(cta_m == 128 || cta_m == 256) << "cta_m must be 128 or 256";
  TVM_FFI_ICHECK(a_path == 0 || a_path == 1) << "a_path must be 0 (n1) or 1 (n2)";

  const int tokens = static_cast<int>(topk_ids.size(0));
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int rows = tokens * top_k;
  const ASrc a1 = weight_of(w1_codes, w1_sf, w1_geom, "w1");
  const ASrc a2 = weight_of(w2_codes, w2_sf, w2_geom, "w2");
  const int hidden = a1.k, inter = a2.k;
  TVM_FFI_ICHECK_EQ(a2.rows, hidden) << "w2 rows must equal the hidden size";
  TVM_FFI_ICHECK_EQ(a1.rows, kModuleGated ? 2 * inter : inter)
      << "w1 rows must be " << (kModuleGated ? "2 * " : "") << "the intermediate size";
  TVM_FFI_ICHECK_EQ(a1.pair != Pair::kNone, kModuleGated)
      << "w1 gate/up pairing does not match the activation";
  TVM_FFI_ICHECK(a2.pair == Pair::kNone) << "w2 takes no gate/up pairing";
  TVM_FFI_ICHECK(a1.num_experts == num_local_experts && a2.num_experts == num_local_experts)
      << "w1 and w2 must hold num_local_experts experts";
  TVM_FFI_ICHECK_EQ(alpha1.numel(), num_local_experts) << "alpha1 must be [num_local_experts]";
  TVM_FFI_ICHECK_EQ(alpha2.numel(), num_local_experts) << "alpha2 must be [num_local_experts]";
  TVM_FFI_ICHECK_EQ(src_permuted.numel(), rows) << "src_permuted must be [num_tokens * top_k]";
  TVM_FFI_ICHECK(y.size(0) >= rows && y.size(1) == hidden)
      << "y must be [>= num_tokens * top_k, hidden]";
  if (per_token) {
    TVM_FFI_ICHECK(token_scale.has_value()) << "per-token NVFP4 needs the token scales";
    CHECK_INPUT_AND_TYPE(token_scale.value(), dl_float32);
    CHECK_DEVICE(token_scale.value(), x);
    TVM_FFI_ICHECK_EQ(token_scale.value().numel(), tokens) << "token scales must be [num_tokens]";
  }
  if (x_global.has_value()) {
    CHECK_INPUT_AND_TYPE(x_global.value(), dl_float32);
    CHECK_DEVICE(x_global.value(), x);
    TVM_FFI_ICHECK_EQ(x_global.value().numel(), 1) << "input global scale must be a scalar";
  }
  TVM_FFI_ICHECK(!quantize || x_global.has_value()) << "input quantization needs a global scale";
  if (fq_global.has_value()) {
    CHECK_INPUT_AND_TYPE(fq_global.value(), dl_float32);
    CHECK_DEVICE(fq_global.value(), x);
    TVM_FFI_ICHECK(fq_global.value().numel() == 1 ||
                   (!per_token && fq_global.value().numel() == num_local_experts))
        << "fq_global must be a scalar (per tensor also [num_local_experts])";
  }
  if (out_scale1.has_value()) {
    TVM_FFI_ICHECK(per_token) << "out_scale1 scales GEMM1's per-token rows";
    CHECK_INPUT_AND_TYPE(out_scale1.value(), dl_float32);
    CHECK_DEVICE(out_scale1.value(), x);
    TVM_FFI_ICHECK_EQ(out_scale1.value().numel(), num_local_experts)
        << "out_scale1 must be [num_local_experts]";
  }
  const bool fc2 = fc2_codes.has_value();
  TVM_FFI_ICHECK(fc2 == fc2_sf.has_value() && fc2 == fc2_tok.has_value())
      << "fc2_input_out takes codes, block scales and token scales together";
  int fq_cols = 0;
  if (fc2) {
    TVM_FFI_ICHECK(per_token) << "fc2_input_out requires per-token NvFP4";
    const TensorView c = fc2_codes.value(), sf = fc2_sf.value(), tk = fc2_tok.value();
    CHECK_INPUT_AND_TYPE(c, dl_uint8);
    CHECK_INPUT_AND_TYPE(sf, dl_uint8);
    CHECK_INPUT_AND_TYPE(tk, dl_float32);
    for (const TensorView& t : {c, sf, tk}) {
      CHECK_DEVICE(t, x);
    }
    CHECK_DIM(3, c);
    CHECK_DIM(3, sf);
    CHECK_DIM(2, tk);
    fq_cols = static_cast<int>(2 * c.size(2));
    TVM_FFI_ICHECK(fq_cols >= inter && fq_cols % 16 == 0)
        << "fc2_input_out codes must be [>= num_tokens, top_k, I / 2] with I >= the GEMM2 K "
           "and I % 16 == 0";
    TVM_FFI_ICHECK(c.size(0) >= tokens && c.size(1) == top_k && sf.size(0) >= tokens &&
                   sf.size(1) == top_k && sf.size(2) == fq_cols / 16 && tk.size(0) >= tokens &&
                   tk.size(1) == top_k)
        << "fc2_input_out tensors must be [>= num_tokens, top_k, ...] of the codes' width";
  }
  TVM_FFI_ICHECK_EQ(lora.has_value(), act_out.has_value())
      << "gemm1_lora_delta and its activation output go together";
  if (lora.has_value()) {
    TVM_FFI_ICHECK(kModuleGated && !per_token) << "gemm1_lora_delta needs a gated activation "
                                                  "per tensor";
    const TensorView d = lora.value(), o = act_out.value();
    CHECK_INPUT_AND_TYPE(d, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(o, kModuleF16 ? dl_float16 : dl_bfloat16);
    for (const TensorView& t : {d, o}) {
      CHECK_DEVICE(t, x);
    }
    CHECK_DIM(3, d);
    CHECK_DIM(2, o);
    TVM_FFI_ICHECK(d.size(0) == tokens && d.size(1) == top_k && d.size(2) % 2 == 0 &&
                   d.size(2) >= 2 * inter)
        << "gemm1_lora_delta must be [num_tokens, top_k, 2 * I] with I >= the GEMM2 K";
    TVM_FFI_ICHECK(o.size(0) >= rows && o.size(1) >= inter)
        << "the activation output must be [>= num_tokens * top_k, >= the GEMM2 K]";
  }
  // GEMM1 activation parameters (cutlass_fused_moe's ActivationParams)
  for (const Optional<TensorView>* p : {&act_x, &act_y, &act_z}) {
    if (!p->has_value()) continue;
    CHECK_INPUT_AND_TYPE(p->value(), dl_float32);
    CHECK_DEVICE(p->value(), x);
    CHECK_DIM(1, p->value());
    TVM_FFI_ICHECK_EQ(p->value().size(0), num_local_experts)
        << "activation parameters must be [num_local_experts]";
  }
  // Per-expert biases of the GEMM rows: the output dtype on the logical rows, or TRT-LLM's FP32
  // on the rows of TRT-LLM shuffled weights (accumulator units, per tensor)
  for (const auto& [b, a] : {std::pair{&bias1, &a1}, std::pair{&bias2, &a2}}) {
    if (!b->has_value()) continue;
    CHECK_INPUT(b->value());
    CHECK_DEVICE(b->value(), x);
    const bool trtllm_rows =
        afmt_trtllm(a->fmt) && (a->pair == Pair::kNone || a->pair == Pair::kTrtllmW31);
    TVM_FFI_ICHECK(b->value().dtype() == y.dtype() ||
                   (b->value().dtype() == dl_float32 && trtllm_rows && !per_token))
        << "biases must have the output dtype, or be FP32 per tensor on TRT-LLM shuffled weights";
    CHECK_DIM(2, b->value());
    TVM_FFI_ICHECK(b->value().size(0) == num_local_experts && b->value().size(1) == a->rows)
        << "biases must be [num_local_experts, GEMM rows]";
  }

  // A localized launch (SM100's runLocalized) runs each GEMM per domain on its stream with
  // loc_sm_count SMs: GEMM1 domain d writes the intermediate columns [d I / 2, (d + 1) I / 2) from
  // its w1 shard (rows [d I / 2, (d + 1) I / 2) of each half), GEMM2 the hidden columns
  // [d H / 2, (d + 1) H / 2) from its w2 shard, after both GEMM1s.
  const bool localized = loc_sm_count > 0;
  ASrc loc1[kNumDomains] = {}, loc2[kNumDomains] = {};
  cudaStream_t loc_stream[kNumDomains] = {};
  if (localized) {
    TVM_FFI_ICHECK(kModuleGated && !per_token && !bias1.has_value() && !bias2.has_value() &&
                   !lora.has_value())
        << "a localized launch is gated and per tensor, without biases or LoRA deltas";
    TVM_FFI_ICHECK(loc_weights.size() == 4 * kNumDomains && loc_geom.size() == 22 * kNumDomains &&
                   loc_streams.size() == kNumDomains)
        << "a localized launch takes two domains' shards, geometries and streams";
    TVM_FFI_ICHECK(hidden % 256 == 0 && inter % 256 == 0)
        << "a localized launch needs hidden and intermediate sizes that are multiples of 256";
    for (int d = 0; d < kNumDomains; ++d) {
      const Tensor w[4] = {loc_weights[4 * d], loc_weights[4 * d + 1], loc_weights[4 * d + 2],
                           loc_weights[4 * d + 3]};
      for (const Tensor& t : w) {
        CHECK_DEVICE(TensorView(t), x);
      }
      auto geom = [&](int i) {
        const auto g = loc_geom.begin() + 22 * d + 11 * i;
        return Array<int64_t>(std::vector<int64_t>(g, g + 11));
      };
      loc1[d] = weight_of(w[0], w[1], geom(0), "w1 shard");
      loc2[d] = weight_of(w[2], w[3], geom(1), "w2 shard");
      TVM_FFI_ICHECK(loc1[d].fmt == a1.fmt && loc1[d].pair == a1.pair && loc1[d].rows == inter &&
                     loc1[d].k == hidden && loc2[d].fmt == a2.fmt && 2 * loc2[d].rows == hidden &&
                     loc2[d].k == inter && loc2[d].pair == Pair::kNone &&
                     loc1[d].num_experts == num_local_experts &&
                     loc2[d].num_experts == num_local_experts)
          << "domain " << d << "'s shards must be w1's and w2's rows of the domain";
      loc_stream[d] = reinterpret_cast<cudaStream_t>(loc_streams[d]);
    }
  }

  CHECK_DIM(2, x);
  TVM_FFI_ICHECK_EQ(x.size(0), tokens) << "x must be [num_tokens, ...]";
  TVM_FFI_ICHECK(x_fmt >= int64_t(XFmt::kBf16) && x_fmt <= int64_t(XFmt::kMxfp4Swz128x4))
      << "x_fmt " << x_fmt << " is not an SM90 input format";
  if (x_fmt == int64_t(XFmt::kBf16) || x_fmt == int64_t(XFmt::kF16)) {
    TVM_FFI_ICHECK(x.dtype() == (x_fmt == int64_t(XFmt::kF16) ? dl_float16 : dl_bfloat16))
        << "x dtype does not match x_fmt " << x_fmt;
    TVM_FFI_ICHECK_EQ(x.size(1), hidden) << "x must be [num_tokens, hidden]";
  } else {
    TVM_FFI_ICHECK(x.dtype() == dl_uint8 && x_sf.has_value()) << "an FP4 x is uint8 codes + scales";
    TVM_FFI_ICHECK(!quantize) << "an FP4 x is already quantized";
    TVM_FFI_ICHECK_GE(2 * x.size(1), hidden) << "FP4 x must be [num_tokens, >= hidden / 2]";
  }

  const PrepArgs shape = prep_shape(tokens, top_k, num_experts, local_expert_offset,
                                    num_local_experts, hidden, x_fmt, quantize, n);
  const Layout l = make_layout(shape, inter, per_token);
  TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.numel()), l.total)
      << "workspace must hold " << l.total << " bytes";
  uint8_t* ws = static_cast<uint8_t*>(workspace.data_ptr());
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(ws) % kAlign, 0u)
      << "workspace must be 256 B aligned";
  auto at = [ws](size_t off) { return ws + off; };

  const APath ap = a_path ? APath::kN2 : APath::kN1;
  const int v1 = gemm1_variant(per_token, a1.fmt), v2 = gemm2_variant(a2.fmt);
  int s1[kNumShapes];
  stages_of(v1, ap, s1);
  int shape_index = -1;
  for (int i = 0; i < kNumShapes; ++i) {
    if (kShapes[i].cta_m == cta_m && kShapes[i].n == n) shape_index = i;
  }
  TVM_FFI_ICHECK(shape_index >= 0) << "tactic (" << cta_m << ", " << n << ") is not a D7 shape";
  TVM_FFI_ICHECK(stages == 0 || stages == s1[shape_index])
      << "tactic stages " << stages << " differ from the module's " << s1[shape_index];

  ffi::CUDADeviceGuard guard(x.device().device_id);
  const cudaStream_t stream = get_stream(x.device());
  int num_sms = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().device_id),
      cudaSuccess)
      << "cudaDeviceGetAttribute(SM count)";
  TVM_FFI_ICHECK(!localized || (loc_stream[0] != loc_stream[1] && loc_stream[0] != stream &&
                                loc_stream[1] != stream))
      << "localized streams must be distinct and differ from the current stream";
  // One GEMM: plain, or localized with part(d, args) narrowing it to domain d.
  auto gemm = [&](int variant, const GemmArgs& g, const char* name, auto part) {
    auto launch = [&](const GemmArgs& args, int sms, cudaStream_t s) {
      TVM_FFI_ICHECK_GE(
          dispatch_gemm(variant, ap, args, static_cast<int>(cta_m), static_cast<int>(n), sms, s), 2)
          << name << " tactic (" << cta_m << ", " << n << ") has no kernel (fewer than 2 stages)";
    };
    if (!localized) return launch(g, num_sms, stream);
    DomainForkJoin fork_join(stream, loc_stream);
    try {
      fork_join.fork();
      for (int d = 0; d < kNumDomains; ++d) {
        GemmArgs gd = g;
        part(d, gd);
        launch(gd, static_cast<int>(loc_sm_count), loc_stream[d]);
      }
      fork_join.join();
    } catch (...) {
      fork_join.join_no_throw();
      throw;
    }
  };

  // prep: routing sort, n-tile list, activation (de)quantization into GEMM1 B tiles
  PrepArgs prep = shape;
  prep.topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  prep.x = x.data_ptr();
  prep.x_sf = x_sf.has_value() ? static_cast<const uint8_t*>(x_sf.value().data_ptr()) : nullptr;
  prep.x_ld = x.stride(0);
  // GEMM1 reads the first `hidden` columns of a wider x (valid_hidden_size); its scale rows keep
  // the full width
  prep.x_sf_cols =
      x_sf.has_value()
          ? static_cast<int>(2 * x.size(1) / (x_fmt == int64_t(XFmt::kMxfp4Swz128x4) ? 32 : 16))
          : 0;
  prep.x_global =
      x_global.has_value() ? static_cast<const float*>(x_global.value().data_ptr()) : nullptr;
  prep.expert_offset = reinterpret_cast<int32_t*>(at(l.expert_offset));
  prep.expert_tile_base = reinterpret_cast<int32_t*>(at(l.expert_tile_base));
  prep.permuted_src = reinterpret_cast<int32_t*>(at(l.permuted_src));
  prep.src_permuted = static_cast<int32_t*>(src_permuted.data_ptr());
  prep.tile_expert = reinterpret_cast<int32_t*>(at(l.tile_expert));
  prep.tile_n0 = reinterpret_cast<int32_t*>(at(l.tile_n0));
  prep.num_tiles = reinterpret_cast<int32_t*>(at(l.num_tiles));
  prep.x_tiles = at(l.x_tiles);
  prep.tok_scale = nullptr;
  prep.workspace = at(l.prep);
  launch_fp4_sm90_prep(prep, stream);

  float* tok1 = per_token ? reinterpret_cast<float*>(at(l.tok1)) : nullptr;
  float* tok2 = per_token ? reinterpret_cast<float*>(at(l.tok2)) : nullptr;
  if (per_token && rows > 0) {
    constexpr int kThreads = 256;
    gather_token_scale<<<(rows + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const float*>(token_scale.value().data_ptr()), prep.permuted_src,
        prep.expert_offset + num_local_experts, top_k, rows, tok1);
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess) << "gather_token_scale launch";
  }

  Sched sched{};
  sched.tile_expert = prep.tile_expert;
  sched.tile_n0 = prep.tile_n0;
  sched.num_tiles = prep.num_tiles;
  sched.expert_offset = prep.expert_offset;
  sched.expert_tile_base = prep.expert_tile_base;
  sched.max_tiles = prep.max_tiles;

  // GEMM1: BF16 rows (token-wise) or NVFP4-quantized GEMM2 B tiles (per tensor)
  GemmArgs g1{};
  g1.a = a1;
  g1.b.x = prep.x_tiles;
  g1.b.tokens = rows;
  g1.b.k = hidden;
  g1.b.x_ld = 0;
  g1.b.tiled = 1;
  g1.epi.alpha = static_cast<const float*>(alpha1.data_ptr());
  g1.epi.tok_scale = tok1;
  auto ptr = [](const Optional<TensorView>& t) -> void* {
    return t.has_value() ? t.value().data_ptr() : nullptr;
  };
  g1.epi.act_x = static_cast<const float*>(ptr(act_x));
  g1.epi.act_y = static_cast<const float*>(ptr(act_y));
  g1.epi.act_z = static_cast<const float*>(ptr(act_z));
  g1.epi.bias = ptr(bias1);
  g1.epi.bias_f16 = y.dtype() == dl_float16;
  g1.epi.bias_trtllm = bias1.has_value() && bias1.value().dtype() == dl_float32;
  g1.epi.out = at(l.act);
  g1.epi.out_ld = inter;
  g1.epi.out_scale = static_cast<const float*>(ptr(out_scale1));
  g1.epi.lora = ptr(lora);
  g1.epi.lora_row = lora.has_value() ? prep.permuted_src : nullptr;
  g1.epi.lora_ld = lora.has_value() ? lora.value().size(2) : 0;
  g1.epi.lora_half = g1.epi.lora_ld / 2;
  g1.epi.act_out = ptr(act_out);
  g1.epi.act_out_ld = act_out.has_value() ? act_out.value().size(1) : 0;
  // fq_global: per tensor GEMM1's output global scale, per token the row quantization's inverse
  // global scale (null: the recipe's)
  const float* const fq_g =
      fq_global.has_value() ? static_cast<const float*>(fq_global.value().data_ptr()) : nullptr;
  g1.epi.fq_global = per_token ? nullptr : fq_g;
  g1.epi.fq_per_expert = fq_global.has_value() && fq_global.value().numel() > 1;
  g1.epi.act = static_cast<uint8_t>(act);
  g1.epi.fq = static_cast<uint8_t>(per_token ? Fq::kNone : Fq::kNvfp4);
  g1.epi.recipe = per_token ? 0 : kModuleRecipe;
  g1.epi.out_f16 = per_token ? 0 : kModuleF16;
  g1.epi.out_tiled = per_token ? 0 : 1;
  g1.sched = sched;
  gemm(v1, g1, "GEMM1", [&](int d, GemmArgs& g) {
    g.a = loc1[d];
    // GEMM2's B tiles from K tile d I / 256 (out_ld keeps their K)
    g.epi.out = at(l.act) + int64_t(d) * (inter / 256) * 2 * n * 128;
  });

  // token-wise recipes: quantize GEMM1's rows into GEMM2 B tiles with per-row scales
  if (per_token) {
    RowQuantArgs rq{};
    rq.in = at(l.act);
    rq.in_ld = inter;
    rq.rows = rows;
    rq.cols = inter;
    rq.in_f16 = 0;
    rq.recipe = kModuleRecipe;
    rq.out_f16 = kModuleF16;
    rq.n = static_cast<int>(n);
    rq.gsi = fq_g;
    rq.expert_offset = prep.expert_offset;
    rq.expert_tile_base = prep.expert_tile_base;
    rq.num_tiles = prep.num_tiles;
    rq.out_tiles = at(l.x2_tiles);
    rq.tok_scale = tok2;
    rq.fq_codes = static_cast<uint8_t*>(ptr(fc2_codes));
    rq.fq_sf = static_cast<uint8_t*>(ptr(fc2_sf));
    rq.fq_tok = static_cast<float*>(ptr(fc2_tok));
    rq.fq_src = fc2 ? prep.permuted_src : nullptr;
    rq.fq_dst = fc2 ? prep.src_permuted : nullptr;
    rq.fq_cols = fq_cols;
    launch_fp4_sm90_row_quant(rq, stream);
  }

  // GEMM2: per (token, slot) rows, summed by finalize (D6)
  GemmArgs g2{};
  g2.a = a2;
  g2.b.x = per_token ? at(l.x2_tiles) : at(l.act);
  g2.b.tokens = rows;
  g2.b.k = inter;
  g2.b.x_ld = 0;
  g2.b.tiled = 1;
  g2.epi.alpha = static_cast<const float*>(alpha2.data_ptr());
  g2.epi.tok_scale = tok2;
  g2.epi.bias = ptr(bias2);
  g2.epi.bias_f16 = y.dtype() == dl_float16;
  g2.epi.bias_trtllm = bias2.has_value() && bias2.value().dtype() == dl_float32;
  g2.epi.out = y.data_ptr();
  g2.epi.out_ld = y.stride(0);
  g2.epi.fq_global = nullptr;
  g2.epi.act = kActIdentity;
  g2.epi.fq = static_cast<uint8_t>(Fq::kNone);
  g2.epi.recipe = 0;
  g2.epi.out_f16 = y.dtype() == dl_float16;
  g2.epi.out_tiled = 0;
  g2.sched = sched;
  gemm(v2, g2, "GEMM2", [&](int d, GemmArgs& g) {
    g.a = loc2[d];
    g.epi.out = static_cast<char*>(y.data_ptr()) + int64_t(d) * (hidden / 2) * 2;  // 16-bit y
  });

  if (out.has_value()) {
    TensorView o = out.value();
    CHECK_CUDA(o);
    CHECK_DEVICE(o, x);
    TVM_FFI_ICHECK(o.dtype() == dl_bfloat16 || o.dtype() == dl_float16)
        << "out must be BF16 or FP16";
    TVM_FFI_ICHECK(o.size(0) == tokens && o.size(1) >= 1 && o.size(1) <= hidden && o.stride(1) == 1)
        << "out must be [num_tokens, <= hidden] with unit column stride";
    FinalizeArgs fin{};
    fin.y = y.data_ptr();
    fin.y_ld = y.stride(0);
    fin.y_dtype = y.dtype() == dl_float16 ? 2 : 1;
    fin.src_permuted = prep.src_permuted;
    fin.topk_weights = nullptr;
    fin.w_dtype = 0;
    if (topk_weights.has_value()) {
      TensorView w = topk_weights.value();
      CHECK_INPUT(w);
      CHECK_DEVICE(w, x);
      TVM_FFI_ICHECK_EQ(w.numel(), rows) << "topk_weights must be [num_tokens, top_k]";
      TVM_FFI_ICHECK(w.dtype() == dl_float32 || w.dtype() == dl_bfloat16 || w.dtype() == dl_float16)
          << "topk_weights must be FP32, BF16 or FP16";
      fin.topk_weights = w.data_ptr();
      fin.w_dtype = w.dtype() == dl_float32 ? 0 : w.dtype() == dl_bfloat16 ? 1 : 2;
    }
    fin.out = o.data_ptr();
    fin.out_ld = o.stride(0);
    fin.out_f16 = o.dtype() == dl_float16;
    fin.tokens = tokens;
    fin.top_k = top_k;
    fin.hidden = static_cast<int>(o.size(1));  // the first out columns of y
    launch_fp4_sm90_finalize(fin, stream);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_moe_sm90_run, fp4_moe_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_moe_sm90_workspace_bytes, fp4_moe_sm90_workspace_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_moe_sm90_tactics, fp4_moe_sm90_tactics);

#endif  // FP4_MOE_SM90_PART
