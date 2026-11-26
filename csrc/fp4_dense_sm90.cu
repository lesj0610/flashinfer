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
// SM90 (wgmma) dense FP4 GEMM: out[M, N] = (x[M, K] @ dequant(W[N, K])^T) * alpha [* tok_scale].
//
// One module per (weight format, f16), see gen_fp4_dense_sm90_module. The weight is the wgmma A
// operand (swap-AB). A 16-bit x is read by the GEMM itself (BSrc::tiled == 0); an FP4 x is
// dequantized into B tiles by the prep kernel first. The GEMM kernel table (D7) is split over
// kNumParts translation units: a part (FP4_DENSE_SM90_PART defined) only instantiates
// run_gemm_one<shape, APath>; the main unit holds the TVM-FFI binding.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fp4_sm90/kernel.cuh"
#include "fp4_sm90/params.cuh"

#if !defined(FP4_DENSE_SM90_AFMT) || !defined(FP4_DENSE_SM90_F16)
#error "fp4_dense_sm90.cu is built by gen_fp4_dense_sm90_module (module defines missing)"
#endif

namespace flashinfer {
namespace fp4_sm90 {
namespace dense {

constexpr AFmt kModuleAFmt = static_cast<AFmt>(FP4_DENSE_SM90_AFMT);
constexpr bool kModuleF16 = FP4_DENSE_SM90_F16 != 0;
constexpr int kEpi = epi_code(Fq::kNone, 0, false, false);
constexpr int kActIdentity = 9;  // ActivationType.Identity
constexpr int kMaxStages = 32;

// D7 shapes: CTA_M in {128, 256} x N in {8, ..., 256}, (CTA_M / 128) * N / 2 <= 128.
struct Shape {
  int cta_m, n;
};
constexpr Shape kShapes[] = {{128, 8}, {128, 16}, {128, 32}, {128, 64}, {128, 128}, {128, 256},
                             {256, 8}, {256, 16}, {256, 32}, {256, 64}, {256, 128}};
constexpr int kNumShapes = sizeof(kShapes) / sizeof(kShapes[0]);
constexpr int kNumParts = kNumShapes * 2;  // x APath; FP4_DENSE_SM90_PARTS in Python

// Largest stage count whose shared memory fits; 0 when fewer than 2 fit (shape dropped, D7).
template <int I, APath AP, int S = kMaxStages>
constexpr int max_stages() {
  if constexpr (S < 2) {
    return 0;
  } else if constexpr (fp4_sm90_gemm_smem_bytes<Cfg<kShapes[I].cta_m, kShapes[I].n, S, kModuleAFmt,
                                                    kModuleF16, kEpi, AP>>() <=
                       size_t(kSmemLimit)) {
    return S;
  } else {
    return max_stages<I, AP, S - 1>();
  }
}

// Launches shape I; returns its stage count (0: not instantiated). Defined in the part units.
template <int I, APath AP>
int run_gemm_one(const GemmArgs& args, int num_sms, cudaStream_t stream);

#if defined(FP4_DENSE_SM90_PART)
template <int I, APath AP>
int run_gemm_one(const GemmArgs& args, int num_sms, cudaStream_t stream) {
  constexpr int S = max_stages<I, AP>();
  if constexpr (S >= 2) {
    launch_fp4_sm90_gemm<Cfg<kShapes[I].cta_m, kShapes[I].n, S, kModuleAFmt, kModuleF16, kEpi, AP>>(
        args, num_sms, stream);
  }
  return S;
}
#endif

}  // namespace dense
}  // namespace fp4_sm90
}  // namespace flashinfer

#if defined(FP4_DENSE_SM90_PART)

static_assert(FP4_DENSE_SM90_PART >= 0 &&
                  FP4_DENSE_SM90_PART < flashinfer::fp4_sm90::dense::kNumParts,
              "FP4_DENSE_SM90_PART out of range");
template int flashinfer::fp4_sm90::dense::run_gemm_one<
    (FP4_DENSE_SM90_PART) / 2, static_cast<flashinfer::fp4_sm90::APath>((FP4_DENSE_SM90_PART) % 2)>(
    const flashinfer::fp4_sm90::GemmArgs&, int, cudaStream_t);

#else  // main unit

#include "fp4_sm90/prep.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace flashinfer {
namespace fp4_sm90 {
namespace dense {

template <APath AP, int I = 0>
int dispatch_gemm(const GemmArgs& args, int cta_m, int n, int num_sms, cudaStream_t stream) {
  if constexpr (I == kNumShapes) {
    return 0;
  } else {
    if (kShapes[I].cta_m == cta_m && kShapes[I].n == n) {
      return run_gemm_one<I, AP>(args, num_sms, stream);
    }
    return dispatch_gemm<AP, I + 1>(args, cta_m, n, num_sms, stream);
  }
}

template <APath AP, int I = 0>
void stage_table(int* out) {
  if constexpr (I < kNumShapes) {
    out[I] = max_stages<I, AP>();
    stage_table<AP, I + 1>(out);
  }
}

constexpr size_t kAlign = 256;
inline size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

bool valid_n(int64_t n) { return n >= 8 && n <= 256 && (n & (n - 1)) == 0; }

bool is_fp4_x(int64_t x_fmt) {
  return x_fmt == int64_t(XFmt::kNvfp4Linear) || x_fmt == int64_t(XFmt::kNvfp4Swz128x4) ||
         x_fmt == int64_t(XFmt::kNvfp4Swz8x4);
}

// B tiles written by prep for an FP4 x (dense: ceil(M / n) n-tiles x ceil(K / 128) K tiles).
size_t x_tiles_bytes(int64_t m, int64_t k, int64_t n) {
  const size_t tiles = size_t((m + n - 1) / n);
  return tiles * size_t((k + 127) / 128) * size_t(n) * 256;
}

}  // namespace dense
}  // namespace fp4_sm90
}  // namespace flashinfer

using namespace flashinfer::fp4_sm90;
using namespace flashinfer::fp4_sm90::dense;

// Bytes of the workspace of one call: the B tiles of an FP4 x (0 for a 16-bit x).
int64_t fp4_dense_sm90_workspace_bytes(int64_t m, int64_t k, int64_t x_fmt, int64_t n) {
  TVM_FFI_ICHECK(valid_n(n)) << "n must be a power of two in [8, 256]";
  return is_fp4_x(x_fmt) ? static_cast<int64_t>(align_up(x_tiles_bytes(m, k, n))) : 0;
}

// Tactics (cta_m, n, stages) of this module's weight format whose kernel holds 2 or more stages.
Array<int64_t> fp4_dense_sm90_tactics(int64_t a_path) {
  int stages[kNumShapes];
  if (a_path) {
    stage_table<APath::kN2>(stages);
  } else {
    stage_table<APath::kN1>(stages);
  }
  std::vector<int64_t> out;
  for (int i = 0; i < kNumShapes; ++i) {
    if (stages[i] >= 2) {
      out.push_back(kShapes[i].cta_m);
      out.push_back(kShapes[i].n);
      out.push_back(stages[i]);
    }
  }
  return Array<int64_t>(out);
}

// w_geom: fmt, pair, rows, rows_phys, k, k_phys, half_rows, pair_g, codes stride, sf stride
// (the Sm90WeightView geometry; dense: pair 0, strides 0). x: BF16/FP16 [M, K] or FP4 codes
// [M, K / 2] with x_sf scales in x_fmt (XFmt). alpha: fp32 scalar; tok_scale: fp32 [M] or none.
void fp4_dense_sm90_run(TensorView x, Optional<TensorView> x_sf, int64_t x_fmt, TensorView w_codes,
                        TensorView w_sf, Array<int64_t> w_geom, TensorView alpha,
                        Optional<TensorView> tok_scale, TensorView workspace, TensorView out,
                        int64_t cta_m, int64_t n, int64_t stages, int64_t a_path) {
  CHECK_CUDA(x);
  CHECK_CUDA(w_codes);
  CHECK_CUDA(w_sf);
  CHECK_CUDA(out);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  // Every consumed tensor on x's device (the launch runs there).
  CHECK_DEVICE(w_codes, x);
  CHECK_DEVICE(w_sf, x);
  CHECK_DEVICE(out, x);
  CHECK_DEVICE(alpha, x);
  CHECK_DEVICE(workspace, x);
  CHECK_DIM(2, x);
  CHECK_DIM(2, out);
  TVM_FFI_ICHECK_EQ(w_geom.size(), 10) << "weight geometry must hold 10 values";
  TVM_FFI_ICHECK(w_codes.dtype() == dl_uint8 && w_sf.dtype() == dl_uint8)
      << "weight codes and block scales must be uint8 views";
  TVM_FFI_ICHECK(out.dtype() == dl_bfloat16 || out.dtype() == dl_float16)
      << "out must be BF16 or FP16";
  TVM_FFI_ICHECK(valid_n(n)) << "n must be a power of two in [8, 256]";
  TVM_FFI_ICHECK(cta_m == 128 || cta_m == 256) << "cta_m must be 128 or 256";
  TVM_FFI_ICHECK(a_path == 0 || a_path == 1) << "a_path must be 0 (n1) or 1 (n2)";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must be a scalar";

  ASrc a{};
  a.codes = static_cast<const uint8_t*>(w_codes.data_ptr());
  a.sf = static_cast<const uint8_t*>(w_sf.data_ptr());
  a.fmt = static_cast<AFmt>(w_geom[0]);
  a.pair = static_cast<Pair>(w_geom[1]);
  a.rows = static_cast<int>(w_geom[2]);
  a.rows_phys = static_cast<int>(w_geom[3]);
  a.k = static_cast<int>(w_geom[4]);
  a.k_phys = static_cast<int>(w_geom[5]);
  a.half_rows = static_cast<int>(w_geom[6]);
  a.pair_g = static_cast<int>(w_geom[7]);
  a.codes_expert_stride = w_geom[8];
  a.sf_expert_stride = w_geom[9];
  a.num_experts = 1;
  TVM_FFI_ICHECK(a.fmt == kModuleAFmt)
      << "weight format " << int(a.fmt) << " is not this module's (" << int(kModuleAFmt) << ")";
  TVM_FFI_ICHECK(a.pair == Pair::kNone) << "a dense weight takes no gate/up pairing";

  const bool fp4_x = is_fp4_x(x_fmt);
  const int m = static_cast<int>(x.size(0));
  const int k = static_cast<int>(fp4_x ? 2 * x.size(1) : x.size(1));
  TVM_FFI_ICHECK_EQ(a.k, k) << "x has K = " << k << " but the weight K = " << a.k;
  TVM_FFI_ICHECK_EQ(out.size(0), m) << "out must be [M, N]";
  TVM_FFI_ICHECK_EQ(out.size(1), a.rows) << "out must be [M, N] with N = the weight rows";
  TVM_FFI_ICHECK_EQ(out.stride(1), 1) << "out rows must be contiguous";
  TVM_FFI_ICHECK(x.stride(1) == 1) << "x rows must be contiguous";
  if (fp4_x) {
    TVM_FFI_ICHECK(x.dtype() == dl_uint8 && x_sf.has_value()) << "an FP4 x is uint8 codes + scales";
    CHECK_CUDA(x_sf.value());
    CHECK_DEVICE(x_sf.value(), x);
  } else {
    TVM_FFI_ICHECK(x_fmt == int64_t(XFmt::kBf16) || x_fmt == int64_t(XFmt::kF16))
        << "x_fmt " << x_fmt << " is not a dense input format";
    TVM_FFI_ICHECK((x_fmt == int64_t(XFmt::kF16)) == kModuleF16)
        << "a 16-bit x must have the wgmma operand type of the module (f16 = " << kModuleF16 << ")";
    TVM_FFI_ICHECK(x.dtype() == (kModuleF16 ? dl_float16 : dl_bfloat16))
        << "x dtype and x_fmt differ";
    TVM_FFI_ICHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0)
        << "a 16-bit x needs 16-byte aligned rows";
  }
  if (tok_scale.has_value()) {
    CHECK_INPUT_AND_TYPE(tok_scale.value(), dl_float32);
    CHECK_DEVICE(tok_scale.value(), x);
    TVM_FFI_ICHECK_EQ(tok_scale.value().numel(), m) << "tok_scale must be [M]";
  }

  const APath ap = a_path ? APath::kN2 : APath::kN1;
  int stage_of[kNumShapes];
  if (a_path) {
    stage_table<APath::kN2>(stage_of);
  } else {
    stage_table<APath::kN1>(stage_of);
  }
  int shape_index = -1;
  for (int i = 0; i < kNumShapes; ++i) {
    if (kShapes[i].cta_m == cta_m && kShapes[i].n == n) shape_index = i;
  }
  TVM_FFI_ICHECK(shape_index >= 0) << "tactic (" << cta_m << ", " << n << ") is not a D7 shape";
  TVM_FFI_ICHECK(stages == 0 || stages == stage_of[shape_index])
      << "tactic stages " << stages << " differ from the module's " << stage_of[shape_index];
  TVM_FFI_ICHECK(stage_of[shape_index] >= 2)
      << "tactic (" << cta_m << ", " << n << ") holds fewer than two stages in this module";
  if (m == 0) return;

  ffi::CUDADeviceGuard guard(x.device().device_id);
  const cudaStream_t stream = get_stream(x.device());
  int num_sms = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().device_id),
      cudaSuccess)
      << "cudaDeviceGetAttribute(SM count)";

  GemmArgs g{};
  g.a = a;
  g.b.tokens = m;
  g.b.k = k;
  g.sched.max_tiles = (m + static_cast<int>(n) - 1) / static_cast<int>(n);  // dense: expert 0
  if (fp4_x) {
    // prep restores the FP4 activations to exact BF16/FP16 values in expert-order B tiles
    const size_t need = x_tiles_bytes(m, k, n);
    TVM_FFI_ICHECK_GE(static_cast<size_t>(workspace.numel()), need)
        << "workspace must hold " << need << " bytes";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % kAlign, 0u)
        << "workspace must be 256 B aligned";
    PrepArgs prep{};
    prep.topk_ids = nullptr;  // dense
    prep.tokens = m;
    prep.top_k = 1;
    prep.num_experts = 1;
    prep.num_local_experts = 1;
    prep.x = x.data_ptr();
    prep.x_sf = static_cast<const uint8_t*>(x_sf.value().data_ptr());
    prep.x_ld = x.stride(0);  // bytes of an FP4 row
    prep.k = k;
    prep.x_fmt = static_cast<XFmt>(x_fmt);
    prep.f16 = kModuleF16;
    prep.n = static_cast<int>(n);
    prep.max_tiles = g.sched.max_tiles;
    prep.x_tiles = workspace.data_ptr();
    launch_fp4_sm90_prep(prep, stream);
    g.b.x = workspace.data_ptr();
    g.b.tiled = 1;
  } else {
    g.b.x = x.data_ptr();
    g.b.x_ld = x.stride(0);
    g.b.tiled = 0;
  }

  g.epi.alpha = static_cast<const float*>(alpha.data_ptr());
  g.epi.tok_scale =
      tok_scale.has_value() ? static_cast<const float*>(tok_scale.value().data_ptr()) : nullptr;
  g.epi.out = out.data_ptr();
  g.epi.out_ld = out.stride(0);
  g.epi.fq_global = nullptr;
  g.epi.act = kActIdentity;
  g.epi.fq = static_cast<uint8_t>(Fq::kNone);
  g.epi.recipe = 0;
  g.epi.out_f16 = out.dtype() == dl_float16;
  g.epi.out_tiled = 0;

  const int launched = ap == APath::kN1
                           ? dispatch_gemm<APath::kN1>(g, static_cast<int>(cta_m),
                                                       static_cast<int>(n), num_sms, stream)
                           : dispatch_gemm<APath::kN2>(g, static_cast<int>(cta_m),
                                                       static_cast<int>(n), num_sms, stream);
  TVM_FFI_ICHECK_GE(launched, 2) << "the GEMM of tactic (" << cta_m << ", " << n
                                 << ") is not built";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_dense_sm90_run, fp4_dense_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_dense_sm90_workspace_bytes, fp4_dense_sm90_workspace_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp4_dense_sm90_tactics, fp4_dense_sm90_tactics);

#endif  // FP4_DENSE_SM90_PART
