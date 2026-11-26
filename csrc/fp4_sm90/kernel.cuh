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
// SM90 FP4 GEMM kernel (warp-specialized, persistent) and its host entry points:
//   fp4_sm90_gemm_smem_bytes<Cfg>()  dynamic shared memory of a configuration (callers drop the
//                                    configurations above kSmemLimit)
//   launch_fp4_sm90_gemm<Cfg>()      argument checks, A tensor maps, launch on `stream`
// CTA: warp group 0 producer (setmaxnreg.dec 40), warp groups 1, 2 consumers (setmaxnreg.inc 232,
// epilogue warp groups 0, 1). Tile t * m_tiles + m_tile, grid min(tiles, num_sms).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "fp4_sm90/a_layout.cuh"
#include "fp4_sm90/mainloop.cuh"
#include "fp4_sm90/params.cuh"
#include "fp4_sm90/wgmma.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace fp4_sm90 {

template <class Cfg>
constexpr size_t fp4_sm90_gemm_smem_bytes() {
  using P = SmemPlan<Cfg>;
  constexpr int kStaging = epi_staging_bytes<Cfg>();
  if constexpr (kStaging < 0) {
    return ~size_t(0) >> 1;  // no epilogue staging fits: never selected
  } else {
    return size_t(1024) + size_t(Cfg::kStages) * size_t(P::kStageBytes) + size_t(kStaging) +
           size_t(P::kBarBytes);
  }
}

template <class Cfg>
__global__ void __launch_bounds__(384, 1)
    fp4_sm90_gemm(const __grid_constant__ GemmArgs args, const __grid_constant__ CUtensorMap map0,
                  const __grid_constant__ CUtensorMap map1) {
  using P = SmemPlan<Cfg>;
  static_assert(Cfg::kThreads == 384, "one producer and two consumer warp groups");
  static_assert(fp4_sm90_gemm_smem_bytes<Cfg>() <= size_t(kSmemLimit), "shared memory");
  constexpr int kStaging = epi_staging_bytes<Cfg>();
  extern __shared__ __align__(1024) uint8_t fp4_sm90_smem[];
  const uint32_t smem = (ptx::smem_u32(fp4_sm90_smem) + 1023u) & ~1023u;
  const uint32_t bar0 = smem + uint32_t(P::kStagingOff + kStaging);
  const Bars bars{bar0, bar0 + 8u * Cfg::kStages, bar0 + 16u * Cfg::kStages};
  const int tid = threadIdx.x;
  if (tid == 0) {
    // full: thread 0 (expect_tx or plain) + 128 cp.async noinc (+ 128 byte-loaded scale stores).
    const uint32_t full_count =
        129u + (a_sf_load(args.a, Cfg::kAFmt) == ASfLoad::kBytes ? 128u : 0u);
#pragma unroll 1
    for (int s = 0; s < Cfg::kStages; ++s) {
      ptx::mbar_init(bars.full + 8u * s, full_count);
      ptx::mbar_init(bars.empty + 8u * s, 8u);  // one arrival per consumer warp
      if constexpr (P::kN2) ptx::mbar_init(bars.full2 + 8u * s, 128u);
    }
    ptx::fence_mbar_init();
  }
  __syncthreads();
  if (tid < 128) {
    ptx::setmaxnreg_dec<Cfg::kProducerRegs>();
    producer_loop<Cfg>(args, &map0, &map1, smem, bars, tid);
  } else {
    ptx::setmaxnreg_inc<Cfg::kConsumerRegs>();
    const int ct = tid - 128;
    consumer_loop<Cfg>(args, smem, bars, ct >> 7, (ct >> 5) & 3, ct & 31);
  }
}

// Checks of the arguments the kernel relies on (layout contracts of params.cuh).
template <class Cfg>
inline void check_fp4_sm90_gemm_args(const GemmArgs& g) {
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  constexpr AFmt kF = Cfg::kAFmt;
  const ASrc& a = g.a;
  const BSrc& b = g.b;
  TVM_FFI_CHECK(a.fmt == kF, ValueError) << "SM90 FP4 GEMM: weight format " << int(a.fmt)
                                         << " does not match the kernel (" << int(kF) << ")";
  TVM_FFI_CHECK(a.rows > 0 && a.k > 0 && a.rows_phys > 0, ValueError)
      << "SM90 FP4 GEMM: empty weight (rows " << a.rows << ", k " << a.k << ")";
  TVM_FFI_CHECK(a.k % sf_block(kF) == 0, ValueError)
      << "SM90 FP4 GEMM: K " << a.k << " must be a multiple of the scale block " << sf_block(kF);
  TVM_FFI_CHECK(a.k_phys >= a.k && a.k_phys % 16 == 0, ValueError)
      << "SM90 FP4 GEMM: physical K " << a.k_phys << " must be >= K " << a.k
      << " and a multiple of 16";
  TVM_FFI_CHECK(a.codes != nullptr && a.sf != nullptr, ValueError)
      << "SM90 FP4 GEMM: missing weight codes or scales";
  if (kGated) {
    TVM_FFI_CHECK(a.pair != Pair::kNone && a.rows == 2 * a.half_rows, ValueError)
        << "SM90 FP4 GEMM: gated weights need a gate/up pairing and rows == 2 * half_rows";
  } else {
    TVM_FFI_CHECK(a.pair == Pair::kNone, ValueError)
        << "SM90 FP4 GEMM: non-gated weights take no gate/up pairing";
  }
  if (afmt_trtllm(kF)) {
    TVM_FFI_CHECK(a.rows_phys % 32 == 0, ValueError)
        << "SM90 FP4 GEMM: TRTLLM-shuffled weights need physical rows % 32 == 0";
    TVM_FFI_CHECK(!kGated || a.pair == Pair::kTrtllmW31, ValueError)
        << "SM90 FP4 GEMM: gated TRTLLM-shuffled weights use the w3/w1 row pairing";
  }
  if (a.pair == Pair::kTrtllmW31) {
    TVM_FFI_CHECK(afmt_trtllm(kF), ValueError)
        << "SM90 FP4 GEMM: the w3/w1 pairing needs TRTLLM-shuffled weights";
  }
  if (a.pair == Pair::kGroupInterleave) {
    TVM_FFI_CHECK(a.pair_g > 0 && a.pair_g % 8 == 0 && (Cfg::kCtaM / 2) % a.pair_g == 0, ValueError)
        << "SM90 FP4 GEMM: group interleave size " << a.pair_g << " must divide " << Cfg::kCtaM / 2
        << " and be a multiple of 8";
  }
  if (kF == AFmt::kDenseCuteSm12x) {
    TVM_FFI_CHECK(!kGated && a.codes_expert_stride == 0 && a.rows_phys % 64 == 0 &&
                      a.k_phys % 16 == 0 && (reinterpret_cast<uintptr_t>(a.codes) & 15) == 0,
                  ValueError)
        << "SM90 FP4 GEMM: CuTe SM12x packed weights are dense, rows % 64 == 0, 16-byte aligned";
  }
  const ACodeLoad cmode = a_code_load(a, kF, kGated);
  if (cmode == ACodeLoad::kCpAsync) {
    TVM_FFI_CHECK((reinterpret_cast<uintptr_t>(a.codes) & 7) == 0 && a.codes_expert_stride % 8 == 0,
                  ValueError)
        << "SM90 FP4 GEMM: weight codes must be 8-byte aligned";
  }
  if (a_sf_load(a, kF) == ASfLoad::kCp4) {
    TVM_FFI_CHECK((reinterpret_cast<uintptr_t>(a.sf) & 3) == 0 && a.sf_expert_stride % 4 == 0,
                  ValueError)
        << "SM90 FP4 GEMM: weight scales must be 4-byte aligned";
  }
  TVM_FFI_CHECK(b.x != nullptr && (reinterpret_cast<uintptr_t>(b.x) & 15) == 0, ValueError)
      << "SM90 FP4 GEMM: activations must be 16-byte aligned";
  TVM_FFI_CHECK(b.k == a.k, ValueError)
      << "SM90 FP4 GEMM: activation K " << b.k << " != weight K " << a.k;
  if (b.tiled == 0) {
    TVM_FFI_CHECK(b.x_ld >= b.k && b.x_ld % 8 == 0, ValueError)
        << "SM90 FP4 GEMM: activation row stride " << b.x_ld << " must be >= K and % 8 == 0";
    TVM_FFI_CHECK(g.sched.expert_offset == nullptr, ValueError)
        << "SM90 FP4 GEMM: row-major activations take a single expert";
  }
  TVM_FFI_CHECK(g.sched.max_tiles >= 0, ValueError) << "SM90 FP4 GEMM: negative max_tiles";
  TVM_FFI_CHECK(g.sched.tile_expert == nullptr || g.sched.expert_offset != nullptr, ValueError)
      << "SM90 FP4 GEMM: an expert schedule needs expert_offset";
}

template <class Cfg>
void launch_fp4_sm90_gemm(const GemmArgs& args, int num_sms, cudaStream_t stream) {
  constexpr size_t kSmem = fp4_sm90_gemm_smem_bytes<Cfg>();
  static_assert(kSmem <= size_t(kSmemLimit), "configuration exceeds the shared memory limit");
  check_fp4_sm90_gemm_args<Cfg>(args);
  const int m_tiles = (args.a.rows + Cfg::kCtaM - 1) / Cfg::kCtaM;
  const int64_t tiles = int64_t(m_tiles) * args.sched.max_tiles;
  if (tiles == 0) return;
  TVM_FFI_CHECK(num_sms > 0, ValueError) << "SM90 FP4 GEMM: num_sms must be positive";
  CUtensorMap map0, map1;
  make_a_tmaps(args.a, Cfg::kAFmt, epi_gated(Cfg::kEpi), Cfg::kCtaM, &map0, &map1);
  auto* kernel = fp4_sm90_gemm<Cfg>;
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmem));
  TVM_FFI_CHECK(err == cudaSuccess, RuntimeError)
      << "SM90 FP4 GEMM: cudaFuncSetAttribute failed: " << cudaGetErrorString(err);
  const int grid = int(tiles < num_sms ? tiles : num_sms);
  kernel<<<grid, Cfg::kThreads, kSmem, stream>>>(args, map0, map1);
  err = cudaGetLastError();
  TVM_FFI_CHECK(err == cudaSuccess, RuntimeError)
      << "SM90 FP4 GEMM: launch failed: " << cudaGetErrorString(err);
}

}  // namespace fp4_sm90
}  // namespace flashinfer
