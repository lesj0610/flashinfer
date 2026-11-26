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
// Producer / consumer loops of the SM90 FP4 GEMM (swap-AB: weights are the wgmma A operand in
// registers, dequantized per k16 step; tokens are the B operand in shared memory).
//
// Shared memory (base 1024-aligned):
//   stage s (kStages of them, 1024-aligned): [A: ASmem<Cfg>][B: kN rows x K 128, two SW128 halves]
//   epilogue staging (Epilogue::kStagingBytes)
//   barriers: full[kStages] (load), empty[kStages], (n2) full2[kStages] (re-staged A)
//
// Pipeline (one global counter `it` over (tile, K tile); stage it % kStages, parity it / kStages):
//   producer warp group: wait empty -> thread 0 arms full with the TMA / bulk bytes and issues
//     them; every thread issues its cp.async (scales, fallback codes, gathered B) and arrives with
//     cp.async.mbarrier.arrive.noinc. full counts 1 + 128 (+ 128 for byte-loaded scales).
//     (n2): a stage is re-staged kLag iterations after its load (wait full, re-stage, arrive
//     full2 with 128 arrivals); kLag = kStages - 2 keeps kStages - 1 loads in flight, kStages 2
//     re-stages right after the load.
//   consumer warp groups: wait full (and full2) -> fence.proxy.async (cp.async B is a generic
//     proxy write read by wgmma) -> per k16 step: dequantize into A registers, wgmma.fence,
//     mma_async per m64 block, commit, wait_group 1. A registers are double buffered by step
//     parity; the previous stage is released (empty, one arrival per warp) after the wait_group 1
//     of step 0, i.e. once its last wgmma has completed. The accumulators are read only after
//     wait_group 0.
#pragma once

#include <cuda.h>

#include <cstdint>
#include <type_traits>

#include "fp4_sm90/a_layout.cuh"
#include "fp4_sm90/dequant.cuh"
#include "fp4_sm90/epilogue.cuh"
#include "fp4_sm90/params.cuh"
#include "fp4_sm90/wgmma.cuh"

namespace flashinfer {
namespace fp4_sm90 {

template <class Cfg>
struct SmemPlan {
  static constexpr bool kN2 = Cfg::kAPath == APath::kN2;
  static constexpr int kBOff = ASmem<Cfg>::kBytes;
  static constexpr int kBBytes = Cfg::kN * 256;
  static constexpr int kStageBytes = (kBOff + kBBytes + 1023) / 1024 * 1024;
  static constexpr int kStagingOff = Cfg::kStages * kStageBytes;
  static constexpr int kBarsPerStage = kN2 ? 3 : 2;
  static constexpr int kBarBytes = Cfg::kStages * kBarsPerStage * 8;
  // (n2) re-staging lag in iterations (see above).
  static constexpr int kLag = Cfg::kStages >= 3 ? Cfg::kStages - 2 : 0;
};

// Configuration seen by the epilogue: the accumulators carry value * 2^-fold (dequant.cuh), the
// epilogue scales them by kAccScale. kStagingN is the D7 staging chunk Nc in {32, 16, 8} (at most
// N): the largest whose total shared memory (fp4_sm90_gemm_smem_bytes, with this kernel's stage
// and barrier bytes) fits kSmemLimit; 0 when none fits.
template <class C>
struct EpiCfg : C {
  static constexpr float kAccScale = float(1u << dq_fold_log2_fmt<C::kAFmt, C::kF16, 0>());
  static constexpr int pick_staging_n() {
    constexpr int kWgCols = epi_gated(C::kEpi) ? C::kCtaM / 4 : C::kCtaM / 2;
    for (int nc = 32; nc >= 8; nc /= 2) {
      const int m = nc < C::kN ? nc : C::kN;
      const long bytes = 1024L + long(C::kStages) * SmemPlan<C>::kStageBytes +
                         2L * kWgCols * m * 2 + SmemPlan<C>::kBarBytes;
      if (bytes <= long(kSmemLimit)) return m;
    }
    return 0;
  }
  static constexpr int kStagingN = pick_staging_n();
};

// Epilogue staging bytes, or -1 when the epilogue has no staging chunk at this stage count (the
// configuration does not fit; Epilogue<> is not instantiated then).
template <class Cfg>
constexpr int epi_staging_bytes() {
  using E = EpiCfg<Cfg>;
  if constexpr (epi_fq(Cfg::kEpi) == Fq::kNvfp4 && epi::StagingN<E>::value == 0) {
    return -1;
  } else {
    return Epilogue<E>::kStagingBytes;
  }
}

struct Bars {
  uint32_t full, empty, full2;  // stage s at + 8 s
};

template <int I, int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    static_for<I + 1, N>(f);
  }
}

// ---------------------------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------------------------
struct TileInfo {
  int t;        // n-tile
  int m0;       // first kernel row
  int e;        // local expert
  int n0;       // first expanded row (token)
  int row_end;  // end of the expert's rows
  int n_valid;  // valid rows of the n-tile
};

__device__ __forceinline__ int sched_n_tiles(const Sched& s) {
  return s.num_tiles != nullptr ? __ldg(s.num_tiles) : s.max_tiles;
}

template <class Cfg>
__device__ __forceinline__ TileInfo tile_info(const GemmArgs& g, int tile, int m_tiles) {
  TileInfo ti;
  ti.t = tile / m_tiles;
  ti.m0 = (tile - ti.t * m_tiles) * Cfg::kCtaM;
  ti.e = g.sched.tile_expert != nullptr ? __ldg(g.sched.tile_expert + ti.t) : 0;
  ti.n0 = g.sched.tile_n0 != nullptr ? __ldg(g.sched.tile_n0 + ti.t) : ti.t * Cfg::kN;
  ti.row_end =
      g.sched.expert_offset != nullptr ? __ldg(g.sched.expert_offset + ti.e + 1) : g.b.tokens;
  const int v = ti.row_end - ti.n0;
  ti.n_valid = v < 0 ? 0 : (v > Cfg::kN ? Cfg::kN : v);
  return ti;
}

// ---------------------------------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------------------------------
// Gathered B (BSrc::tiled == 0, a single expert): 16-byte chunk c of row r -> b_tile_byte
// layout; rows past BSrc::tokens and K past BSrc::k are zero-filled.
template <class Cfg>
__device__ __forceinline__ void b_gather(const BSrc& b, uint32_t dst, int n0, int kt, int tid) {
  constexpr int kN = Cfg::kN;
  const int c = tid & 15, nb = tid >> 4;
  const int k = kt * 128 + 8 * c;
  const bool k_ok = k < b.k;
  const uint32_t d0 = dst + uint32_t((c >> 3) * kN * 128 + nb * 128 + (((c & 7) ^ nb) << 4));
  const uint8_t* x = static_cast<const uint8_t*>(b.x);
#pragma unroll 4
  for (int i = 0; i < kN / 8; ++i) {
    const int p = n0 + nb + 8 * i;
    const bool ok = k_ok && p < b.tokens;
    const uint8_t* src = x;
    if (ok) src = x + (int64_t(p) * b.x_ld + k) * 2;
    ptx::cp_async_16(d0 + uint32_t(i * 1024), src, ok ? 16u : 0u);
  }
}

// (n2) re-staging of iteration j's stage.
template <class Cfg>
__device__ __forceinline__ void producer_restage(const GemmArgs& g, uint32_t smem, const Bars& bars,
                                                 int j, int tid) {
  using P = SmemPlan<Cfg>;
  const int s = j % Cfg::kStages;
  const uint32_t ph = uint32_t(j / Cfg::kStages) & 1u;
  ptx::mbar_wait(bars.full + 8u * s, ph);
  a_prepack<Cfg>(smem + uint32_t(s * P::kStageBytes), g.a.pair, g.a.pair_g, tid);
  ptx::mbar_arrive(bars.full2 + 8u * s);
}

// kATma: the A codes arrive by TMA (otherwise producer cp.async). A template argument because a
// runtime flag stays live across the k loop in the 40-register producer and spills.
template <class Cfg, bool kATma>
__device__ __forceinline__ void producer_loop_mode(const GemmArgs& g, const CUtensorMap* map0,
                                                   const CUtensorMap* map1, uint32_t smem,
                                                   const Bars& bars, int tid) {
  using P = SmemPlan<Cfg>;
  constexpr bool kGated = epi_gated(Cfg::kEpi);
  const ACodeLoad cmode = a_code_load(g.a, Cfg::kAFmt, kGated);
  const ASfLoad smode = a_sf_load(g.a, Cfg::kAFmt);
  constexpr bool a_tma = kATma;
  const bool a_halves = cmode == ACodeLoad::kTmaHalves;
  const bool b_bulk = g.b.tiled != 0;
  const uint32_t tx =
      (a_tma ? uint32_t(Cfg::kCtaM * 64) : 0u) + (b_bulk ? uint32_t(P::kBBytes) : 0u);
  const int m_tiles = (g.a.rows + Cfg::kCtaM - 1) / Cfg::kCtaM;
  const int k_tiles = (g.a.k + 127) / 128;
  const int b_k_tiles = (g.b.k + 127) / 128;
  const int total = sched_n_tiles(g.sched) * m_tiles;
  if (tid == 0 && a_tma) {
    ptx::prefetch_tmap(map0);
    if (a_halves) ptx::prefetch_tmap(map1);
  }
  int it = 0;
#pragma unroll 1
  for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
    const TileInfo ti = tile_info<Cfg>(g, tile, m_tiles);
#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const int s = it % Cfg::kStages;
      const uint32_t ph = uint32_t(it / Cfg::kStages) & 1u;
      const uint32_t st = smem + uint32_t(s * P::kStageBytes);
      const uint32_t full = bars.full + 8u * s;
      ptx::mbar_wait(bars.empty + 8u * s, ph ^ 1u);
      if (tid == 0) {
        if (tx != 0) {
          ptx::mbar_arrive_expect_tx(full, tx);
        } else {
          ptx::mbar_arrive(full);
        }
        if (a_tma) a_issue_tma<Cfg>(a_halves, map0, map1, st, ti.e, ti.m0, kt, full);
        if (b_bulk) {
          const uint8_t* src = static_cast<const uint8_t*>(g.b.x) +
                               b_tile_byte(ti.t, b_k_tiles, kt, 0, Cfg::kN, 0, 0);
          ptx::bulk_g2s(st + uint32_t(P::kBOff), src, uint32_t(P::kBBytes), full);
        }
      }
      if (!a_tma) a_codes_cp_async<Cfg>(g.a, st, ti.e, ti.m0, kt, tid);
      a_sf_load_stage<Cfg>(g.a, smode, st + uint32_t(ASmem<Cfg>::kSfOff), ti.e, ti.m0, kt, tid);
      if (!b_bulk) b_gather<Cfg>(g.b, st + uint32_t(P::kBOff), ti.n0, kt, tid);
      ptx::cp_async_arrive_noinc(full);
      if (smode == ASfLoad::kBytes) ptx::mbar_arrive(full);
      if constexpr (P::kN2) {
        if (it >= P::kLag) producer_restage<Cfg>(g, smem, bars, it - P::kLag, tid);
      }
    }
  }
  if constexpr (P::kN2) {
#pragma unroll 1
    for (int j = it - P::kLag > 0 ? it - P::kLag : 0; j < it; ++j) {
      producer_restage<Cfg>(g, smem, bars, j, tid);
    }
  }
  ptx::cp_async_wait_all();
}

template <class Cfg>
__device__ __forceinline__ void producer_loop(const GemmArgs& g, const CUtensorMap* map0,
                                              const CUtensorMap* map1, uint32_t smem,
                                              const Bars& bars, int tid) {
  if (a_code_load_is_tma(a_code_load(g.a, Cfg::kAFmt, epi_gated(Cfg::kEpi)))) {
    producer_loop_mode<Cfg, true>(g, map0, map1, smem, bars, tid);
  } else {
    producer_loop_mode<Cfg, false>(g, map0, map1, smem, bars, tid);
  }
}

// ---------------------------------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------------------------------
template <class Cfg>
__device__ __forceinline__ void consumer_loop(const GemmArgs& g, uint32_t smem, const Bars& bars,
                                              int wg, int warp, int lane) {
  using P = SmemPlan<Cfg>;
  using Epi = Epilogue<EpiCfg<Cfg>>;
  constexpr AFmt kF = Cfg::kAFmt;
  constexpr int kN = Cfg::kN;
  constexpr int kM64 = Cfg::kM64PerWarpGroup;
  constexpr bool kF16 = Cfg::kF16;
  constexpr bool kSm12x = kF == AFmt::kDenseCuteSm12x;
  using Mma = ptx::WgmmaRS<kN, kF16>;

  AFragRows fr[kM64];
#pragma unroll
  for (int mi = 0; mi < kM64; ++mi) {
    fr[mi] = a_frag_init<Cfg>(g.a.pair, g.a.pair_g, wg * kM64 + mi, warp, lane);
  }
  const uint32_t selq = sel_q(lane & 3);
  const uint32_t staging = smem + uint32_t(P::kStagingOff);
  const int m_tiles = (g.a.rows + Cfg::kCtaM - 1) / Cfg::kCtaM;
  const int k_tiles = (g.a.k + 127) / 128;
  const int total = sched_n_tiles(g.sched) * m_tiles;

  float acc[Cfg::kAccPerThread];
  uint32_t areg[2][kM64][4];
  int it = 0;
#pragma unroll 1
  for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
    const TileInfo ti = tile_info<Cfg>(g, tile, m_tiles);
#pragma unroll
    for (int i = 0; i < Cfg::kAccPerThread; ++i) acc[i] = 0.f;
    ptx::fence_operands(acc);
#pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt, ++it) {
      const int s_stage = it % Cfg::kStages;
      const uint32_t ph = uint32_t(it / Cfg::kStages) & 1u;
      const uint32_t st = smem + uint32_t(s_stage * P::kStageBytes);
      ptx::mbar_wait(bars.full + 8u * s_stage, ph);
      if constexpr (P::kN2) ptx::mbar_wait(bars.full2 + 8u * s_stage, ph);
      ptx::fence_proxy_async_smem();
      const uint32_t b_st = st + uint32_t(P::kBOff);
      const uint32_t prev_empty =
          bars.empty + 8u * uint32_t((it + Cfg::kStages - 1) % Cfg::kStages);

      uint32_t sf[kM64][4];
#pragma unroll
      for (int mi = 0; mi < kM64; ++mi) a_load_sf<Cfg>(st, fr[mi], sf[mi]);

      // Code words of this stage: (n2) 2 x LDS.128 per m64 block; native: 16-byte chunks of
      // rows g / g + 8, double buffered over chunk pairs.
      uint4 cw[kM64][2];
      uint4 ch[2][kM64][2];
      if constexpr (P::kN2) {
#pragma unroll
        for (int mi = 0; mi < kM64; ++mi) {
          cw[mi][0] = ptx::lds128(st + fr[mi].c0);
          cw[mi][1] = ptx::lds128(st + fr[mi].c1);
        }
      } else if constexpr (!kSm12x) {
#pragma unroll
        for (int mi = 0; mi < kM64; ++mi) a_load_chunk(st, fr[mi], 0, ch[0][mi][0], ch[0][mi][1]);
      }

      static_for<0, 8>([&](auto step) {
        constexpr int s = decltype(step)::value;
        if constexpr (!P::kN2 && !kSm12x && (s & 1) == 0 && s < 6) {
#pragma unroll
          for (int mi = 0; mi < kM64; ++mi) {
            a_load_chunk(st, fr[mi], (s >> 1) + 1, ch[((s >> 1) + 1) & 1][mi][0],
                         ch[((s >> 1) + 1) & 1][mi][1]);
          }
        }
#pragma unroll
        for (int mi = 0; mi < kM64; ++mi) {
          uint32_t w8;
          if constexpr (P::kN2) {
            w8 = u4_word(cw[mi][s >> 2], s & 3);
          } else if constexpr (kSm12x) {
            w8 = w8_sm12x(st + uint32_t(s * Cfg::kCtaM * 8), fr[mi].c0);
          } else {
            w8 = w8_from_rows(ch[(s >> 1) & 1][mi][0], ch[(s >> 1) & 1][mi][1], (s & 1) * 2, selq);
          }
          dq_step<kF, kF16, 0>(w8, sf_pair<kF>(sf[mi], s), areg[s & 1][mi]);
        }
        const uint64_t desc = ptx::b_desc<kN>(b_st, s);
        ptx::wgmma_fence();
        Mma::template run<0>(acc, areg[s & 1][0], desc);
        if constexpr (kM64 == 2) Mma::template run<kN / 2>(acc, areg[s & 1][kM64 - 1], desc);
        ptx::wgmma_commit();
        ptx::wgmma_wait<1>();
        if constexpr (s == 0) {
          // The previous stage is free once the wait above retired its last wgmma; the wait is
          // .sync.aligned, so every lane of the warp is past its reads of that stage.
          ptx::mbar_arrive_if(prev_empty, uint32_t(kt > 0 && lane == 0));
        }
      });
    }
    ptx::wgmma_wait<0>();
    ptx::fence_operands(acc);
    __syncwarp();
    if (lane == 0) {
      ptx::mbar_arrive(bars.empty + 8u * uint32_t((it + Cfg::kStages - 1) % Cfg::kStages));
    }
    Epi::apply(acc, g, ti.e, ti.m0, ti.t, ti.n0, ti.n_valid, wg, warp, lane, staging);
  }
}

}  // namespace fp4_sm90
}  // namespace flashinfer
