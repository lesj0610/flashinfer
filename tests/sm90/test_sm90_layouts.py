"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SM90 FP4 GEMM weight (A) loading (``csrc/fp4_sm90/a_layout.cuh``): a dump kernel runs the real
producer loads of one stage (TMA through ``make_a_tmaps``, cp.async fallbacks, scale loads,
(n2) re-staging) and the consumer fragment reads, and writes every consumer thread's W8 word and
scale pair per k16 step. The expected words come from the logical weights: kernel row ->
logical row (gate / up pairing of the GEMM1 contract), FP4 codes of the wgmma A fragment,
scales of the row; physical layouts are built with FlashInfer's own row helpers
(``get_shuffle_matrix_a_row_indices``, ``get_reorder_rows_for_gated_act_gemm_row_indices``,
``_interleave_linear_and_gate``, ``_cute_dsl_pack_fp4_weight``) and the 128x4 scale offset
(checked against ``block_scale_interleave`` in the F1 layout check). Physical padding (rows,
K, scale columns) holds random bytes: codes there are not compared, scales there must read 0.

Every weight format and gate / up pairing, CTA_M 128 / 256, both A paths ((n1) native, (n2)
re-staged), aligned K (TMA), padded K (cp.async fallback, byte-loaded linear scales) and a dense
weight. The loader mode the host selects is checked. Shared-memory bank wavefronts of the
consumer loads (from the dumped per-thread offsets) are printed as ``SM90-LDS-BANK`` records.
"""

import functools
import math

import pytest
import torch

from flashinfer.fused_moe.core import get_reorder_rows_for_gated_act_gemm_row_indices
from flashinfer.fused_moe.prepare import _interleave_linear_and_gate
from flashinfer.fused_moe.sm90.spec import AFmt, Pair
from flashinfer.gemm.gemm_bf16_fp4_cute_dsl import _cute_dsl_pack_fp4_weight
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec, sm90a_nvcc_flags
from flashinfer.jit.utils import write_if_different
from flashinfer.utils import get_compute_capability, get_shuffle_matrix_a_row_indices

_SOURCE = r"""
#include "fp4_sm90/a_layout.cuh"
#include "tvm_ffi_utils.h"

namespace fs = flashinfer::fp4_sm90;
using fs::AFmt;
using fs::APath;
using fs::ASrc;
using fs::Pair;
using tvm::ffi::Array;

// out:  [E, m_tiles, k_tiles, 2 (warp group), kM64, 4 (warp), 32 (lane), 8 (k16 step), 2]
//       = {W8, scale pair}
// frag: [2, kM64, 4, 32, 6] = AFragRows {c0, c1, x0, x1, sf0, sf1} (CTA (0, 0, 0))
template <class C>
__global__ void __launch_bounds__(384, 1)
    dump_kernel(const __grid_constant__ ASrc a, const __grid_constant__ CUtensorMap map0,
                const __grid_constant__ CUtensorMap map1, uint32_t* out, uint32_t* frag) {
  constexpr bool kGated = fs::epi_gated(C::kEpi);
  constexpr int kM64 = C::kM64PerWarpGroup;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t st = (fs::ptx::smem_u32(smem_raw) + 1023u) & ~1023u;
  const uint32_t bar = st + uint32_t(fs::ASmem<C>::kBytes);
  const int kt = blockIdx.x, mt = blockIdx.y, e = blockIdx.z;
  const int m0 = mt * C::kCtaM, tid = threadIdx.x;
  const fs::ACodeLoad cmode = fs::a_code_load(a, C::kAFmt, kGated);
  const fs::ASfLoad smode = fs::a_sf_load(a, C::kAFmt);
  const bool tma = fs::a_code_load_is_tma(cmode);
  if (tid == 0) {
    fs::ptx::mbar_init(bar, 129u + (smode == fs::ASfLoad::kBytes ? 128u : 0u));
    fs::ptx::fence_mbar_init();
  }
  __syncthreads();
  if (tid < 128) {
    if (tid == 0) {
      if (tma) {
        fs::ptx::mbar_arrive_expect_tx(bar, uint32_t(C::kCtaM * 64));
        fs::a_issue_tma<C>(cmode == fs::ACodeLoad::kTmaHalves, &map0, &map1, st, e, m0, kt, bar);
      } else {
        fs::ptx::mbar_arrive(bar);
      }
    }
    if (!tma) fs::a_codes_cp_async<C>(a, st, e, m0, kt, tid);
    fs::a_sf_load_stage<C>(a, smode, st + uint32_t(fs::ASmem<C>::kSfOff), e, m0, kt, tid);
    fs::ptx::cp_async_arrive_noinc(bar);
    if (smode == fs::ASfLoad::kBytes) fs::ptx::mbar_arrive(bar);
  }
  fs::ptx::mbar_wait(bar, 0);
  if constexpr (C::kAPath == APath::kN2) {
    if (tid < 128) fs::a_prepack<C>(st, a.pair, a.pair_g, tid);
    __syncthreads();
  }
  if (tid < 128) return;
  const int ct = tid - 128, wg = ct >> 7, w = (ct >> 5) & 3, lane = ct & 31;
  const uint32_t selq = fs::sel_q(lane & 3);
  const size_t k_tiles = gridDim.x, m_tiles = gridDim.y;
#pragma unroll
  for (int mi = 0; mi < kM64; ++mi) {
    const fs::AFragRows f = fs::a_frag_init<C>(a.pair, a.pair_g, wg * kM64 + mi, w, lane);
    uint32_t sf[4];
    fs::a_load_sf<C>(st, f, sf);
    const size_t t =
        (((((size_t(e) * m_tiles + mt) * k_tiles + kt) * 2 + wg) * kM64 + mi) * 4 + w) * 32 +
        lane;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      out[(t * 8 + s) * 2] = fs::a_read_w8<C>(st, f, s, selq);
      out[(t * 8 + s) * 2 + 1] = fs::sf_pair<C::kAFmt>(sf, s);
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      uint32_t* fp = frag + size_t(((wg * kM64 + mi) * 4 + w) * 32 + lane) * 6;
      fp[0] = f.c0;
      fp[1] = f.c1;
      fp[2] = f.x0;
      fp[3] = f.x1;
      fp[4] = f.sf0;
      fp[5] = f.sf1;
    }
  }
}

// meta: fmt, pair, rows, rows_phys, k, k_phys, half_rows, pair_g, codes_expert_stride,
//       sf_expert_stride, experts, cta_m, a_path
ASrc make_src(const TensorView& codes, const TensorView& sf, const Array<int64_t>& m) {
  TVM_FFI_ICHECK_EQ(m.size(), 13);
  ASrc a{};
  a.codes = static_cast<const uint8_t*>(codes.data_ptr());
  a.sf = static_cast<const uint8_t*>(sf.data_ptr());
  a.fmt = static_cast<AFmt>(m[0]);
  a.pair = static_cast<Pair>(m[1]);
  a.rows = int(m[2]);
  a.rows_phys = int(m[3]);
  a.k = int(m[4]);
  a.k_phys = int(m[5]);
  a.half_rows = int(m[6]);
  a.pair_g = int(m[7]);
  a.codes_expert_stride = m[8];
  a.sf_expert_stride = m[9];
  a.num_experts = int(m[10]);
  return a;
}

template <AFmt AF, bool G, int CM, APath AP>
void launch_one(const ASrc& a, int experts, uint32_t* out, uint32_t* frag, cudaStream_t stream) {
  using C = fs::Cfg<CM, 8, 2, AF, false, fs::epi_code(fs::Fq::kNone, 0, G, false), AP>;
  CUtensorMap m0, m1;
  fs::make_a_tmaps(a, AF, G, CM, &m0, &m1);
  constexpr int kSmem = fs::ASmem<C>::kBytes + 1024 + 64;
  auto* kernel = dump_kernel<C>;
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
  const dim3 grid((a.k + 127) / 128, (a.rows + CM - 1) / CM, experts);
  kernel<<<grid, 384, kSmem, stream>>>(a, m0, m1, out, frag);
  err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

template <AFmt AF, bool G>
void by_cfg(const ASrc& a, int experts, int cta_m, int ap, uint32_t* out, uint32_t* frag,
            cudaStream_t s) {
  if (cta_m == 128) {
    ap ? launch_one<AF, G, 128, APath::kN2>(a, experts, out, frag, s)
       : launch_one<AF, G, 128, APath::kN1>(a, experts, out, frag, s);
  } else {
    TVM_FFI_ICHECK_EQ(cta_m, 256);
    ap ? launch_one<AF, G, 256, APath::kN2>(a, experts, out, frag, s)
       : launch_one<AF, G, 256, APath::kN1>(a, experts, out, frag, s);
  }
}

template <AFmt AF>
void by_gated(const ASrc& a, int experts, int cta_m, int ap, uint32_t* out, uint32_t* frag,
              cudaStream_t s) {
  const bool gated = a.pair != Pair::kNone;
  if constexpr (AF == AFmt::kDenseCuteSm12x) {
    TVM_FFI_ICHECK(!gated);
    by_cfg<AF, false>(a, experts, cta_m, ap, out, frag, s);
  } else {
    gated ? by_cfg<AF, true>(a, experts, cta_m, ap, out, frag, s)
          : by_cfg<AF, false>(a, experts, cta_m, ap, out, frag, s);
  }
}

int64_t modes(TensorView codes, TensorView sf, Array<int64_t> meta) {
  const ASrc a = make_src(codes, sf, meta);
  const bool gated = a.pair != Pair::kNone;
  return int64_t(fs::a_code_load(a, a.fmt, gated)) * 16 + int64_t(fs::a_sf_load(a, a.fmt));
}

void run(TensorView codes, TensorView sf, Array<int64_t> meta, TensorView out, TensorView frag) {
  const ASrc a = make_src(codes, sf, meta);
  const int experts = int(meta[10]), cta_m = int(meta[11]), ap = int(meta[12]);
  auto* o = static_cast<uint32_t*>(out.data_ptr());
  auto* f = static_cast<uint32_t*>(frag.data_ptr());
  ffi::CUDADeviceGuard guard(codes.device().device_id);
  const cudaStream_t s = get_stream(codes.device());
  switch (a.fmt) {
    case AFmt::kNvfp4TrtllmShuffled:
      by_gated<AFmt::kNvfp4TrtllmShuffled>(a, experts, cta_m, ap, o, f, s);
      break;
    case AFmt::kNvfp4Swz128x4:
      by_gated<AFmt::kNvfp4Swz128x4>(a, experts, cta_m, ap, o, f, s);
      break;
    case AFmt::kNvfp4LinearSf:
      by_gated<AFmt::kNvfp4LinearSf>(a, experts, cta_m, ap, o, f, s);
      break;
    case AFmt::kMxfp4TrtllmShuffled:
      by_gated<AFmt::kMxfp4TrtllmShuffled>(a, experts, cta_m, ap, o, f, s);
      break;
    case AFmt::kMxfp4Swz128x4:
      by_gated<AFmt::kMxfp4Swz128x4>(a, experts, cta_m, ap, o, f, s);
      break;
    case AFmt::kDenseCuteSm12x:
      by_gated<AFmt::kDenseCuteSm12x>(a, experts, cta_m, ap, o, f, s);
      break;
    default:
      TVM_FFI_ICHECK(false) << "unknown weight format " << int(a.fmt);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(modes, modes);
"""

_TRTLLM = (AFmt.NVFP4_TRTLLM_SHUFFLED, AFmt.MXFP4_TRTLLM_SHUFFLED)
_MX = (AFmt.MXFP4_TRTLLM_SHUFFLED, AFmt.MXFP4_SWZ128X4)
_SF128X4 = _TRTLLM + (AFmt.NVFP4_SWZ128X4, AFmt.MXFP4_SWZ128X4)

# (format, pairing, group size)
_COMBOS = [
    (AFmt.NVFP4_TRTLLM_SHUFFLED, Pair.NONE, 0),
    (AFmt.NVFP4_TRTLLM_SHUFFLED, Pair.TRTLLM_W31, 0),
    (AFmt.NVFP4_SWZ128X4, Pair.NONE, 0),
    (AFmt.NVFP4_SWZ128X4, Pair.UP_GATE_HALVES, 0),
    (AFmt.NVFP4_SWZ128X4, Pair.GROUP_INTERLEAVE, 64),
    (AFmt.NVFP4_SWZ128X4, Pair.GROUP_INTERLEAVE, 16),
    (AFmt.NVFP4_LINEAR_SF, Pair.NONE, 0),
    (AFmt.NVFP4_LINEAR_SF, Pair.UP_GATE_HALVES, 0),
    (AFmt.NVFP4_LINEAR_SF, Pair.GROUP_INTERLEAVE, 64),
    (AFmt.MXFP4_TRTLLM_SHUFFLED, Pair.NONE, 0),
    (AFmt.MXFP4_TRTLLM_SHUFFLED, Pair.TRTLLM_W31, 0),
    (AFmt.MXFP4_SWZ128X4, Pair.NONE, 0),
    (AFmt.MXFP4_SWZ128X4, Pair.UP_GATE_HALVES, 0),
    (AFmt.MXFP4_SWZ128X4, Pair.GROUP_INTERLEAVE, 64),
    (AFmt.DENSE_CUTE_SM12X, Pair.NONE, 0),
]

# (name, K, physical K, experts): aligned rows (TMA), padded rows (cp.async fallback, linear
# scales with a pitch that is not a multiple of 4), dense weight (no expert stride).
_SHAPES = [("aligned", 256, 256, 2), ("padded", 288, 304, 2), ("dense", 256, 256, 1)]

_NONGATED_ROWS, _NONGATED_ROWS_PHYS = 300, 320
_GATED_HALF = 192


def _device() -> torch.device:
    return torch.device("cuda:0")


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(_device()) != (9, 0):
        pytest.skip("the SM90 FP4 GEMM loaders need SM90 (sm_90a)")


@functools.cache
def _module():
    name = "test_sm90_fp4_layouts"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "layouts_test.cu"
    write_if_different(source, _SOURCE)
    return gen_jit_spec(
        name, [source], extra_cuda_cflags=sm90a_nvcc_flags
    ).build_and_load()


@pytest.fixture(scope="module")
def mod():
    _require_sm90()
    yield _module()
    print("WGEMU-TEST-END")


def _pad(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ----------------------------------------------------------------------------------------------
# Physical layouts


def _phys_of_logical(fmt, pair, rows, rows_phys, half, g) -> torch.Tensor:
    """Physical row of each logical row L (non-gated: L = q; gated: L = h * half + i with h 0 up,
    1 gate, i.e. the [up; gate] halves)."""
    if pair == Pair.NONE:
        if fmt not in _TRTLLM:
            return torch.arange(rows)
        held = get_shuffle_matrix_a_row_indices(
            torch.zeros(rows_phys, 1, dtype=torch.uint8), 128
        )
    elif pair == Pair.UP_GATE_HALVES:
        return torch.arange(2 * half)
    elif pair == Pair.TRTLLM_W31:
        n = 2 * half
        order = get_reorder_rows_for_gated_act_gemm_row_indices(
            torch.zeros(n, 1)
        )  # q -> L
        shuffle = get_shuffle_matrix_a_row_indices(
            torch.zeros(n, 1, dtype=torch.uint8), 128
        )
        held = order[shuffle]  # physical row -> L
    else:
        n = 2 * half
        held = _interleave_linear_and_gate(
            torch.arange(n).view(n, 1), group_size=g, dim=-2
        )
        held = held.view(n)
    phys_of = torch.empty(held.numel(), dtype=torch.long)
    phys_of[held] = torch.arange(held.numel())
    return phys_of[:rows]


def _sf128x4_offset(r: torch.Tensor, c: torch.Tensor, cols: int) -> torch.Tensor:
    return (
        (r // 128) * ((cols + 3) // 4) * 512
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )


def _build(fmt, pair, g, k, k_phys, experts):
    gated = pair != Pair.NONE
    half = _GATED_HALF if gated else 0
    rows = 2 * half if gated else _NONGATED_ROWS
    rows_phys = rows if gated else _NONGATED_ROWS_PHYS
    sm12x = fmt == AFmt.DENSE_CUTE_SM12X
    if sm12x:
        experts = 1
    dense = experts == 1
    blk = 32 if fmt in _MX else 16
    cols, cols_phys = k // blk, math.ceil(k_phys / blk)
    rb = k_phys // 2
    gen = torch.Generator().manual_seed(
        int(fmt) * 1000 + int(pair) * 100 + g + k_phys + experts
    )

    def rand(*shape, hi=256):
        return torch.randint(0, hi, shape, generator=gen, dtype=torch.long)

    codes_log = rand(experts, rows, k, hi=16)
    sf_log = rand(experts, rows, cols)
    phys_of = _phys_of_logical(fmt, pair, rows, rows_phys, half, g)
    packed = codes_log[..., 0::2] | (codes_log[..., 1::2] << 4)  # [E, rows, k / 2]

    if sm12x:
        rowmajor = rand(rows_phys, rb)
        rowmajor[phys_of, : k // 2] = packed[0]
        codes = _cute_dsl_pack_fp4_weight(rowmajor.to(torch.uint8).t().contiguous())
        codes = codes.view(torch.uint8).flatten()
        sf = rand(cols_phys, rows_phys)
        sf[:cols, phys_of] = sf_log[0].t()
        sf = sf.flatten()
        codes_stride = sf_stride = 0
    else:
        codes = rand(experts, rows_phys, rb)
        codes[:, phys_of, : k // 2] = packed
        codes_stride = rows_phys * rb
        if fmt in _SF128X4:
            sf_stride = _pad(rows_phys, 128) * _pad(cols_phys, 4)
            sf = rand(experts, sf_stride)
            off = _sf128x4_offset(
                phys_of[:, None], torch.arange(cols)[None, :], cols_phys
            )
            sf[:, off] = sf_log
        else:
            sf_stride = rows_phys * cols_phys
            sf = rand(experts, rows_phys, cols_phys)
            sf[:, phys_of, :cols] = sf_log
        if dense:
            codes_stride = sf_stride = 0
    meta = [
        int(fmt),
        int(pair),
        rows,
        rows_phys,
        k,
        k_phys,
        half,
        g,
        codes_stride,
        sf_stride,
        experts,
    ]
    return dict(
        gated=gated,
        half=half,
        rows=rows,
        k=k,
        cols=cols,
        rb=rb,
        experts=experts,
        codes=codes.to(torch.uint8).flatten(),
        sf=sf.to(torch.uint8).flatten(),
        codes_log=codes_log,
        sf_log=sf_log,
        meta=meta,
    )


def _expected_modes(fmt, pair, case) -> int:
    if fmt == AFmt.DENSE_CUTE_SM12X:
        code = 4  # kSm12x
    elif case["rb"] % 16 != 0:
        code = 3  # kCpAsync
    elif pair == Pair.UP_GATE_HALVES:
        code = 1  # kTmaHalves
    elif fmt in _TRTLLM:
        code = 2  # kTmaTrtllm
    else:
        code = 0  # kTma
    cols_phys = math.ceil(
        case["meta"][5] / (32 if fmt in _MX else 16)
    )  # meta[5]: physical K
    sf = 1 if fmt == AFmt.NVFP4_LINEAR_SF and cols_phys % 4 != 0 else 0  # kBytes
    return code * 16 + sf


# ----------------------------------------------------------------------------------------------
# Expected fragments


def _expected(case, fmt, cta_m):
    """W8, nibble mask and scale pair of every dumped (thread, k16 step)."""
    rows, k, half = case["rows"], case["k"], case["half"]
    kM64 = cta_m // 128
    shape = (
        case["experts"],
        math.ceil(rows / cta_m),
        math.ceil(k / 128),
        2,
        kM64,
        4,
        32,
        8,
    )
    e, mt, kt, wg, mi, w, lane, s = torch.meshgrid(
        *[torch.arange(n) for n in shape], indexing="ij"
    )
    g, q = lane >> 2, lane & 3
    r0 = mt * cta_m + 64 * (wg * kM64 + mi) + 16 * w + g
    ks = kt * 8 + s
    codes_log, sf_log = case["codes_log"], case["sf_log"]

    def logical(r):
        if case["gated"]:
            h = (r % 16 < 8).long()  # rows 0-7 of a 16-row group: gate
            i = 8 * (r // 16) + r % 8
            return (h * half + i).clamp(max=rows - 1), (r < rows) & (i < half)
        return r.clamp(max=rows - 1), r < rows

    w8 = torch.zeros(shape, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.long)
    pair = torch.zeros(shape, dtype=torch.long)
    blk = ks // 2 if fmt in _MX else ks
    for half_row, r in enumerate((r0, r0 + 8)):
        lrow, valid = logical(r)
        for dk in (0, 8):
            j = (
                2 * half_row + dk // 8
            )  # a0, a2 (row g), a1, a3 (row g + 8): W8 j 0, 1, 2, 3
            for odd in (0, 1):
                kk = 16 * ks + 2 * q + dk + odd
                ok = valid & (kk < k)
                nib = codes_log[e, lrow, kk.clamp(max=k - 1)] * ok
                sh = 4 * (j + 4 * odd)
                w8 |= nib << sh
                mask |= ok.long() * (0xF << sh)
        sok = valid & (blk < case["cols"])
        pair |= (sf_log[e, lrow, blk.clamp(max=case["cols"] - 1)] * sok) << (
            8 * half_row
        )
    return shape, w8, mask, pair


# ----------------------------------------------------------------------------------------------
# Shared-memory bank wavefronts of the consumer loads


def _wavefronts(addrs, width):
    """(wavefronts, ideal) of one warp-wide shared load of `width` bytes per lane."""
    lanes = 32 if width <= 4 else (16 if width == 8 else 8)
    total = 0
    for p0 in range(0, 32, lanes):
        banks = {}
        for a in addrs[p0 : p0 + lanes]:
            for wd in range(max(width, 4) // 4):
                word = a // 4 + wd
                banks.setdefault(word % 32, set()).add(word)
        total += max(len(v) for v in banks.values())
    return total, 32 // lanes


def _bank_record(frag, fmt, cta_m, a_path):
    """Per K tile: (code wavefronts, ideal), (scale wavefronts, ideal), summed over warps."""
    frag = frag.view(2, cta_m // 128, 4, 32, 6).long().tolist()
    code, scale = [0, 0], [0, 0]

    def add(acc, addrs, width):
        t, i = _wavefronts(addrs, width)
        acc[0] += t
        acc[1] += i

    for wg_rows in frag:
        for mi_rows in wg_rows:
            for warp in mi_rows:
                c0, c1, x0, x1, sf0, sf1 = zip(*warp, strict=False)
                if a_path == 1:
                    add(code, c0, 16)
                    add(code, c1, 16)
                    add(scale, sf0, 16)
                elif fmt == AFmt.DENSE_CUTE_SM12X:
                    for s in range(8):
                        seg = s * cta_m * 8
                        add(code, [seg + x for x in c0], 2)
                        add(code, [seg + x + 256 for x in c0], 2)
                        for rows in (sf0, sf1):
                            add(scale, [s * cta_m + x for x in rows], 1)
                else:
                    for j in range(4):
                        add(
                            code,
                            [b + ((x ^ j) << 4) for b, x in zip(c0, x0, strict=False)],
                            16,
                        )
                        add(
                            code,
                            [b + ((x ^ j) << 4) for b, x in zip(c1, x1, strict=False)],
                            16,
                        )
                    width = 4 if fmt in _MX else 8
                    add(scale, sf0, width)
                    add(scale, sf1, width)
    return code, scale


# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
@pytest.mark.parametrize("a_path", [0, 1], ids=["n1", "n2"])
@pytest.mark.parametrize("cta_m", [128, 256])
@pytest.mark.parametrize(
    "combo", _COMBOS, ids=[f"{c[0].name}-{c[1].name}{c[2] or ''}" for c in _COMBOS]
)
def test_a_fragments(mod, combo, cta_m, a_path, shape):
    fmt, pair, g = combo
    shape_name, k, k_phys, experts = shape
    case = _build(fmt, pair, g, k, k_phys, experts)
    name = f"sm90.layouts.{fmt.name}.{pair.name}{g or ''}.m{cta_m}.{'n2' if a_path else 'n1'}"
    name += f".{shape_name}"
    dev = _device()
    codes, sf = case["codes"].to(dev), case["sf"].to(dev)
    meta = case["meta"] + [cta_m, a_path]

    modes = mod.modes(codes, sf, meta)
    want = _expected_modes(fmt, pair, case)
    print(f"WGEMU-TEST name={name}.loader cases=1 mismatches={int(modes != want)}")
    assert modes == want, f"{name}: loader mode {modes:#x}, expected {want:#x}"

    shape_e, w8, mask, pair_ref = _expected(case, fmt, cta_m)
    out = torch.zeros(*shape_e, 2, dtype=torch.int32, device=dev)
    frag = torch.zeros(2 * (cta_m // 128) * 4 * 32 * 6, dtype=torch.int32, device=dev)
    mod.run(codes, sf, meta, out, frag)
    torch.cuda.synchronize()
    out = out.cpu().long() & 0xFFFFFFFF
    bad = (((out[..., 0] ^ w8) & mask) != 0) | (out[..., 1] != pair_ref)
    print(f"WGEMU-TEST name={name} cases={bad.numel()} mismatches={int(bad.sum())}")
    for idx in torch.nonzero(bad)[:4].tolist():
        t = tuple(idx)
        print(
            f"WGEMU-MISMATCH name={name} at(e,mt,kt,wg,mi,w,lane,s)={t} "
            f"w8=0x{int(out[t][0]):08x} ref=0x{int(w8[t]):08x} mask=0x{int(mask[t]):08x} "
            f"pair=0x{int(out[t][1]):04x} ref=0x{int(pair_ref[t]):04x}"
        )
    code, scale = _bank_record(frag.cpu(), fmt, cta_m, a_path)
    print(
        f"SM90-LDS-BANK name={name} codes_wavefronts={code[0]} codes_ideal={code[1]} "
        f"scales_wavefronts={scale[0]} scales_ideal={scale[1]}"
    )
    assert not bool(bad.any()), f"{name}: {int(bad.sum())} mismatching fragments"
