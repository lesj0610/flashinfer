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

SM90 FP4 epilogue, token-wise GEMM2-input quantization and ordered top-k finalize
(``csrc/fp4_sm90/{epilogue,row_quant,finalize}.cuh``), built for the running SM8x / SM90 device
with FlashInfer's NVFP4 software path:

- the NVFP4 epilogue (accumulator fragments in the wgmma layout, B tiles or rows) against
  ``fp4_quantize`` of the same activations, bit for bit, for several recipes;
- per-token and GEMM2 epilogues against the SM100+ rounding chain;
- ``row_quant`` against ``nvfp4_quantize(per_token_activation=True)`` (scales, codes, SF and the
  dequantized B tiles, zero padding included) for every recipe;
- finalize against a sequential FP32 sum on inputs whose sum depends on the slot order.
"""

import functools
import itertools
import math

import pytest
import torch

from flashinfer import SfLayout, fp4_quantize, nvfp4_quantize
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.jit.fp4_quantization import sw_cvt_nvcc_flags
from flashinfer.jit.fused_moe import fp4_sm90_recipe_bits
from flashinfer.jit.utils import write_if_different
from flashinfer.quantization.fp4_quantization import NVFP4_QUANT_ENV_VARS
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    nvfp4_per_token_scale_inv,
)
from flashinfer.tllm_enums import ActivationType as A
from flashinfer.utils import get_compute_capability

_SOURCE = r"""
#include "fp4_sm90/epilogue.cuh"
#include "fp4_sm90/finalize.cuh"
#include "fp4_sm90/row_quant.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::fp4_sm90;
using tvm::ffi::Array;
using tvm::ffi::Optional;

template <class T>
T* opt(const Optional<TensorView>& t) {
  return t.has_value() ? static_cast<T*>(t.value().data_ptr()) : nullptr;
}

__global__ void act_eval_kernel(const float* g, const float* u, float* y, int n, const float* ap,
                                int act, int gated, int pt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const epi::ActQ q = epi::act_params(act, ap, 0);
  y[i] = gated ? epi::act_gated<-1>(act, g[i], u[i], q) : epi::act_plain<-1>(act, g[i], q, pt);
}

void act_eval(TensorView g, TensorView u, TensorView y, Optional<TensorView> ap, int64_t act,
              bool gated, bool pt) {
  const int n = static_cast<int>(g.numel());
  ffi::CUDADeviceGuard guard(g.device().device_id);
  act_eval_kernel<<<(n + 127) / 128, 128, 0, get_stream(g.device())>>>(
      static_cast<const float*>(g.data_ptr()), static_cast<const float*>(u.data_ptr()),
      static_cast<float*>(y.data_ptr()), n, opt<const float>(ap), int(act), gated, pt);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

// One (m-tile, n-tile) epilogue by both consumer warp groups (threads 0-255).
template <class C>
__global__ void __launch_bounds__(256) epi_kernel(const float* acc, GemmArgs args, int expert,
                                                  int m0, int tile, int n0, int n_valid) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x;
  float a[C::kAccPerThread];
#pragma unroll
  for (int i = 0; i < C::kAccPerThread; ++i) a[i] = acc[tid * C::kAccPerThread + i];
  Epilogue<C>::apply(a, args, expert, m0, tile, n0, n_valid, tid >> 7, (tid >> 5) & 3, tid & 31,
                     static_cast<uint32_t>(__cvta_generic_to_shared(smem)));
}

template <class C>
void launch_epi(const float* acc, GemmArgs g, const Array<int64_t>& meta, cudaStream_t s) {
  g.epi.fq = uint8_t(epi_fq(C::kEpi));
  g.epi.recipe = epi_recipe(C::kEpi);
  const int smem = Epilogue<C>::kStagingBytes > 0 ? Epilogue<C>::kStagingBytes : 16;
  epi_kernel<C><<<1, 256, smem, s>>>(acc, g, int(meta[1]), int(meta[2]), int(meta[3]),
                                     int(meta[4]), int(meta[5]));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

constexpr AFmt kAF = AFmt::kNvfp4Swz128x4;
constexpr uint8_t kR7 = kRecipe4over6 | kRecipeE4m3Max256 | kRecipeErrMse;
#if defined(FLASHINFER_FP4_SM90_EXACT)
using V0 = Cfg<128, 64, 8, kAF, false, epi_code(Fq::kNvfp4, 0, true, false)>;
using V1 = Cfg<128, 64, 8, kAF, false, epi_code(Fq::kNvfp4, kRecipe4over6, true, false)>;
using V2 = Cfg<128, 64, 8, kAF, false, epi_code(Fq::kNvfp4, kR7, true, false)>;
using V3 = Cfg<256, 16, 2, kAF, true, epi_code(Fq::kNvfp4, kRecipe4over6, false, false)>;
#endif
using V4 = Cfg<128, 16, 2, kAF, false, epi_code(Fq::kNone, 0, true, false)>;
using V5 = Cfg<128, 16, 2, kAF, false, epi_code(Fq::kNone, 0, false, false)>;
using V6 = Cfg<128, 16, 2, kAF, false, epi_code(Fq::kNone, 0, false, true)>;

// meta: rows, expert, m0, tile, n0, n_valid, act, out_f16, out_tiled, out_ld
void epi_run(int64_t variant, TensorView acc, Array<int64_t> meta, Optional<TensorView> alpha,
             Optional<TensorView> tok_scale, Optional<TensorView> act_params,
             Optional<TensorView> bias, TensorView out, Optional<TensorView> fq_global) {
  TVM_FFI_ICHECK_EQ(meta.size(), 10);
  GemmArgs g{};
  g.a.rows = int(meta[0]);
  g.epi.alpha = opt<const float>(alpha);
  g.epi.tok_scale = opt<const float>(tok_scale);
  g.epi.act_params = opt<const float>(act_params);
  g.epi.bias = opt<const void>(bias);
  g.epi.out = out.data_ptr();
  g.epi.out_ld = meta[9];
  g.epi.fq_global = opt<const float>(fq_global);
  g.epi.act = uint8_t(meta[6]);
  g.epi.out_f16 = uint8_t(meta[7]);
  g.epi.out_tiled = uint8_t(meta[8]);
  const float* a = static_cast<const float*>(acc.data_ptr());
  ffi::CUDADeviceGuard guard(acc.device().device_id);
  cudaStream_t s = get_stream(acc.device());
  switch (variant) {
#if defined(FLASHINFER_FP4_SM90_EXACT)
    case 0: return launch_epi<V0>(a, g, meta, s);
    case 1: return launch_epi<V1>(a, g, meta, s);
    case 2: return launch_epi<V2>(a, g, meta, s);
    case 3: return launch_epi<V3>(a, g, meta, s);
#endif
    case 4: return launch_epi<V4>(a, g, meta, s);
    case 5: return launch_epi<V5>(a, g, meta, s);
    case 6: return launch_epi<V6>(a, g, meta, s);
    default: TVM_FFI_ICHECK(false) << "variant " << variant << " is not built in this module";
  }
}

// meta: in_ld, rows, cols, in_f16, recipe, out_f16, n
void rq_run(TensorView in, Array<int64_t> meta, Optional<TensorView> expert_offset,
            Optional<TensorView> expert_tile_base, Optional<TensorView> num_tiles,
            TensorView out_tiles, TensorView tok_scale, Optional<TensorView> fq_codes,
            Optional<TensorView> fq_sf) {
  TVM_FFI_ICHECK_EQ(meta.size(), 7);
  RowQuantArgs a{};
  a.in = in.data_ptr();
  a.in_ld = meta[0];
  a.rows = int(meta[1]);
  a.cols = int(meta[2]);
  a.in_f16 = uint8_t(meta[3]);
  a.recipe = uint8_t(meta[4]);
  a.out_f16 = uint8_t(meta[5]);
  a.n = int(meta[6]);
  a.expert_offset = opt<const int32_t>(expert_offset);
  a.expert_tile_base = opt<const int32_t>(expert_tile_base);
  a.num_tiles = opt<const int32_t>(num_tiles);
  a.out_tiles = out_tiles.data_ptr();
  a.tok_scale = static_cast<float*>(tok_scale.data_ptr());
  a.fq_codes = opt<uint8_t>(fq_codes);
  a.fq_sf = opt<uint8_t>(fq_sf);
  ffi::CUDADeviceGuard guard(in.device().device_id);
  launch_fp4_sm90_row_quant(a, get_stream(in.device()));
}

// meta: y_dtype, w_dtype, out_f16, tokens, top_k, hidden, y_ld, out_ld
void fin_run(TensorView y, TensorView src_permuted, Optional<TensorView> w, TensorView out,
             Array<int64_t> meta) {
  TVM_FFI_ICHECK_EQ(meta.size(), 8);
  FinalizeArgs a{};
  a.y = y.data_ptr();
  a.y_dtype = uint8_t(meta[0]);
  a.src_permuted = static_cast<const int32_t*>(src_permuted.data_ptr());
  a.topk_weights = opt<const void>(w);
  a.w_dtype = uint8_t(meta[1]);
  a.out = out.data_ptr();
  a.out_f16 = uint8_t(meta[2]);
  a.tokens = int(meta[3]);
  a.top_k = int(meta[4]);
  a.hidden = int(meta[5]);
  a.y_ld = meta[6];
  a.out_ld = meta[7];
  ffi::CUDADeviceGuard guard(y.device().device_id);
  launch_fp4_sm90_finalize(a, get_stream(y.device()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(act_eval, act_eval);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(epi_run, epi_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rq_run, rq_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fin_run, fin_run);
"""

_RECIPES = [None] + [
    NVFP44Over6Config(e4m3_max=m, err_mode=e, err_use_fast_math=f)
    for m, e, f in itertools.product((448, 256), ("MAE", "MSE"), (False, True))
]
# Epilogue variants of the harness with an NVFP4 output: (variant, recipe of its Cfg).
_FQ_GATED = [
    (0, None),
    (1, NVFP44Over6Config()),
    (2, NVFP44Over6Config(e4m3_max=256, err_mode="MSE")),
]
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 += [-v for v in _E2M1]  # code 8 is -0.0
_NAN = float("nan")
_FILL = 0x7F7F  # int16 sentinel of bytes never written


def _dev() -> torch.device:
    return torch.device("cuda:0")


def _arch() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(_dev())
    if major not in (8, 9):
        pytest.skip(
            "the SM90 FP4 epilogue runs on SM90 (and builds for SM80 to SM89 in tests)"
        )
    return "90" if major == 9 else "80"


@functools.cache
def _module(arch: str, mode: str):
    """mode "e0" / "e1": FLASHINFER_FP4_SM90_EXACT=0/1 (NVFP4 epilogue variants built);
    "rt": no define, row_quant reads FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH at launch."""
    name = f"test_sm90_epilogue_{arch}_{mode}"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "epilogue_test.cu"
    write_if_different(source, _SOURCE)
    nv_internal = jit_env.FLASHINFER_CSRC_DIR / "nv_internal"
    common = nv_internal / "cpp" / "common"
    defines = ["-DENABLE_BF16", "-DENABLE_FP8"]
    defines += ["-DENABLE_FP4"] if is_cuda_version_at_least("12.8") else []
    if mode != "rt":
        defines.append(f"-DFLASHINFER_FP4_SM90_EXACT={mode[1]}")
    # The flags of FlashInfer's own quantization module of that arch, -use_fast_math included,
    # so the quantized bits are comparable with fp4_quantize / nvfp4_quantize.
    return gen_jit_spec(
        name,
        [source]
        + [
            common / f
            for f in (
                "envUtils.cpp",
                "logger.cpp",
                "stringUtils.cpp",
                "tllmException.cpp",
            )
        ],
        extra_cflags=defines,
        extra_cuda_cflags=sw_cvt_nvcc_flags(arch) + defines,
        extra_include_paths=[nv_internal, nv_internal / "include"],
    ).build_and_load()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Real device, and recipes / exactness only from the arguments below.
    for name in NVFP4_QUANT_ENV_VARS + (
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
        "WGEMU_SPOOF_CC",
    ):
        monkeypatch.delenv(name, raising=False)


def _exact_env(monkeypatch, exact: bool):
    if exact:
        monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")


# ----------------------------------------------------------------------------------------------
# Helpers


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _assert_bits(got: torch.Tensor, want: torch.Tensor, nan_ok: bool = False):
    same = _bits(got) == _bits(want)
    if nan_ok:
        same |= torch.isnan(got.float()) & torch.isnan(want.float())
    bad = (~same).nonzero()
    assert bad.numel() == 0, (
        f"{bad.shape[0]} values differ, first at {bad[0].tolist()}: "
        f"got {got[tuple(bad[0])].item()}, want {want[tuple(bad[0])].item()}"
    )


def _dequant(codes: torch.Tensor, sf: torch.Tensor, dtype) -> torch.Tensor:
    """code x E4M3 SF of [M, K / 2] codes and [M, K / 16] linear SF, exact, as dtype."""
    lut = torch.tensor(_E2M1, device=codes.device)
    c = codes.view(torch.uint8).to(torch.int64)
    vals = torch.stack([lut[c & 15], lut[c >> 4]], dim=-1).reshape(c.shape[0], -1)
    s = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return (vals * s).to(dtype)


def _tile_index(t: int, k_tiles: int, n: int, kp: int) -> torch.Tensor:
    """int16 index of (row r, value c) of B tile t (params.cuh b_tile_byte)."""
    r = torch.arange(n, device=_dev())[:, None]
    c = torch.arange(kp, device=_dev())[None, :]
    byte = (
        ((t * k_tiles + c // 128) * 2 + (c // 64) % 2) * n * 128
        + r * 128
        + (((c % 64) // 8) ^ (r % 8)) * 16
        + (c % 8) * 2
    )
    return byte // 2


def _read_tile(buf, t, k_tiles, n, kp, dtype) -> torch.Tensor:
    return buf.view(torch.int16)[_tile_index(t, k_tiles, n, kp)].view(dtype)


def _acc(d: torch.Tensor, cta_m: int, n: int) -> torch.Tensor:
    """wgmma accumulator fragments [256 threads, kAccPerThread] of the m-tile matrix d[row, col]:
    acc[(mi (n / 8) + j) 4 + v], v0 = (row lane / 4, col 8 j + 2 (lane % 4)), v1 col + 1,
    v2 row + 8, v3 both; row = wg cta_m / 2 + 64 mi + 16 warp + lane / 4."""
    j_n, kacc = n // 8, (cta_m // 128) * n // 2
    tid = torch.arange(256, device=d.device)[:, None]
    i = torch.arange(kacc, device=d.device)[None, :]
    v, j, mi = i % 4, (i // 4) % j_n, i // (4 * j_n)
    wg, warp, lane = tid // 128, (tid // 32) % 4, tid % 32
    row = wg * (cta_m // 2) + 64 * mi + 16 * warp + lane // 4 + 8 * (v >= 2).long()
    col = 8 * j + 2 * (lane % 4) + v % 2
    return d[row, col].contiguous()


def _tile_rows(m0, cta_m, n_valid, gate_or_a, up=None) -> torch.Tensor:
    """Accumulators [cta_m, n] of kernel rows m0 + kr: gated rows 0-7 of a 16-row group are
    gate[i], rows 8-15 up[i] (i = 8 group + row % 8). Rows and n-tile rows past the data: NaN."""
    rk = m0 + torch.arange(cta_m, device=_dev())
    if up is not None:
        i = 8 * (rk // 16) + rk % 8
        ic = i.clamp(max=gate_or_a.shape[0] - 1)
        d = torch.where((rk % 16 >= 8)[:, None], up[ic], gate_or_a[ic])
        bad = i >= gate_or_a.shape[0]
    else:
        d = gate_or_a[rk.clamp(max=gate_or_a.shape[0] - 1)]
        bad = rk >= gate_or_a.shape[0]
    d = d.clone()
    d[bad] = _NAN
    d[:, n_valid:] = _NAN
    return d


def _epi(
    mod,
    variant,
    d,
    cta_m,
    n,
    meta,
    out,
    alpha=None,
    tok=None,
    ap=None,
    bias=None,
    fq=1.0,
):
    fq_t = torch.tensor([fq], device=out.device, dtype=torch.float32)
    mod.epi_run(variant, _acc(d, cta_m, n), meta, alpha, tok, ap, bias, out, fq_t)


def _act(mod, act, g, u=None, ap=None, pt=False) -> torch.Tensor:
    y = torch.empty_like(g)
    gated = u is not None
    mod.act_eval(
        g.contiguous(), (u if gated else g).contiguous(), y, ap, int(act), gated, pt
    )
    return y


def _f16(dtype) -> int:
    return int(dtype == torch.float16)


def _sentinel(numel: int) -> torch.Tensor:
    return torch.full((numel,), _FILL, dtype=torch.int16, device=_dev())


def _randn(gen, *shape, scale=1.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen) * scale).to(_dev())


# ----------------------------------------------------------------------------------------------
# Activations: the device functions against the CUTLASS adaptor formulas (loose: approximate
# exp / tanh); the epilogue tests below compare against these same device functions.


def _ref_act(act, g, u, q):
    g, u = g.double(), u.double()
    x, y, z = q
    sig = torch.sigmoid

    def silu(v):
        return v * sig(v)

    def gelu(v):
        return 0.5 * v * (1 + torch.erf(v / math.sqrt(2)))

    def clamp(v):
        return v.clamp(-z, z)

    if act == A.Gelu:
        return gelu(g)
    if act == A.Relu:
        return g.clamp_min(0)
    if act == A.Silu:
        return silu(g)
    if act == A.Identity:
        return g
    if act == A.Relu2:
        return g.clamp_min(0) ** 2
    if act == A.ClampedRelu2:
        return (torch.tanh(g.clamp_min(0) / z) * z) ** 2
    if act == A.Swiglu:
        return silu(g) * u
    if act == A.Geglu:
        return gelu(g) * u
    if act == A.GegluTanh:
        k0 = math.sqrt(2 / math.pi)
        return 0.5 * g * (1 + torch.tanh(k0 * (g + 0.044715 * g**3))) * u
    if act == A.SwigluBias:
        gc = g.clamp_max(z)
        return gc * sig(gc * x) * (clamp(u) + y)
    if act == A.SwigluStep:
        return silu(g).clamp_max(z) * clamp(u)
    if act == A.PowLU:
        s = sig(g)
        a = torch.where(
            g > 0, g.clamp_min(1e-30) ** (x / (g.clamp_min(0).sqrt() + 1)) * s, g * s
        )
        return a.clamp_max(z) * clamp(u)
    if act == A.Situ:
        return (torch.tanh(g / x) * sig(g) * x) * (torch.tanh(u / y) * y)
    raise AssertionError(act)


def _defaults(act):
    q = [1.0, 0.0, math.inf]
    if act == A.SwigluStep:
        q[2] = 7.0
    if act == A.PowLU:
        q[0] = 2.5
    if act == A.Situ:
        q[0], q[1] = 4.0, 25.0
    return q


_GATED = {A.Swiglu, A.Geglu, A.GegluTanh, A.SwigluBias, A.SwigluStep, A.PowLU, A.Situ}
_ACT_CASES = [
    (A.Gelu, None),
    (A.Relu, None),
    (A.Silu, None),
    (A.Identity, None),
    (A.Relu2, None),
    (A.ClampedRelu2, (_NAN, _NAN, 6.0)),
    (A.Swiglu, None),
    (A.Geglu, None),
    (A.GegluTanh, None),
    (A.SwigluBias, None),
    (A.SwigluBias, (1.702, 1.0, 7.0)),
    (A.SwigluStep, None),
    (A.SwigluStep, (_NAN, _NAN, 5.0)),
    (A.PowLU, None),
    (A.PowLU, (2.0, _NAN, 3.0)),
    (A.Situ, None),
    (A.Situ, (2.0, 10.0, _NAN)),
]


@pytest.mark.parametrize("act,params", _ACT_CASES, ids=repr)
def test_activations(act, params):
    mod = _module(_arch(), "rt")
    gen = torch.Generator().manual_seed(int(act))
    g = (torch.rand(4096, generator=gen) * 12 - 6).to(_dev())
    u = (torch.rand(4096, generator=gen) * 12 - 6).to(_dev())
    q = _defaults(act)
    ap = None
    if params is not None:
        q = [d if math.isnan(p) else p for d, p in zip(q, params, strict=True)]
        ap = torch.tensor([params], dtype=torch.float32, device=_dev())
    gated = act in _GATED
    y = _act(mod, act, g, u if gated else None, ap)
    torch.testing.assert_close(y.double(), _ref_act(act, g, u, q), rtol=1e-2, atol=1e-2)


# ----------------------------------------------------------------------------------------------
# NVFP4 epilogue (per-tensor recipes): GEMM2 B tiles against fp4_quantize


def _fq_reference(mod, y, n, n_valid, kp, recipe, dtype):
    """Quantized-dequantized B rows [n, kp] of the activations y [cols, n] (tokens past n_valid
    and values past cols are 0), with the fq_global used."""
    yt = torch.zeros(n_valid, kp, dtype=dtype, device=_dev())
    yt[:, : y.shape[0]] = y[:, :n_valid].T
    e4m3_max = recipe.e4m3_max if recipe is not None else 448
    gs = (e4m3_max * 6.0 / yt.float().abs().max()).reshape(1).float()
    codes, sf = fp4_quantize(
        yt,
        gs,
        16,
        False,
        is_sf_swizzled_layout=False,
        backend="cuda",
        nvfp4_4over6=recipe,
    )
    want = torch.zeros(n, kp, dtype=dtype, device=_dev())
    want[:n_valid] = _dequant(
        codes, sf.view(torch.uint8).reshape(n_valid, kp // 16), dtype
    )
    return want, gs.item()


@pytest.mark.parametrize("exact", [0, 1])
@pytest.mark.parametrize("variant,recipe", _FQ_GATED, ids=repr)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("act", [A.Swiglu, A.SwigluStep])
@pytest.mark.parametrize("ncols,with_bias", [(40, False), (80, True)])
def test_epilogue_fq_gated(
    monkeypatch, exact, variant, recipe, out_dtype, act, ncols, with_bias
):
    mod = _module(_arch(), f"e{exact}")
    _exact_env(monkeypatch, bool(exact))
    cta_m, n, n_valid, tile, n0, expert = 128, 64, 45, 1, 64, 1
    rows, k_tiles = 2 * ncols, (ncols + 127) // 128
    kp = k_tiles * 128
    gen = torch.Generator().manual_seed(ncols * 3 + variant)
    gate, up = _randn(gen, ncols, n, scale=3.0), _randn(gen, ncols, n, scale=3.0)
    alpha = torch.tensor([0.5, 1.375], device=_dev())
    bias = _randn(gen, 2, rows, scale=0.5).to(out_dtype) if with_bias else None
    # SM100+ order: T(acc alpha), + bias in FP32, activation, T(y), NVFP4 of T(y)
    g = (gate * alpha[expert]).to(out_dtype).float()
    u = (up * alpha[expert]).to(out_dtype).float()
    if bias is not None:  # [up; gate] halves of the expert's rows
        u += bias[expert, :ncols].float()[:, None]
        g += bias[expert, ncols:].float()[:, None]
    y = _act(mod, act, g, u).to(out_dtype)
    want, fq = _fq_reference(mod, y, n, n_valid, kp, recipe, out_dtype)
    out = _sentinel((tile + 1) * k_tiles * 2 * n * 128 // 2)
    for m0 in range(0, rows, cta_m):
        meta = [rows, expert, m0, tile, n0, n_valid, int(act), _f16(out_dtype), 1, 0]
        _epi(
            mod,
            variant,
            _tile_rows(m0, cta_m, n_valid, gate, up),
            cta_m,
            n,
            meta,
            out,
            alpha=alpha,
            bias=bias,
            fq=fq,
        )
    _assert_bits(_read_tile(out, tile, k_tiles, n, kp, out_dtype), want)
    assert (out.view(torch.int16)[_tile_index(0, k_tiles, n, kp)] == _FILL).all()


@pytest.mark.parametrize("exact", [0, 1])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_tiled", [1, 0])
def test_epilogue_fq_plain_cta256(monkeypatch, exact, out_dtype, out_tiled):
    """Non-gated NVFP4 epilogue, CTA_M 256 (two m64 per warp group), Relu2; rows also written
    row-major (16-byte stores and the scalar tail)."""
    mod = _module(_arch(), f"e{exact}")
    _exact_env(monkeypatch, bool(exact))
    recipe = NVFP44Over6Config()
    cta_m, n, n_valid, tile, n0, expert, rows = 256, 16, 13, 0, 0, 0, 200
    k_tiles, kp, out_ld = 2, 256, 208
    gen = torch.Generator().manual_seed(11 + exact)
    a_mat = _randn(gen, rows, n, scale=2.0)
    alpha = torch.tensor([0.625], device=_dev())
    x = (a_mat * alpha[0]).to(out_dtype).float()
    y = _act(mod, A.Relu2, x).to(out_dtype)
    want, fq = _fq_reference(mod, y, n, n_valid, kp, recipe, out_dtype)
    numel = k_tiles * 2 * n * 128 // 2 if out_tiled else (n0 + n) * out_ld
    out = _sentinel(numel)
    meta = [
        rows,
        expert,
        0,
        tile,
        n0,
        n_valid,
        int(A.Relu2),
        _f16(out_dtype),
        out_tiled,
        out_ld,
    ]
    _epi(
        mod,
        3,
        _tile_rows(0, cta_m, n_valid, a_mat),
        cta_m,
        n,
        meta,
        out,
        alpha=alpha,
        fq=fq,
    )
    if out_tiled:
        _assert_bits(_read_tile(out, tile, k_tiles, n, kp, out_dtype), want)
    else:
        got = out.view(out_dtype).reshape(n0 + n, out_ld)
        expect = _sentinel(numel).view(out_dtype).reshape(n0 + n, out_ld).clone()
        expect[n0 : n0 + n_valid, :rows] = want[:n_valid, :rows]
        _assert_bits(got, expect)


# ----------------------------------------------------------------------------------------------
# BF16/FP16 epilogues: per-token GEMM1 rows, GEMM2 rows, GEMM1 B tiles without quantization


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_epilogue_per_token(gated, with_bias, out_dtype):
    """Per-token GEMM1 (row_quant input): T(acc (alpha s)) with FTZ products, + bias after the
    rounding, activation (Relu2 keeps NaN), row-major rows; rows of other tokens untouched."""
    mod = _module(_arch(), "rt")
    cta_m, n, n_valid, n0, expert = 128, 16, 13, 16, 1
    ncols = 40 if gated else 100
    rows, out_ld = (2 * ncols if gated else ncols), ncols + 8
    act = A.Swiglu if gated else A.Relu2
    gen = torch.Generator().manual_seed(5 + int(gated) * 2 + int(with_bias))
    gate = _randn(gen, ncols, n, scale=3.0)
    up = _randn(gen, ncols, n, scale=3.0) if gated else None
    if not gated:
        gate[7, 2] = _NAN  # NaN propagates through the per-token Relu2
    tok = torch.full((n0 + n,), _NAN, device=_dev())
    tok[n0 : n0 + n_valid] = (torch.rand(n_valid, generator=gen) * 2 + 0.25).to(_dev())
    alpha = torch.tensor([0.75, 1.5], device=_dev())
    bias = _randn(gen, 2, rows, scale=0.5).to(out_dtype) if with_bias else None
    rs = alpha[expert] * tok[n0 : n0 + n]
    g = (gate * rs).to(out_dtype).float()
    if gated:
        u = (up * rs).to(out_dtype).float()
        if bias is not None:
            u += bias[expert, :ncols].float()[:, None]
            g += bias[expert, ncols:].float()[:, None]
        y = _act(mod, act, g, u, pt=True)
    else:
        if bias is not None:
            g += bias[expert].float()[:, None]
        y = _act(mod, act, g, pt=True)
    out = _sentinel((n0 + n + 3) * out_ld)
    for m0 in range(0, rows, cta_m):
        meta = [rows, expert, m0, 0, n0, n_valid, int(act), _f16(out_dtype), 0, out_ld]
        _epi(
            mod,
            4 if gated else 5,
            _tile_rows(m0, cta_m, n_valid, gate, up),
            cta_m,
            n,
            meta,
            out,
            alpha=alpha,
            tok=tok,
            bias=bias,
        )
    got = out.view(out_dtype).reshape(-1, out_ld)
    expect = _sentinel(out.numel()).view(out_dtype).reshape(-1, out_ld).clone()
    expect[n0 : n0 + n_valid, :ncols] = y[:, :n_valid].T.to(out_dtype)
    _assert_bits(got, expect, nan_ok=True)
    if not gated:
        assert torch.isnan(got[n0 + 2, 7].float())


@pytest.mark.parametrize("pt", [True, False])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_epilogue_gemm2(pt, with_bias, out_dtype):
    """GEMM2 rows: T(acc (alpha s) + bias) per token, else T(acc alpha + bias) (FMA or not)."""
    mod = _module(_arch(), "rt")
    cta_m, n, n_valid, n0, expert, hidden = 128, 16, 11, 32, 2, 100
    out_ld = hidden + 4
    gen = torch.Generator().manual_seed(21 + int(pt) * 2 + int(with_bias))
    a_mat = _randn(gen, hidden, n, scale=4.0)
    alpha = torch.tensor([0.5, 2.0, 0.8125], device=_dev())
    bias = _randn(gen, 3, hidden).to(out_dtype) if with_bias else None
    tok = None
    scale = alpha[expert].expand(n)
    if pt:
        tok = torch.full((n0 + n,), _NAN, device=_dev())
        tok[n0 : n0 + n_valid] = (torch.rand(n_valid, generator=gen) + 0.5).to(_dev())
        scale = alpha[expert] * tok[n0 : n0 + n]
    b = bias[expert].float()[:, None] if bias is not None else 0.0
    two_step = (a_mat * scale + b).to(out_dtype)
    fused = a_mat.double() * scale.double() + (b.double() if bias is not None else 0.0)
    fused = fused.float().to(out_dtype)
    out = _sentinel((n0 + n) * out_ld)
    meta = [
        hidden,
        expert,
        0,
        0,
        n0,
        n_valid,
        int(A.Identity),
        _f16(out_dtype),
        0,
        out_ld,
    ]
    _epi(
        mod,
        6,
        _tile_rows(0, cta_m, n_valid, a_mat),
        cta_m,
        n,
        meta,
        out,
        alpha=alpha,
        tok=tok,
        bias=bias,
    )
    got = out.view(out_dtype).reshape(-1, out_ld)
    region = got[n0 : n0 + n_valid, :hidden]
    ok = _bits(region) == _bits(two_step[:, :n_valid].T.contiguous())
    if not pt and bias is not None:
        ok |= _bits(region) == _bits(fused[:, :n_valid].T.contiguous())
    assert ok.all(), f"{(~ok).sum().item()} GEMM2 values differ"
    untouched = torch.ones_like(got, dtype=torch.bool)
    untouched[n0 : n0 + n_valid, :hidden] = False
    assert (_bits(got)[untouched] == _FILL).all()


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_epilogue_tiles_unquantized(out_dtype):
    """GEMM1 into GEMM2 B tiles without quantization (W4A16 rounding: act(acc alpha)); the last
    m-tile zeroes the K half past the columns it covers."""
    mod = _module(_arch(), "rt")
    cta_m, n, n_valid, tile, n0, expert, ncols = 128, 16, 9, 2, 32, 0, 40
    rows, k_tiles, kp = 2 * ncols, 1, 128
    gen = torch.Generator().manual_seed(31)
    gate, up = _randn(gen, ncols, n, scale=2.0), _randn(gen, ncols, n, scale=2.0)
    alpha = torch.tensor([1.25], device=_dev())
    y = _act(mod, A.Swiglu, gate * alpha[0], up * alpha[0]).to(out_dtype)
    want = torch.zeros(n, kp, dtype=out_dtype, device=_dev())
    want[:n_valid, :ncols] = y[:, :n_valid].T
    out = _sentinel((tile + 1) * k_tiles * 2 * n * 128 // 2)
    meta = [rows, expert, 0, tile, n0, n_valid, int(A.Swiglu), _f16(out_dtype), 1, 0]
    _epi(
        mod,
        4,
        _tile_rows(0, cta_m, n_valid, gate, up),
        cta_m,
        n,
        meta,
        out,
        alpha=alpha,
    )
    _assert_bits(_read_tile(out, tile, k_tiles, n, kp, out_dtype), want)
    for t in range(tile):
        assert (out.view(torch.int16)[_tile_index(t, k_tiles, n, kp)] == _FILL).all()


# ----------------------------------------------------------------------------------------------
# row_quant: per-token NVFP4 of GEMM1 rows into GEMM2 B tiles


@pytest.mark.parametrize("mode", ["e0", "e1", "rt0", "rt1"])
@pytest.mark.parametrize("recipe", _RECIPES, ids=repr)
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dense", [False, True])
def test_row_quant(monkeypatch, mode, recipe, in_dtype, out_dtype, dense):
    mod = _module(_arch(), "rt" if mode.startswith("rt") else mode)
    _exact_env(monkeypatch, mode.endswith("1"))
    n, cols, in_ld, kp = 8, 80, 88, 128
    counts = [27] if dense else [5, 0, 19]
    local = sum(counts)
    rows = local + (0 if dense else 3)  # three expanded rows of non-local experts
    gen = torch.Generator().manual_seed(int(dense) * 100 + _RECIPES.index(recipe))
    scale = torch.logspace(-3, 3, rows)[:, None]
    x = (torch.randn(rows, in_ld, generator=gen) * scale).to(in_dtype).to(_dev())
    x[:, cols:] = _NAN  # never read
    x[3, :cols] = 0  # amax 0
    x[local:] = _NAN  # rows past the local experts are never read
    if dense:
        eo = etb = nt = None
        loc = [(p // n, p % n) for p in range(rows)]
        num_tiles = (rows + n - 1) // n
    else:
        off, base, loc = [0], [0], []
        for e, c in enumerate(counts):
            loc += [(base[e] + i // n, i % n) for i in range(c)]
            off.append(off[-1] + c)
            base.append(base[-1] + (c + n - 1) // n)
        num_tiles = base[-1]
        eo = torch.tensor(off, dtype=torch.int32, device=_dev())
        etb = torch.tensor(base, dtype=torch.int32, device=_dev())
        nt = torch.tensor([num_tiles], dtype=torch.int32, device=_dev())
    scale_inv = torch.tensor([nvfp4_per_token_scale_inv(recipe)], device=_dev())
    codes, sf, pts = nvfp4_quantize(
        x[:local, :cols].contiguous(),
        scale_inv,
        sfLayout=SfLayout.layout_linear,
        backend="cuda",
        per_token_activation=True,
        nvfp4_4over6=recipe,
    )
    codes = codes.view(torch.uint8).reshape(local, cols // 2)
    sf = sf.view(torch.uint8).reshape(local, cols // 16)
    deq = _dequant(codes, sf, out_dtype)

    tiles = _sentinel(num_tiles * 2 * n * 128 // 2)
    tok = torch.full((rows,), _NAN, device=_dev())
    fq_codes = torch.full((rows, cols // 2), 0x55, dtype=torch.uint8, device=_dev())
    fq_sf = torch.full((rows, cols // 16), 0x55, dtype=torch.uint8, device=_dev())
    meta = [
        in_ld,
        rows,
        cols,
        _f16(in_dtype),
        fp4_sm90_recipe_bits(recipe),
        _f16(out_dtype),
        n,
    ]
    mod.rq_run(x, meta, eo, etb, nt, tiles, tok, fq_codes, fq_sf)

    _assert_bits(tok[:local], pts.flatten().float())
    assert torch.equal(fq_codes[:local], codes) and torch.equal(fq_sf[:local], sf)
    assert (tok[local:] == 0).all() and (fq_codes[local:] == 0).all()
    assert (fq_sf[local:] == 0).all()
    want = torch.zeros(num_tiles, n, kp, dtype=out_dtype, device=_dev())
    for p, (t, r) in enumerate(loc):
        want[t, r, :cols] = deq[p]
    got = torch.stack(
        [_read_tile(tiles, t, 1, n, kp, out_dtype) for t in range(num_tiles)]
    )
    _assert_bits(got, want)


def test_row_quant_errors():
    mod = _module(_arch(), "rt")
    x = torch.zeros(4, 32, dtype=torch.bfloat16, device=_dev())
    tiles = torch.zeros(2 * 8 * 128, dtype=torch.int16, device=_dev())
    tok = torch.zeros(4, device=_dev())
    with pytest.raises(Exception, match="cols"):
        mod.rq_run(x, [32, 4, 24, 0, 0, 0, 8], None, None, None, tiles, tok, None, None)
    with pytest.raises(Exception, match="recipe"):
        mod.rq_run(x, [32, 4, 32, 0, 2, 0, 8], None, None, None, tiles, tok, None, None)
    with pytest.raises(Exception, match="n must"):
        mod.rq_run(
            x, [32, 4, 32, 0, 0, 0, 12], None, None, None, tiles, tok, None, None
        )


# ----------------------------------------------------------------------------------------------
# finalize: ordered top-k sum


_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _finalize_inputs(gen, tokens, top_k, hidden, y_ld, y_dtype, w_dtype):
    """Products w y from {+-2^11, +-2^-13, 2^-12} (exact in every dtype): an FP32 sum of 2^11 and
    2^-13 rounds, so the result depends on the slot order. ~15% of the slots are not local."""
    slots = tokens * top_k
    rows = slots + 5
    perm = torch.randperm(rows, generator=gen)[:slots].to(torch.int32)
    perm[torch.rand(slots, generator=gen) < 0.15] = -1
    choices = torch.tensor([2.0**11, -(2.0**11), 2.0**-13, -(2.0**-13), 2.0**-12])
    prod = choices[torch.randint(0, 5, (slots, hidden), generator=gen)]
    w = None
    w_val = torch.ones(slots)
    if w_dtype is not None:
        w_val = torch.tensor([0.5, 1.0, 2.0])[
            torch.randint(0, 3, (slots,), generator=gen)
        ]
        w = w_val.to(w_dtype).to(_dev())
    y = torch.zeros(rows, y_ld)
    valid = perm >= 0
    y[perm[valid].long(), :hidden] = prod[valid] / w_val[valid][:, None]
    return y.to(y_dtype).to(_dev()), perm.to(_dev()), w, w_val.to(_dev())


def _ordered_sum(y, perm, w_val, tokens, top_k, hidden, order):
    acc = torch.zeros(tokens, hidden, device=_dev())
    yf = y.float()
    for k in order:
        rows = perm.view(tokens, top_k)[:, k]
        prod = (
            w_val.view(tokens, top_k)[:, k, None]
            * yf[rows.clamp_min(0).long(), :hidden]
        )
        acc = torch.where((rows >= 0)[:, None], acc + prod, acc)
    return acc


@pytest.mark.parametrize("y_dtype", _DTYPES)
@pytest.mark.parametrize("w_dtype", _DTYPES + [None])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,y_ld,out_ld", [(136, 144, 136), (129, 131, 133)])
def test_finalize(y_dtype, w_dtype, out_dtype, hidden, y_ld, out_ld):
    mod = _module(_arch(), "rt")
    tokens, top_k = 33, 4
    gen = torch.Generator().manual_seed(hidden + 7 * _DTYPES.index(y_dtype))
    y, perm, w, w_val = _finalize_inputs(
        gen, tokens, top_k, hidden, y_ld, y_dtype, w_dtype
    )
    seq = _ordered_sum(y, perm, w_val, tokens, top_k, hidden, range(top_k))
    rev = _ordered_sum(y, perm, w_val, tokens, top_k, hidden, reversed(range(top_k)))
    # The order must still show after the output rounding (cancelling +-2^11 terms).
    assert not torch.equal(seq.to(out_dtype), rev.to(out_dtype)), (
        "inputs must be order-sensitive"
    )
    codes = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
    meta = [
        codes[y_dtype],
        codes.get(w_dtype, 0),
        _f16(out_dtype),
        tokens,
        top_k,
        hidden,
    ]
    meta += [y_ld, out_ld]
    outs = []
    for _ in range(2):
        out = torch.full((tokens, out_ld), _NAN, dtype=out_dtype, device=_dev())
        mod.fin_run(y, perm, w, out, meta)
        outs.append(out)
    _assert_bits(outs[0][:, :hidden], seq.to(out_dtype))
    assert torch.equal(_bits(outs[0]), _bits(outs[1]))
    assert torch.isnan(outs[0][:, hidden:].float()).all()
