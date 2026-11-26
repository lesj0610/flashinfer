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

SM90 FP4 prep (``csrc/fp4_sm90/prep.cuh``): routing sort and n-tile list against a CPU
stable sort, GEMM1 B tiles (layout, zero padding) against the input, and the in-prep NVFP4
quantization against FlashInfer's own quantizers (``fp4_quantize`` and per-token
``nvfp4_quantize``, CUDA backend, same device) for every recipe.
"""

import functools
import itertools

import pytest
import torch

from flashinfer import SfLayout, block_scale_interleave, fp4_quantize, nvfp4_quantize
from flashinfer.fused_moe.sm90.spec import XFmt
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.jit.fp4_quantization import sw_cvt_nvcc_flags
from flashinfer.jit.fused_moe import fp4_sm90_recipe_bits
from flashinfer.jit.utils import write_if_different
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    nvfp4_per_token_scale_inv,
)
from flashinfer.utils import get_compute_capability

_SOURCE = r"""
#include "fp4_sm90/prep.cuh"
#include "tvm_ffi_utils.h"

using flashinfer::fp4_sm90::PrepArgs;
using flashinfer::fp4_sm90::XFmt;
using tvm::ffi::Array;
using tvm::ffi::Optional;

template <class T>
T* opt(const Optional<TensorView>& t) {
  return t.has_value() ? static_cast<T*>(t.value().data_ptr()) : nullptr;
}

// meta: tokens, top_k, num_experts, local_expert_offset, num_local_experts, k, x_ld, x_fmt,
// quantize, per_token, recipe, f16, n, max_tiles
PrepArgs make_args(const Optional<TensorView>& topk_ids, const Array<int64_t>& meta) {
  TVM_FFI_ICHECK_EQ(meta.size(), 14);
  PrepArgs a{};
  a.topk_ids = opt<const int32_t>(topk_ids);
  a.tokens = int(meta[0]);
  a.top_k = int(meta[1]);
  a.num_experts = int(meta[2]);
  a.local_expert_offset = int(meta[3]);
  a.num_local_experts = int(meta[4]);
  a.k = int(meta[5]);
  a.x_ld = meta[6];
  a.x_fmt = static_cast<XFmt>(meta[7]);
  a.quantize = uint8_t(meta[8]);
  a.per_token = uint8_t(meta[9]);
  a.recipe = uint8_t(meta[10]);
  a.f16 = uint8_t(meta[11]);
  a.n = int(meta[12]);
  a.max_tiles = int(meta[13]);
  return a;
}

int64_t workspace_bytes(Optional<TensorView> topk_ids, Array<int64_t> meta) {
  return int64_t(flashinfer::fp4_sm90::fp4_sm90_prep_workspace_bytes(make_args(topk_ids, meta)));
}

void run(TensorView x, Optional<TensorView> topk_ids, Optional<TensorView> x_sf,
         Optional<TensorView> x_global, Array<int64_t> meta, Optional<TensorView> expert_offset,
         Optional<TensorView> expert_tile_base, Optional<TensorView> permuted_src,
         Optional<TensorView> src_permuted, Optional<TensorView> tile_expert,
         Optional<TensorView> tile_n0, Optional<TensorView> num_tiles,
         Optional<TensorView> x_tiles, Optional<TensorView> tok_scale,
         Optional<TensorView> workspace) {
  PrepArgs a = make_args(topk_ids, meta);
  a.x = x.data_ptr();
  a.x_sf = opt<const uint8_t>(x_sf);
  a.x_global = opt<const float>(x_global);
  a.expert_offset = opt<int32_t>(expert_offset);
  a.expert_tile_base = opt<int32_t>(expert_tile_base);
  a.permuted_src = opt<int32_t>(permuted_src);
  a.src_permuted = opt<int32_t>(src_permuted);
  a.tile_expert = opt<int32_t>(tile_expert);
  a.tile_n0 = opt<int32_t>(tile_n0);
  a.num_tiles = opt<int32_t>(num_tiles);
  a.x_tiles = opt<void>(x_tiles);
  a.tok_scale = opt<float>(tok_scale);
  a.workspace = opt<void>(workspace);
  ffi::CUDADeviceGuard guard(x.device().device_id);
  flashinfer::fp4_sm90::launch_fp4_sm90_prep(a, get_stream(x.device()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_bytes, workspace_bytes);
"""

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 += [-v for v in _E2M1]  # code 8 is -0.0

_RECIPES = [None] + [
    NVFP44Over6Config(e4m3_max=m, err_mode=e, err_use_fast_math=f)
    for m, e, f in itertools.product((448, 256), ("MAE", "MSE"), (False, True))
]


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(_device())
    if major not in (8, 9):
        pytest.skip("the SM90 prep runs on SM90 (and builds for SM80 to SM89 in tests)")
    return "90" if major == 9 else "80"


@functools.cache
def _module(arch: str):
    name = f"test_sm90_prep_quant_{arch}"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "prep_test.cu"
    write_if_different(source, _SOURCE)
    nv_internal = jit_env.FLASHINFER_CSRC_DIR / "nv_internal"
    common = nv_internal / "cpp" / "common"
    defines = ["-DENABLE_BF16", "-DENABLE_FP8"]
    defines += ["-DENABLE_FP4"] if is_cuda_version_at_least("12.8") else []
    # The flags of FlashInfer's own quantization module of that arch, -use_fast_math included.
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


@pytest.fixture(scope="module")
def mod():
    return _module(_arch())


# ----------------------------------------------------------------------------------------------
# References


def _ref_schedule(ids, top_k, local_off, experts, n, tokens):
    """CPU stable sort by local expert, then the n-tile list (dense: ids None, one expert)."""
    total = tokens * top_k
    flat = list(range(total)) if ids is None else ids.flatten().cpu().tolist()
    local = [0] * total if ids is None else [e - local_off for e in flat]
    order = sorted(
        (i for i in range(total) if 0 <= local[i] < experts),
        key=lambda i: (local[i], i),
    )
    counts = [0] * experts
    for i in order:
        counts[local[i]] += 1
    offset, base = [0], [0]
    for c in counts:
        offset.append(offset[-1] + c)
        base.append(base[-1] + (c + n - 1) // n)
    src_permuted = [-1] * total
    for p, i in enumerate(order):
        src_permuted[i] = p
    tile_expert, tile_n0 = [], []
    for e in range(experts):
        for t in range(base[e + 1] - base[e]):
            tile_expert.append(e)
            tile_n0.append(offset[e] + t * n)
    return dict(
        expert_offset=offset,
        expert_tile_base=base,
        permuted_src=order,
        src_permuted=src_permuted,
        tile_expert=tile_expert,
        tile_n0=tile_n0,
        num_tiles=base[-1],
    )


def _max_tiles(total, experts, n):
    return (total + min(experts, total) * (n - 1)) // n


def _b_rows(tiles: torch.Tensor, num_tiles: int, n: int, k: int, dtype) -> torch.Tensor:
    """B tile bytes (params.cuh b_tile_byte) back to [num_tiles * n, ceil(k / 128) * 128]."""
    k_tiles = (k + 127) // 128
    dev = tiles.device
    g = torch.arange(num_tiles * n, device=dev)[:, None]
    t, r = g // n, g % n
    col = torch.arange(k_tiles * 128, device=dev)[None, :]
    kt, kh, c = col // 128, (col % 128) // 64, col % 64
    byte = ((t * k_tiles + kt) * 2 + kh) * n * 128 + r * 128 + ((c // 8) ^ (r % 8)) * 16
    byte = byte + (c % 8) * 2
    return tiles.view(dtype).flatten()[byte // 2]


def _expected_rows(sched, values, n, top_k, k, dense, tokens):
    """Rows of B per tile row: the token's values, 0 past K and on rows past the expert."""
    rows = sched["num_tiles"] * n
    k_pad = (k + 127) // 128 * 128
    out = torch.zeros(rows, k_pad, dtype=values.dtype, device=values.device)
    src = []
    for g in range(rows):
        t, r = divmod(g, n)
        if dense:
            src.append(g if g < tokens else -1)
            continue
        e, p = sched["tile_expert"][t], sched["tile_n0"][t] + r
        src.append(
            sched["permuted_src"][p] // top_k
            if p < sched["expert_offset"][e + 1]
            else -1
        )
    src_t = torch.tensor(src, device=values.device)
    valid = src_t >= 0
    out[valid, :k] = values[src_t[valid]]
    return out, src


def _dequant(codes: torch.Tensor, sf_linear: torch.Tensor, dtype) -> torch.Tensor:
    """code x E4M3 scale of FP4 codes [T, K / 2] and linear scales [T, K / 16] (exact)."""
    table = torch.tensor(_E2M1, device=codes.device)
    nib = (
        torch.stack((codes & 0xF, codes >> 4), dim=-1)
        .reshape(codes.shape[0], -1)
        .long()
    )
    sf = sf_linear.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return (table[nib] * sf).to(dtype)


def _dequant_mx(codes: torch.Tensor, sf_linear: torch.Tensor) -> torch.Tensor:
    """code x 2^(e-127) of MXFP4 codes [T, K / 2] and UE8M0 scales [T, K / 32] as BF16.

    Computed on the CPU in FP32 (subnormals kept); e = 0 is 2^-127, e = 255 is NaN.
    """
    codes, sf_linear = codes.cpu(), sf_linear.cpu()
    table = torch.tensor(_E2M1)
    nib = (
        torch.stack((codes & 0xF, codes >> 4), dim=-1)
        .reshape(codes.shape[0], -1)
        .long()
    )
    e = sf_linear.int().repeat_interleave(32, dim=1)
    bits = torch.where(e == 0, 0x00400000, e << 23)
    bits = torch.where(e == 255, 0x7FC00000, bits)
    return (table[nib] * bits.int().view(torch.float32)).to(torch.bfloat16)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _bits_equal_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit equality with any NaN matching any NaN."""
    na, nb = torch.isnan(a), torch.isnan(b)
    return torch.equal(na, nb) and _bits_equal(a[~na], b[~nb])


def _input(tokens, k, dtype, seed=0):
    """Rows of very different magnitudes, an all-zero row and a row of tiny values."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tokens, k, generator=gen)
    x *= torch.pow(10.0, torch.empty(tokens, 1).uniform_(-3, 3, generator=gen))
    if tokens > 2:
        x[1] = 0
        x[2] *= 1e-6
    return x.to(dtype).to(_device())


# ----------------------------------------------------------------------------------------------
# Runner


def _run(
    mod,
    x,
    k,
    *,
    ids=None,
    tokens=None,
    top_k=1,
    num_experts=1,
    local_off=0,
    experts=1,
    n=64,
    x_fmt=XFmt.BF16,
    x_sf=None,
    x_global=None,
    quantize=False,
    per_token=False,
    recipe=0,
    f16=False,
    tiles=True,
    max_tiles=None,
    dense_sched=True,
):
    dev = _device()
    tokens = x.shape[0] if tokens is None else tokens
    total = tokens * top_k
    dense = ids is None
    if max_tiles is None:
        max_tiles = _max_tiles(total, experts, n)
    meta = [tokens, top_k, num_experts, local_off, experts, k, x.stride(0), int(x_fmt)]
    meta += [int(quantize), int(per_token), recipe, int(f16), n, max_tiles]
    i32 = functools.partial(torch.full, dtype=torch.int32, device=dev, fill_value=-7)
    want_sched = not dense or dense_sched
    out = dict(
        expert_offset=i32((experts + 1,)) if want_sched else None,
        expert_tile_base=i32((experts + 1,)) if want_sched else None,
        permuted_src=i32((max(total, 1),)) if want_sched else None,
        src_permuted=i32((max(total, 1),)) if want_sched else None,
        tile_expert=i32((max(max_tiles, 1),)) if want_sched else None,
        tile_n0=i32((max(max_tiles, 1),)) if want_sched else None,
        num_tiles=i32((1,)) if want_sched else None,
    )
    k_tiles = (k + 127) // 128
    # Dense without num_tiles: the GEMM walks max_tiles tiles and prep writes them all.
    n_tiles = (total + n - 1) // n if dense and dense_sched else max_tiles
    out["x_tiles"] = (
        torch.full(
            (max(n_tiles, 1) * k_tiles * n * 256,), 0xFF, dtype=torch.uint8, device=dev
        )
        if tiles
        else None
    )
    out["tok_scale"] = (
        torch.full((max(total, 1),), float("nan"), device=dev) if per_token else None
    )
    ws_bytes = mod.workspace_bytes(ids, meta)
    ws = (
        torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=dev)
        if ws_bytes
        else None
    )
    mod.run(
        x,
        ids,
        x_sf,
        x_global,
        meta,
        out["expert_offset"],
        out["expert_tile_base"],
        out["permuted_src"],
        out["src_permuted"],
        out["tile_expert"],
        out["tile_n0"],
        out["num_tiles"],
        out["x_tiles"],
        out["tok_scale"],
        ws,
    )
    torch.cuda.synchronize()
    return out


def _check_sched(out, ref, total):
    E = len(ref["expert_offset"]) - 1
    assert out["expert_offset"].tolist() == ref["expert_offset"]
    assert out["expert_tile_base"].tolist() == ref["expert_tile_base"]
    assert out["num_tiles"].item() == ref["num_tiles"]
    count = ref["expert_offset"][E]
    assert out["permuted_src"][:count].tolist() == ref["permuted_src"]
    if total:
        assert out["src_permuted"][:total].tolist() == ref["src_permuted"]
    nt = ref["num_tiles"]
    assert out["tile_expert"][:nt].tolist() == ref["tile_expert"]
    assert out["tile_n0"][:nt].tolist() == ref["tile_n0"]


# ----------------------------------------------------------------------------------------------
# Routing sort and tile list


@pytest.mark.parametrize("tokens", [1, 7, 300, 5000])
@pytest.mark.parametrize("top_k", [1, 4, 8])
@pytest.mark.parametrize("n", [8, 64, 256])
@pytest.mark.parametrize(
    "num_experts,local_off,experts", [(8, 0, 8), (64, 16, 16), (512, 0, 512)]
)
def test_sort_and_tiles(mod, tokens, top_k, n, num_experts, local_off, experts):
    gen = torch.Generator(device="cpu").manual_seed(tokens * 131 + top_k * 17 + n)
    ids = torch.randint(
        0, num_experts, (tokens, top_k), generator=gen, dtype=torch.int32
    )
    ids = ids.to(_device())
    x = torch.zeros(tokens, 128, dtype=torch.bfloat16, device=_device())
    out = _run(
        mod,
        x,
        128,
        ids=ids,
        top_k=top_k,
        num_experts=num_experts,
        local_off=local_off,
        experts=experts,
        n=n,
        tiles=False,
    )
    _check_sched(
        out, _ref_schedule(ids, top_k, local_off, experts, n, tokens), tokens * top_k
    )


@pytest.mark.parametrize("tokens", [1, 100, 257])
@pytest.mark.parametrize("n", [8, 128])
def test_dense_schedule(mod, tokens, n):
    x = torch.zeros(tokens, 64, dtype=torch.bfloat16, device=_device())
    out = _run(mod, x, 64, n=n, tiles=False)
    _check_sched(out, _ref_schedule(None, 1, 0, 1, n, tokens), tokens)


# ----------------------------------------------------------------------------------------------
# B tiles: copy (layout and zero padding)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("k", [200, 256, 392])
@pytest.mark.parametrize("dense", [True, False])
def test_copy_tiles(mod, dtype, k, dense):
    tokens, top_k, experts, n = 37, 2, 4, 32
    base = _input(tokens, k + 8, dtype)
    x = base[:, :k]  # rows k + 8 elements apart
    ids = None
    if not dense:
        gen = torch.Generator(device="cpu").manual_seed(k)
        ids = torch.randint(
            0, experts, (tokens, top_k), generator=gen, dtype=torch.int32
        )
        ids = ids.to(_device())
    top_k = 1 if dense else top_k
    out = _run(
        mod,
        x,
        k,
        ids=ids,
        top_k=top_k,
        num_experts=experts,
        experts=1 if dense else experts,
        n=n,
        x_fmt=XFmt.F16 if dtype == torch.float16 else XFmt.BF16,
        f16=dtype == torch.float16,
    )
    sched = _ref_schedule(ids, top_k, 0, 1 if dense else experts, n, tokens)
    got = _b_rows(out["x_tiles"], sched["num_tiles"], n, k, dtype)
    want, _ = _expected_rows(sched, x, n, top_k, k, dense, tokens)
    assert _bits_equal(got, want)


# ----------------------------------------------------------------------------------------------
# B tiles: in-prep NVFP4 quantization against FlashInfer's quantizers


@pytest.mark.parametrize("recipe", _RECIPES, ids=repr)
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("b_f16", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("dense", [True, False])
def test_quant_tiles(
    mod, monkeypatch, recipe, exact, in_dtype, b_f16, per_token, dense
):
    if exact:
        monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")
    else:
        monkeypatch.delenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", raising=False)
        monkeypatch.delenv("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH", raising=False)
    tokens, k, top_k, experts, n = 37, 320, 2, 4, 64
    x = _input(tokens, k, in_dtype, seed=int(exact) * 7 + int(per_token))
    b_dtype = torch.float16 if b_f16 else torch.bfloat16
    if per_token:
        scale_inv = torch.tensor([nvfp4_per_token_scale_inv(recipe)], device=_device())
        codes, sf, pts = nvfp4_quantize(
            x,
            scale_inv,
            sfLayout=SfLayout.layout_linear,
            backend="cuda",
            per_token_activation=True,
            nvfp4_4over6=recipe,
        )
        # null x_global: the recipe's default inverse global scale
        x_global = None if in_dtype == torch.float16 else scale_inv
    else:
        amax = x.float().abs().max()
        gs = (
            torch.tensor([(recipe.e4m3_max if recipe else 448) * 6.0], device=_device())
            / amax
        )
        codes, sf = fp4_quantize(
            x,
            gs,
            16,
            False,
            is_sf_swizzled_layout=False,
            backend="cuda",
            nvfp4_4over6=recipe,
        )
        x_global = gs
    deq = _dequant(codes, sf.view(torch.uint8).reshape(tokens, k // 16), b_dtype)
    ids = None
    if not dense:
        gen = torch.Generator(device="cpu").manual_seed(3)
        ids = torch.randint(
            0, experts, (tokens, top_k), generator=gen, dtype=torch.int32
        )
        ids = ids.to(_device())
    top_k = 1 if dense else top_k
    out = _run(
        mod,
        x,
        k,
        ids=ids,
        top_k=top_k,
        num_experts=experts,
        experts=1 if dense else experts,
        n=n,
        x_fmt=XFmt.F16 if in_dtype == torch.float16 else XFmt.BF16,
        x_global=x_global,
        quantize=True,
        per_token=per_token,
        recipe=fp4_sm90_recipe_bits(recipe),
        f16=b_f16,
    )
    sched = _ref_schedule(ids, top_k, 0, 1 if dense else experts, n, tokens)
    got = _b_rows(out["x_tiles"], sched["num_tiles"], n, k, b_dtype)
    want, src = _expected_rows(sched, deq, n, top_k, k, dense, tokens)
    if per_token:  # first stage: the per-token scale of each expanded row
        rows = sched["permuted_src"] if not dense else list(range(tokens))
        want_pts = pts.flatten()[torch.tensor(rows, device=_device()) // top_k]
        assert torch.equal(out["tok_scale"][: len(rows)], want_pts)
    mismatch = (got.view(torch.int16) != want.view(torch.int16)).nonzero()
    assert mismatch.numel() == 0, (
        f"{mismatch.shape[0]} B values differ, first at (row, col) {mismatch[0].tolist()}, "
        f"token {src[mismatch[0, 0].item()]}"
    )


# ----------------------------------------------------------------------------------------------
# B tiles: FP4 input (exact dequantized values)


@pytest.mark.parametrize("layout", ["linear", "128x4", "8x4"])
@pytest.mark.parametrize("b_f16", [False, True])
@pytest.mark.parametrize("dense", [True, False])
def test_fp4_input_tiles(mod, layout, b_f16, dense):
    tokens, k, top_k, experts, n = 37, 320, 2, 4, 64
    x = _input(tokens, k, torch.bfloat16, seed=5)
    gs = torch.tensor([448 * 6.0], device=_device()) / x.float().abs().max()
    codes, sf_lin = fp4_quantize(
        x, gs, 16, False, is_sf_swizzled_layout=False, backend="cuda", nvfp4_4over6=None
    )
    codes_l, sf_in = codes, sf_lin
    if layout != "linear":
        codes_l, sf_in = fp4_quantize(
            x,
            gs,
            16,
            False,
            is_sf_swizzled_layout=True,
            is_sf_8x4_layout=layout == "8x4",
            backend="cuda",
            nvfp4_4over6=None,
        )
        assert torch.equal(codes_l, codes)
    fmt = {
        "linear": XFmt.NVFP4_LINEAR,
        "128x4": XFmt.NVFP4_SWZ128X4,
        "8x4": XFmt.NVFP4_SWZ8X4,
    }
    b_dtype = torch.float16 if b_f16 else torch.bfloat16
    ids = None
    if not dense:
        gen = torch.Generator(device="cpu").manual_seed(11)
        ids = torch.randint(
            0, experts, (tokens, top_k), generator=gen, dtype=torch.int32
        )
        ids = ids.to(_device())
    top_k = 1 if dense else top_k
    out = _run(
        mod,
        codes,
        k,
        ids=ids,
        top_k=top_k,
        num_experts=experts,
        experts=1 if dense else experts,
        n=n,
        x_fmt=fmt[layout],
        x_sf=sf_in.view(torch.uint8).flatten(),
        f16=b_f16,
    )
    deq = _dequant(codes, sf_lin.view(torch.uint8).reshape(tokens, k // 16), b_dtype)
    sched = _ref_schedule(ids, top_k, 0, 1 if dense else experts, n, tokens)
    got = _b_rows(out["x_tiles"], sched["num_tiles"], n, k, b_dtype)
    want, _ = _expected_rows(sched, deq, n, top_k, k, dense, tokens)
    assert _bits_equal(got, want)


@pytest.mark.parametrize("dense", [True, False])
def test_mxfp4_input_tiles(mod, dense):
    """Every UE8M0 byte (0 and 255 included) and every code, SF swizzled by FI."""
    tokens, k, top_k, experts, n = 37, 320, 2, 4, 64
    gen = torch.Generator(device="cpu").manual_seed(13)
    codes = torch.randint(0, 256, (tokens, k // 2), generator=gen, dtype=torch.uint8)
    sf_lin = torch.randint(0, 256, (tokens, k // 32), generator=gen, dtype=torch.uint8)
    sf_lin.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
    sf_swz = block_scale_interleave(sf_lin.to(_device()))
    ids = None
    if not dense:
        gen = torch.Generator(device="cpu").manual_seed(17)
        ids = torch.randint(
            0, experts, (tokens, top_k), generator=gen, dtype=torch.int32
        )
        ids = ids.to(_device())
    top_k = 1 if dense else top_k
    out = _run(
        mod,
        codes.to(_device()),
        k,
        ids=ids,
        top_k=top_k,
        num_experts=experts,
        experts=1 if dense else experts,
        n=n,
        x_fmt=XFmt.MXFP4_SWZ128X4,
        x_sf=sf_swz,
    )
    deq = _dequant_mx(codes, sf_lin).to(_device())
    sched = _ref_schedule(ids, top_k, 0, 1 if dense else experts, n, tokens)
    got = _b_rows(out["x_tiles"], sched["num_tiles"], n, k, torch.bfloat16)
    want, _ = _expected_rows(sched, deq, n, top_k, k, dense, tokens)
    assert _bits_equal_nan(got, want)


# ----------------------------------------------------------------------------------------------
# Dense with num_tiles null: max_tiles tiles written, zero past the tokens


def test_dense_tiles_without_num_tiles(mod):
    tokens, k, n = 37, 256, 16
    x = _input(tokens, k, torch.bfloat16, seed=7)
    max_tiles = (tokens + n - 1) // n + 2
    out = _run(mod, x, k, n=n, max_tiles=max_tiles, dense_sched=False)
    got = _b_rows(out["x_tiles"], max_tiles, n, k, torch.bfloat16)
    want = torch.zeros_like(got)
    want[:tokens, :k] = x
    assert _bits_equal(got, want)


# ----------------------------------------------------------------------------------------------
# SM90: FlashInfer's TMA quantizer (large inputs) against the prep's per-row quantizer


@pytest.mark.parametrize("recipe", [None, NVFP44Over6Config()], ids=repr)
def test_quant_tiles_tma_branch(mod, recipe):
    if _arch() != "90":
        pytest.skip("FlashInfer's TMA FP4 quantizer is built for SM90 only below SM100")
    tokens, k, n = (
        8192,
        4096,
        128,
    )  # m >= 1024, 64 MiB of BF16, enough work tiles for TMA
    x = _input(tokens, k, torch.bfloat16, seed=9)
    gs = torch.tensor([(recipe.e4m3_max if recipe else 448) * 6.0], device=_device())
    gs = gs / x.float().abs().max()
    codes, sf = fp4_quantize(
        x,
        gs,
        16,
        False,
        is_sf_swizzled_layout=False,
        backend="cuda",
        nvfp4_4over6=recipe,
    )
    out = _run(
        mod,
        x,
        k,
        n=n,
        x_global=gs,
        quantize=True,
        recipe=fp4_sm90_recipe_bits(recipe),
        dense_sched=False,
    )
    deq = _dequant(codes, sf.view(torch.uint8).reshape(tokens, k // 16), torch.bfloat16)
    got = _b_rows(out["x_tiles"], tokens // n, n, k, torch.bfloat16)
    assert _bits_equal(got, deq)


# ----------------------------------------------------------------------------------------------
# Rejected arguments


def test_errors(mod):
    dev = _device()
    x = _input(16, 128, torch.bfloat16)
    ids = torch.zeros(16, 2, dtype=torch.int32, device=dev)
    kw = dict(ids=ids, top_k=2, num_experts=4, experts=4, n=64)
    with pytest.raises(Exception, match="need x_tiles"):
        _run(mod, x, 128, quantize=True, tiles=False, **kw)
    with pytest.raises(Exception, match="below the routing bound"):
        _run(mod, x, 128, max_tiles=_max_tiles(32, 4, 64) - 1, **kw)
    with pytest.raises(Exception, match="need the 4over6 bit"):
        _run(mod, x, 128, quantize=True, recipe=2, **kw)
    with pytest.raises(Exception, match="power of two"):
        _run(mod, x, 128, ids=ids, top_k=2, num_experts=4, experts=4, n=48)
    with pytest.raises(Exception, match="dtype must be the B dtype"):
        _run(mod, x, 128, f16=True, **kw)
    with pytest.raises(Exception, match="per_token needs quantize"):
        _run(mod, x, 128, per_token=True, **kw)
    codes = torch.zeros(16, 64, dtype=torch.uint8, device=dev)
    sf = torch.zeros(16 * 8, dtype=torch.uint8, device=dev)
    with pytest.raises(Exception, match="neither quantized again"):
        _run(mod, codes, 128, x_fmt=XFmt.NVFP4_LINEAR, x_sf=sf, quantize=True, **kw)
    with pytest.raises(Exception, match="MXFP4 input needs BF16 B"):
        _run(mod, codes, 128, x_fmt=XFmt.MXFP4_SWZ128X4, x_sf=sf, f16=True, **kw)
    with pytest.raises(Exception, match="k % 32 == 0"):
        _run(mod, codes[:, :56], 112, x_fmt=XFmt.MXFP4_SWZ128X4, x_sf=sf, **kw)
