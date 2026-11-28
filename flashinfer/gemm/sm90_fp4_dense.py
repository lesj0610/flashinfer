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

SM90 (wgmma) dense FP4 GEMM (``csrc/fp4_dense_sm90.cu``).

The SM100+ dense entry points (``mm_bf16_fp4``, ``mm_fp4`` and the experimental cake
per-token backend) run on SM90 here, on the operands in each entry point's own SM100+
format. No call reorders a weight or caches a data pointer: the kernel reads the bytes the
caller prepared (``Sm90WeightView``). Every backend name of an entry point keeps its SM100+
requirements and reaches this module only after them.

Weight formats (``AFmt``) by entry point and backend:

* ``mm_bf16_fp4``  cudnn, blackwell-native: ``LINEAR_SF`` (uint8 ``[N, K/2]``, linear E4M3
  ``[N, K/16]``); cute-dsl uint8 (SM10x prep) and cute-dsl-native: ``NVFP4_SWZ128X4``;
  cute-dsl int32 (SM12x prep) and blackwell-tiled: ``DENSE_CUTE_SM12X``.
* ``mm_fp4`` NVFP4: ``NVFP4_SWZ128X4``, trtllm ``NVFP4_TRTLLM_SHUFFLED``; MXFP4:
  ``MXFP4_SWZ128X4``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..api_logging import _is_current_stream_capturing
from ..autotuner import AutoTuner, OptimizationProfile, TunableRunner
from ..fused_moe.sm90.spec import AFmt, Pair, Sm90WeightView, XFmt
from ..jit.fp4_dense_sm90 import gen_fp4_dense_sm90_module

__all__ = [
    "SM90_DENSE_TACTICS",
    "Sm90Fp4DenseRunner",
    "default_sm90_dense_tactic",
    "sm90_dense_weight_view",
    "sm90_fp4_dense",
    "sm90_mm_bf16_fp4",
    "sm90_mm_fp4",
]

#: D7 shapes (cta_m, n, stages); stages 0 = the most that fit shared memory, as built.
SM90_DENSE_TACTICS: Tuple[Tuple[int, int, int], ...] = tuple(
    (cta_m, n, 0)
    for cta_m in (128, 256)
    for n in (8, 16, 32, 64, 128, 256)
    if (cta_m // 128) * n // 2 <= 128
)

_DENSE_A_FMTS = {
    AFmt.NVFP4_TRTLLM_SHUFFLED: "nvfp4_trtllm_shuffled",
    AFmt.NVFP4_SWZ128X4: "nvfp4_swz128x4",
    AFmt.NVFP4_LINEAR_SF: "nvfp4_linear_sf",
    AFmt.MXFP4_SWZ128X4: "mxfp4_swz128x4",
    AFmt.DENSE_CUTE_SM12X: "dense_cute_sm12x",
}
_ACT_LAYOUTS = {
    "128x4": XFmt.NVFP4_SWZ128X4,
    "8x4": XFmt.NVFP4_SWZ8X4,
    "linear": XFmt.NVFP4_LINEAR,
}


@functools.cache
def get_fp4_dense_sm90_module(f16: bool, a_fmt: AFmt):
    if a_fmt not in _DENSE_A_FMTS:
        raise ValueError(f"SM90 dense FP4 has no weight format {a_fmt!r}")
    return gen_fp4_dense_sm90_module(f16, _DENSE_A_FMTS[a_fmt]).build_and_load()


@functools.cache
def _module_tactics(
    f16: bool, a_fmt: AFmt, a_path: int
) -> Tuple[Tuple[int, int, int], ...]:
    flat = list(get_fp4_dense_sm90_module(f16, a_fmt).fp4_dense_sm90_tactics(a_path))
    return tuple(tuple(flat[i : i + 3]) for i in range(0, len(flat), 3))


def default_sm90_dense_tactic(m: int) -> Tuple[int, int, int]:
    """Untuned tactic (D7): CTA_M 128, n the smallest power of two in [8, 256] at least M."""
    n = 8
    while n < m and n < 256:
        n *= 2
    return (128, n, 0)


_ones_cache: dict = {}


def _ones(device: torch.device) -> torch.Tensor:
    """The fp32 scalar 1 for calls without an alpha (a constant, not derived from a weight)."""
    key = (device.type, device.index)
    value = _ones_cache.get(key)
    if value is None:
        if _is_current_stream_capturing():
            raise RuntimeError(
                "SM90 dense FP4 needs a constant on the device; run one call before "
                "capturing it in a CUDA graph."
            )
        value = _ones_cache[key] = torch.ones(1, dtype=torch.float32, device=device)
    return value


def _pad(x: int, y: int) -> int:
    return -(-x // y) * y


def _flat_bytes(t: torch.Tensor, nbytes: int) -> torch.Tensor:
    """``nbytes`` bytes of ``t``'s storage from its first element, as a flat uint8 alias (no
    copy). A strided view of one dense block (the 6-D scale view of the SM10x cute-dsl
    prep) is read by its base pointer."""
    if t.is_contiguous():
        return t.view(torch.uint8).reshape(-1)
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        t.untyped_storage(), t.storage_offset() * t.element_size(), (nbytes,), (1,)
    )


def _sf_bytes_needed(a_fmt: AFmt, rows: int, k: int) -> int:
    if a_fmt in (AFmt.NVFP4_LINEAR_SF, AFmt.DENSE_CUTE_SM12X):
        return rows * (k // 16)
    block = 32 if a_fmt == AFmt.MXFP4_SWZ128X4 else 16
    return _pad(rows, 128) * _pad(-(-k // block), 4)


def sm90_dense_weight_view(
    entry: str, a_fmt: AFmt, codes: torch.Tensor, sf: torch.Tensor, rows: int, k: int
) -> Sm90WeightView:
    """The ``Sm90WeightView`` of a dense weight as an SM100+ entry point prepared it (a
    description of the caller's bytes: no copy). ``codes`` holds ``rows * k / 2`` bytes
    (uint8 ``[rows, k / 2]``, or the int32 ``[k / 16, 2 * rows]`` of ``DENSE_CUTE_SM12X``) and
    ``sf`` the scale bytes of ``a_fmt``; a layout the format does not describe is rejected."""
    if a_fmt not in _DENSE_A_FMTS:
        raise ValueError(f"SM90 dense FP4 has no weight format {a_fmt!r}")
    block = 32 if a_fmt == AFmt.MXFP4_SWZ128X4 else 16
    if rows <= 0 or k <= 0 or k % block:
        raise ValueError(
            f"{entry}: N = {rows} and K = {k} must be positive, K % {block} == 0"
        )
    if not codes.is_contiguous():
        raise ValueError(f"{entry}: the packed weight must be contiguous")
    if codes.numel() * codes.element_size() != rows * k // 2:
        raise ValueError(
            f"{entry}: the packed weight holds {codes.numel() * codes.element_size()} bytes, "
            f"expected {rows * k // 2} for N = {rows}, K = {k}"
        )
    need = _sf_bytes_needed(a_fmt, rows, k)
    have = sf.numel() * sf.element_size()
    exact = a_fmt in (AFmt.NVFP4_LINEAR_SF, AFmt.DENSE_CUTE_SM12X)
    if (have != need) if exact else (have < need):
        raise ValueError(
            f"{entry}: the block scales hold {have} bytes, expected "
            f"{'' if exact else 'at least '}{need} for {a_fmt.name}, N = {rows}, K = {k}"
        )
    return Sm90WeightView(
        entry=entry,
        a_fmt=a_fmt,
        pair=Pair.NONE,
        codes=_flat_bytes(codes, rows * k // 2),
        sf=_flat_bytes(sf, need),
        num_experts=1,
        rows=rows,
        rows_phys=rows,
        k=k,
        k_phys=k,
        codes_expert_stride=0,
        sf_expert_stride=0,
    )


def _geometry(w: Sm90WeightView) -> List[int]:
    return [
        int(w.a_fmt),
        int(w.pair),
        w.rows,
        w.rows_phys,
        w.k,
        w.k_phys,
        w.half_rows,
        w.pair_g,
        w.codes_expert_stride,
        w.sf_expert_stride,
    ]


def _mxfp4_to_bf16(
    codes: torch.Tensor, sf: torch.Tensor, m: int, k: int
) -> torch.Tensor:
    """Exact BF16 values of MXFP4 activations (``[m, k / 2]`` codes, 128x4-swizzled E8M0
    scales). The SM90 prep kernel restores NVFP4 activations only (``XFmt`` has no MXFP4 yet),
    so the MXFP4 ones are restored here, a correctness path outside the GEMM."""
    from .gemm_bf16_fp4 import _unswizzle_sf_128x4

    if k % 32:
        raise ValueError(f"MXFP4 needs K % 32 == 0, got K = {k}")
    need = _sf_bytes_needed(AFmt.MXFP4_SWZ128X4, m, k)
    if sf.numel() * sf.element_size() < need:
        raise ValueError(
            f"the MXFP4 block scales hold {sf.numel() * sf.element_size()} bytes, expected {need}"
        )
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=codes.device,
    )
    c = codes.view(torch.uint8).reshape(m, k // 2)
    values = torch.stack((lut[(c & 0xF).long()], lut[(c >> 4).long()]), dim=-1).reshape(
        m, k
    )
    e8m0 = _unswizzle_sf_128x4(_flat_bytes(sf, need), m, k // 32)
    scale = torch.exp2(e8m0.float() - 127.0)
    scale = torch.where(e8m0 == 255, torch.full_like(scale, float("nan")), scale)
    return (
        (values.reshape(m, k // 32, 32) * scale.unsqueeze(-1))
        .reshape(m, k)
        .to(torch.bfloat16)
    )


def _alpha_args(
    alpha: Optional[torch.Tensor], m: int, device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """(scalar alpha, per-token scales or None): ``alpha`` is None, a scalar, or ``[m]``."""
    if alpha is None:
        return _ones(device), None
    if alpha.dtype != torch.float32 or alpha.device != device:
        raise ValueError(
            f"alpha must be a float32 tensor on {device}, got {alpha.dtype}"
        )
    if alpha.numel() == 1:
        return alpha.reshape(1), None
    if alpha.numel() != m:
        raise ValueError(
            f"alpha must be a scalar or one scale per row of a, got {alpha.numel()} for m = {m}"
        )
    return _ones(device), alpha.reshape(-1).contiguous()


def _run(
    a: torch.Tensor,
    a_sf: Optional[torch.Tensor],
    w: Sm90WeightView,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
    tactic: Tuple[int, int, int],
    a_layout: str,
    a_path: int,
) -> None:
    m = a.shape[0]
    block_mx = w.a_fmt == AFmt.MXFP4_SWZ128X4
    if a.dtype in (torch.bfloat16, torch.float16):
        if a_sf is not None:
            raise ValueError("a 16-bit activation takes no scales")
        f16 = a.dtype == torch.float16
        x_fmt = XFmt.F16 if f16 else XFmt.BF16
        x_sf = None
        k = a.shape[1]
    else:
        if a.dtype != torch.uint8 or a_sf is None:
            raise ValueError(
                "an FP4 activation is uint8 codes [M, K / 2] with block scales"
            )
        if not a.is_contiguous() or not a_sf.is_contiguous():
            raise ValueError("FP4 activation codes and block scales must be contiguous")
        k = a.shape[1] * 2
        f16 = False
        if block_mx:  # MXFP4 activations of an MXFP4 weight
            if a_layout != "128x4":
                raise ValueError("MXFP4 scales are 128x4-swizzled")
            a = _mxfp4_to_bf16(a, a_sf, m, k)
            a_sf, x_fmt, x_sf = None, XFmt.BF16, None
        else:
            if a_layout not in _ACT_LAYOUTS:
                raise ValueError(f"unknown activation scale layout {a_layout!r}")
            x_fmt = _ACT_LAYOUTS[a_layout]
            block, rows = 16, (_pad(m, 8) if a_layout == "8x4" else _pad(m, 128))
            if a_layout == "linear":
                rows = m
            need = rows * (
                _pad(-(-k // block), 4) if a_layout != "linear" else k // block
            )
            if a_sf.numel() * a_sf.element_size() < need:
                raise ValueError(
                    f"the activation block scales hold {a_sf.numel() * a_sf.element_size()} "
                    f"bytes, expected at least {need} for the {a_layout} layout"
                )
            x_sf = _flat_bytes(a_sf, need)
    if f16 and block_mx:
        raise ValueError(
            "MXFP4 weights need the BF16 wgmma operand; use a BF16 activation"
        )
    if k != w.k:
        raise ValueError(f"a has K = {k} but the weight has K = {w.k}")
    if out.shape != (m, w.rows) or out.stride(-1) != 1:
        raise ValueError(f"out must be [{m}, {w.rows}] with contiguous rows")
    alpha_s, tok = _alpha_args(alpha, m, a.device)
    module = get_fp4_dense_sm90_module(f16, w.a_fmt)
    cta_m, n, stages = tactic
    workspace = torch.empty(
        module.fp4_dense_sm90_workspace_bytes(m, k, int(x_fmt), n),
        dtype=torch.uint8,
        device=a.device,
    )
    module.fp4_dense_sm90_run(
        a,
        x_sf,
        int(x_fmt),
        w.codes,
        w.sf,
        _geometry(w),
        alpha_s,
        tok,
        workspace,
        out,
        cta_m,
        n,
        stages,
        a_path,
    )


def sm90_fp4_dense(
    a: torch.Tensor,
    a_sf: Optional[torch.Tensor],
    w: Sm90WeightView,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    tactic: Any = -1,
    out: Optional[torch.Tensor] = None,
    a_layout: str = "128x4",
    a_path: int = 0,
) -> torch.Tensor:
    """``out[M, N] = (a @ dequant(w)^T) * alpha`` on SM90 (wgmma, weights are the A operand).

    Parameters
    ----------
    a: torch.Tensor
        BF16/FP16 activations ``[M, K]`` (W4A16: the GEMM reads them directly), or FP4
        activations: uint8 codes ``[M, K / 2]`` (W4A4: restored to exact BF16 values by the
        prep kernel; NVFP4 for NVFP4 weights, MXFP4 for MXFP4 weights).
    a_sf: Optional[torch.Tensor]
        Block scales of FP4 activations in ``a_layout``, None for 16-bit ``a``.
    w: Sm90WeightView
        The weight (``sm90_dense_weight_view``); no gate/up pairing.
    alpha: Optional[torch.Tensor]
        None (1), an fp32 scalar, or an fp32 tensor with one scale per row of ``a``
        (applied in the epilogue, the per-token alpha of ``mm_fp4``).
    out_dtype: torch.dtype
        ``torch.bfloat16`` or ``torch.float16``.
    tactic: Union[int, Tuple[int, int, int]]
        ``(cta_m, n, stages)`` of the D7 table, or -1 for ``default_sm90_dense_tactic``.
    out: Optional[torch.Tensor]
        Preallocated ``[M, N]`` output of ``out_dtype``.
    a_layout: str
        ``"128x4"``, ``"8x4"`` or ``"linear"``: the NVFP4 activation scale layout.
    a_path: int
        0 or 1: the internal A path of the kernel (D1).

    Returns
    -------
    torch.Tensor
        ``[M, N]`` of ``out_dtype``.
    """
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {out_dtype}; use bfloat16 or float16"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2-D, got {tuple(a.shape)}")
    if w.pair != Pair.NONE or w.num_experts != 1:
        raise ValueError("SM90 dense FP4 takes one weight without gate/up pairing")
    if out is None:
        out = torch.empty((a.shape[0], w.rows), dtype=out_dtype, device=a.device)
    elif out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    if tactic is None or tactic == -1:
        tactic = default_sm90_dense_tactic(a.shape[0])
    _run(a, a_sf, w, alpha, out, tuple(tactic), a_layout, a_path)
    return out


class Sm90Fp4DenseRunner(TunableRunner):
    """``sm90_fp4_dense`` behind one dense entry point, as an autotuner candidate: ``unpack``
    maps the entry point's ``inputs`` to ``(a, a_sf, w, alpha, a_layout)`` and ``out``;
    tactic (cta_m, n, stages)."""

    def __init__(
        self,
        key: tuple,
        unpack: Callable[
            [List[Any]],
            Tuple[
                torch.Tensor,
                Optional[torch.Tensor],
                Sm90WeightView,
                Optional[torch.Tensor],
                str,
            ],
        ],
        out_index: int,
        out_dtype_index: int,
        a_path: int = 0,
    ):
        self.key = key
        self.unpack = unpack
        self.out_index, self.out_dtype_index = out_index, out_dtype_index
        self.a_path = a_path

    def __hash__(self) -> int:
        return hash((self.key, self.a_path))

    def get_cache_key_extras(self, inputs) -> tuple:
        return (self.key, inputs[0].dtype, inputs[self.out_dtype_index])

    def get_valid_tactics(self, inputs, profile: OptimizationProfile):
        a, _, w, _, _ = self.unpack(inputs)
        return list(_module_tactics(a.dtype == torch.float16, w.a_fmt, self.a_path))

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, a_sf, w, alpha, a_layout = self.unpack(inputs)
        if tactic is None or tactic == -1:
            tactic = default_sm90_dense_tactic(a.shape[0])
        out = inputs[self.out_index]
        _run(a, a_sf, w, alpha, out, tuple(tactic), a_layout, self.a_path)
        return out


def _tune(name: str, runner: Sm90Fp4DenseRunner, config, inputs, out: torch.Tensor):
    chosen, tactic = AutoTuner.get().choose_one(name, [runner], config, inputs)
    chosen(inputs=inputs, tactic=tactic)
    return out


# ---------------------------------------------------------------------------------------------
# mm_bf16_fp4 (W4A16)
# ---------------------------------------------------------------------------------------------


def _bf16_fp4_view(
    backend: str, b: torch.Tensor, b_descale: torch.Tensor
) -> Sm90WeightView:
    """The weight of an ``mm_bf16_fp4`` backend's prepared ``(b, b_descale)``."""
    if backend in ("cudnn", "blackwell-native"):
        a_fmt = AFmt.NVFP4_LINEAR_SF
    elif backend == "blackwell-tiled":
        a_fmt = AFmt.DENSE_CUTE_SM12X
    elif backend == "cute-dsl-native":
        a_fmt = AFmt.NVFP4_SWZ128X4
    elif backend == "cute-dsl":
        a_fmt = AFmt.NVFP4_SWZ128X4 if b.dtype == torch.uint8 else AFmt.DENSE_CUTE_SM12X
    else:
        raise ValueError(f"unknown mm_bf16_fp4 backend {backend!r}")
    if b.dtype == torch.int32:  # (K / 16, 2 N)
        k, n = int(b.shape[0]) * 16, int(b.shape[1]) // 2
    else:  # (N, K / 2)
        n, k = int(b.shape[0]), int(b.shape[1]) * 2
    return sm90_dense_weight_view(f"mm_bf16_fp4:{backend}", a_fmt, b, b_descale, n, k)


def sm90_mm_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    *,
    backend: str,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """``mm_bf16_fp4`` on SM90 (after the backend's requirement check): the GEMM of
    ``backend``'s prepared weight, with that backend's output contract."""
    if backend == "cute-dsl" and out_dtype != torch.bfloat16:
        raise NotImplementedError(
            f"cute-dsl currently requires a bfloat16 output; got {out_dtype}."
        )
    if not a.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    w = _bf16_fp4_view(backend, b, b_descale)
    if out is None:
        out = torch.empty((a.shape[0], w.rows), device=a.device, dtype=out_dtype)
    elif tuple(out.shape) != (a.shape[0], w.rows) or out.dtype != out_dtype:
        raise ValueError(
            f"out must have shape/dtype {(a.shape[0], w.rows)}/{out_dtype}, got "
            f"{tuple(out.shape)}/{out.dtype}"
        )
    from .gemm_bf16_fp4_cudnn import _BF16_FP4_TUNING_CONFIG

    def unpack(inputs):
        a_, b_, sf_, alpha_ = inputs[:4]
        return a_, None, _bf16_fp4_view(backend, b_, sf_), alpha_, "128x4"

    runner = Sm90Fp4DenseRunner(("mm_bf16_fp4", backend), unpack, 5, 4)
    inputs = [a, b, b_descale, alpha, out_dtype, out]
    return _tune("bf16_fp4_gemm", runner, _BF16_FP4_TUNING_CONFIG, inputs, out)


# ---------------------------------------------------------------------------------------------
# mm_fp4 (W4A4)
# ---------------------------------------------------------------------------------------------


def _mm_fp4_view(
    backend: str, b: torch.Tensor, b_descale: torch.Tensor, use_nvfp4: bool
) -> Sm90WeightView:
    """The weight of ``mm_fp4``'s ``b`` (``[K / 2, N]``, the column-major view of a contiguous
    ``[N, K / 2]`` weight) and ``b_descale`` (the transposed view of the swizzled scales)."""
    if backend == "trtllm":
        a_fmt = AFmt.NVFP4_TRTLLM_SHUFFLED
    else:
        a_fmt = AFmt.NVFP4_SWZ128X4 if use_nvfp4 else AFmt.MXFP4_SWZ128X4
    codes = b.t()
    if not codes.is_contiguous():
        raise ValueError(
            "SM90 FP4 GEMM requires b as the column-major [K / 2, N] view of a contiguous "
            "[N, K / 2] weight (pass b_fp4.T)"
        )
    sf = b_descale.t() if b_descale.dim() == 2 else b_descale
    if not sf.is_contiguous():
        raise ValueError(
            "SM90 FP4 GEMM requires b_descale as the transposed view of the contiguous "
            "swizzled weight scales (pass b_sf.T)"
        )
    return sm90_dense_weight_view(
        f"mm_fp4:{backend}",
        a_fmt,
        codes.view(torch.uint8),
        sf,
        int(b.shape[1]),
        int(b.shape[0]) * 2,
    )


def sm90_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: torch.Tensor,
    block_size: int,
    use_nvfp4: bool,
    workspace_buffer: torch.Tensor,
    *,
    backends: List[str],
    use_8x4_sf_layout: bool,
    tuning_config,
) -> torch.Tensor:
    """``mm_fp4`` on SM90, after the requirement check of every name in ``backends``. All
    names of one call compute the same product here (they differ by weight format only for
    ``trtllm``), so a call is one candidate, the first name's. The operands are the
    ``fp4_gemm`` inputs of ``mm_fp4`` (``tuning_config`` indexes them)."""
    backend = backends[0]
    a_layout = "8x4" if use_8x4_sf_layout else "128x4"

    def unpack(inputs):
        a_, b_, sfa_, sfb_, alpha_ = inputs[:5]
        return (
            a_.view(torch.uint8),
            sfa_,
            _mm_fp4_view(backend, b_, sfb_, use_nvfp4),
            alpha_,
            a_layout,
        )

    _mm_fp4_view(backend, b, b_descale, use_nvfp4)  # operand check before tuning
    runner = Sm90Fp4DenseRunner(
        ("mm_fp4", backend == "trtllm", use_nvfp4, a_layout), unpack, 6, 5
    )
    inputs = [
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_nvfp4,
        workspace_buffer,
    ]
    return _tune("fp4_gemm", runner, tuning_config, inputs, out)
