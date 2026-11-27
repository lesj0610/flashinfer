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

The prepared contiguous grouped FP8 GEMM programs on SM8x. The grouped GEMM runs on the
SM8x FP8 GEMM (per 128-element K step the groupwise FP32 scales, or the UE8M0 ones as its
block scales; the rows in tiles that each lie in one expert). The prepared grouped GEMM's
launch routes the expert runs on the device per call, its scratch from the caching
allocator. The fused gate_up + SwiGLU + FP8 group quantization stages run it into a BF16
intermediate, then FlashInfer's ``silu_and_mul`` and the group quantization of the
generated activation kernel; preparation makes their routing scratch and buffers and the
calls allocate nothing.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

# The SM8x GEMM's row tile: it divides 128, so with every internal expert boundary a
# multiple of 128 rows each tile lies in one expert.
_ROWS = 64
# The group quantization of the generated activation kernel.
_GROUP = 128
_EPS = 1e-10
_E4M3_MAX = 448.0


def _ue8m0_exponents(words: torch.Tensor, k_blocks: int) -> torch.Tensor:
    """The UE8M0 exponents packed four per int32 word along the last dim (``e`` of K block
    ``q`` the byte ``q % 4`` of ``words[..., q // 4]``, low byte first), for ``q < k_blocks``."""
    q = torch.arange(k_blocks, dtype=torch.int32, device=words.device)
    return ((words[..., q // 4] >> q % 4 * 8) & 255).to(torch.uint8)


class Sm8xContiguousGroupGemmEntry:
    """The generated program of the prepared contiguous grouped FP8 GEMM on SM8x, called
    with the program's bound arguments (named by its argument plan; the grid is not
    launched): ``out[r] = sum_q (a[r, q] . b[g, :, q]) sa[r, q] sb[g, :, q]`` over the
    128-element K blocks ``q`` for ``g = m_indices[r]``, on the SM8x FP8 GEMM routed by
    expert run. The FP32 scales run its groupwise scaling; the packed UE8M0 words (the
    weight's first row of every 128-row block) its block scales, the exponents applied as
    the sum ``ea + eb`` (``e = 255`` NaN), and a ``-1`` padding row is skipped (its output
    row unspecified). Each call routes on the device (no host sync) and takes its scratch
    from the caching allocator."""

    def __init__(self, arg_plan: Sequence[Sequence[str]]) -> None:
        from ..fused_moe.core import _w4a16_moe_sm80_module

        self.names = tuple(name for _, name in arg_plan)
        _w4a16_moe_sm80_module(
            None, False, None, False
        )  # loaded here, not under a capture

    def __call__(self, *arguments: Any) -> None:
        from .sm8x_fp8_dense import _mx_rows, _run_fp8

        bound = dict(zip(self.names, arguments, strict=True))
        a = bound["A"].view(torch.float8_e4m3fn)
        b = bound["B"].view(torch.float8_e4m3fn)
        m, k = (int(v) for v in a.shape)
        groups, n = (int(v) for v in b.shape[:2])
        device = a.device
        m_indices = bound["m_indices"]
        one = torch.ones(1, dtype=torch.float32, device=device)
        if "a_scale" in bound:
            # FP32 scales (the host forward-fills a -1 row with fill_padding)
            experts = torch.arange(groups + 1, dtype=torch.int32, device=device)
            m_indptr = torch.searchsorted(m_indices, experts, out_int32=True)
            _run_fp8(
                a[None], b.transpose(1, 2), bound["a_scale"], bound["b_scale"],
                bound["C"][None], _ROWS, False, gm=1, m_indptr=m_indptr, one=one,
            )  # fmt: skip
            return
        k_blocks = k // 128
        cols = -(-k_blocks // 4)
        if "SFA" in bound:  # words at base[row * rs + col * cs]
            sfa = bound["SFA"].as_strided(
                (m, cols), (bound["sfa_row_stride"], bound["sfa_col_stride"])
            )
            sfb = bound["SFB"].as_strided(
                (groups, n // 128, cols),
                (bound["sfb_group_stride"], bound["sfb_row_stride"], bound["sfb_col_stride"]),
            )  # fmt: skip
        else:  # the swap-AB surfaces [cols, M] and [G * sf_cols, N] (rows repeated)
            sfa = bound["SFA_packed"].view(torch.int32).T
            sfb = bound["SFB_packed"].view(torch.int32)
            sfb = sfb.as_strided(
                (groups, n // 128, cols), (bound["sf_cols"] * n, 128, n)
            )
        ea = _ue8m0_exponents(sfa, k_blocks)
        eb = _ue8m0_exponents(sfb, k_blocks).repeat_interleave(128, 1)
        _run_fp8(
            a[None], b.transpose(1, 2), _mx_rows(ea), _mx_rows(eb), bound["C_tma"][None],
            _ROWS, False, True, m_keys=m_indices, one=one,
        )  # fmt: skip


class Sm8xContiguousGroupGemm:
    """``out[r] = sum_k a[r, k] a_scale[r, k / 128] b[g, n, k] b_scale[g, n / 128, k /
    128]`` in BF16 for ``g = m_indices[r]``: ``a`` ``(M, K)`` FP8 with its ``m_indices``
    sorted and every internal expert boundary a multiple of 128 rows, ``b`` ``(G, N, K)``
    read in place. Each call routes the row tiles from ``m_indices`` on the device."""

    def __init__(self, m: int, k: int, a_dtype: torch.dtype, b, b_scale) -> None:
        from ..fused_moe.core import _w4a16_moe_sm80_module
        from .sm8x_fp4_dense import _pad
        from .sm8x_fp8_dense import _E4M3, _FMT, _I32_MAX, GW_E4M3_ALPHA, _weight

        groups, n = int(b.shape[0]), int(b.shape[1])
        device = b.device
        kp = _pad(k, 32)
        self.k, self.n, self.groups = k, n, groups
        self.fmt = (_FMT[a_dtype], _FMT[b.dtype])
        self.w = _weight(b.transpose(1, 2), k, n, False)
        self.b_scale = b_scale
        # E4M3 weights: their scales over 2^8 per call, alpha 2^8 (gw_e4m3_weight_scales)
        self.gsb = torch.empty_like(b_scale) if b.dtype == _E4M3 else None
        # the GEMM's activation offsets are 32-bit: chunks of whole 128-row blocks
        step = m
        if m * kp > _I32_MAX and m > 128:
            step = max(128, _I32_MAX // kp // 128 * 128)
        self.chunks = []
        for r0 in range(0, m, step):
            rows = min(step, m - r0)
            blocks = -(-rows // _ROWS)
            ints = (1 + 2 * blocks, blocks * _ROWS)
            meta, rowmap = torch.empty(
                sum(ints), dtype=torch.int32, device=device
            ).split(ints)
            self.chunks.append((r0, rows, blocks, meta, rowmap))
        self.act = torch.empty((min(step, m), kp), dtype=torch.bfloat16, device=device)
        self.alpha = torch.full(
            (1,), GW_E4M3_ALPHA if self.gsb is not None else 1.0,
            dtype=torch.float32, device=device,
        )  # fmt: skip
        self.module = _w4a16_moe_sm80_module(None, False, None, False)

    def __call__(self, a, a_scale, m_indices, out) -> None:
        from .sm8x_fp8_dense import gw_e4m3_weight_scales

        w, rs, ks, bs = self.w
        b_scale = self.b_scale
        if self.gsb is not None:
            b_scale = gw_e4m3_weight_scales(b_scale, out=self.gsb)
        x = a.view(torch.uint8)
        for r0, rows, blocks, meta, rowmap in self.chunks:
            act = self.act[:rows]
            # prep_f8 routes the tiles in order, every one to expert 0 ...
            self.module.prep_f8(
                x[r0 : r0 + rows].reshape(-1), act, meta, rowmap, 1, self.k, _ROWS,
                blocks, self.fmt[0],
            )  # fmt: skip
            # ... and each tile's expert is its first row's
            meta[1 : 1 + blocks].copy_(m_indices[r0 : r0 + rows : _ROWS])
            self.module.gemm_f8(
                w, self.alpha, None, act, meta, rowmap, out[r0 : r0 + rows], _ROWS,
                self.n, self.k, blocks, self.groups, rs, ks, bs, self.fmt[1], False, None,
                None, False, 0, a_scale[r0:], b_scale, 1, None,
            )  # fmt: skip


class Sm8xSiluQuant:
    """``out_q``, ``out_s`` of the generated activation kernel from the BF16 gate_up
    ``y`` ``(M, 2H)``: FlashInfer's ``silu_and_mul`` (the chain's activation), then per
    row and group of 128 columns ``scale = max(absmax, 1e-10) / 448`` (an FP32 division)
    and ``q = e4m3(clamp(act * (1 / scale), -448, 448))`` (round to nearest even). As in
    the kernel's ``fmaxf`` / ``fminf``, the absmax and the clamp skip a NaN: a NaN
    element leaves its group's scale to the finite ones and quantizes to 448."""

    def __init__(self, m: int, h: int, device: torch.device) -> None:
        from ..activation import get_act_and_mul_module

        get_act_and_mul_module("silu")  # loaded here, not under a capture
        self.groups = (m, h // _GROUP, _GROUP)
        self.act = torch.empty((m, h), dtype=torch.bfloat16, device=device)
        self.mag = torch.empty_like(self.act)
        self.amax = torch.empty(self.groups[:2], dtype=torch.bfloat16, device=device)
        self.inverse = torch.empty(self.groups[:2], dtype=torch.float32, device=device)
        self.scaled = torch.empty(self.groups, dtype=torch.float32, device=device)
        self.zero = torch.zeros((), dtype=torch.bfloat16, device=device)
        # a device divisor: a true division (a host scalar divides by its reciprocal)
        self.limit = torch.full((), _E4M3_MAX, dtype=torch.float32, device=device)
        self.neg_limit = torch.full((), -_E4M3_MAX, dtype=torch.float32, device=device)

    def __call__(self, y, out_q, out_s) -> None:
        from ..activation import silu_and_mul

        silu_and_mul(y, out=self.act)
        torch.abs(self.act, out=self.mag)
        torch.fmax(self.mag, self.zero, out=self.mag)  # a NaN magnitude counts as 0
        torch.amax(self.mag.view(self.groups), -1, out=self.amax)
        scale = out_s.view(self.groups[:2])
        scale.copy_(self.amax).clamp_(min=_EPS).div_(self.limit)
        torch.reciprocal(scale, out=self.inverse)
        torch.mul(self.act.view(self.groups), self.inverse[..., None], out=self.scaled)
        torch.fmin(self.scaled, self.limit, out=self.scaled)  # NaN -> 448
        torch.fmax(self.scaled, self.neg_limit, out=self.scaled)
        out_q.view(self.groups).copy_(self.scaled)


class Sm8xSiluQuantStage:
    """One generated stage of the fused grouped FP8 gate_up + SwiGLU + quantization
    program, called with the stage's bound arguments (named by its argument plan; the
    grid is not launched): the pair or mixed-schedule stage runs the grouped GEMM into the
    prepared BF16 intermediate, then the activation; the activation stage reads ``y``."""

    def __init__(
        self,
        arg_plan: Sequence[Sequence[str]],
        act: Sm8xSiluQuant,
        gemm: Optional[Sm8xContiguousGroupGemm] = None,
        y: Optional[torch.Tensor] = None,
    ) -> None:
        self.names = tuple(name for _, name in arg_plan)
        self.act, self.gemm, self.y = act, gemm, y

    def __call__(self, *arguments: Any) -> None:
        bound = dict(zip(self.names, arguments, strict=True))
        y = bound.get("y")
        if self.gemm is not None:
            y = self.y
            self.gemm(bound["A"], bound["a_scale"], bound["m_indices"], y)
        self.act(y, bound["out_q"], bound["out_s"])


def sm8x_tail_stage(*arguments: Any) -> None:
    """The pair + tail route's tail stage: the SM8x pair stage covers every row."""
