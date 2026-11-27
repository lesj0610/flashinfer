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


class Sm8xBatchDeepGemm:
    """The prepared masked batch FP8 GEMM on SM8x: ``out[g, r] = sum_q (a[g, r, q] . b[g, :,
    q]) sa[g, r, q] sb[g, :, q]`` over the 128-element K blocks ``q`` for the first
    ``masked_m[g]`` rows of each group ``g`` (the rest stay unwritten), on the SM8x FP8 GEMM
    routed in row tiles that each lie in one group. ``groupwise``: the route's FP32 epilogue,
    the FP32 scales in full; otherwise the UE8M0 exponents the route applies (FP32 ``bits >>
    23``, or the bytes of the packed MN-major words) as the GEMM's block scales (``e = 255``
    NaN). The operands are bound here with the buffers; each call repacks the scales and
    routes the row tiles on the device (no host sync) and allocates nothing."""

    def __init__(self, a, b, a_scale, b_scale, masked_m, out, groupwise: bool) -> None:
        from ..fused_moe.core import _w4a16_moe_sm80_module
        from .sm8x_fp8_dense import _E4M3, _FMT, _I32_MAX, GW_E4M3_ALPHA, _weight

        groups, m, k = (int(v) for v in a.shape)
        n, nka = int(b.shape[1]), k // 128
        device = a.device
        i32 = dict(dtype=torch.int32, device=device)
        self.module = _w4a16_moe_sm80_module(None, False, None, False)
        self.fmt = (_FMT[a.dtype], _FMT[b.dtype])
        self.k, self.n, self.groupwise = k, n, groupwise
        self.masked_m = masked_m
        # the GEMM's activation offsets are 32-bit: calls of whole groups, or of one group's
        # 128-row chunks
        if m * k <= _I32_MAX:
            per = _I32_MAX // (m * k)
            spans = [(g, min(g + per, groups), 0, m) for g in range(0, groups, per)]
        else:
            step = _I32_MAX // k // 128 * 128
            spans = [
                (g, g + 1, r, min(r + step, m)) for g in range(groups) for r in range(0, m, step)
            ]  # fmt: skip
        most = max((g1 - g0) * (r1 - r0) for g0, g1, r0, r1 in spans)
        self.act = torch.empty((most, k), dtype=torch.bfloat16, device=device)
        ints = (1 + 2 * most // _ROWS, most)
        self.meta, self.rowmap = torch.empty(sum(ints), **i32).split(ints)
        self.j = torch.arange(_ROWS, **i32)
        self.alpha = torch.full(
            (1,), GW_E4M3_ALPHA if groupwise and b.dtype == _E4M3 else 1.0,
            dtype=torch.float32, device=device,
        )  # fmt: skip
        x = a.view(torch.uint8)
        if groupwise:
            # E4M3 weights: their scales over 2^8 per call, alpha 2^8 (gw_e4m3_weight_scales)
            self.gsb = torch.empty_like(b_scale) if b.dtype == _E4M3 else b_scale
            self.b_scale = b_scale
        else:
            # A's block scales of the stacked rows and B's of each group (both 128-row
            # atoms, _mx_rows' order), written from the scales per call
            self.asf = torch.empty(
                groups * m * nka * 4, dtype=torch.uint8, device=device
            )
            self.sf = torch.empty(
                groups * n * nka * 4, dtype=torch.uint8, device=device
            )
            if (
                a_scale.dtype == torch.int32
            ):  # [G][K / 512][MN] words, K block 4c + j byte j
                cols = nka // 4

                def words(s, mn, dst):
                    src = s.as_strided((groups, cols, mn), (mn * cols, mn, 1))
                    src = src.view(torch.uint8).view(groups, cols, mn // 128, 4, 32, 4)
                    src = src.permute(0, 2, 1, 5, 4, 3)[..., None]
                    src = src.expand(-1, -1, -1, -1, -1, -1, 4)
                    dst = dst.view(groups, mn // 128, cols, 4, 32, 4, 4)
                    return src, dst

                self.shift: tuple = ()
                self.repack = (words(a_scale, m, self.asf), words(b_scale, n, self.sf))
            else:  # FP32, A's per row and B's per 128 rows: bits >> 23 in int32 (a CUDA
                # shift into a uint8 output shifts the low byte), then the bytes
                ea = torch.empty((groups * m // 128, 4, 32, nka), **i32)
                eb = torch.empty((groups, n // 128, nka), **i32)
                self.shift = (
                    (a_scale.view(torch.int32).view(ea.shape), ea),
                    (b_scale.view(torch.int32), eb),
                )
                self.repack = (
                    (
                        ea.permute(0, 3, 2, 1)[..., None].expand(-1, -1, -1, -1, 4),
                        self.asf.view(groups * m // 128, nka, 32, 4, 4),
                    ),
                    (
                        eb[..., None].expand(groups, n // 128, nka, 512),
                        self.sf.view(groups, n // 128, nka, 512),
                    ),
                )
        self.spans = []
        for g0, g1, r0, r1 in spans:
            cb, mr = g1 - g0, r1 - r0
            blocks = cb * mr // _ROWS
            nbg = torch.zeros((2, cb), **i32)  # row 1 zero: a 2-D scan, not CUB's
            scratch = (
                nbg, torch.empty_like(nbg), torch.empty(cb, **i32),
                torch.arange(blocks, **i32), torch.empty(blocks, **i32),
                torch.empty(blocks, **i32),
                torch.empty((blocks, _ROWS), dtype=torch.bool, device=device),
            )  # fmt: skip
            if groupwise:
                scales = (
                    None, None, 0, a_scale[g0:g1, r0:r1].reshape(cb * mr, nka),
                    self.gsb[g0:g1],
                )  # fmt: skip
            else:
                scales = (
                    self.sf[g0 * n * nka * 4 :], self.asf[(g0 * m + r0) * nka * 4 :],
                    n * nka * 4 if cb > 1 else 0, None, None,
                )  # fmt: skip
            self.spans.append(
                (
                    g0, g1, r0, mr, blocks, x[g0:g1, r0:r1].reshape(-1),
                    out[g0:g1, r0:r1].reshape(cb * mr, n),
                    _weight(b[g0:g1].transpose(1, 2), k, n, False), scales, scratch,
                )
            )  # fmt: skip

    def __call__(self) -> None:
        from .sm8x_fp8_dense import gw_e4m3_weight_scales

        if not self.groupwise:
            for src, dst in self.shift:
                torch.bitwise_right_shift(src, 23, out=dst)
            for src, dst in self.repack:
                dst.copy_(src)
        elif self.gsb is not self.b_scale:
            gw_e4m3_weight_scales(self.b_scale, out=self.gsb)
        for g0, g1, r0, mr, blocks, x, target, weight, scales, scratch in self.spans:
            w, rs, ks, bs = weight
            sf, asf, sf_bstride, gsa, gsb = scales
            nbg, end, size, blk, first, local, past = scratch
            cb = g1 - g0
            act = self.act[: cb * mr]
            meta = self.meta[: 1 + 2 * blocks]
            rowmap = self.rowmap[: blocks * _ROWS].view(blocks, _ROWS)
            # prep_f8 converts every row; the tiles of each group's first masked_m rows
            # are routed below
            self.module.prep_f8(
                x, act, meta, rowmap, cb, self.k, _ROWS, blocks, self.fmt[0]
            )
            expert, rows = meta[1 : 1 + blocks], meta[1 + blocks :]
            torch.sub(self.masked_m[g0:g1], r0, out=size).clamp_(0, mr)
            torch.add(size, _ROWS - 1, out=nbg[0]).div_(_ROWS, rounding_mode="floor")
            torch.cumsum(nbg, 1, dtype=torch.int32, out=end)
            meta[:1].copy_(end[0, -1:])
            torch.searchsorted(end[0], blk, right=True, out_int32=True, out=expert)
            expert.clamp_(max=cb - 1)
            torch.index_select(end[0], 0, expert, out=first)
            torch.index_select(nbg[0], 0, expert, out=local)
            first.sub_(local)  # the group's first tile
            torch.sub(blk, first, out=local)  # the tile within its group
            torch.index_select(size, 0, expert, out=rows)
            rows.sub_(local, alpha=_ROWS).clamp_(0, _ROWS)
            local.mul_(_ROWS).add_(expert, alpha=mr)  # the tile's first row
            torch.add(local[:, None], self.j, out=rowmap)
            torch.ge(self.j, rows[:, None], out=past)
            rowmap.masked_fill_(past, -1)
            self.module.gemm_f8(
                w, self.alpha, None, act, meta, rowmap.view(-1), target, _ROWS, self.n,
                self.k, blocks, cb, rs, ks, bs, self.fmt[1], False, sf, asf, False,
                sf_bstride, gsa, gsb, 1, None,
            )  # fmt: skip
