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

The mixed FP8 x FP4 GEMM on SM8x: the plan's operands on FlashInfer's SM8x FP4 GEMM with
the E4M3 activation (as ``group_gemm_mxfp8_mxfp4_nt_groupwise`` runs on SM8x), the
prepacked UE8M0 words of A (per 32 or per 128 K elements) and B (per 32) gathered to its
128x4-swizzled E8M0 scales on every run, since their contents may change between runs.
Preparation tunes the row tile, routes the rows and allocates the scratch; ``run()``
submits the gathers, the activation's conversion and the GEMM without allocating.
"""

from __future__ import annotations

import torch

# FlashInfer's SM8x GEMM: the package's only FlashInfer imports, taken on SM8x only.
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.core import _w4a16_moe_sm80_module
from flashinfer.gemm.sm8x_fp4_dense import _pad
from flashinfer.gemm.sm8x_fp8_dense import (
    Sm8xRowsFp4Runner,
    _group_rows,
    _rows_tuning_config,
)

_NAME = "fp8_fp4_gemm_sm8x"


def _swizzle(words: torch.Tensor, rows: int, groups: int, gran: int) -> torch.Tensor:
    """The byte offsets in the packed UE8M0 words ``[words, aligned rows]`` (``groups``
    bytes per row, each for ``gran`` K elements) of their 128x4-swizzled E8M0 scales, one
    per 32 K elements, of the first ``rows`` rows (the padding rows: the last row's)."""
    r = torch.arange(_pad(rows, 128), device=words.device).clamp_(max=rows - 1)
    g = torch.arange(groups, device=words.device)
    if gran == 128:
        g = g.repeat_interleave(4)
    blocks = int(g.shape[0])
    x = (g // 4 * int(words.shape[1]) + r[:, None]) * 4 + g % 4
    return x.view(-1, 4, 32, blocks // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


class Sm8xMixedGemm:
    """``plan.run()`` of a prepared mixed GEMM on SM8x: ``out = A[:m] @ B.T`` with the
    scales gathered into buffers made here; the row tile, routing and GEMM are prepared
    here too."""

    def __init__(self, a, b, a_scales, b_scales, out, m, n, k, gran_k_a):
        device = a.device
        self.a = a[:m].view(torch.float8_e4m3fn)
        self.stage = []
        w = b.view(torch.uint8).reshape(-1)
        if w.data_ptr() % 16:  # the GEMM's 16-byte weight loads: staged every run
            self.stage.append((w, torch.empty_like(w)))
            w = self.stage[-1][1]
        self.w, self.shape = w, (m, n, k)
        self.gathers = []
        for words, rows, gran in ((a_scales, m, gran_k_a), (b_scales, n, 32)):
            index = _swizzle(words, rows, k // gran, gran)
            self.gathers.append(
                (
                    words.view(torch.int32).view(torch.uint8).reshape(-1),
                    index,
                    torch.empty(index.shape, dtype=torch.uint8, device=device),
                )
            )
        sfa, sfb = (g[2] for g in self.gathers)
        starts = torch.zeros(1, dtype=torch.int64, device=device)
        ends = torch.full((1,), m, dtype=torch.int64, device=device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=device)
        self.asf_off = torch.zeros(1, dtype=torch.int32, device=device)
        self.out = out
        direct = n % 8 == 0 and out.data_ptr() % 16 == 0
        self.target = (
            out
            if direct
            else torch.empty((m, _pad(n, 8)), dtype=out.dtype, device=device)
        )
        self.module = _w4a16_moe_sm80_module(None, False, None, False)
        for src, index, dst in self.gathers:
            torch.index_select(src, 0, index, out=dst)
        inputs = [
            self.a, b.view(torch.uint8).view(1, n, k // 2), sfa, sfb, starts, ends,
            self.alpha, self.asf_off, out,
        ]  # fmt: skip
        _, tactic = AutoTuner.get().choose_one(
            _NAME, [Sm8xRowsFp4Runner(None)], _rows_tuning_config(32), inputs
        )
        self.r = 64 if tactic in (None, -1) else tactic
        self.max_blocks = m // self.r + 1
        ints = (1 + 2 * self.max_blocks, self.max_blocks * self.r)
        self.meta, self.rowmap = torch.empty(
            sum(ints), dtype=torch.int32, device=device
        ).split(ints)
        _group_rows(starts, ends, m, self.r, self.max_blocks, self.meta, self.rowmap)
        # the activation in its blocks' row order (here the rows' own), the rest zero
        self.act = torch.zeros(
            (self.max_blocks * self.r, k), dtype=torch.bfloat16, device=device
        )

    def __call__(self):
        m, n, k = self.shape
        for src, dst in self.stage:
            dst.copy_(src)
        for src, index, dst in self.gathers:
            torch.index_select(src, 0, index, out=dst)
        # E4M3 restored to BF16 times 2^119, as the GEMM's FP8 activation takes it
        self.act[:m].copy_(self.a).mul_(2.0**119)
        (sfa, sfb) = (g[2] for g in self.gathers)
        self.module.gemm_fp4(
            self.w, sfb, self.alpha, self.act, self.meta, self.rowmap, self.target,
            self.r, n, k, self.max_blocks, False, sfa, False, self.asf_off, 0,
        )  # fmt: skip
        if self.target is not self.out:
            self.out.copy_(self.target[:, :n])
