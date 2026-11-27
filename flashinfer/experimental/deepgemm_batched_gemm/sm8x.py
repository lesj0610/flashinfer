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

The batched FP8 projection on SM8x: the selected program's contract on the SM8x FP8
GEMM (the UE8M0 exponents of the scales, per 128-element K step, as the programs pack
them), then the program's epilogue on its accumulator: BF16 (round to nearest), the
runtime alpha (an FP32 product, then BF16), or the dynamic FP8 output (BF16 values, one
UE8M0 exponent per 32 of them from their BF16 maximum, the BF16 product with its inverse
power of two saturated to E4M3, four exponents per int32 scale word). Preparation
routes the rows, allocates the scratch and selects nothing more; ``run()`` submits
into the prepared buffers without allocating.
"""

from __future__ import annotations

import torch

# The SM8x GEMM's row tile.
_ROWS = 64


def _exponents(scales: torch.Tensor) -> torch.Tensor:
    """The UE8M0 exponents of the FP32 scales (``batched_gemm._pack_scales``)."""
    return ((scales.contiguous().view(torch.int32) >> 23) & 255).to(torch.uint8)


class Sm8xProjection:
    """``plan.run()`` of a prepared projection on SM8x. The rows ``t * H + h`` of
    ``A[T, H, K]`` and ``out[T, H, N]`` are routed to ``B[h]`` (``_run_fp8`` with these
    ``m_keys``, its routing and scratch made here once); the scales are packed and the
    GEMM loaded here, as the programs are."""

    def __init__(self, aq, asf, bq, bsf, out, output_scales, epilogue, alpha):
        # FlashInfer's SM8x GEMM: the package's only FlashInfer import, taken on SM8x only.
        from flashinfer.fused_moe.core import _w4a16_moe_sm80_module
        from flashinfer.gemm.sm8x_fp4_dense import _ones, _out_module, _pad
        from flashinfer.gemm.sm8x_fp8_dense import (
            _FMT,
            _I32_MAX,
            _aligned,
            _group_rows,
            _mx_rows,
            _weight,
        )

        tokens, heads, inner = (int(x) for x in aq.shape)
        width = int(bq.shape[1])
        rows, device = tokens * heads, aq.device
        kp, nka = _pad(inner, 32), inner // 128
        self.shape, self.inner = (tokens, heads, width), inner
        self.out, self.scales = out, output_scales
        self.epilogue, self.alpha = epilogue, alpha
        # operands read in place; a misaligned one staged (copied every run)
        self.stage = []
        x = aq.view(torch.uint8).view(rows, inner)
        if x.data_ptr() % 4:
            self.stage.append((x, torch.empty_like(x)))
            x = self.stage[-1][1]
        b = bq
        if b.data_ptr() % 16:
            self.stage.append((b, torch.empty_like(b)))
            b = self.stage[-1][1]
        self.w = _weight(b.transpose(1, 2), inner, width, False)
        self.sfa = _aligned(_mx_rows(_exponents(asf).reshape(rows, nka)), 4)
        self.sfb = _aligned(_mx_rows(_exponents(bsf).repeat_interleave(128, dim=1)), 16)
        self.sf_bstride = _pad(width, 128) * nka * 4 if heads > 1 else 0
        self.fmt = (_FMT[aq.dtype], _FMT[bq.dtype])
        acc = {"bf16": torch.bfloat16, "alpha": torch.float32, "fp8": torch.bfloat16}
        direct = epilogue == "bf16" and out.data_ptr() % 16 == 0
        self.acc = (
            out.view(rows, width)
            if direct
            else torch.empty((rows, width), dtype=acc[epilogue], device=device)
        )
        self.copy_out = epilogue == "bf16" and not direct
        # the 32-bit activation offsets: 128-row chunks of rows as _run_fp8 splits them
        step = rows
        if rows * kp > _I32_MAX and rows > 128:
            step = max(128, _I32_MAX // kp // 128 * 128)
        keys = torch.arange(rows, device=device) % heads
        self.chunks = []
        for r0 in range(0, rows, step):
            m = min(step, rows - r0)
            max_blocks = m // _ROWS + heads  # each head's last block partial
            ints = (1 + 2 * max_blocks, max_blocks * _ROWS)
            meta, rowmap = torch.empty(
                sum(ints), dtype=torch.int32, device=device
            ).split(ints)
            sk, perm = torch.sort(keys[r0 : r0 + m], stable=True)
            ip = torch.searchsorted(sk, torch.arange(heads + 1, device=device))
            _group_rows(ip[:-1], ip[1:], m, _ROWS, max_blocks, meta, rowmap, perm)
            # prep_f8 writes its own (unrouted) meta and rowmap: kept apart
            scratch = torch.empty(sum(ints), dtype=torch.int32, device=device).split(
                ints
            )
            self.chunks.append(
                (
                    x[r0 : r0 + m].reshape(-1),
                    m,
                    max_blocks,
                    meta,
                    rowmap,
                    scratch,
                    self.acc[r0 : r0 + m],
                    self.sfa[r0 * nka * 4 :],
                )
            )
        self.act = torch.empty(
            (min(step, rows), kp), dtype=torch.bfloat16, device=device
        )
        self.ones = _ones(device)
        self.prep = _w4a16_moe_sm80_module(None, False, None, False)
        self.gemm = (
            self.prep if self.acc.dtype == torch.bfloat16 else _out_module(acc["alpha"])
        )
        if epilogue == "fp8":
            groups = (tokens, heads, width // 32)
            self.tmp = torch.empty((*groups, 32), dtype=torch.bfloat16, device=device)
            self.amax = torch.empty(groups, dtype=torch.bfloat16, device=device)
            self.exponent = torch.empty(groups, dtype=torch.int32, device=device)
            self.inverse32 = torch.empty_like(self.exponent)
            self.inverse = torch.empty(groups, dtype=torch.int16, device=device)
            self.bytes = torch.empty(groups, dtype=torch.uint8, device=device)

    def __call__(self):
        _, heads, width = self.shape
        inner = self.inner
        w, rs, ks, bs = self.w
        for src, dst in self.stage:
            dst.copy_(src)
        for x, m, max_blocks, meta, rowmap, scratch, target, sfa in self.chunks:
            act = self.act[:m]
            self.prep.prep_f8(
                x, act, *scratch, 1, inner, _ROWS, max_blocks, self.fmt[0]
            )
            self.gemm.gemm_f8(
                w, self.ones, None, act, meta, rowmap, target, _ROWS, width, inner,
                max_blocks, heads, rs, ks, bs, self.fmt[1], False, self.sfb, sfa, False,
                self.sf_bstride, None, None, 1, None,
            )  # fmt: skip
        if self.copy_out:
            self.out.view_as(self.acc).copy_(self.acc)
        elif self.epilogue == "alpha":
            self.out.view_as(self.acc).copy_(self.acc.mul_(self.alpha))
        elif self.epilogue == "fp8":
            self._fp8()

    def _fp8(self):
        tokens, heads, width = self.shape
        v = self.acc.view(self.tmp.shape)
        torch.abs(v, out=self.tmp)
        torch.amax(self.tmp, -1, out=self.amax)
        e = self.exponent
        e.copy_(self.amax.view(torch.int16)).bitwise_and_(0xFFFF)
        e.add_(31).bitwise_right_shift_(7).clamp_(min=113).sub_(8)
        self.inverse32.copy_(e).neg_().add_(254).bitwise_left_shift_(7)
        self.inverse.copy_(self.inverse32)
        torch.mul(v, self.inverse.view(torch.bfloat16)[..., None], out=self.tmp)
        self.out.view_as(v).copy_(self.tmp.clamp_(-448.0, 448.0))
        self.bytes.copy_(e)
        words = self.bytes.view(tokens, heads * width // 128, 4).view(torch.int32)
        self.scales.copy_(words[..., 0])
