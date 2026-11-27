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
power of two saturated to E4M3, four exponents per int32 scale word).
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
    ``A[T, H, K]`` and ``out[T, H, N]`` are routed to ``B[h]``; the scales are packed and
    the GEMM and scratch output are prepared here, as the programs are."""

    def __init__(self, aq, asf, bq, bsf, out, output_scales, epilogue, alpha):
        # FlashInfer's SM8x GEMM: the package's only FlashInfer import, taken on SM8x only.
        from flashinfer.fused_moe.core import _w4a16_moe_sm80_module
        from flashinfer.gemm.sm8x_fp4_dense import _ones, _out_module
        from flashinfer.gemm.sm8x_fp8_dense import _mx_rows, _run_fp8

        tokens, heads, inner = (int(x) for x in aq.shape)
        width = int(bq.shape[1])
        rows, device = tokens * heads, aq.device
        self.a = aq.view(1, rows, inner)
        self.b = bq.transpose(1, 2)
        self.sfa = _mx_rows(_exponents(asf).reshape(rows, inner // 128))
        self.sfb = _mx_rows(_exponents(bsf).repeat_interleave(128, dim=1))
        self.keys = torch.arange(rows, dtype=torch.int32, device=device) % heads
        self.out, self.scales = out, output_scales
        self.epilogue, self.alpha = epilogue, alpha
        self.shape = (tokens, heads, width)
        self._run = _run_fp8
        acc = {"bf16": None, "alpha": torch.float32, "fp8": torch.bfloat16}[epilogue]
        self.acc = (
            out.view(1, rows, width)
            if acc is None
            else torch.empty((1, rows, width), dtype=acc, device=device)
        )
        _ones(device)
        _w4a16_moe_sm80_module(None, False, None, False)
        if acc == torch.float32:
            _out_module(acc)

    def __call__(self):
        self._run(
            self.a, self.b, self.sfa, self.sfb, self.acc, _ROWS, False, True,
            m_keys=self.keys,
        )  # fmt: skip
        if self.epilogue == "alpha":
            self.out.view_as(self.acc).copy_(self.acc.mul_(self.alpha))
        elif self.epilogue == "fp8":
            self._fp8()

    def _fp8(self):
        tokens, heads, width = self.shape
        v = self.acc.view(tokens, heads, width // 32, 32)
        bits = v.abs().amax(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        exponent = ((bits + 31) >> 7).clamp_(min=113) - 8
        inverse = ((254 - exponent) << 7).to(torch.int16).view(torch.bfloat16)
        quantized = (v * inverse[..., None]).float().clamp_(-448.0, 448.0)
        self.out.view_as(v).copy_(quantized.to(torch.float8_e4m3fn))
        words = exponent.to(torch.uint8).view(tokens, heads * width // 128, 4)
        self.scales.copy_(words.view(torch.int32)[..., 0])
