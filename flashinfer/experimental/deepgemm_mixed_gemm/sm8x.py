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
prepacked UE8M0 words of A (per 32 or per 128 K elements) and B (per 32) unpacked to its
128x4-swizzled E8M0 scales on every run, since their contents may change between runs.
"""

from __future__ import annotations

import torch

# FlashInfer's SM8x GEMM: the package's only FlashInfer import, taken on SM8x only.
from flashinfer.fused_moe.core import _w4a16_moe_sm80_module
from flashinfer.gemm.sm8x_fp4_dense import _pad
from flashinfer.gemm.sm8x_fp8_dense import _tuned_rows_fp4

_NAME = "fp8_fp4_gemm_sm8x"


def _swizzled(words: torch.Tensor, rows: int, groups: int, gran: int) -> torch.Tensor:
    """The 128x4-swizzled E8M0 scales, one per 32 K elements, of the first ``rows`` rows of
    the packed UE8M0 words ``[words, aligned rows]`` (``groups`` bytes per row, each for
    ``gran`` K elements)."""
    e = words.view(torch.int32).t().contiguous().view(torch.uint8)[:rows, :groups]
    if gran == 128:
        e = e.repeat_interleave(4, dim=1)
    blocks = int(e.shape[1])
    x = e.new_zeros(_pad(rows, 128), blocks)
    x[:rows] = e
    return x.view(-1, 4, 32, blocks // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


class Sm8xMixedGemm:
    """``plan.run()`` of a prepared mixed GEMM on SM8x: ``out = A[:m] @ B.T`` with the
    scales unpacked here; the GEMM is loaded at preparation."""

    def __init__(self, a, b, a_scales, b_scales, out, m, n, k, gran_k_a):
        device = a.device
        self.a = a[:m].view(torch.float8_e4m3fn)
        self.b = b.view(torch.uint8).view(1, n, k // 2)
        self.scales = (a_scales, b_scales)
        self.shape = (m, n, k, gran_k_a)
        self.starts = torch.zeros(1, dtype=torch.int64, device=device)
        self.ends = torch.full((1,), m, dtype=torch.int64, device=device)
        self.alpha = torch.ones(1, dtype=torch.float32, device=device)
        self.asf_off = torch.zeros(1, dtype=torch.int32, device=device)
        self.out = out
        _w4a16_moe_sm80_module(None, False, None, False)

    def __call__(self):
        m, n, k, gran = self.shape
        sfa = _swizzled(self.scales[0], m, k // gran, gran)
        sfb = _swizzled(self.scales[1], n, k // 32, 32)
        _tuned_rows_fp4(
            _NAME, None,
            [self.a, self.b, sfa, sfb, self.starts, self.ends, self.alpha,
             self.asf_off, self.out],
        )  # fmt: skip
