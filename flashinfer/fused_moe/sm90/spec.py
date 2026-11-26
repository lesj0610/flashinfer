#
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Host-side interface of the SM90 (wgmma) FP4 GEMM shared by the MoE and dense paths.

The enum values mirror ``csrc/fp4_sm90/params.cuh``. SM90 has no weight format of its
own: a :class:`Sm90WeightView` describes the bytes an SM100+ entry point already accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import torch


class AFmt(IntEnum):
    """Weight byte layout (``fp4_sm90::AFmt``)."""

    NVFP4_TRTLLM_SHUFFLED = 0
    NVFP4_SWZ128X4 = 1
    NVFP4_LINEAR_SF = 2
    MXFP4_TRTLLM_SHUFFLED = 3
    MXFP4_SWZ128X4 = 4
    DENSE_CUTE_SM12X = 5


class Pair(IntEnum):
    """Gate/up pairing of a gated GEMM1 weight (``fp4_sm90::Pair``)."""

    NONE = 0
    UP_GATE_HALVES = 1
    TRTLLM_W31 = 2
    GROUP_INTERLEAVE = 3


class XFmt(IntEnum):
    """Activation format accepted by the SM90 prep kernel (``fp4_sm90::XFmt``)."""

    BF16 = 0
    F16 = 1
    NVFP4_LINEAR = 2
    NVFP4_SWZ128X4 = 3
    NVFP4_SWZ8X4 = 4
    MXFP4_SWZ128X4 = 5


class Fq(IntEnum):
    """Quantization applied to a GEMM result in the epilogue (``fp4_sm90::Fq``)."""

    NONE = 0
    NVFP4 = 1
    MXFP8 = 2
    FP8 = 3


RECIPE_4OVER6 = 1 << 0
RECIPE_E4M3_MAX_256 = 1 << 1
RECIPE_ERR_MSE = 1 << 2
RECIPE_FAST_MATH = 1 << 3


def epi_code(fq: Fq, recipe: int, gated: bool, gemm2: bool) -> int:
    """Compile-time epilogue code (``fp4_sm90::epi_code``)."""
    return int(fq) | ((recipe & 0xF) << 2) | (int(gated) << 6) | (int(gemm2) << 7)


# Tactic of the SM90 GEMM: (cta_m, n, stages). See D7 of the SM90 plan.
SM90_TACTIC = Tuple[int, int, int]


@dataclass(frozen=True)
class Sm90WeightView:
    """FP4 weight as laid out by an SM100+ entry point, described for the SM90 kernel.

    ``codes``/``sf`` are the caller's tensors (no copy). Strides are in bytes.
    """

    entry: str
    a_fmt: AFmt
    pair: Pair
    codes: torch.Tensor
    sf: torch.Tensor
    num_experts: int
    rows: int
    rows_phys: int
    k: int
    k_phys: int
    codes_expert_stride: int
    sf_expert_stride: int
    half_rows: int = 0
    pair_g: int = 0
    global_scale: Optional[torch.Tensor] = None

    @property
    def gated(self) -> bool:
        return self.pair != Pair.NONE


@dataclass(frozen=True)
class Sm90EpilogueSpec:
    """Epilogue of one GEMM (``fp4_sm90::EpiArgs`` plus the compile-time code)."""

    act: int  # flashinfer.tllm_enums.ActivationType value
    fq: Fq = Fq.NONE
    recipe: int = 0
    gemm2: bool = False
    out_dtype: torch.dtype = torch.bfloat16
    alpha: Optional[torch.Tensor] = None
    tok_scale: Optional[torch.Tensor] = None
    act_params: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    fq_global: Optional[torch.Tensor] = None  # device FP32 scalar; None = 1

    @property
    def out_f16(self) -> bool:
        return self.out_dtype == torch.float16


@dataclass(frozen=True)
class Sm90Routing:
    """Routing result consumed by the SM90 prep kernel.

    ``topk_ids``/``topk_weights`` are ``[tokens, top_k]``. ``local_expert_offset`` and
    ``num_local_experts`` select the experts held by this rank (other ids are skipped).
    """

    topk_ids: torch.Tensor
    topk_weights: Optional[torch.Tensor]
    top_k: int
    num_experts: int
    local_expert_offset: int = 0
    num_local_experts: Optional[int] = None
