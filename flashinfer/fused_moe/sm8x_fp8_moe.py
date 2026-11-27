# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The CUTLASS, cuTile and TRT-LLM FP8-family fused-MoE runners' semantics over the SM8x
grouped GEMMs, on the architectures where the CUTLASS fused-MoE module has no kernel for
the quantization pair, cuTile cannot compile FP8 or the TRT-LLM fused-MoE kernels do not
exist.

``_CutlassRunnerBase``, ``_CuTileFp8Runner``, ``TrtllmFp8BlockRunner`` and
``TrtllmFp8PerTensorRunner`` delegate to these: the cuDNN grouped-GEMM pipeline
(``moe_sort`` -> ``moe_permute`` -> GEMM1 -> activation -> GEMM2 -> ``moe_unpermute``) with
the ``sm8x_fp8_dense`` grouped GEMMs and the delegating runner's activation set and scale
conventions.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.weak import WeakIdKeyDictionary

from ..gemm.sm8x_fp4_dense import _unswizzle
from ..utils import get_shuffle_matrix_a_row_indices
from ..gemm.sm8x_fp8_dense import (
    _run_group_groupwise,
    sm8x_grouped_mm_fp4,
    sm8x_grouped_mm_fp8,
    sm8x_grouped_mm_mxfp8,
)
from .api import ActivationConfig, ActivationType, MoEConfig, SwiGLUStep
from .runners import (
    _check_moe_sort_limits,
    _CudnnGroupedGemmBlockScaleRunnerBase,
    CudnnGroupedGemmBf16Runner,
    _CudnnGroupedGemmRunnerBase,
    _GroupedGemmLayout,
    _GroupedGemmWorkspace,
    _scatter_rows,
)

_Param = Union[torch.Tensor, float]


def _per_row(
    params: Dict[str, Optional[torch.Tensor]],
    name: str,
    row_expert: torch.Tensor,
    default: float,
) -> _Param:
    """``params[name]`` (``[num_experts]`` float32) of every row's expert, or ``default``."""
    value = params.get(name)
    if value is None:
        return default
    return value.index_select(0, row_expert)[:, None]


def cutlass_activation(
    x: torch.Tensor,
    activation_type: ActivationType,
    params: Dict[str, Optional[torch.Tensor]],
    row_expert: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """The CUTLASS fused-MoE activation of float32 GEMM1 rows ``x``: ``[rows, I]``, or
    ``[rows, 2I]`` as ``[up | gate]`` for the gated ones (``up, gate, up, gate, ...``
    columns if ``interleaved``). ``params`` are the runner's optional per-expert
    tensors (``_cutlass_activation_params``) and ``row_expert`` every row's expert; an
    absent tensor takes the kernel's default."""
    t = activation_type
    if t == ActivationType.Identity:
        return x
    if t == ActivationType.Gelu:
        return F.gelu(x)
    if t == ActivationType.Relu:
        return F.relu(x)
    if t == ActivationType.Silu:
        return F.silu(x)
    if t == ActivationType.Relu2:
        return F.relu(x).square()
    if interleaved:
        up, gate = x[:, 0::2], x[:, 1::2]
    else:
        inter = x.shape[1] // 2
        up, gate = x[:, :inter], x[:, inter:]
    if t == ActivationType.Geglu:
        return F.gelu(gate) * up
    if t == ActivationType.GegluTanh:
        return F.gelu(gate, approximate="tanh") * up
    if t == ActivationType.Swiglu:
        if all(
            params.get(name) is None
            for name in ("swiglu_alpha", "swiglu_beta", "swiglu_limit")
        ):
            return F.silu(gate) * up
        # Any of the three present selects the SwigluBias adaptor.
        alpha = _per_row(params, "swiglu_alpha", row_expert, 1.0)
        beta = _per_row(params, "swiglu_beta", row_expert, 0.0)
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + beta)
    if t == ActivationType.SwigluStep:
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        return F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
    if t == ActivationType.Situ:
        beta = _per_row(params, "situ_beta", row_expert, 4.0)
        linear_beta = _per_row(params, "situ_linear_beta", row_expert, 25.0)
        act = torch.tanh(gate / beta) * torch.sigmoid(gate) * beta
        return act * (torch.tanh(up / linear_beta) * linear_beta)
    if t == ActivationType.PowLU:
        # ``m`` rides in the swiglu_alpha slot, the clamp in swiglu_limit.
        m = _per_row(params, "swiglu_alpha", row_expert, 2.5)
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        pos = gate.clamp(min=0)
        act = torch.where(gate > 0, pos.pow(m / (pos.sqrt() + 1)), gate)
        act = act * torch.sigmoid(gate)
        return act.clamp(max=limit) * up.clamp(-limit, limit)
    raise NotImplementedError(f"CUTLASS activation {t!r} has no SM8x implementation.")


class _Sm8xMoEBodyBase(_CudnnGroupedGemmRunnerBase):
    """A delegating runner's pipeline over the SM8x grouped GEMMs. Built and driven by
    that runner, which checked the configuration and packed the inputs; only the
    pipeline's own limits are checked here."""

    _intermediate_alignment: ClassVar[int]

    def _check_support(self) -> None:
        if self._device_arch not in self._supported_archs:
            raise RuntimeError(
                f"{type(self).__name__} does not support SM{self._device_arch}; "
                f"supported architectures are {self._supported_archs}."
            )
        _check_moe_sort_limits(type(self).__name__, self.config)
        intermediate_size = self.config.experts.intermediate_size
        if intermediate_size % self._intermediate_alignment:
            raise NotImplementedError(
                f"{type(self).__name__} requires intermediate_size divisible by "
                f"{self._intermediate_alignment}, got {intermediate_size}."
            )

    def _build(self) -> None:
        from .cute_dsl.moe_utils import _get_moe_utils_module

        _get_moe_utils_module()


class _Sm8xCutlassMoEBase(_Sm8xMoEBodyBase):
    """CUTLASS semantics on the cuDNN pipeline: GEMM1's per-expert alpha and the
    activation in float32, then the GEMM2 operand; ``run`` maps the CUTLASS runner's
    packed inputs onto this pipeline's."""

    # Packed-input indices of GEMM1's and GEMM2's [E] float32 per-expert alphas, if
    # any. CUTLASS scales the float32 accumulator in the GEMM epilogue, before the
    # BF16 store: those GEMMs write float32 here and ``_epilogue`` rounds.
    _gemm1_alpha_input: ClassVar[Optional[int]] = None
    _gemm2_alpha_input: ClassVar[Optional[int]] = None
    # GEMM1's gated columns interleave up and gate (the TRTLLM weight shuffle).
    _interleaved_gemm1: ClassVar[bool] = False

    def __init__(self, config: MoEConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._activation_params: Dict[str, Optional[torch.Tensor]] = {}
        # CUTLASS weights every (token, slot) assignment, a repeated expert included.
        self._repeated_experts = True

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """This pipeline's packed inputs from the CUTLASS runner's (output first)."""
        raise NotImplementedError

    def run(
        self,
        inputs: List[torch.Tensor],
        activation_params: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Run the CUTLASS runner's packed ``inputs`` into their output ``inputs[0]``."""
        out, hidden_states = inputs[0], inputs[1]
        if hidden_states.shape[0] == 0:
            return out
        self._validate_hidden_states(hidden_states)
        self._activation_params = activation_params
        self.forward(self._body_inputs(inputs))
        return out

    def _activate(
        self,
        gemm1_out: torch.Tensor,
        layout: _GroupedGemmLayout,
        workspace: _GroupedGemmWorkspace,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        row_expert = layout.row_expert(gemm1_out.shape[0])
        y = cutlass_activation(
            gemm1_out.float(),
            self.config.activation.type,
            self._activation_params,
            row_expert,
            self._interleaved_gemm1,
        )
        return self._gemm2_operand(y, workspace, inputs)

    @staticmethod
    def _epilogue(
        acc: torch.Tensor,
        alpha_input: Optional[int],
        layout: _GroupedGemmLayout,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """CUTLASS's epilogue ``bf16(alpha[e] * acc)`` on a float32 GEMM output."""
        if alpha_input is None:
            return acc
        row_expert = layout.row_expert(acc.shape[0])
        alpha = inputs[alpha_input].index_select(0, row_expert)[:, None]
        return acc.mul_(alpha).to(torch.bfloat16)

    def _gemm2_operand(
        self,
        y: torch.Tensor,
        workspace: _GroupedGemmWorkspace,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """GEMM2's operand from the float32 activation ``y``."""
        return workspace.intermediate.copy_(y)

    def _permute(
        self, inputs: List[torch.Tensor]
    ) -> tuple[_GroupedGemmLayout, _GroupedGemmWorkspace]:
        layout, workspace = super()._permute(inputs)
        if self._repeated_experts:
            layout = _route_repeated_experts(layout, inputs[1])
        return layout, workspace


class Sm8xCutlassFp8PerTensorMoE(_Sm8xCutlassMoEBase):
    """``CutlassFp8PerTensorRunner`` on SM80 / SM86 / SM87. Inputs:
    ``[hidden, ids, weights, w1, w2, fc1_dequant, fc2_act_quant, fc2_dequant, out]``."""

    backend_key = "cutlass_fp8_per_tensor"
    _supported_archs = (80, 86, 87)
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn
    _expected_num_inputs = 8
    _gemm1_alpha_input = 5
    _fc2_act_quant_input = 6
    _gemm2_alpha_input = 7
    _intermediate_dtype = torch.float8_e4m3fn
    _intermediate_alignment = 16  # GEMM2's K

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [*inputs[1:9], inputs[0]]

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        acc = sm8x_grouped_mm_fp8(
            workspace.permuted_hidden_states,
            inputs[3],
            layout.m_indptr,
            None,
            torch.float32,
            None,
        )
        return self._epilogue(acc, self._gemm1_alpha_input, layout, inputs)

    def _gemm2_operand(self, y, workspace, inputs) -> torch.Tensor:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        return (
            y.mul_(inputs[self._fc2_act_quant_input])
            .clamp_(-fp8_max, fp8_max)
            .to(torch.float8_e4m3fn)
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        acc = sm8x_grouped_mm_fp8(
            intermediate, inputs[4], layout.m_indptr, None, torch.float32, None
        )
        return self._epilogue(acc, self._gemm2_alpha_input, layout, inputs)


class Sm8xCutlassBf16MoE(_Sm8xCutlassMoEBase, CudnnGroupedGemmBf16Runner):
    """``CutlassBf16Runner`` on SM80 / SM86 / SM87, which the CUTLASS module has no
    kernel for: the ``grouped_mm_bf16`` GEMMs with the CUTLASS activation between
    them. Inputs: ``[hidden, ids, weights, w1, w2, out]``."""

    backend_key = "cutlass_bf16"
    _supported_archs = (80, 86, 87)
    _intermediate_alignment = 8  # GEMM2's K in 16-byte BF16 rows

    def _check_support(self) -> None:
        super()._check_support()
        from ..grouped_mm.cudnn import _CUDNN_MOE_MIN_VERSION, _check_cudnn_version

        _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, f"{self.backend_key} MoE")

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [*inputs[1:6], inputs[0]]


class _Sm8xCutlassMxfp8MoEBase(
    _Sm8xCutlassMoEBase, _CudnnGroupedGemmBlockScaleRunnerBase
):
    """MXFP8 activations: E4M3 rows with UE8M0 32-element block scales, the
    intermediate MXFP8-quantized for GEMM2. Inputs: ``[hidden, ids, weights, w1, w2,
    w1_scale, w2_scale, fc1_input_scale, fc2_input_scale, hidden_states_scale, out]``
    with token-major ``hidden_states_scale``."""

    _supported_archs = (80, 86, 87, 89)
    _x_dtype = torch.float8_e4m3fn
    _expected_num_inputs = 10
    _block_size = 32
    _block_scale_dtype = torch.uint8
    _intermediate_alignment = 128  # whole 128x4 scale tiles

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        act_sf = inputs[10]
        if act_sf.ndim == 1:  # QuantConfig(swizzled_scale_factors=True)
            num_tokens, hidden_size = inputs[1].shape
            act_sf = _unswizzle(act_sf, num_tokens, hidden_size // 32)
        return [*inputs[1:10], act_sf.view(torch.uint8), inputs[0]]


class Sm8xCutlassMxfp8MoE(_Sm8xCutlassMxfp8MoEBase):
    """``CutlassMxfp8Runner`` on SM8x: E4M3 weights with int32-packed per-expert
    128x4-swizzled UE8M0 scales; ``fc1_input_scale`` and ``fc2_input_scale`` are the
    two GEMMs' per-expert alphas."""

    backend_key = "cutlass_mxfp8"
    _weight_dtype = torch.float8_e4m3fn
    _gemm1_alpha_input = 7
    _gemm2_alpha_input = 8

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        acc = sm8x_grouped_mm_mxfp8(
            workspace.permuted_hidden_states,
            inputs[3],
            workspace.permuted_hidden_states_scale,
            inputs[5].view(torch.uint8),
            layout.m_indptr,
            torch.float32,
            None,
        )
        return self._epilogue(acc, self._gemm1_alpha_input, layout, inputs)

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _quantize_mxfp8_rows

        quantized, scale = _quantize_mxfp8_rows(intermediate)
        acc = sm8x_grouped_mm_mxfp8(
            quantized,
            inputs[4],
            scale,
            inputs[6].view(torch.uint8),
            layout.m_indptr,
            torch.float32,
            None,
        )
        return self._epilogue(acc, self._gemm2_alpha_input, layout, inputs)


class Sm8xCutlassMxfp8Mxfp4MoE(_Sm8xCutlassMxfp8MoEBase):
    """``CutlassMxfp8Mxfp4Runner`` on SM8x: packed E2M1 weights with per-expert
    128x4-swizzled UE8M0 scales. CUTLASS reads neither input scale."""

    backend_key = "cutlass_mxfp8_mxfp4"
    _weight_dtype = torch.int64

    def _alpha(self, index: Optional[int], inputs: List[torch.Tensor]):
        return None if index is None else inputs[index]

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_fp4(
            workspace.permuted_hidden_states,
            inputs[3].view(torch.uint8),
            workspace.permuted_hidden_states_scale,
            inputs[5],
            layout.m_indptr,
            self._alpha(self._gemm1_alpha_input, inputs),
            torch.bfloat16,
            workspace.gemm1_out,
            32,
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _quantize_mxfp8_rows

        quantized, scale = _quantize_mxfp8_rows(intermediate)
        return sm8x_grouped_mm_fp4(
            quantized,
            inputs[4].view(torch.uint8),
            scale,
            inputs[6],
            layout.m_indptr,
            self._alpha(self._gemm2_alpha_input, inputs),
            torch.bfloat16,
            None,
            32,
        )


class Sm8xFrostMxfp8Mxfp4MoE(Sm8xCutlassMxfp8Mxfp4MoE):
    """``CudnnFrostMxfp8Mxfp4MoeRunner`` on SM8x: the CUTLASS pipeline's packed
    inputs, with Frost's per-expert ``fc1_input_scale`` and ``fc2_input_scale``
    scaling each GEMM's float32 output before its BF16 store."""

    _gemm1_alpha_input = 7
    _gemm2_alpha_input = 8


class Sm8xCutlassNvfp4MoE(_Sm8xCutlassMoEBase, _CudnnGroupedGemmBlockScaleRunnerBase):
    """``CutlassNvfp4Runner`` on SM8x: packed E2M1 activations with linear E4M3
    16-element block scales, packed E2M1 weights with per-expert 128x4-swizzled E4M3
    scales. ``fc1_dequant`` and ``fc2_dequant`` are the two GEMMs' per-expert alphas;
    the BF16 activation output is NVFP4-quantized with ``fc2_act_global_scale`` (one
    value, or one per expert). Inputs: ``[hidden, ids, weights, w1, w2, w1_scale,
    w2_scale, fc1_dequant, fc2_dequant, fc2_act_global_scale, hidden_states_scale,
    out]``."""

    backend_key = "cutlass_nvfp4"
    _supported_archs = (80, 86, 87, 89)
    _x_dtype = torch.uint8
    _weight_dtype = torch.int64
    _k_pack = 2
    _expected_num_inputs = 11
    _gemm1_alpha_input = 7
    _gemm2_alpha_input = 8
    _fc2_act_global_input = 9
    _block_size = 16
    _block_scale_dtype = torch.float8_e4m3fn
    # GEMM2's K in whole 16-byte E2M1 rows: the CUTLASS grouped GEMMs' 128-bit
    # operand alignment.
    _intermediate_alignment = 32

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # CUTLASS: [out, hidden, ids, weights, w1, w2, fc1_act_global, w1_scale,
        # fc1_dequant, fc2_act_global, w2_scale, fc2_dequant, hidden_states_scale].
        # GEMM1 reads the pre-quantized activations, so fc1_act_global is unused.
        return [
            *inputs[1:6],
            inputs[7],
            inputs[10],
            inputs[8],
            inputs[11],
            inputs[9],
            inputs[12].view(torch.float8_e4m3fn),
            inputs[0],
        ]

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> tuple[int, int]:
        # CUTLASS NVFP4 rows are whole 4-column scale tiles (hidden_size % 64), so
        # the swizzled permuted scales keep the row-major size.
        num_tokens, hidden_size = _CudnnGroupedGemmRunnerBase._validate_hidden_states(
            self, hidden_states
        )
        if hidden_size % 64:
            raise ValueError(
                f"{type(self).__name__} requires hidden_size divisible by 64, "
                f"got {hidden_size}."
            )
        return num_tokens, hidden_size

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_fp4(
            workspace.permuted_hidden_states,
            inputs[3].view(torch.uint8),
            workspace.permuted_hidden_states_scale,
            inputs[5],
            layout.m_indptr,
            inputs[self._gemm1_alpha_input],
            torch.bfloat16,
            workspace.gemm1_out,
            16,
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from ..quantization.fp4_quantization import fp4_quantize

        global_scale = inputs[self._fc2_act_global_input]
        if global_scale.ndim == 0:
            global_scale = global_scale.reshape(1)
        else:  # one per local expert
            row_expert = layout.row_expert(intermediate.shape[0])
            global_scale = global_scale.index_select(0, row_expert)
        quantized, scale = fp4_quantize(
            intermediate,
            global_scale,
            16,
            False,
            True,
            nvfp4_4over6=self.config.quant.nvfp4_4over6,
        )
        return sm8x_grouped_mm_fp4(
            quantized.view(torch.uint8),
            inputs[4].view(torch.uint8),
            scale,
            inputs[6],
            layout.m_indptr,
            inputs[self._gemm2_alpha_input],
            torch.bfloat16,
            None,
            16,
        )


def cutile_fp8_quantize(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_scaled: bool,
    valid_rows: Optional[torch.Tensor] = None,
    scale_ready: bool = False,
) -> None:
    """``cutile.fp8.quantize`` on SM8x: ``x`` into E4M3 ``q`` with UE8M0 32-element
    ``scale`` rows (``block_scaled``) or one float32 ``scale``, ``max(amax / 448, 2^-126)``
    over the first ``valid_rows`` rows (a one-element device tensor; None: all) unless
    ``scale_ready``."""
    values = x.float()
    if block_scaled:
        groups = values.unflatten(-1, (-1, 32))
        amax = groups.abs().amax(-1, keepdim=True)
        exponent = torch.ceil(torch.log2((amax / 448).clamp_min(2.0**-127)))
        exponent = exponent.clamp_max(127)
        scale.view(torch.uint8).copy_((exponent + 127).squeeze(-1))
        values = (groups / torch.exp2(exponent)).flatten(-2)
    else:
        if not scale_ready:
            amax = values.abs().amax(-1)
            if valid_rows is not None:
                rows = torch.arange(amax.shape[0], device=amax.device)
                amax = torch.where(rows < valid_rows.reshape(1), amax, 0.0)
            scale.copy_((amax.amax() / 448).clamp_min(2.0**-126).reshape(1))
        values = values / scale
    q.copy_(values.clamp(-448, 448))


def cutile_activation(
    x: torch.Tensor, activation: ActivationConfig, out: torch.Tensor
) -> torch.Tensor:
    """cuTile's activation of BF16 GEMM1 rows ``x`` (``[rows, I]``, or ``[rows, 2I]`` as
    ``[gate | up]`` for the gated ones) in float32, into BF16 ``out``."""
    from .cutile.activation import _activation_kernel_args

    kind, p1, p2, p3 = _activation_kernel_args(activation)
    t = ActivationType(kind)
    v = x.float()
    if not activation.is_gated:
        if t == ActivationType.Gelu:
            v = F.gelu(v)
        elif t == ActivationType.Relu:
            v = F.relu(v)
        elif t == ActivationType.Silu:
            v = F.silu(v)
        elif t == ActivationType.Relu2:
            v = F.relu(v).square()
        return out.copy_(v)
    inter = v.shape[1] // 2
    gate, up = v[:, :inter], v[:, inter:]
    if t in (ActivationType.Swiglu, ActivationType.Situ) and p3 > 0:
        gate, up = gate.clamp(max=p3), up.clamp(-p3, p3)
    if t == ActivationType.Swiglu:
        y = gate * torch.sigmoid(p1 * gate) * (up + p2)
    elif t == ActivationType.SwigluStep:
        y = F.silu(gate).clamp(max=p1) * up.clamp(-p1, p1)
    elif t == ActivationType.Geglu:
        y = F.gelu(gate) * up
    elif t == ActivationType.GegluTanh:
        y = F.gelu(gate, approximate="tanh") * up
    elif t == ActivationType.Situ:
        gate = p1 * torch.tanh(gate / p1) * torch.sigmoid(gate)
        if p2 > 0:
            up = p2 * torch.tanh(up / p2)
        y = gate * up
    else:
        raise NotImplementedError(
            f"cuTile activation {t!r} has no SM8x implementation."
        )
    return out.copy_(y)


# The 128x4-swizzled copy of every prepared [E, N, K / 32] UE8M0 weight-scale tensor
# the SM8x GEMMs read, keyed by the tensor and stamped with its address and version.
_SWIZZLED_WEIGHT_SCALES: WeakIdKeyDictionary = WeakIdKeyDictionary()


def _swizzled_weight_scales(scale: torch.Tensor) -> torch.Tensor:
    from ..quantization.fp4_quantization import block_scale_interleave

    # An inference tensor has no version counter, and a graph replay never re-runs
    # this check: both read scale itself each call, as the native kernels do.
    if scale.is_inference() or torch.cuda.is_current_stream_capturing():
        return block_scale_interleave(scale.view(torch.uint8))
    stamp = (scale.data_ptr(), scale._version)
    cached = _SWIZZLED_WEIGHT_SCALES.get(scale)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    swizzled = block_scale_interleave(scale.view(torch.uint8))
    _SWIZZLED_WEIGHT_SCALES[scale] = (stamp, swizzled)
    return swizzled


def _route_repeated_experts(
    layout: _GroupedGemmLayout, topk_ids: torch.Tensor
) -> _GroupedGemmLayout:
    """``moe_sort`` gives a token one row per distinct expert and -1 to repeats;
    the cuTile kernels weight every assignment, so point each repeat at its
    expert's row."""
    rows = layout.token_to_row
    same = topk_ids.unsqueeze(2) == topk_ids.unsqueeze(1)
    rows = torch.where(same, rows.unsqueeze(1), -1).amax(2)
    return _GroupedGemmLayout(m_indptr=layout.m_indptr, token_to_row=rows)


@dataclass
class _CuTileFp8Workspace(_GroupedGemmWorkspace):
    """The grouped-GEMM stage buffers plus the E4M3 hidden and intermediate rows and
    their scales: one float32 per tensor, or token-major UE8M0 rows and the GEMMs'
    128x4-swizzled copies."""

    hidden_q: Optional[torch.Tensor] = None  # [bucket, H] token order
    hidden_scale: Optional[torch.Tensor] = None  # [1] | [bucket, H / 32]
    intermediate_q: Optional[torch.Tensor] = None  # [rows, I]
    intermediate_scale: Optional[torch.Tensor] = None  # [1] | swizzled
    intermediate_scale_rows: Optional[torch.Tensor] = None  # [rows, I / 32]


class _Sm8xCuTileFp8MoEBase(_Sm8xMoEBodyBase):
    """cuTile FP8 semantics on the cuDNN pipeline: the hidden states quantized in token
    order, each GEMM's output rounded to BF16 after its scales, cuTile's activation in
    float32 and the intermediate requantized over the live rows. Inputs: ``[hidden, ids,
    weights, w1, w1_scale, w2, w2_scale, out]`` with BF16 ``hidden``."""

    _supported_archs = (80, 86, 87, 89)
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn
    _expected_num_inputs = 7
    _intermediate_dtype = torch.float8_e4m3fn
    _intermediate_alignment = 32  # GEMM2's K, whole UE8M0 blocks
    _block_scaled: ClassVar[bool] = True

    def run(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Run the cuTile runner's packed ``inputs`` into their output ``inputs[0]``."""
        out = inputs[0]
        if inputs[1].shape[0] == 0:
            return out
        self.forward([*inputs[1:], out])
        return out

    def _allocate_workspace(
        self, bucket: int, hidden_size: int
    ) -> _GroupedGemmWorkspace:
        from ..quantization.fp4_quantization import _compute_swizzled_layout_sf_size

        base = super()._allocate_workspace(bucket, hidden_size)
        rows, inter = base.num_rows, self.config.experts.intermediate_size

        def empty(*shape: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
            return torch.empty(*shape, dtype=dtype, device=self.device)

        def swizzled(k: int) -> torch.Tensor:
            return empty(_compute_swizzled_layout_sf_size(rows, k // 32))

        workspace = _CuTileFp8Workspace(
            **vars(base),
            hidden_q=empty(bucket, hidden_size, dtype=self._x_dtype),
            intermediate_q=empty(rows, inter, dtype=self._x_dtype),
        )
        if self._block_scaled:
            workspace.hidden_scale = empty(bucket, hidden_size // 32)
            workspace.permuted_hidden_states_scale_rows = torch.zeros(
                rows + 1, hidden_size // 32, dtype=torch.uint8, device=self.device
            )
            workspace.permuted_hidden_states_scale = swizzled(hidden_size)
            workspace.intermediate_scale_rows = empty(rows, inter // 32)
            workspace.intermediate_scale = swizzled(inter)
        else:
            workspace.hidden_scale = empty(1, dtype=torch.float32)
            workspace.intermediate_scale = empty(1, dtype=torch.float32)
        return workspace

    @staticmethod
    def _cutile_workspace(workspace: _GroupedGemmWorkspace) -> _CuTileFp8Workspace:
        assert isinstance(workspace, _CuTileFp8Workspace)
        return workspace

    def _permute(
        self, inputs: List[torch.Tensor]
    ) -> tuple[_GroupedGemmLayout, _GroupedGemmWorkspace]:
        """Quantize the hidden states in token order, then permute the E4M3 rows and,
        block-scaled, scatter and swizzle their scale rows."""
        hidden_states = inputs[0]
        num_tokens = hidden_states.shape[0]
        workspace = self._cutile_workspace(self._ensure_workspace(*hidden_states.shape))
        assert workspace.hidden_q is not None and workspace.hidden_scale is not None
        q = workspace.hidden_q[:num_tokens]
        scale = workspace.hidden_scale
        if self._block_scaled:
            scale = scale[:num_tokens]
        cutile_fp8_quantize(hidden_states, q, scale, block_scaled=self._block_scaled)
        layout, _ = super()._permute([q, *inputs[1:3]])
        layout = _route_repeated_experts(layout, inputs[1])
        if self._block_scaled:
            from ..quantization.fp4_quantization import block_scale_interleave

            rows = workspace.permuted_hidden_states_scale_rows
            assert (
                rows is not None and workspace.permuted_hidden_states_scale is not None
            )
            _scatter_rows(scale, layout.token_to_row, out=rows)
            block_scale_interleave(
                rows[: workspace.num_rows], out=workspace.permuted_hidden_states_scale
            )
        return layout, workspace

    def _grouped_mm(
        self,
        a: torch.Tensor,
        a_scale: torch.Tensor,
        weights: torch.Tensor,
        weight_scale: torch.Tensor,
        layout: _GroupedGemmLayout,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One grouped GEMM's BF16 rows ``a @ weights[e]^T`` after both operands' scales."""
        raise NotImplementedError

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        workspace = self._cutile_workspace(workspace)
        a_scale = (
            workspace.permuted_hidden_states_scale
            if self._block_scaled
            else workspace.hidden_scale
        )
        assert a_scale is not None
        return self._grouped_mm(
            workspace.permuted_hidden_states,
            a_scale,
            inputs[3],
            inputs[4],
            layout,
            workspace.gemm1_out,
        )

    def _activate(self, gemm1_out, layout, workspace, inputs) -> torch.Tensor:
        """cuTile's activation of every row, then its E4M3 requantization; the per-tensor
        scale covers the live rows only."""
        workspace = self._cutile_workspace(workspace)
        intermediate, q = workspace.intermediate, workspace.intermediate_q
        assert q is not None and workspace.intermediate_scale is not None
        cutile_activation(gemm1_out, self.config.activation, intermediate)
        if self._block_scaled:
            from ..quantization.fp4_quantization import block_scale_interleave

            rows = workspace.intermediate_scale_rows
            assert rows is not None
            cutile_fp8_quantize(intermediate, q, rows, block_scaled=True)
            block_scale_interleave(rows, out=workspace.intermediate_scale)
            return q
        slots = layout.token_to_row.reshape(-1)
        row_amax = intermediate.abs().amax(1).index_select(0, slots.clamp(min=0))
        amax = torch.where(slots >= 0, row_amax.float(), 0.0).amax()
        workspace.intermediate_scale.copy_((amax / 448).clamp_min(2.0**-126).reshape(1))
        cutile_fp8_quantize(
            intermediate,
            q,
            workspace.intermediate_scale,
            block_scaled=False,
            scale_ready=True,
        )
        return q

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        workspace = self._cutile_workspace(workspace)
        assert workspace.intermediate_scale is not None
        return self._grouped_mm(
            intermediate,
            workspace.intermediate_scale,
            inputs[5],
            inputs[6],
            layout,
            None,
        )


class Sm8xCuTileFp8PerTensorMoE(_Sm8xCuTileFp8MoEBase):
    """``CuTileFp8PerTensorRunner`` on SM8x: one float32 scale per hidden and
    intermediate tensor and per expert weight, ``bf16(acc * (w_scale[e] * x_scale))``."""

    backend_key = "cutile_fp8_per_tensor"
    _block_scaled = False

    def _grouped_mm(self, a, a_scale, weights, weight_scale, layout, out):
        acc = sm8x_grouped_mm_fp8(
            a, weights, layout.m_indptr, None, torch.float32, None
        )
        row_expert = layout.row_expert(acc.shape[0])
        alpha = weight_scale.index_select(0, row_expert) * a_scale
        acc.mul_(alpha[:, None])
        return acc.to(torch.bfloat16) if out is None else out.copy_(acc)


class Sm8xCuTileMxfp8MoE(_Sm8xCuTileFp8MoEBase):
    """``CuTileMxfp8Runner`` on SM8x: E4M3 weights with ``[E, N, K / 32]`` UE8M0 scales."""

    backend_key = "cutile_mxfp8"

    def _grouped_mm(self, a, a_scale, weights, weight_scale, layout, out):
        return sm8x_grouped_mm_mxfp8(
            a,
            weights,
            a_scale,
            _swizzled_weight_scales(weight_scale),
            layout.m_indptr,
            torch.bfloat16,
            out,
        )


class Sm8xCuTileMxfp4Mxfp8MoE(_Sm8xCuTileFp8MoEBase):
    """``CuTileMxfp4Mxfp8Runner`` on SM8x: packed E2M1 weights ``[E, N, K / 2]`` with
    ``[E, N, K / 32]`` UE8M0 scales."""

    backend_key = "cutile_mxfp4"
    _weight_dtype = torch.uint8

    def _grouped_mm(self, a, a_scale, weights, weight_scale, layout, out):
        return sm8x_grouped_mm_fp4(
            a,
            weights,
            a_scale,
            _swizzled_weight_scales(weight_scale),
            layout.m_indptr,
            None,
            torch.bfloat16,
            out,
            32,
        )


# Launch kwargs of the TRT-LLM FP8 kernels the SM8x pipeline has no counterpart for;
# ``TrtllmFp8BlockRunner`` and ``TrtllmFp8PerTensorRunner`` never pack them.
_TRTLLM_UNSUPPORTED_KWARGS = (
    "gemm1_bias",
    "gemm2_bias",
    "valid_hidden_size",
    "valid_intermediate_size",
    "workspace_buffer",
)
_TRTLLM_UNSUPPORTED_SEQUENCE_KWARGS = ("da_routing_metadata", "da_body_workspace")


class _Sm8xTrtllmFp8MoEBase(_Sm8xCutlassMoEBase):
    """The TRT-LLM FP8 runners' semantics on the cuDNN pipeline: TRT-LLM's routing below
    SM100 (``_trtllm_routing_sm80``) or the caller's experts, then the GEMMs over the
    TRT-LLM weights. As on SM100 the fused shared experts follow the routed ones in the
    ids and in the weights (local experts, then the shared), so the pipeline runs
    ``num_experts + S`` experts of which ``local_num_experts + S`` are local. Inputs:
    ``[hidden, ids, weights, *weights and scales, hidden_states_scale?, out]``."""

    _supported_archs = (80, 86, 87, 89)
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn

    def __init__(self, config: MoEConfig, device: torch.device) -> None:
        routing, shared = config.routing, config.experts.num_fused_shared_experts
        if shared:
            experts = config.experts
            local = (
                routing.num_experts
                if experts.local_num_experts is None
                else experts.local_num_experts
            )
            config = dataclasses.replace(
                config,
                routing=dataclasses.replace(
                    routing,
                    num_experts=routing.num_experts + shared,
                    top_k=routing.top_k + shared,
                ),
                experts=dataclasses.replace(
                    experts,
                    local_num_experts=local + shared,
                    num_fused_shared_experts=0,
                ),
            )
        super().__init__(config, device)
        self._routing, self._num_shared = routing, shared
        # Precomputed experts may repeat within a token; TRT-LLM's routing never does.
        self._repeated_experts = False
        self._config_limit_tensor: Optional[torch.Tensor] = None

    def _weight_inputs(
        self, kwargs: Mapping[str, Any], hidden_states_scale: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        """This pipeline's inputs after ``[hidden, ids, weights]`` from the launch kwargs."""
        raise NotImplementedError

    def _activation_inputs(
        self, kwargs: Mapping[str, Any]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """``cutlass_activation``'s per-expert tensors from the launch kwargs."""
        limit = kwargs.get("gemm1_clamp_limit")
        if limit is None and self.config.activation.type == ActivationType.SwigluStep:
            limit = self._config_limit()
        return dict(
            swiglu_alpha=kwargs.get("gemm1_alpha"),
            swiglu_beta=kwargs.get("gemm1_beta"),
            swiglu_limit=limit,
        )

    def _config_limit(self) -> torch.Tensor:
        """``SwiGLUStep.limit`` of every expert, the kernels' value without a view limit."""
        cached = self._config_limit_tensor
        if cached is None:
            activation = self.config.activation
            assert isinstance(activation, SwiGLUStep)
            cached = torch.full(
                (self._num_local_experts,),
                activation.limit,
                dtype=torch.float32,
                device=self.device,
            )
            self._config_limit_tensor = cached
        return cached

    def run(  # type: ignore[override]
        self, inputs: List[Optional[torch.Tensor]], kwargs: Mapping[str, Any]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run the TRT-LLM runner's packed ``inputs`` (``MoeRunnerInputs.to_list()``) under
        its launch ``kwargs``: the finalized output ``inputs[0]``, or SM100's unfinalized
        ``[gemm2_out, expert_weights, expanded_idx_to_permuted_idx]``."""
        from .api import RoutingInputMode
        from .core import _trtllm_routing_sm80, _validate_routing_replay_out

        name = type(self).__name__
        out, logits, topk_ids, expert_weights, hidden, hidden_scale, lora, pts = inputs
        unsupported = [
            key for key in _TRTLLM_UNSUPPORTED_KWARGS if kwargs.get(key) is not None
        ] + [key for key in _TRTLLM_UNSUPPORTED_SEQUENCE_KWARGS if kwargs.get(key)]
        if lora is not None or pts is not None:
            unsupported.append(
                "gemm1_lora_delta" if lora is not None else "per_token_scale"
            )
        if unsupported:
            raise NotImplementedError(f"{name} has no SM8x path for {unsupported}.")
        assert (
            hidden is not None and topk_ids is not None and expert_weights is not None
        )
        top_k, num_tokens = self._routing.top_k, hidden.shape[0]
        mode = RoutingInputMode(kwargs["routing_input_mode"])
        replay = kwargs.get("routing_replay_out")
        _validate_routing_replay_out(replay, top_k, num_tokens)
        if mode == RoutingInputMode.FromLogits:
            ids, weights = _trtllm_routing_sm80(
                logits,
                kwargs.get("routing_bias"),
                kwargs["routing_method_type"],
                top_k,
                kwargs.get("n_group"),
                kwargs.get("topk_group"),
                kwargs.get("routed_scaling_factor"),
                kwargs.get("norm_topk_prob", True),
                kwargs.get("enable_pdl"),
                self._num_shared,
            )
            if replay is not None:
                replay[:num_tokens].copy_(ids[:, :top_k])
            # SM100's routing writes the caller's buffers when it passes them.
            if topk_ids.numel():
                topk_ids.copy_(ids)
            if expert_weights.numel():
                expert_weights.copy_(weights)
        elif mode == RoutingInputMode.PackedPrecomputed:
            # (expert << 16) | BF16 weight bits; the BF16 halves go into the weight
            # buffer when the caller passes one (packed calls may pass it empty).
            ids = topk_ids >> 16
            weights = topk_ids.view(torch.bfloat16)[:, 0::2].contiguous()
            if expert_weights.numel():
                weights = expert_weights.copy_(weights)
        else:
            ids, weights = topk_ids, expert_weights
        self._repeated_experts = mode != RoutingInputMode.FromLogits
        finalize = self.config.finalize.do_finalize
        if num_tokens == 0:
            if finalize:
                return out
            return [
                hidden.new_empty((0, hidden.shape[1]), dtype=torch.bfloat16),
                weights,
                ids.new_empty((0,)),
            ]
        self._validate_hidden_states(hidden)
        self._activation_params = self._activation_inputs(kwargs)
        body = [
            hidden,
            ids,
            self._finalize_weights(weights, kwargs),
            *self._weight_inputs(kwargs, hidden_scale),
        ]
        if finalize:
            body.append(out)
        result = self.forward(body)
        return out if finalize else result

    def _finalize_weights(
        self, weights: torch.Tensor, kwargs: Mapping[str, Any]
    ) -> torch.Tensor:
        """The routing weights ``moe_unpermute`` applies (and an unfinalized call returns)."""
        return weights


class Sm8xTrtllmMxfp8MoE(_Sm8xTrtllmFp8MoEBase, _CudnnGroupedGemmBlockScaleRunnerBase):
    """``TrtllmFp8BlockRunner`` with MXFP8 on SM8x: the shuffled E4M3 weights with
    their UE8M0 32-element block scales, GEMM1's gated columns ``up, gate, ...``, the
    intermediate MXFP8-quantized for GEMM2. Inputs: ``[hidden, ids, weights, w1,
    w1_scale, w2, w2_scale, hidden_states_scale]`` with linear ``[T, H / 32]`` UE8M0
    ``hidden_states_scale``."""

    backend_key = "trtllm_fp8_block"
    _expected_num_inputs = 8
    _block_size = 32
    _block_scale_dtype = torch.uint8
    _intermediate_alignment = 128  # whole 128x4 scale tiles
    _interleaved_gemm1 = True

    def _weight_inputs(self, kwargs, hidden_states_scale) -> List[torch.Tensor]:
        assert hidden_states_scale is not None
        num_tokens = hidden_states_scale.shape[0]
        return [
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
            hidden_states_scale.view(torch.uint8).reshape(num_tokens, -1),
        ]

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_mxfp8(
            workspace.permuted_hidden_states,
            inputs[3],
            workspace.permuted_hidden_states_scale,
            inputs[4].view(torch.uint8),
            layout.m_indptr,
            torch.float32,
            None,
            shuffled=True,
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _quantize_mxfp8_rows

        quantized, scale = _quantize_mxfp8_rows(intermediate)
        return sm8x_grouped_mm_mxfp8(
            quantized,
            inputs[5],
            scale,
            inputs[6].view(torch.uint8),
            layout.m_indptr,
            torch.bfloat16,
            None,
            shuffled=True,
        )


class Sm8xTrtllmDeepSeekFp8MoE(_Sm8xTrtllmFp8MoEBase):
    """``TrtllmFp8BlockRunner`` with DeepSeek FP8 on SM8x: E4M3 weights with float32
    128x128 block scales, the hidden states' and the intermediate's float32 1x128 block
    scales, GEMM1's gated columns ``[up | gate]``. Inputs: ``[hidden, ids, weights, w1,
    w1_scale, w2, w2_scale, hidden_states_scale]`` with ``[H / 128, T]``
    ``hidden_states_scale``."""

    backend_key = "trtllm_fp8_block"
    _expected_num_inputs = 8
    _intermediate_alignment = 128  # whole 128-element K blocks of GEMM2

    def _weight_inputs(self, kwargs, hidden_states_scale) -> List[torch.Tensor]:
        assert hidden_states_scale is not None
        return [
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
            hidden_states_scale,
        ]

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> tuple[int, int]:
        num_tokens, hidden_size = super()._validate_hidden_states(hidden_states)
        if hidden_size % 128:
            raise ValueError(
                f"{type(self).__name__} requires hidden_size divisible by 128, "
                f"got {hidden_size}."
            )
        return num_tokens, hidden_size

    def _allocate_workspace(
        self, bucket: int, hidden_size: int
    ) -> _GroupedGemmWorkspace:
        workspace = super()._allocate_workspace(bucket, hidden_size)
        # Token-major float32 scale rows, the spare last row for the -1 slots.
        workspace.permuted_hidden_states_scale_rows = torch.zeros(
            workspace.num_rows + 1,
            hidden_size // 128,
            dtype=torch.float32,
            device=self.device,
        )
        return workspace

    def _permute(
        self, inputs: List[torch.Tensor]
    ) -> tuple[_GroupedGemmLayout, _GroupedGemmWorkspace]:
        layout, workspace = super()._permute(inputs)
        rows = workspace.permuted_hidden_states_scale_rows
        assert rows is not None
        _scatter_rows(inputs[7].t(), layout.token_to_row, out=rows)
        return layout, workspace

    @staticmethod
    def _segments(layout: _GroupedGemmLayout, rows: int) -> torch.Tensor:
        """The groupwise GEMM's segment offsets: the last segment to ``rows``."""
        starts = layout.m_indptr[:-1]
        return torch.cat((starts, starts.new_full((1,), rows)))

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        rows = workspace.num_rows
        scale_rows = workspace.permuted_hidden_states_scale_rows
        assert scale_rows is not None
        out = torch.empty(
            rows, self._gemm1_rows, dtype=torch.float32, device=self.device
        )
        _run_group_groupwise(
            workspace.permuted_hidden_states,
            inputs[3],
            scale_rows[:rows],
            inputs[4],
            self._segments(layout, rows),
            out,
            1,
        )
        return out

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _deepseek_fp8_quantize_activations

        rows = intermediate.shape[0]
        quantized, scale = _deepseek_fp8_quantize_activations(intermediate)
        out = torch.empty(
            rows, inputs[5].shape[1], dtype=torch.bfloat16, device=self.device
        )
        _run_group_groupwise(
            quantized,
            inputs[5],
            scale.t(),
            inputs[6],
            self._segments(layout, rows),
            out,
            1,
        )
        return out


class Sm8xPrimsTsDeepSeekFp8MoE(Sm8xTrtllmDeepSeekFp8MoE):
    """``PrimsTsRunner`` with DeepSeek FP8 on SM8x: the TRT-LLM DeepSeek pipeline over
    the Prims-TS weights, whose E4M3 rows ``shuffle_matrix_a`` reordered with epilogue
    tile 64 (within 16-row blocks; the 128x128 scales stay unshuffled). The order stays
    within every 128-row scale block and alike in both halves of ``[up | gate]``, so
    GEMM1 yields the intermediate's columns and GEMM2 the hidden ones in the shuffled
    order: the intermediate goes back to GEMM2's unshuffled K order, and the output
    (after the finalize, which acts per column) to the hidden order."""

    def __init__(self, config: MoEConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._column_orders: Dict[int, torch.Tensor] = {}

    def _column_order(self, size: int) -> torch.Tensor:
        """``order`` with ``unshuffled[:, c] == shuffled[:, order[c]]`` for ``size``
        shuffled columns."""
        order = self._column_orders.get(size)
        if order is None:
            shuffled_from = get_shuffle_matrix_a_row_indices(torch.empty(size, 0), 64)
            order = torch.empty_like(shuffled_from)
            order[shuffled_from] = torch.arange(size)
            order = order.to(self.device)
            self._column_orders[size] = order
        return order

    def _allocate_workspace(
        self, bucket: int, hidden_size: int
    ) -> _GroupedGemmWorkspace:
        workspace = super()._allocate_workspace(bucket, hidden_size)
        # Outside any graph capture, like the workspace.
        self._column_order(self.config.experts.intermediate_size)
        self._column_order(hidden_size)
        return workspace

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        order = self._column_orders[intermediate.shape[1]]
        return super()._gemm2(
            inputs, layout, workspace, intermediate.index_select(1, order), tactic
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not self.config.finalize.do_finalize:
            result = super().forward(inputs, tactic, do_preparation, **kwargs)
            result[0] = result[0].index_select(
                1, self._column_orders[result[0].shape[1]]
            )
            return result
        out = inputs[-1]
        staged = torch.empty_like(out)
        super().forward([*inputs[:-1], staged], tactic, do_preparation, **kwargs)
        return torch.index_select(staged, 1, self._column_orders[out.shape[1]], out=out)


class Sm8xTrtllmFp8PerTensorMoE(_Sm8xTrtllmFp8MoEBase):
    """``TrtllmFp8PerTensorRunner`` on SM8x: the shuffled E4M3 weights and three
    per-expert float32 scales, GEMM1's gated columns ``up, gate, ...``. For Llama4
    routing GEMM1's output first takes the token's routing weight, which the finalize
    then skips. ``output1_scales_gate_scalar`` dequantizes the gate half (all of a
    non-gated output) while the up half stays in accumulator units, its dequant folded
    into ``output1_scales_scalar``, which times the activation into GEMM2's E4M3 input.
    GEMM2's output times ``output2_scales_scalar``. Inputs: ``[hidden, ids, weights, w1,
    w2, output1_scales_scalar, output1_scales_gate_scalar, output2_scales_scalar]``."""

    backend_key = "trtllm_fp8_per_tensor"
    _expected_num_inputs = 8
    _gemm2_alpha_input = 7
    _intermediate_dtype = torch.float8_e4m3fn
    _intermediate_alignment = 16  # GEMM2's K
    _interleaved_gemm1 = True

    def _weight_inputs(self, kwargs, hidden_states_scale) -> List[torch.Tensor]:
        return [
            kwargs["gemm1_weights"],
            kwargs["gemm2_weights"],
            kwargs["output1_scales_scalar"],
            kwargs["output1_scales_gate_scalar"],
            kwargs["output2_scales_scalar"],
        ]

    def _activation_inputs(
        self, kwargs: Mapping[str, Any]
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.config.activation.type != ActivationType.SwigluStep:
            return {}
        # The gate's SiLU clamps at the physical limit, the up half (accumulator units)
        # at the accumulator-unit one. A view limit is in accumulator units (prepare
        # divides by the gate scale); without one the kernels use SwiGLUStep.limit.
        gate_scale = kwargs["output1_scales_gate_scalar"]
        limit = kwargs.get("gemm1_clamp_limit")
        if limit is None:
            physical = self._config_limit()
            return dict(gate_limit=physical, up_limit=physical / gate_scale)
        return dict(gate_limit=limit * gate_scale, up_limit=limit)

    def run(  # type: ignore[override]
        self, inputs: List[Optional[torch.Tensor]], kwargs: Mapping[str, Any]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._input_weights: Optional[torch.Tensor] = None
        return super().run(inputs, kwargs)

    def _finalize_weights(self, weights, kwargs) -> torch.Tensor:
        if not kwargs.get("use_routing_scales_on_input"):
            return weights
        # Llama4: the routing weight scales GEMM1's input instead of the finalize.
        self._input_weights = weights
        if not self.config.finalize.do_finalize:
            return weights
        return torch.ones_like(weights)

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        return sm8x_grouped_mm_fp8(
            workspace.permuted_hidden_states,
            inputs[3],
            layout.m_indptr,
            None,
            torch.float32,
            None,
            shuffled=True,
        )

    def _activate(self, gemm1_out, layout, workspace, inputs) -> torch.Tensor:
        rows = gemm1_out.shape[0]
        row_expert = layout.row_expert(rows)
        x = gemm1_out
        if self._input_weights is not None:
            slots = layout.token_to_row.reshape(-1)
            row_weight = x.new_zeros(rows + 1).index_copy_(
                0,
                torch.where(slots >= 0, slots, rows).long(),
                self._input_weights.reshape(-1).float(),
            )
            x.mul_(row_weight[:rows, None])
        gate_scale = inputs[6].index_select(0, row_expert)[:, None]
        activation_type = self.config.activation.type
        if not self.config.activation.is_gated:
            y = cutlass_activation(x.mul_(gate_scale), activation_type, {}, row_expert)
        else:
            up, gate = x[:, 0::2], x[:, 1::2]
            y = F.silu(gate.mul_(gate_scale))
            if activation_type == ActivationType.SwigluStep:
                params = self._activation_params
                # fmin: a zero gate scale with an infinite view limit gives NaN here.
                y = torch.fmin(y, _per_row(params, "gate_limit", row_expert, math.inf))
                up_limit = _per_row(params, "up_limit", row_expert, math.inf)
                up = up.clamp(-up_limit, up_limit)
            elif activation_type != ActivationType.Swiglu:
                raise NotImplementedError(
                    f"{type(self).__name__} has no {activation_type!r} activation."
                )
            y = y.mul_(up)
        scale = inputs[5].index_select(0, row_expert)[:, None]
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        return y.mul_(scale).clamp_(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        acc = sm8x_grouped_mm_fp8(
            intermediate,
            inputs[4],
            layout.m_indptr,
            None,
            torch.float32,
            None,
            shuffled=True,
        )
        return self._epilogue(acc, self._gemm2_alpha_input, layout, inputs)
