# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The CUTLASS and cuTile FP8-family fused-MoE runners' semantics over the SM8x grouped
GEMMs, on the architectures where the CUTLASS fused-MoE module has no kernel for the
quantization pair or cuTile cannot compile FP8.

``_CutlassRunnerBase`` and ``_CuTileFp8Runner`` delegate to these: the cuDNN grouped-GEMM
pipeline (``moe_sort`` -> ``moe_permute`` -> GEMM1 -> activation -> GEMM2 ->
``moe_unpermute``) with the ``sm8x_fp8_dense`` grouped GEMMs and the delegating runner's
activation set and scale conventions.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.weak import WeakIdKeyDictionary

from ..gemm.sm8x_fp4_dense import _unswizzle
from ..gemm.sm8x_fp8_dense import (
    sm8x_grouped_mm_fp4,
    sm8x_grouped_mm_fp8,
    sm8x_grouped_mm_mxfp8,
)
from .api import ActivationConfig, ActivationType, MoEConfig
from .runners import (
    _check_moe_sort_limits,
    _CudnnGroupedGemmBlockScaleRunnerBase,
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
) -> torch.Tensor:
    """The CUTLASS fused-MoE activation of float32 GEMM1 rows ``x``: ``[rows, I]``, or
    ``[rows, 2I]`` as ``[up | gate]`` for the gated ones. ``params`` are the runner's
    optional per-expert tensors (``_cutlass_activation_params``) and ``row_expert``
    every row's expert; an absent tensor takes the kernel's default."""
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

    def __init__(self, config: MoEConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._activation_params: Dict[str, Optional[torch.Tensor]] = {}

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

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_fp4(
            workspace.permuted_hidden_states,
            inputs[3].view(torch.uint8),
            workspace.permuted_hidden_states_scale,
            inputs[5],
            layout.m_indptr,
            None,
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
            None,
            torch.bfloat16,
            None,
            32,
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
