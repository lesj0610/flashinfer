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

SM90 (wgmma) FP4 MoE driver (``csrc/fp4_moe_sm90.cu``).

The SM100+ MoE entry points (``trtllm_fp4_block_scale_moe`` and its routed variant,
``cutlass_fused_moe``, ``cute_dsl_fused_moe``) run their NVFP4 tuples here on SM90,
on the weights in each entry point's own SM100+ layout. Other tuples raise on SM90.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ...autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...jit.fused_moe import fp4_sm90_recipe_bits, gen_fp4_moe_sm90_module
from ...quantization.nvfp4_quantization_utils import (
    env_flag_enabled,
    nvfp4_4over6_cache_key,
    resolve_nvfp4_4over6,
)
from ...tllm_enums import (
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
    ActivationType,
)
from ..shared.validation import resolve_moe_act_sf_layout
from ..utils import make_hybrid_bucket_mapper, map_to_hybrid_bucket
from .spec import (
    SM90_TACTIC,
    AFmt,
    Fq,
    Pair,
    Sm90EpilogueSpec,
    Sm90Routing,
    Sm90WeightView,
    XFmt,
)
from .weights import sm90_weight_view

__all__ = [
    "SM90_FP4_TACTICS",
    "Sm90Fp4MoeRunner",
    "default_sm90_fp4_tactic",
    "sm90_fp4_moe",
    "sm90_fp4_moe_supports",
    "sm90_fp4_valid_tactics",
]

#: D7 shapes (cta_m, n, stages); stages 0 = the most that fit shared memory, as built.
#: ``Sm90Fp4MoeRunner.get_valid_tactics`` reports the built stage counts and drops shapes
#: whose kernels hold fewer than two stages.
SM90_FP4_TACTICS: Tuple[SM90_TACTIC, ...] = tuple(
    (cta_m, n, 0)
    for cta_m in (128, 256)
    for n in (8, 16, 32, 64, 128, 256)
    if (cta_m // 128) * n // 2 <= 128
)

#: Activations of the ``fp4_sm90`` GEMM1 epilogue: ``cutlass_fused_moe``'s (gate/up pairing
#: follows ``ActivationType.is_gated``).
SM90_FP4_ACTIVATIONS = tuple(
    a
    for a in ActivationType
    if a not in (ActivationType.InvalidType, ActivationType.ClampedRelu2)
)


def _not_on_sm90(what: str) -> NotImplementedError:
    return NotImplementedError(f"SM90 FP4 MoE: {what} is not implemented on SM90 yet.")


def default_sm90_fp4_tactic(
    num_tokens: int, top_k: int, num_local_experts: int
) -> SM90_TACTIC:
    """Untuned tactic (D7): CTA_M 128, n the smallest power of two in [8, 256] at least the
    mean rows per local expert."""
    rows = -(-num_tokens * top_k // max(num_local_experts, 1))
    n = 8
    while n < rows and n < 256:
        n *= 2
    return (128, n, 0)


@functools.cache
def get_fp4_moe_sm90_module(recipe=None, exact=False, act=None, f16=False):
    return gen_fp4_moe_sm90_module(recipe, exact, act, f16).build_and_load()


@functools.cache
def _module_tactics(
    recipe, exact, act, f16, per_token, w1_fmt, w2_fmt, a_path
) -> Tuple[SM90_TACTIC, ...]:
    flat = list(
        get_fp4_moe_sm90_module(recipe, exact, act, f16).fp4_moe_sm90_tactics(
            per_token, int(w1_fmt), int(w2_fmt), a_path
        )
    )
    return tuple(
        (int(flat[i]), int(flat[i + 1]), int(flat[i + 2]))
        for i in range(0, len(flat), 3)
    )


#: Weight layout of each entry point's view (``fp4_sm90::AFmt``; CuTe DSL's SF is 128x4 bytes).
_ENTRY_AFMT = {
    "trtllm": AFmt.NVFP4_TRTLLM_SHUFFLED,
    "cutlass": AFmt.NVFP4_SWZ128X4,
    "cute_dsl": AFmt.NVFP4_SWZ128X4,
}


#: Activations each entry point runs on SM90 (at their default parameters).
_ENTRY_ACTIVATIONS = {
    "trtllm": (ActivationType.Swiglu, ActivationType.Relu2),
    "cutlass": SM90_FP4_ACTIVATIONS,
    "cute_dsl": (ActivationType.Swiglu, ActivationType.Relu2),
}


def sm90_fp4_moe_supports(entry: str, *, per_token: bool, activation_type: int) -> bool:
    """Whether SM90 runs this NVFP4 x NVFP4 tuple of ``entry`` (``"trtllm"``, ``"cutlass"`` or
    ``"cute_dsl"``): per-token activation scales for TRT-LLM and CuTe DSL, per-tensor for
    CUTLASS, and an activation the entry runs on SM90. The entry points reject the rest on
    SM90 with ``NotImplementedError``."""
    return (
        per_token == (entry != "cutlass")
        and ActivationType(activation_type) in _ENTRY_ACTIVATIONS[entry]
    )


def sm90_fp4_valid_tactics(
    entry: str, recipe, act: int, per_token: bool, a_path: int = 0
) -> List[SM90_TACTIC]:
    """Tactics of ``entry``'s (``"trtllm"``, ``"cutlass"`` or ``"cute_dsl"``) M3 tuple, as
    ``Sm90Fp4MoeRunner.get_valid_tactics`` reports them (builds the module)."""
    fmt = _ENTRY_AFMT[entry]
    return list(
        _module_tactics(recipe, _exact(), int(act), False, per_token, fmt, fmt, a_path)
    )


def _scalar(t: Optional[torch.Tensor]) -> bool:
    return t is not None and t.numel() == 1


def _device_scalar(t: torch.Tensor, what: str) -> torch.Tensor:
    """``t`` as the device FP32 scalar the kernel reads (no host read, graph-capture safe)."""
    if not t.is_cuda or t.dtype != torch.float32 or t.numel() != 1:
        raise TypeError(f"SM90 FP4 MoE: {what} must be a CUDA FP32 scalar tensor")
    return t.reshape(1)


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
        w.num_experts,
    ]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _tuning(
    num_experts: int,
    num_tokens: int,
    tune_max_num_tokens: int,
    inputs: Sequence[Optional[torch.Tensor]],
    x_sf_swizzled: bool,
) -> TuningConfig:
    """Tuning of ``Sm90Fp4MoeRunner``'s inputs [x, topk_ids, topk_weights, x_sf, token_scale,
    out] at ``num_tokens``' bucket: seeded inputs, ids and softmaxed weights, cold L2."""

    def seeded(fn, *args):
        def init(shape, dtype, device):
            generator = torch.Generator(device).manual_seed(515)
            return fn(*args, shape, dtype=dtype, device=device, generator=generator)

        return init

    def empty(shape, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device)

    def given(i):
        return inputs[i] is not None

    codes, randn = inputs[0].dtype == torch.uint8, seeded(torch.randn)
    dynamic = [0, 1] + [2] * given(2) + [3] * (given(3) and not x_sf_swizzled)
    dynamic += [4] * given(4) + [5] * given(5)
    initializers = [
        (0, seeded(torch.randint, 0, 256) if codes else randn),
        (1, seeded(torch.randint, 0, num_experts)),
    ]
    if given(2):
        initializers.append((2, lambda *args: randn(*args).softmax(-1)))
    if given(3):  # E4M3 scales by their bytes in [2^-1, 2^2): FP8 has no random fill
        scales = seeded(torch.randint, 48, 72)
        initializers.append(
            (3, lambda s, dtype, d: scales(s, torch.uint8, d).view(dtype))
        )
    if given(4):
        initializers.append((4, seeded(torch.rand)))
    if given(5):
        initializers.append((5, empty))
    constraints = []
    if given(3) and x_sf_swizzled:  # 128x4 tiles over [tokens, K / 16]
        constraints.append(
            ConstraintSpec(
                3, 0, lambda s: -(-s[0][0] // 128) * 128 * -(-s[0][1] // 32) * 4
            )
        )
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                tuple(dynamic),
                (0,) * len(dynamic),
                (map_to_hybrid_bucket(num_tokens, tune_max_num_tokens),),
                make_hybrid_bucket_mapper(tune_max_num_tokens),
            ),
        ),
        constraint_specs=tuple(constraints),
        tensor_initializers=tuple(initializers),
        use_cold_l2_cache=True,
    )


class Sm90Fp4MoeRunner(TunableRunner):
    """``sm90_fp4_moe`` on fixed weights and epilogues; inputs [x, topk_ids, topk_weights,
    x_sf, token_scale, out], tactic (cta_m, n, stages)."""

    def __init__(
        self,
        w1: Sm90WeightView,
        w2: Sm90WeightView,
        epi1: Sm90EpilogueSpec,
        epi2: Sm90EpilogueSpec,
        *,
        num_experts: int,
        local_expert_offset: int,
        num_local_experts: int,
        recipe=None,
        exact: bool = False,
        x_global: Optional[torch.Tensor] = None,
        x_sf_swizzled: bool = False,
        do_finalize: bool = True,
        a_path: int = 0,
    ):
        self.w1, self.w2, self.epi1, self.epi2 = w1, w2, epi1, epi2
        self.num_experts, self.num_local_experts = num_experts, num_local_experts
        self.local_expert_offset = local_expert_offset
        self.recipe, self.exact, self.x_global = recipe, exact, x_global
        self.x_sf_swizzled, self.do_finalize, self.a_path = (
            x_sf_swizzled,
            do_finalize,
            a_path,
        )
        self.per_token = epi1.fq == Fq.NONE
        self.module_key = (recipe, exact, int(epi1.act), False)
        self.key = (
            nvfp4_4over6_cache_key(recipe),
            exact,
            int(epi1.act),
            self.per_token,
            tuple(_geometry(w1)),
            tuple(_geometry(w2)),
            num_experts,
            local_expert_offset,
            num_local_experts,
            x_sf_swizzled,
            do_finalize,
            a_path,
            epi2.out_dtype,
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, self.x_global is not None)

    def get_valid_tactics(
        self, inputs, profile: OptimizationProfile
    ) -> List[SM90_TACTIC]:
        return list(
            _module_tactics(
                *self.module_key,
                self.per_token,
                self.w1.a_fmt,
                self.w2.a_fmt,
                self.a_path,
            )
        )

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        x, ids, weights, x_sf, token_scale, out = inputs
        tokens, top_k = ids.shape
        if tactic is None or tactic == -1:
            tactic = default_sm90_fp4_tactic(tokens, top_k, self.num_local_experts)
        cta_m, n, stages = tactic
        if x.dtype == torch.uint8:
            x_fmt = XFmt.NVFP4_SWZ128X4 if self.x_sf_swizzled else XFmt.NVFP4_LINEAR
            quantize = 0
        else:
            x_fmt = XFmt.F16 if x.dtype == torch.float16 else XFmt.BF16
            quantize = 1
        module = get_fp4_moe_sm90_module(*self.module_key)
        hidden, inter = self.w1.k, self.w2.k
        size = module.fp4_moe_sm90_workspace_bytes(
            tokens,
            top_k,
            self.num_experts,
            self.local_expert_offset,
            self.num_local_experts,
            hidden,
            inter,
            int(x_fmt),
            quantize,
            self.per_token,
            n,
        )
        workspace = torch.empty(size, dtype=torch.uint8, device=x.device)
        src_permuted = torch.empty(tokens * top_k, dtype=torch.int32, device=x.device)
        out_dtype = self.epi2.out_dtype
        y = torch.empty(tokens * top_k, hidden, dtype=out_dtype, device=x.device)
        if self.do_finalize and out is None:
            out = torch.empty(tokens, hidden, dtype=out_dtype, device=x.device)
        module.fp4_moe_sm90_run(
            x,
            None if x_sf is None else _bytes(x_sf),
            int(x_fmt),
            self.x_global,
            quantize,
            token_scale,
            ids,
            weights,
            self.num_experts,
            self.local_expert_offset,
            self.num_local_experts,
            _bytes(self.w1.codes),
            _bytes(self.w1.sf),
            _geometry(self.w1),
            _bytes(self.w2.codes),
            _bytes(self.w2.sf),
            _geometry(self.w2),
            self.epi1.alpha,
            self.epi2.alpha,
            *self.epi1.act_params,
            int(self.epi1.act),
            self.per_token,
            self.epi1.fq_global,
            self.epi1.bias,
            self.epi2.bias,
            workspace,
            src_permuted,
            y,
            out if self.do_finalize else None,
            cta_m,
            n,
            stages,
            self.a_path,
        )
        if self.do_finalize:
            return [out]
        return [y, weights, src_permuted.view(tokens, top_k)]


def sm90_fp4_moe(
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    w1: Sm90WeightView,
    w2: Sm90WeightView,
    routing: Sm90Routing,
    epi1: Sm90EpilogueSpec,
    epi2: Sm90EpilogueSpec,
    *,
    out: Optional[torch.Tensor] = None,
    tactic: Any = -1,
    recipe=None,
    exact: bool = False,
    x_global: Optional[torch.Tensor] = None,
    token_scale: Optional[torch.Tensor] = None,
    x_sf_swizzled: bool = False,
    do_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    a_path: int = 0,
) -> List[torch.Tensor]:
    """SM90 FP4 MoE: prep, GEMM1, row quantization (token-wise), GEMM2, finalize.

    Parameters
    ----------
    x : torch.Tensor
        BF16/FP16 ``[num_tokens, hidden]`` (quantized in prep with ``recipe`` under
        ``x_global``) or NVFP4 codes uint8 ``[num_tokens, hidden / 2]``.
    x_sf : Optional[torch.Tensor]
        E4M3 block scales of NVFP4 ``x``: linear ``[num_tokens, hidden / 16]`` or
        128x4-swizzled (``x_sf_swizzled``).
    w1, w2 : Sm90WeightView
        GEMM1 (gated: gate/up pair) and GEMM2 weights in an SM100+ entry point's layout.
    routing : Sm90Routing
        Global expert ids int32 ``[num_tokens, top_k]`` and their weights.
    epi1 : Sm90EpilogueSpec
        GEMM1: activation, its ``act_params``, ``alpha`` FP32 ``[num_local_experts]``, ``bias``;
        ``fq`` ``Fq.NONE`` for the token-wise recipe (rows requantized with ``recipe`` per
        row, ``token_scale`` per token on GEMM1) or ``Fq.NVFP4`` per tensor (under
        ``fq_global``).
    epi2 : Sm90EpilogueSpec
        GEMM2: ``alpha`` FP32 ``[num_local_experts]``, ``bias``, ``out_dtype`` BF16 or FP16.
    out : Optional[torch.Tensor]
        Finalized output ``[num_tokens, hidden]`` (allocated if None).
    tactic : tuple or int
        ``(cta_m, n, stages)``; -1 autotunes (or the untuned default outside tuning).
    recipe : Optional[NVFP44Over6Config]
        Resolved NVFP4 recipe of every quantization in the call (the module's).
    exact : bool
        Quantizers without fast math (``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH``).
    x_global : Optional[torch.Tensor]
        FP32 scalar global scale of the in-prep input quantization (16-bit ``x``).
    token_scale : Optional[torch.Tensor]
        FP32 ``[num_tokens]`` per-token scales of NVFP4 ``x`` (token-wise recipe).
    x_sf_swizzled : bool
        ``x_sf`` 128x4-swizzled instead of linear.
    do_finalize : bool
        ``[out]`` if True, else ``[y, topk_weights, src_permuted]`` with ``y`` the GEMM2 rows
        ``[num_tokens * top_k, hidden]`` and ``src_permuted`` ``[num_tokens, top_k]`` the row
        of each (token, slot), -1 for other ranks' experts.
    tune_max_num_tokens : int
        Largest token bucket profiled.
    a_path : int
        Weight staging path of the GEMM kernels: 0 (n1) or 1 (n2).

    Returns
    -------
    List[torch.Tensor]
        See ``do_finalize``.
    """
    act = ActivationType(epi1.act)
    per_token = epi1.fq == Fq.NONE
    num_local = routing.num_local_experts
    if num_local is None:
        num_local = routing.num_experts
    if (
        act not in SM90_FP4_ACTIVATIONS
        or epi1.fq not in (Fq.NONE, Fq.NVFP4)
        or epi2.fq != Fq.NONE
        or not epi2.gemm2
        or epi2.out_dtype not in (torch.bfloat16, torch.float16)
        or (not per_token and epi1.recipe != fp4_sm90_recipe_bits(recipe))
        or w1.gated != act.is_gated
        or {w1.a_fmt, w2.a_fmt} - {AFmt.NVFP4_TRTLLM_SHUFFLED, AFmt.NVFP4_SWZ128X4}
        or w2.pair != Pair.NONE
    ):
        raise _not_on_sm90(
            f"activation {act.name}, epilogues {epi1.fq.name}/{epi2.fq.name} or weights "
            f"{w1.a_fmt.name}/{w2.a_fmt.name} (NVFP4 only)"
        )
    if per_token and (token_scale is None or x.dtype != torch.uint8):
        raise _not_on_sm90("the token-wise recipe without NVFP4 per-token activations")
    if x.dtype != torch.uint8 and not _scalar(x_global):
        raise _not_on_sm90("16-bit input quantization without one global scale")
    if min(w1.num_experts, w2.num_experts) < num_local or epi1.alpha is None:
        raise ValueError("SM90 FP4 MoE: weights and alphas of every local expert")
    ids = routing.topk_ids
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        ids = ids.to(torch.int32).contiguous()
    weights = routing.topk_weights
    if weights is not None and not weights.is_contiguous():
        weights = weights.contiguous()
    alpha = [e.alpha.float().reshape(-1)[:num_local].contiguous() for e in (epi1, epi2)]
    runner = Sm90Fp4MoeRunner(
        w1,
        w2,
        replace(epi1, alpha=alpha[0]),
        replace(epi2, alpha=alpha[1]),
        num_experts=routing.num_experts,
        local_expert_offset=routing.local_expert_offset,
        num_local_experts=num_local,
        recipe=recipe,
        exact=exact,
        x_global=None if x.dtype == torch.uint8 else x_global.float().reshape(1),
        x_sf_swizzled=x_sf_swizzled,
        do_finalize=do_finalize,
        a_path=a_path,
    )
    inputs = [x, ids, weights, x_sf, token_scale, out if do_finalize else None]
    if isinstance(tactic, int) and tactic == -1:
        tuning = _tuning(
            routing.num_experts, x.shape[0], tune_max_num_tokens, inputs, x_sf_swizzled
        )
        _, tactic = AutoTuner.get().choose_one(
            "flashinfer::fp4_moe_sm90", [runner], tuning, inputs
        )
    return runner(inputs, tactic=tactic)


def _exact() -> bool:
    return any(
        env_flag_enabled(f"{p}_DISABLE_FP4_QUANT_FAST_MATH")
        for p in ("FLASHINFER", "TRTLLM")
    )


def _per_expert(t: torch.Tensor, experts: int) -> torch.Tensor:
    """One FP32 scale per local expert from one for all or [>= local experts] (the local first,
    as SM100 reads them)."""
    if t.dim() > 1 or t.numel() not in (1,) and t.numel() < experts:
        raise ValueError(
            f"expected one scale or [local_num_experts], got {tuple(t.shape)}"
        )
    t = t.float().reshape(-1)
    return t.expand(experts) if t.numel() == 1 else t[:experts]


# ---------------------------------------------------------------------------------------
# Entry points: the SM100+ public APIs' arguments, their M3 tuples on SM90.
# ---------------------------------------------------------------------------------------


def trtllm_fp4_sm90_moe(
    a: Dict[str, Any], ids: torch.Tensor, weights: torch.Tensor
) -> List[torch.Tensor]:
    """``trtllm_fp4_block_scale_moe`` / ``_routed_moe`` (their arguments ``a``; ``tactic`` an SM90
    tactic or -1) on SM90 with the global expert ``ids`` and BF16 ``weights``: NVFP4 x NVFP4
    per token."""
    x, x_scale, pt = a["hidden_states"], a["hidden_states_scale"], a["per_token_scale"]
    act = ActivationType(a["activation_type"])
    s1, s_gate, s2 = (
        a[f"output{i}_scalar"] for i in ("1_scale", "1_scale_gate", "2_scale")
    )
    recipe = resolve_nvfp4_4over6(a["nvfp4_4over6"]) if pt is not None else None
    unsupported = [
        n
        for n in (
            "gemm1_bias",
            "gemm1_alpha",
            "gemm1_beta",
            "gemm1_clamp_limit",
            "gemm2_bias",
            "gemm1_lora_delta",
            "fc2_input_out",
            "workspace_buffer",
            "routing_metadata",
            "valid_hidden_size",
            "valid_intermediate_size",
            "localized_weights",
        )
        if a.get(n) is not None
    ]
    if (
        x.dtype != torch.uint8
        or x_scale is None
        or pt is None
        or act not in (ActivationType.Swiglu, ActivationType.Relu2)
        or unsupported
        or a.get("_nfse")
        or None in (s1, s_gate, s2)
        or s_gate.dim() > 1
        or a["gemm1_weights"].dtype != torch.uint8
    ):
        raise _not_on_sm90(
            "TRT-LLM FP4 MoE outside NVFP4 x NVFP4 per-token with Swiglu or Relu2 "
            f"(set: {unsupported})"
        )
    if "hidden_states_scale_layout" in a:  # SM100's: linear only
        resolve_moe_act_sf_layout(a["hidden_states_scale_layout"])
    tokens, hidden = x.shape[0], x.shape[1] * 2
    if pt.dtype != torch.float32 or pt.numel() != tokens or not pt.is_contiguous():
        raise ValueError("per_token_scale must be FP32 [num_tokens], contiguous.")
    if x_scale.numel() != tokens * (hidden // 16):
        raise _not_on_sm90(
            "hidden_states_scale other than linear [num_tokens, hidden / 16]"
        )
    experts, inter = a["local_num_experts"], a["intermediate_size"]
    gated = act.is_gated
    finalize, out = a["do_finalize"], a["output"]
    if (
        finalize
        and out is not None
        and (out.shape != (tokens, hidden) or out.dtype != torch.bfloat16)
    ):
        raise _not_on_sm90("an output other than BF16 [num_tokens, hidden]")
    w1 = sm90_weight_view(
        "trtllm",
        a["gemm1_weights"],
        a["gemm1_weights_scale"],
        rows=inter * (2 if gated else 1),
        k=hidden,
        gated=gated,
        scale_kind="nvfp4",
    )
    w2 = sm90_weight_view(
        "trtllm",
        a["gemm2_weights"],
        a["gemm2_weights_scale"],
        rows=hidden,
        k=inter,
        gated=False,
        scale_kind="nvfp4",
    )
    # As SM100: GEMM1 under the gate scalar, GEMM2 input under output1_scale_scalar (gated over
    # the gate scalar, folded into GEMM2's alpha since the rows are requantized per token).
    alpha1 = _per_expert(s_gate, experts)
    fc2 = _per_expert(s1, experts)
    fc2 = fc2 / alpha1 if gated else fc2
    alpha2 = _per_expert(s2, experts) * fc2
    return sm90_fp4_moe(
        x.view(torch.uint8),
        x_scale.view(torch.uint8).reshape(tokens, hidden // 16),
        w1,
        w2,
        Sm90Routing(
            ids,
            weights,
            ids.shape[1],
            a["num_experts"],
            a["local_expert_offset"],
            experts,
        ),
        Sm90EpilogueSpec(int(act), alpha=alpha1),
        Sm90EpilogueSpec(int(ActivationType.Identity), gemm2=True, alpha=alpha2),
        out=out if finalize else None,
        tactic=a.get("tactic", -1),
        recipe=recipe,
        exact=_exact(),
        token_scale=pt,
        do_finalize=finalize,
        tune_max_num_tokens=a["tune_max_num_tokens"],
    )


def cutlass_fp4_sm90_moe(a: Dict[str, Any], recipe) -> torch.Tensor:
    """``cutlass_fused_moe`` (its arguments ``a``, ``output`` allocated; unset optional ones at
    their defaults, ``tactic`` an SM90 tactic or -1) on SM90 with the resolved ``recipe``:
    NVFP4 x NVFP4 per tensor, BF16/FP16 or NVFP4 input."""
    qs, act = a["quant_scales"], ActivationType(a["activation_type"])
    x, x_sf, out = a["input"], a.get("input_sf"), a["output"]
    fc1, fc2 = a["fc1_expert_weights"], a["fc2_expert_weights"]
    profile_ids = a.get("profile_ids")
    if profile_ids is not None and len(profile_ids) != 2:
        raise ValueError("profile_ids must contain [gemm1_profile, gemm2_profile]")
    unsupported = [n for n in ("workspace_buffer",) if a.get(n) is not None]
    # As SM100: -1 keeps a GEMM's default tactic without the tuner; SM100 tactic ids mean
    # nothing here.
    if profile_ids is not None and list(profile_ids) != [-1, -1]:
        unsupported.append("profile_ids")
    biases = (a.get("fc1_expert_biases"), a.get("fc2_expert_biases"))
    swiglu = tuple(a.get(n) for n in ("swiglu_alpha", "swiglu_beta", "swiglu_limit"))
    # As cutlass_fused_moe: Swiglu with SwiGLU parameters runs SwigluBias; SiTU reads situ_beta
    # and situ_linear_beta.
    if act == ActivationType.Swiglu and any(t is not None for t in swiglu):
        act = ActivationType.SwigluBias
    act_params = (
        (a.get("situ_beta"), a.get("situ_linear_beta"), None)
        if act == ActivationType.Situ
        else swiglu
    )
    codes = x.dtype == torch.uint8
    ep_size, ep_rank = a.get("ep_size", 1), a.get("ep_rank", 0)
    if (
        len(qs) != 6
        or act not in SM90_FP4_ACTIVATIONS
        or unsupported
        or a.get("cluster_size", 1) != 1
        or a.get("use_packed_weights", False)
        or (a.get("enable_alltoall", False) and ep_size > 1)
        or a["output_dtype"] not in (torch.bfloat16, torch.float16)
        or codes != (x_sf is not None)
        or (not codes and x.dtype not in (torch.bfloat16, torch.float16))
        or (not codes and not _scalar(qs[0]))
        or not _scalar(qs[3])
        or qs[2].numel() != fc1.shape[0]
        or qs[5].numel() != fc1.shape[0]
    ):
        raise _not_on_sm90(
            "CUTLASS FP4 MoE outside NVFP4 x NVFP4 per tensor with one global input scale "
            f"per tensor (set: {unsupported})"
        )
    if (biases[0] is None) != (biases[1] is None):
        raise ValueError("fc1_expert_biases and fc2_expert_biases go together")
    if (
        a.get("tp_rank", 0) != 0
    ):  # as cutlass_fused_moe's finalize: fc2 biases on TP rank 0
        biases = (biases[0], None)
    gated = act.is_gated
    experts, hidden = fc2.shape[0], fc2.shape[1]
    inter = fc2.shape[2] * (16 if fc2.dtype == torch.int64 else 2)
    if fc1.shape[0] != experts or fc1.shape[1] != inter * (2 if gated else 1):
        raise ValueError(
            "fc1_expert_weights must be [num_experts, (2 *) intermediate_size, ...] as "
            "fc2_expert_weights."
        )
    w1 = sm90_weight_view(
        "cutlass",
        fc1.view(torch.uint8),
        qs[1],
        rows=fc1.shape[1],
        k=hidden,
        gated=gated,
        scale_kind="nvfp4",
    )
    w2 = sm90_weight_view(
        "cutlass",
        fc2.view(torch.uint8),
        qs[4],
        rows=hidden,
        k=inter,
        gated=False,
        scale_kind="nvfp4",
    )
    bits = fp4_sm90_recipe_bits(recipe)
    sm90_fp4_moe(
        x.view(torch.uint8) if codes else x,
        x_sf,
        w1,
        w2,
        Sm90Routing(
            a["token_selected_experts"],
            a["token_final_scales"],
            a["token_selected_experts"].shape[1],
            experts * ep_size,
            experts * ep_rank,
            experts,
        ),
        Sm90EpilogueSpec(
            int(act),
            fq=Fq.NVFP4,
            recipe=bits,
            alpha=qs[2],
            act_params=act_params,
            bias=biases[0],
            fq_global=_device_scalar(qs[3], "the GEMM2 input global scale"),
        ),
        Sm90EpilogueSpec(
            int(ActivationType.Identity),
            gemm2=True,
            out_dtype=a["output_dtype"],
            alpha=qs[5],
            bias=biases[1],
        ),
        out=out,
        tactic=None if profile_ids is not None else a.get("tactic", -1),
        recipe=recipe,
        exact=_exact(),
        x_global=None if codes else qs[0],
        x_sf_swizzled=codes and a.get("swizzled_input_sf", False),
        tune_max_num_tokens=a.get("tune_max_num_tokens", 8192),
    )
    return out


def cute_dsl_fp4_sm90_moe(
    a: Dict[str, Any], activation: ActivationType, swap_ab: bool
) -> torch.Tensor:
    """``cute_dsl_fused_moe`` (its arguments ``a``) on SM90: W4A4 per token, weights in
    the CuTe DSL layout (group interleave 64, or 16 with ``swap_ab``)."""
    x, x_sf, pt = a["x"], a["x_sf"], a["per_token_scale"]
    recipe = resolve_nvfp4_4over6(a["nvfp4_4over6"]) if pt is not None else None
    swiglu = (a["swiglu_alpha"], a["swiglu_beta"], a["swiglu_limit"])
    defaults = (DEFAULT_SWIGLU_ALPHA, DEFAULT_SWIGLU_BETA, DEFAULT_SWIGLU_LIMIT)
    output_dtype = a["output_dtype"]
    if (
        a["quant_mode"] != "w4a4"
        or pt is None
        or activation not in (ActivationType.Swiglu, ActivationType.Relu2)
        or swiglu != defaults
        or a["situ_beta"] is not None
        or a["clamped_relu2_limit"] is not None
        or a["routing_weights_before_fc2"]
        or a["localized_weights"] is not None
        or output_dtype not in (torch.bfloat16, torch.float16)
        or x.dtype != torch.uint8
        or x_sf is None
        or a["fc2_input_scale"] is None
    ):
        raise _not_on_sm90(
            "CuTe DSL FP4 MoE outside W4A4 per-token with Swiglu or Relu2 at the "
            "default activation parameters"
        )
    tokens, hidden = x.shape[0], x.shape[1] * 2
    if pt.dtype != torch.float32 or pt.numel() != tokens or not pt.is_contiguous():
        raise ValueError("per_token_scale must be FP32 [num_tokens], contiguous.")
    # As SM100's per-token quantizer: the GEMM2 input rows under this inverse global scale.
    gsi = a["fc2_input_scale"]
    if isinstance(gsi, torch.Tensor):
        gsi = gsi.float().reshape(1).contiguous().to(x.device)
    else:
        gsi = torch.tensor([float(gsi)], dtype=torch.float32, device=x.device)
    if x_sf.dtype not in (torch.uint8, torch.float8_e4m3fn):  # as _routed_nvfp4
        raise TypeError("x_sf must have dtype uint8 or float8_e4m3fn")
    if x_sf.numel() != tokens * (hidden // 16):
        raise ValueError("x_sf must be linear: [num_tokens, hidden / 16] E4M3 scales.")
    w1w, w2w = a["w1_weight"], a["w2_weight"]
    inter = w2w.shape[2] * 2
    gated = activation.is_gated
    local = a["num_local_experts"]
    local = a["num_experts"] if local is None else local
    w1 = sm90_weight_view(
        "cute_dsl",
        w1w,
        a["w1_weight_sf"],
        rows=inter * (2 if gated else 1),
        k=hidden,
        gated=gated,
        scale_kind="nvfp4",
        pair_g=16 if swap_ab else 64,
    )
    w2 = sm90_weight_view(
        "cute_dsl",
        w2w,
        a["w2_weight_sf"],
        rows=hidden,
        k=inter,
        gated=False,
        scale_kind="nvfp4",
    )
    out = a["moe_output"]
    ids = a["token_selected_experts"]
    if ids.shape[1:] != (a["top_k"],):
        raise ValueError("token_selected_experts must be [num_tokens, top_k].")
    sm90_fp4_moe(
        x.view(torch.uint8),
        x_sf.view(torch.uint8).reshape(tokens, hidden // 16),
        w1,
        w2,
        Sm90Routing(
            ids,
            a["token_final_scales"],
            a["top_k"],
            a["num_experts"],
            a["local_expert_offset"],
            local,
        ),
        Sm90EpilogueSpec(int(activation), alpha=a["w1_alpha"], fq_global=gsi),
        Sm90EpilogueSpec(
            int(ActivationType.Identity),
            gemm2=True,
            out_dtype=output_dtype,
            alpha=a["w2_alpha"],
        ),
        out=out,
        tactic=a["tactic"] if a.get("tactic") is not None else -1,
        recipe=recipe,
        exact=_exact(),
        token_scale=pt,
    )
    return out
