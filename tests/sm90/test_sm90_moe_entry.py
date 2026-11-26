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

SM90 (wgmma) FP4 MoE entry points (``flashinfer.fused_moe.sm90``). The SM100+ MoE entry
points run the NVFP4 x NVFP4 4over6 tuples on SM90 (TRT-LLM and CuTe DSL per token, CUTLASS
per tensor) on their own SM100+ weights, and reject every other tuple on SM90 with an error.

Each running case asserts through the CUDA profiler that the SM90 chain ran and nothing else:
prep, the two ``fp4_sm90_gemm`` launches (GEMM1 and GEMM2), the per-token GEMM2-input
quantization exactly when the activations are per token, finalize exactly when
``do_finalize``, and no kernel of another MoE backend. Accuracy is left to each entry point's
own SM100+ test, which runs these tuples on SM90 too.
"""

import pytest
import torch

import flashinfer.fused_moe as fused_moe
from flashinfer import CuteDslMoEWrapper, cute_dsl_fused_moe, fp4_quantize
from flashinfer.autotuner import autotune
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe import (
    CuteDslConfig,
    CuteDslRunner,
    CutlassNvfp4Config,
    CutlassNvfp4Runner,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RecordsMxfp4Config,
    RecordsNvfp4Bf16Config,
    RecordsNvfp4Config,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SwiGLU,
    SwiGLUStep,
    TrtllmFp4Config,
    TrtllmFp4RoutedRunner,
)
from flashinfer.fused_moe.api import BackendOptions, _RECORDS_NVFP4_ARCHS
from flashinfer.fused_moe.sm90.driver import (
    SM90_FP4_TACTICS,
    default_sm90_fp4_tactic,
    sm90_fp4_moe_supports,
)
from flashinfer.quantization.nvfp4_quantization_utils import (
    NVFP44Over6Config,
    nvfp4_per_token_scale_inv,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import get_compute_capability
from tests.moe import utils as moe_utils
from tests.moe.test_records_nvfp4_moe import _model, _quant_pt, _routing, _trt
from tests.moe.test_trtllm_cutlass_fused_moe import compute_routing
from tests.moe.test_trtllm_gen_per_token_moe import _per_token_routed_case
from tests.moe.test_unified_moe_cutlass import _make_nvfp4_case
from tests.moe.utils import create_moe_tensors
from tests.quantization.test_helpers.utils_fp4 import nvfp4_global_encode_scale_te

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) != (9, 0),
    reason="SM90 FP4 MoE entry points",
)

# Keys on ``use_4over6``: pins every FLASHINFER_NVFP4_4OVER6* variable for the case.
set_nvfp4_4over6_env = moe_utils.set_nvfp4_4over6_env

cute_dsl_required = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuTe DSL is not installed"
)

RECIPE = NVFP44Over6Config()

# Kernels of the SM90 chain (demangled names as the profiler reports them).
GEMM = "fp4_sm90_gemm"
PREP = "prep_detail::prep_"
ROW_QUANT = "rq::row_quant_kernel"
FINALIZE = "fin::finalize_kernel"
OWN = ("fp4_sm90", "gather_token_scale")
# Kernels of the other MoE backends: CUTLASS and CuTe DSL GEMMs, the trtllm-gen batched GEMMs
# and their activation and finalize kernels, and the SM80-SM89 records.
FOREIGN = (
    "cutlass",
    "bmm_",
    "sm100",
    "moe::dev::activation",
    "moe::dev::finalize",
    "w4a16",
)


def _kernel_counts(fn):
    """Run ``fn`` (after one warm-up call that builds the modules) under the CUDA profiler;
    return its result and the launches per kernel name."""
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    counts = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA":
            counts[e.key] = counts.get(e.key, 0) + e.count
    return result, counts


def _launches(counts, symbol):
    return sum(c for name, c in counts.items() if symbol in name)


def _assert_sm90_chain(counts, *, per_token: bool, finalize: bool, what: str):
    names = sorted(counts)
    assert _launches(counts, GEMM) == 2, f"{what}: GEMM1 and GEMM2 expected: {names}"
    assert _launches(counts, PREP), f"{what}: no SM90 prep: {names}"
    assert _launches(counts, ROW_QUANT) == int(per_token), f"{what}: row quant: {names}"
    assert _launches(counts, FINALIZE) == int(finalize), f"{what}: finalize: {names}"
    foreign = [
        n
        for n in names
        if not any(s in n for s in OWN) and any(s in n.lower() for s in FOREIGN)
    ]
    assert not foreign, f"{what}: another MoE backend ran: {foreign}"


def _assert_finite(out, what):
    assert torch.isfinite(out.float()).all(), f"{what}: non-finite output"
    assert out.float().abs().sum() > 0, f"{what}: all-zero output"


# ---------------------------------------------------------------------------
# Host: backend gates and tactics
# ---------------------------------------------------------------------------


def test_backend_gates():
    """SM90 takes the SM100+ configs of the M3 tuples, and no records config."""
    for config in (TrtllmFp4Config, CuteDslConfig, CutlassNvfp4Config):
        assert config.supported(90), config.__name__
    for config in (RecordsNvfp4Bf16Config, RecordsNvfp4Config, RecordsMxfp4Config):
        assert not config.supported(90), config.__name__
    assert 90 not in _RECORDS_NVFP4_ARCHS


@pytest.mark.parametrize(
    "entry,per_token,recipe,act,runs",
    [
        ("trtllm", True, RECIPE, ActivationType.Swiglu, True),
        ("trtllm", True, RECIPE, ActivationType.Relu2, True),
        ("trtllm", True, None, ActivationType.Swiglu, False),
        ("trtllm", False, RECIPE, ActivationType.Swiglu, False),
        ("cute_dsl", True, RECIPE, ActivationType.Swiglu, True),
        ("cute_dsl", True, RECIPE, ActivationType.Relu2, True),
        ("cute_dsl", True, RECIPE, ActivationType.Geglu, False),
        ("cute_dsl", False, RECIPE, ActivationType.Swiglu, False),
        ("cutlass", False, RECIPE, ActivationType.Swiglu, True),
        ("cutlass", False, RECIPE, ActivationType.SwigluStep, True),
        ("cutlass", False, RECIPE, ActivationType.Relu2, True),
        ("cutlass", False, None, ActivationType.Swiglu, False),
        ("cutlass", True, RECIPE, ActivationType.Swiglu, False),
    ],
)
def test_supports(entry, per_token, recipe, act, runs):
    assert (
        sm90_fp4_moe_supports(
            entry, per_token=per_token, nvfp4_4over6=recipe, activation_type=int(act)
        )
        == runs
    )


def test_tactics():
    """D7: CTA_M 128 or 256, n 8 to 256, at most 128 accumulators per thread."""
    assert len(SM90_FP4_TACTICS) == 11
    assert all((m // 128) * n // 2 <= 128 for m, n, _ in SM90_FP4_TACTICS)
    assert (256, 256, 0) not in SM90_FP4_TACTICS
    assert default_sm90_fp4_tactic(1, 4, 32) == (128, 8, 0)
    assert default_sm90_fp4_tactic(1024, 4, 32) == (128, 128, 0)
    assert default_sm90_fp4_tactic(8192, 8, 8) == (128, 256, 0)


# ---------------------------------------------------------------------------
# TRT-LLM flat entry points: NVFP4 x NVFP4 per token, 4over6
# ---------------------------------------------------------------------------


def _trtllm_case(activation_type, num_tokens=64):
    kwargs, _, _ = _per_token_routed_case(
        num_tokens=num_tokens,
        hidden_size=1024,
        intermediate_size=2048,
        num_experts=32,
        top_k=4,
        use_4over6=True,
        weights_use_4over6=False,
        activation_type=activation_type,
        zero_rows=False,
        device=torch.device("cuda:0"),
    )
    return kwargs


def _from_logits(kwargs):
    """The routed call's arguments for ``trtllm_fp4_block_scale_moe`` (Renormalize)."""
    kw = dict(kwargs)
    tokens = kw.pop("topk_ids").shape[0]
    kw["routing_logits"] = torch.randn(
        tokens, kw["num_experts"], device="cuda", dtype=torch.bfloat16
    )
    kw["routing_method_type"] = RoutingMethodType.Renormalize.value
    return kw


@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize(
    "activation_type", [ActivationType.Swiglu, ActivationType.Relu2]
)
@pytest.mark.parametrize("routed", [True, False])
@pytest.mark.parametrize("do_finalize", [True, False])
def test_trtllm_runs_sm90(use_4over6, activation_type, routed, do_finalize):
    kwargs = _trtllm_case(activation_type)
    kwargs["do_finalize"] = do_finalize
    if routed:
        entry = fused_moe.trtllm_fp4_block_scale_routed_moe
    else:
        entry, kwargs = fused_moe.trtllm_fp4_block_scale_moe, _from_logits(kwargs)

    def run():
        with autotune(False):
            return entry(**kwargs, nvfp4_4over6=RECIPE)[0]

    out, counts = _kernel_counts(run)
    what = f"{entry.__name__} {activation_type.name}"
    _assert_sm90_chain(counts, per_token=True, finalize=do_finalize, what=what)
    _assert_finite(out, what)


@pytest.mark.parametrize("use_4over6", [False])
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(dict(nvfp4_4over6=None), id="4over6-off"),
        pytest.param(dict(per_token_scale=None), id="per-tensor"),
        pytest.param("bf16", id="w4a16"),
        pytest.param("gemm1_alpha", id="activation-params"),
    ],
)
def test_trtllm_m6_rejected(use_4over6, change):
    kwargs = _trtllm_case(ActivationType.Swiglu, num_tokens=16)
    if change == "bf16":
        tokens, hidden = kwargs["hidden_states"].shape[0], 1024
        change = dict(
            hidden_states=torch.randn(tokens, hidden, device="cuda").bfloat16(),
            hidden_states_scale=None,
            per_token_scale=None,
        )
    elif change == "gemm1_alpha":
        change = dict(gemm1_alpha=torch.ones(32, device="cuda"))
    for entry, kw in (
        (fused_moe.trtllm_fp4_block_scale_routed_moe, dict(kwargs, **change)),
        (fused_moe.trtllm_fp4_block_scale_moe, _from_logits(dict(kwargs, **change))),
    ):
        with pytest.raises(NotImplementedError, match="SM90"), autotune(False):
            entry(**kw)


# ---------------------------------------------------------------------------
# CUTLASS flat entry point: NVFP4 x NVFP4 per tensor, 4over6
# ---------------------------------------------------------------------------


def _cutlass_case(activation_type, quantized_input):
    """``cutlass_fused_moe`` arguments: SM100 NVFP4 weights (int64 codes, int32 128x4 scales),
    BF16 input or NVFP4 input with its 128x4 scales."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    e, m, n, k, top_k = 8, 64, 256, 512, 2
    rows = 2 * n if activation_type.is_gated else n

    def quantize(w):
        gs = torch.stack(
            [
                nvfp4_global_encode_scale_te(w[i].abs().max().float(), None)
                for i in range(e)
            ]
        )
        q = [fp4_quantize(w[i], gs[i]) for i in range(e)]
        codes = torch.stack([c.view(torch.uint8) for c, _ in q])
        sf = torch.stack([s.view(torch.uint8).reshape(w.shape[1], -1) for _, s in q])
        return codes.view(torch.long), sf.view(torch.int32), gs

    with moe_utils.nvfp4_4over6_env(False):  # caller-quantized weights
        w1, w1_sf, w1_gs = quantize(
            torch.randn(e, rows, k, device=device).bfloat16() / 10
        )
        w2, w2_sf, w2_gs = quantize(torch.randn(e, k, n, device=device).bfloat16() / 10)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    a1_gs = torch.tensor(1.0, device=device)
    a2_gs = torch.tensor(1.0, device=device)
    weights, ids = compute_routing(torch.randn(m, e, device=device), top_k)
    kwargs = dict(
        input=x,
        token_selected_experts=ids.to(torch.int),
        token_final_scales=weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        output_dtype=torch.bfloat16,
        quant_scales=[
            a1_gs,
            w1_sf,
            1 / (a1_gs * w1_gs),
            a2_gs,
            w2_sf,
            1 / (a2_gs * w2_gs),
        ],
        activation_type=activation_type,
    )
    if quantized_input:
        x_q, x_sf = fp4_quantize(x, a1_gs)
        kwargs.update(input=x_q, input_sf=x_sf)
    return kwargs


@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Swiglu, ActivationType.SwigluStep, ActivationType.Relu2],
)
@pytest.mark.parametrize("quantized_input", [False, True])
def test_cutlass_runs_sm90(use_4over6, activation_type, quantized_input):
    kwargs = _cutlass_case(activation_type, quantized_input)

    def run():
        return fused_moe.cutlass_fused_moe(**kwargs, nvfp4_4over6=RECIPE)[0]

    out, counts = _kernel_counts(run)
    what = f"cutlass_fused_moe {activation_type.name}"
    _assert_sm90_chain(counts, per_token=False, finalize=True, what=what)
    _assert_finite(out, what)


@pytest.mark.parametrize("use_4over6", [False])
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(dict(nvfp4_4over6=None), id="4over6-off"),
        pytest.param(dict(nvfp4_4over6=RECIPE, swiglu_alpha=1), id="activation-params"),
    ],
)
def test_cutlass_m6_rejected(use_4over6, change):
    kwargs = _cutlass_case(ActivationType.Swiglu, False)
    if "swiglu_alpha" in change:
        change = dict(change, swiglu_alpha=torch.ones(8, device="cuda"))
    with pytest.raises(NotImplementedError, match="SM90"):
        fused_moe.cutlass_fused_moe(**kwargs, **change)


def test_cutlass_mxfp4_rejected():
    """MXFP4 weights (int64) with FP8 input: M6 on SM90, no records."""
    x = torch.zeros(4, 256, device="cuda").to(torch.float8_e4m3fn)
    w1 = torch.zeros(2, 256, 16, device="cuda", dtype=torch.long)
    w2 = torch.zeros(2, 256, 8, device="cuda", dtype=torch.long)
    ids = torch.zeros(4, 1, device="cuda", dtype=torch.int)
    weights = torch.ones(4, 1, device="cuda")
    with pytest.raises(NotImplementedError, match="SM90"):
        fused_moe.cutlass_fused_moe(x, ids, weights, w1, w2, torch.bfloat16, [])


# ---------------------------------------------------------------------------
# CuTe DSL flat entry points: W4A4 per token, 4over6
# ---------------------------------------------------------------------------


def _cute_case(gated, swap_ab):
    with moe_utils.nvfp4_4over6_env(True):
        t = create_moe_tensors(
            num_tokens=64,
            hidden_size=256,
            intermediate_size=512,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            gated=gated,
            use_per_token_activation=True,
            swap_ab=swap_ab,
        )
    common = dict(
        x=t["x"],
        x_sf=t["x_sf"],
        token_selected_experts=t["token_selected_experts"],
        token_final_scales=t["token_final_scales"],
        w1_weight=t["w1_weight"],
        w1_weight_sf=t["w1_weight_sf"],
        w1_alpha=t["w1_alpha"],
        fc2_input_scale=torch.tensor(
            [nvfp4_per_token_scale_inv(RECIPE)], device="cuda", dtype=torch.float32
        ),
        w2_weight=t["w2_weight"],
        w2_weight_sf=t["w2_weight_sf"],
        w2_alpha=t["w2_alpha"],
        per_token_scale=t["x_per_token_scale"],
    )
    return t, common


@cute_dsl_required
@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize(
    "activation_type", [ActivationType.Swiglu, ActivationType.Relu2]
)
@pytest.mark.parametrize("swap_ab", [False, True])
def test_cute_dsl_runs_sm90(use_4over6, activation_type, swap_ab):
    _, common = _cute_case(activation_type.is_gated, swap_ab)

    def run():
        return cute_dsl_fused_moe(
            **common,
            num_experts=8,
            top_k=2,
            activation_type=activation_type.value,
            swap_ab=swap_ab,
            nvfp4_4over6=RECIPE,
        )

    out, counts = _kernel_counts(run)
    what = f"cute_dsl_fused_moe {activation_type.name} swap_ab={swap_ab}"
    _assert_sm90_chain(counts, per_token=True, finalize=True, what=what)
    _assert_finite(out, what)
    if activation_type == ActivationType.Swiglu:
        wrapper = CuteDslMoEWrapper(
            num_experts=8,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            quant_mode="w4a4",
            swap_ab=swap_ab,
            use_fused_finalize=not swap_ab,
        )
        out, counts = _kernel_counts(lambda: wrapper.run(**common, nvfp4_4over6=RECIPE))
        _assert_sm90_chain(
            counts, per_token=True, finalize=True, what="CuteDslMoEWrapper"
        )
        _assert_finite(out, "CuteDslMoEWrapper")


@cute_dsl_required
@pytest.mark.parametrize("use_4over6", [False])
@pytest.mark.parametrize(
    "mode", ["4over6-off", "per-tensor", "w4a16", "w4a8", "activation-params"]
)
def test_cute_dsl_m6_rejected(use_4over6, mode):
    t, common = _cute_case(True, False)
    extra = dict(nvfp4_4over6=RECIPE)
    if mode == "4over6-off":
        common["fc2_input_scale"] = torch.tensor(
            [nvfp4_per_token_scale_inv(None)], device="cuda", dtype=torch.float32
        )
        extra = dict(nvfp4_4over6=None)
    elif mode == "per-tensor":
        common["per_token_scale"], extra = None, {}
    elif mode == "w4a16":
        common.update(
            x=t["x_bf16"], x_sf=None, fc2_input_scale=None, per_token_scale=None
        )
        extra = dict(quant_mode="w4a16")
    elif mode == "w4a8":
        common.update(fc2_input_scale=None, per_token_scale=None)
        extra = dict(quant_mode="w4a8")
    else:
        extra = dict(extra, swiglu_alpha=1.5)
    with pytest.raises(NotImplementedError, match="SM90"):
        cute_dsl_fused_moe(**common, num_experts=8, top_k=2, **extra)


# ---------------------------------------------------------------------------
# Unified API: the SM100+ runners on SM90
# ---------------------------------------------------------------------------


def _moe_config(
    activation, per_token=True, recipe=RECIPE, pair=None, backend=None, **routing
):
    weight, act = pair or (QuantFormat.NVFP4, QuantFormat.NVFP4)
    kw = {} if backend is None else dict(backend=BackendOptions((backend,)))
    # QuantConfig rejects any nvfp4_4over6 value when the activation is not NVFP4.
    qkw = dict(nvfp4_4over6=recipe) if act is QuantFormat.NVFP4 else {}
    return MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=2, **routing),
        quant=QuantConfig(
            weight=weight, activation=act, per_token_scale=per_token, **qkw
        ),
        experts=ExpertConfig(intermediate_size=512, local_num_experts=8),
        activation=activation,
        **kw,
    )


def _unified_tensors(activation, swap_ab=False):
    with moe_utils.nvfp4_4over6_env(True):
        return create_moe_tensors(
            num_tokens=64,
            hidden_size=1024,
            intermediate_size=512,
            num_experts=8,
            num_local_experts=8,
            top_k=2,
            gated=activation.is_gated,
            use_per_token_activation=True,
            use_nontrivial_alphas=False,
            swap_ab=swap_ab,
        )


def _direct(runner_cls, config):
    runner = runner_cls(config, device=torch.device("cuda:0"))
    runner.check_support()
    runner.build()
    return runner


@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2()], ids=["SwiGLU", "ReLU2"])
@pytest.mark.parametrize(
    "mode",
    [RoutingInputMode.UnpackedPrecomputed, RoutingInputMode.FromLogits],
    ids=["precomputed", "from-logits"],
)
def test_unified_trtllm_runs_sm90(use_4over6, activation, mode):
    t = _unified_tensors(activation)
    from_logits = mode == RoutingInputMode.FromLogits
    routing = dict(method=RoutingMethodType.Renormalize) if from_logits else {}
    runner = _direct(TrtllmFp4RoutedRunner, _moe_config(activation, **routing))
    weights = MoEWeightPack()
    weights.prepare_for(
        runner.backend_key,
        TrtllmFp4Config.prepare_weights(
            t["w1_weight_bf16"],
            t["w2_weight_bf16"],
            num_local_experts=8,
            hidden_size=1024,
            intermediate_size=512,
            activation=activation,
            device=torch.device("cuda:0"),
        ),
    )
    act = dict(
        hidden_states_q=t["x"],
        hidden_states_scale=t["x_sf"].squeeze(-1),
        per_token_scale=t["x_per_token_scale"],
        routing_input_mode=mode,
    )
    if from_logits:
        act["routing_logits"] = torch.randn(64, 8, device="cuda", dtype=torch.bfloat16)
    else:
        act.update(
            topk_ids=t["token_selected_experts"],
            topk_weights=t["token_final_scales"].to(torch.bfloat16),
        )
    inputs = runner.pack_inputs(MoEActivationPack(**act), weights)
    out, counts = _kernel_counts(lambda: runner.forward(inputs, tactic=-1))
    what = f"TrtllmFp4RoutedRunner {type(activation).__name__} {mode.name}"
    _assert_sm90_chain(counts, per_token=True, finalize=True, what=what)
    _assert_finite(out, what)


@cute_dsl_required
@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2()], ids=["SwiGLU", "ReLU2"])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_unified_cute_dsl_runs_sm90(use_4over6, activation, swap_ab):
    t = _unified_tensors(activation, swap_ab)
    config = _moe_config(activation, backend=CuteDslConfig(swap_ab=swap_ab))
    runner = _direct(CuteDslRunner, config)
    weights = MoEWeightPack()
    weights.prepare_for(
        runner.backend_key,
        CuteDslConfig.prepare_weights(
            t["w1_weight_bf16"],
            t["w2_weight_bf16"],
            quant=config.quant,
            num_local_experts=8,
            hidden_size=1024,
            intermediate_size=512,
            activation=activation,
            device=torch.device("cuda:0"),
            nvfp4_4over6=RECIPE,
            swap_ab=swap_ab,
        ),
    )
    act = MoEActivationPack(
        t["x"],
        t["x_sf"].squeeze(-1),
        t["token_selected_experts"],
        t["token_final_scales"],
        per_token_scale=t["x_per_token_scale"],
    )
    inputs = runner.pack_inputs(act, weights)
    out, counts = _kernel_counts(lambda: runner.forward(inputs, tactic=-1))
    what = f"CuteDslRunner {type(activation).__name__} swap_ab={swap_ab}"
    _assert_sm90_chain(counts, per_token=True, finalize=True, what=what)
    _assert_finite(out, what)


@pytest.mark.parametrize("use_4over6", [True])
@pytest.mark.parametrize(
    "activation",
    [SwiGLU(), SwiGLUStep(), ReLU2()],
    ids=["SwiGLU", "SwiGLUStep", "ReLU2"],
)
def test_unified_cutlass_runs_sm90(use_4over6, activation):
    import dataclasses

    config, act, weights, _ = _make_nvfp4_case(num_tokens=16, activation=activation)
    config = dataclasses.replace(
        config, quant=dataclasses.replace(config.quant, nvfp4_4over6=RECIPE)
    )
    layer = MoELayer(config)
    assert isinstance(layer.runners[0], CutlassNvfp4Runner)
    out, counts = _kernel_counts(lambda: layer(act, weights))
    what = f"CutlassNvfp4Runner {type(activation).__name__}"
    _assert_sm90_chain(counts, per_token=False, finalize=True, what=what)
    _assert_finite(out, what)


@pytest.mark.parametrize(
    "runner_cls,config",
    [
        (TrtllmFp4RoutedRunner, _moe_config(SwiGLU(), recipe=None)),
        (TrtllmFp4RoutedRunner, _moe_config(SwiGLU(), per_token=False, recipe=None)),
        (TrtllmFp4RoutedRunner, _moe_config(SwiGLUStep())),
        (
            TrtllmFp4RoutedRunner,
            _moe_config(
                SwiGLU(),
                per_token=False,
                pair=(QuantFormat.MXFP4, QuantFormat.MXFP8),
            ),
        ),
        (CuteDslRunner, _moe_config(SwiGLU(), recipe=None)),
        (CuteDslRunner, _moe_config(SwiGLU(), per_token=False, recipe=None)),
        (
            CuteDslRunner,
            _moe_config(
                SwiGLU(),
                per_token=False,
                pair=(QuantFormat.NVFP4, QuantFormat.BF16),
            ),
        ),
        (CutlassNvfp4Runner, _moe_config(SwiGLU(), per_token=False, recipe=None)),
        (CutlassNvfp4Runner, _moe_config(SwiGLU(alpha=1.5), per_token=False)),
    ],
)
def test_unified_m6_rejected(runner_cls, config):
    runner = runner_cls(config, device=torch.device("cuda:0"))
    with pytest.raises(NotImplementedError, match="SM90"):
        runner.check_support()


# ---------------------------------------------------------------------------
# Other SM100+ MoE entry points SM90 does not run, and records input
# ---------------------------------------------------------------------------


def test_alphamoe_routed_rejected():
    t = torch.zeros(1, device="cuda")
    with pytest.raises(NotImplementedError, match="SM90"):
        fused_moe.alphamoe_nvfp4_routed_moe(*([t] * 16), 1)


@cute_dsl_required
def test_grouped_gemm_nt_masked_rejected():
    from flashinfer.gemm import grouped_gemm_nt_masked

    t = torch.zeros(1, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="SM100"):
        grouped_gemm_nt_masked(
            (t, t),
            (t, t),
            t,
            t,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
        )


@pytest.mark.parametrize("entry", ["trtllm", "cutlass", "cute_dsl"])
def test_records_input_rejected(entry):
    """Records (``prepare_records_nvfp4_weights``, the SM80-SM89 format) are not an SM90
    input: the M3 tuple on records fails in the SM90 weight view instead of running."""
    if entry == "cute_dsl" and not is_cute_dsl_available():
        pytest.skip("CuTe DSL is not installed")
    m = _model(True)
    tokens, experts, top_k = 16, m["r1"].shape[0], 2
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    ids, weights = _routing(tokens, top_k, experts)
    x_q, x_sf, pt, _ = _quant_pt(x, RECIPE)
    ones = torch.ones(experts, device="cuda")
    fc2_in = torch.tensor([nvfp4_per_token_scale_inv(RECIPE)], device="cuda")
    with pytest.raises((TypeError, ValueError), match="SM90 FP4 weights"):
        if entry == "trtllm":
            _trt(
                m,
                x_q,
                x_sf,
                ids,
                weights,
                ones,
                ones,
                ones,
                per_token_scale=pt,
                nvfp4_4over6=RECIPE,
            )
        elif entry == "cutlass":
            one = torch.tensor(1.0, device="cuda")
            fused_moe.cutlass_fused_moe(
                x, ids, weights, m["w1"].view(torch.long), m["w2"].view(torch.long),
                torch.bfloat16, [one, m["r1"], ones, one, m["r2"], ones],
                activation_type=m["act"], nvfp4_4over6=RECIPE,
            )  # fmt: skip
        else:
            cute_dsl_fused_moe(
                x_q, x_sf, ids, weights, m["w1"], m["r1"], ones, fc2_in, m["w2"], m["r2"],
                ones, experts, top_k, per_token_scale=pt, nvfp4_4over6=RECIPE,
            )  # fmt: skip


def test_fc2_input_scale_checked_in_graph_capture():
    """The SM90 row quantization uses the recipe's GEMM2-input scale, not the tensor: a tensor
    (or tensor version) not read before a CUDA graph capture raises instead of passing
    unchecked, and NaN fails the check."""
    from flashinfer.fused_moe.sm90.driver import _check_fc2_input_scale

    good = nvfp4_per_token_scale_inv(RECIPE)
    warmed = torch.tensor([good], device="cuda", dtype=torch.float32)
    _check_fc2_input_scale(warmed, RECIPE)  # warmup outside capture reads and checks it
    fresh = torch.tensor([2 * good], device="cuda", dtype=torch.float32)
    mutated = torch.tensor([good], device="cuda", dtype=torch.float32)
    _check_fc2_input_scale(mutated, RECIPE)
    mutated.mul_(2)  # new version, never read on the host
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    errors = []
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _check_fc2_input_scale(warmed, RECIPE)  # same tensor and version: cached value
        for t in (fresh, mutated):
            try:
                _check_fc2_input_scale(t, RECIPE)
            except RuntimeError as e:
                errors.append(str(e))
    assert len(errors) == 2 and all("before capturing" in e for e in errors), errors
    with pytest.raises(NotImplementedError, match="fc2_input_scale"):
        _check_fc2_input_scale(
            torch.tensor([float("nan")], device="cuda", dtype=torch.float32), RECIPE
        )
