"""CPU-only tests of the SM90 FP4 MoE entry points' input checks.

Each entry point of ``flashinfer.fused_moe.sm90`` checks its inputs as the SM100 op does, in the
same order and with the same exception type and wording, before it launches anything. These
tests reach only those checks, with CPU tensors, so they run without a GPU.

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
"""

import pytest
import torch

from flashinfer.fused_moe import core
from flashinfer.fused_moe.cute_dsl.fused_moe import _check_clamped_relu2_supported
from flashinfer.fused_moe.sm90 import driver
from flashinfer.tllm_enums import (
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
    ActivationType,
)

_TOKENS, _HIDDEN, _TOP_K = 4, 128, 2


def _trtllm_args(**overrides):
    """``trtllm_fp4_sm90_moe``'s arguments up to its output check (CPU tensors)."""
    ones = torch.ones(8, dtype=torch.float32)
    args = dict(
        hidden_states=torch.zeros(_TOKENS, _HIDDEN // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(_TOKENS * (_HIDDEN // 16), dtype=torch.uint8),
        per_token_scale=None,
        activation_type=int(ActivationType.Relu2),
        output1_scale_scalar=ones,
        output1_scale_gate_scalar=ones,
        output2_scale_scalar=ones,
        nvfp4_4over6=None,
        gemm1_weights=torch.zeros(8, 256, _HIDDEN // 2, dtype=torch.uint8),
        intermediate_size=256,
        local_num_experts=8,
        num_experts=8,
        do_finalize=True,
        output=None,
    )
    args.update(overrides)
    return args


def _trtllm_call(**overrides):
    ids = torch.zeros(_TOKENS, _TOP_K, dtype=torch.int32)
    weights = torch.ones(_TOKENS, _TOP_K, dtype=torch.bfloat16)
    return driver.trtllm_fp4_sm90_moe(_trtllm_args(**overrides), ids, weights)


def _cute_dsl_args(**overrides):
    """``cute_dsl_fp4_sm90_moe``'s arguments (CPU tensors), per tensor."""
    args = dict(
        x=torch.zeros(_TOKENS, _HIDDEN // 2, dtype=torch.uint8),
        x_sf=torch.zeros(_TOKENS * (_HIDDEN // 16), dtype=torch.uint8),
        per_token_scale=None,
        nvfp4_4over6=None,
        swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT,
        output_dtype=torch.bfloat16,
        quant_mode="w4a4",
        fc2_input_scale=torch.ones(1, dtype=torch.float32),
        situ_beta=None,
        clamped_relu2_limit=None,
        routing_weights_before_fc2=False,
        localized_weights=None,
    )
    args.update(overrides)
    return args


class TestLocalizedWeights:
    def test_routing_metadata_is_rejected_first(self, monkeypatch):
        """core.py's check of the SM100 op comes before every other localized check."""
        monkeypatch.setattr(
            core, "_flatten_localized_moe_args", lambda *_: ([torch.zeros(1)], [1], 1)
        )
        with pytest.raises(
            ValueError, match="localized_weights does not support routing_metadata"
        ):
            driver._check_localized(
                {"routing_metadata": object(), "workspace_buffer": object()},
                torch.zeros(1, dtype=torch.uint8),
            )


class TestTrtllmOutput:
    """``output`` of a finalized call, checked as core.py's SM100 op does."""

    def test_dtype(self):
        output = torch.zeros(_TOKENS, _HIDDEN, dtype=torch.float16)
        with pytest.raises(ValueError, match="Invalid dtype of output"):
            _trtllm_call(output=output)

    def test_rows(self):
        output = torch.zeros(_TOKENS + 1, _HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(
            AssertionError,
            match=rf"output.shape\[0\]={_TOKENS + 1} must be equal to {_TOKENS}",
        ):
            _trtllm_call(output=output)

    def test_columns(self):
        output = torch.zeros(_TOKENS, _HIDDEN + 1, dtype=torch.bfloat16)
        with pytest.raises(
            AssertionError,
            match=rf"output.shape\[1\]={_HIDDEN + 1} must be less than or equal to "
            rf"{_HIDDEN}",
        ):
            _trtllm_call(output=output)

    def test_dtype_is_checked_before_rows(self):
        output = torch.zeros(_TOKENS + 1, _HIDDEN, dtype=torch.float16)
        with pytest.raises(ValueError, match="Invalid dtype of output"):
            _trtllm_call(output=output)


class TestTrtllmIgnoredActivationParameters:
    """Non-gated activations ignore gemm1_alpha, gemm1_beta and gemm1_clamp_limit."""

    @pytest.mark.parametrize(
        "activation", [ActivationType.Relu2, ActivationType.Identity]
    )
    @pytest.mark.parametrize("name", ["gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"])
    def test_not_rejected_as_unsupported(self, activation, name):
        # The call stops at the output check, past the check that rejects parameters.
        output = torch.zeros(_TOKENS, _HIDDEN, dtype=torch.float16)
        with pytest.raises(ValueError, match="Invalid dtype of output"):
            _trtllm_call(
                activation_type=int(activation),
                output=output,
                **{name: torch.ones(8, dtype=torch.float32)},
            )

    @pytest.mark.parametrize("name", ["gemm1_bias", "gemm2_bias"])
    def test_biases_are_unchanged(self, name):
        output = torch.zeros(_TOKENS, _HIDDEN, dtype=torch.float16)
        with pytest.raises(ValueError, match="Invalid dtype of output"):
            _trtllm_call(output=output, **{name: torch.ones(8, 256)})

    def test_other_unsupported_input_is_still_rejected(self):
        with pytest.raises(NotImplementedError, match="routing_metadata"):
            _trtllm_call(routing_metadata=object())


class TestCuteDslW4A4Checks:
    """W4A4 inputs, checked in SM100's order (TypeError, then the two ValueErrors)."""

    def _call(self, **overrides):
        return driver.cute_dsl_fp4_sm90_moe(
            _cute_dsl_args(**overrides), ActivationType.Swiglu, False
        )

    def test_packed_input(self):
        x = torch.zeros(_TOKENS, _HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(
            TypeError, match="quant_mode='w4a4' requires packed uint8 input"
        ):
            self._call(x=x)

    def test_input_scale(self):
        with pytest.raises(ValueError, match="x_sf is required for block-scaled MoE"):
            self._call(x_sf=None)

    def test_fc2_input_scale(self):
        with pytest.raises(
            ValueError, match="fc2_input_scale is required when quant_mode='w4a4'"
        ):
            self._call(fc2_input_scale=None)

    def test_order(self):
        x = torch.zeros(_TOKENS, _HIDDEN, dtype=torch.bfloat16)
        with pytest.raises(TypeError):
            self._call(x=x, x_sf=None, fc2_input_scale=None)
        with pytest.raises(ValueError, match="x_sf is required"):
            self._call(x_sf=None, fc2_input_scale=None)

    @pytest.mark.parametrize(
        "per_token", [False, True], ids=["per-tensor", "per-token"]
    )
    def test_empty_fc2_input_scale(self, per_token):
        """An empty scale fails in ``reshape(1)``, as SM100's per-token quantizer's does
        (SM100's per-tensor path has no check of its own)."""
        pt = torch.ones(_TOKENS, dtype=torch.float32) if per_token else None
        with pytest.raises(
            RuntimeError, match=r"shape '\[1\]' is invalid for input of size 0"
        ):
            self._call(fc2_input_scale=torch.ones(0), per_token_scale=pt)

    @pytest.mark.parametrize(
        "scale",
        [1.0, torch.ones(1, dtype=torch.bfloat16), torch.ones(1, dtype=torch.float64)],
        ids=["float", "bf16", "fp64"],
    )
    def test_per_tensor_fc2_input_scale_type(self, scale):
        """Per tensor, SM100 reads a float32 pointer: no float, no other dtype."""
        with pytest.raises(TypeError, match="must be a float32 tensor"):
            self._call(fc2_input_scale=scale)

    def test_per_tensor_fc2_input_scale_device(self):
        with pytest.raises(ValueError, match="must be on the input device"):
            self._call(fc2_input_scale=torch.ones(1, device="meta"))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
    def test_intermediate_dtype(self, dtype):
        with pytest.raises(
            ValueError,
            match="supports only torch.float16 and torch.bfloat16 intermediate dtypes",
        ):
            self._call(output_dtype=dtype)

    @pytest.mark.parametrize("name", ["situ_beta", "clamped_relu2_limit"])
    def test_unsupported_params_are_not_implemented(self, name):
        with pytest.raises(NotImplementedError, match="SM90 FP4 MoE"):
            self._call(**{name: 1.0})


_INTER, _EXPERTS = 256, 8
_LOG2E = 1.4426950408889634


def _w4a16_args(**overrides):
    """``cute_dsl_fp4_sm90_moe``'s arguments for ``quant_mode="w4a16"`` (CPU tensors): BF16 ``x``
    and no activation scales, up to the checks the CPU reaches (before the weight views)."""
    args = _cute_dsl_args(
        x=torch.zeros(_TOKENS, _HIDDEN, dtype=torch.bfloat16),
        x_sf=None,
        fc2_input_scale=None,
        quant_mode="w4a16",
        situ_linear_beta=None,
        token_selected_experts=torch.zeros(_TOKENS, _TOP_K, dtype=torch.int32),
        token_final_scales=torch.ones(_TOKENS, _TOP_K, dtype=torch.float32),
        w1_weight=torch.zeros(_EXPERTS, 2 * _INTER, _HIDDEN // 2, dtype=torch.uint8),
        w2_weight=torch.zeros(_EXPERTS, _HIDDEN, _INTER // 2, dtype=torch.uint8),
        moe_output=None,
        num_experts=_EXPERTS,
        num_local_experts=None,
        local_expert_offset=0,
        tactic=None,
        use_fused_finalize=True,
    )
    args.update(overrides)
    return args


class TestCuteDslW4A16Checks:
    """W4A16 inputs, checked in SM100's order and wording (``cute_dsl_fused_moe``'s)."""

    def _call(self, activation=ActivationType.Swiglu, **overrides):
        return driver.cute_dsl_fp4_sm90_moe(_w4a16_args(**overrides), activation, False)

    def test_localized_weights(self):
        with pytest.raises(
            NotImplementedError,
            match="locality-domain localization is implemented for the W4A4 path only",
        ):
            self._call(localized_weights=object())

    @pytest.mark.parametrize(
        "name,value",
        [
            ("x_sf", torch.zeros(_TOKENS, dtype=torch.uint8)),
            ("fc2_input_scale", torch.ones(1, dtype=torch.float32)),
            ("per_token_scale", torch.ones(_TOKENS, dtype=torch.float32)),
        ],
    )
    def test_activation_scales(self, name, value):
        with pytest.raises(
            ValueError,
            match="x_sf, fc2_input_scale, and per_token_scale must be None "
            "when quant_mode='w4a16'",
        ):
            self._call(**{name: value})

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtype(self, dtype):
        with pytest.raises(ValueError, match="W4A16 only supports BF16 output"):
            self._call(output_dtype=dtype)

    def test_fp16_input(self):
        with pytest.raises(TypeError, match="W4A16 requires x.dtype=torch.bfloat16"):
            self._call(x=torch.zeros(_TOKENS, _HIDDEN, dtype=torch.float16))

    def test_packed_input(self):
        with pytest.raises(TypeError, match="W4A16 requires x.dtype=torch.bfloat16"):
            self._call(x=torch.zeros(_TOKENS, _HIDDEN // 2, dtype=torch.uint8))

    def test_token_final_scales_dtype(self):
        with pytest.raises(
            TypeError, match="W4A16 requires token_final_scales.dtype=torch.float32"
        ):
            self._call(
                token_final_scales=torch.ones(_TOKENS, _TOP_K, dtype=torch.bfloat16)
            )

    def test_input_shape(self):
        with pytest.raises(ValueError, match="x must have shape"):
            self._call(x=torch.zeros(_TOKENS, _HIDDEN // 2, dtype=torch.bfloat16))

    def test_output_shape(self):
        with pytest.raises(ValueError, match="moe_output must have shape"):
            self._call(
                moe_output=torch.zeros(_TOKENS, _HIDDEN // 2, dtype=torch.bfloat16)
            )

    @pytest.mark.parametrize(
        "activation,rows",
        [
            (ActivationType.Swiglu, _INTER),
            (ActivationType.GegluTanh, _INTER),
            (ActivationType.Relu2, 2 * _INTER),
        ],
    )
    def test_gemm1_rows(self, activation, rows):
        """The last check the CPU reaches: GEMM1's rows follow the activation's gating."""
        w1 = torch.zeros(_EXPERTS, rows, _HIDDEN // 2, dtype=torch.uint8)
        with pytest.raises(ValueError, match="w1_weight dim 1 must be"):
            self._call(activation, w1_weight=w1)

    def test_order(self):
        """localization, scales, output dtype, ``x`` dtype, then the routing weights' dtype."""
        bad = dict(
            localized_weights=object(),
            x_sf=torch.zeros(1, dtype=torch.uint8),
            output_dtype=torch.float16,
            x=torch.zeros(_TOKENS, _HIDDEN, dtype=torch.float16),
            token_final_scales=torch.ones(_TOKENS, _TOP_K, dtype=torch.bfloat16),
        )
        for name, error in [
            ("localized_weights", NotImplementedError),
            ("x_sf", ValueError),
            ("output_dtype", ValueError),
            ("x", TypeError),
            ("token_final_scales", TypeError),
        ]:
            with pytest.raises(error):
                self._call(**bad)
            del bad[name]

    def test_clamped_relu2(self):
        """The hook's rejection, ahead of every check above (SM100's place and wording)."""
        with pytest.raises(
            ValueError,
            match="ActivationType.ClampedRelu2 is not supported when quant_mode='w4a16'",
        ):
            _check_clamped_relu2_supported(
                ActivationType.ClampedRelu2, "w4a16", torch.device("cpu")
            )

    def test_clamped_relu2_is_not_a_w4a16_activation(self):
        assert not driver.sm90_fp4_moe_supports(
            "cute_dsl",
            per_token=False,
            activation_type=int(ActivationType.ClampedRelu2),
            w4a16=True,
        )


class TestW4A16Constants:
    """``Sm90W4A16``: SM100's FP32 activation constants (host double) and flags."""

    def _epilogue(self, activation=ActivationType.Swiglu, **overrides):
        args = _w4a16_args(**overrides)
        return driver._w4a16_epilogue(activation, args)

    def test_swiglu_default(self):
        """The plain SwiGLU reads no constants."""
        act, w4a16 = self._epilogue()
        assert act == ActivationType.Swiglu and w4a16 == driver.Sm90W4A16(None, 0)

    @pytest.mark.parametrize(
        "name,value",
        [("swiglu_alpha", 1.702), ("swiglu_beta", 1.0), ("swiglu_limit", 7.0)],
    )
    def test_swiglu_parameters(self, name, value):
        act, w4a16 = self._epilogue(**{name: value})
        assert act == ActivationType.Swiglu
        assert w4a16.flags == driver.W4A16_FLAG_SWIGLU_PARAMS
        alpha = value if name == "swiglu_alpha" else DEFAULT_SWIGLU_ALPHA
        assert w4a16.consts[0] == -alpha * _LOG2E

    @pytest.mark.parametrize(
        "activation", [ActivationType.GegluTanh, ActivationType.Relu2]
    )
    def test_no_constants(self, activation):
        act, w4a16 = self._epilogue(activation, swiglu_alpha=1.702)
        assert act == activation
        assert w4a16 == driver.Sm90W4A16(None, 0)

    def test_situ(self):
        act, w4a16 = self._epilogue(situ_beta=4.0, swiglu_alpha=1.702)
        assert (
            act == ActivationType.Situ
        )  # SiTU is Swiglu with situ_beta; swiglu_* unread
        assert w4a16 == driver.Sm90W4A16((4.0, 1.0 / 4.0, 0.0, 0.0), 0)

    def test_situ_linear_beta(self):
        act, w4a16 = self._epilogue(situ_beta=4.0, situ_linear_beta=25.0)
        assert act == ActivationType.Situ
        assert w4a16 == driver.Sm90W4A16(
            (4.0, 1.0 / 4.0, 25.0, 1.0 / 25.0), driver.W4A16_FLAG_SITU_LINEAR_BETA
        )

    @pytest.mark.parametrize("use_fused_finalize", [False, True])
    def test_fused_finalize_flag(self, monkeypatch, use_fused_finalize):
        """``use_fused_finalize`` (SM100's default) sets flag bit 2 beside the activation's."""
        seen = {}
        monkeypatch.setattr(driver, "sm90_weight_view", lambda *_, **__: None)
        monkeypatch.setattr(driver, "sm90_fp4_moe", lambda *_, **kw: seen.update(kw))
        args = _w4a16_args(
            use_fused_finalize=use_fused_finalize,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            w1_weight_sf=None,
            w2_weight_sf=None,
            w1_alpha=None,
            w2_alpha=None,
        )
        driver._cute_dsl_w4a16_sm90_moe(args, ActivationType.Swiglu, False)
        fused = driver.W4A16_FLAG_FUSED_FINALIZE if use_fused_finalize else 0
        assert seen["w4a16"] == driver.Sm90W4A16(
            (4.0, 1.0 / 4.0, 25.0, 1.0 / 25.0),
            driver.W4A16_FLAG_SITU_LINEAR_BETA | fused,
        )

    def test_constants_tensor(self):
        t = driver._w4a16_consts_tensor((-_LOG2E, 0.0, 7.0, 0.0))
        assert t.device.type == "cpu" and t.dtype == torch.float32
        assert t.shape == (4,) and t.is_contiguous()
        assert t[0].item() == torch.tensor(-_LOG2E, dtype=torch.float32).item()

    @pytest.mark.parametrize(
        "activation,expected",
        [
            (ActivationType.Swiglu, True),
            (ActivationType.Situ, True),
            (ActivationType.GegluTanh, True),
            (ActivationType.Relu2, True),
            (ActivationType.ClampedRelu2, False),
            (ActivationType.Geglu, False),
        ],
    )
    def test_supports(self, activation, expected):
        for entry in ("trtllm", "cutlass", "cute_dsl"):
            assert driver.sm90_fp4_moe_supports(
                entry, per_token=False, activation_type=int(activation), w4a16=True
            ) is (expected and entry == "cute_dsl")
