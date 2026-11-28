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
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from .utils import write_if_different

#: Weight formats (``fp4_sm90::AFmt`` names) a dense module is built for. The TRTLLM-shuffled
#: MXFP4 format is a MoE layout and has no dense entry point.
FP4_DENSE_SM90_A_FMTS = {
    "nvfp4_trtllm_shuffled": 0,
    "nvfp4_swz128x4": 1,
    "nvfp4_linear_sf": 2,
    "mxfp4_swz128x4": 4,
    "dense_cute_sm12x": 5,
}

#: GEMM instantiation parts of ``csrc/fp4_dense_sm90.cu`` (D7 shape x APath), one source each
#: so that the wgmma kernels compile in parallel. Must match ``kNumParts`` there.
FP4_DENSE_SM90_PARTS = 22


def gen_fp4_dense_sm90_module(f16: bool, a_fmt: str) -> JitSpec:
    """SM90 (wgmma) dense FP4 GEMM (``csrc/fp4_dense_sm90.cu``), sm_90a only.

    One module per weight (wgmma A operand) format ``a_fmt`` (a name of
    ``FP4_DENSE_SM90_A_FMTS``) and per ``f16``, the wgmma B operand type (FP16 instead of
    BF16; MXFP4 weights are BF16 only, their E8M0 scales overflow FP16). The output type is a
    runtime argument. The FP4 activation dequantization of the prep kernel runs in software
    (FLASHINFER_FP4_SW_CVT, as the SM90 quantizers).
    """
    if a_fmt not in FP4_DENSE_SM90_A_FMTS:
        raise ValueError(
            f"unknown SM90 dense FP4 weight format {a_fmt!r}; "
            f"expected one of {sorted(FP4_DENSE_SM90_A_FMTS)}"
        )
    if f16 and a_fmt.startswith("mxfp4"):
        raise ValueError(
            "SM90 dense MXFP4 weights need the BF16 wgmma operand (E8M0 scales overflow FP16)"
        )
    name = f"fp4_dense_sm90_{a_fmt}" + ("_f16" if f16 else "")
    defines = [
        "ENABLE_BF16",
        "ENABLE_FP8",
        "ENABLE_FP4",
        "FLASHINFER_FP4_SW_CVT",
        "FP4_SM90_RECIPE=0",  # no in-prep quantization: one (unused) quantizer is compiled
        "FLASHINFER_FP4_SM90_EXACT=0",
        "FP4_SM90_ACT=9",  # ActivationType.Identity: the epilogue skips the run-time switch
        f"FP4_DENSE_SM90_AFMT={FP4_DENSE_SM90_A_FMTS[a_fmt]}",
        f"FP4_DENSE_SM90_F16={int(f16)}",
    ]
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / name
    parts = []
    for part in range(FP4_DENSE_SM90_PARTS):
        path = gen_directory / f"fp4_dense_sm90_part{part}.cu"
        write_if_different(
            path, f'#define FP4_DENSE_SM90_PART {part}\n#include "fp4_dense_sm90.cu"\n'
        )
        parts.append(path)
    nv_internal = jit_env.FLASHINFER_CSRC_DIR / "nv_internal"
    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / "fp4_dense_sm90.cu", *parts],
        extra_cuda_cflags=sm90a_nvcc_flags + [f"-D{d}" for d in defines],
        extra_include_paths=[nv_internal, nv_internal / "include"],
    )
