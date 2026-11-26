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

SM90 FP4 weight dequantization (``csrc/fp4_sm90/dequant.cuh``), bit-exact against a CPU
reference: every E2M1 code times every scale byte, for each scale format (E4M3, UE8M0,
S0E5M3), each operand type (BF16, FP16) and both NVFP4 variants (DQ 0 "fold7", DQ 1).

The kernel output is value * 2^-fold (``dq_fold_log2``); the reference rounds the exact product
E2M1 * scale * 2^-fold to the operand type (round to nearest even). NaN matches any NaN,
signed zeros must match.

Scale semantics (PTX / OCP MX): E4M3 is signed (0x80 = -0.0, 0x7F / 0xFF NaN); UE8M0 e is
2^(e - 127) for every e in [0, 254] (e = 0 is 2^-127, e in {253, 254} overflow to inf after the
E2M1 product where the operand type saturates) and 255 is NaN; S0E5M3 is FP16 with bits
byte << 7 (0xF8 inf, 0xF9 - 0xFF NaN). The UE8M0 bytes 253 and 254 are also reported on their
own line.
"""

import functools

import pytest
import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec, sm90a_nvcc_flags
from flashinfer.jit.utils import write_if_different
from flashinfer.utils import get_compute_capability

_SOURCE = r"""
#include "fp4_sm90/dequant.cuh"
#include "tvm_ffi_utils.h"

namespace fs = flashinfer::fp4_sm90;
using fs::SfKind;

// Variant v: 0 / 1 NVFP4 DQ 0 / 1 BF16, 2 / 3 NVFP4 DQ 0 / 1 FP16, 4 / 5 MXFP4 BF16 / FP16,
// 6 / 7 S0E5M3 BF16 / FP16.
template <int V>
__device__ __forceinline__ void dq_variant(uint32_t w8, uint32_t sf2, uint32_t (&a)[4]) {
  if constexpr (V == 0) fs::dq_nvfp4<0, false>(w8, uint16_t(sf2), a);
  if constexpr (V == 1) fs::dq_nvfp4<1, false>(w8, uint16_t(sf2), a);
  if constexpr (V == 2) fs::dq_nvfp4<0, true>(w8, uint16_t(sf2), a);
  if constexpr (V == 3) fs::dq_nvfp4<1, true>(w8, uint16_t(sf2), a);
  if constexpr (V == 4) fs::dq_mxfp4<false>(w8, uint16_t(sf2), a);
  if constexpr (V == 5) fs::dq_mxfp4<true>(w8, uint16_t(sf2), a);
  if constexpr (V == 6) fs::dq_s0e5m3<false>(w8, uint16_t(sf2), a);
  if constexpr (V == 7) fs::dq_s0e5m3<true>(w8, uint16_t(sf2), a);
}

template <int V>
__global__ void dq_kernel(const uint32_t* w8, const uint32_t* sf2, uint32_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a[4];
  dq_variant<V>(w8[i], sf2[i], a);
#pragma unroll
  for (int r = 0; r < 4; ++r) out[4 * i + r] = a[r];
}

constexpr int kFold[8] = {
    fs::dq_fold_log2<SfKind::kE4M3, false, 0>(),  fs::dq_fold_log2<SfKind::kE4M3, false, 1>(),
    fs::dq_fold_log2<SfKind::kE4M3, true, 0>(),   fs::dq_fold_log2<SfKind::kE4M3, true, 1>(),
    fs::dq_fold_log2<SfKind::kUE8M0, false, 0>(), fs::dq_fold_log2<SfKind::kUE8M0, true, 0>(),
    fs::dq_fold_log2<SfKind::kS0E5M3, false, 0>(), fs::dq_fold_log2<SfKind::kS0E5M3, true, 0>()};

int64_t fold(int64_t variant) {
  TVM_FFI_ICHECK(variant >= 0 && variant < 8);
  return kFold[variant];
}

void run(int64_t variant, TensorView w8, TensorView sf2, TensorView out) {
  const int n = static_cast<int>(w8.size(0));
  const auto* w = static_cast<const uint32_t*>(w8.data_ptr());
  const auto* s = static_cast<const uint32_t*>(sf2.data_ptr());
  auto* o = static_cast<uint32_t*>(out.data_ptr());
  ffi::CUDADeviceGuard guard(w8.device().device_id);
  const cudaStream_t stream = get_stream(w8.device());
  const dim3 grid((n + 255) / 256), block(256);
  switch (variant) {
    case 0: dq_kernel<0><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 1: dq_kernel<1><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 2: dq_kernel<2><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 3: dq_kernel<3><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 4: dq_kernel<4><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 5: dq_kernel<5><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 6: dq_kernel<6><<<grid, block, 0, stream>>>(w, s, o, n); break;
    case 7: dq_kernel<7><<<grid, block, 0, stream>>>(w, s, o, n); break;
    default: TVM_FFI_ICHECK(false) << "unknown variant " << variant;
  }
  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fold, fold);
"""

# (name, scale kind, FP16 operands, expected fold)
_VARIANTS = [
    ("nvfp4_dq0_bf16", "e4m3", False, 7),
    ("nvfp4_dq1_bf16", "e4m3", False, 0),
    ("nvfp4_dq0_f16", "e4m3", True, 7),
    ("nvfp4_dq1_f16", "e4m3", True, 0),
    ("mxfp4_bf16", "ue8m0", False, 0),
    ("mxfp4_f16", "ue8m0", True, 0),
    ("s0e5m3_bf16", "s0e5m3", False, 15),
    ("s0e5m3_f16", "s0e5m3", True, 6),
]

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 += [-v for v in _E2M1]  # code 8 is -0.0

# Register r of the A fragment holds (low, high) = nibbles (j, j + 4) of W8 and the scale of row
# g (low byte of the pair) or g + 8 (high byte): a0 <- j 0, a1 <- j 2, a2 <- j 1, a3 <- j 3.
_REG_NIBBLE = [0, 2, 1, 3]
_REG_SCALE_BYTE = [0, 1, 0, 1]


def _device() -> torch.device:
    return torch.device("cuda:0")


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(_device()) != (9, 0):
        pytest.skip("the SM90 FP4 GEMM dequantization needs SM90 (sm_90a)")


@functools.cache
def _module():
    name = "test_sm90_fp4_dequant"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "dequant_test.cu"
    write_if_different(source, _SOURCE)
    return gen_jit_spec(
        name, [source], extra_cuda_cflags=sm90a_nvcc_flags
    ).build_and_load()


@pytest.fixture(scope="module")
def mod():
    _require_sm90()
    yield _module()
    print("WGEMU-TEST-END")


def _scale_table(kind: str) -> torch.Tensor:
    """float64 value of every scale byte."""
    b = torch.arange(256, dtype=torch.int64)
    if kind == "e4m3":
        return (
            b.to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .to(torch.float64)
        )
    if kind == "ue8m0":
        v = torch.pow(2.0, (b - 127).to(torch.float64))
        v[255] = float("nan")
        return v
    return (b << 7).to(torch.int16).view(torch.float16).to(torch.float64)


def _cases():
    """(w8, sf2): every code in all 8 nibbles, plus two nibble-placement words, per scale byte.
    The second row's scale byte is b ^ 0x5A so both scale lanes see different values."""
    words = [c * 0x11111111 for c in range(16)] + [0x76543210, 0xFEDCBA98]
    w8, sf2 = [], []
    for b in range(256):
        for w in words:
            w8.append(w)
            sf2.append(b | ((b ^ 0x5A) << 8))
    return torch.tensor(w8, dtype=torch.int64), torch.tensor(sf2, dtype=torch.int64)


def _to_i32(x: torch.Tensor) -> torch.Tensor:
    return (((x & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000).to(torch.int32)


def _reference(w8, sf2, kind, f16, fold):
    """[n, 4, 2] int16 bit patterns of the expected registers (low half first)."""
    e2m1 = torch.tensor(_E2M1, dtype=torch.float64)
    scale = _scale_table(kind)
    dtype = torch.float16 if f16 else torch.bfloat16
    vals = torch.empty(w8.numel(), 4, 2, dtype=torch.float64)
    for r in range(4):
        j = _REG_NIBBLE[r]
        s = scale[(sf2 >> (8 * _REG_SCALE_BYTE[r])) & 0xFF]
        vals[:, r, 0] = e2m1[(w8 >> (4 * j)) & 0xF] * s
        vals[:, r, 1] = e2m1[(w8 >> (4 * (j + 4))) & 0xF] * s
    # The products are exact in float64; scaling by 2^-fold is exact; to() rounds once.
    return (vals * 2.0**-fold).to(dtype).view(torch.int16)


def _mismatch(got: torch.Tensor, ref: torch.Tensor, f16: bool) -> torch.Tensor:
    dtype = torch.float16 if f16 else torch.bfloat16
    both_nan = torch.isnan(got.view(dtype)) & torch.isnan(ref.view(dtype))
    return (got != ref) & ~both_nan


@pytest.mark.parametrize(
    "variant", range(len(_VARIANTS)), ids=[v[0] for v in _VARIANTS]
)
def test_dequant_exact(mod, variant):
    name, kind, f16, fold = _VARIANTS[variant]
    assert mod.fold(variant) == fold, (
        f"{name}: dq_fold_log2 {mod.fold(variant)} != {fold}"
    )
    w8, sf2 = _cases()
    out = torch.empty(w8.numel(), 4, dtype=torch.int32, device=_device())
    mod.run(variant, _to_i32(w8).to(_device()), _to_i32(sf2).to(_device()), out)
    torch.cuda.synchronize()
    got = out.cpu().view(torch.int16).view(-1, 4, 2)
    ref = _reference(w8, sf2, kind, f16, fold)
    bad = _mismatch(got, ref, f16)
    bad_case = bad.flatten(1).any(dim=1)
    print(
        f"WGEMU-TEST name=sm90.dequant.{name} cases={w8.numel()} "
        f"mismatches={int(bad_case.sum())}"
    )
    for i in torch.nonzero(bad_case).flatten()[:4].tolist():
        print(
            f"WGEMU-MISMATCH name=sm90.dequant.{name} w8=0x{int(w8[i]):08x} "
            f"sf2=0x{int(sf2[i]):04x} got={got[i].flatten().tolist()} "
            f"ref={ref[i].flatten().tolist()}"
        )
    if kind == "ue8m0":
        high = (((sf2 & 0xFF) >= 253) & ((sf2 & 0xFF) <= 254)) | (
            ((sf2 >> 8) >= 253) & ((sf2 >> 8) <= 254)
        )
        print(
            f"WGEMU-TEST name=sm90.dequant.{name}.e253_254 cases={int(high.sum())} "
            f"mismatches={int((bad_case & high).sum())}"
        )
    assert not bool(bad_case.any()), f"{name}: {int(bad_case.sum())} mismatching cases"
