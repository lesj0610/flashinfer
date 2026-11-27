# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The dense GEMM on SM8x: the SM8x FP8 or NVFP4 GEMM to FP32, then the kernel's
epilogue (``kernel.py``) on that accumulator in the same order."""

from __future__ import annotations

from typing import Optional

import torch

from ...gemm.sm8x_fp4_dense import _flat_bytes, _ones, _run_fp4
from ...gemm.sm8x_fp8_dense import _run_fp8
from .config import PrimsTsGemmConfig

_FP8 = torch.float8_e4m3fn
_HEAD = 128
# RN-even, satfinite E2M1: the midpoints between the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(_FP8)


def _e2m1(x: torch.Tensor) -> torch.Tensor:
    mag = x.abs()
    lo = sum((mag > t).to(torch.uint8) for t in _E2M1_MIDS)
    hi = sum((mag >= t).to(torch.uint8) for t in _E2M1_MIDS)
    code = torch.where(lo % 2 == 0, lo, hi)  # a tie takes the even code
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def _store_nvfp4(v: torch.Tensor, c: torch.Tensor, sf_c: torch.Tensor, scale_c) -> None:
    """One E4M3 scale per 16 values, encoded against the stored scale; the low nibble
    holds the even element and the scales go to the 128x4 swizzle (rows < M only)."""
    m, n = v.shape
    blocks = v.view(m, n // 16, 16)
    sf = (blocks.abs().amax(-1) * scale_c * (1.0 / 6.0)).clamp(max=448.0).to(_FP8)
    codes = _e2m1(blocks * (scale_c / sf.float().clamp(min=1e-12))[..., None])
    c.view(m, n // 2).copy_((codes[..., 0::2] | (codes[..., 1::2] << 4)).view(m, -1))
    padded = -(-(n // 16) // 4) * 4
    rows = torch.arange(m, device=v.device)[:, None]
    cols = torch.arange(n // 16, device=v.device)[None, :]
    idx = (
        cols % 4
        + cols // 4 * 512
        + rows % 32 * 16
        + rows % 128 // 32 * 4
        + rows // 128 * 128 * padded
    )
    _flat_bytes(sf_c)[idx] = sf.view(torch.uint8)


def _qknorm_rope(
    v: torch.Tensor,
    rounded: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Q and K RMSNorm per 128-column head of the output-rounded values, then the
    interleaved-pair RoPE at each row's position (outside ``[0, M)``: 0); V as is."""
    m, n = v.shape
    qk = 2 * (n // 3)
    heads = rounded[:, :qk].reshape(m, -1, _HEAD)
    norm = torch.rsqrt(heads.square().sum(-1, keepdim=True) / _HEAD + 1e-6)
    weight = torch.stack((q_norm, k_norm)).float().repeat_interleave(n // 3 // _HEAD, 0)
    x = heads * norm * weight
    pos = torch.where((positions < 0) | (positions >= m), 0, positions)
    cos, sin = cos_sin[pos].view(m, 1, 2, _HEAD // 2).unbind(2)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
    return torch.cat((rotated.view(m, qk), v[:, qk:]), 1)


def run(
    config: PrimsTsGemmConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    mnk: tuple[int, int, int],
    bias: Optional[torch.Tensor],
    scale: float,
    x_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    q_norm: Optional[torch.Tensor],
    k_norm: Optional[torch.Tensor],
    cos_sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    sfa: Optional[torch.Tensor],
    sfb: Optional[torch.Tensor],
    sf_c: Optional[torch.Tensor],
    scale_c: Optional[torch.Tensor],
    scale_gate: Optional[torch.Tensor],
    qkv_scale: Optional[torch.Tensor],
) -> None:
    """The compiled launcher's arguments, in its order."""
    m, n, _ = mnk
    if m == 0:
        return
    one = _ones(a.device)
    fp8 = config.operand_format == "fp8_e4m3"
    acc = torch.empty(
        (1, m, n) if fp8 else (m, n), dtype=torch.float32, device=a.device
    )
    if fp8:
        _run_fp8(a[None], b.t()[None], one, one, acc, 64, False)
        v = acc[0] * x_scale[:, None] * weight_scale
        if bias is not None:
            v += bias.float()
    else:
        _run_fp4(a, b.t(), sfa, sfb, one, acc, 64)
        v = acc if bias is None else acc + bias.float()
    if qkv_scale is not None:
        v = (v.view(m, 3, -1) * qkv_scale.view(1, 3, 1)).view(m, n)
    out_fmt = config.output_format
    if not fp8 and out_fmt != "nvfp4_e2m1" and scale_gate is None and qkv_scale is None:
        v = v * scale
    if config.epilogue == "qkv_qknorm_rope":
        rounded = (v.to(torch.bfloat16) if out_fmt == "bf16" else _to_fp8(v)).float()
        v = _qknorm_rope(v, rounded, q_norm, k_norm, cos_sin, positions)
    elif config.epilogue == "swiglu":
        gate, act = v[:, 0::2], v[:, 1::2]
        if scale_gate is None:
            v = gate * (act * torch.sigmoid(act))
        else:
            v = (gate * scale_gate) * (act * torch.sigmoid(act * scale_gate))
    if out_fmt == "nvfp4_e2m1":
        _store_nvfp4(v, c, sf_c, scale if scale_gate is None else scale_c)
        return
    if scale_gate is not None:
        v = v * scale_c
    v = v.to(torch.bfloat16) if out_fmt == "bf16" else _to_fp8(v)
    if config.separate_qkv_output:
        v = v.view(m, 3, -1).transpose(0, 1)
    c.view(v.shape).copy_(v)
