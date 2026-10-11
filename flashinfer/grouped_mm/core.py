"""Grouped Matrix Multiplication for Mixture-of-Experts (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each expert
in a Mixture-of-Experts layer has its own weight matrix and tokens are routed
to experts via ``m_indptr``.

This module is a backend-agnostic facade: it owns the public ``grouped_mm_*``
entry points, argument validation, and ``backend`` dispatch.  Backend-specific
implementations live in sibling subpackages (e.g. :mod:`.cudnn`).  Adding a
new backend means adding a new subpackage and wiring an extra ``elif`` branch
in the dispatch sections below.

Public APIs: ``grouped_mm_bf16``, ``grouped_mm_fp8``, ``grouped_mm_mxfp8``,
``grouped_mm_fp4``.
"""

from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..utils import (
    backend_requirement,
    get_compute_capability,
    supported_compute_capability,
)
from .cudnn import (
    _CUDNN_MOE_MIN_VERSION,
    _check_cudnn_version,
    _cudnn_moe_block_scale_min_version,
    _run_cudnn_moe_block_scale_grouped_gemm_fp4,
    _run_cudnn_moe_block_scale_grouped_gemm_mxfp8,
    _run_cudnn_moe_grouped_gemm,
)

# =========================================================================
# grouped_mm_bf16
# =========================================================================


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 107, 110, 120, 121])
def _check_grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(f"a and b must be bfloat16, got {a.dtype} and {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_bf16)
@flashinfer_api
def grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with BF16/FP16 data types (cuDNN MOE backend, or the experimental Cake backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_bf16` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, bf16 or fp16.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  ``"cudnn"`` (default), or the experimental
        ``"cake"`` backend (:mod:`flashinfer.experimental.cake_moe_grouped_gemm`,
        SM100 / SM103 / SM107): generated ragged BF16 programs that read
        ``m_indptr`` on the device, need no padding, produce bfloat16 outputs
        only and require ``n % 256 == 0`` and ``k % 64 == 0``.  Naming it is the
        opt-in and emits an ``ExperimentalWarning`` once.
    tactic : int
        cuDNN execution-plan index.  ``-1`` (default) uses the heuristic-best
        plan; non-negative values select a specific plan.  The Cake backend has
        a single program per architecture and accepts only ``-1``.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Examples
    --------
    >>> import torch, flashinfer
    >>> E, tpe, k, n = 8, 128, 4096, 2048
    >>> a = torch.randn(E * tpe, k, dtype=torch.bfloat16, device="cuda")
    >>> b = torch.randn(E, n, k, dtype=torch.bfloat16, device="cuda")
    >>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
    >>> out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr)
    >>> out_cake = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")
    """
    if backend == "cake":
        from ..experimental.cake_moe_grouped_gemm.cake_backend import (
            grouped_mm_bf16_cake,
        )

        return grouped_mm_bf16_cake(
            a, b, m_indptr, out=out, out_dtype=out_dtype, tactic=tactic
        )

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, "grouped_mm_bf16")
        return _run_cudnn_moe_grouped_gemm(
            a, b, m_indptr, out_dtype=out_dtype, out=out, tactic=tactic
        )
    else:
        raise ValueError("backend does not support grouped_mm_bf16")


# =========================================================================
# grouped_mm_fp8
# =========================================================================


@supported_compute_capability([89, 90, 100, 103, 107, 110, 120, 121])
def _check_grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"a must be float8_e4m3fn or float8_e5m2, got {a.dtype}")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"b must be float8_e4m3fn or float8_e5m2, got {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if alpha is not None:
        if alpha.dtype != torch.float32:
            raise ValueError(f"alpha must be float32, got {alpha.dtype}")
        if alpha.shape != (1,):
            raise ValueError(f"alpha must be a scalar, got {alpha.shape}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_fp8)
@flashinfer_api
def grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with FP8 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_fp8` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, e4m3 or e5m2.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    alpha : Optional[torch.Tensor]
        Scaling factor for the output, shape ``(1,)``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.
    tactic : int
        cuDNN execution plan index.  ``-1`` (default) uses the heuristic-best
        plan.  Non-negative values select a specific plan.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Examples
    --------
    >>> import torch, flashinfer
    >>> E, tpe, k, n = 8, 128, 4096, 2048
    >>> a = torch.randn(E * tpe, k, dtype=torch.float8_e4m3, device="cuda")
    >>> b = torch.randn(E, n, k, dtype=torch.float8_e4m3, device="cuda")
    >>> m_indptr = (torch.arange(E + 1, device="cuda") * tpe).to(torch.int32)
    >>> out = flashinfer.grouped_mm.grouped_mm_fp8(a, b, m_indptr, alpha=torch.tensor(1.0, device="cuda"))
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        _check_cudnn_version(_CUDNN_MOE_MIN_VERSION, "grouped_mm_fp8")
        return _run_cudnn_moe_grouped_gemm(
            a,
            b,
            m_indptr,
            alpha=alpha,
            out_dtype=out_dtype,
            out=out,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp8")


# =========================================================================
# grouped_mm_mxfp8
# =========================================================================


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 107, 110, 120, 121])
def _check_grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"a must be float8_e4m3fn or float8_e5m2, got {a.dtype}")
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"b must be float8_e4m3fn or float8_e5m2, got {b.dtype}")
    if a_descale.dtype != torch.uint8:
        raise ValueError(f"a_descale must be uint8, got {a_descale.dtype}")
    if b_descale.dtype != torch.uint8:
        raise ValueError(f"b_descale must be uint8, got {b_descale.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if a_descale.ndim != 2:
        raise ValueError(
            f"a_descale must be 2D (cum_m, k // 32), got {a_descale.shape}"
        )
    if b_descale.ndim != 3:
        raise ValueError(
            f"b_descale must be 3D (num_experts, n, k // 32), got {b_descale.shape}"
        )
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    return True


@backend_requirement({}, common_check=_check_grouped_mm_mxfp8)
@flashinfer_api
def grouped_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with MXFP8 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_mxfp8` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, e4m3 or e5m2.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``, e4m3 or e5m2.
    a_descale : torch.Tensor
        Block scale tensor for A. Can be:
        - 2D swizzled 128x4: shape (cum_m, k // 32)
        dtype: uint8.
    b_descale : torch.Tensor
        Block scale tensor for B. Can be:
        - 3D swizzled 128x4: shape (batch_size, n, k // 32)
        dtype: uint8.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.
    tactic : int
        cuDNN execution-plan index.  ``-1`` (default) uses the heuristic-best
        plan; non-negative values select a specific plan.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Below SM100 the operands are dequantized to BF16 and multiplied by
    :func:`grouped_mm_bf16` in FP32, with each row's largest scale taken out
    before the GEMM and put back after. This is exact unless a block's scale is
    more than ``2**124`` below its row's largest one. There are no cuDNN plans
    to pick from, so any ``tactic >= 0`` returns ``None``.
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        major, minor = get_compute_capability(a.device)
        if major < 10:
            if tactic >= 0:
                return None
            num_experts, n = b.shape[:2]
            a_bf16, a_exp = _scale_rows_to_bf16(
                a.bfloat16(), a_descale, 1, a.shape[0], 32
            )
            b_bf16, b_exp = _scale_rows_to_bf16(
                b.bfloat16(), b_descale, num_experts, n, 32
            )
            return _dequantized_grouped_mm(
                a_bf16, b_bf16, a_exp, b_exp, m_indptr, None, out, out_dtype
            )
        _check_cudnn_version(
            _cudnn_moe_block_scale_min_version(major * 10 + minor), "grouped_mm_mxfp8"
        )
        return _run_cudnn_moe_block_scale_grouped_gemm_mxfp8(
            a,
            b,
            a_descale,
            b_descale,
            m_indptr,
            out_dtype=out_dtype,
            out=out,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_mxfp8")


# =========================================================================
# grouped_mm_fp4
# =========================================================================


def _e2m1_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """BF16 values of packed E2M1 bytes, low nibble first."""
    # BF16 E2M1 pair of every byte, built on the device so CUDA graph capture
    # needs no host copy.
    m = torch.arange(8, device=x.device)
    e, f = m >> 1, (m & 1).float()
    mag = torch.where(e == 0, f * 0.5, torch.exp2((e - 1).float()) * (1 + f * 0.5))
    lut = torch.cat((mag, -mag)).bfloat16()
    byte = torch.arange(256, device=x.device)
    pairs = torch.stack((lut[byte & 0xF], lut[byte >> 4]), dim=-1)
    codes = x.view(torch.uint8).reshape(-1).int()  # index_select keeps int32
    return pairs.index_select(0, codes)


def _scale_rows_to_bf16(
    values: torch.Tensor,
    descale: torch.Tensor,
    batch: int,
    rows: int,
    block_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """BF16 ``[batch * rows, k]`` of ``values`` times 128x4-swizzled block scales.

    E4M3 scales (``block_size=16``) are applied exactly, with ``None`` row
    exponents. E8M0 scales (``block_size=32``) are applied relative to the
    largest scale of each row, which keeps every value in BF16 range, and that
    scale's exponent is returned per row. The values stay exact unless a block's
    scale is more than ``2**124`` below its row's largest one; a 255 (NaN)
    scale gives NaN.
    """
    from ..trace.templates.gemm import _unswizzle_batched_sf_128x4

    out = values.view(batch * rows, -1, block_size)
    scale = _unswizzle_batched_sf_128x4(
        descale.reshape(-1), batch, rows, out.shape[1]
    ).reshape(batch * rows, -1)
    if block_size == 16:
        scale = scale.view(torch.float8_e4m3fn).bfloat16()
        return out.mul_(scale.unsqueeze(-1)).flatten(1), None
    exp = scale.view(torch.uint8).int()
    nan = exp == 255
    row_exp = exp.masked_fill(nan, 0).amax(dim=1, keepdim=True)
    rel = torch.exp2((exp - row_exp).float()).masked_fill(nan, float("nan"))
    return out.mul_(rel.bfloat16().unsqueeze(-1)).flatten(1), row_exp.squeeze(1) - 127


def _dequantized_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_exp: Optional[torch.Tensor],
    b_exp: Optional[torch.Tensor],
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FP32 :func:`grouped_mm_bf16` of :func:`_scale_rows_to_bf16` operands."""
    num_experts = m_indptr.shape[0] - 1
    result = grouped_mm_bf16(
        a, b.view(num_experts, -1, a.shape[1]), m_indptr, out_dtype=torch.float32
    )
    if a_exp is not None:
        rows = torch.arange(result.shape[0], dtype=torch.int32, device=a.device)
        expert = torch.searchsorted(m_indptr[1:], rows, right=True)
        exp = (
            a_exp[:, None]
            + b_exp.view(num_experts, -1)[expert.clamp_(max=num_experts - 1)]
        )
        # Two factors within FP32 range restore row exponents from -254 to 254.
        result.mul_(torch.exp2((exp // 2).float()))
        result.mul_(torch.exp2((exp - exp // 2).float()))
    if alpha is not None:
        result.mul_(alpha)
    return result.to(out_dtype) if out is None else out.copy_(result)


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 107, 110, 120, 121])
def _check_grouped_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
):
    if a.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise ValueError(f"a must be float4_e2m1fn_x2 or uint8, got {a.dtype}")
    if b.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise ValueError(f"b must be float4_e2m1fn_x2 or uint8, got {b.dtype}")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"out_dtype must be float16, bfloat16, or float32, got {out_dtype}"
        )
    if a.ndim != 2:
        raise ValueError(f"a must be 2D (cum_m, k), got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"b must be 3D (num_experts, n, k), got {b.shape}")
    if a_descale.ndim != 2:
        raise ValueError(
            f"a_descale must be 2D (cum_m, k // 32), got {a_descale.shape}"
        )
    if b_descale.ndim != 3:
        raise ValueError(
            f"b_descale must be 3D (num_experts, n, k // 32), got {b_descale.shape}"
        )
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be int32, got {m_indptr.dtype}")
    num_experts = b.shape[0]
    if m_indptr.shape[0] != num_experts + 1:
        raise ValueError(
            f"m_indptr length {m_indptr.shape[0]} != num_experts+1 ({num_experts + 1})"
        )
    if a.shape[1] != b.shape[2]:
        raise ValueError(f"K mismatch: a has {a.shape[1]}, b has {b.shape[2]}")
    if alpha is not None:
        if alpha.dtype != torch.float32:
            raise ValueError(f"alpha must be float32, got {alpha.dtype}")
        if alpha.shape != (1,):
            raise ValueError(f"alpha must be a scalar, got {alpha.shape}")
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {expected_shape}"
            )
        if out.device != a.device:
            raise ValueError(
                f"out device {out.device} must match input device {a.device}"
            )
    if block_size == 16:  # nvfp4
        if (
            a_descale.dtype != torch.float8_e4m3fn
            or b_descale.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(
                f"a_descale and b_descale must be float8_e4m3fn, got {a_descale.dtype} and {b_descale.dtype}"
            )
    elif block_size == 32:  # mxfp4
        if a_descale.dtype != torch.uint8 or b_descale.dtype != torch.uint8:
            raise ValueError(
                f"a_descale and b_descale must be uint8, got {a_descale.dtype} and {b_descale.dtype}"
            )
    else:
        raise ValueError(f"block_size must be 16 or 32, got {block_size}")

    return True


@backend_requirement({}, common_check=_check_grouped_mm_fp4)
@flashinfer_api
def grouped_mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
    *,
    backend: str = "cudnn",
    tactic: int = -1,
) -> torch.Tensor:
    r"""Grouped matrix multiplication with FP4 data types (cuDNN MOE backend).

    Performs a grouped GEMM across experts, as used in Mixture-of-Experts layers.
    Mirrors :func:`flashinfer.mm_fp4` but for expert-partitioned inputs.

    .. math::

        \text{out}[\text{start}:\text{end}] = a[\text{start}:\text{end}] \times b[e]^T
        \quad \text{for each expert } e

    where ``start, end = m_indptr[e], m_indptr[e+1]``.

    Parameters
    ----------
    a : torch.Tensor
        Token activations, shape ``(cum_m, k)``, fp4 e2m1fn_x2 or uint8.
    b : torch.Tensor
        Expert weights, shape ``(batch_size, n, k)``, fp4 e2m1fn_x2 or uint8.
    a_descale : torch.Tensor
        Block scale tensor for A. Can be:
        - 2D swizzled 128x4: shape (cum_m, k // block_size)
        dtype: float8_e4m3fn or uint8.
    b_descale : torch.Tensor
        Block scale tensor for B. Can be:
        - 3D swizzled 128x4: shape (batch_size, n, k // block_size)
        dtype: float8_e4m3fn or uint8.
    m_indptr : torch.Tensor
        Cumulative token counts, shape ``(batch_size + 1,)``, ``int32``.
    alpha : Optional[torch.Tensor]
        Scaling factor for the output, shape ``(1,)``.
    out : Optional[torch.Tensor]
        Pre-allocated output ``(m_out, n)``.
    out_dtype : torch.dtype
        Output data type.  ``torch.bfloat16`` (default) or ``torch.float16``, ``torch.float32``.
    block_size : int
        Block size used for the FP4 scale layout.  ``16`` selects NVFP4 (with
        ``float8_e4m3fn`` scales) and ``32`` selects MXFP4 (with ``uint8``
        scales).  Defaults to ``16``.
    backend : str
        Backend selector.  Currently only ``"cudnn"`` is supported.
    tactic : int
        cuDNN execution-plan index.  ``-1`` (default) uses the heuristic-best
        plan; non-negative values select a specific plan.

    Returns
    -------
    torch.Tensor
        Output tensor ``(m_out, n)``.

    Below SM100 the operands are dequantized to BF16 and multiplied by
    :func:`grouped_mm_bf16` in FP32. NVFP4 is exact. MXFP4 takes each row's
    largest scale out before the GEMM and puts it back after, so it is exact
    unless a block's scale is more than ``2**124`` below its row's largest one.
    There are no cuDNN plans to pick from, so any ``tactic >= 0`` returns
    ``None`` like an out-of-range index.
    """

    if out is not None:
        out_dtype = out.dtype

    if backend == "cudnn":
        major, minor = get_compute_capability(a.device)
        if major < 10:
            if tactic >= 0:
                return None
            num_experts, n = b.shape[:2]
            a_bf16, a_exp = _scale_rows_to_bf16(
                _e2m1_to_bf16(a), a_descale, 1, a.shape[0], block_size
            )
            b_bf16, b_exp = _scale_rows_to_bf16(
                _e2m1_to_bf16(b), b_descale, num_experts, n, block_size
            )
            return _dequantized_grouped_mm(
                a_bf16, b_bf16, a_exp, b_exp, m_indptr, alpha, out, out_dtype
            )
        _check_cudnn_version(
            _cudnn_moe_block_scale_min_version(major * 10 + minor), "grouped_mm_fp4"
        )
        return _run_cudnn_moe_block_scale_grouped_gemm_fp4(
            a,
            b,
            a_descale,
            b_descale,
            m_indptr,
            alpha=alpha,
            out_dtype=out_dtype,
            out=out,
            block_size=block_size,
            tactic=tactic,
        )
    else:
        raise ValueError("backend does not support grouped_mm_fp4")
