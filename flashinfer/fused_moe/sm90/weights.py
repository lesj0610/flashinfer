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

"""SM90 views of the FP4 weights an SM100+ entry point takes, without a copy.

SM90 has no weight format of its own: each entry point's native SM100+ layout is checked by
shape and dtype and described as a :class:`~.spec.Sm90WeightView` (byte layout, gate/up
pairing, strides). Records (``prepare_records_*``, the SM80-SM89 format) and layouts outside
the SM100+ entry points are rejected.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .spec import AFmt, Pair, Sm90WeightView

_SCALE_KINDS = ("nvfp4", "mxfp4")
_MOE_ENTRIES = ("trtllm", "cutlass", "cute_dsl")
_DENSE_ENTRIES = ("dense_linear", "dense_swz128x4", "dense_trtllm", "dense_cute_sm12x")
_FP8_SF = (torch.float8_e4m3fn, torch.uint8)


def _pad(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _swz_bytes(rows: int, sf_cols: int) -> int:
    # 128x4-swizzled scales of one expert: pad128(rows) x pad4(sf_cols) bytes
    return _pad(rows, 128) * _pad(sf_cols, 4)


def _err(entry: str, msg: str) -> str:
    return f"SM90 FP4 weights ({entry}): {msg}"


def _check_dtype(
    entry: str, name: str, t: torch.Tensor, dtypes: Tuple[torch.dtype, ...]
):
    if t.dtype not in dtypes:
        raise TypeError(
            _err(
                entry,
                f"{name} must be {' or '.join(str(d) for d in dtypes)}, got {t.dtype}",
            )
        )


_RECORDS_HINT = (
    "records from prepare_records_* are not an SM90 format: pass the SM100+ layout"
)


def _check_shape(
    entry: str,
    name: str,
    t: torch.Tensor,
    shape: Tuple[int, ...],
    hint: str = _RECORDS_HINT,
):
    if tuple(t.shape) != tuple(shape):
        raise ValueError(
            _err(entry, f"{name} must be {tuple(shape)}, got {tuple(t.shape)} ({hint})")
        )


def _row_major_2d(
    entry: str, name: str, t: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """[rows, cols] row-major bytes, given as is or as the transpose view an SM100+ GEMM takes."""
    if t.dim() == 2 and tuple(t.shape) == (rows, cols) and t.is_contiguous():
        return t
    if t.dim() == 2 and tuple(t.shape) == (cols, rows) and t.t().is_contiguous():
        return t.t()
    raise ValueError(
        _err(
            entry,
            f"{name} must be [{rows}, {cols}] row-major (or its transpose view), got "
            f"shape {tuple(t.shape)} stride {tuple(t.stride())}",
        )
    )


def _expert_bytes(
    entry: str, name: str, sf: torch.Tensor, experts: int, per: int
) -> torch.Tensor:
    """The [experts, per] bytes of a contiguous scale tensor of ``experts`` leading rows."""
    if sf.dim() < 1 or sf.shape[0] != experts or not sf.is_contiguous():
        raise ValueError(
            _err(
                entry,
                f"{name} must be contiguous with {experts} experts in dim 0, got shape "
                f"{tuple(sf.shape)}",
            )
        )
    if sf.numel() * sf.element_size() != experts * per:
        raise ValueError(
            _err(
                entry,
                f"{name} must hold pad128(rows) * pad4(K / block) = {per} bytes per "
                f"expert, got {sf.numel() * sf.element_size() // max(experts, 1)}",
            )
        )
    return sf.view(torch.uint8).reshape(experts, per)


def _mma_view_bytes(
    sf: torch.Tensor, rows: int, sf_cols: int, experts: int
) -> Optional[torch.Tensor]:
    """CuTe DSL "MMA layout" view of the 128x4 bytes (convert_sf_to_mma_layout), else None."""
    m_tiles, k_tiles = (rows + 127) // 128, (sf_cols + 3) // 4
    shape = (32, 4, m_tiles, 4, k_tiles, experts)
    stride = (16, 4, k_tiles * 512, 1, 512, m_tiles * k_tiles * 512)
    if sf.dim() != 6 or tuple(sf.shape) != shape or tuple(sf.stride()) != stride:
        return None
    per = m_tiles * k_tiles * 512
    return sf.view(torch.uint8).as_strided((experts, per), (per, 1))


def sm90_weight_view(
    entry: str,
    codes: torch.Tensor,
    sf: torch.Tensor,
    *,
    rows: int,
    k: int,
    gated: bool,
    scale_kind: str,
    pair_g: int = 64,
    global_scale: Optional[torch.Tensor] = None,
) -> Sm90WeightView:
    r"""Describe an SM100+ entry point's FP4 weight for the SM90 GEMM (no copy).

    Parameters
    ----------
    entry : str
        Layout owner. MoE (``[E, ...]`` weights):

        - ``"trtllm"``: ``trtllm_fp4_block_scale_moe``; codes uint8 ``[E, rows, k / 2]`` with
          the TRT-LLM row shuffle (gated: w3/w1 interleave first), scales float8_e4m3fn with
          ``pad128(rows) * pad4(k / block)`` bytes per expert (128x4 on the shuffled rows).
        - ``"cutlass"``: ``cutlass_fused_moe`` NVFP4; codes int64 ``[E, rows, k / 16]`` (or
          their uint8 view), gated rows ``[up; gate]``; scales int32
          ``[E, pad128(rows), pad64(k) / 64]`` (128x4).
        - ``"cute_dsl"``: CuTe DSL MoE; codes uint8 ``[E, rows, k / 2]``, gated rows interleaved
          by groups of ``pair_g`` (64, or 16 for ``swap_ab``); scales the
          ``convert_sf_to_mma_layout`` view, or the same bytes contiguous with E in dim 0.

        Dense (one expert):

        - ``"dense_linear"``: codes uint8 ``[rows, k / 2]``, scales linear ``[rows, k / 16]``.
        - ``"dense_swz128x4"``: codes uint8 ``[rows, k / 2]``, scales 128x4
          (``pad128(rows) * pad4(k / block)`` bytes).
        - ``"dense_trtllm"``: as ``dense_swz128x4`` on TRT-LLM shuffled rows.
        - ``"dense_cute_sm12x"``: cute-dsl SM12x prep; codes int32 ``[k / 16, 2 * rows]``,
          scales uint8 ``[k / 16, rows]`` (S0E5M3).

        Dense codes and linear or 128x4 scales may be the transpose views ``mm_fp4`` takes.
    codes, sf : torch.Tensor
        The entry point's own weight tensors.
    rows : int
        Logical rows (output features; gated: both halves).
    k : int
        Logical K (FP4 values per row).
    gated : bool
        GEMM1 of a gated activation (a gate/up pair).
    scale_kind : str
        ``"nvfp4"`` (E4M3 scales of 16 values) or ``"mxfp4"`` (UE8M0 scales of 32 values).
    pair_g : int
        ``"cute_dsl"`` gated group size.
    global_scale : Optional[torch.Tensor]
        Kept on the view for the caller.

    Returns
    -------
    Sm90WeightView
        Byte layout, pairing and strides of ``codes`` and ``sf`` (their uint8 views).
    """
    if entry not in _MOE_ENTRIES + _DENSE_ENTRIES:
        raise ValueError(f"SM90 FP4 weights: unknown entry {entry!r}")
    if scale_kind not in _SCALE_KINDS:
        raise ValueError(
            _err(entry, f"scale_kind must be one of {_SCALE_KINDS}, got {scale_kind!r}")
        )
    if codes.is_meta or sf.is_meta:
        raise ValueError(
            _err(
                entry,
                "weights without data (the shapes-only weights of the SM80-SM89 "
                "records path) are not an SM90 input: pass the SM100+ layout",
            )
        )
    if codes.device != sf.device:
        raise ValueError(_err(entry, f"codes on {codes.device}, scales on {sf.device}"))
    mx = scale_kind == "mxfp4"
    block = 32 if mx else 16
    if rows <= 0 or k <= 0 or k % block:
        raise ValueError(
            _err(entry, f"rows {rows} and k {k} (a multiple of {block}) must be > 0")
        )
    if gated and rows % 2:
        raise ValueError(_err(entry, f"gated rows {rows} must be even"))
    sf_cols = k // block
    half = rows // 2 if gated else 0

    if entry in _MOE_ENTRIES:
        if codes.dim() != 3:
            raise ValueError(
                _err(
                    entry,
                    f"codes must be [num_experts, rows, ...], got {tuple(codes.shape)} "
                    f"({_RECORDS_HINT})",
                )
            )
        experts = codes.shape[0]
        if entry == "trtllm":
            _check_dtype(entry, "codes", codes, (torch.uint8,))
            _check_shape(entry, "codes", codes, (experts, rows, k // 2))
            _check_dtype(entry, "scales", sf, (torch.float8_e4m3fn,))
            if rows % 32:
                raise ValueError(
                    _err(entry, f"rows {rows} must be a multiple of the 32-row shuffle")
                )
            sf_bytes = _expert_bytes(
                entry, "scales", sf, experts, _swz_bytes(rows, sf_cols)
            )
            fmt = AFmt.MXFP4_TRTLLM_SHUFFLED if mx else AFmt.NVFP4_TRTLLM_SHUFFLED
            pair, g = (Pair.TRTLLM_W31 if gated else Pair.NONE), 0
        elif entry == "cutlass":
            if mx:
                raise NotImplementedError(
                    _err(
                        entry,
                        "MXFP4 cutlass_fused_moe weights are not supported on SM90 yet",
                    )
                )
            _check_dtype(entry, "codes", codes, (torch.int64, torch.uint8))
            per_row = k // 16 if codes.dtype == torch.int64 else k // 2
            _check_shape(entry, "codes", codes, (experts, rows, per_row))
            _check_dtype(entry, "scales", sf, (torch.int32,))
            if not sf.is_contiguous():
                raise ValueError(_err(entry, "scales must be contiguous"))
            # The SM100+ kernels step experts by pad128(rows) * pad64(k) / 16 bytes; a non-gated
            # GEMM1 of rows % 128 in (0, 64] (binding shape pad64(rows)) has no such layout.
            _check_shape(
                entry,
                "scales",
                sf,
                (experts, _pad(rows, 128), _pad(k, 64) // 64),
                "experts are pad128(rows) * pad64(k) / 16 bytes apart; pad the rows to 128",
            )
            sf_bytes = sf.view(torch.uint8).reshape(experts, -1)
            fmt, pair, g = (
                AFmt.NVFP4_SWZ128X4,
                (Pair.UP_GATE_HALVES if gated else Pair.NONE),
                0,
            )
        else:  # cute_dsl
            _check_dtype(entry, "codes", codes, (torch.uint8,))
            _check_shape(entry, "codes", codes, (experts, rows, k // 2))
            _check_dtype(entry, "scales", sf, _FP8_SF)
            mma = _mma_view_bytes(sf, rows, sf_cols, experts)
            sf_bytes = (
                mma
                if mma is not None
                else _expert_bytes(
                    entry, "scales", sf, experts, _swz_bytes(rows, sf_cols)
                )
            )
            fmt = AFmt.MXFP4_SWZ128X4 if mx else AFmt.NVFP4_SWZ128X4
            pair, g = Pair.NONE, 0
            if gated:
                if pair_g not in (16, 64) or rows % (2 * pair_g):
                    raise ValueError(
                        _err(
                            entry,
                            f"gated rows {rows} must be a multiple of 2 * pair_g, "
                            f"pair_g 64 or 16 (got {pair_g})",
                        )
                    )
                pair, g = Pair.GROUP_INTERLEAVE, pair_g
        if not codes.is_contiguous():
            raise ValueError(_err(entry, "codes must be contiguous"))
        codes_bytes = codes.view(torch.uint8)
        return Sm90WeightView(
            entry=entry,
            a_fmt=fmt,
            pair=pair,
            codes=codes_bytes,
            sf=sf_bytes,
            num_experts=experts,
            rows=rows,
            rows_phys=rows,
            k=k,
            k_phys=k,
            codes_expert_stride=rows * k // 2,
            sf_expert_stride=sf_bytes.shape[1],
            half_rows=half,
            pair_g=g,
            global_scale=global_scale,
        )

    # dense: one expert, no gate/up pair
    if gated:
        raise ValueError(_err(entry, "dense weights have no gate/up pair"))
    if entry == "dense_cute_sm12x":
        if mx:
            raise ValueError(
                _err(entry, "the SM12x cute-dsl prep holds NVFP4 weights only")
            )
        _check_dtype(entry, "codes", codes, (torch.int32,))
        _check_shape(entry, "codes", codes, (k // 16, 2 * rows))
        _check_dtype(entry, "scales", sf, (torch.uint8,))
        _check_shape(entry, "scales", sf, (k // 16, rows))
        if rows % 64 or not codes.is_contiguous() or not sf.is_contiguous():
            raise ValueError(
                _err(
                    entry, "rows must be a multiple of 64, codes and scales contiguous"
                )
            )
        codes_bytes, sf_bytes, fmt = codes.view(torch.uint8), sf, AFmt.DENSE_CUTE_SM12X
    else:
        _check_dtype(entry, "codes", codes, (torch.uint8,))
        codes_bytes = _row_major_2d(entry, "codes", codes, rows, k // 2)
        _check_dtype(entry, "scales", sf, _FP8_SF)
        if entry == "dense_linear":
            if mx:
                raise ValueError(
                    _err(entry, "linear MXFP4 weight scales are not an SM100+ layout")
                )
            sf_bytes = _row_major_2d(entry, "scales", sf, rows, sf_cols)
            fmt = AFmt.NVFP4_LINEAR_SF
        else:
            if sf.dim() == 2 and not sf.is_contiguous() and sf.t().is_contiguous():
                sf = sf.t()  # the transpose view mm_fp4 takes
            if not sf.is_contiguous():
                raise ValueError(
                    _err(entry, "scales must be contiguous or a transpose view")
                )
            sf_bytes = _expert_bytes(
                entry, "scales", sf.reshape(1, -1), 1, _swz_bytes(rows, sf_cols)
            )
            if entry == "dense_trtllm":
                if rows % 32:
                    raise ValueError(
                        _err(
                            entry,
                            f"rows {rows} must be a multiple of the 32-row shuffle",
                        )
                    )
                fmt = AFmt.MXFP4_TRTLLM_SHUFFLED if mx else AFmt.NVFP4_TRTLLM_SHUFFLED
            else:
                fmt = AFmt.MXFP4_SWZ128X4 if mx else AFmt.NVFP4_SWZ128X4
        sf_bytes = sf_bytes.view(torch.uint8)
    return Sm90WeightView(
        entry=entry,
        a_fmt=fmt,
        pair=Pair.NONE,
        codes=codes_bytes,
        sf=sf_bytes,
        num_experts=1,
        rows=rows,
        rows_phys=rows,
        k=k,
        k_phys=k,
        codes_expert_stride=0,
        sf_expert_stride=0,
        global_scale=global_scale,
    )
