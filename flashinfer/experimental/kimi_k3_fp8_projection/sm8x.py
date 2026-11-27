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

The Kimi-K3 FP8 projection on SM8x: the SM100 route, plan and launch sequence as they
are, each planned launch run on FlashInfer's SM8x FP8 GEMM. The quantization launch
writes the programs' activation contract into the workspace (per row and 128 K
elements: the absolute maximum clamped to 1e-4, the UE8M0 ceiling of its quotient by
448 as the FP32 scale -- kept at the start of ``workspace.sf`` -- and the E4M3 product
with the scale's exact inverse in ``workspace.q``); the GEMM / decode launch runs the
groupwise SM8x GEMM on it against a row-major copy of the prepared weight (one per
prepared weight) and its FP32 UE8M0 scales; the fused decode launch does both. The
launcher's receipts are host records naming the planned launch; the scratch of a
workspace (per ``workspace.q`` region, so disjoint workspaces carved from one
allocation never share it) lives as long as the workspace, so a captured graph keeps
it.
"""

from __future__ import annotations

import itertools
import weakref

import torch
from torch.utils.weak import WeakIdKeyDictionary

from .cake_backend import _REBIND_SOURCES, _current_raw_stream, _device_index

# The SM8x GEMM's row tile.
_ROWS = 64
# A planned receipt starts with this int64 and its registry id.
_MAGIC = 0x384D_5338_4B33_4650
_RECEIPT_MIN_BYTES = 64
_IDS = itertools.count(1)
# id -> (program, stage) of every live planned receipt (dropped with the receipt)
_RECEIPTS: dict[int, tuple[Sm8xProgram, Sm8xStage]] = {}
# workspace storage -> {(workspace.q address, name, shape, dtype): buffer}
_SCRATCH: WeakIdKeyDictionary = WeakIdKeyDictionary()
# prepared weight -> (row-major weight bytes, FP32 weight scales)
_WEIGHTS: WeakIdKeyDictionary = WeakIdKeyDictionary()
_WORKSPACE_Q = ("a_q", "A", "X")
_WORKSPACE_SF = ("a_sf", "SFA", "SFX")


def _scratch(q, name, shape, dtype) -> torch.Tensor:
    """The buffer ``name`` of the workspace whose activation is ``q``: one per
    ``q`` region of its storage, as concurrent runs on disjoint workspaces need."""
    storage = q.untyped_storage()
    buffers = _SCRATCH.get(storage)
    if buffers is None:
        buffers = _SCRATCH[storage] = {}
    key = (q.data_ptr(), name, shape, dtype)
    buffer = buffers.get(key)
    if buffer is None:
        buffer = buffers[key] = torch.empty(shape, dtype=dtype, device=q.device)
    return buffer


def _weight(prepared) -> tuple[torch.Tensor, torch.Tensor]:
    weight = _WEIGHTS.get(prepared)
    if weight is None:
        weight = _WEIGHTS[prepared] = (
            prepared.weight_q.contiguous().view(torch.uint8).reshape(-1),
            prepared.weight_scale_ue8m0.unsqueeze(0),
        )
    return weight


class Sm8xStage:
    """Launch ``index`` of ``plan`` on SM8x, bound to the workspace ``(q, sf)``: the
    quantization of ``x`` (the first of two launches), the GEMM / decode into ``out``
    (the second) or both (a single fused launch). Its scratch is made here."""

    def __init__(self, prepared, plan, index, q, sf, M, ldo, out_ptr):
        # FlashInfer's SM8x GEMM: the package's only FlashInfer imports, taken on SM8x only.
        from flashinfer.fused_moe.core import _w4a16_moe_sm80_module
        from flashinfer.gemm.sm8x_fp4_dense import _ones, _pad
        from flashinfer.gemm.sm8x_fp8_dense import _FMT, _I32_MAX

        fused = plan.quant_units is None
        self.quant, self.gemm = fused or index == 0, fused or index > 0
        K, n, device = int(prepared.K), int(prepared.n_valid), q.device
        nka = K // 128
        self.shape, self.ldo, self.out_ptr = (M, K, n), ldo, out_ptr
        self.q = q.view(M, nka, 128)
        self.gsa = sf[: M * nka * 4].view(torch.float32).view(M, nka)
        if self.quant:
            self.lo, self.hi = (
                _scratch(q, name, (M, nka), torch.bfloat16) for name in ("lo", "hi")
            )
            self.t = _scratch(q, "t", (M, nka), torch.float32)
        if not self.gemm:
            return
        self.w, self.gsb = _weight(prepared)
        self.ones = _ones(device)
        self.module = _w4a16_moe_sm80_module(None, False, None, False)
        self.fmt = _FMT[torch.float8_e4m3fn]
        # the GEMM writes rows of n columns: out itself when its rows are, else staged
        self.direct = n % 8 == 0 and (ldo == n or M == 1) and out_ptr % 16 == 0
        width, kp = _pad(n, 8), _pad(K, 32)
        self.target = None
        if not self.direct:
            self.target = _scratch(q, "target", (M, width), torch.bfloat16)
        # the GEMM's activation and output offsets are 32-bit
        step = M
        if M * max(kp, width) > _I32_MAX and M > 1:
            step = max(1, _I32_MAX // max(kp, width))
        self.act = _scratch(q, "act", (step, kp), torch.bfloat16)
        x = q.view(torch.uint8)
        self.chunks = []
        for r0 in range(0, M, step):
            m = min(step, M - r0)
            blocks = -(-m // _ROWS)
            ints = (1 + 2 * blocks, blocks * _ROWS)
            meta, rowmap = _scratch(q, ("ints", r0), (sum(ints),), torch.int32).split(
                ints
            )
            self.chunks.append(
                (r0, m, blocks, x[r0 : r0 + m].reshape(-1), meta, rowmap)
            )

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> None:
        if self.quant:
            self._quantize(x)
        if self.gemm:
            self._project(out)

    def _quantize(self, x: torch.Tensor) -> None:
        lo, hi, t, gsa = self.lo, self.hi, self.t, self.gsa
        xv = x.view(self.q.shape)
        torch.aminmax(xv, dim=-1, out=(lo, hi))
        torch.maximum(lo.neg_(), hi, out=hi)
        t.copy_(hi).clamp_(min=1e-4).div_(448.0)
        # the UE8M0 ceiling of t into gsa, then t = its inverse (exponent 254 - e)
        tb, e = t.view(torch.int32), gsa.view(torch.int32)
        torch.bitwise_right_shift(tb, 23, out=e).bitwise_and_(0xFF)
        e.add_(tb.bitwise_and_(0x7FFFFF).clamp_(max=1)).clamp_(1, 254)
        e.bitwise_left_shift_(23)
        tb.copy_(e).neg_().add_(254 << 23)
        torch.mul(xv, t.unsqueeze(-1), out=self.q)

    def _project(self, out: torch.Tensor) -> None:
        M, K, n = self.shape
        target = self.target
        if target is None:
            target = out if self.ldo == n else out.as_strided((M, n), (n, 1))
        for r0, m, blocks, x, meta, rowmap in self.chunks:
            act = self.act[:m]
            self.module.prep_f8(x, act, meta, rowmap, 1, K, _ROWS, blocks, self.fmt)
            self.module.gemm_f8(
                self.w, self.ones, None, act, meta, rowmap, target[r0 : r0 + m], _ROWS,
                n, K, blocks, 1, K, 128, 0, self.fmt, False, None, None, False, 0,
                self.gsa[r0 : r0 + m], self.gsb, 1, None,
            )  # fmt: skip
        if self.target is not None:
            out.copy_(target[:, :n])


def _receipt(record: torch.Tensor):
    size = record.numel() * record.element_size()
    if size < _RECEIPT_MIN_BYTES:
        raise ValueError(
            f"launch receipt needs at least {_RECEIPT_MIN_BYTES} bytes, got {size}"
        )
    magic, ident = record.view(torch.uint8)[:16].view(torch.int64).tolist()
    return _RECEIPTS.get(ident) if magic == _MAGIC else None


class Sm8xProgram:
    """The ``run_plan`` / ``run_launch`` / ``run_chain`` entries of launch ``index`` of a
    template on SM8x, as the generated programs take them: ``run_plan`` freezes the
    launch (its stage on the planned workspace) into a host receipt; a launch re-binds
    the call's ``x`` / ``out`` at the argument positions ``names`` maps to them."""

    def __init__(self, prepared, plan, index, names, M, ldo, out_ptr):
        self.prepared, self.plan, self.index = prepared, plan, index
        self.names = tuple(names)
        self.M, self.ldo, self.out_ptr = M, ldo, out_ptr
        self.device_index = _device_index(prepared.device)
        self.roles = {
            position: _REBIND_SOURCES[name]
            for position, name in enumerate(self.names)
            if name in _REBIND_SOURCES
        }

    def run_plan(self, record: torch.Tensor, *args) -> None:
        _receipt(record)
        q = next(
            a for a, name in zip(args, self.names, strict=True) if name in _WORKSPACE_Q
        )
        sf = next(
            a for a, name in zip(args, self.names, strict=True) if name in _WORKSPACE_SF
        )
        stage = Sm8xStage(
            self.prepared, self.plan, self.index, q.view(torch.float8_e4m3fn),
            sf.view(torch.uint8).reshape(-1), self.M, self.ldo, self.out_ptr,
        )  # fmt: skip
        ident = next(_IDS)
        _RECEIPTS[ident] = (self, stage)
        weakref.finalize(record, _RECEIPTS.pop, ident, None)
        header = record.view(torch.uint8)[:16].view(torch.int64)
        header[0], header[1] = _MAGIC, ident

    def run_launch(self, record, stream, i0, t0, i1, t1, i2, t2, i3, t3) -> None:
        entry = _receipt(record)
        if entry is None or entry[0] is not self:
            raise ValueError("launch receipt was not planned by this launcher")
        x, out = self._bind(((i0, t0), (i1, t1), (i2, t2), (i3, t3)), entry[1])
        _run(stream, self.device_index, entry[1], x, out)

    def run_chain(self, *args) -> None:
        if not 5 <= len(args) <= 11 or (len(args) - 3) % 2:
            raise TypeError(
                "run_chain takes the stream, x, out and one to four (receipt, slots) "
                f"pairs, got {len(args)} arguments"
            )
        stream, x, out = args[:3]
        calls = (x, out)
        launches = []
        for record, codes in zip(args[3::2], args[4::2], strict=True):
            entry = _receipt(record)
            if entry is None:
                raise ValueError(
                    "launch receipt was not planned by a launcher of this export"
                )
            program, stage = entry
            pairs = []
            for lane in range(4):
                code = (int(codes) >> (16 * lane)) & 0xFFFF
                if code:
                    pairs.append(((code - 1) >> 1, calls[(code - 1) & 1]))
            launches.append((stage, *program._bind(pairs, stage)))
        for stage, x_, out_ in launches:
            _run(stream, self.device_index, stage, x_, out_)

    def _bind(self, pairs, stage):
        """The call's ``(x, out)`` re-bound at ``pairs`` (validated, nothing issued)."""
        bound = [None, None]
        for position, tensor in pairs:
            if position < 0:
                continue
            role = self.roles.get(position)
            if role is None:
                raise ValueError(
                    f"argument {position} is not a re-bindable tensor slot of this launcher"
                )
            bound[role] = tensor
        x, out = bound
        M, K, n = stage.shape
        if stage.quant and (
            x is None or x.dtype != torch.bfloat16 or tuple(x.shape) != (M, K)
        ):
            raise ValueError(f"the launch must re-bind x as a bf16 [{M}, {K}] tensor")
        if stage.gemm and (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (M, n)
            or out.stride() != (self.ldo, 1)
            or (out.data_ptr() - self.out_ptr) % 32
        ):
            raise ValueError(
                f"the launch must re-bind out as the planned bf16 [{M}, {n}] view class "
                f"(row stride {self.ldo})"
            )
        return x, out


def _run(stream: int, device_index: int, stage: Sm8xStage, x, out) -> None:
    if stream == _current_raw_stream(device_index):
        stage(x, out)
        return
    external = torch.cuda.ExternalStream(stream, device=device_index)
    with torch.cuda.stream(external):
        stage(x, out)
