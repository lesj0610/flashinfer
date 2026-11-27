"""Asynchronous-copy, TMA, mbarrier, tensor-map and sm_90 fence rules (PLAN sections 1, 5, 10).

Owner: A. Library: ``runtime/lib_async.cu``; entry points and flag values are documented in
``include/wgmma_emu/async_device.cuh`` (keep the constants below in sync with it).

Every handled instruction is rewritten into calls of the emulation library or into native
sm_80 PTX:

* ``mbarrier.*`` (all of them, including the sm_80 forms, because the barrier word uses the
  emulation format) and ``cp.async.mbarrier.arrive{.noinc}``.
* ``cp.async.bulk`` (global->shared::cta/cluster, shared::cta->shared::cluster,
  shared::cta->global), ``cp.reduce.async.bulk``, ``cp.async.bulk.tensor`` (load / store,
  tile / im2col, 1-5D, multicast), ``cp.reduce.async.bulk.tensor``,
  ``cp.async.bulk.commit_group`` / ``wait_group{.read}``. The issuing thread performs the copy
  synchronously (one valid execution of the asynchronous operation). commit / wait emit no
  code outside check mode; check mode records bulk sources in a per-function ``.local`` table
  and reports a source overwritten before the matching ``wait_group``.
* ``cp.async.bulk.prefetch{.tensor}`` and ``prefetch{.space}.tensormap``: no code (hints).
* ``st.async`` / ``red.async``, ``tensormap.replace`` / ``tensormap.cp_fenceproxy`` /
  ``tensormap.query.alloc_class_large``.
* the sm_90 ``fence`` / ``membar.proxy`` forms (``.cluster`` scope, ``.acquire`` /
  ``.release``, ``.sync_restrict``, ``.mbarrier_init``, ``proxy.async*``,
  ``proxy.tensormap::generic``, ``proxy.async::generic``, ``proxy.alias`` with semantics,
  ``fence.alias``) are replaced by an equal or stronger native fence. Cluster size is 1, so
  ``.cluster`` scope becomes ``.cta``. Native sm_70 forms (``fence.sc|acq_rel.{cta,gpu,sys}``,
  ``fence.proxy.alias``, ``membar.{cta,gl,sys}``, ``membar.proxy.alias``) are not matched.

Qualifiers may appear in any order (the PTX ISA examples reorder them); a qualifier that is
unknown, repeated, or needs a newer architecture than sm_90a is a translation error.

Instrumentation hooks (``instrument.py``, translator flags ``--async-delay`` / ``--count``):
with E1 the bulk global->shared copy and the tensor loads call the ``wgemu_dl_*`` variants
(poison at issue, data delivered when a wait observes the phase), ``cp.async.mbarrier.arrive``
binds the thread's pending ``cp.async`` copies to the barrier phase, and every
``mbarrier.try_wait`` / ``test_wait`` passes its result to ``instrument.after_wait`` (E1
delivers the copies of the observed phase on success). With E2 the copies add their byte
counts and successful waits count ``mbar_waits``. Without either flag the output is unchanged.
"""

from __future__ import annotations

import re

from . import instrument as _ins
from .core import (
    Ctx,
    Func,
    Stmt,
    parse_address,
    parse_int,
    parse_tensor_address,
    parse_vector,
    split_dests,
)

# ---------------------------------------------------------------------------------------------
# Constants shared with include/wgmma_emu/async_device.cuh
# ---------------------------------------------------------------------------------------------

AF_CHECK = 0x80000000
AF_TRAP = 0x40000000
AF_IM2COL = 0x8
AF_MULTICAST = 0x10
AF_MB_V1 = 0x1
AF_MB_NOCOMPLETE = 0x2
AF_MB_DROP = 0x4
AF_MB_EXPECT_TX = 0x8
AF_MB_HINT = 0x10
AF_NOINC = 0x2
AF_READ = 0x1
AF_OP_SHIFT = 8
AF_TYPE_SHIFT = 12

RED_OP = {"add": 0, "min": 1, "max": 2, "inc": 3, "dec": 4, "and": 5, "or": 6, "xor": 7}
RED_TYPE = {
    "b32": 0,
    "u32": 1,
    "s32": 2,
    "b64": 3,
    "u64": 4,
    "s64": 5,
    "f16": 6,
    "bf16": 7,
    "f32": 8,
    "f64": 9,
}

TMR_FIELD = {
    "rank": 0,
    "box_dim": 1,
    "global_dim": 2,
    "global_stride": 3,
    "element_stride": 4,
    "elemtype": 5,
    "interleave_layout": 6,
    "swizzle_mode": 7,
    "fill_mode": 9,
    "global_address": 10,
}
TMR_ALLOC_ONLY = 11
ALLOC_CLASS = {"alloc_small": 1, "alloc_large": 2, "alloc_same": 3}

BULK_MAX = 32
BULK_STATE_BYTES = 8 + 16 * BULK_MAX
BULK_STATE_KEY = "async_bulk_state"
# func_state owner: rules_misc reaches _bulk_state through _delegate_async, and
# function_prologue must see the same register.
STATE_OWNER = __name__

LIB = {
    "mbar_init": "wgemu_mbar_init",
    "mbar_inval": "wgemu_mbar_inval",
    "mbar_arrive": "wgemu_mbar_arrive",
    "mbar_expect_tx": "wgemu_mbar_expect_tx",
    "mbar_complete_tx": "wgemu_mbar_complete_tx",
    "mbar_test_wait": "wgemu_mbar_test_wait",
    "mbar_test_wait_parity": "wgemu_mbar_test_wait_parity",
    "mbar_try_wait": "wgemu_mbar_try_wait",
    "mbar_try_wait_parity": "wgemu_mbar_try_wait_parity",
    "mbar_pending_count": "wgemu_mbar_pending_count",
    "mbar_check_layout": "wgemu_mbar_check_layout",
    "cp_async_mbar_arrive": "wgemu_cp_async_mbar_arrive",
    "tma_load": "wgemu_tma_load",
    "tma_store": "wgemu_tma_store",
    "tma_reduce": "wgemu_tma_reduce",
    "bulk_g2s": "wgemu_bulk_g2s",
    "bulk_s2s": "wgemu_bulk_s2s",
    "bulk_s2g": "wgemu_bulk_s2g",
    "bulk_red_s2s": "wgemu_bulk_red_s2s",
    "bulk_red_s2g": "wgemu_bulk_red_s2g",
    "bulk_state_init": "wgemu_bulk_state_init",
    "bulk_commit": "wgemu_bulk_commit",
    "bulk_wait": "wgemu_bulk_wait",
    "st_async": "wgemu_st_async",
    "red_async": "wgemu_red_async",
    "tmap_replace": "wgemu_tmap_replace",
    "tmap_copy": "wgemu_tmap_copy",
    "tmap_query_large": "wgemu_tmap_query_large",
}

# Qualifiers that exist in PTX ISA 9.5 but need an architecture newer than sm_90a.
NEWER = {
    "cp_mask": "sm_100",
    "multicast::cluster::32b": "sm_107f",
    "tile::gather4": "sm_100",
    "tile::scatter4": "sm_100",
    "im2col::w": "sm_100",
    "im2col::w::128": "sm_100",
    "im2col_no_offs::w": "sm_107f",
    "cta_group::1": "sm_100",
    "cta_group::2": "sm_100",
    "override::global_address": "sm_107f",
    "override::global_dim": "sm_107f",
    "override::global_dim_stride": "sm_107f",
    "L2::evict_last": "sm_107f",
    "mmio": "sm_100",
    "sync_restrict::proxy::async": "sm_100",
    "swizzle_atomicity": "sm_100",
    "generic::fabric": "sm_100",
    "fabric::generic": "sm_100",
    "fabric::fabric": "sm_100",
}

SCOPE_MAP = {"cta": "cta", "cluster": "cta", "gpu": "gpu", "sys": "sys"}

_INT_RE = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)U?\Z")


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def _hex(v: int) -> str:
    return f"0x{v & 0xFFFFFFFF:08X}"


def _hex64(v: int) -> str:
    return f"0x{v & 0xFFFFFFFFFFFFFFFF:016X}"


def _base_flags(ctx: Ctx) -> int:
    f = AF_CHECK if ctx.mode == "check" else 0
    if ctx.trap:
        f |= AF_TRAP
    return f


def _is_imm(op: str) -> bool:
    return bool(_INT_RE.match(op.replace(" ", "")))


def _classify(
    ctx: Ctx, toks: list[str], cats: dict[str, set[str]], what: str
) -> dict[str, str | None]:
    """Assign every qualifier token to exactly one category (categories are disjoint)."""
    out: dict[str, str | None] = dict.fromkeys(cats)
    for t in toks:
        if t in NEWER or t.startswith("mbarrier::report"):
            arch = NEWER.get(t, "sm_107f")
            ctx.error(
                f"{what}: qualifier .{t} requires {arch}; the sm_90 emulation does not support it"
            )
        hit = [k for k, vals in cats.items() if t in vals]
        if not hit:
            ctx.error(f"{what}: unknown or misplaced qualifier .{t}")
        k = hit[0]
        if out[k] is not None:
            ctx.error(f"{what}: qualifiers .{out[k]} and .{t} both given")
        out[k] = t
    return out


def _forbid(
    ctx: Ctx, q: dict[str, str | None], keys: tuple[str, ...], what: str
) -> None:
    for k in keys:
        if q.get(k) is not None:
            ctx.error(f"{what}: qualifier .{q[k]} is not valid in this form")


def _expect_ops(ctx: Ctx, stmt: Stmt, n: int | tuple[int, ...], what: str) -> None:
    ok = (n,) if isinstance(n, int) else n
    if len(stmt.operands) not in ok:
        ctx.error(
            f"{what}: expected {' or '.join(map(str, ok))} operands, got {len(stmt.operands)}"
        )


def _reg_bits(ctx: Ctx, op: str, what: str) -> int:
    if ctx.reg_type(op) is None:
        ctx.error(f"{what}: operand {op!r} is not a declared register or an integer")
    bits = ctx.reg_bits(op)
    if bits is None:
        ctx.error(f"{what}: register {op} has an unknown width")
    return bits


def _copy_reg(ctx: Ctx, op: str, bits: int) -> tuple[list[str], str]:
    r = ctx.new_reg(f"b{bits}")
    return [f"mov.b{bits} \t{r}, {op};"], r


def _val32(ctx: Ctx, op: str, what: str, signed: bool = False) -> tuple[list[str], str]:
    """32-bit call argument (register or immediate) from a 16/32-bit register or immediate."""
    op = op.strip()
    if _is_imm(op):
        return [], _hex(parse_int(op))
    bits = _reg_bits(ctx, op, what)
    if bits == 32:
        ty = ctx.reg_type(op) or ""
        if op.startswith("%") and not ty.startswith("f"):
            return [], op
        return _copy_reg(ctx, op, 32)
    if bits == 16:
        lines, src = _copy_reg(ctx, op, 16)
        r = ctx.new_reg("b32")
        cvt = "cvt.s32.s16" if signed else "cvt.u32.u16"
        return [*lines, f"{cvt} \t{r}, {src};"], r
    ctx.error(f"{what}: operand {op} must be a 16- or 32-bit value, not {bits}-bit")
    raise AssertionError  # unreachable


def _val64(ctx: Ctx, op: str, what: str) -> tuple[list[str], str]:
    """64-bit call argument from a 32/64-bit register (zero-extended) or immediate."""
    op = op.strip()
    if _is_imm(op):
        return [], _hex64(parse_int(op))
    bits = _reg_bits(ctx, op, what)
    if bits == 64:
        ty = ctx.reg_type(op) or ""
        if op.startswith("%") and not ty.startswith("f"):
            return [], op
        return _copy_reg(ctx, op, 64)
    if bits == 32:
        lines, src = _copy_reg(ctx, op, 32)
        r = ctx.new_reg("b64")
        return [*lines, f"cvt.u64.u32 \t{r}, {src};"], r
    ctx.error(f"{what}: operand {op} must be a 32- or 64-bit value, not {bits}-bit")
    raise AssertionError  # unreachable


def _norm_addr(ctx: Ctx, op: str) -> tuple[list[str], str]:
    """Copy a register base without '%' (declared in an inline-asm block) into a %wgemu register,
    because the core address helpers and call() only accept '%' registers."""
    base, off = parse_address(op)
    if base and not base.startswith("%") and ctx.reg_type(base) is not None:
        bits = ctx.reg_bits(base)
        if bits not in (32, 64):
            ctx.error(f"address register {base} must be 32- or 64-bit")
        lines, r = _copy_reg(ctx, base, bits)
        return lines, (f"[{r}+{off}]" if off else f"[{r}]")
    return [], op


def _smem(ctx: Ctx, op: str, space: str | None) -> tuple[list[str], str]:
    l1, a = _norm_addr(ctx, op)
    l2, r = ctx.to_shared_u32(a, space)
    return l1 + l2, r


def _gen(ctx: Ctx, op: str, space: str | None) -> tuple[list[str], str]:
    l1, a = _norm_addr(ctx, op)
    l2, r = ctx.to_generic_u64(a, space)
    return l1 + l2, r


def _set_pred(ctx: Ctx, dest: str, r: str, what: str) -> list[str]:
    if dest == "_" or ctx.reg_type(dest) != "pred":
        ctx.error(f"{what}: destination {dest!r} must be a predicate register")
    return [f"setp.ne.u32 \t{dest}, {r}, 0;"]


def _tmap_addr(
    ctx: Ctx, base: str, space: str | None, what: str
) -> tuple[list[str], str]:
    """Generic u64 address of a tensor-map object.

    ``base`` is the address part of a tensor operand ('[tmap, {...}]') or of '[addr]'. A
    register holds a generic address (space None) or an address in ``space``. A symbol is
    converted from its own state space; a ``.param`` symbol of a non-entry function is not
    addressable with cvta, so its 128 bytes are copied to a ``.local`` buffer first.
    """
    lines, a = _norm_addr(ctx, base if base.startswith("[") else f"[{base}]")
    b, off = parse_address(a)
    if b.startswith("%"):
        if space is None and ctx.reg_bits(b) != 64:
            ctx.error(
                f"{what}: tensor-map address register {b} must be a 64-bit generic address"
            )
        l2, r = ctx.to_generic_u64(a, space)
        return lines + l2, r
    if not b:
        ctx.error(f"{what}: absolute tensor-map address {base!r}")
    sp = ctx.symbol_space(b)
    if sp is None:
        ctx.error(f"{what}: unknown tensor-map symbol {b}")
    if sp == "param" and ctx.func is not None and not ctx.func.is_entry:
        buf = ctx.local_buf(f"tmap_{b}_{off}", 128, 64)
        t = ctx.new_reg("b64")
        for i in range(16):
            lines.append(f"ld.param.b64 \t{t}, [{b}+{off + 8 * i}];")
            lines.append(f"st.local.b64 \t[{buf}+{8 * i}], {t};")
        l2, r = ctx.generic_addr_of_local(buf)
        return lines + l2, r
    if sp not in ("param", "const", "global", "shared"):
        ctx.error(f"{what}: tensor-map symbol {b} is in .{sp}")
    if space is not None and space != sp:
        ctx.error(f"{what}: tensor-map symbol {b} is in .{sp}, not .{space}")
    l2, r = ctx.to_generic_u64(a, sp)
    return lines + l2, r


def _bulk_state(ctx: Ctx) -> str:
    """b64 register with the generic address of the check-mode bulk table ('0' otherwise)."""
    if ctx.mode != "check":
        return "0"
    st = ctx.func_state(STATE_OWNER)
    if "bulk_reg" not in st:
        st["bulk_sym"] = ctx.local_buf(BULK_STATE_KEY, BULK_STATE_BYTES, 16)
        st["bulk_reg"] = ctx.new_reg("b64")
    return st["bulk_reg"]


def _split_opcode(stmt: Stmt, n: int) -> list[str]:
    return stmt.opcode.split(".")[n:]


# ---------------------------------------------------------------------------------------------
# mbarrier
# ---------------------------------------------------------------------------------------------

_MB_SPACE = {"shared", "shared::cta", "shared::cluster"}
_MB_CATS = {
    "sem": {"release", "relaxed", "acquire"},
    "scope": {"cta", "cluster"},
    "space": _MB_SPACE,
    "type": {"b64"},
    "layout": {"layout::v0", "layout::v1"},
    "phase": {"phase_type::primary", "phase_type::conditional"},
    "sub": {"expect_tx", "noComplete"},
    "parity": {"parity"},
    "noinc": {"noinc"},
}


def _mb_quals(ctx: Ctx, stmt: Stmt, toks: list[str], allowed: tuple[str, ...]) -> dict:
    what = stmt.opcode
    q = _classify(ctx, toks, _MB_CATS, what)
    _forbid(ctx, q, tuple(k for k in _MB_CATS if k not in allowed), what)
    if q["type"] is None:
        ctx.error(f"{what}: missing .b64")
    return q


def _rw_mbarrier(stmt: Stmt, ctx: Ctx) -> list[str]:
    toks = _split_opcode(stmt, 1)
    what = stmt.opcode
    if not toks:
        ctx.error(f"{what}: missing operation")
    op, rest = toks[0], toks[1:]
    base = _base_flags(ctx)
    ops = stmt.operands
    out: list[str] = []

    if op == "init":
        q = _mb_quals(ctx, stmt, rest, ("layout", "space", "type"))
        if q["space"] == "shared::cluster":
            ctx.error(f"{what}: .shared::cluster is not valid for mbarrier.init")
        _expect_ops(ctx, stmt, 2, what)
        l1, bar = _smem(ctx, ops[0], q["space"])
        l2, cnt = _val32(ctx, ops[1], what)
        flags = base | (AF_MB_V1 if q["layout"] == "layout::v1" else 0)
        return [
            *l1,
            *l2,
            *ctx.call(
                LIB["mbar_init"], [("b32", bar), ("b32", cnt), ("b32", _hex(flags))]
            ),
        ]

    if op == "inval":
        q = _mb_quals(ctx, stmt, rest, ("space", "type"))
        if q["space"] == "shared::cluster":
            ctx.error(f"{what}: .shared::cluster is not valid for mbarrier.inval")
        _expect_ops(ctx, stmt, 1, what)
        l1, bar = _smem(ctx, ops[0], q["space"])
        return [*l1, *ctx.call(LIB["mbar_inval"], [("b32", bar), ("b32", _hex(base))])]

    if op in ("expect_tx", "complete_tx"):
        q = _mb_quals(ctx, stmt, rest, ("sem", "scope", "space", "type"))
        if q["sem"] not in (None, "relaxed"):
            ctx.error(f"{what}: .{q['sem']} is not valid (only .relaxed)")
        _expect_ops(ctx, stmt, 2, what)
        l1, bar = _smem(ctx, ops[0], q["space"])
        l2, tx = _val32(ctx, ops[1], what)
        fn = LIB["mbar_expect_tx"] if op == "expect_tx" else LIB["mbar_complete_tx"]
        return [
            *l1,
            *l2,
            *ctx.call(fn, [("b32", bar), ("b32", tx), ("b32", _hex(base))]),
        ]

    if op in ("arrive", "arrive_drop"):
        q = _mb_quals(ctx, stmt, rest, ("sem", "scope", "space", "type", "sub"))
        if q["sem"] == "acquire":
            ctx.error(f"{what}: .acquire is not valid for arrive")
        sub = q["sub"]
        flags = base | (AF_MB_DROP if op == "arrive_drop" else 0)
        if sub == "noComplete":
            if q["sem"] not in (None, "release") or q["scope"] not in (None, "cta"):
                ctx.error(f"{what}: .noComplete only allows .release.cta")
            if q["space"] == "shared::cluster":
                ctx.error(f"{what}: .noComplete is not valid with .shared::cluster")
        if len(ops) < 2:
            ctx.error(f"{what}: expected state, [addr], ...")
        dest, addr, extra = ops[0], ops[1], ops[2:]
        count, tx = "1", "0"
        if sub == "expect_tx":
            if len(extra) != 1:
                ctx.error(f"{what}: expected state, [addr], txCount")
            tx = extra[0]
            flags |= AF_MB_EXPECT_TX
        elif sub == "noComplete":
            if len(extra) != 1:
                ctx.error(f"{what}: expected state, [addr], count")
            count = extra[0]
            flags |= AF_MB_NOCOMPLETE
        else:
            if len(extra) > 1:
                ctx.error(f"{what}: expected state, [addr]{{, count}}")
            if extra:
                count = extra[0]
        if q["space"] == "shared::cluster" and dest != "_":
            ctx.error(
                f"{what}: the state operand of a .shared::cluster arrive must be '_'"
            )
        l1, bar = _smem(ctx, addr, q["space"])
        l2, c = _val32(ctx, count, what)
        l3, t = _val32(ctx, tx, what)
        r = ctx.new_reg("b32")
        out = [*l1, *l2, *l3]
        out += ctx.call(
            LIB["mbar_arrive"],
            [("b32", bar), ("b32", c), ("b32", t), ("b32", _hex(flags))],
            ret=("b32", r),
        )
        if dest != "_":
            if _reg_bits(ctx, dest, what) != 64:
                ctx.error(f"{what}: state register {dest} must be 64-bit")
            out.append(f"cvt.u64.u32 \t{dest}, {r};")
        return out

    if op in ("test_wait", "try_wait"):
        q = _mb_quals(
            ctx, stmt, rest, ("parity", "phase", "sem", "scope", "space", "type")
        )
        if q["space"] == "shared::cluster":
            ctx.error(f"{what}: .shared::cluster is not valid for {op}")
        if q["sem"] == "release":
            ctx.error(f"{what}: .release is not valid for {op}")
        if q["phase"] == "phase_type::conditional" and q["parity"] is None:
            ctx.error(f"{what}: .phase_type::conditional requires .parity")
        if not ops:
            ctx.error(f"{what}: missing operands")
        dests = split_dests(ops[0])
        if len(dests) > 2:
            ctx.error(f"{what}: malformed destination {ops[0]!r}")
        i = 1
        report_v = None
        if len(ops) > 1 and not ops[1].startswith("["):
            report_v = ops[1]
            i = 2
        if report_v is not None and len(dests) != 2:
            ctx.error(
                f"{what}: reportValue needs the waitComplete|reportPredicate form"
            )
        if len(dests) == 2 and q["phase"] != "phase_type::primary":
            ctx.error(
                f"{what}: reportPredicate / reportValue need .phase_type::primary"
            )
        args = ops[i:]
        nmax = 3 if op == "try_wait" else 2
        if not (2 <= len(args) <= nmax):
            ctx.error(
                f"{what}: expected [addr], {'phaseParity' if q['parity'] else 'state'}"
                f"{', timeHint' if op == 'try_wait' else ''}"
            )
        l1, bar = _smem(ctx, args[0], q["space"])
        out = list(l1)
        flags = base
        if q["parity"]:
            l2, ph = _val32(ctx, args[1], what)
            arg = ("b32", ph)
        else:
            l2, st = _val64(ctx, args[1], what)
            arg = ("b64", st)
        out += l2
        cargs = [("b32", bar), arg]
        if op == "try_wait":
            hint = "0"
            if len(args) == 3:
                l3, hint = _val32(ctx, args[2], what)
                out += l3
                flags |= AF_MB_HINT
            cargs.append(("b32", hint))
            fn = LIB["mbar_try_wait_parity"] if q["parity"] else LIB["mbar_try_wait"]
        else:
            fn = LIB["mbar_test_wait_parity"] if q["parity"] else LIB["mbar_test_wait"]
        cargs.append(("b32", _hex(flags)))
        r = ctx.new_reg("b32")
        out += ctx.call(fn, cargs, ret=("b32", r))
        if _ins.delay_on(ctx) or ctx.count:
            if q["parity"]:
                par = ph
            elif _is_imm(st):
                par = _hex((parse_int(st) >> 31) & 1)
            else:
                t64 = ctx.new_reg("b64")
                par = ctx.new_reg("b32")
                out += [f"shr.b64 \t{t64}, {st}, 31;", f"cvt.u32.u64 \t{par}, {t64};"]
            out += _ins.after_wait(ctx, bar, par, r, flags)
        out += _set_pred(ctx, dests[0], r, what)
        # reportPredicate is false and reportValue is 0: always so for .layout::v0 (ptx.txt
        # 21691); for .layout::v1 no operation that sets a report exists on sm_90 [inferred].
        if len(dests) == 2:
            if ctx.reg_type(dests[1]) != "pred":
                ctx.error(f"{what}: reportPredicate {dests[1]} must be a predicate")
            out.append(f"setp.ne.u32 \t{dests[1]}, {r}, {r};")
        if report_v is not None:
            bits = _reg_bits(ctx, report_v, what)
            if bits == 8:
                z = ctx.new_reg("b32")
                out += [f"mov.u32 \t{z}, 0;", f"cvt.u8.u32 \t{report_v}, {z};"]
            elif bits in (16, 32, 64):
                out.append(f"mov.b{bits} \t{report_v}, 0;")
            else:
                ctx.error(f"{what}: reportValue {report_v} must be a .b8 register")
        return out

    if op == "pending_count":
        q = _mb_quals(ctx, stmt, rest, ("layout", "type"))
        if q["layout"] not in (None, "layout::v0"):
            ctx.error(f"{what}: only .layout::v0 is valid")
        _expect_ops(ctx, stmt, 2, what)
        l1, st = _val64(ctx, ops[1], what)
        if _reg_bits(ctx, ops[0], what) != 32:
            ctx.error(f"{what}: count register {ops[0]} must be 32-bit")
        r = ctx.new_reg("b32")
        return [
            *l1,
            *ctx.call(LIB["mbar_pending_count"], [("b64", st)], ret=("b32", r)),
            f"mov.b32 \t{ops[0]}, {r};",
        ]

    if op == "check_layout":
        q = _mb_quals(ctx, stmt, rest, ("layout", "space", "type"))
        if q["layout"] is None:
            ctx.error(f"{what}: missing .layout::v0 / .layout::v1")
        if q["space"] not in (None, "shared::cta"):
            ctx.error(f"{what}: only .shared::cta is valid")
        _expect_ops(ctx, stmt, 2, what)
        l1, bar = _smem(ctx, ops[1], q["space"])
        r = ctx.new_reg("b32")
        v1 = "1" if q["layout"] == "layout::v1" else "0"
        out = [
            *l1,
            *ctx.call(
                LIB["mbar_check_layout"], [("b32", bar), ("b32", v1)], ret=("b32", r)
            ),
        ]
        return out + _set_pred(ctx, ops[0], r, what)

    ctx.error(f"{what}: unsupported mbarrier operation .{op}")
    raise AssertionError  # unreachable


def _rw_cp_async_mbar_arrive(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    q = _mb_quals(ctx, stmt, _split_opcode(stmt, 4), ("noinc", "space", "type"))
    if q["space"] == "shared::cluster":
        ctx.error(f"{what}: .shared::cluster is not valid")
    _expect_ops(ctx, stmt, 1, what)
    l1, bar = _smem(ctx, stmt.operands[0], q["space"])
    flags = _base_flags(ctx) | (AF_NOINC if q["noinc"] else 0)
    if _ins.delay_on(ctx):
        lh, h = _ins.hdr(ctx)
        return [
            *l1,
            *lh,
            *ctx.call(
                _ins.LIB["dl_cp_mbar_arrive"],
                [("b32", h), ("b32", bar), ("b32", _hex(flags))],
            ),
        ]
    return [
        *l1,
        *ctx.call(LIB["cp_async_mbar_arrive"], [("b32", bar), ("b32", _hex(flags))]),
    ]


# ---------------------------------------------------------------------------------------------
# cp.async.bulk / cp.reduce.async.bulk (non-tensor)
# ---------------------------------------------------------------------------------------------

_SPACE_TOKS = {"shared::cta", "shared::cluster", "global", "shared"}
_COMPLETION = {"mbarrier::complete_tx::bytes", "bulk_group"}
_BULK_CATS = {
    "sem": {"weak", "relaxed"},
    "scope": {"cta", "cluster", "gpu", "sys"},
    "completion": _COMPLETION,
    "hint": {"L2::cache_hint"},
    "oob": {"ignore_oob"},
    "multicast": {"multicast::cluster", "multicast::cluster::16b"},
    "type": {"b128"},
}
_BRED_CATS = {
    "sem": {"relaxed"},
    "scope": {"cta", "cluster", "gpu", "sys"},
    "completion": _COMPLETION,
    "hint": {"L2::cache_hint"},
    "op": set(RED_OP),
    "type": set(RED_TYPE),
    "noftz": {"noftz"},
}


def _spaces(ctx: Ctx, toks: list[str], what: str) -> tuple[list[str], list[str]]:
    sp = [t for t in toks if t in _SPACE_TOKS]
    rest = [t for t in toks if t not in _SPACE_TOKS]
    if "shared" in sp:
        ctx.error(
            f"{what}: plain .shared is not valid; use .shared::cta or .shared::cluster"
        )
    if len(sp) != 2:
        ctx.error(
            f"{what}: expected .dst.src state spaces, got {['.' + s for s in sp]}"
        )
    return sp, rest


def _direction(ctx: Ctx, sp: list[str], completion: str | None, what: str) -> str:
    """'g2s', 's2s' or 's2g' from the .dst.src pair (dst first, PTX convention) and the
    completion mechanism, which must agree with it."""
    dst, src = sp
    if dst in ("shared::cta", "shared::cluster") and src == "global":
        kind, need = "g2s", "mbarrier::complete_tx::bytes"
    elif dst == "shared::cluster" and src == "shared::cta":
        kind, need = "s2s", "mbarrier::complete_tx::bytes"
    elif dst == "global" and src == "shared::cta":
        kind, need = "s2g", "bulk_group"
    else:
        ctx.error(f"{what}: unsupported direction .{dst}.{src}")
        raise AssertionError  # unreachable
    if completion != need:
        ctx.error(f"{what}: direction .{dst}.{src} needs .{need}")
    return kind


def _optional_tail(
    ctx: Ctx, ops: list[str], names: list[tuple[str, bool]], what: str
) -> dict[str, str | None]:
    """Assign trailing operands to the enabled optional names, in order; counts must match."""
    enabled = [n for n, on in names if on]
    if len(ops) != len(enabled):
        ctx.error(
            f"{what}: expected {len(enabled)} trailing operand(s) ({', '.join(enabled) or 'none'}), got {len(ops)}"
        )
    out: dict[str, str | None] = {n: None for n, _ in names}
    for n, v in zip(enabled, ops, strict=True):
        out[n] = v
    return out


def _check_policy(ctx: Ctx, pol: str | None, what: str) -> None:
    if pol is not None and not _is_imm(pol) and _reg_bits(ctx, pol, what) != 64:
        ctx.error(f"{what}: cache_policy {pol} must be a 64-bit register")


def _rw_bulk(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    sp, rest = _spaces(ctx, _split_opcode(stmt, 3), what)
    q = _classify(ctx, rest, _BULK_CATS, what)
    kind = _direction(ctx, sp, q["completion"], what)
    # ptx.txt 15323-15415: '.relaxed' comes with '.scope' and '.b128' in every direction (the
    # example 'cp.async.bulk.relaxed.shared::cta.global...b128' without a scope contradicts the
    # syntax and is rejected) [inferred]. The copy itself is done with 16-byte accesses.
    if q["sem"] == "relaxed":
        if q["scope"] is None or q["type"] is None:
            ctx.error(f"{what}: .relaxed needs .scope and .b128")
    elif q["scope"] is not None or q["type"] is not None:
        ctx.error(f"{what}: .scope / .b128 need .relaxed")
    if kind == "s2s" and q["scope"] not in (None, "cta", "cluster"):
        ctx.error(
            f"{what}: .{q['scope']} is not valid for shared::cta -> shared::cluster"
        )
    ops = stmt.operands
    base = _base_flags(ctx)
    if len(ops) < 3:
        ctx.error(f"{what}: expected [dstMem], [srcMem], size, ...")
    out: list[str] = []
    if kind == "g2s":
        cta_dst = sp[0] == "shared::cta"
        if cta_dst:
            _forbid(ctx, q, ("multicast",), what)
        else:
            _forbid(ctx, q, ("oob",), what)
        mid = 3 + (2 if q["oob"] else 0)
        if len(ops) < mid + 1:
            ctx.error(f"{what}: missing [mbar]")
        tail = _optional_tail(
            ctx,
            ops[mid + 1 :],
            [
                ("ctaMask", q["multicast"] is not None),
                ("policy", q["hint"] is not None),
            ],
            what,
        )
        _check_policy(ctx, tail["policy"], what)
        l1, dst = _smem(ctx, ops[0], sp[0])
        l2, src = _gen(ctx, ops[1], "global")
        l3, size = _val32(ctx, ops[2], what)
        out += l1 + l2 + l3
        il, ir = "0", "0"
        if q["oob"]:
            l4a, il = _val32(ctx, ops[3], what)
            l4b, ir = _val32(ctx, ops[4], what)
            out += l4a + l4b
        l5, mbar = _smem(ctx, ops[mid], sp[0])
        out += l5
        mask = "1"
        flags = base
        if q["multicast"]:
            l6, mask = _val32(ctx, tail["ctaMask"], what)
            out += l6
            flags |= AF_MULTICAST
        args = [
            ("b32", dst),
            ("b64", src),
            ("b32", size),
            ("b32", mbar),
            ("b32", il),
            ("b32", ir),
            ("b32", mask),
            ("b32", _hex(flags)),
        ]
        if _ins.delay_on(ctx):
            lh, h = _ins.hdr(ctx)
            out += lh + ctx.call(_ins.LIB["dl_bulk_g2s"], [*args, ("b32", h)])
        else:
            out += ctx.call(LIB["bulk_g2s"], args)
        return out + _ins.count_add(ctx, _ins.CNT_BULK, size)
    if kind == "s2s":
        _forbid(ctx, q, ("hint", "oob", "multicast"), what)
        _expect_ops(ctx, stmt, 4, what)
        l1, dst = _smem(ctx, ops[0], "shared::cluster")
        l2, src = _smem(ctx, ops[1], "shared::cta")
        l3, size = _val32(ctx, ops[2], what)
        l4, mbar = _smem(ctx, ops[3], "shared::cluster")
        return [
            *l1,
            *l2,
            *l3,
            *l4,
            *ctx.call(
                LIB["bulk_s2s"],
                [
                    ("b32", dst),
                    ("b32", src),
                    ("b32", size),
                    ("b32", mbar),
                    ("b32", _hex(base)),
                ],
            ),
            *_ins.count_add(ctx, _ins.CNT_BULK, size),
        ]
    # s2g
    _forbid(ctx, q, ("oob", "multicast"), what)
    tail = _optional_tail(ctx, ops[3:], [("policy", q["hint"] is not None)], what)
    _check_policy(ctx, tail["policy"], what)
    l1, dst = _gen(ctx, ops[0], "global")
    l2, src = _smem(ctx, ops[1], "shared::cta")
    l3, size = _val32(ctx, ops[2], what)
    return [
        *l1,
        *l2,
        *l3,
        *ctx.call(
            LIB["bulk_s2g"],
            [
                ("b64", dst),
                ("b32", src),
                ("b32", size),
                ("b32", _hex(base)),
                ("b64", _bulk_state(ctx)),
            ],
        ),
        *_ins.count_add(ctx, _ins.CNT_BULK, size),
    ]


# Valid (redOp, type) pairs, ptx.txt 15635-15676 (cp.reduce.async.bulk table: bitwise is .b32
# for .shared::cluster, .b32/.b64 for .global). Bitwise operations also accept the integer
# types of the same width, because the ISA example at ptx.txt 15745 uses .xor.s32 [inferred].
_BRED_S2S = {
    "add": {"u32", "s32", "u64"},
    "min": {"u32", "s32"},
    "max": {"u32", "s32"},
    "inc": {"u32"},
    "dec": {"u32"},
    "and": {"b32", "u32", "s32"},
    "or": {"b32", "u32", "s32"},
    "xor": {"b32", "u32", "s32"},
}
_BRED_S2G = {
    "add": {"u32", "s32", "u64", "f32", "f64", "f16", "bf16"},
    "min": {"u32", "s32", "u64", "s64", "f16", "bf16"},
    "max": {"u32", "s32", "u64", "s64", "f16", "bf16"},
    "inc": {"u32"},
    "dec": {"u32"},
    "and": {"b32", "u32", "s32", "b64", "u64", "s64"},
    "or": {"b32", "u32", "s32", "b64", "u64", "s64"},
    "xor": {"b32", "u32", "s32", "b64", "u64", "s64"},
}


def _red_code(
    ctx: Ctx, op: str, ty: str, noftz: bool, table: dict[str, set[str]], what: str
) -> int:
    if ty not in table.get(op, set()):
        ctx.error(f"{what}: .{op}.{ty} is not a valid reduction")
    if ty in ("f16", "bf16") and op == "add" and not noftz:
        ctx.error(f"{what}: .add.{ty} requires .noftz")
    if noftz and not (op == "add" and ty in ("f16", "bf16", "f32")):
        ctx.error(f"{what}: .noftz is only valid with .add.f16/.bf16/.f32")
    if op in ("and", "or", "xor"):
        ty = "b64" if ty in ("b64", "u64", "s64") else "b32"
    return (RED_OP[op] << AF_OP_SHIFT) | (RED_TYPE[ty] << AF_TYPE_SHIFT)


def _rw_bulk_reduce(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    sp, rest = _spaces(ctx, _split_opcode(stmt, 4), what)
    q = _classify(ctx, rest, _BRED_CATS, what)
    kind = _direction(ctx, sp, q["completion"], what)
    if kind == "g2s":
        ctx.error(f"{what}: global -> shared is not a cp.reduce.async.bulk direction")
    if q["op"] is None or q["type"] is None:
        ctx.error(f"{what}: missing .redOp / .type")
    if (q["sem"] is None) != (q["scope"] is None):
        ctx.error(f"{what}: .sem and .scope go together")
    ops = stmt.operands
    base = _base_flags(ctx)
    if kind == "s2s":
        _forbid(ctx, q, ("hint",), what)
        if q["scope"] not in (None, "cta", "cluster"):
            ctx.error(
                f"{what}: .{q['scope']} is not valid for a shared::cluster destination"
            )
        code = _red_code(
            ctx, q["op"], q["type"], q["noftz"] is not None, _BRED_S2S, what
        )
        _expect_ops(ctx, stmt, 4, what)
        l1, dst = _smem(ctx, ops[0], "shared::cluster")
        l2, src = _smem(ctx, ops[1], "shared::cta")
        l3, size = _val32(ctx, ops[2], what)
        l4, mbar = _smem(ctx, ops[3], "shared::cluster")
        return [
            *l1,
            *l2,
            *l3,
            *l4,
            *ctx.call(
                LIB["bulk_red_s2s"],
                [
                    ("b32", dst),
                    ("b32", src),
                    ("b32", size),
                    ("b32", mbar),
                    ("b32", _hex(base | code)),
                ],
            ),
            *_ins.count_add(ctx, _ins.CNT_BULK, size),
        ]
    code = _red_code(ctx, q["op"], q["type"], q["noftz"] is not None, _BRED_S2G, what)
    if len(ops) < 3:
        ctx.error(f"{what}: expected [dstMem], [srcMem], size{{, cache_policy}}")
    tail = _optional_tail(ctx, ops[3:], [("policy", q["hint"] is not None)], what)
    _check_policy(ctx, tail["policy"], what)
    l1, dst = _gen(ctx, ops[0], "global")
    l2, src = _smem(ctx, ops[1], "shared::cta")
    l3, size = _val32(ctx, ops[2], what)
    return [
        *l1,
        *l2,
        *l3,
        *ctx.call(
            LIB["bulk_red_s2g"],
            [
                ("b64", dst),
                ("b32", src),
                ("b32", size),
                ("b32", _hex(base | code)),
                ("b64", _bulk_state(ctx)),
            ],
        ),
        *_ins.count_add(ctx, _ins.CNT_BULK, size),
    ]


# ---------------------------------------------------------------------------------------------
# cp.async.bulk.tensor / cp.reduce.async.bulk.tensor
# ---------------------------------------------------------------------------------------------

_DIMS = {"1d": 1, "2d": 2, "3d": 3, "4d": 4, "5d": 5}
_TENSOR_CATS = {
    "dim": set(_DIMS),
    "mode": {"tile", "im2col", "im2col_no_offs"},
    "completion": _COMPLETION,
    "multicast": {"multicast::cluster", "multicast::cluster::16b"},
    "hint": {"L2::cache_hint"},
    "op": set(RED_OP),
}


def _coords(
    ctx: Ctx, elems: list[str], dim: int, what: str
) -> tuple[list[str], list[str]]:
    if len(elems) != dim:
        ctx.error(f"{what}: .{dim}d needs {dim} coordinates, got {len(elems)}")
    lines: list[str] = []
    vals: list[str] = []
    for e in elems:
        ln, v = _val32(ctx, e, what, signed=True)
        lines += ln
        vals.append(v)
    return lines, vals + ["0"] * (5 - dim)


def _rw_tensor(stmt: Stmt, ctx: Ctx, reduce: bool) -> list[str]:
    what = stmt.opcode
    # 'cp.reduce.async.bulk.tensor.*' has one more leading token than 'cp.async.bulk.tensor.*'.
    sp, rest = _spaces(ctx, _split_opcode(stmt, 5 if reduce else 4), what)
    q = _classify(ctx, rest, _TENSOR_CATS, what)
    kind = _direction(ctx, sp, q["completion"], what)
    if q["dim"] is None:
        ctx.error(f"{what}: missing .dim")
    dim = _DIMS[q["dim"]]
    mode = q["mode"] or "tile"
    ops = stmt.operands
    base = _base_flags(ctx) | dim
    if kind == "s2s":
        ctx.error(
            f"{what}: shared::cta -> shared::cluster is not a tensor copy direction"
        )
    if kind == "g2s":
        if reduce:
            ctx.error(
                f"{what}: cp.reduce.async.bulk.tensor only reduces into global memory"
            )
        _forbid(ctx, q, ("op",), what)
        if mode == "im2col_no_offs":
            ctx.error(f"{what}: .im2col_no_offs is a store mode; loads use .im2col")
        if sp[0] == "shared::cta":
            _forbid(ctx, q, ("multicast",), what)
        im2col = mode == "im2col"
        if im2col and dim < 3:
            ctx.error(f"{what}: .im2col needs .3d to .5d")
        if len(ops) < 3:
            ctx.error(
                f"{what}: expected [dstMem], [tensorMap, tensorCoords], [mbar], ..."
            )
        tail = _optional_tail(
            ctx,
            ops[3:],
            [
                ("im2colInfo", im2col),
                ("ctaMask", q["multicast"] is not None),
                ("policy", q["hint"] is not None),
            ],
            what,
        )
        _check_policy(ctx, tail["policy"], what)
        tbase, elems = parse_tensor_address(ops[1])
        out: list[str] = []
        l1, dst = _smem(ctx, ops[0], sp[0])
        l2, tmap = _tmap_addr(ctx, tbase, None, what)
        l3, cs = _coords(ctx, elems, dim, what)
        l4, mbar = _smem(ctx, ops[2], sp[0])
        out += l1 + l2 + l3 + l4
        offs = ["0", "0", "0"]
        flags = base
        if im2col:
            flags |= AF_IM2COL
            info = parse_vector(tail["im2colInfo"])
            if len(info) != dim - 2:
                ctx.error(
                    f"{what}: im2colInfo needs {dim - 2} offsets, got {len(info)}"
                )
            for k, e in enumerate(info):
                ln, offs[k] = _val32(ctx, e, what)
                out += ln
        mask = "1"
        if q["multicast"]:
            ln, mask = _val32(ctx, tail["ctaMask"], what)
            out += ln
            flags |= AF_MULTICAST
        args = (
            [("b32", dst), ("b64", tmap), ("b32", mbar)]
            + [("b32", c) for c in cs]
            + [("b32", o) for o in offs]
            + [("b32", mask), ("b32", _hex(flags))]
        )
        if _ins.delay_on(ctx):
            lh, h = _ins.hdr(ctx)
            out += lh + ctx.call(_ins.LIB["dl_tma_load"], [*args, ("b32", h)])
        else:
            out += ctx.call(LIB["tma_load"], args)
        return out + _ins.count_tma(ctx, tmap, flags)
    # s2g: store or reduce
    _forbid(ctx, q, ("multicast",), what)
    if reduce and q["op"] is None:
        ctx.error(f"{what}: missing .redOp")
    if not reduce:
        _forbid(ctx, q, ("op",), what)
    # '.im2col' on a store / reduce is read as .im2col_no_offs (the ISA example
    # 'cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.im2col.bulk_group') [inferred].
    flags = base
    if mode in ("im2col", "im2col_no_offs"):
        if dim < 3:
            ctx.error(f"{what}: .{mode} needs .3d to .5d")
        flags |= AF_IM2COL
    if reduce:
        flags |= RED_OP[q["op"]] << AF_OP_SHIFT
    if len(ops) < 2:
        ctx.error(
            f"{what}: expected [tensorMap, tensorCoords], [srcMem]{{, cache_policy}}"
        )
    tail = _optional_tail(ctx, ops[2:], [("policy", q["hint"] is not None)], what)
    _check_policy(ctx, tail["policy"], what)
    tbase, elems = parse_tensor_address(ops[0])
    l1, tmap = _tmap_addr(ctx, tbase, None, what)
    l2, cs = _coords(ctx, elems, dim, what)
    l3, src = _smem(ctx, ops[1], "shared::cta")
    fn = LIB["tma_reduce"] if reduce else LIB["tma_store"]
    return [
        *l1,
        *l2,
        *l3,
        *ctx.call(
            fn,
            [("b64", tmap), ("b32", src)]
            + [("b32", c) for c in cs]
            + [("b32", _hex(flags)), ("b64", _bulk_state(ctx))],
        ),
        *_ins.count_tma(ctx, tmap, flags),
    ]


_PREFETCH_CATS = {
    "dim": set(_DIMS),
    "mode": {"tile", "im2col"},
    "hint": {"L2::cache_hint"},
    "dst": {"L2"},
    "src": {"global"},
}


def _rw_prefetch(stmt: Stmt, ctx: Ctx, tensor: bool) -> list[str]:
    """cp.async.bulk.prefetch{.tensor}: an L2 hint without memory effect -> no code."""
    what = stmt.opcode
    toks = _split_opcode(stmt, 5 if tensor else 4)
    q = _classify(ctx, toks, _PREFETCH_CATS, what)
    if q["dst"] is None or q["src"] is None:
        ctx.error(f"{what}: expected .L2.global")
    if tensor:
        if q["dim"] is None:
            ctx.error(f"{what}: missing .dim")
        parse_tensor_address(stmt.operands[0] if stmt.operands else "")
    else:
        _forbid(ctx, q, ("dim", "mode"), what)
    return []


def _rw_commit_wait(stmt: Stmt, ctx: Ctx, op: str) -> list[str]:
    what = stmt.opcode
    toks = _split_opcode(stmt, 4)
    q = _classify(ctx, toks, {"read": {"read"}}, what)
    if op == "commit_group":
        _forbid(ctx, q, ("read",), what)
        _expect_ops(ctx, stmt, 0, what)
        if ctx.mode != "check":
            return []
        return ctx.call(LIB["bulk_commit"], [("b64", _bulk_state(ctx))])
    _expect_ops(ctx, stmt, 1, what)
    if not _is_imm(stmt.operands[0]):
        ctx.error(f"{what}: N must be an integer constant")
    n = parse_int(stmt.operands[0])
    if n < 0:
        ctx.error(f"{what}: N must be non-negative")
    if ctx.mode != "check":
        return []
    flags = _base_flags(ctx) | (AF_READ if q["read"] else 0)
    return ctx.call(
        LIB["bulk_wait"],
        [("b64", _bulk_state(ctx)), ("b32", str(n)), ("b32", _hex(flags))],
    )


# ---------------------------------------------------------------------------------------------
# st.async / red.async
# ---------------------------------------------------------------------------------------------

_ST_TYPES = {"b32", "b64", "b128", "u32", "u64", "s32", "s64", "f32", "f64"}
_RED_ASYNC = {
    "inc": {"u32"},
    "dec": {"u32"},
    "min": {"u32", "s32"},
    "max": {"u32", "s32"},
    "and": {"b32"},
    "or": {"b32"},
    "xor": {"b32"},
    "add": {"u32", "s32", "u64"},
}
_NEWER_ASYNC = {
    "release": "sm_100",
    "global": "sm_100",
    "gpu": "sm_100",
    "sys": "sm_100",
}


def _b32_reg(ctx: Ctx, op: str, what: str) -> tuple[list[str], str]:
    if _is_imm(op):
        r = ctx.new_reg("b32")
        return [f"mov.b32 \t{r}, {_hex(parse_int(op))};"], r
    if _reg_bits(ctx, op, what) != 32:
        ctx.error(f"{what}: {op} must be a 32-bit register")
    return _copy_reg(ctx, op, 32)


def _b64_reg(ctx: Ctx, op: str, what: str) -> tuple[list[str], str]:
    if _is_imm(op):
        r = ctx.new_reg("b64")
        return [f"mov.b64 \t{r}, {_hex64(parse_int(op))};"], r
    if _reg_bits(ctx, op, what) != 64:
        ctx.error(f"{what}: {op} must be a 64-bit register")
    return _copy_reg(ctx, op, 64)


def _rw_st_async(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    toks = _split_opcode(stmt, 2)
    for t in toks:
        if t in _NEWER_ASYNC:
            ctx.error(f"{what}: .{t} requires {_NEWER_ASYNC[t]}")
    q = _classify(
        ctx,
        toks,
        {
            "sem": {"weak"},
            "space": {"shared::cluster"},
            "completion": {"mbarrier::complete_tx::bytes"},
            "vec": {"v2", "v4"},
            "type": _ST_TYPES,
        },
        what,
    )
    if q["completion"] is None or q["type"] is None:
        ctx.error(f"{what}: needs .mbarrier::complete_tx::bytes and a type")
    ty, vec = q["type"], q["vec"]
    bits = 128 if ty == "b128" else int(ty[1:])
    if vec == "v4" and bits != 32:
        ctx.error(f"{what}: .v4 is only valid with 32-bit types")
    if vec is not None and bits == 128:
        ctx.error(f"{what}: .{vec} is not valid with .b128")
    _expect_ops(ctx, stmt, 3, what)
    ops = stmt.operands
    out: list[str] = []
    elems = parse_vector(ops[1]) if vec else [ops[1]]
    nvec = {None: 1, "v2": 2, "v4": 4}[vec]
    if len(elems) != nvec:
        ctx.error(f"{what}: expected {nvec} source value(s)")
    nbytes = bits // 8 * nvec
    lo, hi = "0", "0"
    if bits == 128:
        if _reg_bits(ctx, elems[0], what) != 128:
            ctx.error(f"{what}: .b128 source {elems[0]} must be a 128-bit register")
        lo, hi = ctx.new_reg("b64"), ctx.new_reg("b64")
        out.append(f"mov.b128 \t{{{lo}, {hi}}}, {elems[0]};")
    elif bits == 64:
        regs = []
        for e in elems:
            ln, r = _b64_reg(ctx, e, what)
            out += ln
            regs.append(r)
        lo = regs[0]
        hi = regs[1] if nvec == 2 else "0"
    else:
        regs = []
        for e in elems:
            ln, r = _b32_reg(ctx, e, what)
            out += ln
            regs.append(r)
        zero = ctx.new_reg("b32")
        out.append(f"mov.b32 \t{zero}, 0;")
        regs += [zero] * (4 - len(regs))
        lo, hi = ctx.new_reg("b64"), ctx.new_reg("b64")
        out.append(f"mov.b64 \t{lo}, {{{regs[0]}, {regs[1]}}};")
        out.append(f"mov.b64 \t{hi}, {{{regs[2]}, {regs[3]}}};")
    l1, addr = _smem(ctx, ops[0], q["space"])
    l2, mbar = _smem(ctx, ops[2], q["space"])
    flags = _base_flags(ctx) | nbytes
    return [
        *out,
        *l1,
        *l2,
        *ctx.call(
            LIB["st_async"],
            [
                ("b32", addr),
                ("b64", lo),
                ("b64", hi),
                ("b32", mbar),
                ("b32", _hex(flags)),
            ],
        ),
    ]


def _rw_red_async(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    toks = _split_opcode(stmt, 2)
    for t in toks:
        if t in _NEWER_ASYNC:
            ctx.error(f"{what}: .{t} requires {_NEWER_ASYNC[t]}")
    q = _classify(
        ctx,
        toks,
        {
            "sem": {"relaxed"},
            "scope": {"cluster"},
            "space": {"shared::cluster"},
            "completion": {"mbarrier::complete_tx::bytes"},
            "op": set(RED_OP),
            "type": {"b32", "u32", "s32", "u64"},
        },
        what,
    )
    if q["sem"] is None or q["scope"] is None or q["op"] is None or q["type"] is None:
        ctx.error(f"{what}: needs .relaxed.cluster, .op and .type")
    if q["completion"] is None:
        # The forms without an mbarrier (.release / .global / .gpu / .sys) are sm_100+.
        ctx.error(f"{what}: the sm_90 form needs .mbarrier::complete_tx::bytes")
    op, ty = q["op"], q["type"]
    if ty not in _RED_ASYNC[op]:
        ctx.error(f"{what}: .{op}.{ty} is not a valid red.async reduction")
    _expect_ops(ctx, stmt, 3, what)
    ops = stmt.operands
    l1, addr = _smem(ctx, ops[0], q["space"])
    l2, val = _val64(ctx, ops[1], what)
    l3, mbar = _smem(ctx, ops[2], q["space"])
    flags = (
        _base_flags(ctx) | (RED_OP[op] << AF_OP_SHIFT) | (RED_TYPE[ty] << AF_TYPE_SHIFT)
    )
    return [
        *l1,
        *l2,
        *l3,
        *ctx.call(
            LIB["red_async"],
            [("b32", addr), ("b64", val), ("b32", mbar), ("b32", _hex(flags))],
        ),
    ]


# ---------------------------------------------------------------------------------------------
# tensormap.*
# ---------------------------------------------------------------------------------------------

_FIELD3_MAX = {
    "elemtype": 12,
    "interleave_layout": 2,
    "swizzle_mode": 3,
    "fill_mode": 1,
}
_FIELD3_NEWER = {
    "elemtype": "sm_100 (values 13-15)",
    "swizzle_mode": "sm_103a (value 4)",
}


def _rw_tensormap(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    toks = _split_opcode(stmt, 1)
    if not toks:
        ctx.error(f"{what}: missing operation")
    op = toks[0]
    ops = stmt.operands

    if op == "replace":
        q = _classify(
            ctx,
            toks[1:],
            {
                "mode": {"tile"},
                "field": set(TMR_FIELD),
                "alloc": set(ALLOC_CLASS),
                "space": {"global", "shared::cta", "shared"},
                "size": {"b1024"},
                "type": {"b32", "b64"},
            },
            what,
        )
        if q["mode"] is None or q["size"] is None:
            ctx.error(f"{what}: needs .tile and .b1024")
        field, alloc = q["field"], q["alloc"]
        space = {
            "global": "global",
            "shared::cta": "shared",
            "shared": "shared",
            None: None,
        }[q["space"]]
        if field is None:
            if alloc not in ("alloc_small", "alloc_large") or q["type"] is not None:
                ctx.error(
                    f"{what}: without a field only .alloc_small / .alloc_large are valid"
                )
            _expect_ops(ctx, stmt, 1, what)
            l1, t = _tmap_addr(ctx, ops[0], space, what)
            return [
                *l1,
                *ctx.call(
                    LIB["tmap_replace"],
                    [
                        ("b64", t),
                        ("b32", str(TMR_ALLOC_ONLY)),
                        ("b32", "0"),
                        ("b64", "0"),
                        ("b32", _hex(ALLOC_CLASS[alloc])),
                    ],
                ),
            ]
        if alloc is not None and field != "global_address":
            ctx.error(f"{what}: .{alloc} is only valid with .global_address")
        want = "b64" if field in ("global_address", "global_stride") else "b32"
        if q["type"] != want:
            ctx.error(f"{what}: .{field} needs .{want}")
        ord_ = "0"
        if field in ("box_dim", "global_dim", "global_stride", "element_stride"):
            _expect_ops(ctx, stmt, 3, what)
            if not _is_imm(ops[1]):
                ctx.error(f"{what}: ord must be an integer constant")
            o = parse_int(ops[1])
            if not 0 <= o <= (3 if field == "global_stride" else 4):
                ctx.error(f"{what}: ord {o} out of range")
            ord_ = str(o)
            val = ops[2]
        else:
            _expect_ops(ctx, stmt, 2, what)
            val = ops[1]
        if field in _FIELD3_MAX:
            if not _is_imm(val):
                ctx.error(f"{what}: new_val of .{field} must be an immediate")
            v = parse_int(val)
            if v > _FIELD3_MAX[field] and field in _FIELD3_NEWER and v <= 15:
                ctx.error(f"{what}: .{field} value {v} requires {_FIELD3_NEWER[field]}")
            if not 0 <= v <= _FIELD3_MAX[field]:
                ctx.error(f"{what}: .{field} value {v} out of range")
        l1, t = _tmap_addr(ctx, ops[0], space, what)
        l2, nv = _val64(ctx, val, what)
        cls = ALLOC_CLASS[alloc] if alloc else 0
        return [
            *l1,
            *l2,
            *ctx.call(
                LIB["tmap_replace"],
                [
                    ("b64", t),
                    ("b32", str(TMR_FIELD[field])),
                    ("b32", ord_),
                    ("b64", nv),
                    ("b32", _hex(cls)),
                ],
            ),
        ]

    if op == "cp_fenceproxy":
        toks2 = toks[1:]
        sp = [t for t in toks2 if t in ("global", "shared::cta")]
        q = _classify(
            ctx,
            [t for t in toks2 if t not in ("global", "shared::cta")],
            {
                "proxy": {"tensormap::generic"},
                "sem": {"release"},
                "scope": set(SCOPE_MAP),
                "sync": {"sync"},
                "aligned": {"aligned"},
            },
            what,
        )
        if sp != ["global", "shared::cta"] or any(v is None for v in q.values()):
            ctx.error(
                f"{what}: expected .global.shared::cta.tensormap::generic.release.scope.sync.aligned"
            )
        _expect_ops(ctx, stmt, 3, what)
        if not _is_imm(ops[2]) or parse_int(ops[2]) != 128:
            ctx.error(f"{what}: size must be the constant 128")
        l1, dst = _gen(ctx, ops[0], "global")
        l2, src = _smem(ctx, ops[1], "shared::cta")
        return [
            *l1,
            *l2,
            *ctx.call(LIB["tmap_copy"], [("b64", dst), ("b32", src), ("b32", "128")]),
            f"fence.acq_rel.{SCOPE_MAP[q['scope']]};",
        ]

    if op == "query":
        q = _classify(
            ctx,
            toks[1:],
            {
                "what": {"alloc_class_large"},
                "space": {"global", "shared::cta", "const", "param", "param::entry"},
                "size": {"b1024"},
            },
            what,
        )
        if q["what"] is None or q["size"] is None:
            ctx.error(
                f"{what}: expected tensormap.query.alloc_class_large{{.ss}}.b1024"
            )
        _expect_ops(ctx, stmt, 2, what)
        space = {
            "global": "global",
            "shared::cta": "shared",
            "const": "const",
            "param": "param",
            "param::entry": "param",
            None: None,
        }[q["space"]]
        l1, t = _tmap_addr(ctx, ops[1], space, what)
        r = ctx.new_reg("b32")
        return [
            *l1,
            *ctx.call(LIB["tmap_query_large"], [("b64", t)], ret=("b32", r)),
        ] + _set_pred(ctx, ops[0], r, what)

    ctx.error(f"{what}: unsupported tensormap operation .{op}")
    raise AssertionError  # unreachable


def _rw_prefetch_tensormap(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    q = _classify(
        ctx,
        _split_opcode(stmt, 1),
        {"space": {"const", "param"}, "tm": {"tensormap"}},
        what,
    )
    if q["tm"] is None:
        ctx.error(f"{what}: not a tensormap prefetch")
    _expect_ops(ctx, stmt, 1, what)
    return []  # prefetch hint: no memory effect


# ---------------------------------------------------------------------------------------------
# fence / membar
# ---------------------------------------------------------------------------------------------

_SEMS = {"sc", "acq_rel", "acquire", "release"}
_SYNC_RESTRICT = {
    "sync_restrict::shared::cta",
    "sync_restrict::shared::cluster",
    "sync_restrict::shared::cluster::read",
    "sync_restrict::read",
}


def _fence_needs_rewrite(toks: list[str]) -> bool:
    """Whether a fence / membar (tokens after the head) is an sm_90 form this rule handles."""
    if not toks:
        return False
    if toks[0] == "proxy":
        if len(toks) == 2 and toks[1] == "alias":
            return False  # fence.proxy.alias / membar.proxy.alias: native sm_70
        return len(toks) >= 2
    if toks[0] == "alias":
        return True
    if any(t in ("cluster", "acquire", "release", "mbarrier_init") for t in toks):
        return True
    return any(t.startswith("sync_restrict") for t in toks)


def _fence_scope(ctx: Ctx, scope: str | None, what: str) -> str:
    if scope is None:
        ctx.error(f"{what}: missing scope")
    return SCOPE_MAP[scope]


def _rw_fence(stmt: Stmt, ctx: Ctx) -> list[str]:
    what = stmt.opcode
    head = stmt.opcode.split(".", 1)[0]
    toks = _split_opcode(stmt, 1)
    ops = stmt.operands
    if head == "membar" and toks[:2] != ["proxy", "async"]:
        # Only membar.proxy.async{.space} reaches here (same grammar as fence.proxy.async).
        ctx.error(f"{what}: unsupported membar form")
    if toks[0] == "proxy":
        rest = toks[1:]
        kind = rest[0]
        if kind in NEWER:
            ctx.error(f"{what}: .{kind} requires {NEWER[kind]}")
        if kind == "tensormap::generic":
            q = _classify(
                ctx,
                rest[1:],
                {"sem": {"release", "acquire"}, "scope": set(SCOPE_MAP)},
                what,
            )
            if q["sem"] is None:
                ctx.error(f"{what}: needs .release or .acquire")
            sc = _fence_scope(ctx, q["scope"], what)
            if q["sem"] == "acquire":
                _expect_ops(ctx, stmt, 2, what)
                parse_address(ops[0])
                if not _is_imm(ops[1]) or parse_int(ops[1]) != 128:
                    ctx.error(f"{what}: size must be the constant 128")
            else:
                _expect_ops(ctx, stmt, 0, what)
            # The tensor-map proxy is emulated with generic loads (lib_async.cu tmap_read), so a
            # generic acq_rel fence at the same scope gives the required ordering.
            return [f"fence.acq_rel.{sc};"]
        if kind == "async::generic":
            q = _classify(
                ctx,
                rest[1:],
                {
                    "sem": {"acquire", "release"},
                    "restrict": _SYNC_RESTRICT,
                    "scope": {"cluster"},
                },
                what,
            )
            if any(v is None for v in q.values()):
                ctx.error(
                    f"{what}: expected .acquire|.release.sync_restrict::*.cluster"
                )
            ok = {
                ("acquire", "sync_restrict::shared::cluster"),
                ("release", "sync_restrict::shared::cta"),
                ("release", "sync_restrict::shared::cluster::read"),
            }
            if (q["sem"], q["restrict"]) not in ok:
                ctx.error(f"{what}: invalid .{q['sem']} / .{q['restrict']} combination")
            _expect_ops(ctx, stmt, 0, what)
            return ["fence.acq_rel.cta;"]
        if kind in ("async", "alias"):
            q = _classify(
                ctx,
                rest[1:],
                {
                    "space": {"global", "shared::cta", "shared::cluster"},
                    "sem": _SEMS,
                    "scope": set(SCOPE_MAP),
                },
                what,
            )
            _expect_ops(ctx, stmt, 0, what)
            if kind == "alias" and q["space"] is not None:
                ctx.error(f"{what}: .alias takes no state space")
            pre = ["fence.proxy.alias;"] if kind == "alias" else []
            if q["scope"] is not None:
                sem = "sc" if q["sem"] == "sc" else "acq_rel"
                return [*pre, f"fence.{sem}.{SCOPE_MAP[q['scope']]};"]
            if q["sem"] is not None:
                ctx.error(f"{what}: .{q['sem']} needs a scope")
            # Bi-directional async-proxy fence. The async proxy is emulated by generic accesses of
            # the issuing thread, so program order already holds; a CTA (shared) or GPU (global /
            # all spaces) acq_rel fence is kept as a conservative native equivalent.
            sc = "cta" if q["space"] in ("shared::cta", "shared::cluster") else "gpu"
            return [*pre, f"fence.acq_rel.{sc};"]
        ctx.error(f"{what}: unsupported proxy kind .{kind}")
    if toks[0] == "alias":
        # fence.alias.sync_restrict::read.release.sys: releases prior reads across virtual aliases
        # [inferred: fence.proxy.alias plus a sys acq_rel fence is at least as strong].
        q = _classify(
            ctx,
            toks[1:],
            {"restrict": {"sync_restrict::read"}, "sem": {"release"}, "scope": {"sys"}},
            what,
        )
        if any(v is None for v in q.values()):
            ctx.error(f"{what}: expected fence.alias.sync_restrict::read.release.sys")
        _expect_ops(ctx, stmt, 0, what)
        return ["fence.proxy.alias;", "fence.acq_rel.sys;"]
    q = _classify(
        ctx,
        toks,
        {
            "op": {"mbarrier_init"},
            "sem": _SEMS,
            "restrict": _SYNC_RESTRICT,
            "scope": set(SCOPE_MAP),
        },
        what,
    )
    _expect_ops(ctx, stmt, 0, what)
    sc = _fence_scope(ctx, q["scope"], what)
    if q["op"] is not None:
        if (
            q["sem"] != "release"
            or q["scope"] != "cluster"
            or q["restrict"] is not None
        ):
            ctx.error(f"{what}: expected fence.mbarrier_init.release.cluster")
        return [f"fence.acq_rel.{sc};"]
    if q["restrict"] is not None:
        ok = {
            ("acquire", "sync_restrict::shared::cluster"),
            ("release", "sync_restrict::shared::cta"),
        }
        if (q["sem"], q["restrict"]) not in ok or q["scope"] != "cluster":
            ctx.error(f"{what}: invalid .sync_restrict thread fence")
        return [f"fence.acq_rel.{sc};"]
    sem = "sc" if q["sem"] == "sc" else "acq_rel"
    return [f"fence.{sem}.{sc};"]


# ---------------------------------------------------------------------------------------------
# Rule module contract
# ---------------------------------------------------------------------------------------------


def _kind(stmt: Stmt) -> str | None:
    toks = stmt.opcode.split(".")
    head = toks[0]
    if head == "mbarrier":
        return "mbarrier"
    if toks[:4] == ["cp", "async", "mbarrier", "arrive"]:
        return "cp_async_mbar_arrive"
    if toks[:3] == ["cp", "async", "bulk"]:
        nxt = toks[3] if len(toks) > 3 else ""
        if nxt == "tensor":
            return "tensor_copy"
        if nxt == "prefetch":
            return (
                "prefetch_tensor"
                if len(toks) > 4 and toks[4] == "tensor"
                else "prefetch"
            )
        if nxt in ("commit_group", "wait_group"):
            return nxt
        return "bulk"
    if toks[:4] == ["cp", "reduce", "async", "bulk"]:
        return (
            "tensor_reduce" if len(toks) > 4 and toks[4] == "tensor" else "bulk_reduce"
        )
    if head in ("st", "red") and len(toks) > 1 and toks[1] == "async":
        return f"{head}_async"
    if head == "tensormap":
        return "tensormap"
    if head == "prefetch" and "tensormap" in toks[1:]:
        return "prefetch_tensormap"
    if head in ("fence", "membar") and _fence_needs_rewrite(toks[1:]):
        if head == "membar" and toks[1:3] != ["proxy", "async"]:
            return None
        return "fence"
    return None


def scan(func: Func, ctx: Ctx) -> None:
    """Nothing to pre-compute: check-mode state is created on first use (see _bulk_state)."""
    return None


def function_prologue(ctx: Ctx) -> list[str]:
    st = ctx.func_state(STATE_OWNER)
    if "bulk_reg" not in st:
        return []
    t = ctx.new_reg("b64")
    return [
        f"mov.u64 \t{t}, {st['bulk_sym']};",
        f"cvta.local.u64 \t{st['bulk_reg']}, {t};",
        *ctx.call(LIB["bulk_state_init"], [("b64", st["bulk_reg"])]),
    ]


def match(stmt: Stmt) -> bool:
    return _kind(stmt) is not None


def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]:
    kind = _kind(stmt)
    if kind == "mbarrier":
        return _rw_mbarrier(stmt, ctx)
    if kind == "cp_async_mbar_arrive":
        return _rw_cp_async_mbar_arrive(stmt, ctx)
    if kind == "bulk":
        return _rw_bulk(stmt, ctx)
    if kind == "bulk_reduce":
        return _rw_bulk_reduce(stmt, ctx)
    if kind == "tensor_copy":
        return _rw_tensor(stmt, ctx, reduce=False)
    if kind == "tensor_reduce":
        return _rw_tensor(stmt, ctx, reduce=True)
    if kind == "prefetch":
        return _rw_prefetch(stmt, ctx, tensor=False)
    if kind == "prefetch_tensor":
        return _rw_prefetch(stmt, ctx, tensor=True)
    if kind in ("commit_group", "wait_group"):
        return _rw_commit_wait(stmt, ctx, kind)
    if kind == "st_async":
        return _rw_st_async(stmt, ctx)
    if kind == "red_async":
        return _rw_red_async(stmt, ctx)
    if kind == "tensormap":
        return _rw_tensormap(stmt, ctx)
    if kind == "prefetch_tensormap":
        return _rw_prefetch_tensormap(stmt, ctx)
    if kind == "fence":
        return _rw_fence(stmt, ctx)
    ctx.error(f"rules_async: no rewrite for {stmt.opcode}")
    raise AssertionError  # unreachable
