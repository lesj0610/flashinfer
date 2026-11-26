"""E1 delayed delivery and E2 operation counters (translation side).

Enabled only by ``translate(async_delay=True)`` / ``translate(count=True)`` (CLI ``--async-delay``
/ ``--count``, nvcc_emu ``WGEMU_ASYNC_DELAY=1`` / ``WGEMU_COUNT=1``). Then ``core.translate``
puts this module in front of the rule list and calls :func:`before` for every statement; with
both off nothing here runs and the output is unchanged. Library side: ``runtime/lib_async.cu``,
layouts and constants: "E1 delayed delivery" / "E2 counters" in
``include/wgmma_emu/async_device.cuh`` (keep the constants below in sync with it).

E1 (``async_delay``), for modules whose PTX contains ``cp.async``:

* every entry starts with ``wgemu_dl_init`` (per-CTA header in module ``.shared``, pool slot in
  the module ``.global`` pool) and calls ``wgemu_dl_exit`` before each ``ret``; every function
  calls it before each ``exit``;
* ``cp.async.bulk`` global->shared and ``cp.async.bulk.tensor`` loads call the ``wgemu_dl_*``
  variants (rules_async): the destination gets poison, the data is kept in the slot;
* every ``mbarrier.test_wait`` / ``try_wait`` is followed by ``wgemu_dl_observe`` (rules_async):
  a successful wait delivers the copies that complete the observed phase;
* non-bulk ``cp.async`` / ``commit_group`` / ``wait_group`` / ``wait_all`` (native on sm_80)
  become ``wgemu_dl_cp_*`` calls (this module's rule); ``cp.async.mbarrier.arrive`` ties the
  thread's pending copies to the barrier phase (rules_async).

E2 (``count``): the module gets ``.weak .global __wgemu_global_count_v1`` and the counted
instructions add to it; host/shim.c prints one ``WGEMU-COUNT`` line per launch and zeroes it.

* ``wgmma[...]``: ``wgmma.mma_async`` executions, one per warpgroup;
* ``bulk_bytes``: size operand of every ``cp.async.bulk`` / ``cp.reduce.async.bulk`` (all
  directions; prefetch excluded), per executing thread;
* ``tma_bytes``: box bytes of every ``cp.async.bulk.tensor`` / ``cp.reduce.async.bulk.tensor``
  (prefetch excluded), per executing thread;
* ``cpasync_bytes``: cp-size of every non-bulk ``cp.async``, per executing thread;
* ``ldg_bytes``: bytes of every ``ld.global`` / ``ldu.global`` (explicit ``.global`` only),
  per executing thread;
* ``mbar_waits``: ``mbarrier.test_wait`` / ``try_wait`` executions that returned true.
"""

from __future__ import annotations

from .core import Ctx, Func, Stmt, TranslateError, parse_int

# ---- constants shared with include/wgmma_emu/async_device.cuh --------------------------------
AF_CHECK = 0x80000000
AF_TRAP = 0x40000000
AF_DL_ALL = 0x1

DL_SMEM_BYTES = 233472
DL_SLOT_RECORDS = 125184
DL_REC_BYTES = 32
DL_HDR_BYTES = 32
DL_MAX_RECORDS = 0xFFFE
DL_HDR_KEY = "async_delay_hdr"
DL_POOL_KEY = "async_delay_pool"

CNT_KEY = "count_v1"  # symbol __wgemu_global_count_v1 (read by host/shim.c)
CNT_BYTES = 1088
CNT_BULK = 0
CNT_TMA = 8
CNT_CPASYNC = 16
CNT_LDG = 24
CNT_MBAR_WAITS = 32

# wgmma code: bit 0 valid, 1 sparse, 2-10 N, 11-19 K, 20-23 D type, 24-27 A type, 28-31 B type
# (type = index in WGMMA_TYPES; host/shim.c decodes it with the same table).
WGMMA_TYPES = ("f16", "bf16", "tf32", "f32", "e4m3", "e5m2", "s8", "u8", "s32", "b1")

LIB = {
    "dl_init": "wgemu_dl_init",
    "dl_exit": "wgemu_dl_exit",
    "dl_tma_load": "wgemu_dl_tma_load",
    "dl_bulk_g2s": "wgemu_dl_bulk_g2s",
    "dl_observe": "wgemu_dl_observe",
    "dl_cp_async": "wgemu_dl_cp_async",
    "dl_cp_commit": "wgemu_dl_cp_commit",
    "dl_cp_wait": "wgemu_dl_cp_wait",
    "dl_cp_mbar_arrive": "wgemu_dl_cp_mbar_arrive",
    "count_add": "wgemu_count_add",
    "count_wgmma": "wgemu_count_wgmma",
    "tma_bytes": "wgemu_tma_bytes",
}

_TYPE_BYTES = {
    "b8": 1,
    "u8": 1,
    "s8": 1,
    "b16": 2,
    "u16": 2,
    "s16": 2,
    "f16": 2,
    "bf16": 2,
    "b32": 4,
    "u32": 4,
    "s32": 4,
    "f32": 4,
    "f16x2": 4,
    "bf16x2": 4,
    "b64": 8,
    "u64": 8,
    "s64": 8,
    "f64": 8,
    "b128": 16,
}
_VEC = {"v2": 2, "v4": 4, "v8": 8}


def slot_bytes(records: int) -> int:
    """Bytes of one E1 pool slot (WGEMU_DL_SLOT_BYTES)."""
    return DL_SLOT_RECORDS + DL_REC_BYTES * records + DL_SMEM_BYTES


def check_shape(slots: int, records: int) -> None:
    if not 1 <= slots <= 65536:
        raise TranslateError(f"E1 pool: slot count {slots} outside 1..65536")
    if not 1 <= records <= DL_MAX_RECORDS:
        raise TranslateError(
            f"E1 pool: record count {records} outside 1..{DL_MAX_RECORDS}"
        )


def _hex(v: int) -> str:
    return f"0x{v & 0xFFFFFFFF:08X}"


def _flags(ctx: Ctx) -> int:
    f = AF_CHECK if ctx.mode == "check" else 0
    if ctx.trap:
        f |= AF_TRAP
    return f


def delay_on(ctx: Ctx) -> bool:
    """E1 applies to this module: --async-delay and the module issues cp.async of some form."""
    if not ctx.async_delay:
        return False
    st = ctx.module_state()
    if "_dl_on" not in st:
        st["_dl_on"] = "cp.async" in (ctx.module_text or "")
    return bool(st["_dl_on"])


def hdr(ctx: Ctx) -> tuple[list[str], str]:
    """b32 register with the shared address of the E1 per-CTA header."""
    sym = ctx.shared_var(DL_HDR_KEY, DL_HDR_BYTES, 16)
    r = ctx.new_reg("b32")
    return [f"mov.u32 \t{r}, {sym};"], r


def _global_addr(ctx: Ctx, sym: str) -> tuple[list[str], str]:
    t = ctx.new_reg("b64")
    g = ctx.new_reg("b64")
    return [f"mov.u64 \t{t}, {sym};", f"cvta.global.u64 \t{g}, {t};"], g


def count_add(ctx: Ctx, off: int, val: str) -> list[str]:
    """E2: counter at byte offset ``off`` += ``val`` (b32 register or immediate), summed over
    the active lanes. No code when counting is off."""
    if not ctx.count:
        return []
    lines, c = _global_addr(ctx, ctx.global_var(CNT_KEY, CNT_BYTES, 16))
    return lines + ctx.call(
        LIB["count_add"], [("b64", c), ("b32", str(off)), ("b32", val)]
    )


def count_tma(ctx: Ctx, tmap: str, flags: int) -> list[str]:
    """E2: tma_bytes += box bytes of the tensor map at generic address ``tmap``."""
    if not ctx.count:
        return []
    r = ctx.new_reg("b32")
    lines = ctx.call(
        LIB["tma_bytes"], [("b64", tmap), ("b32", _hex(flags))], ret=("b32", r)
    )
    return lines + count_add(ctx, CNT_TMA, r)


def after_wait(ctx: Ctx, bar: str, parity: str, ok: str, flags: int) -> list[str]:
    """After an mbarrier test_wait / try_wait with result ``ok`` (b32 0/1) on phase ``parity``:
    E1 delivery of the copies of an observed phase, E2 count of successful waits."""
    out: list[str] = []
    if delay_on(ctx):
        lh, h = hdr(ctx)
        out += lh + ctx.call(
            LIB["dl_observe"],
            [
                ("b32", h),
                ("b32", bar),
                ("b32", parity),
                ("b32", ok),
                ("b32", _hex(flags)),
            ],
        )
    return out + count_add(ctx, CNT_MBAR_WAITS, ok)


# ---- statement classification ------------------------------------------------------------


def _cp_async_kind(stmt: Stmt) -> str | None:
    """'copy', 'commit', 'wait' or 'wait_all' for non-bulk cp.async forms (native on sm_80)."""
    t = stmt.opcode.split(".")
    if t[:2] != ["cp", "async"] or len(t) < 3:
        return None
    if t[2] in ("ca", "cg"):
        return "copy"
    if t[2] == "commit_group":
        return "commit"
    if t[2] == "wait_group":
        return "wait"
    if t[2] == "wait_all":
        return "wait_all"
    return None


def _wgmma_code(ctx: Ctx, toks: list[str]) -> int:
    shape = next((t for t in toks if t.startswith("m64n")), None)
    if shape is None or "k" not in shape:
        ctx.error(f"wgmma count: no m64nNkK shape in {'.'.join(toks)}")
        raise AssertionError  # unreachable
    n_s, k_s = shape[4:].split("k", 1)
    tys = [t for t in toks[toks.index(shape) + 1 :] if t in WGMMA_TYPES]
    if not n_s.isdigit() or not k_s.isdigit() or len(tys) < 3:
        ctx.error(f"wgmma count: cannot read shape/types of {'.'.join(toks)}")
    n, k = int(n_s), int(k_s)
    if n > 511 or k > 511:
        ctx.error(f"wgmma count: shape {shape} does not fit the counter code")
    d, a, b = (WGMMA_TYPES.index(x) for x in tys[:3])
    sp = 1 if "sp" in toks else 0
    return 1 | sp << 1 | n << 2 | k << 11 | d << 20 | a << 24 | b << 28


def _ld_global_bytes(toks: list[str]) -> int | None:
    if toks[0] not in ("ld", "ldu") or "global" not in toks:
        return None
    vec = next((_VEC[t] for t in toks if t in _VEC), 1)
    ty = next((_TYPE_BYTES[t] for t in reversed(toks) if t in _TYPE_BYTES), None)
    return None if ty is None else vec * ty


def _guarded(ctx: Ctx, stmt: Stmt, lines: list[str]) -> list[str]:
    if not lines or not stmt.guard:
        return lines
    g = stmt.guard
    neg = "@" + g[2:] if g.startswith("@!") else "@!" + g[1:]
    lab = ctx.new_label("INSTR_SKIP")
    return [f"{neg} bra \t{lab};", *lines, f"{lab}:"]


def before(stmt: Stmt, ctx: Ctx) -> list[str]:
    """Lines placed before ``stmt`` whatever rule rewrites it (core calls this when E1 or E2
    is on): E2 counters of wgmma / ld.global / cp.async, E1 wgemu_dl_exit before exit and
    before ret of an entry. Guarded statements get guarded lines."""
    toks = stmt.opcode.split(".")
    out: list[str] = []
    if ctx.count:
        if toks[:2] == ["wgmma", "mma_async"]:
            sym = ctx.global_var(CNT_KEY, CNT_BYTES, 16)
            lines, c = _global_addr(ctx, sym)
            code = _wgmma_code(ctx, toks)
            out += lines + ctx.call(
                LIB["count_wgmma"], [("b64", c), ("b32", _hex(code))]
            )
        elif _cp_async_kind(stmt) == "copy":
            if len(stmt.operands) < 3:
                ctx.error(f"{stmt.opcode}: expected [dst], [src], cp-size, ...")
            out += count_add(ctx, CNT_CPASYNC, _hex(parse_int(stmt.operands[2])))
        else:
            nb = _ld_global_bytes(toks)
            if nb is not None:
                out += count_add(ctx, CNT_LDG, str(nb))
    if delay_on(ctx):
        is_entry = ctx.func is not None and ctx.func.is_entry
        if toks[0] == "exit" or (toks[0] == "ret" and is_entry):
            out += _hdr_call(ctx, LIB["dl_exit"], [], _flags(ctx))
    return _guarded(ctx, stmt, out)


# ---- rule module contract (E1 non-bulk cp.async) ------------------------------------------

_STATE = {"delay": False}


def scan(func: Func, ctx: Ctx) -> None:
    _STATE["delay"] = delay_on(ctx)


def function_prologue(ctx: Ctx) -> list[str]:
    """E1: per-CTA header and pool slot, at the start of every entry."""
    if not delay_on(ctx) or ctx.func is None or not ctx.func.is_entry:
        return []
    slots, records = ctx.dl_shape
    lh, h = hdr(ctx)
    lp, pool = _global_addr(
        ctx, ctx.global_var(DL_POOL_KEY, slots * slot_bytes(records), 16)
    )
    return [
        *lh,
        *lp,
        *ctx.call(
            LIB["dl_init"],
            [
                ("b32", h),
                ("b64", pool),
                ("b32", str(slots)),
                ("b32", str(records)),
                ("b32", _hex(_flags(ctx))),
            ],
        ),
    ]


def match(stmt: Stmt) -> bool:
    return bool(_STATE["delay"]) and _cp_async_kind(stmt) is not None


def _rw_copy(stmt: Stmt, ctx: Ctx) -> list[str]:
    from . import rules_async as ra

    what = stmt.opcode
    toks = stmt.opcode.split(".")[3:]
    spaces = [t for t in toks if t in ("shared", "shared::cta", "global")]
    if len(spaces) != 2 or spaces[1] != "global":
        ctx.error(f"{what}: expected .shared{{::cta}}.global")
    hint = "L2::cache_hint" in toks
    ops = [o.strip() for o in stmt.operands]
    extra = len(ops) - 3 - (1 if hint else 0)
    if len(ops) < 3 or extra not in (0, 1):
        ctx.error(f"{what}: expected [dst], [src], cp-size{{, src-size | ignore-src}}")
    cp_size = parse_int(ops[2])
    if cp_size not in (4, 8, 16) or (
        stmt.opcode.split(".")[2] == "cg" and cp_size != 16
    ):
        ctx.error(f"{what}: invalid cp-size {ops[2]}")
    l1, dst = ra._smem(ctx, ops[0], "shared")
    l2, src = ra._gen(ctx, ops[1], "global")
    out = l1 + l2
    size = _hex(cp_size)
    if extra:
        op = ops[3]
        pred = op.lstrip("!")
        if ctx.reg_type(pred) == "pred":
            # ignore-src: when true, no source bytes are read (zero fill).
            size = ctx.new_reg("b32")
            zero_first = not op.startswith("!")
            a, b = ("0", str(cp_size)) if zero_first else (str(cp_size), "0")
            out.append(f"selp.b32 \t{size}, {a}, {b}, {pred};")
        else:
            l3, size = ra._val32(ctx, op, what)
            out += l3
    return out + _hdr_call(
        ctx,
        LIB["dl_cp_async"],
        [("b32", dst), ("b64", src), ("b32", _hex(cp_size)), ("b32", size)],
        _flags(ctx),
    )


def _hdr_call(
    ctx: Ctx, fn: str, args: list[tuple[str, str]], flags: int | None
) -> list[str]:
    """call fn(hdr, *args{, flags})."""
    lh, h = hdr(ctx)
    full = [("b32", h), *args]
    if flags is not None:
        full.append(("b32", _hex(flags)))
    return lh + ctx.call(fn, full)


def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]:
    kind = _cp_async_kind(stmt)
    what = stmt.opcode
    if kind == "copy":
        return _rw_copy(stmt, ctx)
    if stmt.operands and kind in ("commit", "wait_all"):
        ctx.error(f"{what}: unexpected operands")
    if kind == "commit":
        return _hdr_call(ctx, LIB["dl_cp_commit"], [], None)
    if kind == "wait_all":
        return _hdr_call(
            ctx, LIB["dl_cp_wait"], [("b32", "0")], _flags(ctx) | AF_DL_ALL
        )
    if kind == "wait":
        if len(stmt.operands) != 1:
            ctx.error(f"{what}: expected one immediate operand")
        n = parse_int(stmt.operands[0])
        return _hdr_call(ctx, LIB["dl_cp_wait"], [("b32", str(n))], _flags(ctx))
    ctx.error(f"instrument: no rewrite for {what}")
    raise AssertionError  # unreachable
