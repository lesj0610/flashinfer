"""E4 shadow shared memory (translation side).

Enabled by ``translate(e4=True)`` (CLI ``--smem-global``, nvcc_emu ``WGEMU_SMEM_GLOBAL=1``). The CTA
shared window of every entry that touches shared memory lives in a per-CTA slot of a global pool;
shared addresses become virtual offsets from ``E4_ORIGIN``; real shared memory holds only the 32 B
header ``__wgemu_e4_hdr_v1`` that the library defines. Library side: ``include/wgmma_emu/e4_device.cuh``
and ``runtime/lib_async.cu`` built with ``-DWGEMU_SMEM_GLOBAL``; host side: ``host/shim.c``.

Pipeline (``core.translate``):

1. every function is translated by the ordinary rules; this module is the first rule of the list and
   rewrites the non-bulk ``cp.async`` family (``match`` / ``rewrite``);
2. :func:`apply` runs over all rewritten functions: classifies entries (E4 iff the call graph reaches
   a shared access, a shared symbol, ``%dynamic_smem_size`` / ``%total_smem_size``, an indirect call or
   a library function that references the header), lays the shared variables out in the virtual
   window, rewrites every remaining shared access to the shadow slot and adds the entry prologue,
   the ``wgemu_e4_exit`` hooks, the hidden ``__wgemu_e4_meta`` parameter and the per-entry marker
   global;
3. :func:`selfcheck` verifies the final text.

Unsupported shared forms are translation errors (fail closed), never silently kept.

Entry selection (``--smem-global-entries REGEX`` / ``WGEMU_E4_ENTRIES``): only entries whose mangled
name matches (``re.search``) are E4 candidates. The others are translated exactly as without E4
(real shared memory, ``plan_off`` fixes the set of such functions before translation: the entries
and the functions only they reach; the cp.async rewrite above is not applied to them). Their shared
declarations stay. A shared-touching function reachable from both kinds of entry fails closed, as
does a call from a real-shared function into a library function that uses the E4 header. A regex
that selects no entry leaves the whole module in the ordinary translation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import core, rules_misc
from .core import (
    Ctx,
    Stmt,
    TranslateError,
    parse_address,
    parse_int,
    parse_stmt,
    render_stmt,
)

# ---- constants shared with include/wgmma_emu/async_device.cuh (WGEMU_SMEM_GLOBAL) ----------------
E4_ORIGIN = 0x400
E4_WINDOW_MAX = 0x40000
E4_STRIDE = 0x40000
E4_SM90_BLOCK_MAX = 232448
E4_MAGIC = 0x57474534
E4_SLOTS_DEFAULT = 512
E4_SLOTS_MAX = 65535
E4_ALIGN_MAX = 1024

HDR = "__wgemu_e4_hdr_v1"
POOL = "__wgemu_e4_pool_v1"
LEASE = "__wgemu_e4_lease_v1"
STATS = "__wgemu_e4_stats_v1"
META = "__wgemu_e4_meta"
MARK_PREFIX = "__wgemu_e4f_"
E4_PREFIX = "__wgemu_e4"

LIB = {
    "enter": "wgemu_e4_enter",
    "exit": "wgemu_e4_exit",
    "cp_async": "wgemu_e4_cp_async",
    "cp_async_arrive": "wgemu_e4_cp_async_arrive",
    "dyn_smem": "wgemu_e4_dyn_smem",
    "total_smem": "wgemu_e4_total_smem",
}

# extern functions an E4 entry may call without the translator knowing their body
_EXTERN_OK = frozenset(
    {"vprintf", "malloc", "free", "__assertfail", "memcpy", "memset", "memmove"}
)

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

_SREG_RE = re.compile(
    r"%(?:dynamic_smem_size|total_smem_size|aggr_smem_size|reserved_smem_offset)(?![\w$])"
)
_REFID_RE = re.compile(r"(?<![\w$%.@])[A-Za-z_$][\w$]*")
_REG_RE = re.compile(r"%[A-Za-z_$][\w$]*")
_SYMREF_RE = re.compile(r"\[?\s*([A-Za-z_$][\w$]*)\s*(?:\+\s*(\d+)\s*)?\]?\Z")
_CALLEE_RE = re.compile(
    r"call(?:\.uni)?\s*(?:\([^)]*\)\s*,\s*)?([A-Za-z_$%][\w$]*)\s*,"
)
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_HDR_ADDR_RE = re.compile(r"\[\s*" + HDR + r"\s*(?:\+\s*\d+\s*)?\]")
_SHARED_DECL_TOK_RE = re.compile(r"(?<![\w$:])\.shared(?![\w$:])")


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def marker_name(entry: str) -> str:
    return f"{MARK_PREFIX}{fnv1a64(entry):016x}"


def compile_selector(pattern: str | None) -> "re.Pattern[str] | None":
    """None / empty: no selection (every shared-using entry is E4)."""
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as e:
        raise TranslateError(
            f"invalid E4 entry regex {pattern!r} (--smem-global-entries / WGEMU_E4_ENTRIES): {e}"
        ) from None


def check_shape(slots: int) -> None:
    if not 1 <= slots <= E4_SLOTS_MAX:
        raise TranslateError(
            f"E4 slots {slots} out of range (1..{E4_SLOTS_MAX}, WGEMU_E4_SLOTS)"
        )


# ---------------------------------------------------------------------------------------------
# Rule module part: non-bulk cp.async family (phase 1, runs with the other rules)
# ---------------------------------------------------------------------------------------------


def _kind(stmt: Stmt) -> str | None:
    t = stmt.opcode.split(".")
    if t[:2] != ["cp", "async"] or len(t) < 3:
        return None
    if t[2] in ("ca", "cg"):
        return "copy"
    if t[2] in ("commit_group", "wait_group", "wait_all"):
        return t[2]
    if t[2] == "mbarrier" and len(t) > 3 and t[3] == "arrive":
        return "arrive"
    return None


def match(stmt: Stmt) -> bool:
    return _kind(stmt) is not None


def _hex(v: int) -> str:
    return f"0x{v & 0xFFFFFFFF:08X}"


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
            # ignore-src: when true no source bytes are read (zero fill)
            size = ctx.new_reg("b32")
            zero_first = not op.startswith("!")
            a, b = ("0", str(cp_size)) if zero_first else (str(cp_size), "0")
            out.append(f"selp.b32 \t{size}, {a}, {b}, {pred};")
        else:
            l3, size = ra._val32(ctx, op, what)
            out += l3
    return out + ctx.call(
        LIB["cp_async"],
        [("b32", dst), ("b64", src), ("b32", _hex(cp_size)), ("b32", size)],
    )


def _rw_arrive(stmt: Stmt, ctx: Ctx) -> list[str]:
    from . import rules_async as ra

    what = stmt.opcode
    toks = stmt.opcode.split(".")[4:]
    bad = [t for t in toks if t not in ("noinc", "shared", "shared::cta", "b64")]
    if bad:
        ctx.error(f"{what}: unsupported qualifier .{bad[0]}")
    if len(stmt.operands) != 1:
        ctx.error(f"{what}: expected one address operand")
    l1, bar = ra._smem(ctx, stmt.operands[0], "shared")
    noinc = "1" if "noinc" in toks else "0"
    return l1 + ctx.call(LIB["cp_async_arrive"], [("b32", bar), ("b32", noinc)])


def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]:
    kind = _kind(stmt)
    what = stmt.opcode
    if kind == "copy":
        return _rw_copy(stmt, ctx)
    if kind == "arrive":
        return _rw_arrive(stmt, ctx)
    # copies complete inside wgemu_e4_cp_async: groups and waits have nothing left to do
    if kind in ("commit_group", "wait_all"):
        if stmt.operands:
            ctx.error(f"{what}: unexpected operands")
        return []
    if kind == "wait_group":
        if len(stmt.operands) != 1:
            ctx.error(f"{what}: expected one immediate operand")
        parse_int(stmt.operands[0])
        return []
    ctx.error(f"rules_smem: no rewrite for {what}")
    raise AssertionError  # unreachable


# ---------------------------------------------------------------------------------------------
# Module pass
# ---------------------------------------------------------------------------------------------


@dataclass
class _Var:
    key: tuple
    name: str
    size: int
    align: int
    extern: bool
    func: str | None = None  # function-scope variable: owning function
    off: int = 0


class _P:
    """One parsed body item: kind is stmt | call | decl | brace | label | other."""

    __slots__ = ("kind", "items", "stmt", "orig")

    def __init__(self, kind: str, items: list, stmt: Stmt | None, orig: str) -> None:
        self.kind = kind
        self.items = items
        self.stmt = stmt
        self.orig = orig


@dataclass
class _F:
    idx: int
    fp: "core._FuncPass"
    name: str
    is_entry: bool
    body: list[_P] = field(default_factory=list)
    prologue: list[_P] = field(default_factory=list)
    local_vars: dict[str, _Var] = field(default_factory=dict)
    callees: set[str] = field(default_factory=set)
    indirect: bool = False
    refs: set[tuple] = field(default_factory=set)
    uses_shared: bool = False
    has_exit: bool = False
    e4: bool = False
    bw: dict[str, str] = field(default_factory=dict)  # 'B' / 'W' -> register
    counts: dict[str, int] = field(default_factory=dict)
    vars: dict[str, _Var] = field(default_factory=dict)  # visible shared variables
    last_orig: str = ""
    ok64: set[str] = field(default_factory=set)
    off: bool = False  # stays in real shared memory (entry not selected, see plan_off)

    def count(self, k: str) -> None:
        self.counts[k] = self.counts.get(k, 0) + 1


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _fail(F: _F, msg: str, stmt: Stmt | None = None) -> None:
    ctx = F.fp.ctx
    loc = ctx._m.filename
    if stmt is not None and stmt.lineno:
        loc += f":{stmt.lineno}"
    tail = ""
    if stmt is not None:
        tail = f" [{' '.join(stmt.text.split())[:160]}]"
    if F.last_orig and (stmt is None or not stmt.lineno):
        tail += f" (from {F.last_orig})"
    raise core._located(f"{loc}: in {F.name}: E4 shadow shared memory: {msg}{tail}")


def _parse_items(F: _F, items: list) -> list[_P]:
    out: list[_P] = []
    i = 0
    n = len(items)
    while i < n:
        it = items[i]
        if isinstance(it, Stmt):
            out.append(_P("stmt", [it], it, ""))
            i += 1
            continue
        t = it.strip()
        if t.startswith("{ // callseq"):
            j = i
            while j < n and not (
                isinstance(items[j], str)
                and items[j].lstrip().startswith("} // callseq")
            ):
                j += 1
            if j >= n:
                _fail(F, "unterminated callseq block")
            out.append(_P("call", items[i : j + 1], None, ""))
            i = j + 1
            continue
        if not t or t.startswith(("//", "/*")):
            out.append(_P("other", [it], None, t))
        elif t in ("{", "}"):
            out.append(_P("brace", [it], None, t))
        elif t.startswith("."):
            out.append(_P("decl", [it], None, t))
        elif core._LABEL_LINE_RE.match(t):
            out.append(_P("label", [it], None, t))
        else:
            text = t[:-1].rstrip() if t.endswith(";") else t
            if ";" in text or "\n" in text:
                _fail(F, f"cannot parse generated line {t[:120]!r}")
            try:
                st = parse_stmt(" ".join(text.split()), 0)
            except TranslateError as e:
                _fail(F, f"cannot parse generated line {t[:120]!r}: {e}")
            out.append(_P("stmt", [it], st, ""))
        i += 1
    return out


def _parse_shared_decl(text: str, func: str | None, order: int) -> _Var:
    s = text.strip().rstrip(";").strip()
    if "=" in s:
        raise TranslateError(
            f"shared variable with an initializer is not supported in E4: {s[:120]}"
        )
    m = re.match(
        r"((?:\.(?:visible|extern|weak|common)\s+)*)\.shared(?![\w$:])\s*(.*)\Z",
        s,
        re.S,
    )
    if not m:
        raise TranslateError(f"cannot parse shared declaration {s[:120]!r}")
    extern = ".extern" in m.group(1).split()
    rest = m.group(2)
    name = core._var_name(s)
    am = re.search(r"\.align\s+(\d+)", rest)
    tm = re.search(
        r"\.(b8|b16|b32|b64|b128|u8|u16|u32|u64|s8|s16|s32|s64|f16x2|f16|bf16x2|bf16|f32|f64)(?![\w$])",
        rest,
    )
    if tm is None:
        raise TranslateError(
            f"cannot find the element type of shared declaration {s[:120]!r}"
        )
    elem = _TYPE_BYTES[tm.group(1)]
    vm = re.search(r"\.(v[248])(?![\w$])", rest)
    if vm:
        elem *= _VEC[vm.group(1)]
    align = int(am.group(1)) if am else elem
    tail = rest[rest.rindex(name) + len(name) :] if name in rest else ""
    dims = re.findall(r"\[\s*(\d*)\s*\]", tail)
    if extern:
        if any(d for d in dims):
            raise TranslateError(f".extern .shared variable with a size: {s[:120]}")
        size = 0
    else:
        if any(d == "" for d in dims):
            raise TranslateError(f"unsized non-extern .shared variable: {s[:120]}")
        size = elem
        for d in dims:
            size *= int(d)
    if align <= 0 or align & (align - 1):
        raise TranslateError(f"invalid alignment {align} of shared variable {name}")
    if align > E4_ALIGN_MAX:
        raise TranslateError(
            f"shared variable {name} has alignment {align} > {E4_ALIGN_MAX}"
        )
    key = ("f", func, name) if func else ("m", name)
    return _Var(key, name, size, align, extern, func)


def _call_target(stmt: Stmt) -> str | None:
    if stmt.opcode not in ("call", "call.uni"):
        return None
    ops = stmt.operands
    if not ops:
        return None
    if ops[0].startswith("("):
        return ops[1] if len(ops) > 1 else None
    return ops[0]


def selects_any(module: "core.Module", selector: "re.Pattern[str]") -> bool:
    """True when the selector matches the mangled name of at least one entry of the module."""
    return any(
        it.func.is_entry and selector.search(it.func.name)
        for it in module.items
        if it.kind == "func" and it.func is not None
    )


def plan_off(module: "core.Module", selector: "re.Pattern[str]") -> set[str]:
    """Names of the functions that stay in real shared memory, computed on the input module before
    translation: the entries the selector does not match plus the non-entry functions reachable only
    from those. An indirect call reaches every non-entry function. When no entry is selected the
    whole module is real shared."""
    funcs = {
        it.func.name: it.func
        for it in module.items
        if it.kind == "func" and it.func is not None
    }
    entries = [f for f in funcs.values() if f.is_entry]
    sel = [f.name for f in entries if selector.search(f.name)]
    if not sel:
        return set(funcs)
    plain = {n for n, f in funcs.items() if not f.is_entry}
    callees: dict[str, set[str]] = {}
    for n, f in funcs.items():
        cs: set[str] = set()
        for it in f.body:
            if isinstance(it, Stmt):
                tgt = _call_target(it)
                if tgt is None:
                    continue
                if tgt.startswith("%"):
                    cs |= plain
                elif tgt in funcs:
                    cs.add(tgt)
        callees[n] = cs

    def reach(starts: list[str]) -> set[str]:
        seen: set[str] = set()
        todo = list(starts)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(callees[n])
        return seen

    excluded = [f.name for f in entries if f.name not in sel]
    return set(excluded) | ((reach(excluded) - reach(sel)) & plain)


def _is_shared_tok(t: str) -> bool:
    return t.startswith("shared")


def _shared_toks(stmt: Stmt) -> list[str]:
    return [t for t in stmt.opcode.split(".")[1:] if _is_shared_tok(t)]


def _analyze(F: _F, lib_users: set[str], lib_names: set[str]) -> None:
    for sec in (F.body, F.prologue):
        for p in sec:
            if p.kind == "call":
                m = _CALLEE_RE.search(" ".join(x.strip() for x in p.items))
                if m:
                    F.callees.add(m.group(1))
                continue
            if p.kind != "stmt":
                continue
            st = p.stmt
            tgt = _call_target(st)
            if tgt is not None:
                if tgt.startswith("%"):
                    F.indirect = True
                else:
                    F.callees.add(tgt)
            if st.opcode.split(".")[0] == "exit":
                F.has_exit = True
            if _shared_toks(st) or _SREG_RE.search(st.text):
                F.uses_shared = True
            for ident in _REFID_RE.findall(st.text):
                v = F.vars.get(ident)
                if v is not None:
                    F.refs.add(v.key)
    if F.callees & lib_users:
        F.uses_shared = True
    if F.refs:
        F.uses_shared = True


# ---- rewriting -----------------------------------------------------------------------------


def _guarded(F: _F, stmt: Stmt, lines: list[str]) -> list[str]:
    if not stmt.guard or not lines:
        return lines
    g = stmt.guard
    neg = "@" + g[2:] if g.startswith("@!") else "@!" + g[1:]
    lab = F.fp.ctx.new_label("E4_SKIP")
    return [f"{neg} bra \t{lab};", *lines, f"{lab}:"]


def _reg_BW(F: _F, which: str) -> str:
    r = F.bw.get(which)
    if r is None:
        r = F.fp.ctx.new_reg("b64")
        F.bw[which] = r
    return r


def _bw_loads(F: _F) -> list[str]:
    ctx = F.fp.ctx
    out: list[str] = []
    if "B" in F.bw:
        out.append(f"ld.shared.u64 \t{F.bw['B']}, [{HDR}];")
    if "W" in F.bw:
        t = ctx.new_reg("b32")
        out += [
            f"ld.shared.u32 \t{t}, [{HDR}+8];",
            f"cvt.u64.u32 \t{F.bw['W']}, {t};",
        ]
    return out


def _access_width(F: _F, stmt: Stmt) -> int:
    toks = stmt.opcode.split(".")[1:]
    vec = next((_VEC[t] for t in toks if t in _VEC), 1)
    ty = next((_TYPE_BYTES[t] for t in reversed(toks) if t in _TYPE_BYTES), None)
    if ty is None:
        _fail(F, "cannot determine the access width", stmt)
    return vec * ty


def _addr_index(stmt: Stmt) -> int | None:
    head = stmt.opcode.split(".")[0]
    if head in ("st", "red"):
        return 0
    if head in ("ld", "atom"):
        return 1
    return None


def _shadow_addr(
    F: _F, stmt: Stmt, op: str, width: int, dyn_base: int
) -> tuple[list[str], str]:
    ctx = F.fp.ctx
    try:
        base, off = parse_address(op)
    except TranslateError as e:
        _fail(F, f"unsupported shared address operand {op!r}: {e}", stmt)
        raise
    lines: list[str] = []
    const: int | None = None
    if base == "":
        const = off
    elif not base.startswith("%"):
        v = F.vars.get(base)
        if v is None:
            _fail(F, f"address symbol {base} is not a shared variable", stmt)
            raise AssertionError
        const = v.off + off
    if const is not None:
        if const < E4_ORIGIN:
            _fail(
                F,
                f"shared address {const:#x} is below the window origin {E4_ORIGIN:#x}",
                stmt,
            )
        B = _reg_BW(F, "B")
        g = ctx.new_reg("b64")
        if const + width > dyn_base:
            W = _reg_BW(F, "W")
            q = ctx.new_reg("pred")
            lines += [f"setp.lt.u64 \t{q}, {W}, {const + width};", f"@{q} trap;"]
            F.count("checks")
        else:
            F.count("checks_elided")
        lines.append(f"add.s64 \t{g}, {B}, {const};")
        return lines, g
    bits = ctx.reg_bits(base)
    a = ctx.new_reg("b64")
    if bits == 32:
        lines.append(f"cvt.u64.u32 \t{a}, {base};")
    elif bits == 64:
        lines.append(f"and.b64 \t{a}, {base}, 0xFFFFFFFF;")
    else:
        _fail(F, f"address register {base} has unknown or invalid width", stmt)
    if off:
        lines.append(f"add.s64 \t{a}, {a}, {off};")
    B = _reg_BW(F, "B")
    W = _reg_BW(F, "W")
    p = ctx.new_reg("pred")
    q = ctx.new_reg("pred")
    e = ctx.new_reg("b64")
    g = ctx.new_reg("b64")
    lines += [
        f"setp.lt.s64 \t{p}, {a}, {E4_ORIGIN};",
        f"add.s64 \t{e}, {a}, {width};",
        f"setp.gt.s64 \t{q}, {e}, {W};",
        f"or.pred \t{p}, {p}, {q};",
        f"@{p} trap;",
        f"add.s64 \t{g}, {B}, {a};",
    ]
    F.count("checks")
    return lines, g


def _rw_access(F: _F, stmt: Stmt, dyn_base: int) -> list[str]:
    head = stmt.opcode.split(".")[0]
    toks = stmt.opcode.split(".")
    sh = [t for t in toks[1:] if _is_shared_tok(t)]
    if len(sh) != 1 or sh[0] not in ("shared", "shared::cta"):
        _fail(F, f"unsupported state space .{'.'.join(sh)}", stmt)
    idx = _addr_index(stmt)
    assert idx is not None
    if len(stmt.operands) <= idx:
        _fail(F, "missing address operand", stmt)
    width = _access_width(F, stmt)
    lines, g = _shadow_addr(F, stmt, stmt.operands[idx], width, dyn_base)
    new_toks = [("global" if t == sh[0] else t) for t in toks]
    ops = list(stmt.operands)
    ops[idx] = f"[{g}]"
    F.count(head)
    if head in ("atom", "red") and "add" in toks and "f32" in toks:
        # ptx.txt 20049-20051 / 20370: .add.f32 keeps subnormals on shared memory and flushes
        # them on global memory, so the relocated access is the per-element CAS loop that
        # rules_misc emits for .global.add.noftz.f32.
        a = rules_misc._parse_atom(F.fp.ctx, stmt, new_toks)
        a["noftz"] = True
        d = ops[0] if head == "atom" else None
        lines += rules_misc._atom_scalar(F.fp.ctx, a, d, ops[idx], ops[idx + 1], None)
        return _guarded(F, stmt, lines)
    lines.append(render_stmt(None, ".".join(new_toks), ops) + ";")
    return _guarded(F, stmt, lines)


def _sym_ref(F: _F, op: str) -> tuple[_Var, int] | None:
    m = _SYMREF_RE.match(op.strip())
    if not m:
        return None
    v = F.vars.get(m.group(1))
    if v is None:
        return None
    return v, int(m.group(2) or 0)


def _is_mov_sym(F: _F, st: Stmt) -> tuple[int, _Var, int] | None:
    """(bits, var, extra offset) for ``mov.{u,b,s}{32,64} r, sym{+k}`` of a shared variable."""
    toks = st.opcode.split(".")
    if toks[0] != "mov" or len(toks) != 2 or len(st.operands) != 2:
        return None
    ty = toks[1]
    if ty not in ("u32", "b32", "s32", "u64", "b64", "s64"):
        return None
    if st.operands[1].startswith("["):
        return None
    r = _sym_ref(F, st.operands[1])
    if r is None:
        return None
    return int(ty[1:]), r[0], r[1]


def _find_ok64(F: _F) -> None:
    """Registers that receive a shared symbol address through 64-bit mov and are used only as the
    base of shared access addresses (contract: otherwise fail closed)."""
    defs: dict[str, Stmt] = {}
    for sec in (F.body, F.prologue):
        for p in sec:
            if p.kind == "stmt":
                m = _is_mov_sym(F, p.stmt)
                if m and m[0] == 64:
                    defs[p.stmt.operands[0]] = p.stmt
    if not defs:
        return
    bad: set[str] = set()
    for sec in (F.body, F.prologue):
        for p in sec:
            if p.kind == "call":
                txt = " ".join(p.items)
                bad |= set(_REG_RE.findall(txt)) & defs.keys()
                continue
            if p.kind != "stmt":
                continue
            st = p.stmt
            used = set(_REG_RE.findall(st.text)) & defs.keys()
            if not used:
                continue
            m = _is_mov_sym(F, st)
            if m and m[0] == 64:
                used.discard(st.operands[0])
                bad |= used
                continue
            head = st.opcode.split(".")[0]
            idx = _addr_index(st)
            sh = _shared_toks(st)
            if (
                head in ("ld", "st", "atom", "red")
                and sh
                and idx is not None
                and len(st.operands) > idx
            ):
                try:
                    base, _ = parse_address(st.operands[idx])
                except TranslateError:
                    base = None
                others = set()
                for k, o in enumerate(st.operands):
                    if k != idx:
                        others |= set(_REG_RE.findall(o))
                if base in used and not (others & defs.keys()) and used == {base}:
                    continue
            bad |= used
    F.ok64 = set(defs) - bad
    for r, st in defs.items():
        if r in bad:
            _fail(
                F,
                f"shared symbol address in 64-bit register {r} is used other than as a shared access address",
                st,
            )


def _rw_stmt(F: _F, st: Stmt, dyn_base: int, entry_hooks: bool) -> list:
    ctx = F.fp.ctx
    toks = st.opcode.split(".")
    head = toks[0]
    sh = _shared_toks(st)

    # special registers
    if _SREG_RE.search(st.text):
        if (
            head == "mov"
            and len(st.operands) == 2
            and st.operands[1] in ("%dynamic_smem_size", "%total_smem_size")
            and toks[1:] in (["u32"], ["b32"], ["s32"])
        ):
            fn = "dyn_smem" if st.operands[1] == "%dynamic_smem_size" else "total_smem"
            F.count(fn)
            dst = st.operands[0]
            tmp = ctx.new_reg("b32")
            lines = ctx.call(LIB[fn], [], ret=("b32", tmp))
            lines.append(f"mov.b32 \t{dst}, {tmp};")
            return _guarded(F, st, lines)
        _fail(F, f"unsupported use of {_SREG_RE.search(st.text).group(0)}", st)

    m = _is_mov_sym(F, st)
    if m is not None:
        bits, v, extra = m
        if bits == 64 and st.operands[0] not in F.ok64:
            _fail(F, "shared symbol address in a 64-bit register", st)
        F.count("mov_sym64" if bits == 64 else "mov_sym")
        imm = v.off + extra
        line = render_stmt(None, st.opcode, [st.operands[0], str(imm)]) + ";"
        return _guarded(F, st, [line])

    if sh:
        if head in ("ld", "st", "atom", "red") and len(sh) == 1:
            return _rw_access(F, st, dyn_base)
        what = "cluster shared memory" if "shared::cluster" in sh else "shared form"
        _fail(F, f"unsupported {what} .{'.'.join(sh)} in {st.opcode}", st)

    # any other mention of a shared symbol
    for ident in _REFID_RE.findall(st.text):
        if ident in F.vars:
            _fail(
                F, f"shared symbol {ident} used outside mov / ld / st / atom / red", st
            )

    if head == "exit" or (head == "ret" and F.is_entry):
        if head == "ret" and (len(toks) > 2 or st.operands):
            _fail(F, "ret with operands in an entry", st)
        if entry_hooks:
            F.count("exit_hook")
            return [*_guarded(F, st, ctx.call(LIB["exit"], [])), st]
    return [st]


def _rw_section(F: _F, sec: list[_P], dyn_base: int) -> list:
    ctx = F.fp.ctx
    ctx._scopes = [core._Scope()]
    out: list = []
    for p in sec:
        if p.kind == "stmt":
            ctx.stmt = p.stmt
            out.extend(_rw_stmt(F, p.stmt, dyn_base, True))
            ctx.stmt = None
            continue
        if p.kind == "other":
            if p.orig.startswith("// wgemu "):
                F.last_orig = p.orig[:160]
            out.extend(p.items)
            continue
        if p.kind == "decl":
            if _SHARED_DECL_TOK_RE.search(p.orig):
                continue  # removed: the variable became a virtual offset
            ctx._track(p.items[0])
        elif p.kind == "brace":
            ctx._track(p.items[0])
        out.extend(p.items)
    return out


# ---- driver --------------------------------------------------------------------------------


def _lib_hdr_users(lib_src: str) -> set[str]:
    lib = core._parse_lib(lib_src)
    has_hdr = any(
        it.kind == "var" and it.name == HDR and it.space == "shared" for it in lib.items
    )
    if not has_hdr:
        raise TranslateError(
            f"E4 shadow shared memory needs an emulation library built with -DWGEMU_SMEM_GLOBAL "
            f"(it defines no .shared {HDR}); wrong library variant"
        )
    names = {it.name for it in lib.items if it.name}
    refs: dict[str, set[str]] = {}
    for it in lib.items:
        if it.name and it.kind in ("func", "var"):
            ids = set(core._IDENT_RE.findall(core._item_text(lib_src, it)))
            ids.discard(it.name)
            refs.setdefault(it.name, set()).update(ids & names)
    users: set[str] = set()
    changed = True
    while changed:
        changed = False
        for n, r in refs.items():
            if n not in users and (HDR in r or r & users):
                users.add(n)
                changed = True
    users.discard(HDR)
    return users


def apply(
    module: "core.Module",
    passes: "list[core._FuncPass]",
    mctx: "core._ModuleCtx",
    lib_src: str,
    rep: dict,
    selector: "re.Pattern[str] | None" = None,
    off: "set[str] | frozenset[str]" = frozenset(),
) -> tuple[dict[int, list[str]], dict]:
    """E4 pass over all rewritten functions. Returns ({pass index: lines for the head of the
    function body}, info). Mutates the bodies / headers in place. ``selector`` / ``off``: entry
    selection (:func:`plan_off`); the functions in ``off`` are left as the ordinary rules made them."""
    slots = mctx.e4_slots
    lib_users = _lib_hdr_users(lib_src)
    lib_names = {it.name for it in core._parse_lib(lib_src).items if it.name}
    for sym in mctx.symbols:
        if sym.startswith(E4_PREFIX):
            raise core._located(
                f"{mctx.filename}: input symbol {sym} uses a reserved E4 prefix"
            )

    # shared variables ------------------------------------------------------------------------
    mod_vars: dict[str, _Var] = {}
    try:
        for it in module.items:
            if it.kind == "var" and it.space == "shared":
                v = _parse_shared_decl(it.text, None, 0)
                mod_vars[v.name] = v
    except TranslateError as e:
        raise core._located(f"{mctx.filename}: {e}") from None

    funcs: list[_F] = []
    for i, fp in enumerate(passes):
        F = _F(i, fp, fp.func.name, fp.func.is_entry)
        F.off = F.name in off
        F.body = _parse_items(F, fp.body)
        F.prologue = _parse_items(F, fp.prologue)
        for p in F.body:
            if F.off:
                break  # real shared: its declarations are not E4 variables
            if p.kind == "decl" and _SHARED_DECL_TOK_RE.search(p.orig):
                try:
                    v = _parse_shared_decl(p.orig, F.name, 0)
                except TranslateError as e:
                    _fail(F, str(e))
                F.local_vars[v.name] = v
        funcs.append(F)
    # declaration order: module vars, function-scope vars at their function, translator vars
    seq: list[_Var] = []
    fi = {F.name: F for F in funcs}
    for it in module.items:
        if it.kind == "var" and it.space == "shared":
            seq.append(mod_vars[it.name])
        elif it.kind == "func" and it.func is not None and it.name in fi:
            seq.extend(fi[it.name].local_vars.values())
    for sym, (n, a) in mctx.shared.items():
        v = _Var(("m", sym), sym, n, a, False)
        if v.align > E4_ALIGN_MAX:
            raise core._located(
                f"{mctx.filename}: translator shared {sym} alignment {a} > {E4_ALIGN_MAX}"
            )
        mod_vars[sym] = v
        seq.append(v)

    for F in funcs:
        F.vars = {**mod_vars, **F.local_vars}
        _analyze(F, lib_users, lib_names)

    # classification --------------------------------------------------------------------------
    byname = {F.name: F for F in funcs}
    allfuncs = {F.name for F in funcs if not F.is_entry}

    def closure(entry: _F) -> set[str]:
        seen: set[str] = set()
        todo = [entry.name]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            F = byname.get(n)
            if F is None:
                continue
            todo.extend(c for c in F.callees if c in byname)
            if F.indirect:
                todo.extend(allfuncs)
        return seen

    entries = [F for F in funcs if F.is_entry]
    clos = {F.name: closure(F) for F in entries}

    def selected(F: _F) -> bool:
        return selector is None or selector.search(F.name) is not None

    def touches(F: _F) -> bool:
        return any(byname[n].uses_shared for n in clos[F.name] if n in byname)

    e4_entries = [F for F in entries if selected(F) and touches(F)]
    e4_names = {F.name for F in e4_entries}
    excluded = [F for F in entries if not selected(F)]
    e4_excluded = [F for F in excluded if touches(F)]
    reach_e4: set[str] = set()
    reach_other: set[str] = set()
    for F in entries:
        (reach_e4 if F.name in e4_names else reach_other).update(clos[F.name])
    for F in funcs:
        F.e4 = not F.off and (F.name in reach_e4 or (not F.is_entry and F.uses_shared))
    # real shared functions (entries not selected and what only they reach)
    for F in funcs:
        if not F.off:
            continue
        if F.name in reach_e4:
            _fail(
                F,
                "function reachable from an E4 entry was planned as real shared (internal)",
            )
        bad = sorted(F.callees & lib_users)
        if bad:
            who = next((X.name for X in excluded if F.name in clos[X.name]), F.name)
            _fail(
                F,
                f"call to emulation library function {bad[0]}, which uses the E4 window, from "
                f"{who} (not selected by WGEMU_E4_ENTRIES, real shared memory): the E4 library "
                "variant cannot serve it; select the entry or use the library without "
                "-DWGEMU_SMEM_GLOBAL",
            )
    # one function body cannot be both modes
    for G in funcs:
        if G.is_entry or not G.uses_shared:
            continue
        a = next((E for E in e4_entries if G.name in clos[E.name]), None)
        b = next((X for X in excluded if G.name in clos[X.name]), None)
        if a is not None and b is not None:
            _fail(
                G,
                f"function {G.name} touches shared memory and is reachable from E4 entry {a.name} "
                f"and from entry {b.name} (not selected by WGEMU_E4_ENTRIES); one function body "
                "cannot be both E4 and real shared memory",
            )
    # calls out of an E4 entry to code the translator cannot see
    for F in e4_entries:
        for n in clos[F.name]:
            G = byname.get(n)
            if G is None:
                if (
                    n in lib_names
                    or n in _EXTERN_OK
                    or n.startswith(("__nv_", "__internal_"))
                ):
                    continue
                raise core._located(
                    f"{mctx.filename}: in {F.name}: E4 shadow shared memory: call to {n}, which is neither "
                    "defined in the module nor in the emulation library (shared memory use unknown)"
                )
    for F in funcs:
        if (
            F.has_exit
            and not F.is_entry
            and F.name in reach_e4
            and F.name in reach_other
        ):
            _fail(F, "exit in a function reachable from both E4 and non-E4 entries")

    # layout (overlay: entries share nothing they do not both reference) ----------------------
    sets: dict[tuple, set[str]] = {}
    for F in e4_entries:
        for n in clos[F.name]:
            G = byname.get(n)
            if G is not None:
                for k in G.refs:
                    sets.setdefault(k, set()).add(F.name)
    end: dict[str, int] = {F.name: E4_ORIGIN for F in e4_entries}
    dyn_align = 1
    for v in seq:
        if v.extern:
            dyn_align = max(dyn_align, v.align)
            continue
        es = sets.get(v.key, set())
        base = max((end[e] for e in es), default=E4_ORIGIN)
        v.off = _align_up(base, v.align)
        for e in es:
            end[e] = v.off + v.size
    dyn_base = _align_up(
        max(end.values(), default=E4_ORIGIN), max(E4_ALIGN_MAX, dyn_align)
    )
    for v in seq:
        if v.extern:
            v.off = dyn_base
    if dyn_base > E4_WINDOW_MAX or dyn_base > E4_STRIDE:
        raise core._located(
            f"{mctx.filename}: E4 static shared memory ({dyn_base - E4_ORIGIN} B after the "
            f"{E4_ORIGIN:#x} origin and alignment) does not fit the {E4_WINDOW_MAX:#x} virtual window"
        )
    if dyn_base - E4_ORIGIN > E4_SM90_BLOCK_MAX:
        raise core._located(
            f"{mctx.filename}: E4 static shared memory {dyn_base - E4_ORIGIN} B exceeds the SM90 limit {E4_SM90_BLOCK_MAX}"
        )

    # rewrite ---------------------------------------------------------------------------------
    heads: dict[int, list[str]] = {}
    totals: dict[str, int] = {}
    for F in funcs:
        ctx = F.fp.ctx
        if F.off:
            continue  # real shared: exactly the ordinary translation
        if not F.e4:
            # a function that only declares shared variables: drop the declarations
            F.fp.body = [
                it
                for p in F.body
                if not (p.kind == "decl" and _SHARED_DECL_TOK_RE.search(p.orig))
                for it in p.items
            ]
            continue
        _find_ok64(F)
        new_body = _rw_section(F, F.body, dyn_base)
        new_prologue = _rw_section(F, F.prologue, dyn_base)
        head: list[str] = []
        if F.is_entry and F.name in e4_names:
            func = F.fp.func
            func.header = _append_param(func.header, f".param .u64 {META}")
            func.params.append((".u64", META))
            mk = marker_name(F.name)
            mctx.globals[mk] = [4, 4]
            meta = ctx.new_reg("b64")
            tp, pool = ctx.new_reg("b64"), ctx.new_reg("b64")
            tl, lease = ctx.new_reg("b64"), ctx.new_reg("b64")
            ts, stats = ctx.new_reg("b64"), ctx.new_reg("b64")
            head += [
                f"ld.param.u64 \t{meta}, [{META}];",
                f"mov.u64 \t{tp}, {POOL};",
                f"cvta.global.u64 \t{pool}, {tp};",
                f"mov.u64 \t{tl}, {LEASE};",
                f"cvta.global.u64 \t{lease}, {tl};",
                f"mov.u64 \t{ts}, {STATS};",
                f"cvta.global.u64 \t{stats}, {ts};",
                *ctx.call(
                    LIB["enter"],
                    [
                        ("b64", meta),
                        ("b32", str(dyn_base)),
                        ("b64", pool),
                        ("b64", lease),
                        ("b64", stats),
                        ("b32", str(slots)),
                    ],
                ),
            ]
        head += _bw_loads(F)
        F.fp.body = new_body
        F.fp.prologue = new_prologue
        heads[F.idx] = head
        for k, c in F.counts.items():
            totals[k] = totals.get(k, 0) + c

    if e4_entries:
        mctx.globals[POOL] = [slots * E4_STRIDE, 1024]
        mctx.globals[LEASE] = [slots * 4, 8]
        mctx.globals[STATS] = [32, 8]
    cp = mctx.rewrites.get("rules_smem", {})
    # shared variables a real shared function references keep their declaration (an E4 function
    # that references the same variable uses its virtual offset instead)
    kept: set[tuple] = set()
    for F in funcs:
        if F.off:
            kept |= F.refs
    if selector is not None and not e4_entries:
        kept |= {
            v.key for v in seq
        }  # nothing is E4: the module is the ordinary translation
    e4_used = {k for F in funcs if F.e4 for k in F.refs}
    gone = [v for v in seq if v.key not in kept or v.key in e4_used]
    removed = sorted({v.name for v in gone})
    info = {
        "entries": [F.name for F in e4_entries],
        "e4_entries": [F.name for F in e4_entries],
        "e4_excluded": [
            {"entry": F.name, "reason": "not selected by WGEMU_E4_ENTRIES"}
            for F in e4_excluded
        ],
        "entry_regex": selector.pattern if selector is not None else None,
        "non_e4_entries": [F.name for F in entries if F.name not in e4_names],
        "off_funcs": sorted(off),
        "kept_shared": [
            v.name
            for v in seq
            if v.key in kept and v.key[0] == "m" and v.name not in mctx.shared
        ],
        "kept_translator_shared": [s for s in mctx.shared if ("m", s) in kept],
        "markers": {F.name: marker_name(F.name) for F in e4_entries},
        "dyn_base": dyn_base,
        "slots": slots,
        "window_max": E4_WINDOW_MAX,
        "rewritten": {
            **totals,
            "cp_async": sum(cp.values()),
            "cp_async_by_opcode": dict(cp),
        },
        "layout": {
            v.name: {
                "off": v.off,
                "bytes": v.size,
                "align": v.align,
                "extern": v.extern,
            }
            for v in gone
        },
        "functions": [F.name for F in funcs if F.e4 and not F.is_entry],
        "removed_shared": removed,
    }
    return heads, info


def _append_param(header: str, decl: str) -> str:
    m = core._LINKAGE_RE.match(header)
    if not m:
        raise TranslateError(f"not a function header: {header[:120]!r}")
    s = header
    pos = core._skip_ws(s, m.end())
    m2 = core._IDENT_RE.match(s, pos)
    if not m2:
        raise TranslateError(f"cannot find the entry name in {header[:120]!r}")
    pos = core._skip_ws(s, m2.end())
    if pos < len(s) and s[pos] == "(":
        end = core._match_close(s, pos)
        if s[pos + 1 : end].strip():
            return s[:end].rstrip() + ",\n\t" + decl + "\n" + s[end:]
        return s[: pos + 1] + "\n\t" + decl + "\n" + s[end:]
    return s[: m2.end()] + "(\n\t" + decl + "\n)" + s[m2.end() :]


# ---------------------------------------------------------------------------------------------
# Self check
# ---------------------------------------------------------------------------------------------


def _statements(text: str) -> list[str]:
    t = _STR_RE.sub('""', _COMMENT_RE.sub(" ", text))
    out = []
    for piece in re.split(r"[;{}]", t):
        s = piece.strip()
        while True:
            m = re.match(r"[A-Za-z_$][\w$]*\s*:(?!:)\s*", s)
            if not m:
                break
            s = s[m.end() :]
        if s:
            out.append(" ".join(s.split()))
    return out


def selfcheck(lib_text: str, user_text: str, removed: list[str]) -> None:
    """Contract section 5: the output has no shared state space except ld/st/atom.shared on the
    header, no .shared declaration but the header, no shared special registers outside the library
    and no reference to a removed shared symbol."""
    bad: list[str] = []
    rem = set(removed)

    def scan(text: str, lib: bool) -> None:
        for s in _statements(text):
            if s.startswith("."):
                if _SHARED_DECL_TOK_RE.search(s):
                    nm = (
                        core._var_name(s)
                        if "=" not in s
                        else s.split("=")[0].split()[-1]
                    )
                    if nm != HDR:
                        bad.append(f"shared declaration: {s[:100]}")
                continue
            m = re.match(r"(?:@!?%?[\w$]+\s+)?([A-Za-z_][\w.:]*)(.*)\Z", s)
            if not m:
                continue
            opc, rest = m.group(1), m.group(2)
            toks = opc.split(".")
            if any(_is_shared_tok(t) for t in toks[1:]):
                if toks[0] not in ("ld", "st", "atom") or not _HDR_ADDR_RE.search(rest):
                    bad.append(f"shared instruction: {s[:100]}")
                continue
            if not lib and _SREG_RE.search(rest):
                bad.append(f"shared special register: {s[:100]}")
            if not lib and rem:
                for ident in _REFID_RE.findall(rest):
                    if ident in rem:
                        bad.append(
                            f"reference to removed shared symbol {ident}: {s[:100]}"
                        )
                        break

    scan(lib_text, True)
    scan(user_text, False)
    if bad:
        head = "; ".join(bad[:5])
        raise core._located(f"E4 self-check failed ({len(bad)} violation(s)): {head}")
