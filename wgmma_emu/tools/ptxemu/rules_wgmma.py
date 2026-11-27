"""wgmma translation rules (PLAN sections 5, 7, 9). Owner: W.

Rewrites every ``wgmma.*`` instruction of PTX ISA 9.5 section 9.7.15 into calls of the
emulation library (``runtime/lib_wgmma.cu``):

* ``wgmma.mma_async{.sp}``: the accumulator registers are stored to a function-scope
  ``.local`` buffer, ``wgemu_wgmma`` is called with a generic pointer to that buffer and the
  registers are reloaded from it.
* ``wgmma.fence`` / ``wgmma.commit_group`` / ``wgmma.wait_group``: no code in exact and fast
  modes (results are written at issue time, which is one valid execution of the asynchronous
  operation). Check mode calls ``wgemu_wgmma_fence`` / ``_commit`` / ``_wait``; after a wait
  every accumulator id is fetched (``wgemu_wgmma_fetch`` + predicated reload) and every RS A
  register list is compared (``wgemu_wgmma_check_a``).
* Check mode also instruments non-wgmma instructions that name accumulator registers with
  ``wgemu_wgmma_check_acc`` (``rewrite_operand``), once per accumulator id per basic block.

The variant word passed to ``wgemu_wgmma`` (keep identical to
``include/wgmma_emu/wgmma_device.cuh``)::

    bits  0..5   N/8 - 1            (N = 8..256)
    bits  6..9   atype              (EType value)
    bits 10..13  btype              (EType value)
    bits 14..17  dtype              (EType value: F16=0, F32=8, S32=9)
    bit  18      rs                 A from registers a0..a3 (else from adesc)
    bit  19      sparse             wgmma.mma_async.sp
    bit  20      neg_a              imm-scale-a == -1
    bit  21      neg_b              imm-scale-b == -1
    bit  22      trans_a            imm-trans-a == 1 (A is M-major in smem)
    bit  23      trans_b            imm-trans-b == 1 (B is N-major in smem)
    bit  24      satfinite          integer .satfinite
    bits 25..26  sp_sel             sparsity selector (sp-sel)
    bits 27..28  mode               0 exact, 1 fast, 2 check
    bit  29      trap               check mode: __trap() after reporting a violation
    bits 30..31  reserved, must be 0

ids word: bits 0..15 accumulator id, bits 16..31 RS A-register id (0xFFFF = SS, no A id).
Check-mode table bytes: 64 + 544 * n_acc + 48 * n_a + 64 * 48 (``table_bytes``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .core import Ctx, Func, Stmt, parse_vector

# ---------------------------------------------------------------------------------------------
# Encoding tables (mirrors of wgmma_device.cuh)
# ---------------------------------------------------------------------------------------------

ETYPE = {
    "f16": 0,
    "bf16": 1,
    "tf32": 2,
    "e4m3": 3,
    "e5m2": 4,
    "s8": 5,
    "u8": 6,
    "b1": 7,
    "f32": 8,
    "s32": 9,
}
ETYPE_NAME = {v: k for k, v in ETYPE.items()}
FLOAT_IN = ("f16", "bf16", "tf32", "e4m3", "e5m2")
FP8 = ("e4m3", "e5m2")
INT8 = ("s8", "u8")

VAR_N_SHIFT = 0
VAR_N_MASK = 0x3F
VAR_ATYPE_SHIFT = 6
VAR_BTYPE_SHIFT = 10
VAR_DTYPE_SHIFT = 14
VAR_TYPE_MASK = 0xF
VAR_RS = 1 << 18
VAR_SPARSE = 1 << 19
VAR_NEG_A = 1 << 20
VAR_NEG_B = 1 << 21
VAR_TRANS_A = 1 << 22
VAR_TRANS_B = 1 << 23
VAR_SATFINITE = 1 << 24
VAR_SPSEL_SHIFT = 25
VAR_SPSEL_MASK = 0x3
VAR_MODE_SHIFT = 27
VAR_MODE_MASK = 0x3
VAR_TRAP = 1 << 29
VAR_RESERVED = 3 << 30

MODE_CODE = {"exact": 0, "fast": 1, "check": 2}
IDS_NO_A = 0xFFFF
MAX_IDS = 0xFFFF  # ids are 16-bit; 0xFFFF is reserved for "no A id"

WG_HDR_BYTES = 64
WG_ACC_SLOT_BYTES = 544
WG_A_SLOT_BYTES = 48
WG_MAX_OPS = 64
WG_OP_BYTES = 48
WG_FLAG_TRAP = 1
D_BUF_BYTES = 128 * 4  # largest accumulator: f32/s32 n256 -> 128 registers

LIB_MMA = "wgemu_wgmma"
LIB_INIT = "wgemu_wgmma_check_init"
LIB_FENCE = "wgemu_wgmma_fence"
LIB_COMMIT = "wgemu_wgmma_commit"
LIB_WAIT = "wgemu_wgmma_wait"
LIB_FETCH = "wgemu_wgmma_fetch"
LIB_CHECK_A = "wgemu_wgmma_check_a"
LIB_CHECK_ACC = "wgemu_wgmma_check_acc"


def table_bytes(n_acc: int, n_a: int) -> int:
    return (
        WG_HDR_BYTES
        + n_acc * WG_ACC_SLOT_BYTES
        + n_a * WG_A_SLOT_BYTES
        + WG_MAX_OPS * WG_OP_BYTES
    )


@dataclass
class Variant:
    n: int
    at: str
    bt: str
    dt: str
    rs: bool = False
    sparse: bool = False
    neg_a: bool = False
    neg_b: bool = False
    trans_a: bool = False
    trans_b: bool = False
    satfinite: bool = False
    sp_sel: int = 0
    mode: str = "exact"
    trap: bool = False

    def encode(self) -> int:
        x = ((self.n // 8 - 1) & VAR_N_MASK) << VAR_N_SHIFT
        x |= (ETYPE[self.at] & VAR_TYPE_MASK) << VAR_ATYPE_SHIFT
        x |= (ETYPE[self.bt] & VAR_TYPE_MASK) << VAR_BTYPE_SHIFT
        x |= (ETYPE[self.dt] & VAR_TYPE_MASK) << VAR_DTYPE_SHIFT
        x |= VAR_RS if self.rs else 0
        x |= VAR_SPARSE if self.sparse else 0
        x |= VAR_NEG_A if self.neg_a else 0
        x |= VAR_NEG_B if self.neg_b else 0
        x |= VAR_TRANS_A if self.trans_a else 0
        x |= VAR_TRANS_B if self.trans_b else 0
        x |= VAR_SATFINITE if self.satfinite else 0
        x |= (self.sp_sel & VAR_SPSEL_MASK) << VAR_SPSEL_SHIFT
        x |= (MODE_CODE[self.mode] & VAR_MODE_MASK) << VAR_MODE_SHIFT
        x |= VAR_TRAP if self.trap else 0
        return x


def decode_variant(x: int) -> Variant:
    if x & VAR_RESERVED:
        raise ValueError(f"variant 0x{x:08x} has reserved bits set")
    mode = {v: k for k, v in MODE_CODE.items()}.get(
        (x >> VAR_MODE_SHIFT) & VAR_MODE_MASK
    )
    if mode is None:
        raise ValueError(f"variant 0x{x:08x} has an invalid mode")

    def ty(shift: int) -> str:
        code = (x >> shift) & VAR_TYPE_MASK
        if code not in ETYPE_NAME:
            raise ValueError(f"variant 0x{x:08x}: type code {code}")
        return ETYPE_NAME[code]

    return Variant(
        n=(((x >> VAR_N_SHIFT) & VAR_N_MASK) + 1) * 8,
        at=ty(VAR_ATYPE_SHIFT),
        bt=ty(VAR_BTYPE_SHIFT),
        dt=ty(VAR_DTYPE_SHIFT),
        rs=bool(x & VAR_RS),
        sparse=bool(x & VAR_SPARSE),
        neg_a=bool(x & VAR_NEG_A),
        neg_b=bool(x & VAR_NEG_B),
        trans_a=bool(x & VAR_TRANS_A),
        trans_b=bool(x & VAR_TRANS_B),
        satfinite=bool(x & VAR_SATFINITE),
        sp_sel=(x >> VAR_SPSEL_SHIFT) & VAR_SPSEL_MASK,
        mode=mode,
        trap=bool(x & VAR_TRAP),
    )


def wg_k(at: str, sparse: bool) -> int:
    """Logical K of one wgmma (PTX 9.5 shape table)."""
    if at in ("f16", "bf16"):
        return 32 if sparse else 16
    if at == "tf32":
        return 16 if sparse else 8
    if at in FP8 or at in INT8:
        return 64 if sparse else 32
    if at == "b1":
        return 256
    return 0


def wg_valid_n(at: str, n: int) -> bool:
    """Floating point: 8..256 step 8. Integer / b1: 8, 16, 24, 32, 48..256 step 16 (PTX 9.5
    shape table; the dense integer syntax block stops at n224, the superset is accepted)."""
    if n < 8 or n > 256 or n % 8:
        return False
    if at in INT8 or at == "b1":
        return n <= 32 or n % 16 == 0
    return True


def d_regs(v: Variant) -> int:
    return v.n // 4 if v.dt == "f16" else v.n // 2


def variant_error(v: Variant) -> str | None:
    """None when ``v`` is a legal PTX 9.5 wgmma.mma_async{.sp}, else a reason (mirrors the
    C++ ``variant_error``)."""
    if v.at not in ETYPE or v.bt not in ETYPE or v.dt not in ETYPE:
        return "types"
    if not wg_valid_n(v.at, v.n):
        return f"n{v.n} is not a valid N for .{v.at}"
    at, bt, dt = v.at, v.bt, v.dt
    if at == "f16":
        ok = bt == "f16" and dt in ("f16", "f32")
    elif at in ("bf16", "tf32"):
        ok = bt == at and dt == "f32"
    elif at in FP8:
        ok = bt in FP8 and dt in ("f16", "f32")
    elif at in INT8:
        ok = bt in INT8 and dt == "s32"
    elif at == "b1":
        ok = bt == "b1" and dt == "s32"
        if ok and v.sparse:
            return "b1 has no sparse wgmma"
    else:
        ok = False
    if not ok:
        return f"type combination .{dt}.{at}.{bt} does not exist"
    trans_ok = at in ("f16", "bf16")
    if v.trans_a and (not trans_ok or v.rs):
        return "imm-trans-a exists only for .f16/.bf16 with A in shared memory"
    if v.trans_b and not trans_ok:
        return "imm-trans-b exists only for .f16/.bf16"
    if (v.neg_a or v.neg_b) and at not in FLOAT_IN:
        return "imm-scale exists only for floating point types"
    if v.satfinite and at not in INT8:
        return ".satfinite exists only for .s8/.u8"
    if not v.sparse:
        if v.sp_sel:
            return "sp-sel on a dense wgmma"
    elif at in ("f16", "bf16", "tf32"):
        if v.sp_sel > 1:
            return f"sp-sel {v.sp_sel} (must be 0 or 1 for .{at})"
    elif v.sp_sel:
        return f"sp-sel {v.sp_sel} (must be 0 for .{at})"
    return None


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------

_SHAPE_RE = re.compile(r"m(\d+)n(\d+)k(\d+)\Z")
_REG_RE = re.compile(r"%?[A-Za-z_$][\w$]*\Z")
_PRED_RE = re.compile(r"(!?)\s*(%?[A-Za-z_$][\w$]*)\Z")
_INT_RE = re.compile(
    r"([+-]?)\s*(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)[Uu]?\Z"
)
_LABEL_RE = re.compile(r"\s*[A-Za-z_$][\w$]*\s*:\s*\Z")


def _fail(ctx: Ctx, msg: str) -> None:
    ctx.error(msg)
    raise RuntimeError(msg)  # ctx.error raises; never continue silently


def _imm(op: str) -> int | None:
    m = _INT_RE.fullmatch(op.strip())
    if not m:
        return None
    s = m.group(2)
    if s[:2] in ("0x", "0X"):
        v = int(s[2:], 16)
    elif s[:2] in ("0b", "0B"):
        v = int(s[2:], 2)
    elif len(s) > 1 and s[0] == "0":
        v = int(s[1:], 8)
    else:
        v = int(s, 10)
    return -v if m.group(1) == "-" else v


def _is_reg(op: str) -> bool:
    return bool(_REG_RE.fullmatch(op.strip())) and _imm(op) is None


def _kind(stmt: Stmt) -> str:
    toks = stmt.opcode.split(".")
    return toks[1] if len(toks) > 1 else ""


def _sync_aligned(ctx: Ctx, stmt: Stmt, toks: list[str]) -> None:
    for t in ("sync", "aligned"):
        c = toks.count(t)
        if c != 1:
            _fail(
                ctx,
                f"{stmt.opcode}: qualifier .{t} must appear exactly once (found {c})",
            )


@dataclass
class Mma:
    v: Variant
    d: list[str]
    a_regs: list[str] | None
    adesc: str | None
    bdesc: str
    meta: str | None
    scale_d: str  # '0', '1' or a predicate operand with optional '!'


def parse_mma(stmt: Stmt, ctx: Ctx) -> Mma:
    toks = stmt.opcode.split(".")
    if toks[:2] != ["wgmma", "mma_async"]:
        _fail(ctx, f"not a wgmma.mma_async: {stmt.opcode}")
    rest = toks[2:]
    _sync_aligned(ctx, stmt, rest)
    sparse = False
    satfinite = False
    and_popc = []
    shape = None
    types: list[str] = []
    for t in rest:
        if t in ("sync", "aligned"):
            continue
        if t == "sp":
            if sparse:
                _fail(ctx, f"{stmt.opcode}: duplicate .sp")
            sparse = True
        elif t == "satfinite":
            if satfinite:
                _fail(ctx, f"{stmt.opcode}: duplicate .satfinite")
            satfinite = True
        elif t in ("and", "popc"):
            if t in and_popc:
                _fail(ctx, f"{stmt.opcode}: duplicate .{t}")
            and_popc.append(t)
        elif _SHAPE_RE.fullmatch(t):
            if shape is not None:
                _fail(ctx, f"{stmt.opcode}: two shapes")
            shape = t
        elif t in ETYPE:
            types.append(t)
        else:
            _fail(ctx, f"{stmt.opcode}: unknown qualifier .{t}")
    if shape is None:
        _fail(ctx, f"{stmt.opcode}: missing .m64nNkK shape")
    if len(types) != 3:
        _fail(ctx, f"{stmt.opcode}: expected .dtype.atype.btype, found {types}")
    dt, at, bt = types
    m, n, k = (int(x) for x in _SHAPE_RE.fullmatch(shape).groups())
    if m != 64:
        _fail(ctx, f"{stmt.opcode}: M must be 64 (got {m})")
    if at == "b1":
        if sorted(and_popc) != ["and", "popc"]:
            _fail(ctx, f"{stmt.opcode}: .b1 requires .and.popc")
    elif and_popc:
        _fail(ctx, f"{stmt.opcode}: .and/.popc only exist for .b1")
    if k != wg_k(at, sparse):
        _fail(
            ctx,
            f"{stmt.opcode}: K={k} does not match .{at}{' sparse' if sparse else ''}",
        )

    ops = [o.strip() for o in stmt.operands]
    if len(ops) < 4:
        _fail(ctx, f"{stmt.opcode}: too few operands ({len(ops)})")
    if not (ops[0].startswith("{") and ops[0].endswith("}")):
        _fail(ctx, f"{stmt.opcode}: accumulator operand {ops[0]!r} is not a vector")
    d = parse_vector(ops[0])
    rs = ops[1].startswith("{")
    a_regs = parse_vector(ops[1]) if rs else None
    adesc = None if rs else ops[1]
    bdesc = ops[2]
    i = 3
    meta = None
    sp_sel = 0
    if sparse:
        if len(ops) < 6:
            _fail(ctx, f"{stmt.opcode}: sparse wgmma needs sp-meta and sp-sel")
        meta = ops[3]
        sel = _imm(ops[4])
        if sel is None:
            _fail(ctx, f"{stmt.opcode}: sp-sel {ops[4]!r} is not an integer constant")
        sp_sel = sel
        if sp_sel < 0 or sp_sel > 3:
            _fail(ctx, f"{stmt.opcode}: sp-sel {sp_sel} out of range")
        i = 5
    scale_d = ops[i]
    i += 1
    is_float = at in FLOAT_IN
    n_trans = (1 if rs else 2) if at in ("f16", "bf16") else 0
    expect = i + (2 if is_float else 0) + n_trans
    if len(ops) != expect:
        _fail(
            ctx,
            f"{stmt.opcode}: {len(ops)} operands, expected {expect} for this form "
            f"({'RS' if rs else 'SS'}, .{at})",
        )
    neg_a = neg_b = trans_a = trans_b = False
    if is_float:
        sa, sb = _imm(ops[i]), _imm(ops[i + 1])
        for name, val, op in (
            ("imm-scale-a", sa, ops[i]),
            ("imm-scale-b", sb, ops[i + 1]),
        ):
            if val not in (1, -1):
                _fail(ctx, f"{stmt.opcode}: {name} {op!r} must be the constant 1 or -1")
        neg_a, neg_b = sa == -1, sb == -1
        i += 2
    if n_trans:
        vals = []
        for op in ops[i : i + n_trans]:
            val = _imm(op)
            if val not in (0, 1):
                _fail(
                    ctx, f"{stmt.opcode}: imm-trans {op!r} must be the constant 0 or 1"
                )
            vals.append(val == 1)
        if rs:
            trans_b = vals[0]
        else:
            trans_a, trans_b = vals
    v = Variant(
        n=n,
        at=at,
        bt=bt,
        dt=dt,
        rs=rs,
        sparse=sparse,
        neg_a=neg_a,
        neg_b=neg_b,
        trans_a=trans_a,
        trans_b=trans_b,
        satfinite=satfinite,
        sp_sel=sp_sel,
        mode=ctx.mode,
        trap=bool(getattr(ctx, "trap", False)),
    )
    err = variant_error(v)
    if err:
        _fail(ctx, f"{stmt.opcode}: {err}")
    if len(d) != d_regs(v):
        _fail(
            ctx,
            f"{stmt.opcode}: accumulator has {len(d)} registers, expected {d_regs(v)}",
        )
    for r in d:
        if not _is_reg(r):
            _fail(ctx, f"{stmt.opcode}: accumulator element {r!r} is not a register")
    if a_regs is not None:
        if len(a_regs) != 4:
            _fail(
                ctx,
                f"{stmt.opcode}: A register vector has {len(a_regs)} elements, expected 4",
            )
        for r in a_regs:
            if not _is_reg(r):
                _fail(ctx, f"{stmt.opcode}: A element {r!r} is not a register")
    for name, op in (("a-desc", adesc), ("b-desc", bdesc), ("sp-meta", meta)):
        if op is not None and not (_is_reg(op) or _imm(op) is not None):
            _fail(
                ctx,
                f"{stmt.opcode}: {name} {op!r} is neither a register nor a constant",
            )
    sd = _imm(scale_d)
    if sd is not None:
        if sd not in (0, 1):
            _fail(ctx, f"{stmt.opcode}: scale-d constant {scale_d!r} must be 0 or 1")
    elif not _PRED_RE.fullmatch(scale_d):
        _fail(ctx, f"{stmt.opcode}: scale-d {scale_d!r} is neither a predicate nor 0/1")
    else:
        pm = _PRED_RE.fullmatch(scale_d)
        ty = ctx.reg_type(pm.group(2)) if hasattr(ctx, "reg_type") else None
        if ty is not None and ty != "pred":
            _fail(
                ctx,
                f"{stmt.opcode}: scale-d register {pm.group(2)} is .{ty}, not .pred",
            )
    return Mma(v, d, a_regs, adesc, bdesc, meta, scale_d)


# ---------------------------------------------------------------------------------------------
# Per-function state
# ---------------------------------------------------------------------------------------------


def _operand_regs(op: str) -> list[str]:
    """Plain registers named by an operand: the operand itself or the elements of a vector.
    Address operands are not instrumented."""
    op = op.strip()
    if op.startswith("{") and op.endswith("}"):
        return [e.strip() for e in parse_vector(op) if _is_reg(e)]
    if _is_reg(op):
        return [op]
    return []


def _state(ctx: Ctx) -> dict:
    return ctx.func_state()


def _ensure(ctx: Ctx) -> dict:
    """Buffers and generic pointers, created once per function (normally in scan)."""
    st = _state(ctx)
    if st.get("ready"):
        return st
    st.setdefault("acc_ids", {})
    st.setdefault("a_ids", {})
    st.setdefault("acc_of_reg", {})
    st.setdefault("plan", {})
    st.setdefault("emitted", set())
    st["active"] = True
    dsym = ctx.local_buf("wgmma_d", D_BUF_BYTES, 16)
    lines, dptr = ctx.generic_addr_of_local(dsym)
    st["dsym"], st["dptr"] = dsym, dptr
    st["pro"] = list(lines)
    if ctx.mode == "check":
        nbytes = table_bytes(max(len(st["acc_ids"]), 1), max(len(st["a_ids"]), 1))
        psym = ctx.local_buf("wgmma_pend", nbytes, 16)
        plines, pptr = ctx.generic_addr_of_local(psym)
        st["psym"], st["pptr"] = psym, pptr
        st["pro"] += plines
    else:
        st["pptr"] = "0"
    st["ready"] = True
    return st


def scan(func: Func, ctx: Ctx) -> None:
    st = _state(ctx)
    wg = [
        it
        for it in func.body
        if isinstance(it, Stmt) and it.opcode.startswith("wgmma.")
    ]
    if not wg:
        st["active"] = False
        return
    if ctx.src_target != "sm_90a":
        _fail(ctx, f"wgmma requires .target sm_90a (module target is {ctx.src_target})")
    acc_ids: dict[tuple[str, ...], int] = {}
    a_ids: dict[tuple[str, ...], int] = {}
    acc_of_reg: dict[str, set[int]] = {}
    for s in wg:
        if _kind(s) != "mma_async":
            continue
        ops = [o.strip() for o in s.operands]
        if not ops or not ops[0].startswith("{"):
            continue  # parse_mma reports it during rewrite
        dk = tuple(e.strip() for e in parse_vector(ops[0]))
        if dk not in acc_ids:
            acc_ids[dk] = len(acc_ids)
            for r in dk:
                acc_of_reg.setdefault(r, set()).add(acc_ids[dk])
        if len(ops) > 1 and ops[1].startswith("{"):
            ak = tuple(e.strip() for e in parse_vector(ops[1]))
            if ak not in a_ids:
                a_ids[ak] = len(a_ids)
    if len(acc_ids) >= MAX_IDS or len(a_ids) >= MAX_IDS:
        _fail(
            ctx,
            f"too many wgmma accumulator / A register lists ({len(acc_ids)}, {len(a_ids)})",
        )
    st["acc_ids"], st["a_ids"], st["acc_of_reg"] = acc_ids, a_ids, acc_of_reg
    st["plan"], st["emitted"] = {}, set()
    _ensure(ctx)
    if ctx.mode == "check":
        if len(acc_ids) > 64:
            ctx.warn(
                f"check mode: {len(acc_ids)} accumulator lists in {func.name}; the pending table "
                f"takes {table_bytes(len(acc_ids), len(a_ids))} bytes of local memory per thread"
            )
        _plan_acc_checks(func, st)


def _plan_acc_checks(func: Func, st: dict) -> None:
    """For each non-wgmma statement, the accumulator ids to check before it: the first access
    of an id in a basic block (labels start blocks) after the last mma_async on that id."""
    acc_of_reg = st["acc_of_reg"]
    plan: dict[int, list[int]] = {}
    checked: set[int] = set()
    for it in func.body:
        if isinstance(it, str):
            if _LABEL_RE.fullmatch(it):
                checked = set()
            continue
        if it.opcode.startswith("wgmma."):
            if _kind(it) == "mma_async" and it.operands:
                op0 = it.operands[0].strip()
                if op0.startswith("{"):
                    for r in parse_vector(op0):
                        checked -= acc_of_reg.get(r.strip(), set())
            continue
        touched: set[int] = set()
        for op in it.operands:
            for r in _operand_regs(op):
                touched |= acc_of_reg.get(r, set())
        need = touched if it.guard else touched - checked
        if need:
            plan[id(it)] = sorted(need)
            if not it.guard:
                checked |= need
    st["plan"] = plan


def function_prologue(ctx: Ctx) -> list[str]:
    st = _state(ctx)
    if not st.get("active"):
        return []
    lines = list(st["pro"])
    if ctx.mode == "check":
        flags = WG_FLAG_TRAP if getattr(ctx, "trap", False) else 0
        lines += ctx.call(
            LIB_INIT,
            [
                ("b64", st["pptr"]),
                ("b32", str(len(st["acc_ids"]))),
                ("b32", str(len(st["a_ids"]))),
                ("b32", str(flags)),
            ],
        )
    return lines


# ---------------------------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------------------------


def match(stmt: Stmt) -> bool:
    return stmt.opcode.startswith("wgmma.")


def _pred_to_u32(ctx: Ctx, st: dict, pred_op: str) -> tuple[list[str], str]:
    m = _PRED_RE.fullmatch(pred_op.strip())
    if m is None:
        _fail(ctx, f"bad predicate operand {pred_op!r}")
    if "sd_reg" not in st:
        st["sd_reg"] = ctx.new_reg("b32")
    r = st["sd_reg"]
    t, f = ("0", "1") if m.group(1) == "!" else ("1", "0")
    return [f"selp.u32 \t{r}, {t}, {f}, {m.group(2)};"], r


def _rewrite_mma(stmt: Stmt, ctx: Ctx) -> list[str]:
    st = _ensure(ctx)
    mm = parse_mma(stmt, ctx)
    dsym, dptr, pptr = st["dsym"], st["dptr"], st["pptr"]
    acc_id = st["acc_ids"].get(tuple(mm.d))
    if acc_id is None:
        _fail(ctx, "internal: accumulator list missing from scan")
    if mm.a_regs is not None:
        a_id = st["a_ids"].get(tuple(mm.a_regs))
        if a_id is None:
            _fail(ctx, "internal: A register list missing from scan")
    else:
        a_id = IDS_NO_A
    ids = acc_id | (a_id << 16)
    lines: list[str] = []
    sd_imm = _imm(mm.scale_d)
    if not (ctx.mode in ("exact", "fast") and sd_imm == 0):
        for i, r in enumerate(mm.d):
            lines.append(f"st.local.b32 \t[{dsym}+{4 * i}], {r};")
    if sd_imm is None:
        pl, sd_op = _pred_to_u32(ctx, st, mm.scale_d)
        lines += pl
    else:
        sd_op = str(sd_imm)
    a = mm.a_regs if mm.a_regs is not None else ["0", "0", "0", "0"]
    adesc = mm.adesc if mm.adesc is not None else "0"
    args = [
        ("b64", dptr),
        ("b64", _arg(adesc)),
        ("b32", a[0]),
        ("b32", a[1]),
        ("b32", a[2]),
        ("b32", a[3]),
        ("b64", _arg(mm.bdesc)),
        ("b32", _arg(mm.meta) if mm.meta is not None else "0"),
        ("b32", sd_op),
        ("b32", str(mm.v.encode())),
        ("b32", str(ids)),
        ("b64", pptr),
    ]
    lines += ctx.call(LIB_MMA, args)
    for i, r in enumerate(mm.d):
        lines.append(f"ld.local.b32 \t{r}, [{dsym}+{4 * i}];")
    return lines


def _arg(op: str) -> str:
    """Register operands pass through; constants become decimal (unsigned for 64-bit)."""
    v = _imm(op)
    if v is None:
        return op.strip()
    return str(v & 0xFFFFFFFFFFFFFFFF)


def _rewrite_wait(stmt: Stmt, ctx: Ctx) -> list[str]:
    if len(stmt.operands) != 1:
        _fail(ctx, f"{stmt.opcode}: expected one operand N")
    n = _imm(stmt.operands[0])
    if n is None or n < 0:
        _fail(
            ctx,
            f"{stmt.opcode}: N {stmt.operands[0]!r} must be a non-negative integer constant",
        )
    if ctx.mode != "check":
        return []
    st = _ensure(ctx)
    pptr, dsym, dptr = st["pptr"], st["dsym"], st["dptr"]
    lines = ctx.call(LIB_WAIT, [("b64", pptr), ("b32", str(n))])
    if st["acc_ids"] and "fetch_p" not in st:
        st["fetch_p"] = ctx.new_reg("pred")
        st["fetch_r"] = ctx.new_reg("b32")
    for regs, acc_id in st["acc_ids"].items():
        for i, r in enumerate(regs):
            lines.append(f"st.local.b32 \t[{dsym}+{4 * i}], {r};")
        lines += ctx.call(
            LIB_FETCH,
            [("b64", pptr), ("b32", str(acc_id)), ("b64", dptr)],
            ret=("b32", st["fetch_r"]),
        )
        lines.append(f"setp.ne.u32 \t{st['fetch_p']}, {st['fetch_r']}, 0;")
        for i, r in enumerate(regs):
            lines.append(f"@{st['fetch_p']} ld.local.b32 \t{r}, [{dsym}+{4 * i}];")
    for regs, a_id in st["a_ids"].items():
        lines += ctx.call(
            LIB_CHECK_A,
            [("b64", pptr), ("b32", str(a_id))] + [("b32", r) for r in regs],
        )
    return lines


def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]:
    toks = stmt.opcode.split(".")
    kind = _kind(stmt)
    if kind == "mma_async":
        return _rewrite_mma(stmt, ctx)
    if kind in ("fence", "commit_group", "wait_group"):
        _sync_aligned(ctx, stmt, toks[2:])
        extra = [t for t in toks[2:] if t not in ("sync", "aligned")]
        if extra:
            _fail(ctx, f"{stmt.opcode}: unknown qualifiers {extra}")
        if kind == "wait_group":
            return _rewrite_wait(stmt, ctx)
        if stmt.operands:
            _fail(ctx, f"{stmt.opcode}: takes no operands")
        if ctx.mode != "check":
            return []
        st = _ensure(ctx)
        fn = LIB_FENCE if kind == "fence" else LIB_COMMIT
        return ctx.call(fn, [("b64", st["pptr"])])
    _fail(ctx, f"unsupported wgmma instruction {stmt.opcode}")
    return []


def rewrite_operand(op: str, stmt: Stmt, ctx: Ctx) -> tuple[list[str], str] | None:
    """Check mode: before the first instruction of a basic block that names a register of an
    accumulator id, call wgemu_wgmma_check_acc for every planned id. Only plain register
    operands (or vector elements, which the core offers one by one) trigger it, so operands
    that other rule modules rewrite are never claimed."""
    if ctx.mode != "check" or stmt.opcode.startswith("wgmma."):
        return None
    st = _state(ctx)
    if not st.get("active"):
        return None
    ids = st["plan"].get(id(stmt))
    if not ids or id(stmt) in st["emitted"]:
        return None
    o = op.strip()
    if not _is_reg(o) or o not in st["acc_of_reg"]:
        return None
    st["emitted"].add(id(stmt))
    lines: list[str] = []
    if stmt.guard:
        g = stmt.guard
        pred = ("!" if g.startswith("@!") else "") + (
            g[2:] if g.startswith("@!") else g[1:]
        )
        if "acc_kind" not in st:
            st["acc_kind"] = ctx.new_reg("b32")
        pl, kind = _pred_to_kind(st["acc_kind"], pred)
        lines += pl
    else:
        kind = "1"
    for acc_id in ids:
        lines += ctx.call(
            LIB_CHECK_ACC, [("b64", st["pptr"]), ("b32", str(acc_id)), ("b32", kind)]
        )
    return lines, op


def _pred_to_kind(reg: str, pred: str) -> tuple[list[str], str]:
    m = _PRED_RE.fullmatch(pred.strip())
    t, f = ("0", "1") if m.group(1) == "!" else ("1", "0")
    return [f"selp.u32 \t{reg}, {t}, {f}, {m.group(2)};"], reg
