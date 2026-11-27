"""Miscellaneous sm_86..sm_90a translation rules (PLAN sections 5 and 8). Owner: T2.

Every ``inventory.tsv`` row whose handling is ``inline``, ``noop`` or ``lib:wgemu_*`` and that is
not a wgmma (``rules_wgmma``) or async-proxy (``rules_async``) instruction is rewritten here:

* special registers of the cluster (``rewrite_operand``): with a cluster of one CTA,
  ``%clusterid`` = ``%ctaid``, ``%nclusterid`` = ``%nctaid``, ``%cluster_ctaid`` = 0,
  ``%cluster_nctaid`` = 1, ``%cluster_ctarank`` = 0, ``%cluster_nctarank`` = 1,
  ``%is_explicit_cluster`` = 1 in an entry with ``.explicitcluster`` or
  ``.reqnctapercluster`` (rejected elsewhere: the value depends on launch attributes, which
  host/shim.c drops for a 1x1x1 cluster), ``%aggr_smem_size`` =
  ``%total_smem_size + %reserved_smem_offset_cap``;
* ``barrier.cluster.arrive/wait``: a software CTA-wide split barrier in a module ``.shared``
  word (layout in ``include/wgmma_emu/misc_device.cuh``, ``kCbar*``) with warp-aggregated
  arrivals; exited threads leave the barrier (every ``exit``, and ``ret`` of an entry, of an
  instrumented function drops the thread first);
* ``cvt``: fp8 (``lib:wgemu_cvt_*fp8*``), ``.rn/.rz`` tf32 (``lib:wgemu_cvt_f32_to_tf32``),
  bf16 destinations SM80 lacks (``lib:wgemu_cvt_to_bf16``), bf16 sources (f32 round trip),
  ``.satfinite`` f16/bf16/tf32 (native conversion plus an inf -> MAX_NORM fix);
* bf16 ``add/sub/mul`` (``fma.rn`` identities), ``fma.oob``, ``min/max.xorsign.abs``, integer
  SIMD ``add/min/max`` on ``u16x2/s16x2`` and ``min/max.relu.s32``, ``tanh/ex2`` bf16,
  ``set/setp`` with bf16;
* memory: ``.cluster`` scope -> ``.gpu``, ``shared::cluster`` -> ``shared``, ``.unified`` and
  ``.proxy::readonly`` dropped, vector / bf16 / ``.noftz.f32`` atom and red (per element, CAS
  loops where SM80 has no native form), ``.b128`` atom (``lib:wgemu_atom_b128`` with a global
  lock table; one ``.weak`` table per linked device program, so -rdc units linked together
  share it and separately loaded modules do not; ``.sys`` scope is rejected), ``multimem.ld_reduce/st/red`` (one replica), ``multimem.cp{.reduce}.async.bulk``
  (one replica, handed to ``rules_async``), ``prefetch .L1::32B.valid_addr``,
  ``mapa``, ``getctarank``;
* ``elect.sync``, ``setmaxnreg`` / ``griddepcontrol`` (no-ops);
* ``stmatrix.m8n8`` and ``ldmatrix.m8n16 .s8.s4`` (shuffles and shared loads/stores);
* ``mma`` fp8 (``lib:wgemu_mma_fp8``), ``mma.sp`` fp8 (``lib:wgemu_mma_sp_fp8``) and f64
  m16n8k{4,8,16} (``lib:wgemu_mma_f64``).

The sm_90 ``fence`` / ``membar.proxy`` forms belong to ``rules_async`` (loaded first).
Forms of these instructions that need sm_100 or later are translation errors, never dropped.

Library calls (``runtime/lib_misc.cu``; flag words below must match it)::

    u32 wgemu_cvt_f32x2_to_fp8x2(u32 a, u32 b, u32 flags)      bit0 e5m2, bit1 relu, bit2 satfinite
    u32 wgemu_cvt_f16x2_to_fp8x2(u32 a, u32 flags)
    u32 wgemu_cvt_fp8x2_to_f16x2(u32 a, u32 flags)             a in the low 16 bits
    u32 wgemu_cvt_f32_to_tf32(u32 a, u32 ctrl)                 bits 0..1 rnd (0 rn, 1 rz), bit2 relu
    u32 wgemu_cvt_to_bf16(u64 a, u32 ctrl)                     bits 0..3 source kind, 4..6 rnd,
                                                               bit8 ftz, bit9 sat
    void wgemu_mma_fp8(u32 a0..a3, u32 b0, b1, u32 c0..c3, u64 out, u32 cfg)
    void wgemu_mma_sp_fp8(u32 a0..a3, u32 b0..b3, u32 c0..c3, u32 e, u64 out, u32 cfg)
    void wgemu_mma_f64(u64 a0..a7, u64 b0..b3, u64 c0..c3, u64 out, u32 cfg)
    void wgemu_atom_b128(u64 addr, u64 locks, u32 cfg, u64 cmp_lo, cmp_hi, new_lo, new_hi, u64 out)

Inferences ([inference], also listed in the T2 report):

* OOB-NaN bit patterns 0x7FFF / 0x7FFFFFFF and "either operand" = any of a, b, c (fma.oob);
* %clusterid.w / %nclusterid.w / %cluster_ctaid.w / %cluster_nctaid.w are 0 (as %tid.w);
* %aggr_smem_size = %total_smem_size + %reserved_smem_offset_cap;
* elect.sync elects the lowest lane of membermask;
* set with a bf16 source: the 1.0 / 0xffff selection follows dtype (the PTX pseudo code tests
  ``stype`` where the description says dtype);
* multimem.* act on the local replica only (multimem.cp{.reduce}.async.bulk become the plain
  sm_90 bulk forms of rules_async); multimem.red.add.f32 does not flush subnormals;
* ldmatrix .s8.s4: the lower nibble of each source byte is the lower column;
* barrier.cluster.wait without a prior arrive traps (PTX leaves it undefined).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .core import (
    Ctx,
    Func,
    Stmt,
    TranslateError,
    parse_address,
    parse_int,
    parse_vector,
    render_stmt,
    split_dests,
)

# ---------------------------------------------------------------------------------------------
# Library entry points and encodings (keep identical to runtime/lib_misc.cu)
# ---------------------------------------------------------------------------------------------

LIB_CVT_F32X2_FP8X2 = "wgemu_cvt_f32x2_to_fp8x2"
LIB_CVT_F16X2_FP8X2 = "wgemu_cvt_f16x2_to_fp8x2"
LIB_CVT_FP8X2_F16X2 = "wgemu_cvt_fp8x2_to_f16x2"
LIB_CVT_TF32 = "wgemu_cvt_f32_to_tf32"
LIB_CVT_BF16 = "wgemu_cvt_to_bf16"
LIB_MMA_FP8 = "wgemu_mma_fp8"
LIB_MMA_SP_FP8 = "wgemu_mma_sp_fp8"
LIB_MMA_F64 = "wgemu_mma_f64"
LIB_ATOM_B128 = "wgemu_atom_b128"

FP8_E5M2 = 1
FP8_RELU = 2
FP8_SATFINITE = 4
TF32_RELU = 4
RND = {"rn": 0, "rz": 1, "rm": 2, "rp": 3}
BF16_SRC = {
    "f32": 0,
    "f16": 1,
    "f64": 2,
    "u8": 3,
    "u16": 4,
    "u32": 5,
    "u64": 6,
    "s8": 7,
    "s16": 8,
    "s32": 9,
    "s64": 10,
}
BF16_RND_SHIFT = 4
BF16_FTZ = 1 << 8
BF16_SAT = 1 << 9
MMA_A_E5M2 = 1
MMA_B_E5M2 = 2
MMA_D_F16 = 4
MMA_K32 = 8
MMA_ADA = 16
F64_K = {4: 0, 8: 1, 16: 2}
F64_RND_SHIFT = 2
B128_CAS = 1
B128_SYS = 2  # library bit; not emitted, atom.sys.b128 is rejected

# include/wgmma_emu/misc_device.cuh
CBAR_BYTES = 8 + 1024
CBAR_CNT_MASK = 0x7FF
CBAR_EXP_SHIFT = 11
CBAR_GEN_SHIFT = 22
B128_LOCK_BYTES = 4 * 1024
OOB_NAN16 = 0x7FFF

# ---------------------------------------------------------------------------------------------
# Operand helpers
# ---------------------------------------------------------------------------------------------

_REG_RE = re.compile(r"%[A-Za-z_$][\w$]*\Z")
_F32_IMM_RE = re.compile(r"0[fF]([0-9a-fA-F]{8})\Z")
_F64_IMM_RE = re.compile(r"0[dD]([0-9a-fA-F]{16})\Z")
_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*\Z")
_SHAPE_RE = re.compile(r"m(\d+)n(\d+)k(\d+)\Z")

_INT_T = ("u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64")
_FP8 = ("e4m3", "e5m2")
_FP8_ALL = ("e4m3", "e5m2", "e3m2", "e2m3", "e2m1")
_FP8X2 = ("e4m3x2", "e5m2x2")
_SCOPES = ("cta", "cluster", "gpu", "sys")


def _is_reg(op: str) -> bool:
    return bool(_REG_RE.match(op.strip()))


def _imm(op: str, bits: int) -> int | None:
    """Value of an integer or 0f/0d immediate masked to ``bits``, or None."""
    s = op.strip()
    m = _F32_IMM_RE.match(s)
    if m:
        return int(m.group(1), 16) if bits == 32 else None
    m = _F64_IMM_RE.match(s)
    if m:
        return int(m.group(1), 16) if bits == 64 else None
    try:
        v = parse_int(s)
    except TranslateError:
        return None
    return v & ((1 << bits) - 1)


def _reg(ctx: Ctx, op: str, bits: int) -> tuple[list[str], str]:
    """``op`` as a register: registers unchanged, immediates moved into a new b<bits>."""
    s = op.strip()
    if _is_reg(s):
        return [], s
    v = _imm(s, bits)
    if v is None:
        ctx.error(f"operand {op!r} is neither a register nor a {bits}-bit immediate")
    r = ctx.new_reg(f"b{bits}")
    return [f"mov.b{bits} \t{r}, 0x{v:X};"], r


def _u32_of16(ctx: Ctx, op: str) -> tuple[list[str], str]:
    """b32 register holding the low 16 bits of ``op`` zero-extended."""
    s = op.strip()
    r = ctx.new_reg("b32")
    if _is_reg(s):
        bits = ctx.reg_bits(s)
        if bits == 16:
            h = ctx.new_reg("b16")
            return [f"mov.b16 \t{h}, {s};", f"cvt.u32.u16 \t{r}, {h};"], r
        if bits == 32:
            return [f"and.b32 \t{r}, {s}, 65535;"], r
        ctx.error(f"16-bit operand {s} has width {bits}")
    v = _imm(s, 16)
    if v is None:
        ctx.error(f"operand {op!r} is neither a register nor a 16-bit immediate")
    return [f"mov.b32 \t{r}, {v};"], r


def _put16(ctx: Ctx, dst: str, r32: str) -> list[str]:
    """Write the low 16 bits of b32 ``r32`` to ``dst`` (16-bit, or zero-extended into 32-bit)."""
    d = dst.strip()
    if d == "_":
        return []
    bits = ctx.reg_bits(d)
    if bits == 16:
        h = ctx.new_reg("b16")
        return [f"cvt.u16.u32 \t{h}, {r32};", f"mov.b16 \t{d}, {h};"]
    if bits == 32:
        return [f"and.b32 \t{d}, {r32}, 65535;"]
    ctx.error(f"16-bit destination {d} has width {bits}")
    return []


def _put16_from16(ctx: Ctx, dst: str, h: str) -> list[str]:
    """Write b16 ``h`` to ``dst`` (16-bit register, or zero-extended into a 32-bit register)."""
    d = dst.strip()
    bits = ctx.reg_bits(d)
    if bits == 16:
        return [f"mov.b16 \t{d}, {h};"]
    if bits == 32:
        return [f"cvt.u32.u16 \t{d}, {h};"]
    ctx.error(f"16-bit destination {d} has width {bits}")
    return []


def _u64_arg(ctx: Ctx, op: str, ty: str) -> tuple[list[str], str]:
    """b64 register holding ``op`` of PTX type ``ty`` for wgemu_cvt_to_bf16 (raw bits,
    integers extended from their own width; the library re-truncates)."""
    s = op.strip()
    r = ctx.new_reg("b64")
    if not _is_reg(s):
        bits = {"f32": 32, "f16": 16, "f64": 64}.get(ty) or int(ty[1:])
        v = _imm(s, bits)
        if v is None:
            ctx.error(f"operand {op!r} is not a valid .{ty} immediate")
        return [f"mov.b64 \t{r}, 0x{v:X};"], r
    if ty == "f64" or ty in ("u64", "s64"):
        return [f"mov.b64 \t{r}, {s};"], r
    if ty == "f32":
        t = ctx.new_reg("b32")
        return [f"mov.b32 \t{t}, {s};", f"cvt.u64.u32 \t{r}, {t};"], r
    if ty == "f16":
        h = ctx.new_reg("b16")
        return [f"mov.b16 \t{h}, {s};", f"cvt.u64.u16 \t{r}, {h};"], r
    if ty in _INT_T:
        sign = "s" if ty[0] == "s" else "u"
        return [f"cvt.{sign}64.{ty} \t{r}, {s};"], r
    ctx.error(f"unsupported source type .{ty}")
    return [], r


def _addr_plus(op: str, delta: int) -> str:
    """Address operand ``op`` ('[base+off]') displaced by ``delta`` bytes."""
    base, off = parse_address(op)
    o = off + delta
    if not base:
        return f"[{o}]"
    if o == 0:
        return f"[{base}]"
    return f"[{base}+{o}]" if o > 0 else f"[{base}-{-o}]"


def _vec(op: str) -> list[str]:
    s = op.strip()
    return parse_vector(s) if s.startswith("{") else [s]


def _plain(stmt: Stmt) -> str:
    return render_stmt(None, stmt.opcode, stmt.operands) + ";"


def _local_ptr(ctx: Ctx, key: str, nbytes: int) -> tuple[str, str]:
    """Function-scope .local buffer and a b64 register with its generic address (set up once
    in the function prologue)."""
    st = ctx.func_state()
    bufs = st.setdefault("bufs", {})
    if key not in bufs:
        sym = ctx.local_buf(key, nbytes, 16)
        a = ctx.new_reg("b64")
        g = ctx.new_reg("b64")
        st.setdefault("pro", []).extend(
            [f"mov.u64 \t{a}, {sym};", f"cvta.local.u64 \t{g}, {a};"]
        )
        bufs[key] = (sym, g)
    return bufs[key]


def _src_arch(ctx: Ctx) -> str:
    m = re.match(r"sm_(\d+)", ctx.src_target or "")
    n = int(m.group(1)) if m else 0
    if n == 89:
        return "ada"
    if n >= 90:
        return "hopper"
    ctx.error(
        f"fp8 mma needs .target sm_89 or sm_90 (module target is {ctx.src_target})"
    )
    return ""


# ---------------------------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------------------------

Pred = Callable[[Stmt, list[str]], bool]
Handler = Callable[[Stmt, list[str], Ctx], list[str]]
_RULES: list[tuple[Pred, Handler]] = []


def _rule(pred: Pred) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        _RULES.append((pred, fn))
        return fn

    return deco


def _find(stmt: Stmt) -> Handler | None:
    toks = stmt.opcode.split(".")
    for pred, fn in _RULES:
        if pred(stmt, toks):
            return fn
    return None


def match(stmt: Stmt) -> bool:
    return _find(stmt) is not None


def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]:
    fn = _find(stmt)
    if fn is None:
        ctx.error(f"rules_misc.rewrite called for unmatched {stmt.opcode}")
    assert fn is not None
    return fn(stmt, stmt.opcode.split("."), ctx)


def _redispatch(stmt: Stmt, opcode: str, operands: list[str], ctx: Ctx) -> list[str]:
    new = Stmt(
        None,
        opcode,
        operands,
        render_stmt(None, opcode, operands),
        stmt.lineno,
        stmt.in_asm,
    )
    fn = _find(new)
    if fn is None:
        return [new.text + ";"]
    return fn(new, opcode.split("."), ctx)


# ---------------------------------------------------------------------------------------------
# Per-function scan: software cluster barrier instrumentation
# ---------------------------------------------------------------------------------------------

# Rule state of the function being translated. match() has no Ctx, so scan() publishes the
# exit/ret instrumentation decision here; the core translates one function at a time
# (scan, statements, prologue), which this relies on.
_FN: dict = {"entry": False, "entry_drop": False, "exit_drop": False}

_CBAR_RE = re.compile(r"(?<![\w.$])barrier\.cluster\b")
_BUILTIN_CALLEES = frozenset({"vprintf", "__assertfail", "malloc", "free"})


def _call_target(stmt: Stmt) -> str:
    for op in stmt.operands:
        s = op.strip()
        if s.startswith("("):
            continue
        return s if _IDENT_RE.match(s) else "<indirect>"
    return "<indirect>"


def _closure(name: str, funcs: dict) -> set[str]:
    seen = {name}
    todo = [name]
    while todo:
        n = todo.pop()
        for c in funcs.get(n, (False, set(), False))[1]:
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return seen


def _reaches_cbar(name: str, funcs: dict) -> bool:
    return any(funcs[n][0] for n in _closure(name, funcs) if n in funcs)


def scan(func: Func, ctx: Ctx) -> None:
    st = ctx.func_state()
    ms = ctx.module_state()
    funcs: dict = ms.setdefault("funcs", {})
    stmts = [it for it in func.body if isinstance(it, Stmt)]
    uses = any(s.opcode.startswith("barrier.cluster") for s in stmts)
    has_exit = any(s.opcode.split(".")[0] == "exit" for s in stmts)
    callees = {_call_target(s) for s in stmts if s.opcode.split(".")[0] == "call"}
    callees = {
        c for c in callees if not (c.startswith("wgemu_") or c in _BUILTIN_CALLEES)
    }
    funcs[func.name] = (uses, callees, has_exit)
    if uses and not func.is_entry and func.linkage in (".visible", ".weak"):
        ctx.warn(
            f"{func.name} uses barrier.cluster; callers in other modules (-rdc) do not set up "
            "the software cluster barrier of this module"
        )
    text = getattr(ctx, "module_text", None)
    if text is not None:
        if "need" not in ms:
            ms["need"] = bool(_CBAR_RE.search(text))
        entry_drop = func.is_entry and ms["need"]
        exit_drop = ms["need"]
    else:
        # Without the module text, only functions seen so far are known. Entries and exit
        # sites decided before a later barrier.cluster appears are re-checked on every scan.
        ms["seen"] = ms.get("seen", False) or uses
        entry_drop = func.is_entry and _reaches_cbar(func.name, funcs)
        exit_drop = ms["seen"] or _reaches_cbar(func.name, funcs)
        if func.is_entry and not entry_drop:
            ms.setdefault("plain_entries", []).append(func.name)
        if has_exit and not func.is_entry and not exit_drop:
            ms.setdefault("plain_exit_funcs", []).append(func.name)
        for e in ms.get("plain_entries", []):
            if _reaches_cbar(e, funcs):
                ctx.error(
                    f"entry {e} reaches barrier.cluster through a function defined after it; "
                    "rules_misc needs Ctx.module_text (T1) to instrument it"
                )
        if ms["seen"] and ms.get("plain_exit_funcs"):
            ctx.error(
                f"exit in {ms['plain_exit_funcs']} precedes the first barrier.cluster of the "
                "module; rules_misc needs Ctx.module_text (T1) to instrument it"
            )
    if entry_drop:
        last = stmts[-1] if stmts else None
        if (
            last is None
            or last.guard
            or last.opcode.split(".")[0] not in ("ret", "exit", "bra")
        ):
            ctx.error(
                f"entry {func.name} does not end with an unconditional ret/exit/bra"
            )
    st["entry"] = func.is_entry
    st["entry_drop"] = entry_drop
    st["exit_drop"] = exit_drop
    _FN.update(entry=func.is_entry, entry_drop=entry_drop, exit_drop=exit_drop)


def _cbar(ctx: Ctx) -> dict:
    st = ctx.func_state()
    cb = st.get("cbar")
    if cb is None:
        cb = {
            "sym": ctx.shared_var("cluster_barrier", CBAR_BYTES, 8),
            "base": ctx.new_reg("b32"),
            "rec": ctx.new_reg("b32"),
        }
        st["cbar"] = cb
    return cb


def function_prologue(ctx: Ctx) -> list[str]:
    st = ctx.func_state()
    if st.get("entry_drop"):
        _cbar(ctx)
    lines = list(st.get("pro", []))
    cb = st.get("cbar")
    if cb is None:
        return lines
    n = ctx.new_reg
    tx, ty, tz, nx, ny, nz, t1, lt = (n("b32") for _ in range(8))
    lines += [
        f"mov.u32 \t{cb['base']}, {cb['sym']};",
        f"mov.u32 \t{tx}, %tid.x;",
        f"mov.u32 \t{ty}, %tid.y;",
        f"mov.u32 \t{tz}, %tid.z;",
        f"mov.u32 \t{nx}, %ntid.x;",
        f"mov.u32 \t{ny}, %ntid.y;",
        f"mov.u32 \t{nz}, %ntid.z;",
        f"mad.lo.u32 \t{t1}, {tz}, {ny}, {ty};",
        f"mad.lo.u32 \t{lt}, {t1}, {nx}, {tx};",
        f"add.u32 \t{t1}, {cb['base']}, {lt};",
        f"add.u32 \t{cb['rec']}, {t1}, 8;",
    ]
    if st.get("entry_drop"):
        nn, w, z = n("b32"), n("b32"), n("b32")
        p = n("pred")
        skip = ctx.new_label("cbar_init")
        lines += [
            f"mul.lo.u32 \t{nn}, {nx}, {ny};",
            f"mul.lo.u32 \t{nn}, {nn}, {nz};",
            f"shl.b32 \t{w}, {nn}, {CBAR_EXP_SHIFT};",
            f"mov.u32 \t{z}, 0;",
            f"setp.ne.u32 \t{p}, {lt}, 0;",
            f"@{p} bra \t{skip};",
            f"st.shared.u32 \t[{cb['base']}], {w};",
            f"{skip}:",
            f"st.shared.u8 \t[{cb['rec']}], {z};",
            "bar.sync \t0;",
        ]
    return lines


def _leader(ctx: Ctx) -> tuple[list[str], str, str, str, str]:
    """(lines, activemask, its popcount, lowest active lane, pred 'this lane is not it')."""
    n = ctx.new_reg
    m, cnt, ln, neg, lsb, ldr = (n("b32") for _ in range(6))
    notl = n("pred")
    lines = [
        f"activemask.b32 \t{m};",
        f"popc.b32 \t{cnt}, {m};",
        f"mov.u32 \t{ln}, %laneid;",
        f"neg.s32 \t{neg}, {m};",
        f"and.b32 \t{lsb}, {m}, {neg};",
        f"bfind.u32 \t{ldr}, {lsb};",
        f"setp.ne.u32 \t{notl}, {ln}, {ldr};",
    ]
    return lines, m, cnt, ldr, notl


def _cbar_fields(ctx: Ctx, old: str) -> tuple[list[str], str, str, str]:
    cnt, exp, gen = ctx.new_reg("b32"), ctx.new_reg("b32"), ctx.new_reg("b32")
    return (
        [
            f"and.b32 \t{cnt}, {old}, {CBAR_CNT_MASK};",
            f"bfe.u32 \t{exp}, {old}, {CBAR_EXP_SHIFT}, 11;",
            f"shr.u32 \t{gen}, {old}, {CBAR_GEN_SHIFT};",
        ],
        cnt,
        exp,
        gen,
    )


def _cbar_arrive(ctx: Ctx) -> list[str]:
    cb = _cbar(ctx)
    base, rec = cb["base"], cb["rec"]
    n = ctx.new_reg
    lead, m, cnt_n, ldr, notl = _leader(ctx)
    loop, done = ctx.new_label("cbar_arrive"), ctx.new_label("cbar_arrived")
    old, c2, g1, es, vfull, vmore, nv, prev, gsh, par = (n("b32") for _ in range(10))
    full, retry = n("pred"), n("pred")
    fl, cnt, exp, gen = _cbar_fields(ctx, old)
    return [
        *lead,
        f"@{notl} bra \t{done};",
        f"{loop}:",
        f"ld.relaxed.cta.shared.u32 \t{old}, [{base}];",
        *fl,
        f"add.u32 \t{c2}, {cnt}, {cnt_n};",
        f"setp.eq.u32 \t{full}, {c2}, {exp};",
        f"add.u32 \t{g1}, {gen}, 1;",
        f"shl.b32 \t{g1}, {g1}, {CBAR_GEN_SHIFT};",
        f"shl.b32 \t{es}, {exp}, {CBAR_EXP_SHIFT};",
        f"or.b32 \t{vfull}, {g1}, {es};",
        f"add.u32 \t{vmore}, {old}, {cnt_n};",
        f"selp.b32 \t{nv}, {vfull}, {vmore}, {full};",
        f"atom.acq_rel.cta.shared.cas.b32 \t{prev}, [{base}], {old}, {nv};",
        f"setp.ne.b32 \t{retry}, {prev}, {old};",
        f"@{retry} bra \t{loop};",
        f"{done}:",
        f"shfl.sync.idx.b32 \t{gsh}, {gen}, {ldr}, 31, {m};",
        f"and.b32 \t{par}, {gsh}, 1;",
        f"add.u32 \t{par}, {par}, 1;",
        f"st.shared.u8 \t[{rec}], {par};",
    ]


def _cbar_wait(ctx: Ctx) -> list[str]:
    cb = _cbar(ctx)
    base, rec = cb["base"], cb["rec"]
    n = ctx.new_reg
    r, par, cur, g = (n("b32") for _ in range(4))
    none, done = n("pred"), n("pred")
    loop, end = ctx.new_label("cbar_wait"), ctx.new_label("cbar_waited")
    return [
        f"ld.shared.u8 \t{r}, [{rec}];",
        f"setp.eq.u32 \t{none}, {r}, 0;",
        f"@{none} trap;",
        f"sub.u32 \t{par}, {r}, 1;",
        f"{loop}:",
        f"ld.acquire.cta.shared.u32 \t{cur}, [{base}];",
        f"shr.u32 \t{g}, {cur}, {CBAR_GEN_SHIFT};",
        f"and.b32 \t{g}, {g}, 1;",
        f"setp.ne.u32 \t{done}, {g}, {par};",
        f"@{done} bra \t{end};",
        "nanosleep.u32 \t32;",
        f"bra \t{loop};",
        f"{end}:",
    ]


def _cbar_drop(ctx: Ctx) -> list[str]:
    """The executing threads leave the barrier: expected -= #threads, arrived -= #threads that
    arrived in the current phase; completes the phase if everyone left has arrived."""
    cb = _cbar(ctx)
    base, rec = cb["base"], cb["rec"]
    n = ctx.new_reg
    lead, m, cnt_n, ldr, notl = _leader(ctx)
    r, b1, b2, old, par, bb, arr, c2, e2, es, g1, vfull, gs, vmore, nv, prev = (
        n("b32") for _ in range(16)
    )
    q1, q2, podd, nz, eq, full, retry = (n("pred") for _ in range(7))
    loop, done = ctx.new_label("cbar_drop"), ctx.new_label("cbar_dropped")
    fl, cnt, exp, gen = _cbar_fields(ctx, old)
    return [
        f"ld.shared.u8 \t{r}, [{rec}];",
        *lead,
        f"setp.eq.u32 \t{q1}, {r}, 1;",
        f"vote.sync.ballot.b32 \t{b1}, {q1}, {m};",
        f"setp.eq.u32 \t{q2}, {r}, 2;",
        f"vote.sync.ballot.b32 \t{b2}, {q2}, {m};",
        f"@{notl} bra \t{done};",
        f"{loop}:",
        f"ld.relaxed.cta.shared.u32 \t{old}, [{base}];",
        *fl,
        f"and.b32 \t{par}, {gen}, 1;",
        f"setp.ne.u32 \t{podd}, {par}, 0;",
        f"selp.b32 \t{bb}, {b2}, {b1}, {podd};",
        f"popc.b32 \t{arr}, {bb};",
        f"sub.u32 \t{c2}, {cnt}, {arr};",
        f"sub.u32 \t{e2}, {exp}, {cnt_n};",
        f"setp.ne.u32 \t{nz}, {e2}, 0;",
        f"setp.eq.u32 \t{eq}, {c2}, {e2};",
        f"and.pred \t{full}, {nz}, {eq};",
        f"shl.b32 \t{es}, {e2}, {CBAR_EXP_SHIFT};",
        f"add.u32 \t{g1}, {gen}, 1;",
        f"shl.b32 \t{g1}, {g1}, {CBAR_GEN_SHIFT};",
        f"or.b32 \t{vfull}, {g1}, {es};",
        f"shl.b32 \t{gs}, {gen}, {CBAR_GEN_SHIFT};",
        f"or.b32 \t{vmore}, {gs}, {es};",
        f"or.b32 \t{vmore}, {vmore}, {c2};",
        f"selp.b32 \t{nv}, {vfull}, {vmore}, {full};",
        f"atom.acq_rel.cta.shared.cas.b32 \t{prev}, [{base}], {old}, {nv};",
        f"setp.ne.b32 \t{retry}, {prev}, {old};",
        f"@{retry} bra \t{loop};",
        f"{done}:",
    ]


def _is_drop_point(stmt: Stmt, toks: list[str]) -> bool:
    if toks[0] == "exit":
        return bool(_FN["entry_drop"] if _FN["entry"] else _FN["exit_drop"])
    if toks[0] == "ret":
        return bool(_FN["entry"] and _FN["entry_drop"])
    return False


@_rule(_is_drop_point)
def _rw_drop(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    return [*_cbar_drop(ctx), _plain(stmt)]


@_rule(lambda s, t: t[0] == "barrier" and len(t) > 2 and t[1] == "cluster")
def _rw_cbar(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if stmt.operands:
        ctx.error(f"{stmt.opcode} takes no operands")
    mods = set(toks[3:])
    if toks[2] == "arrive" and mods <= {"release", "relaxed", "aligned"}:
        return _cbar_arrive(ctx)
    if toks[2] == "wait" and mods <= {"acquire", "aligned"}:
        return _cbar_wait(ctx)
    ctx.error(f"unsupported barrier.cluster form {stmt.opcode}")
    return []


# ---------------------------------------------------------------------------------------------
# Special registers
# ---------------------------------------------------------------------------------------------

_SREG_VEC = ("clusterid", "nclusterid", "cluster_ctaid", "cluster_nctaid")
_SREG_RE = re.compile(
    r"%(clusterid|nclusterid|cluster_ctaid|cluster_nctaid|cluster_ctarank|cluster_nctarank"
    r"|is_explicit_cluster|aggr_smem_size)(?:\.([xyzw]))?\Z"
)
_SREG_VEC_RE = re.compile(r"%(" + "|".join(_SREG_VEC) + r")\Z")


def _sreg_value(name: str, comp: str | None) -> str:
    if name == "clusterid":
        return "0" if comp == "w" else f"%ctaid.{comp}"
    if name == "nclusterid":
        return "0" if comp == "w" else f"%nctaid.{comp}"
    if name == "cluster_ctaid":
        return "0"
    if name == "cluster_nctaid":
        return "0" if comp == "w" else "1"
    if name == "cluster_ctarank":
        return "0"
    if name == "cluster_nctarank":
        return "1"
    raise TranslateError(f"no value for %{name}")


def _explicit_cluster(ctx: Ctx) -> bool:
    """%is_explicit_cluster as a constant (ptx.txt 34853-34861): true when the entry's own
    .explicitcluster / .reqnctapercluster make every launch explicit. Otherwise the value
    depends on the launch (a cudaLaunchKernelEx cluster attribute, which host/shim.c drops
    for 1x1x1), so the read is rejected rather than folded to false."""
    info = getattr(ctx, "cluster_info", None)
    if info is None:
        ctx.error(
            "%is_explicit_cluster outside an entry: the launch is not known here; "
            "unsupported"
        )
    if not (info.get("explicitcluster") or info.get("reqnctapercluster")):
        ctx.error(
            "%is_explicit_cluster in an entry without .explicitcluster or "
            ".reqnctapercluster: the value depends on launch attributes that the "
            "emulation drops; unsupported"
        )
    return True


def rewrite_operand(op: str, stmt: Stmt, ctx: Ctx) -> tuple[list[str], str] | None:
    m = _SREG_RE.match(op.strip())
    if not m:
        return None
    name, comp = m.group(1), m.group(2)
    vec = name in _SREG_VEC
    if vec and comp is None:
        if stmt.opcode.startswith("mov.v"):
            return None  # whole-vector mov: rewritten by _rw_sreg_mov
        ctx.error(f"%{name} without a component outside mov.v*")
    if not vec and comp is not None:
        ctx.error(f"%{name} has no component .{comp}")
    if name == "is_explicit_cluster":
        t = ctx.new_reg("b32")
        p = ctx.new_reg("pred")
        flag = 1 if _explicit_cluster(ctx) else 0
        return [f"mov.u32 \t{t}, {flag};", f"setp.ne.u32 \t{p}, {t}, 0;"], p
    r = ctx.new_reg("b32")
    if name == "aggr_smem_size":
        t1, t2 = ctx.new_reg("b32"), ctx.new_reg("b32")
        return [
            f"mov.u32 \t{t1}, %total_smem_size;",
            f"mov.u32 \t{t2}, %reserved_smem_offset_cap;",
            f"add.u32 \t{r}, {t1}, {t2};",
        ], r
    return [f"mov.u32 \t{r}, {_sreg_value(name, comp)};"], r


@_rule(
    lambda s, t: (
        t[0] == "mov"
        and len(s.operands) == 2
        and bool(_SREG_VEC_RE.match(s.operands[1].strip()))
    )
)
def _rw_sreg_mov(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    name = stmt.operands[1].strip()[1:]
    dsts = _vec(stmt.operands[0])
    if not 1 < len(dsts) <= 4:
        ctx.error(f"mov of %{name} needs a 2..4 element vector destination")
    lines = []
    for d, comp in zip(dsts, "xyzw", strict=False):
        if d != "_":
            lines.append(f"mov.u32 \t{d}, {_sreg_value(name, comp)};")
    return lines


# ---------------------------------------------------------------------------------------------
# cvt
# ---------------------------------------------------------------------------------------------

_CVT_SM100_T = frozenset(
    {
        "e2m1x2",
        "e2m3x2",
        "e3m2x2",
        "ue8m0x2",
        "ue5m3x2",
        "s2f6x2",
        "e2m1x4",
        "e4m3x4",
        "e5m2x4",
        "e2m3x4",
        "e3m2x4",
    }
)
_SATFIN_DT = ("f16", "bf16", "f16x2", "bf16x2", "tf32")
_IRND = ("rni", "rzi", "rmi", "rpi")
_FRND = ("rn", "rz", "rm", "rp")


def _cvt_sm100(dt: str, at: str, mods: list[str]) -> bool:
    return (
        dt in _CVT_SM100_T
        or at in _CVT_SM100_T
        or any(m in ("rs", "pzo") or m.startswith("scaled") for m in mods)
    )


def _cvt_match(s: Stmt, t: list[str]) -> bool:
    if t[0] != "cvt" or len(t) < 3:
        return False
    dt, at, mods = t[-2], t[-1], t[1:-2]
    if _cvt_sm100(dt, at, mods) or dt in _FP8X2 or at in _FP8X2:
        return True
    if dt == "tf32":
        return any(m in ("rn", "rz", "satfinite") for m in mods)
    if "satfinite" in mods and dt in _SATFIN_DT and at == "f32":
        return True
    if dt == "bf16" and at != "f32":
        return True
    if at == "bf16" and (dt != "f32" or bool(mods)):
        return True
    return dt == "bf16" and at == "f32" and ("rm" in mods or "rp" in mods)


@_rule(_cvt_match)
def _rw_cvt(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    dt, at, mods = toks[-2], toks[-1], toks[1:-2]
    if _cvt_sm100(dt, at, mods):
        ctx.error(f"{stmt.opcode} needs sm_100 or later; not emulated")
    if dt in _FP8X2 or at in _FP8X2:
        return _cvt_fp8(stmt, dt, at, mods, ctx)
    if dt == "tf32":
        if "satfinite" in mods:
            if "rna" not in mods:
                ctx.error(
                    f"{stmt.opcode}: .rn/.rz with .satfinite needs sm_100 or later"
                )
            return _cvt_satfinite(stmt, dt, at, mods, ctx)
        return _cvt_tf32(stmt, mods, ctx)
    if "satfinite" in mods and at == "f32" and dt in _SATFIN_DT:
        return _cvt_satfinite(stmt, dt, at, mods, ctx)
    if at == "bf16":
        return _cvt_from_bf16(stmt, dt, mods, ctx)
    return _cvt_to_bf16(stmt, at, mods, ctx)


def _ops(ctx: Ctx, stmt: Stmt, n: int) -> list[str]:
    if len(stmt.operands) != n:
        ctx.error(f"{stmt.opcode} expects {n} operands, got {len(stmt.operands)}")
    return [o.strip() for o in stmt.operands]


def _cvt_fp8(stmt: Stmt, dt: str, at: str, mods: list[str], ctx: Ctx) -> list[str]:
    ms = set(mods)
    if not ms <= {"rn", "relu", "satfinite"} or "rn" not in ms:
        ctx.error(
            f"{stmt.opcode}: fp8 cvt on sm_89/sm_90 is .rn{{.satfinite}}{{.relu}} only"
        )
    e5 = FP8_E5M2 if "e5m2x2" in (dt, at) else 0
    relu = FP8_RELU if "relu" in ms else 0
    lines: list[str] = []
    r = ctx.new_reg("b32")
    if dt in _FP8X2:
        if "satfinite" not in ms:
            ctx.error(f"{stmt.opcode}: .satfinite is required for an fp8 destination")
        flags = e5 | relu | FP8_SATFINITE
        if at == "f32":
            d, a, b = _ops(ctx, stmt, 3)
            la, ra = _reg(ctx, a, 32)
            lb, rb = _reg(ctx, b, 32)
            lines += la + lb
            lines += ctx.call(
                LIB_CVT_F32X2_FP8X2,
                [("b32", ra), ("b32", rb), ("b32", str(flags))],
                ("b32", r),
            )
        elif at == "f16x2":
            d, a = _ops(ctx, stmt, 2)
            la, ra = _reg(ctx, a, 32)
            lines += la
            lines += ctx.call(
                LIB_CVT_F16X2_FP8X2, [("b32", ra), ("b32", str(flags))], ("b32", r)
            )
        else:
            ctx.error(f"{stmt.opcode}: .{at} source needs sm_100 or later")
            return []
        return lines + _put16(ctx, d, r)
    if dt != "f16x2" or "satfinite" in ms:
        ctx.error(
            f"{stmt.opcode}: only cvt.rn{{.relu}}.f16x2.{at} exists before sm_100"
        )
    d, a = _ops(ctx, stmt, 2)
    la, ra = _u32_of16(ctx, a)
    lines += la
    lines += ctx.call(
        LIB_CVT_FP8X2_F16X2, [("b32", ra), ("b32", str(e5 | relu))], ("b32", r)
    )
    return [*lines, f"mov.b32 \t{d}, {r};"]


def _cvt_tf32(stmt: Stmt, mods: list[str], ctx: Ctx) -> list[str]:
    ms = set(mods)
    rnd = [m for m in mods if m in ("rn", "rz")]
    if len(rnd) != 1 or not ms <= {"rn", "rz", "relu"}:
        ctx.error(f"unsupported {stmt.opcode}")
    d, a = _ops(ctx, stmt, 2)
    la, ra = _reg(ctx, a, 32)
    r = ctx.new_reg("b32")
    ctrl = RND[rnd[0]] | (TF32_RELU if "relu" in ms else 0)
    return [
        *la,
        *ctx.call(LIB_CVT_TF32, [("b32", ra), ("b32", str(ctrl))], ("b32", r)),
        f"mov.b32 \t{d}, {r};",
    ]


def _fix16(ctx: Ctx, h: str, inf: int) -> tuple[list[str], str]:
    """b16 ``h`` with +-inf replaced by +-MAX_NORM (inf - 1 keeps the sign)."""
    m, hm, o = ctx.new_reg("b16"), ctx.new_reg("b16"), ctx.new_reg("b16")
    p = ctx.new_reg("pred")
    return [
        f"and.b16 \t{m}, {h}, 0x7FFF;",
        f"setp.eq.u16 \t{p}, {m}, 0x{inf:X};",
        f"sub.u16 \t{hm}, {h}, 1;",
        f"selp.b16 \t{o}, {hm}, {h}, {p};",
    ], o


def _cvt_satfinite(
    stmt: Stmt, dt: str, at: str, mods: list[str], ctx: Ctx
) -> list[str]:
    base = [m for m in mods if m != "satfinite"]
    rnd = [m for m in base if m in ("rn", "rz", "rna")]
    if len(rnd) != 1 or not set(base) <= {"rn", "rz", "rna", "relu"}:
        ctx.error(f"unsupported {stmt.opcode}")
    x2 = dt in ("f16x2", "bf16x2")
    ops = _ops(ctx, stmt, 3 if x2 else 2)
    d, srcs = ops[0], ops[1:]
    opc = ".".join(["cvt", *base, dt, at])
    if dt == "tf32":
        n = ctx.new_reg
        t, m, tm, x, xm, r = n("b32"), n("b32"), n("b32"), n("b32"), n("b32"), n("b32")
        p, pn = n("pred"), n("pred")
        # The native conversion can round a NaN whose payload is only in the low 13 bits to
        # the inf pattern (e2e misc.cvt 11-26: 0x7F800001 -> MAX_NORM), so NaN inputs are
        # classified on the source and give the canonical NaN.
        return [
            f"{opc} \t{t}, {srcs[0]};",
            f"and.b32 \t{m}, {t}, 0x7FFFFFFF;",
            f"setp.eq.u32 \t{p}, {m}, 0x7F800000;",
            f"sub.u32 \t{tm}, {t}, 0x2000;",
            f"selp.b32 \t{r}, {tm}, {t}, {p};",
            f"mov.b32 \t{x}, {srcs[0]};",
            f"and.b32 \t{xm}, {x}, 0x7FFFFFFF;",
            f"setp.gt.u32 \t{pn}, {xm}, 0x7F800000;",
            f"selp.b32 \t{d}, 0x7FFFFFFF, {r}, {pn};",
        ]
    inf = 0x7C00 if dt.startswith("f16") else 0x7F80
    if not x2:
        t = ctx.new_reg("b16")
        lines = [f"{opc} \t{t}, {srcs[0]};"]
        fl, o = _fix16(ctx, t, inf)
        return lines + fl + _put16_from16(ctx, d, o)
    t, lo, hi = ctx.new_reg("b32"), ctx.new_reg("b16"), ctx.new_reg("b16")
    lines = [f"{opc} \t{t}, {srcs[0]}, {srcs[1]};", f"mov.b32 \t{{{lo}, {hi}}}, {t};"]
    fl1, olo = _fix16(ctx, lo, inf)
    fl2, ohi = _fix16(ctx, hi, inf)
    return lines + fl1 + fl2 + [f"mov.b32 \t{d}, {{{olo}, {ohi}}};"]


def _cvt_to_bf16(stmt: Stmt, at: str, mods: list[str], ctx: Ctx) -> list[str]:
    if at not in BF16_SRC:
        ctx.error(f"unsupported {stmt.opcode}")
    ms = set(mods)
    rnd = [m for m in mods if m in _FRND]
    if not ms <= {"rn", "rz", "rm", "rp", "ftz", "sat"} or len(rnd) != 1:
        ctx.error(
            f"{stmt.opcode}: needs exactly one of .rn/.rz/.rm/.rp and only .ftz/.sat"
        )
    d, a = _ops(ctx, stmt, 2)
    ctrl = BF16_SRC[at] | (RND[rnd[0]] << BF16_RND_SHIFT)
    ctrl |= BF16_FTZ if "ftz" in ms else 0
    ctrl |= BF16_SAT if "sat" in ms else 0
    la, ra = _u64_arg(ctx, a, at)
    r = ctx.new_reg("b32")
    return (
        la
        + ctx.call(LIB_CVT_BF16, [("b64", ra), ("b32", str(ctrl))], ("b32", r))
        + _put16(ctx, d, r)
    )


def _ftz32(ctx: Ctx, f: str) -> tuple[list[str], str]:
    e, sg, o = ctx.new_reg("b32"), ctx.new_reg("b32"), ctx.new_reg("b32")
    z = ctx.new_reg("pred")
    return [
        f"and.b32 \t{e}, {f}, 0x7F800000;",
        f"setp.eq.u32 \t{z}, {e}, 0;",
        f"and.b32 \t{sg}, {f}, 0x80000000;",
        f"selp.b32 \t{o}, {sg}, {f}, {z};",
    ], o


def _cvt_from_bf16(stmt: Stmt, dt: str, mods: list[str], ctx: Ctx) -> list[str]:
    ms = set(mods)
    if not ms <= set(_IRND) | set(_FRND) | {"ftz", "sat"}:
        ctx.error(f"unsupported {stmt.opcode}")
    irnd = [m for m in mods if m in _IRND]
    frnd = [m for m in mods if m in _FRND]
    if len(irnd) + len(frnd) > 1:
        ctx.error(f"{stmt.opcode}: more than one rounding modifier")
    ftz = ".ftz" if "ftz" in ms else ""
    sat = ".sat" if "sat" in ms else ""
    d, a = _ops(ctx, stmt, 2)
    la, ra = _reg(ctx, a, 16)
    f = ctx.new_reg("f32")
    lines = [*la, f"cvt.f32.bf16 \t{f}, {ra};"]
    if dt in _INT_T:
        if not irnd:
            ctx.error(f"{stmt.opcode}: integer destination needs .rni/.rzi/.rmi/.rpi")
        return [*lines, f"cvt.{irnd[0]}{ftz}{sat}.{dt}.f32 \t{d}, {f};"]
    if dt == "f16":
        if not frnd:
            ctx.error(f"{stmt.opcode}: .f16 destination needs .rn/.rz/.rm/.rp")
        return [*lines, f"cvt.{frnd[0]}{ftz}{sat}.f16.f32 \t{d}, {f};"]
    if dt == "f64":
        if irnd:
            ctx.error(f"{stmt.opcode}: integer rounding with a float destination")
        return [
            *lines,
            f"cvt{ftz}{sat}.f64.f32 \t{d}, {f};",
        ]  # exact; .frnd has no effect
    if dt not in ("f32", "bf16"):
        ctx.error(f"unsupported {stmt.opcode}")
    if dt == "bf16" and not (irnd or ftz or sat):
        return [*la, f"mov.b16 \t{d}, {ra};"]
    g = f
    if irnd:
        g = ctx.new_reg("f32")
        lines.append(f"cvt.{irnd[0]}{ftz}{sat}.f32.f32 \t{g}, {f};")
    else:
        if ftz:
            fl, g = _ftz32(ctx, f)
            lines += fl
        if sat:
            g2 = ctx.new_reg("f32")
            lines.append(f"cvt.sat.f32.f32 \t{g2}, {g};")
            g = g2
    if dt == "f32":
        return [*lines, f"mov.b32 \t{d}, {g};"]
    return [*lines, f"cvt.rn.bf16.f32 \t{d}, {g};"]


# ---------------------------------------------------------------------------------------------
# Half precision and integer SIMD arithmetic
# ---------------------------------------------------------------------------------------------


def _w(ty: str) -> int:
    return 32 if ty.endswith("x2") or ty in ("f32", "u32", "s32", "b32") else 16


@_rule(lambda s, t: t[0] in ("add", "sub", "mul") and t[-1] in ("bf16", "bf16x2"))
def _rw_bf16_arith(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if not set(toks[1:-1]) <= {"rn"}:
        ctx.error(f"unsupported {stmt.opcode}")
    ty = toks[-1]
    w = _w(ty)
    d, a, b = _ops(ctx, stmt, 3)
    la, ra = _reg(ctx, a, w)
    lb, rb = _reg(ctx, b, w)
    k = ctx.new_reg(f"b{w}")
    rep = 2 if w == 32 else 1
    one, mone, mzero = (int(f"{v:04X}" * rep, 16) for v in (0x3F80, 0xBF80, 0x8000))
    op = toks[0]
    if op == "add":
        body = [f"mov.b{w} \t{k}, 0x{one:X};", f"fma.rn.{ty} \t{d}, {ra}, {k}, {rb};"]
    elif op == "sub":
        body = [f"mov.b{w} \t{k}, 0x{mone:X};", f"fma.rn.{ty} \t{d}, {rb}, {k}, {ra};"]
    else:
        body = [f"mov.b{w} \t{k}, 0x{mzero:X};", f"fma.rn.{ty} \t{d}, {ra}, {rb}, {k};"]
    return la + lb + body


def _oob16(ctx: Ctx, r: str, srcs: list[str]) -> tuple[list[str], str]:
    ps = [ctx.new_reg("pred") for _ in srcs]
    z, o = ctx.new_reg("b16"), ctx.new_reg("b16")
    lines = [
        f"setp.eq.b16 \t{p}, {s}, 0x{OOB_NAN16:X};"
        for p, s in zip(ps, srcs, strict=True)
    ]
    acc = ps[0]
    for p in ps[1:]:
        q = ctx.new_reg("pred")
        lines.append(f"or.pred \t{q}, {acc}, {p};")
        acc = q
    lines += [f"mov.b16 \t{z}, 0;", f"selp.b16 \t{o}, {z}, {r}, {acc};"]
    return lines, o


@_rule(lambda s, t: t[0] == "fma" and "oob" in t)
def _rw_fma_oob(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    ty = toks[-1]
    mods = set(toks[1:-1])
    if (
        ty not in ("f16", "f16x2", "bf16", "bf16x2")
        or not mods <= {"rn", "oob", "relu"}
        or "rn" not in mods
    ):
        ctx.error(f"unsupported {stmt.opcode}")
    w = _w(ty)
    d, a, b, c = _ops(ctx, stmt, 4)
    lines: list[str] = []
    regs = []
    for x in (a, b, c):
        lx, rx = _reg(ctx, x, w)
        lines += lx
        regs.append(rx)
    r = ctx.new_reg(f"b{w}")
    relu = ".relu" if "relu" in mods else ""
    lines.append(f"fma.rn{relu}.{ty} \t{r}, {regs[0]}, {regs[1]}, {regs[2]};")
    if w == 16:
        lo, o = _oob16(ctx, r, regs)
        return lines + lo + [f"mov.b16 \t{d}, {o};"]
    halves = []
    for x in (*regs, r):
        lo, hi = ctx.new_reg("b16"), ctx.new_reg("b16")
        lines.append(f"mov.b32 \t{{{lo}, {hi}}}, {x};")
        halves.append((lo, hi))
    outs = []
    for i in range(2):
        lo, o = _oob16(ctx, halves[3][i], [h[i] for h in halves[:3]])
        lines += lo
        outs.append(o)
    return [*lines, f"mov.b32 \t{d}, {{{outs[0]}, {outs[1]}}};"]


def _minmax_type(t: list[str]) -> str:
    return t[-2] if t[-1] == "relu" else t[-1]


def _minmax_match(s: Stmt, t: list[str]) -> bool:
    if t[0] not in ("min", "max") or len(t) < 2:
        return False
    ty = _minmax_type(t)
    return (
        "xorsign" in t
        or ty in ("u16x2", "s16x2", "u8x4", "s8x4")
        or (ty == "s32" and "relu" in t)
    )


def _xorsign16(
    ctx: Ctx, op: str, mods: str, ty: str, a: str, b: str
) -> tuple[list[str], str]:
    n = ctx.new_reg
    aa, bb, r, x, sg, rs, m, o = (n("b16") for _ in range(8))
    p = n("pred")
    nan = 0x7C00 if ty == "f16" else 0x7F80
    return [
        f"and.b16 \t{aa}, {a}, 0x7FFF;",
        f"and.b16 \t{bb}, {b}, 0x7FFF;",
        f"{op}{mods}.{ty} \t{r}, {aa}, {bb};",
        f"xor.b16 \t{x}, {a}, {b};",
        f"and.b16 \t{sg}, {x}, 0x8000;",
        f"or.b16 \t{rs}, {r}, {sg};",
        f"and.b16 \t{m}, {r}, 0x7FFF;",
        f"setp.gt.u16 \t{p}, {m}, 0x{nan:X};",
        f"selp.b16 \t{o}, {r}, {rs}, {p};",
    ], o


@_rule(_minmax_match)
def _rw_minmax(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    op = toks[0]
    ty = _minmax_type(toks)
    mods = [m for m in toks[1:] if m != ty]
    if "xorsign" in toks:
        if "abs" not in mods or not set(mods) <= {"ftz", "NaN", "xorsign", "abs"}:
            ctx.error(f"unsupported {stmt.opcode}")
        if "ftz" in mods and ty.startswith("bf16"):
            ctx.error(f"{stmt.opcode}: .ftz is not defined for bf16")
        keep = "".join(f".{m}" for m in ("ftz", "NaN") if m in mods)
        w = _w(ty)
        d, a, b = _ops(ctx, stmt, 3)
        la, ra = _reg(ctx, a, w)
        lb, rb = _reg(ctx, b, w)
        lines = la + lb
        if ty == "f32":
            n = ctx.new_reg
            aa, bb, r, x, sg, rs = (n("b32") for _ in range(6))
            p = n("pred")
            return lines + [
                f"and.b32 \t{aa}, {ra}, 0x7FFFFFFF;",
                f"and.b32 \t{bb}, {rb}, 0x7FFFFFFF;",
                f"{op}{keep}.f32 \t{r}, {aa}, {bb};",
                f"xor.b32 \t{x}, {ra}, {rb};",
                f"and.b32 \t{sg}, {x}, 0x80000000;",
                f"or.b32 \t{rs}, {r}, {sg};",
                f"testp.notanumber.f32 \t{p}, {r};",
                f"selp.b32 \t{d}, {r}, {rs}, {p};",
            ]
        if ty in ("f16", "bf16"):
            fl, o = _xorsign16(ctx, op, keep, ty, ra, rb)
            return lines + fl + [f"mov.b16 \t{d}, {o};"]
        if ty in ("f16x2", "bf16x2"):
            st = ty[:-2]
            al, ah, bl, bh = (ctx.new_reg("b16") for _ in range(4))
            lines += [
                f"mov.b32 \t{{{al}, {ah}}}, {ra};",
                f"mov.b32 \t{{{bl}, {bh}}}, {rb};",
            ]
            f1, o1 = _xorsign16(ctx, op, keep, st, al, bl)
            f2, o2 = _xorsign16(ctx, op, keep, st, ah, bh)
            return lines + f1 + f2 + [f"mov.b32 \t{d}, {{{o1}, {o2}}};"]
        ctx.error(f"unsupported {stmt.opcode}")
    if ty in ("u8x4", "s8x4"):
        ctx.error(f"{stmt.opcode} needs sm_100f/sm_107f or later; not emulated")
    relu = "relu" in mods
    if not set(mods) <= {"relu"} or (relu and ty == "u16x2"):
        ctx.error(f"unsupported {stmt.opcode}")
    d, a, b = _ops(ctx, stmt, 3)
    la, ra = _reg(ctx, a, 32)
    lb, rb = _reg(ctx, b, 32)
    lines = la + lb
    if ty == "s32":
        t = ctx.new_reg("b32")
        return lines + [f"{op}.s32 \t{t}, {ra}, {rb};", f"max.s32 \t{d}, {t}, 0;"]
    et = ty[:3]
    al, ah, bl, bh, lo, hi = (ctx.new_reg("b16") for _ in range(6))
    lines += [
        f"mov.b32 \t{{{al}, {ah}}}, {ra};",
        f"mov.b32 \t{{{bl}, {bh}}}, {rb};",
        f"{op}.{et} \t{lo}, {al}, {bl};",
        f"{op}.{et} \t{hi}, {ah}, {bh};",
    ]
    if relu:
        lo2, hi2 = ctx.new_reg("b16"), ctx.new_reg("b16")
        lines += [f"max.s16 \t{lo2}, {lo}, 0;", f"max.s16 \t{hi2}, {hi}, 0;"]
        lo, hi = lo2, hi2
    return [*lines, f"mov.b32 \t{d}, {{{lo}, {hi}}};"]


@_rule(lambda s, t: t[0] == "add" and t[-1] in ("u16x2", "s16x2", "u8x4", "s8x4"))
def _rw_add_simd(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if toks[-1] in ("u8x4", "s8x4") or "sat" in toks:
        ctx.error(f"{stmt.opcode} needs sm_100f/sm_107f or later; not emulated")
    if len(toks) != 2:
        ctx.error(f"unsupported {stmt.opcode}")
    d, a, b = _ops(ctx, stmt, 3)
    la, ra = _reg(ctx, a, 32)
    lb, rb = _reg(ctx, b, 32)
    al, bl, lo, lo2, ah, bh, hi = (ctx.new_reg("b32") for _ in range(7))
    return (
        la
        + lb
        + [
            f"and.b32 \t{al}, {ra}, 0xFFFF;",
            f"and.b32 \t{bl}, {rb}, 0xFFFF;",
            f"add.u32 \t{lo}, {al}, {bl};",
            f"and.b32 \t{lo2}, {lo}, 0xFFFF;",
            f"and.b32 \t{ah}, {ra}, 0xFFFF0000;",
            f"and.b32 \t{bh}, {rb}, 0xFFFF0000;",
            f"add.u32 \t{hi}, {ah}, {bh};",
            f"or.b32 \t{d}, {hi}, {lo2};",
        ]
    )


@_rule(lambda s, t: t[0] in ("tanh", "ex2") and t[-1] in ("bf16", "bf16x2"))
def _rw_bf16_sfu(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    want = ["approx"] if toks[0] == "tanh" else ["approx", "ftz"]
    if toks[1:-1] != want:
        ctx.error(f"unsupported {stmt.opcode}")
    f32op = "tanh.approx.f32" if toks[0] == "tanh" else "ex2.approx.ftz.f32"
    d, a = _ops(ctx, stmt, 2)
    if toks[-1] == "bf16":
        la, ra = _reg(ctx, a, 16)
        f, g = ctx.new_reg("f32"), ctx.new_reg("f32")
        return la + [
            f"cvt.f32.bf16 \t{f}, {ra};",
            f"{f32op} \t{g}, {f};",
            f"cvt.rn.bf16.f32 \t{d}, {g};",
        ]
    la, ra = _reg(ctx, a, 32)
    lo, hi = ctx.new_reg("b16"), ctx.new_reg("b16")
    fl, fh, gl, gh = (ctx.new_reg("f32") for _ in range(4))
    return la + [
        f"mov.b32 \t{{{lo}, {hi}}}, {ra};",
        f"cvt.f32.bf16 \t{fl}, {lo};",
        f"cvt.f32.bf16 \t{fh}, {hi};",
        f"{f32op} \t{gl}, {fl};",
        f"{f32op} \t{gh}, {fh};",
        f"cvt.rn.bf16x2.f32 \t{d}, {gh}, {gl};",
    ]


# ---------------------------------------------------------------------------------------------
# set / setp with bf16
# ---------------------------------------------------------------------------------------------

_BOOLOPS = ("and", "or", "xor")


def _bf16_f32(ctx: Ctx, h: str) -> tuple[list[str], str]:
    f = ctx.new_reg("f32")
    return [f"cvt.f32.bf16 \t{f}, {h};"], f


def _halves(ctx: Ctx, op: str) -> tuple[list[str], str, str]:
    lr, r = _reg(ctx, op, 32)
    lo, hi = ctx.new_reg("b16"), ctx.new_reg("b16")
    return [*lr, f"mov.b32 \t{{{lo}, {hi}}}, {r};"], lo, hi


def _bf16_cmp(ctx: Ctx, cmp: str, boolop: str, x2: bool, a: str, b: str, c: str | None):
    """Predicates of the bf16(x2) comparison (low half first)."""
    tail = f", {c}" if c is not None else ""
    bo = f".{boolop}" if boolop else ""
    lines: list[str] = []
    pairs = []
    if x2:
        la, al, ah = _halves(ctx, a)
        lb, bl, bh = _halves(ctx, b)
        lines += la + lb
        pairs = [(al, bl), (ah, bh)]
    else:
        la, ra = _reg(ctx, a, 16)
        lb, rb = _reg(ctx, b, 16)
        lines += la + lb
        pairs = [(ra, rb)]
    preds = []
    for x, y in pairs:
        l1, fx = _bf16_f32(ctx, x)
        l2, fy = _bf16_f32(ctx, y)
        p = ctx.new_reg("pred")
        lines += l1 + l2 + [f"setp.{cmp}{bo}.f32 \t{p}, {fx}, {fy}{tail};"]
        preds.append(p)
    return lines, preds


def _cmp_mods(ctx: Ctx, stmt: Stmt, mods: list[str]) -> tuple[str, str]:
    cmp = mods[0] if mods else ""
    rest = mods[1:]
    boolop = rest[0] if rest and rest[0] in _BOOLOPS else ""
    extra = rest[1:] if boolop else rest
    if not cmp or any(m != "ftz" for m in extra):
        ctx.error(f"unsupported {stmt.opcode}")
    return cmp, boolop


@_rule(lambda s, t: t[0] == "setp" and t[-1] in ("bf16", "bf16x2"))
def _rw_setp_bf16(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    cmp, boolop = _cmp_mods(ctx, stmt, toks[1:-1])
    if "ftz" in toks:
        ctx.error(f"{stmt.opcode}: .ftz is not defined for bf16")
    ops = [o.strip() for o in stmt.operands]
    if len(ops) != (4 if boolop else 3):
        ctx.error(f"{stmt.opcode}: wrong operand count")
    x2 = toks[-1] == "bf16x2"
    dsts = split_dests(ops[0])
    if len(dsts) != (2 if x2 else 1):
        ctx.error(f"{stmt.opcode}: destination must be {'p|q' if x2 else 'p'}")
    lines, preds = _bf16_cmp(
        ctx, cmp, boolop, x2, ops[1], ops[2], ops[3] if boolop else None
    )
    for dst, p in zip(dsts, preds, strict=True):
        if dst != "_":
            lines.append(f"mov.pred \t{dst}, {p};")
    return lines


_SET_TRUE = {
    "u32": 0xFFFFFFFF,
    "s32": 0xFFFFFFFF,
    "u16": 0xFFFF,
    "s16": 0xFFFF,
    "bf16": 0x3F80,
}
_SET_STYPES = (
    "b16",
    "b32",
    "b64",
    "u16",
    "u32",
    "u64",
    "s16",
    "s32",
    "s64",
    "f16",
    "f32",
    "f64",
)


@_rule(
    lambda s, t: (
        t[0] == "set" and len(t) >= 4 and ("bf16" in t[-2:] or "bf16x2" in t[-2:])
    )
)
def _rw_set_bf16(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    dt, st = toks[-2], toks[-1]
    cmp, boolop = _cmp_mods(ctx, stmt, toks[1:-2])
    ftz = ".ftz" if "ftz" in toks else ""
    ops = [o.strip() for o in stmt.operands]
    if len(ops) != (4 if boolop else 3):
        ctx.error(f"{stmt.opcode}: wrong operand count")
    d, a, b = ops[:3]
    c = ops[3] if boolop else None
    x2 = st == "bf16x2"
    if x2 and dt not in ("bf16x2", "u32", "s32"):
        ctx.error(f"unsupported {stmt.opcode}")
    if st == "bf16" and dt not in ("u16", "u32", "s16", "s32", "bf16"):
        ctx.error(f"unsupported {stmt.opcode}")
    if dt == "bf16" and st not in ("bf16", *_SET_STYPES):
        ctx.error(f"unsupported {stmt.opcode}")
    if st in ("bf16", "bf16x2"):
        if ftz:
            ctx.error(f"{stmt.opcode}: .ftz is not defined for bf16")
        lines, preds = _bf16_cmp(ctx, cmp, boolop, x2, a, b, c)
    else:
        p = ctx.new_reg("pred")
        bo = f".{boolop}" if boolop else ""
        tail = f", {c}" if c is not None else ""
        lines, preds = [f"setp.{cmp}{bo}{ftz}.{st} \t{p}, {a}, {b}{tail};"], [p]
    if x2:
        tv = 0x3F80 if dt == "bf16x2" else 0xFFFF
        lo, hi = ctx.new_reg("b32"), ctx.new_reg("b32")
        return lines + [
            f"selp.b32 \t{lo}, {tv}, 0, {preds[0]};",
            f"selp.b32 \t{hi}, {tv << 16}, 0, {preds[1]};",
            f"or.b32 \t{d}, {hi}, {lo};",
        ]
    if dt not in _SET_TRUE:
        ctx.error(f"unsupported {stmt.opcode}")
    bits = ctx.reg_bits(d)
    w = 32 if dt in ("u32", "s32") else 16
    if bits == 32 and w == 16:
        w = 32
    elif bits != w:
        ctx.error(f"{stmt.opcode}: destination {d} has width {bits}")
    return [*lines, f"selp.b{w} \t{d}, {_SET_TRUE[dt]}, 0, {preds[0]};"]


# ---------------------------------------------------------------------------------------------
# Memory instructions
# ---------------------------------------------------------------------------------------------


def _mem_rename_match(s: Stmt, t: list[str]) -> bool:
    if t[0] not in ("ld", "st", "cvta", "isspacep") or (
        len(t) > 1 and t[1] in ("async", "bulk")
    ):
        return False
    if "cluster" in t or "shared::cluster" in t:
        return True
    return t[0] == "ld" and (
        "proxy::readonly" in t
        or any(o.strip().endswith(".unified") for o in s.operands)
    )


@_rule(_mem_rename_match)
def _rw_mem_rename(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    out = []
    for t in toks:
        if t == "cluster":
            out.append("gpu")
        elif t == "shared::cluster":
            out.append("shared")
        elif t == "proxy::readonly":
            continue
        else:
            out.append(t)
    ops = [
        o.strip()[: -len(".unified")] if o.strip().endswith(".unified") else o.strip()
        for o in stmt.operands
    ]
    return _redispatch(stmt, ".".join(out), ops, ctx)


_SEMS = ("relaxed", "acquire", "release", "acq_rel")
_ATOM_OPS = ("and", "or", "xor", "add", "inc", "dec", "min", "max", "cas", "exch")
_SPACES = ("global", "shared", "shared::cta", "shared::cluster")
_VECS = ("v2", "v4", "v8")
_ESIZE = {
    "b16": 2,
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


def _atom_match(s: Stmt, t: list[str]) -> bool:
    if t[0] not in ("atom", "red") or (len(t) > 1 and t[1] == "async"):
        return False
    ts = set(t[1:])
    if ts & {"cluster", "shared::cluster", "v2", "v4", "v8", "b128"}:
        return True
    return "noftz" in ts and bool(ts & {"bf16", "bf16x2", "f32"})


def _parse_atom(ctx: Ctx, stmt: Stmt, toks: list[str]) -> dict:
    a: dict = {
        "kind": toks[0],
        "sem": None,
        "scope": None,
        "space": None,
        "op": None,
        "noftz": False,
        "hint": False,
        "vec": None,
        "type": None,
    }
    for t in toks[1:]:
        if t in _SEMS and a["sem"] is None:
            a["sem"] = t
        elif t in _SCOPES and a["scope"] is None:
            a["scope"] = "gpu" if t == "cluster" else t
        elif t in _SPACES and a["space"] is None:
            a["space"] = "shared" if t == "shared::cluster" else t
        elif t in _ATOM_OPS and a["op"] is None:
            a["op"] = t
        elif t == "noftz":
            a["noftz"] = True
        elif t == "L2::cache_hint":
            a["hint"] = True
        elif t in _VECS and a["vec"] is None:
            a["vec"] = int(t[1:])
        elif t in _ESIZE and a["type"] is None:
            a["type"] = t
        else:
            ctx.error(f"unsupported qualifier .{t} in {stmt.opcode}")
    if a["op"] is None or a["type"] is None:
        ctx.error(f"cannot parse {stmt.opcode}")
    return a


def _atom_head(
    a: dict, op: str, ty: str, noftz: bool, hint: bool, kind: str | None = None
) -> str:
    parts = [kind or a["kind"]]
    for k in ("sem", "scope", "space"):
        if a[k]:
            parts.append(a[k])
    parts.append(op)
    if noftz:
        parts.append("noftz")
    if hint:
        parts.append("L2::cache_hint")
    parts.append(ty)
    return ".".join(parts)


def _atom_scalar(
    ctx: Ctx, a: dict, d: str | None, addr: str, b: str, policy: str | None
) -> list[str]:
    """One scalar atom/red element: native where SM80 has it, else a CAS loop."""
    op, ty = a["op"], a["type"]
    half = ty in ("f16", "bf16", "f16x2", "bf16x2")
    shared = a["space"] in ("shared", "shared::cta")
    cas_loop = (
        (ty in ("bf16", "bf16x2") and op == "add")
        or (half and op in ("min", "max"))
        or (ty == "f32" and op == "add" and a["noftz"] and not shared)
    )
    if not cas_loop:
        noftz = a["noftz"] and ty in ("f16", "f16x2")
        if half and op == "add" and not noftz:
            ctx.error(f"{a['kind']}.add.{ty} requires .noftz")
        hint = a["hint"] and policy is not None
        head = _atom_head(a, op, ty, noftz, hint)
        tail = f", {policy}" if hint else ""
        if a["kind"] == "atom":
            return [f"{head} \t{d or '_'}, {addr}, {b}{tail};"]
        return [f"{head} \t{addr}, {b}{tail};"]
    w = _ESIZE[ty] * 8
    lb, rb = _reg(ctx, b, w)
    n = ctx.new_reg
    old, nv, prev = n(f"b{w}"), n(f"b{w}"), n(f"b{w}")
    p = n("pred")
    loop = ctx.new_label("atom_cas")
    sp = f".{a['space']}" if a["space"] else ""
    lines = [
        *lb,
        f"ld.relaxed.{a['scope'] or 'gpu'}{sp}.b{w} \t{old}, {addr};",
        f"{loop}:",
    ]
    if op == "add" and ty == "f32":
        lines.append(f"add.rn.f32 \t{nv}, {old}, {rb};")
    elif op == "add":
        k = n(f"b{w}")
        one = 0x3F803F80 if ty == "bf16x2" else 0x3F80
        lines += [
            f"mov.b{w} \t{k}, 0x{one:X};",
            f"fma.rn.{ty} \t{nv}, {old}, {k}, {rb};",
        ]
    else:
        lines.append(f"{op}.{ty} \t{nv}, {old}, {rb};")
    head = _atom_head(a, "cas", f"b{w}", False, False, kind="atom")
    lines += [
        f"{head} \t{prev}, {addr}, {old}, {nv};",
        f"setp.ne.b{w} \t{p}, {prev}, {old};",
        f"mov.b{w} \t{old}, {prev};",
        f"@{p} bra \t{loop};",
    ]
    if d and d != "_":
        lines.append(f"mov.b{w} \t{d}, {prev};")
    return lines


def _atom_b128(ctx: Ctx, a: dict, ops: list[str]) -> list[str]:
    if a["kind"] != "atom" or a["op"] not in ("cas", "exch") or a["vec"]:
        ctx.error(f"unsupported .b128 {a['kind']}.{a['op']}")
    gv = getattr(ctx, "global_var", None)
    if gv is None:
        ctx.error(
            "atom .b128 needs a module .global lock table (Ctx.global_var from T1)"
        )
    if a["scope"] == "sys":
        ctx.error(
            f"atom.sys.{a['op']}.b128: the lock table is device memory of one linked "
            "program; host and peer-device atomicity is not emulated; unsupported"
        )
    cas = a["op"] == "cas"
    if len(ops) != (4 if cas else 3):
        ctx.error(f"atom.{a['op']}.b128: wrong operand count")
    d, addr = ops[0], ops[1]
    locks = gv("b128_locks", B128_LOCK_BYTES, 16)
    la, ga = ctx.to_generic_u64(addr, a["space"])
    n = ctx.new_reg
    tl, gl = n("b64"), n("b64")
    lines = [*la, f"mov.u64 \t{tl}, {locks};", f"cvta.global.u64 \t{gl}, {tl};"]
    parts = []
    for x in ops[2:]:
        lo, hi = n("b64"), n("b64")
        lines.append(f"mov.b128 \t{{{lo}, {hi}}}, {x};")
        parts += [lo, hi]
    if not cas:
        parts = ["0", "0", *parts]
    sym, gp = _local_ptr(ctx, "misc_b128", 16)
    cfg = B128_CAS if cas else 0
    args = (
        [("b64", ga), ("b64", gl), ("b32", str(cfg))]
        + [("b64", x) for x in parts]
        + [("b64", gp)]
    )
    lines += ctx.call(LIB_ATOM_B128, args)
    rlo, rhi = n("b64"), n("b64")
    lines.append(f"ld.local.v2.b64 \t{{{rlo}, {rhi}}}, [{sym}];")
    if d != "_":
        lines.append(f"mov.b128 \t{d}, {{{rlo}, {rhi}}};")
    return lines


@_rule(_atom_match)
def _rw_atom(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    a = _parse_atom(ctx, stmt, toks)
    ops = [o.strip() for o in stmt.operands]
    if a["type"] == "b128":
        return _atom_b128(ctx, a, ops)
    if a["kind"] == "atom":
        d, rest = ops[0], ops[1:]
    else:
        d, rest = None, ops
    if a["op"] == "cas":
        if a["vec"] or len(rest) != 3:
            ctx.error(f"unsupported {stmt.opcode}")
        head = _atom_head(a, "cas", a["type"], False, False)
        return [f"{head} \t{d}, {rest[0]}, {rest[1]}, {rest[2]};"]
    policy = rest[2] if a["hint"] and len(rest) == 3 else None
    if len(rest) != (3 if policy else 2):
        ctx.error(f"{stmt.opcode}: wrong operand count")
    addr, b = rest[0], rest[1]
    if not a["vec"]:
        return _atom_scalar(ctx, a, d, addr, b, policy)
    if a["space"] not in (None, "global"):
        ctx.error(f"{stmt.opcode}: vector atom/red needs .global or generic addressing")
    bs = _vec(b)
    ds = _vec(d) if d is not None else [None] * len(bs)
    if len(bs) != a["vec"] or len(ds) != a["vec"]:
        ctx.error(f"{stmt.opcode}: vector operands must have {a['vec']} elements")
    lines: list[str] = []
    es = _ESIZE[a["type"]]
    for i in range(a["vec"]):
        lines += _atom_scalar(ctx, a, ds[i], _addr_plus(addr, i * es), bs[i], policy)
    return lines


_MM_SEMS = ("weak", "relaxed", "acquire", "release")
_MM_FP8 = ("e5m2", "e5m2x2", "e5m2x4", "e4m3", "e4m3x2", "e4m3x4")


@_rule(
    lambda s, t: (
        t[0] == "multimem" and len(t) > 1 and t[1] in ("ld_reduce", "st", "red")
    )
)
def _rw_multimem(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    kind = toks[1]
    sem = scope = space = op = ty = None
    vec = 1
    for t in toks[2:]:
        if t in _MM_SEMS and sem is None:
            sem = t
        elif t in _SCOPES and scope is None:
            scope = "gpu" if t == "cluster" else t
        elif t == "global" and space is None:
            space = t
        elif t in ("min", "max", "add", "and", "or", "xor") and op is None:
            op = t
        elif t == "acc::f32":
            continue  # one replica: no intermediate accumulation
        elif t in _VECS:
            vec = int(t[1:])
        elif t in _MM_FP8 or t == "acc::f16":
            ctx.error(f"{stmt.opcode}: .{t} needs sm_100 or later; not emulated")
        elif t in _ESIZE and ty is None:
            ty = t
        else:
            ctx.error(f"unsupported qualifier .{t} in {stmt.opcode}")
    if ty is None or (kind != "st" and op is None):
        ctx.error(f"cannot parse {stmt.opcode}")
    assert ty is not None
    ops = [o.strip() for o in stmt.operands]
    sp = f".{space}" if space else ""
    es = _ESIZE[ty]
    if vec * es > 16:
        ctx.error(f"{stmt.opcode}: .v{vec} of .{ty} is wider than 16 bytes")
    if kind == "red":
        if len(ops) != 2:
            ctx.error(f"{stmt.opcode}: wrong operand count")
        a = {
            "kind": "red",
            "sem": sem or "relaxed",
            "scope": scope or "sys",
            "space": space,
            "op": op,
            "noftz": False,
            "hint": False,
            "vec": None,
            "type": ty,
        }
        if ty in ("f16", "f16x2"):
            a["noftz"] = True
        elif ty == "f32" and op == "add":
            a["noftz"] = True  # [inference] no flush of subnormals
        elif ty == "b32" and op == "add":
            a["type"] = "u32"
        elif ty in ("b64", "s64") and op == "add":
            a["type"] = "u64"
        bs = _vec(ops[1])
        if len(bs) != vec:
            ctx.error(f"{stmt.opcode}: vector operand must have {vec} elements")
        lines: list[str] = []
        for i, b in enumerate(bs):
            lines += _atom_scalar(ctx, a, None, _addr_plus(ops[0], i * es), b, None)
        return lines
    if sem in (None, "weak"):
        head = ("ld" if kind == "ld_reduce" else "st") + sp
    else:
        head = f"{'ld' if kind == 'ld_reduce' else 'st'}.{sem}.{scope or 'sys'}{sp}"
    if kind == "ld_reduce":
        if len(ops) != 2:
            ctx.error(f"{stmt.opcode}: wrong operand count")
        regs, addr = _vec(ops[0]), ops[1]
    else:
        if len(ops) != 2:
            ctx.error(f"{stmt.opcode}: wrong operand count")
        addr, regs = ops[0], _vec(ops[1])
    if len(regs) != vec:
        ctx.error(f"{stmt.opcode}: vector operand must have {vec} elements")
    if es != 2:
        v = f".v{vec}" if vec > 1 else ""
        body = "{" + ", ".join(regs) + "}" if vec > 1 else regs[0]
        if kind == "ld_reduce":
            return [f"{head}{v}.b{es * 8} \t{body}, {addr};"]
        return [f"{head}{v}.b{es * 8} \t{addr}, {body};"]
    if vec == 1 or vec % 2:
        ctx.error(f"{stmt.opcode}: 16-bit elements need .v2/.v4/.v8")
    words = [ctx.new_reg("b32") for _ in range(vec // 2)]
    v = f".v{len(words)}" if len(words) > 1 else ""
    wbody = "{" + ", ".join(words) + "}" if len(words) > 1 else words[0]
    if kind == "ld_reduce":
        lines = [f"{head}{v}.b32 \t{wbody}, {addr};"]
        for i, w in enumerate(words):
            lines.append(f"mov.b32 \t{{{regs[2 * i]}, {regs[2 * i + 1]}}}, {w};")
        return lines
    lines = []
    for i, w in enumerate(words):
        lines.append(f"mov.b32 \t{w}, {{{regs[2 * i]}, {regs[2 * i + 1]}}};")
    return [*lines, f"{head}{v}.b32 \t{addr}, {wbody};"]


# multimem.cp.reduce.async.bulk .redOp / .type combinations (ptx.txt 15896-15921).
_MMCP_RED = {
    "add": {"u32", "s32", "u64", "f32", "f64", "f16", "bf16"},
    "min": {"u32", "s32", "u64", "s64", "f16", "bf16"},
    "max": {"u32", "s32", "u64", "s64", "f16", "bf16"},
    "inc": {"u32"},
    "dec": {"u32"},
    "and": {"b32", "b64"},
    "or": {"b32", "b64"},
    "xor": {"b32", "b64"},
}
_MMCP_TYPES = set().union(*_MMCP_RED.values())


def _delegate_async(
    stmt: Stmt, opcode: str, operands: list[str], ctx: Ctx
) -> list[str]:
    """Rewrite ``opcode operands`` (an sm_90 async-proxy form) with ``rules_async``."""
    from . import rules_async

    new = Stmt(
        None,
        opcode,
        operands,
        render_stmt(None, opcode, operands),
        stmt.lineno,
        stmt.in_asm,
    )
    if not rules_async.match(new):
        ctx.error(f"{stmt.opcode}: rules_async has no rule for {opcode}")
    return rules_async.rewrite(new, ctx)


@_rule(lambda s, t: t[:2] == ["multimem", "cp"])
def _rw_multimem_cp(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    # One replica (as every multimem.* rule): the multimem destination is the local global
    # address and the copy / reduction is the plain sm_90 bulk form, done by rules_async.
    reduce = toks[2:3] == ["reduce"]
    head = ["reduce", "async", "bulk"] if reduce else ["async", "bulk"]
    if toks[2 : 2 + len(head)] != head:
        ctx.error(f"unsupported {stmt.opcode}")
    mods = toks[2 + len(head) :]
    if "cp_mask" in mods:
        ctx.error(f"{stmt.opcode}: .cp_mask needs sm_100 or later; not emulated")
    sem = scope = comp = op = ty = None
    spaces: list[str] = []
    noftz = b128 = False
    for t in mods:
        if t in ("weak", "relaxed") and sem is None:
            sem = t
        elif t in _SCOPES and scope is None:
            scope = t
        elif t in ("global", "shared::cta"):
            spaces.append(t)
        elif t == "bulk_group" and comp is None:
            comp = t
        elif reduce and t in _MMCP_RED and op is None:
            op = t
        elif reduce and t == "noftz" and not noftz:
            noftz = True
        elif reduce and t in _MMCP_TYPES and ty is None:
            ty = t
        elif not reduce and t == "b128" and not b128:
            b128 = True
        else:
            ctx.error(f"unsupported qualifier .{t} in {stmt.opcode}")
    if spaces != ["global", "shared::cta"] or comp is None:
        ctx.error(f"{stmt.opcode}: expected .global.shared::cta.bulk_group")
    ops = [o.strip() for o in stmt.operands]
    if len(ops) != 3:
        ctx.error(f"{stmt.opcode}: expected [dstMem], [srcMem], size")
    if not reduce:
        # ptx.txt 15790-15802: .weak alone, or .relaxed.scope with .b128.
        if sem == "relaxed" and (scope is None or not b128):
            ctx.error(f"{stmt.opcode}: .relaxed needs .scope and .b128")
        if sem != "relaxed" and (scope is not None or b128):
            ctx.error(f"{stmt.opcode}: .scope / .b128 need .relaxed")
        q = [*([sem] if sem else []), *([scope] if scope else [])]
        q += ["global", "shared::cta", "bulk_group", *(["b128"] if b128 else [])]
        return _delegate_async(stmt, "cp.async.bulk." + ".".join(q), ops, ctx)
    if sem == "weak" or (sem is None) != (scope is None):
        ctx.error(f"{stmt.opcode}: .sem.scope must be .relaxed.<scope> or absent")
    if op is None or ty is None or ty not in _MMCP_RED[op]:
        ctx.error(f"{stmt.opcode}: invalid .redOp.type")
    assert op is not None and ty is not None
    if op == "add" and ty in ("f16", "bf16") and not noftz:
        ctx.error(f"{stmt.opcode}: .add.{ty} requires .noftz")
    if noftz and not (op == "add" and ty in ("f16", "bf16", "f32")):
        ctx.error(f"{stmt.opcode}: .noftz is only valid with .add.f16/.bf16/.f32")
    # ptx.txt 15893-15895: .add.f32 keeps subnormals with or without .noftz. Default
    # .sem.scope is .relaxed.sys (ptx.txt 15930).
    noftz = noftz or (op == "add" and ty == "f32")
    q = [sem or "relaxed", scope or "sys", "global", "shared::cta", "bulk_group", op]
    q += [*(["noftz"] if noftz else []), ty]
    return _delegate_async(stmt, "cp.reduce.async.bulk." + ".".join(q), ops, ctx)


@_rule(lambda s, t: t[0] == "prefetch" and ("valid_addr" in t or "L1::32B" in t))
def _rw_prefetch(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    # ptx.txt 13953: .L1::32B always comes with .valid_addr.
    if set(toks[1:]) - {"global"} != {"L1::32B", "valid_addr"}:
        ctx.error(f"unsupported {stmt.opcode}")
    sp = ".global" if "global" in toks else ""
    (a,) = _ops(ctx, stmt, 1)
    return [f"prefetch{sp}.L1 \t{a};"]


@_rule(lambda s, t: t[0] == "mapa")
def _rw_mapa(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    ty = toks[-1]
    mods = toks[1:-1]
    if ty not in ("u32", "u64") or not set(mods) <= {"shared::cluster"}:
        ctx.error(f"unsupported {stmt.opcode}")
    d, a, b = _ops(ctx, stmt, 3)
    lines: list[str] = []
    if _is_reg(b):
        # A rank other than 0 does not exist in a cluster of one CTA (PTX: undefined).
        bad = ctx.new_reg("pred")
        lines += [f"setp.ne.u32 \t{bad}, {b}, 0;", f"@{bad} trap;"]
    else:
        v = _imm(b, 32)
        if v is None or v != 0:
            ctx.error(f"{stmt.opcode}: CTA rank {b} outside a cluster of one CTA")
    w = 32 if ty == "u32" else 64
    if _is_reg(a):
        return [*lines, f"mov.b{w} \t{d}, {a};"]
    if mods:
        la, r = ctx.to_shared_u32(a, "shared::cluster")
        if w == 32:
            return [*lines, *la, f"mov.b32 \t{d}, {r};"]
        return [*lines, *la, f"cvt.u64.u32 \t{d}, {r};"]
    if w != 64:
        ctx.error(f"{stmt.opcode}: generic mapa needs .u64")
    la, g = ctx.to_generic_u64(a, None)
    return [*lines, *la, f"mov.b64 \t{d}, {g};"]


@_rule(lambda s, t: t[0] == "getctarank")
def _rw_getctarank(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if toks[-1] != "u32" or not set(toks[1:-1]) <= {"shared::cluster"}:
        ctx.error(f"unsupported {stmt.opcode}")
    d, _a = _ops(ctx, stmt, 2)
    return [f"mov.u32 \t{d}, 0;"]


# ---------------------------------------------------------------------------------------------
# elect.sync, setmaxnreg, griddepcontrol
# ---------------------------------------------------------------------------------------------


@_rule(lambda s, t: t[0] == "elect")
def _rw_elect(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if toks != ["elect", "sync"]:
        ctx.error(f"unsupported {stmt.opcode}")
    dst, mask = _ops(ctx, stmt, 2)
    parts = split_dests(dst)
    if len(parts) != 2:
        ctx.error("elect.sync destination must be d|p")
    d, p = parts
    n = ctx.new_reg
    one, bal, neg, lsb, ldr, ln = (n("b32") for _ in range(6))
    pt, pl = n("pred"), n("pred")
    lines = [
        f"mov.u32 \t{one}, 1;",
        f"setp.ne.u32 \t{pt}, {one}, 0;",
        f"vote.sync.ballot.b32 \t{bal}, {pt}, {mask};",
        f"neg.s32 \t{neg}, {bal};",
        f"and.b32 \t{lsb}, {bal}, {neg};",
        f"bfind.u32 \t{ldr}, {lsb};",
        f"mov.u32 \t{ln}, %laneid;",
        f"setp.eq.u32 \t{pl}, {ln}, {ldr};",
    ]
    if d != "_":
        lines.append(f"mov.u32 \t{d}, {ldr};")
    if p != "_":
        lines.append(f"mov.pred \t{p}, {pl};")
    return lines


@_rule(lambda s, t: t[0] in ("setmaxnreg", "griddepcontrol"))
def _rw_noop(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if toks[0] == "griddepcontrol" and toks[1:] not in (
        ["launch_dependents"],
        ["wait"],
    ):
        ctx.error(f"unsupported {stmt.opcode}")
    if toks[0] == "setmaxnreg" and (len(toks) < 2 or toks[1] not in ("inc", "dec")):
        ctx.error(f"unsupported {stmt.opcode}")
    return []


# ---------------------------------------------------------------------------------------------
# stmatrix / ldmatrix.m8n16
# ---------------------------------------------------------------------------------------------


def _lane_gq(ctx: Ctx) -> tuple[list[str], str, str]:
    ln, g, q = ctx.new_reg("b32"), ctx.new_reg("b32"), ctx.new_reg("b32")
    return (
        [
            f"mov.u32 \t{ln}, %laneid;",
            f"shr.u32 \t{g}, {ln}, 2;",
            f"and.b32 \t{q}, {ln}, 3;",
        ],
        g,
        q,
    )


def _row_addr(ctx: Ctx, sa: str, src: str) -> tuple[list[str], str]:
    ra = ctx.new_reg("b32")
    return [f"shfl.sync.idx.b32 \t{ra}, {sa}, {src}, 31, -1;"], ra


def _matrix_num(ctx: Ctx, stmt: Stmt, toks: list[str]) -> int:
    nums = [t for t in toks if t in ("x1", "x2", "x4")]
    if len(nums) != 1:
        ctx.error(f"{stmt.opcode}: missing .x1/.x2/.x4")
    return int(nums[0][1:])


def _matrix_space(toks: list[str]) -> str | None:
    return "shared" if ("shared" in toks or "shared::cta" in toks) else None


@_rule(lambda s, t: t[0] == "stmatrix")
def _rw_stmatrix(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if "m8n8" not in toks or toks[-1] != "b16":
        ctx.error(f"{stmt.opcode} (m16n8 / .b8) needs sm_100 or later; not emulated")
    known = {
        "stmatrix",
        "sync",
        "aligned",
        "m8n8",
        "x1",
        "x2",
        "x4",
        "trans",
        "shared",
        "shared::cta",
        "b16",
    }
    if not set(toks) <= known or not {"sync", "aligned"} <= set(toks):
        ctx.error(f"unsupported {stmt.opcode}")
    num = _matrix_num(ctx, stmt, toks)
    p, r = _ops(ctx, stmt, 2)
    regs = _vec(r)
    if len(regs) != num:
        ctx.error(f"{stmt.opcode}: {num} source registers expected")
    lines, sa = ctx.to_shared_u32(p, _matrix_space(toks))
    lg, g, q = _lane_gq(ctx)
    lines += lg
    n = ctx.new_reg
    if "trans" not in toks:
        q4 = n("b32")
        lines.append(f"shl.b32 \t{q4}, {q}, 2;")
        for i, ri in enumerate(regs):
            src, ad = n("b32"), n("b32")
            lines.append(f"add.u32 \t{src}, {g}, {8 * i};")
            lr, ra = _row_addr(ctx, sa, src)
            lines += lr + [
                f"add.u32 \t{ad}, {ra}, {q4};",
                f"st.shared.b32 \t[{ad}], {ri};",
            ]
        return lines
    q2, g2 = n("b32"), n("b32")
    lines += [f"shl.b32 \t{q2}, {q}, 1;", f"shl.b32 \t{g2}, {g}, 1;"]
    for i, ri in enumerate(regs):
        h = [n("b16"), n("b16")]
        lines.append(f"mov.b32 \t{{{h[0]}, {h[1]}}}, {ri};")
        for k in range(2):
            src, ad = n("b32"), n("b32")
            lines.append(f"add.u32 \t{src}, {q2}, {8 * i + k};")
            lr, ra = _row_addr(ctx, sa, src)
            lines += lr + [
                f"add.u32 \t{ad}, {ra}, {g2};",
                f"st.shared.b16 \t[{ad}], {h[k]};",
            ]
    return lines


@_rule(lambda s, t: t[0] == "ldmatrix" and ("m8n16" in t or "m16n16" in t))
def _rw_ldmatrix_s4(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if "m16n16" in toks or toks[-2:] != ["s8", "s4"]:
        ctx.error(f"{stmt.opcode} needs sm_100 or later; not emulated")
    known = {
        "ldmatrix",
        "sync",
        "aligned",
        "m8n16",
        "x1",
        "x2",
        "x4",
        "shared",
        "shared::cta",
        "s8",
        "s4",
    }
    if not set(toks) <= known or not {"sync", "aligned"} <= set(toks):
        ctx.error(f"unsupported {stmt.opcode}")
    num = _matrix_num(ctx, stmt, toks)
    r, p = _ops(ctx, stmt, 2)
    regs = _vec(r)
    if len(regs) != num:
        ctx.error(f"{stmt.opcode}: {num} destination registers expected")
    lines, sa = ctx.to_shared_u32(p, _matrix_space(toks))
    lg, g, q = _lane_gq(ctx)
    lines += lg
    n = ctx.new_reg
    q2 = n("b32")
    lines.append(f"shl.b32 \t{q2}, {q}, 1;")
    for i, ri in enumerate(regs):
        src, ad, v = n("b32"), n("b32"), n("b32")
        lines.append(f"add.u32 \t{src}, {g}, {8 * i};")
        lr, ra = _row_addr(ctx, sa, src)
        lines += lr + [f"add.u32 \t{ad}, {ra}, {q2};", f"ld.shared.u16 \t{v}, [{ad}];"]
        acc = n("b32")
        lines.append(f"mov.b32 \t{acc}, 0;")
        for k in range(4):
            t, nacc = n("b32"), n("b32")
            lines += [
                f"bfe.s32 \t{t}, {v}, {4 * k}, 4;",
                f"bfi.b32 \t{nacc}, {t}, {acc}, {8 * k}, 8;",
            ]
            acc = nacc
        lines.append(f"mov.b32 \t{ri}, {acc};")
    return lines


# ---------------------------------------------------------------------------------------------
# mma: fp8 (dense and sparse) and f64 m16n8k{4,8,16}
# ---------------------------------------------------------------------------------------------


def _mma_shape(toks: list[str]) -> tuple[int, int, int] | None:
    for t in toks:
        m = _SHAPE_RE.match(t)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _mma_match(s: Stmt, t: list[str]) -> bool:
    if t[0] != "mma" or len(t) < 2:
        return False
    if any(x in _FP8_ALL or x.startswith("kind::") or x == "block_scale" for x in t):
        return True
    return "f64" in t and _mma_shape(t) in ((16, 8, 4), (16, 8, 8), (16, 8, 16))


def _regs32(ctx: Ctx, op: str, n: int, what: str) -> tuple[list[str], list[str]]:
    els = _vec(op)
    if len(els) != n:
        ctx.error(f"mma: {what} needs {n} registers, got {len(els)}")
    lines: list[str] = []
    out = []
    for e in els:
        le, r = _reg(ctx, e, 32)
        lines += le
        out.append(r)
    return lines, out


def _regs64(ctx: Ctx, op: str, n: int, what: str) -> tuple[list[str], list[str]]:
    els = _vec(op)
    if len(els) != n:
        ctx.error(f"mma: {what} needs {n} registers, got {len(els)}")
    lines: list[str] = []
    out = []
    for e in els:
        le, r = _reg(ctx, e, 64)
        lines += le
        out.append(r)
    return lines, out


@_rule(_mma_match)
def _rw_mma(stmt: Stmt, toks: list[str], ctx: Ctx) -> list[str]:
    if any(
        x.startswith("kind::") or x == "block_scale" or x in ("e3m2", "e2m3", "e2m1")
        for x in toks
    ):
        ctx.error(f"{stmt.opcode} needs sm_120a or later; not emulated")
    shape = _mma_shape(toks)
    sparse = toks[1].startswith("sp")
    if not {"sync", "aligned", "row", "col"} <= set(toks) or shape is None:
        ctx.error(f"unsupported {stmt.opcode}")
    if "f64" in toks:
        return _mma_f64(stmt, toks, shape, ctx)
    return _mma_fp8(stmt, toks, shape, sparse, ctx)


def _mma_fp8(
    stmt: Stmt, toks: list[str], shape: tuple[int, int, int], sparse: bool, ctx: Ctx
) -> list[str]:
    dt, at, bt, ct = toks[-4:]
    if at not in _FP8 or bt not in _FP8 or dt != ct or dt not in ("f16", "f32"):
        ctx.error(f"unsupported {stmt.opcode}")
    expect = {
        "mma",
        "sync",
        "aligned",
        "row",
        "col",
        dt,
        at,
        bt,
        f"m{shape[0]}n{shape[1]}k{shape[2]}",
    }
    if sparse:
        expect.add(toks[1])
        if toks[1] not in ("sp", "sp::ordered_metadata"):
            ctx.error(f"unsupported {stmt.opcode}")
    if not set(toks) <= expect:
        ctx.error(f"unsupported qualifier in {stmt.opcode}")
    cfg = (MMA_A_E5M2 if at == "e5m2" else 0) | (MMA_B_E5M2 if bt == "e5m2" else 0)
    cfg |= MMA_ADA if _src_arch(ctx) == "ada" else 0
    f16 = dt == "f16"
    cfg |= MMA_D_F16 if f16 else 0
    nd = 2 if f16 else 4
    if sparse:
        if shape != (16, 8, 64) or f16:
            ctx.error(f"{stmt.opcode} needs sm_120a or later; not emulated")
        d, a, b, c, e, f = _ops(ctx, stmt, 6)
        if _imm(f, 32) != 0:
            ctx.error(f"{stmt.opcode}: sparsity selector must be 0 for 8-bit types")
        lines, ra = _regs32(ctx, a, 4, "A")
        lb, rb = _regs32(ctx, b, 4, "B")
        lc, rc = _regs32(ctx, c, 4, "C")
        le, re_ = _reg(ctx, e, 32)
        lines += lb + lc + le
        sym, gp = _local_ptr(ctx, "misc_mma", 64)
        args = [("b32", x) for x in (*ra, *rb, *rc, re_)] + [
            ("b64", gp),
            ("b32", str(cfg)),
        ]
        lines += ctx.call(LIB_MMA_SP_FP8, args)
    else:
        if shape not in ((16, 8, 16), (16, 8, 32)):
            ctx.error(f"unsupported {stmt.opcode}")
        k32 = shape[2] == 32
        cfg |= MMA_K32 if k32 else 0
        d, a, b, c = _ops(ctx, stmt, 4)
        lines, ra = _regs32(ctx, a, 4 if k32 else 2, "A")
        lb, rb = _regs32(ctx, b, 2 if k32 else 1, "B")
        lc, rc = _regs32(ctx, c, nd, "C")
        lines += lb + lc
        ra += ["0"] * (4 - len(ra))
        rb += ["0"] * (2 - len(rb))
        rc += ["0"] * (4 - len(rc))
        sym, gp = _local_ptr(ctx, "misc_mma", 64)
        args = [("b32", x) for x in (*ra, *rb, *rc)] + [("b64", gp), ("b32", str(cfg))]
        lines += ctx.call(LIB_MMA_FP8, args)
    ds = _vec(d)
    if len(ds) != nd:
        ctx.error(f"{stmt.opcode}: D needs {nd} registers")
    return [*lines, f"ld.local.v{nd}.b32 \t{{{', '.join(ds)}}}, [{sym}];"]


def _mma_f64(
    stmt: Stmt, toks: list[str], shape: tuple[int, int, int], ctx: Ctx
) -> list[str]:
    rnd = [t for t in toks if t in RND]
    base = {
        "mma",
        "sync",
        "aligned",
        "row",
        "col",
        "f64",
        f"m{shape[0]}n{shape[1]}k{shape[2]}",
    }
    if len(rnd) > 1 or not set(toks) <= base | set(RND) or toks.count("f64") != 4:
        ctx.error(f"unsupported {stmt.opcode}")
    k = shape[2]
    d, a, b, c = _ops(ctx, stmt, 4)
    lines, ra = _regs64(ctx, a, k // 2, "A")
    lb, rb = _regs64(ctx, b, k // 4, "B")
    lc, rc = _regs64(ctx, c, 4, "C")
    lines += lb + lc
    ra += ["0"] * (8 - len(ra))
    rb += ["0"] * (4 - len(rb))
    sym, gp = _local_ptr(ctx, "misc_mma", 64)
    cfg = F64_K[k] | (RND[rnd[0] if rnd else "rn"] << F64_RND_SHIFT)
    args = [("b64", x) for x in (*ra, *rb, *rc)] + [("b64", gp), ("b32", str(cfg))]
    lines += ctx.call(LIB_MMA_F64, args)
    ds = _vec(d)
    if len(ds) != 4:
        ctx.error(f"{stmt.opcode}: D needs 4 registers")
    return [
        *lines,
        f"ld.local.v2.b64 \t{{{ds[0]}, {ds[1]}}}, [{sym}];",
        f"ld.local.v2.b64 \t{{{ds[2]}, {ds[3]}}}, [{sym}+16];",
    ]
