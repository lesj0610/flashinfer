"""PTX translator core (PLAN section 5).

Parses a PTX module produced by cicc (or hand written), runs the rule modules over every
instruction, wraps guarded rewrites, declares the registers / ``.local`` buffers / module
``.shared`` variables the rules ask for, merges the emulation library PTX and emits PTX for
``.target sm_80``.

Rule module contract (``rules_wgmma`` -> ``rules_async`` -> ``rules_misc``, first match wins)::

    def scan(func: Func, ctx: Ctx) -> None                     # optional, once per function
    def function_prologue(ctx: Ctx) -> list[str]               # optional, called after the body
                                                               # was rewritten; lines go right
                                                               # after the body declarations
    def match(stmt: Stmt) -> bool
    def rewrite(stmt: Stmt, ctx: Ctx) -> list[str]            # replacement lines, no guard
    def rewrite_operand(op: str, stmt: Stmt, ctx: Ctx)         # optional; (lines, new_op) | None
        -> tuple[list[str], str] | None

``rewrite_operand`` is offered every top-level operand of every instruction (and each element
of a vector operand) before ``match``; the first rule returning non-None wins for that operand.
Its lines are placed before the statement (inside the guard wrapper when a rule then rewrites
the statement).
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from . import lexer as lx

DEFAULT_RULE_MODULES = ("ptxemu.rules_wgmma", "ptxemu.rules_async", "ptxemu.rules_misc")
MODES = ("exact", "fast", "check")
DST_TARGET = "sm_80"
LIB_ALLOWED_EXTERNS = frozenset({"vprintf"})
LIB_PUBLIC_PREFIX = "wgemu_"
LIB_RENAME_PREFIX = "__wgemulib_"
REG_PREFIX = "%wgemu_"
LOCAL_PREFIX = "__wgemu_local_"
SHARED_PREFIX = "__wgemu_shared_"
GLOBAL_PREFIX = "__wgemu_global_"
SKIP_LABEL_PREFIX = "$WGEMU_SKIP_"
LABEL_PREFIX = "$WGEMU_"

REG_TYPES = frozenset(
    {
        "pred",
        "b8",
        "b16",
        "b32",
        "b64",
        "b128",
        "u8",
        "u16",
        "u32",
        "u64",
        "s8",
        "s16",
        "s32",
        "s64",
        "f16",
        "f16x2",
        "bf16",
        "bf16x2",
        "f32",
        "f64",
    }
)

_CALL_TYPES = {
    "b32": "b32",
    "u32": "b32",
    "s32": "b32",
    "b64": "b64",
    "u64": "b64",
    "s64": "b64",
}
_SPACES = {
    None: None,
    "generic": None,
    "shared": "shared",
    "shared::cta": "shared",
    "shared::cluster": "shared",  # cluster size 1: the cluster window is the CTA window
    "global": "global",
    "local": "local",
    "const": "const",
    "param": "param",
    "param::entry": "param",
}


class TranslateError(Exception):
    """Translation failure. Raised for every unsupported or malformed input."""

    located = False


def _located(msg: str) -> TranslateError:
    e = TranslateError(msg)
    e.located = True
    return e


# ---------------------------------------------------------------------------------------------
# Statements and operand helpers
# ---------------------------------------------------------------------------------------------


@dataclass
class Stmt:
    guard: str | None  # '@%p1', '@!%p1' or None
    opcode: str  # full dotted opcode
    operands: list[str]  # top-level comma split; vectors / addresses stay one operand
    text: str  # original text without the trailing ';' (includes the guard)
    lineno: int
    in_asm: bool = False  # statement comes from an inline-asm expansion


_WS_RE = re.compile(r"\s+")
_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")
_GUARD_RE = re.compile(r"@!?%?[A-Za-z_$][\w$]*")
_OPCODE_RE = re.compile(r"[A-Za-z_][\w.:]*")
_INT_RE = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)U?\Z")
_ADDR_RE = re.compile(
    r"(?P<base>%?[A-Za-z_$][\w$]*)\s*(?:(?P<sign>[+-])\s*(?P<off>[+-]?\s*\w+))?\Z"
)
_LABEL_LINE_RE = re.compile(r"[A-Za-z_$][\w$]*\s*:\Z")


def parse_int(s: str) -> int:
    """Parse a PTX integer literal (decimal, 0x hex, 0b binary, 0 octal, optional U suffix)."""
    t = s.replace(" ", "")
    if not _INT_RE.match(t):
        raise TranslateError(f"not an integer literal: {s!r}")
    neg = t.startswith("-")
    t = t.lstrip("+-").rstrip("U")
    if t[:2] in ("0x", "0X"):
        v = int(t[2:], 16)
    elif t[:2] in ("0b", "0B"):
        v = int(t[2:], 2)
    elif len(t) > 1 and t[0] == "0":
        v = int(t, 8)
    else:
        v = int(t)
    return -v if neg else v


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split ``s`` at ``sep`` characters that are outside (), [], {} and strings."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            j = s.find('"', i + 1)
            while j > 0 and s[j - 1] == "\\":
                j = s.find('"', j + 1)
            if j < 0:
                raise TranslateError(f"unterminated string in {s!r}")
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                raise TranslateError(f"unbalanced brackets in {s!r}")
        elif c == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    if depth != 0:
        raise TranslateError(f"unbalanced brackets in {s!r}")
    parts.append(s[start:])
    return parts


def _norm_operand(op: str) -> str:
    s = _WS_RE.sub(" ", op.strip())
    s = re.sub(r"([\[({])\s+", r"\1", s)
    return re.sub(r"\s+([\])}])", r"\1", s)


def split_operands(rest: str) -> list[str]:
    """Split an operand list at top-level commas, normalising whitespace inside operands."""
    if not rest.strip():
        return []
    parts = _split_top(rest)
    out = []
    for p in parts:
        if not p.strip():
            raise TranslateError(f"empty operand in {rest.strip()!r}")
        out.append(_norm_operand(p))
    return out


def parse_vector(op: str) -> list[str]:
    """'{%f1, %f2}' -> ['%f1', '%f2']."""
    s = op.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise TranslateError(f"not a vector operand: {op!r}")
    inner = s[1:-1]
    if not inner.strip():
        return []
    elems = [_norm_operand(e) for e in _split_top(inner)]
    if any(not e for e in elems):
        raise TranslateError(f"empty element in vector operand {op!r}")
    return elems


def parse_address(op: str) -> tuple[str, int]:
    """'[%rd1+16]' -> ('%rd1', 16); '[sym]' -> ('sym', 0); '[%rd1+-8]' -> ('%rd1', -8).

    The brackets are optional. An absolute address '[64]' gives ('', 64). Tensor addresses
    ('[tmap, {...}]') are rejected; use :func:`parse_tensor_address`.
    """
    s = op.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise TranslateError(f"malformed address operand {op!r}")
        s = s[1:-1].strip()
    if not s or "," in s or "{" in s:
        raise TranslateError(f"not a plain address operand: {op!r}")
    m = _ADDR_RE.match(s)
    if m:
        off = 0
        if m.group("off") is not None:
            off = parse_int(m.group("off"))
            if m.group("sign") == "-":
                off = -off
        return m.group("base"), off
    try:
        return "", parse_int(s)
    except TranslateError:
        raise TranslateError(f"cannot parse address operand {op!r}") from None


def parse_tensor_address(op: str) -> tuple[str, list[str]]:
    """'[tmap, {%r1, %r2}]' -> ('tmap', ['%r1', '%r2'])."""
    s = op.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise TranslateError(f"not a tensor address operand: {op!r}")
    parts = _split_top(s[1:-1])
    if len(parts) != 2:
        raise TranslateError(f"tensor address must be [base, {{coords}}]: {op!r}")
    return _norm_operand(parts[0]), parse_vector(parts[1])


def split_dests(op: str) -> list[str]:
    """'%r1|%p1' -> ['%r1', '%p1']; an operand without '|' gives a one-element list."""
    parts = [p.strip() for p in op.split("|")]
    if any(not p for p in parts):
        raise TranslateError(f"malformed '|' destination operand {op!r}")
    return parts


def render_stmt(guard: str | None, opcode: str, operands: list[str]) -> str:
    """Statement text (without ';') from its parts."""
    head = f"{guard} {opcode}" if guard else opcode
    return f"{head} \t{', '.join(operands)}" if operands else head


def parse_stmt(text: str, lineno: int = 0, in_asm: bool = False) -> Stmt:
    """Parse one instruction statement (text without ';')."""
    s = text.strip()
    guard = None
    pos = 0
    m = _GUARD_RE.match(s)
    if m:
        guard = m.group(0)
        pos = m.end()
        while pos < len(s) and s[pos].isspace():
            pos += 1
    m = _OPCODE_RE.match(s, pos)
    if not m:
        raise TranslateError(f"line {lineno}: cannot parse instruction {s!r}")
    rest = s[m.end() :]
    if rest and not (rest[0].isspace() or rest[0] in "({["):
        raise TranslateError(f"line {lineno}: cannot parse opcode in {s!r}")
    try:
        operands = split_operands(rest)
    except TranslateError as e:
        raise TranslateError(f"line {lineno}: {e}") from None
    return Stmt(guard, m.group(0), operands, s, lineno, in_asm)


# ---------------------------------------------------------------------------------------------
# Module model
# ---------------------------------------------------------------------------------------------


@dataclass
class Func:
    name: str
    is_entry: bool
    params: list[
        tuple[str, str]
    ]  # (type text without '.param', name); array dims in the type
    body: list[Stmt | str]
    linkage: str = ""  # '.visible', '.weak', '.extern', '' ...
    rets: list[tuple[str, str]] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)  # e.g. ['.maxntid 128, 1, 1']
    header: str = ""  # header text (without '{'), as emitted
    lineno: int = 0


@dataclass
class ModItem:
    kind: str  # 'raw' | 'func' | 'func_decl' | 'var'
    text: str = ""
    name: str | None = None
    linkage: str = ""
    func: Func | None = None
    space: str | None = None
    start: int = 0
    end: int = 0
    lineno: int = 0


@dataclass
class Module:
    preamble: list[str]
    version: str | None
    target: str | None
    target_opts: list[str]
    address_size: str | None
    items: list[ModItem]
    filename: str = "<input>"


@dataclass
class _Header:
    linkage: str
    kind: str
    rets_raw: str | None
    name: str
    params_raw: str | None
    rest: str


_LINKAGE_RE = re.compile(
    r"\s*((?:\.(?:visible|extern|weak|common)\s+)*)\.(entry|func)(?![\w$])"
)
_VAR_RE = re.compile(
    r"\s*((?:\.(?:visible|extern|weak|common)\s+)*)"
    r"\.(global|shared|const|local|tex|surf|sampler|texref|surfref|samplerref)(?![\w$])"
)
_PARAM_RE = re.compile(
    r"\.(?:param|reg)\s+(?P<ty>.*?)\s*(?P<name>[A-Za-z_$%][\w$]*)\s*(?P<dims>(?:\[\s*\w*\s*\]\s*)*)\Z",
    re.S,
)
_DIR_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[^\s,;]+')
_CLUSTER_DIR_RE = re.compile(
    r"\.(reqnctapercluster)\s+(\d+(?:\s*,\s*\d+)*)"
    r"|\.(explicitcluster)(?![\w$])"
    r"|\.(maxclusterrank)\s+(\d+)"
    r"|\.(blocksareclusters)(?![\w$])"
)


def _skip_ws(s: str, pos: int) -> int:
    while pos < len(s) and s[pos].isspace():
        pos += 1
    return pos


def _match_close(s: str, pos: int) -> int:
    depth = 0
    for i in range(pos, len(s)):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
    raise TranslateError(f"unbalanced brackets in {s[:120]!r}")


def _parse_header(text: str) -> _Header:
    m = _LINKAGE_RE.match(text)
    if not m:
        raise TranslateError(f"not a function header: {text[:120]!r}")
    linkage = " ".join(m.group(1).split())
    kind = m.group(2)
    s = text
    pos = _skip_ws(s, m.end())
    rets = None
    if kind == "func" and pos < len(s) and s[pos] == "(":
        end = _match_close(s, pos)
        rets = s[pos + 1 : end]
        pos = _skip_ws(s, end + 1)
    m2 = _IDENT_RE.match(s, pos)
    if not m2:
        raise TranslateError(f"cannot find the function name in {text[:120]!r}")
    name = m2.group(0)
    pos = _skip_ws(s, m2.end())
    params = None
    if pos < len(s) and s[pos] == "(":
        end = _match_close(s, pos)
        params = s[pos + 1 : end]
        pos = end + 1
    return _Header(linkage, kind, rets, name, params, s[pos:])


def _parse_params(raw: str | None) -> list[tuple[str, str]]:
    if raw is None or not raw.strip():
        return []
    out = []
    for part in _split_top(raw):
        p = " ".join(part.split())
        m = _PARAM_RE.match(p)
        if not m:
            raise TranslateError(f"cannot parse parameter declaration {p!r}")
        ty = m.group("ty")
        dims = m.group("dims").replace(" ", "")
        out.append((f"{ty} {dims}".strip() if dims else ty, m.group("name")))
    return out


def _parse_directives(rest: str) -> list[str]:
    groups: list[list[str]] = []
    for tok in _DIR_TOKEN_RE.findall(rest):
        if tok.startswith("."):
            groups.append([tok])
        elif groups:
            groups[-1].append(tok)
        else:
            raise TranslateError(
                f"unexpected token {tok!r} after a function parameter list"
            )
    return [g[0] + (" " + ", ".join(g[1:]) if len(g) > 1 else "") for g in groups]


def _var_name(text: str) -> str:
    s = text.rstrip().rstrip(";")
    depth = 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "=" and depth == 0:
            s = s[:i]
            break
    s = re.sub(r"(\s*\[[^\]]*\])+\s*\Z", "", s.rstrip())
    s = re.sub(r"<\s*\d+\s*>\s*\Z", "", s)
    m = re.search(r"([A-Za-z_$][\w$]*)\s*\Z", s)
    if not m:
        raise TranslateError(f"cannot find the variable name in {text[:120]!r}")
    return m.group(1)


def _body_item(el: lx.Element) -> Stmt | str:
    k = el.kind
    if k == lx.STATEMENT:
        if el.text.startswith("."):
            return el.text + ";"
        return parse_stmt(el.text, el.lineno, el.in_asm)
    if k == lx.LABEL:
        return el.text + ":"
    if k == lx.LBRACE:
        return "{"
    if k == lx.RBRACE:
        return "}"
    if k in (lx.COMMENT, lx.ASM_BEGIN, lx.ASM_END, lx.LINE_DIRECTIVE):
        return el.text
    raise TranslateError(
        f"line {el.lineno}: unexpected {k} inside a function body: {el.text[:80]!r}"
    )


def _make_func(hdr_el: lx.Element, body_els: list[lx.Element]) -> Func:
    h = _parse_header(hdr_el.text)
    return Func(
        name=h.name,
        is_entry=h.kind == "entry",
        params=_parse_params(h.params_raw),
        body=[_body_item(el) for el in body_els],
        linkage=h.linkage,
        rets=_parse_params(h.rets_raw),
        directives=_parse_directives(h.rest),
        header=hdr_el.text,
        lineno=hdr_el.lineno,
    )


def _module_stmt_item(el: lx.Element) -> ModItem:
    text = el.text
    common = dict(start=el.start, end=el.end, lineno=el.lineno)
    if lx.is_header_text(text):
        h = _parse_header(text)
        return ModItem(
            "func_decl", text + ";", name=h.name, linkage=h.linkage, **common
        )
    m = _VAR_RE.match(text)
    if m:
        linkage = " ".join(m.group(1).split())
        return ModItem(
            "var",
            text + ";",
            name=_var_name(text),
            linkage=linkage,
            space=m.group(2),
            **common,
        )
    return ModItem("raw", text + ";", **common)


def parse_module(src: str, filename: str = "<input>") -> Module:
    """Parse PTX module text into a :class:`Module`."""
    try:
        els = lx.lex(src)
    except lx.LexError as e:
        raise _located(f"{filename}: {e}") from None
    preamble: list[str] = []
    version = target = address_size = None
    target_opts: list[str] = []
    items: list[ModItem] = []
    i = 0
    while i < len(els):
        el = els[i]
        k = el.kind
        try:
            if k in (lx.COMMENT, lx.ASM_BEGIN, lx.ASM_END):
                if version is None and not items:
                    preamble.append(el.text)
                else:
                    items.append(
                        ModItem(
                            "raw", el.text, start=el.start, end=el.end, lineno=el.lineno
                        )
                    )
            elif k == lx.LINE_DIRECTIVE:
                word = el.text.split(None, 1)[0]
                arg = el.text[len(word) :].strip()
                if word == ".version":
                    version = arg
                elif word == ".target":
                    parts = [p.strip() for p in arg.split(",")]
                    target, target_opts = parts[0], [p for p in parts[1:] if p]
                elif word == ".address_size":
                    address_size = arg
                else:
                    items.append(
                        ModItem(
                            "raw", el.text, start=el.start, end=el.end, lineno=el.lineno
                        )
                    )
            elif k == lx.BLOCK_DIRECTIVE:
                items.append(
                    ModItem(
                        "raw", el.text, start=el.start, end=el.end, lineno=el.lineno
                    )
                )
            elif k == lx.STATEMENT:
                items.append(_module_stmt_item(el))
            elif k == lx.HEADER:
                if i + 1 >= len(els) or els[i + 1].kind != lx.LBRACE:
                    raise TranslateError("function header without a body")
                j = i + 2
                depth = 1
                body: list[lx.Element] = []
                while j < len(els):
                    kj = els[j].kind
                    if kj == lx.LBRACE:
                        depth += 1
                    elif kj == lx.RBRACE:
                        depth -= 1
                        if depth == 0:
                            break
                    body.append(els[j])
                    j += 1
                else:
                    raise TranslateError("unterminated function body")
                func = _make_func(el, body)
                items.append(
                    ModItem(
                        "func",
                        name=func.name,
                        linkage=func.linkage,
                        func=func,
                        start=el.start,
                        end=els[j].end,
                        lineno=el.lineno,
                    )
                )
                i = j + 1
                continue
            else:
                raise TranslateError(
                    f"unexpected {k} at module scope: {el.text[:80]!r}"
                )
        except TranslateError as e:
            if getattr(e, "located", False):
                raise
            raise _located(f"{filename}:{el.lineno}: {e}") from None
        i += 1
    if version is None or target is None:
        raise _located(f"{filename}: missing .version or .target directive")
    return Module(preamble, version, target, target_opts, address_size, items, filename)


# ---------------------------------------------------------------------------------------------
# Register and symbol bookkeeping
# ---------------------------------------------------------------------------------------------


def _type_bits(ty: str | None) -> int | None:
    if ty is None:
        return None
    t = ty.lstrip(".")
    if t == "pred":
        return 1
    m = re.search(r"(\d+)(x2)?\Z", t)
    if not m:
        return None
    bits = int(m.group(1)) * (2 if m.group(2) else 1)
    v = re.match(r"v(\d+)\.", t)
    return bits * int(v.group(1)) if v else bits


class _Scope:
    __slots__ = ("names", "ranges", "vars")

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.ranges: list[tuple[str, int, str]] = []
        self.vars: dict[str, str] = {}

    def add_decl(self, text: str) -> None:
        s = text.strip().rstrip(";").strip()
        if s.startswith(".reg"):
            m = re.match(r"\.reg((?:\s+\.[A-Za-z_]\w*)+)\s+(.*)\Z", s, re.S)
            if not m:
                raise TranslateError(f"cannot parse register declaration {text!r}")
            toks = [t[1:] for t in m.group(1).split()]
            vec = [t for t in toks if re.fullmatch(r"v\d+", t)]
            ty = toks[-1] if not vec else f"{vec[0]}.{toks[-1]}"
            for name in _split_top(m.group(2)):
                name = name.strip()
                r = re.fullmatch(r"(%?[A-Za-z_$][\w$]*)\s*<\s*(\d+)\s*>", name)
                if r:
                    self.ranges.append((r.group(1), int(r.group(2)), ty))
                elif re.fullmatch(r"%?[A-Za-z_$][\w$]*", name):
                    self.names[name] = ty
                else:
                    raise TranslateError(
                        f"cannot parse register name {name!r} in {text!r}"
                    )
            return
        m = re.match(r"\.(local|shared|const|global|param)(?![\w$])", s)
        if m:
            self.vars[_var_name(s)] = m.group(1)

    def lookup(self, name: str) -> str | None:
        ty = self.names.get(name)
        if ty is not None:
            return ty
        for prefix, count, rty in self.ranges:
            if name.startswith(prefix):
                rest = name[len(prefix) :]
                if (
                    rest.isdigit()
                    and int(rest) < count
                    and (rest == "0" or not rest.startswith("0"))
                ):
                    return rty
        return None


class _ModuleCtx:
    """State shared by all functions of one translated module."""

    def __init__(
        self,
        mode: str = "exact",
        src_target: str = "sm_90a",
        trap: bool = False,
        filename: str = "<input>",
        allow_cluster: bool = False,
        warn_stream: Any = sys.stderr,
        async_delay: bool = False,
        count: bool = False,
        dl_slots: int = 0,
        dl_records: int = 0,
    ) -> None:
        self.mode = mode
        self.src_target = src_target
        self.trap = trap
        self.filename = filename
        self.allow_cluster = allow_cluster
        self.warn_stream = warn_stream
        # E1 delayed delivery / E2 counters (ptxemu.instrument); off: no instrumentation at all.
        self.async_delay = async_delay
        self.count = count
        self.dl_slots = dl_slots
        self.dl_records = dl_records
        self.counters = {"callseq": 0, "label": 0, "skip": 0}
        self.shared: dict[str, list[int]] = {}  # symbol -> [nbytes, align]
        self.globals: dict[str, list[int]] = {}  # symbol -> [nbytes, align]
        self.module_text: str | None = (
            None  # input PTX of the module, before translation
        )
        self.called: dict[str, tuple[tuple[str, ...], str | None]] = {}
        self.warnings: list[str] = []
        self.rule_states: dict[str, dict] = {}
        self.rewrites: dict[str, dict[str, int]] = {}
        self.operand_rewrites: dict[str, int] = {}
        self.vars: dict[str, str] = {}  # module-scope variable -> state space
        self.symbols: set[str] = set()  # every module-scope symbol of the input

    def next(self, counter: str) -> int:
        n = self.counters[counter]
        self.counters[counter] = n + 1
        return n


def _sanitize_key(key: str) -> str:
    k = re.sub(r"[^\w$]", "_", str(key))
    if not k:
        raise TranslateError("empty buffer key")
    return k


class Ctx:
    """Per-function translation context handed to rule modules."""

    def __init__(
        self,
        mode: str = "exact",
        src_target: str = "sm_90a",
        func: Func | None = None,
        module: _ModuleCtx | None = None,
        *,
        trap: bool = False,
    ) -> None:
        if mode not in MODES:
            raise TranslateError(f"unknown mode {mode!r} (expected one of {MODES})")
        self.mode = mode
        self.src_target = src_target
        self.func = func
        self.trap = trap
        self.stmt: Stmt | None = (
            None  # statement being rewritten (None in scan/prologue)
        )
        self._m = module if module is not None else _ModuleCtx(mode, src_target, trap)
        self._rule: str | None = None
        self._regs: dict[str, int] = {}
        self._locals: dict[str, list[int]] = {}  # symbol -> [nbytes, align]
        self._func_states: dict[str, dict] = {}
        self._rewrites: dict[str, int] = {}
        self._scopes: list[_Scope] = [_Scope()]
        self._flat = _Scope()
        # Cluster directives of an entry before _process_cluster removed them (keys
        # explicitcluster, reqnctapercluster, maxclusterrank, blocksareclusters; {} when the
        # entry has none). None for non-entry functions: their launch is not known here.
        self.cluster_info: dict[str, Any] | None = None
        if func is not None:
            for item in func.body:
                if isinstance(item, str) and item.startswith("."):
                    self._flat.add_decl(item)

    # -- PLAN section 5 API --------------------------------------------------------------------
    def new_reg(self, ty: str) -> str:
        """Fresh function-scope register of PTX type ``ty`` ('pred', 'b32', 'u64', 'f32', ...)."""
        t = ty.lstrip(".")
        if t not in REG_TYPES:
            self.error(f"new_reg: unsupported register type {ty!r}")
        n = self._regs.get(t, 0)
        self._regs[t] = n + 1
        return f"{REG_PREFIX}{t}_{n}"

    def local_buf(self, key: str, nbytes: int, align: int = 16) -> str:
        """Function-scope ``.local`` byte array, one per key (largest size/alignment wins)."""
        if nbytes <= 0 or align <= 0 or align & (align - 1):
            self.error(
                f"local_buf({key!r}): invalid size {nbytes} or alignment {align}"
            )
        sym = LOCAL_PREFIX + _sanitize_key(key)
        cur = self._locals.setdefault(sym, [nbytes, align])
        cur[0] = max(cur[0], nbytes)
        cur[1] = max(cur[1], align)
        return sym

    def shared_var(self, key: str, nbytes: int, align: int = 16) -> str:
        """Module-scope ``.shared`` byte array, one per key (largest size/alignment wins)."""
        if nbytes <= 0 or align <= 0 or align & (align - 1):
            self.error(
                f"shared_var({key!r}): invalid size {nbytes} or alignment {align}"
            )
        sym = SHARED_PREFIX + _sanitize_key(key)
        cur = self._m.shared.setdefault(sym, [nbytes, align])
        cur[0] = max(cur[0], nbytes)
        cur[1] = max(cur[1], align)
        return sym

    @property
    def module_text(self) -> str | None:
        """Input PTX text of the whole module (before translation)."""
        return self._m.module_text

    @property
    def async_delay(self) -> bool:
        """E1: asynchronous copies into shared memory are delivered late (ptxemu.instrument)."""
        return self._m.async_delay

    @property
    def count(self) -> bool:
        """E2: per-launch operation counters (ptxemu.instrument)."""
        return self._m.count

    @property
    def dl_shape(self) -> tuple[int, int]:
        """E1 pool shape: (slots, records per slot)."""
        return self._m.dl_slots, self._m.dl_records

    def global_var(self, key: str, nbytes: int, align: int = 16) -> str:
        """Module-scope zero-initialised ``.global`` byte array, one per key.

        Declared ``.weak`` so that -rdc modules that each declare it share one copy.
        """
        if nbytes <= 0 or align <= 0 or align & (align - 1):
            self.error(
                f"global_var({key!r}): invalid size {nbytes} or alignment {align}"
            )
        sym = GLOBAL_PREFIX + _sanitize_key(key)
        cur = self._m.globals.setdefault(sym, [nbytes, align])
        if cur != [nbytes, align]:
            self.error(
                f"global_var({key!r}): size/alignment {nbytes}/{align} differs from "
                f"{cur[0]}/{cur[1]} requested earlier"
            )
        return sym

    def generic_addr_of_local(self, sym: str) -> tuple[list[str], str]:
        """PTX lines and a b64 register holding the generic address of local symbol ``sym``."""
        a = self.new_reg("b64")
        g = self.new_reg("b64")
        return [f"mov.u64 \t{a}, {sym};", f"cvta.local.u64 \t{g}, {a};"], g

    def to_shared_u32(
        self, addr_operand: str, space: str | None
    ) -> tuple[list[str], str]:
        """32-bit shared-window address of an address operand in state space ``space``.

        ``space``: 'shared', 'shared::cta', 'shared::cluster' (identical with cluster size 1)
        or None for a generic address. Accepts '[reg+off]', '[sym+off]', 'reg', 'sym'.
        """
        sp = self._space(space)
        if sp not in (None, "shared"):
            self.error(f"to_shared_u32: state space {space!r} is not shared or generic")
        base, off = self._addr(addr_operand)
        lines: list[str] = []
        if base == "":
            if sp is None:
                self.error(f"to_shared_u32: absolute generic address {addr_operand!r}")
            r = self.new_reg("b32")
            return [f"mov.u32 \t{r}, {off};"], r
        if base.startswith("%"):
            bits = self.reg_bits(base)
            if sp == "shared":
                if bits == 32:
                    cur = base
                elif bits == 64:
                    cur = self.new_reg("b32")
                    lines.append(f"cvt.u32.u64 \t{cur}, {base};")
                else:
                    self.error(
                        f"to_shared_u32: register {base} has unknown or invalid width {bits}"
                    )
            else:
                if bits != 64:
                    self.error(
                        f"to_shared_u32: generic address register {base} is not 64-bit"
                    )
                t = self.new_reg("b64")
                cur = self.new_reg("b32")
                lines += [
                    f"cvta.to.shared.u64 \t{t}, {base};",
                    f"cvt.u32.u64 \t{cur}, {t};",
                ]
        else:
            symsp = self.symbol_space(base)
            if symsp != "shared":
                self.error(
                    f"to_shared_u32: symbol {base} is not a known .shared variable ({symsp})"
                )
            cur = self.new_reg("b32")
            lines.append(f"mov.u32 \t{cur}, {base};")
        if off:
            r = self.new_reg("b32")
            lines.append(f"add.s32 \t{r}, {cur}, {off};")
            cur = r
        return lines, cur

    def to_generic_u64(
        self, addr_operand: str, space: str | None
    ) -> tuple[list[str], str]:
        """64-bit generic address of an address operand in state space ``space``."""
        sp = self._space(space)
        base, off = self._addr(addr_operand)
        lines: list[str] = []
        if base == "":
            if sp not in (None, "global"):
                self.error(f"to_generic_u64: absolute {space} address {addr_operand!r}")
            r = self.new_reg("b64")
            return [f"mov.u64 \t{r}, {off};"], r
        if base.startswith("%"):
            bits = self.reg_bits(base)
            if sp is None:
                if bits != 64:
                    self.error(
                        f"to_generic_u64: generic address register {base} is not 64-bit"
                    )
                cur = base
            else:
                if bits == 32:
                    if sp not in ("shared", "local"):
                        self.error(
                            f"to_generic_u64: 32-bit {sp} address register {base}"
                        )
                    src = self.new_reg("b64")
                    lines.append(f"cvt.u64.u32 \t{src}, {base};")
                elif bits == 64:
                    src = base
                else:
                    self.error(
                        f"to_generic_u64: register {base} has unknown or invalid width {bits}"
                    )
                cur = self.new_reg("b64")
                lines.append(f"cvta.{sp}.u64 \t{cur}, {src};")
        else:
            symsp = self.symbol_space(base)
            if sp is not None and symsp is not None and symsp != sp:
                self.error(f"to_generic_u64: symbol {base} is in .{symsp}, not .{sp}")
            eff = sp or symsp
            if eff is None:
                self.error(f"to_generic_u64: unknown symbol {base}")
            t = self.new_reg("b64")
            cur = self.new_reg("b64")
            lines += [f"mov.u64 \t{t}, {base};", f"cvta.{eff}.u64 \t{cur}, {t};"]
        if off:
            r = self.new_reg("b64")
            lines.append(f"add.s64 \t{r}, {cur}, {off};")
            cur = r
        return lines, cur

    def call(
        self, func: str, args: list[tuple[str, str]], ret: tuple[str, str] | None = None
    ) -> list[str]:
        """nvcc-style callseq block calling library function ``func``.

        ``args``: (PTX type 'b32' | 'b64' (u/s aliases accepted), register or integer
        immediate). ``ret``: (type, destination register) or None.
        """
        if not isinstance(func, str) or not _IDENT_RE.fullmatch(func):
            self.error(f"call: invalid function name {func!r}")
        tys: list[str] = []
        vals: list[str] = []
        for a in args:
            if not (isinstance(a, (tuple, list)) and len(a) == 2):
                self.error(f"call {func}: argument {a!r} is not a (type, operand) pair")
            pty = _CALL_TYPES.get(str(a[0]).lstrip("."))
            if pty is None:
                self.error(f"call {func}: unsupported argument type {a[0]!r}")
            val = str(a[1]).strip()
            if not (
                re.fullmatch(r"%[A-Za-z_$][\w$]*", val)
                or re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|\d+)U?", val)
            ):
                self.error(
                    f"call {func}: argument {val!r} is neither a register nor an integer"
                )
            tys.append(pty)
            vals.append(val)
        rty = None
        rreg = ""
        if ret is not None:
            if not (isinstance(ret, (tuple, list)) and len(ret) == 2):
                self.error(f"call {func}: ret {ret!r} is not a (type, register) pair")
            rty = _CALL_TYPES.get(str(ret[0]).lstrip("."))
            rreg = str(ret[1]).strip()
            if rty is None or not re.fullmatch(r"%[A-Za-z_$][\w$]*", rreg):
                self.error(f"call {func}: invalid return {ret!r}")
        sig = (tuple(tys), rty)
        prev = self._m.called.setdefault(func, sig)
        if prev != sig:
            self.error(
                f"call {func}: signature {sig} differs from an earlier call {prev}"
            )
        n = self._m.next("callseq")
        lines = [f"{{ // callseq wgemu{n}, 0", ".reg .b32 temp_param_reg;"]
        for i, (pty, val) in enumerate(zip(tys, vals, strict=True)):
            lines.append(f".param .{pty} param{i};")
            lines.append(f"st.param.{pty} \t[param{i}+0], {val};")
        if rty is not None:
            lines.append(f".param .{rty} retval0;")
            lines.append("call.uni (retval0), ")
        else:
            lines.append("call.uni ")
        lines.append(f"{func}, ")
        lines.append("(")
        for i in range(len(tys)):
            lines.append(f"param{i}, " if i + 1 < len(tys) else f"param{i}")
        lines.append(");")
        if rty is not None:
            lines.append(f"ld.param.{rty} \t{rreg}, [retval0+0];")
        lines.append(f"}} // callseq wgemu{n}")
        return lines

    def func_state(self, owner: str | None = None) -> dict:
        """Per-function state dict private to the calling rule module, or to ``owner`` (a rule
        whose rewrite can also be reached through another rule, e.g. rules_misc delegating
        multimem.cp to rules_async, keeps its state under its own key)."""
        return self._func_states.setdefault(owner or self._rule or "", {})

    def warn(self, msg: str) -> None:
        full = f"{self._where()}warning: {msg}"
        self._m.warnings.append(full)
        if self._m.warn_stream is not None:
            print(f"ptxemu: {full}", file=self._m.warn_stream)

    def error(self, msg: str) -> None:
        raise _located(f"{self._where()}{msg}")

    # -- extensions (beyond PLAN section 5) ----------------------------------------------------
    def module_state(self) -> dict:
        """Module-wide state dict private to the calling rule module."""
        return self._m.rule_states.setdefault(self._rule or "", {})

    def new_label(self, hint: str = "L") -> str:
        """Module-unique label name ('$WGEMU_<hint>_<n>')."""
        return f"{LABEL_PREFIX}{_sanitize_key(hint)}_{self._m.next('label')}"

    def reg_type(self, name: str) -> str | None:
        """Declared type of register ``name`` ('b32', 'pred', 'v4.b32', ...) or None."""
        if name.startswith(REG_PREFIX):
            rest = name[len(REG_PREFIX) :]
            ty, _, idx = rest.rpartition("_")
            if ty in self._regs and idx.isdigit() and int(idx) < self._regs[ty]:
                return ty
        for scope in reversed(self._scopes):
            ty = scope.lookup(name)
            if ty is not None:
                return ty
        return self._flat.lookup(name)

    def reg_bits(self, name: str) -> int | None:
        """Bit width of register ``name`` (1 for predicates) or None if undeclared."""
        return _type_bits(self.reg_type(name))

    def symbol_space(self, name: str) -> str | None:
        """State space of a variable symbol visible in this function, or None."""
        for scope in reversed(self._scopes):
            if name in scope.vars:
                return scope.vars[name]
        if name in self._flat.vars:
            return self._flat.vars[name]
        if name in self._m.shared:
            return "shared"
        if name in self._m.vars:
            return self._m.vars[name]
        if self.func is not None and any(p[1] == name for p in self.func.params):
            return "param"
        return None

    # -- internals -------------------------------------------------------------------------------
    def _where(self) -> str:
        parts = [self._m.filename]
        if self.stmt is not None:
            parts.append(str(self.stmt.lineno))
        loc = ":".join(parts)
        fn = f" in {self.func.name}" if self.func is not None else ""
        rule = f" [{self._rule}]" if self._rule else ""
        return f"{loc}:{fn}{rule}: "

    def _space(self, space: str | None) -> str | None:
        key = space.lstrip(".") if isinstance(space, str) else space
        if key not in _SPACES:
            self.error(f"unknown state space {space!r}")
        return _SPACES[key]

    def _addr(self, op: str) -> tuple[str, int]:
        try:
            return parse_address(op)
        except TranslateError as e:
            self.error(str(e))
            raise  # unreachable

    def _track(self, item: str) -> None:
        if item == "{":
            self._scopes.append(_Scope())
        elif item == "}":
            if len(self._scopes) == 1:
                self.error("unbalanced '}' in function body")
            self._scopes.pop()
        elif item.startswith("."):
            self._scopes[-1].add_decl(item)

    def _decl_lines(self) -> list[str]:
        lines = [
            f".reg .{ty} \t{REG_PREFIX}{ty}_<{n}>;" for ty, n in self._regs.items()
        ]
        lines += [
            f".local .align {a} .b8 \t{sym}[{n}];"
            for sym, (n, a) in self._locals.items()
        ]
        return lines


# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------

RuleList = list[tuple[str, Any]]


def _validate_rule(mod: Any, name: str) -> None:
    for attr in ("match", "rewrite"):
        if not callable(getattr(mod, attr, None)):
            raise TranslateError(f"rule module {name} has no callable {attr}()")
    for attr in ("scan", "function_prologue", "rewrite_operand"):
        if hasattr(mod, attr) and not callable(getattr(mod, attr)):
            raise TranslateError(f"rule module {name}: {attr} is not callable")


def load_rules(
    names: Iterable[str] = DEFAULT_RULE_MODULES, warn: Any = None
) -> RuleList:
    """Import rule modules in order. A missing module is skipped with a warning; a module
    that exists but fails to import or lacks match()/rewrite() is an error."""
    out: RuleList = []
    for full in names:
        try:
            mod = importlib.import_module(full)
        except ModuleNotFoundError as e:
            if e.name == full:
                msg = f"rule module {full} not found; skipped"
                if warn is not None:
                    warn(msg)
                else:
                    print(f"ptxemu: warning: {msg}", file=sys.stderr)
                continue
            raise TranslateError(f"importing rule module {full} failed: {e}") from e
        _validate_rule(mod, full)
        out.append((full.rsplit(".", 1)[-1], mod))
    return out


def _normalize_rules(rules: Iterable[Any]) -> RuleList:
    out: RuleList = []
    for r in rules:
        if isinstance(r, tuple):
            name, mod = r
        else:
            name, mod = getattr(r, "__name__", "rule").rsplit(".", 1)[-1], r
        _validate_rule(mod, name)
        out.append((name, mod))
    return out


def _check_lines(lines: Any, where: str) -> list[str]:
    if lines is None or isinstance(lines, str) or not isinstance(lines, (list, tuple)):
        raise TranslateError(
            f"{where} must return a list of PTX lines, got {type(lines).__name__}"
        )
    out = []
    for ln in lines:
        if isinstance(ln, Stmt):
            out.append(ln.text + ";")
        elif isinstance(ln, str):
            out.append(ln)
        else:
            raise TranslateError(f"{where} returned a non-string line {ln!r}")
    return out


def _call_rule(ctx: Ctx, name: str, fn: Any, *args: Any) -> Any:
    ctx._rule = name
    try:
        return fn(*args)
    except TranslateError as e:
        if getattr(e, "located", False):
            raise
        raise _located(f"{ctx._where()}{e}") from None
    except Exception as e:  # a crashing rule is a translation failure, never ignored
        raise _located(f"{ctx._where()}rule {name}.{fn.__name__} raised {e!r}") from e
    finally:
        ctx._rule = None


def _one_line(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _rewrite_operand(
    op: str, stmt: Stmt, ctx: Ctx, op_rules: RuleList
) -> tuple[list[str], str] | None:
    for name, mod in op_rules:
        res = _call_rule(ctx, name, mod.rewrite_operand, op, stmt, ctx)
        if res is None:
            continue
        if not (isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], str)):
            raise _located(
                f"{ctx._where()}{name}.rewrite_operand must return (lines, operand) or None"
            )
        lines = _check_lines(res[0], f"{name}.rewrite_operand")
        ctx._m.operand_rewrites[op] = ctx._m.operand_rewrites.get(op, 0) + 1
        return lines, res[1]
    return None


def _translate_stmt(stmt: Stmt, rules: RuleList, ctx: Ctx) -> list[Stmt | str]:
    pre: list[str] = []
    op_rules = [(n, m) for n, m in rules if hasattr(m, "rewrite_operand")]
    if op_rules and stmt.operands:
        ops = list(stmt.operands)
        changed = False
        for idx, op in enumerate(ops):
            res = _rewrite_operand(op, stmt, ctx, op_rules)
            if res is not None:
                pre.extend(res[0])
                ops[idx] = res[1]
                changed = True
                continue
            if op.startswith("{") and op.endswith("}"):
                elems = parse_vector(op)
                ech = False
                for k, el in enumerate(elems):
                    r2 = _rewrite_operand(el, stmt, ctx, op_rules)
                    if r2 is not None:
                        pre.extend(r2[0])
                        elems[k] = r2[1]
                        ech = True
                if ech:
                    ops[idx] = "{" + ", ".join(elems) + "}"
                    changed = True
        if changed:
            stmt = replace(
                stmt, operands=ops, text=render_stmt(stmt.guard, stmt.opcode, ops)
            )
            ctx.stmt = stmt
    for name, mod in rules:
        if not _call_rule(ctx, name, mod.match, stmt):
            continue
        lines = _check_lines(
            _call_rule(ctx, name, mod.rewrite, stmt, ctx), f"{name}.rewrite"
        )
        counts = ctx._m.rewrites.setdefault(name, {})
        counts[stmt.opcode] = counts.get(stmt.opcode, 0) + 1
        ctx._rewrites[stmt.opcode] = ctx._rewrites.get(stmt.opcode, 0) + 1
        body = pre + lines
        if stmt.guard:
            g = stmt.guard
            neg = "@" + g[2:] if g.startswith("@!") else "@!" + g[1:]
            lab = f"{SKIP_LABEL_PREFIX}{ctx._m.next('skip')}"
            body = [f"{neg} bra \t{lab};", *body, f"{lab}:"]
        return [f"// wgemu {name}: {_one_line(stmt.text)}", *body]
    return [*pre, stmt]


def _decl_end(body: list[Stmt | str]) -> int:
    i = 0
    while i < len(body):
        it = body[i]
        if isinstance(it, str):
            if it.startswith((".reg", ".local", ".shared", ".const", ".param")):
                i += 1
                continue
            if it.startswith(("//", "/*")) and not (
                lx._ASM_BEGIN_RE.match(it) or lx._ASM_END_RE.match(it)
            ):
                i += 1
                continue
        break
    return i


def _process_cluster(func: Func, ctx: Ctx, finfo: dict) -> None:
    hdr = _parse_header(func.header)
    found: dict[str, Any] = {}

    def sub(m: re.Match) -> str:
        if m.group(1):
            found["reqnctapercluster"] = [
                int(x) for x in re.split(r"\s*,\s*", m.group(2).strip())
            ]
        elif m.group(3):
            found["explicitcluster"] = True
        elif m.group(4):
            found["maxclusterrank"] = int(m.group(5))
        else:
            found["blocksareclusters"] = True
        return ""

    new_rest = _CLUSTER_DIR_RE.sub(sub, hdr.rest)
    if not found:
        return
    if not func.is_entry:
        ctx.error("cluster directives on a non-entry function")
    dims = found.get("reqnctapercluster") or [1]
    size = 1
    for d in dims:
        size *= d
    if size > 1:
        msg = f".reqnctapercluster {dims} (cluster of {size} CTAs) is not supported before phase 2"
        if ctx._m.allow_cluster:
            ctx.warn(msg + "; directive removed because --allow-cluster was given")
        else:
            ctx.error(msg)
    if found.get("maxclusterrank", 1) > 1:
        ctx.warn(
            f".maxclusterrank {found['maxclusterrank']} removed; launches run with cluster 1x1x1"
        )
    new_rest = "\n".join(ln for ln in new_rest.split("\n") if ln.strip())
    func.header = func.header[: len(func.header) - len(hdr.rest)].rstrip() + (
        "\n" + new_rest if new_rest else ""
    )
    func.directives = _parse_directives(new_rest)
    finfo["cluster"] = found


_UNIFIED_ATTR_RE = re.compile(r"\.attribute\s*\(\s*\.unified\s*\([^()]*\)\s*\)\s*")


def _translate_func(func: Func, rules: RuleList, mctx: _ModuleCtx) -> dict:
    ctx = Ctx(mctx.mode, mctx.src_target, func, mctx, trap=mctx.trap)
    finfo: dict[str, Any] = {"name": func.name, "is_entry": func.is_entry}
    _process_cluster(func, ctx, finfo)
    ctx.cluster_info = dict(finfo.get("cluster", {})) if func.is_entry else None
    for name, mod in rules:
        if hasattr(mod, "scan"):
            _call_rule(ctx, name, mod.scan, func, ctx)
    # E1/E2: lines inserted before a statement whatever rule rewrites it (counters, E1 exit hook).
    before = None
    if mctx.async_delay or mctx.count:
        before = next((m for n, m in rules if n == "instrument"), None)
    new_body: list[Stmt | str] = []
    for item in func.body:
        if isinstance(item, Stmt):
            ctx.stmt = item
            if before is not None:
                res = _call_rule(ctx, "instrument", before.before, item, ctx)
                new_body.extend(_check_lines(res, "instrument.before"))
            new_body.extend(_translate_stmt(item, rules, ctx))
            ctx.stmt = None
        else:
            ctx._track(item)
            new_body.append(item)
    if len(ctx._scopes) != 1:
        ctx.error("unbalanced '{' in function body")
    prologue: list[str] = []
    for name, mod in rules:
        if hasattr(mod, "function_prologue"):
            res = _call_rule(ctx, name, mod.function_prologue, ctx)
            prologue.extend(_check_lines(res, f"{name}.function_prologue"))
    insert = ctx._decl_lines() + prologue
    if insert:
        idx = _decl_end(new_body)
        new_body = new_body[:idx] + insert + new_body[idx:]
    func.body = new_body
    finfo["rewrites"] = dict(ctx._rewrites)
    finfo["new_regs"] = dict(ctx._regs)
    finfo["local_bufs"] = {
        s: {"bytes": n, "align": a} for s, (n, a) in ctx._locals.items()
    }
    return finfo


# ---------------------------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------------------------


def _indent(s: str) -> str:
    t = s.strip("\n")
    if _LABEL_LINE_RE.match(t.strip()):
        return t.strip()
    return t if t.startswith("\t") else "\t" + t.lstrip(" ")


def emit_func(func: Func) -> list[str]:
    lines = [func.header, "{"]
    for it in func.body:
        lines.append("\t" + it.text + ";" if isinstance(it, Stmt) else _indent(it))
    lines.append("}")
    return lines


def _version_tuple(v: str | None) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", v or "")
    if not m:
        raise TranslateError(f"bad .version {v!r}")
    return int(m.group(1)), int(m.group(2))


def _map_target(target: str) -> str:
    m = re.fullmatch(r"sm_(\d+)([a-z]?)", target)
    if not m:
        raise TranslateError(f"unsupported .target {target!r}")
    if int(m.group(1)) > 90:
        raise TranslateError(
            f".target {target} is newer than sm_90a; only sm_8x/sm_90/sm_90a sources are supported"
        )
    return DST_TARGET


# ---------------------------------------------------------------------------------------------
# Library merge
# ---------------------------------------------------------------------------------------------

_LIB_CACHE: dict[int, tuple[str, Module]] = {}
_INTERNALIZE_RE = re.compile(r"\A(\s*)((?:\.(?:visible|weak|common)\s+)+)")


def _parse_lib(lib_src: str) -> Module:
    key = hash(lib_src)
    hit = _LIB_CACHE.get(key)
    if hit is not None and hit[0] == lib_src:
        return hit[1]
    mod = parse_module(lib_src, "<emu_lib.ptx>")
    _LIB_CACHE.clear()
    _LIB_CACHE[key] = (lib_src, mod)
    return mod


def _param_widths(params: list[tuple[str, str]]) -> tuple[int | None, ...]:
    out = []
    for ty, _ in params:
        m = re.fullmatch(r"\.[bsuf](\d+)", ty.strip())
        out.append(int(m.group(1)) if m else None)
    return tuple(out)


def _norm_sig(text: str) -> tuple:
    h = _parse_header(text)
    return (
        h.kind,
        tuple(" ".join(t.split()) for t, _ in _parse_params(h.rets_raw)),
        tuple(" ".join(t.split()) for t, _ in _parse_params(h.params_raw)),
    )


def _rename_ids(text: str, renames: dict[str, str]) -> str:
    if not renames:
        return text
    pat = re.compile(
        r"(?<![\w$%.])("
        + "|".join(re.escape(k) for k in sorted(renames, key=len, reverse=True))
        + r")(?![\w$])"
    )
    return pat.sub(lambda m: renames[m.group(1)], text)


def _item_text(src: str, item: ModItem) -> str:
    if item.kind == "func":
        return src[item.start : item.end]
    return item.text


def _merge_library(
    user: Module, user_text: str, lib_src: str, mctx: _ModuleCtx, info: dict
) -> tuple[list[str], set[int], str]:
    """Returns (library lines, indices of user items to drop, library .version)."""
    lib = _parse_lib(lib_src)
    lib_defs: dict[str, list[ModItem]] = {}
    lib_externs: dict[str, ModItem] = {}
    order: list[ModItem] = []
    for item in lib.items:
        if item.kind == "raw":
            t = item.text.lstrip()
            if t.startswith(("//", "/*")):
                continue
            raise TranslateError(
                f"emulation library: unsupported module item {t[:60]!r} (build it without -G/-lineinfo)"
            )
        assert item.name is not None
        if ".extern" in item.linkage.split():
            lib_externs.setdefault(item.name, item)
        else:
            lib_defs.setdefault(item.name, []).append(item)
        order.append(item)

    user_defs: dict[str, int] = {}
    user_externs: dict[str, int] = {}
    for idx, item in enumerate(user.items):
        if item.name is None:
            continue
        if ".extern" in item.linkage.split():
            user_externs.setdefault(item.name, idx)
        else:
            user_defs.setdefault(item.name, idx)
    taken = (
        set(user_defs)
        | set(user_externs)
        | mctx.symbols
        | set(mctx.shared)
        | set(mctx.globals)
    )

    renames: dict[str, str] = {}
    for name in lib_defs:
        if name.startswith(LIB_PUBLIC_PREFIX):
            if name in user_defs:
                raise TranslateError(
                    f"the input module defines {name}, which the emulation library provides"
                )
            continue
        if name in taken:
            new = LIB_RENAME_PREFIX + name
            while new in taken or new in lib_defs or new in lib_externs:
                new += "_"
            renames[name] = new
    for name in lib_externs:
        if name in user_defs:
            raise TranslateError(
                f"the input module defines {name}, which the emulation library declares .extern"
            )

    texts: dict[int, str] = {}
    final: dict[str, list[ModItem]] = {}
    for item in order:
        t = _rename_ids(_item_text(lib_src, item), renames)
        if ".extern" not in item.linkage.split():
            t = _INTERNALIZE_RE.sub(r"\1", t)
        texts[id(item)] = t
        assert item.name is not None
        final.setdefault(renames.get(item.name, item.name), []).append(item)

    names = set(final)
    refs: dict[str, set[str]] = {}
    for fname, items in final.items():
        ids: set[str] = set()
        for it in items:
            ids.update(_IDENT_RE.findall(texts[id(it)]))
        ids.discard(fname)
        refs[fname] = ids & names
    reach: set[str] = set()
    todo = list(set(_IDENT_RE.findall(user_text)) & names)
    while todo:
        n = todo.pop()
        if n in reach:
            continue
        reach.add(n)
        todo.extend(refs[n] - reach)

    for fname, (tys, rty) in mctx.called.items():
        defs = [it for it in lib_defs.get(fname, []) if it.kind == "func"]
        if not defs:
            raise TranslateError(f"the emulation library has no definition of {fname}")
        f = defs[0].func
        assert f is not None
        want = tuple(32 if t == "b32" else 64 for t in tys)
        have = _param_widths(f.params)
        rwant = None if rty is None else (32 if rty == "b32" else 64)
        rhave = _param_widths(f.rets)
        if have != want or (rhave != (() if rwant is None else (rwant,))):
            raise TranslateError(
                f"call signature of {fname} (args {want}, ret {rwant}) does not match the library "
                f"definition (args {have}, ret {rhave})"
            )

    dropped: set[int] = set()
    for name in sorted(reach):
        if name in lib_externs:
            if name not in LIB_ALLOWED_EXTERNS:
                raise TranslateError(
                    f"emulation library references external symbol {name}; only {sorted(LIB_ALLOWED_EXTERNS)} are allowed"
                )
            if name in user_externs:
                uitem = user.items[user_externs[name]]
                if uitem.kind != lib_externs[name].kind or (
                    uitem.kind == "func_decl"
                    and _norm_sig(uitem.text) != _norm_sig(lib_externs[name].text)
                ):
                    raise TranslateError(
                        f"declaration of {name} differs between the input and the library"
                    )
                dropped.add(user_externs[name])
        elif name.startswith(LIB_PUBLIC_PREFIX) and name in user_externs:
            uitem = user.items[user_externs[name]]
            fdef = next((it.func for it in lib_defs[name] if it.kind == "func"), None)
            if uitem.kind != "func_decl" or fdef is None:
                raise TranslateError(
                    f"input declares {name} but the library does not define it as a function"
                )
            uh = _parse_header(uitem.text)
            if _param_widths(_parse_params(uh.params_raw)) != _param_widths(
                fdef.params
            ) or _param_widths(_parse_params(uh.rets_raw)) != _param_widths(fdef.rets):
                raise TranslateError(
                    f"input declaration of {name} does not match the library definition"
                )
            dropped.add(user_externs[name])

    lines: list[str] = []
    kept: list[str] = []
    for item in order:
        assert item.name is not None
        fname = renames.get(item.name, item.name)
        if fname in reach:
            lines.append(texts[id(item)])
            if item.kind == "func":
                kept.append(fname)
    info.update(
        {
            "functions_kept": kept,
            "functions_total": sum(1 for it in order if it.kind == "func"),
            "renamed": renames,
            "externs": sorted(n for n in reach if n in lib_externs),
            "dropped_input_declarations": sorted(
                str(user.items[i].name) for i in dropped
            ),
        }
    )
    assert lib.version is not None
    return lines, dropped, lib.version


def _extern_protos(user: Module, mctx: _ModuleCtx) -> list[str]:
    have = {it.name for it in user.items if it.name}
    out = []
    for fname, (tys, rty) in mctx.called.items():
        if fname in have:
            continue
        ret = f"(.param .{rty} func_retval0) " if rty else ""
        params = ", ".join(f".param .{t} {fname}_param_{i}" for i, t in enumerate(tys))
        out.append(f".extern .func {ret}{fname}({params});")
    return out


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def translate(
    src: str,
    *,
    mode: str = "exact",
    lib_src: str | None = None,
    trap: bool = False,
    rules: Iterable[Any] | None = None,
    filename: str = "<input>",
    allow_cluster: bool = False,
    report: dict | None = None,
    warn_stream: Any = sys.stderr,
    async_delay: bool = False,
    count: bool = False,
    dl_slots: int = 128,
    dl_records: int = 4096,
) -> str:
    """Translate one PTX module for ``.target sm_80`` and return the new text.

    ``rules``: None loads :data:`DEFAULT_RULE_MODULES`; otherwise an iterable of rule modules
    or (name, module) pairs (used by tests). ``lib_src``: emulation library PTX to merge; when
    None, ``.extern`` prototypes are emitted for the library calls instead. ``report`` (dict)
    is filled with the translation record written by the CLI.

    ``async_delay`` (E1) and ``count`` (E2) put the rule module ``ptxemu.instrument`` in front
    of the rules; ``dl_slots`` / ``dl_records`` size the E1 pool (see instrument.py).
    """
    if mode not in MODES:
        raise TranslateError(f"unknown mode {mode!r} (expected one of {MODES})")
    rep: dict[str, Any] = report if report is not None else {}
    module = parse_module(src, filename)
    assert module.target is not None
    mctx = _ModuleCtx(
        mode,
        module.target,
        trap,
        filename,
        allow_cluster,
        warn_stream,
        async_delay=async_delay,
        count=count,
        dl_slots=dl_slots,
        dl_records=dl_records,
    )
    mctx.module_text = src

    def warn(msg: str) -> None:
        mctx.warnings.append(f"warning: {msg}")
        if warn_stream is not None:
            print(f"ptxemu: warning: {msg}", file=warn_stream)

    try:
        _map_target(module.target)
    except TranslateError as e:
        raise _located(f"{filename}: {e}") from None
    for item in module.items:
        if item.name is not None:
            mctx.symbols.add(item.name)
            if item.kind == "var" and item.space is not None:
                mctx.vars[item.name] = item.space
        if item.kind == "func" and item.func is not None:
            for it in item.func.body:
                if isinstance(it, str) and it.startswith(
                    (".local", ".shared", ".const")
                ):
                    try:
                        mctx.symbols.add(_var_name(it))
                    except TranslateError as e:
                        raise _located(
                            f"{filename}: function {item.name}: {e}"
                        ) from None
    for sym in mctx.symbols:
        if sym.startswith(
            (LOCAL_PREFIX, SHARED_PREFIX, GLOBAL_PREFIX, LIB_RENAME_PREFIX)
        ):
            raise _located(
                f"{filename}: input symbol {sym} uses a reserved wgemu prefix"
            )

    if rules is None:
        names = list(DEFAULT_RULE_MODULES)
        rule_list = load_rules(names, warn=warn)
    else:
        rule_list = _normalize_rules(rules)
        names = [n for n, _ in rule_list]
    if async_delay or count:
        from . import instrument

        instrument.check_shape(dl_slots, dl_records)
        rule_list = [("instrument", instrument), *rule_list]
        names = ["ptxemu.instrument", *names]
    loaded = [n for n, _ in rule_list]
    rep["rules"] = {
        "loaded": loaded,
        "missing": [
            n.rsplit(".", 1)[-1] for n in names if n.rsplit(".", 1)[-1] not in loaded
        ],
    }

    funcs = []
    for item in module.items:
        if item.kind == "func" and item.func is not None:
            funcs.append(_translate_func(item.func, rule_list, mctx))

    chunks: list[list[str]] = []
    for item in module.items:
        if item.kind == "func" and item.func is not None:
            chunks.append(emit_func(item.func))
        else:
            chunks.append([item.text])
    # .attribute(.unified(uuid1, uuid2)) is sm_90 only (ptx.txt 3115-3128); the declaration
    # stays, the attribute goes. rules_misc drops the matching ld operand qualifier.
    n_unified = 0
    for c in chunks:
        for k, ln in enumerate(c):
            new, cnt = _UNIFIED_ATTR_RE.subn("", ln)
            if cnt:
                c[k] = new
                n_unified += cnt
    if n_unified:
        warn(
            f"removed {n_unified} .attribute(.unified(...)); host/device address unification "
            "of those symbols is not emulated"
        )
    user_text = "\n".join("\n".join(c) for c in chunks)

    lib_info: dict[str, Any] = {}
    lib_lines: list[str] = []
    dropped: set[int] = set()
    out_version = module.version
    assert out_version is not None
    if lib_src is not None:
        try:
            lib_lines, dropped, lib_version = _merge_library(
                module, user_text, lib_src, mctx, lib_info
            )
        except TranslateError as e:
            if getattr(e, "located", False):
                raise
            raise _located(f"{filename}: library merge: {e}") from None
        if _version_tuple(lib_version) > _version_tuple(out_version):
            warn(
                f".version raised from {out_version} to {lib_version} for the emulation library"
            )
            out_version = lib_version
        lib_info["version"] = lib_version
    protos = _extern_protos(module, mctx) if lib_src is None else []
    if lib_src is None and mctx.called:
        warn("no emulation library given; emitted .extern prototypes for library calls")

    out: list[str] = list(module.preamble)
    out.append(f".version {out_version}")
    out.append(".target " + ", ".join([DST_TARGET, *module.target_opts]))
    if module.address_size:
        out.append(f".address_size {module.address_size}")
    out.append("")
    if lib_lines:
        out.append("// ---- wgemu emulation library (pruned) ----")
        out.extend(lib_lines)
        out.append("// ---- end of wgemu emulation library ----")
        out.append("")
    if protos:
        out.extend(protos)
        out.append("")
    if mctx.shared:
        for sym, (n, a) in mctx.shared.items():
            out.append(f".shared .align {a} .b8 {sym}[{n}];")
        out.append("")
    if mctx.globals:
        for sym, (n, a) in mctx.globals.items():
            out.append(f".weak .global .align {a} .b8 {sym}[{n}];")
        out.append("")
    for idx, chunk in enumerate(chunks):
        if idx in dropped:
            out.append(
                f"// wgemu: declaration of {module.items[idx].name} provided by the emulation library"
            )
        else:
            out.extend(chunk)

    agg: dict[str, int] = {}
    for counts in mctx.rewrites.values():
        for op, c in counts.items():
            agg[op] = agg.get(op, 0) + c
    rep.update(
        {
            "input": filename,
            "mode": mode,
            "trap": trap,
            "src_target": module.target,
            "dst_target": DST_TARGET,
            "version": {"input": module.version, "output": out_version},
            "functions": funcs,
            "rewritten_opcodes": agg,
            "rewrites_by_rule": mctx.rewrites,
            "rewritten_operands": dict(mctx.operand_rewrites),
            "called": {
                k: {"args": list(v[0]), "ret": v[1]} for k, v in mctx.called.items()
            },
            "shared_vars": {
                s: {"bytes": n, "align": a} for s, (n, a) in mctx.shared.items()
            },
            "global_vars": {
                s: {"bytes": n, "align": a} for s, (n, a) in mctx.globals.items()
            },
            "library": lib_info if lib_src is not None else None,
            "warnings": list(mctx.warnings),
        }
    )
    if async_delay:
        rep["async_delay"] = {"slots": dl_slots, "records": dl_records}
    if count:
        rep["count"] = True
    return "\n".join(out) + "\n"
