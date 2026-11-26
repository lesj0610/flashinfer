"""Find instructions, special registers and directives that sm_80 cannot run.

``inventory.tsv`` (written by T2) is tab separated::

    opcode_regex  min_sm  handling  ptx_section  note

The first column selects what the row describes:

* starts with ``%`` (or ``\\%``): a special register / operand token, searched in instruction
  operands (``%cluster_ctarank``, ``%clusterid.x`` ...);
* starts with ``.`` (or ``\\.``): a directive, searched in directive statements and function
  headers (``.reqnctapercluster`` ...);
* anything else: an instruction opcode. The regex must match a prefix of the full dotted
  opcode that ends at a '.' boundary or at the end of the opcode, so ``cvt`` does not match
  ``cvta.to.shared.u64`` but ``wgmma\\.fence`` matches ``wgmma.fence.sync.aligned``.

Rows with ``min_sm`` at or above the threshold (default 86) are reported when they still occur
in translated PTX; so is a ``.target`` above sm_80. The final gate remains ``ptxas -arch=sm_80``.

CLI: ``python -m ptxemu.check out.ptx [--inventory inventory.tsv] [--min-sm 86]``; exit 3 if
anything is found, 1 on errors.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import lexer as lx

DEFAULT_INVENTORY = Path(__file__).with_name("inventory.tsv")
DEFAULT_MIN_SM = 86
HANDLING_RE = re.compile(
    r"native|inline|noop|strip|unsupported|lib:[A-Za-z_]\w*|rewrite:[A-Za-z_][\w.]*"
)
_OPERAND_TOKEN_RE = re.compile(r"%[A-Za-z_$][\w$]*(?:\.[A-Za-z_]\w*)?")
_GUARD_RE = re.compile(r"@!?%?[A-Za-z_$][\w$]*\s*")
_OPCODE_RE = re.compile(r"[A-Za-z_][\w.:]*")


class InventoryError(ValueError):
    pass


@dataclass(frozen=True)
class InvRow:
    pattern: str
    min_sm: int
    handling: str
    section: str
    note: str
    kind: str  # 'opcode' | 'operand' | 'directive'
    regex: re.Pattern
    lineno: int


@dataclass(frozen=True)
class Hit:
    lineno: int
    kind: str
    token: str
    pattern: str
    min_sm: int
    handling: str
    text: str


def _row_kind(pattern: str) -> str:
    if pattern.startswith(("%", "\\%")):
        return "operand"
    if pattern.startswith(("\\.", ".")):
        return "directive"
    return "opcode"


def load_inventory(path: str | Path) -> list[InvRow]:
    """Parse inventory.tsv. Malformed rows are errors, never skipped."""
    p = Path(path)
    if not p.is_file():
        raise InventoryError(f"inventory file not found: {p}")
    rows: list[InvRow] = []
    for n, raw in enumerate(p.read_text().splitlines(), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0].strip().lower() == "opcode_regex":
            continue  # header row
        if len(cols) < 4:
            raise InventoryError(
                f"{p}:{n}: expected at least 4 tab-separated columns, got {len(cols)}"
            )
        pattern, min_sm_s, handling, section = (c.strip() for c in cols[:4])
        note = "\t".join(cols[4:]).strip()
        if not pattern:
            raise InventoryError(f"{p}:{n}: empty opcode_regex")
        try:
            min_sm = int(min_sm_s)
        except ValueError:
            raise InventoryError(
                f"{p}:{n}: min_sm {min_sm_s!r} is not an integer"
            ) from None
        if not HANDLING_RE.fullmatch(handling):
            raise InventoryError(f"{p}:{n}: unknown handling {handling!r}")
        if min_sm >= 89 and handling == "native":
            raise InventoryError(f"{p}:{n}: min_sm {min_sm} rows cannot be 'native'")
        kind = _row_kind(pattern)
        try:
            if kind == "opcode":
                rx = re.compile(f"(?:{pattern})(?![^.])")
            else:
                rx = re.compile(f"(?<![\\w$])(?:{pattern})(?![\\w$])")
        except re.error as e:
            raise InventoryError(f"{p}:{n}: bad regex {pattern!r}: {e}") from None
        rows.append(InvRow(pattern, min_sm, handling, section, note, kind, rx, n))
    if not rows:
        raise InventoryError(f"{p}: no inventory rows")
    return rows


def _split_instr(text: str) -> tuple[str, str] | None:
    s = text.strip()
    m = _GUARD_RE.match(s)
    if m:
        s = s[m.end() :]
    m = _OPCODE_RE.match(s)
    if not m:
        return None
    return m.group(0), s[m.end() :]


def find_remaining(
    ptx: str, rows: list[InvRow], min_sm: int = DEFAULT_MIN_SM
) -> list[Hit]:
    """Instructions / operands / directives of rows with ``min_sm >= min_sm`` left in ``ptx``."""
    sel = [r for r in rows if r.min_sm >= min_sm]
    ops = [r for r in sel if r.kind == "opcode"]
    regs = [r for r in sel if r.kind == "operand"]
    dirs = [r for r in sel if r.kind == "directive"]
    hits: list[Hit] = []
    for el in lx.lex(ptx):
        if el.kind == lx.LINE_DIRECTIVE and el.text.startswith(".target"):
            words = el.text.split(None, 1)
            arch = words[1].split(",")[0].strip() if len(words) > 1 else ""
            m = re.fullmatch(r"sm_(\d+)[a-z]?", arch)
            if not m or int(m.group(1)) > 80:
                hits.append(
                    Hit(
                        el.lineno,
                        "target",
                        arch,
                        ".target",
                        int(m.group(1)) if m else 0,
                        "",
                        el.text,
                    )
                )
            continue
        if el.kind in (lx.LINE_DIRECTIVE, lx.HEADER) or (
            el.kind == lx.STATEMENT and el.text.startswith(".")
        ):
            for r in dirs:
                for m in r.regex.finditer(el.text):
                    hits.append(
                        Hit(
                            el.lineno,
                            "directive",
                            m.group(0),
                            r.pattern,
                            r.min_sm,
                            r.handling,
                            el.text,
                        )
                    )
            continue
        if el.kind != lx.STATEMENT:
            continue
        parts = _split_instr(el.text)
        if parts is None:
            continue
        opcode, rest = parts
        for r in ops:
            if r.regex.match(opcode):
                hits.append(
                    Hit(
                        el.lineno,
                        "opcode",
                        opcode,
                        r.pattern,
                        r.min_sm,
                        r.handling,
                        el.text,
                    )
                )
                break
        for tok in _OPERAND_TOKEN_RE.findall(rest):
            for r in regs:
                if r.regex.match(tok):
                    hits.append(
                        Hit(
                            el.lineno,
                            "operand",
                            tok,
                            r.pattern,
                            r.min_sm,
                            r.handling,
                            el.text,
                        )
                    )
                    break
    return hits


def format_hits(hits: list[Hit], name: str = "<ptx>") -> str:
    lines = []
    for h in hits:
        text = " ".join(h.text.split())
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(
            f"{name}:{h.lineno}: {h.kind} {h.token} (inventory {h.pattern!r}, min_sm {h.min_sm}, {h.handling or '-'}): {text}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ptxemu.check", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("ptx", help="translated PTX file")
    ap.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    ap.add_argument("--min-sm", type=int, default=DEFAULT_MIN_SM)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    try:
        rows = load_inventory(args.inventory)
        text = Path(args.ptx).read_text()
        hits = find_remaining(text, rows, args.min_sm)
    except (InventoryError, lx.LexError, OSError) as e:
        print(f"ptxemu.check: error: {e}", file=sys.stderr)
        return 1
    if hits:
        print(format_hits(hits, args.ptx), file=sys.stderr)
        print(
            f"ptxemu.check: {len(hits)} sm_{args.min_sm}+ item(s) remain in {args.ptx}",
            file=sys.stderr,
        )
        return 3
    print(f"ptxemu.check: {args.ptx}: no sm_{args.min_sm}+ items remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
