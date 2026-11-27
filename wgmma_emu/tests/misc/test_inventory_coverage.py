#!/usr/bin/env python3
"""Static check that every sm_86..sm_90a inventory row is handled and exercised end to end.

For tools/ptxemu/inventory.tsv rows with min_sm >= 86 and handling != native:

* ``misc.inventory.current``: the TSV is the output of gen_inventory.py (``--check-only``).
* ``misc.inventory.covers``: some positive test of a ``tests/*/e2e_manifest.json`` lists a
  ``covers`` regex that only matches tokens the row matches. A cover proves that when the
  literal prefix of the regex (up to its first metacharacter) reaches, at a token boundary, a
  string the row regex matches in full: every token the cover accepts in the compiled PTX (the
  harness checks that one exists) starts with that string and is then matched by the row with
  the same boundary rule (opcode rows end at '.' or the token end, operand / directive rows at a
  non-word character). The harness's own coverage table (build_and_run.py) repeats this on the
  real PTX after a run.
* ``misc.inventory.features``: every ``#feature`` of gen_inventory.FEATURES has a
  ``// WGEMU-COVERS-FEATURE: <name>`` line in the source of a positive test; unknown names fail.
* ``misc.inventory.routing``: the first rule module (core order rules_wgmma, rules_async,
  rules_misc) whose ``match`` accepts the row's example opcode is the one the handling names
  (inline / noop / lib:* -> rules_misc, rewrite:async -> rules_async or, for rows whose note
  starts with "rules_misc:", rules_misc, rewrite:wgmma -> rules_wgmma); special-register rows
  must be recognised by rules_misc.rewrite_operand's pattern; strip / rewrite:core rows belong
  to the core and are not routed.

Nothing is built or run. Output follows PLAN section 11 (WGEMU-TEST / WGEMU-MISMATCH /
WGEMU-TEST-END). Usage: python tests/misc/test_inventory_coverage.py [--all] [--min-sm 86]
or pytest tests/misc/test_inventory_coverage.py. Exit status 0 when every item passes.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import re
import sys
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLS = ROOT / "tools"
HARNESS = ROOT / "tests" / "e2e" / "common" / "build_and_run.py"
INVENTORY = TOOLS / "ptxemu" / "inventory.tsv"
MAX_MISMATCH_LINES = 10
RULE_MODULES = ("ptxemu.rules_wgmma", "ptxemu.rules_async", "ptxemu.rules_misc")
FEATURE_RE = re.compile(r"WGEMU-COVERS-FEATURE:\s*(\S+)")

sys.path.insert(0, str(TOOLS))

from ptxemu import check  # noqa: E402
from ptxemu import core  # noqa: E402
from ptxemu import gen_inventory as gi  # noqa: E402


class Item:
    def __init__(self, name: str, show_all: bool) -> None:
        self.name = name
        self.cases = 0
        self.bad = 0
        self.show_all = show_all

    def check(self, ok: bool, detail: str) -> None:
        self.cases += 1
        if ok:
            return
        if self.show_all or self.bad < MAX_MISMATCH_LINES:
            print(f"WGEMU-MISMATCH name={self.name} {detail}")
        self.bad += 1

    def done(self) -> int:
        print(
            f"WGEMU-TEST name={self.name} cases={self.cases} mismatches={self.bad}",
            flush=True,
        )
        return self.bad if self.cases else 1


# ---------------------------------------------------------------------------------------------
# Cover regex -> literal prefix
# ---------------------------------------------------------------------------------------------

_META = set(".^$*+?{}[]|()")
_QUANT = set("*+?{")


def _top_level_alternation(rx: str) -> bool:
    depth, i, in_class = 0, 0, False
    while i < len(rx):
        ch = rx[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth == 0:
            return True
        i += 1
    return False


def literal_prefix(rx: str) -> tuple[str, str]:
    """(prefix every match starts with, the regex text after it)."""
    if _top_level_alternation(rx):
        return "", rx
    out: list[str] = []
    i = 0
    while i < len(rx):
        ch = rx[i]
        if ch == "\\":
            if i + 1 >= len(rx) or rx[i + 1].isalnum():
                break  # class escape (\d, \w, ...) or dangling backslash
            nxt, step = rx[i + 1], 2
        elif ch in _META:
            break
        else:
            nxt, step = ch, 1
        if i + step < len(rx) and rx[i + step] in _QUANT:
            break  # the quantifier makes this character optional / repeated
        out.append(nxt)
        i += step
    return "".join(out), rx[i:]


def _kind(rx: str) -> str:
    return check._row_kind(rx)


def _boundaries(prefix: str, rest: str, kind: str) -> list[int]:
    """Lengths L such that a matched token has a boundary right after prefix[:L]."""
    if kind == "opcode":
        cuts = [i for i, ch in enumerate(prefix) if ch == "." and i > 0]
        end_ok = rest == "" or rest.startswith("\\.")
    else:
        cuts = [
            i
            for i, ch in enumerate(prefix)
            if i > 1 and not (ch.isalnum() or ch in "_$")
        ]
        end_ok = rest == "" or (rest.startswith("\\") and not rest[1:2].isalnum())
    return cuts + ([len(prefix)] if end_ok and prefix else [])


def cover_proves(row_pattern: str, cover: str) -> bool:
    kind = _kind(row_pattern)
    if _kind(cover) != kind:
        return False
    prefix, rest = literal_prefix(cover)
    return any(
        re.fullmatch(row_pattern, prefix[:n]) for n in _boundaries(prefix, rest, kind)
    )


# ---------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------


def _load_harness():
    spec = importlib.util.spec_from_file_location("wgemu_build_and_run", HARNESS)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {HARNESS}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def item_current(show_all: bool) -> int:
    it = Item("misc.inventory.current", show_all)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gi.main(["--check-only"])
    sys.stderr.write(buf.getvalue())
    it.check(rc == 0, f"gen_inventory --check-only exit={rc}")
    return it.done()


def item_covers(rows, entries, show_all: bool) -> int:
    it = Item("misc.inventory.covers", show_all)
    positive = [e for e in entries if not e.expect_violation]
    for r in rows:
        ok = any(cover_proves(r.pattern, c) for e in positive for c in e.covers)
        it.check(
            ok, f"row={r.pattern} min_sm={r.min_sm} handling={r.handling} uncovered"
        )
    return it.done()


def item_features(entries, show_all: bool) -> int:
    it = Item("misc.inventory.features", show_all)
    known = {f.name for f in gi.FEATURES}
    marked: dict[str, list[str]] = {}
    for e in entries:
        if e.expect_violation:
            continue
        for name in FEATURE_RE.findall(Path(e.source).read_text()):
            marked.setdefault(name, []).append(e.key)
    for f in gi.FEATURES:
        if f.handling == "unsupported":  # listed for the audit only; no positive test
            continue
        it.check(f.name in marked, f"feature={f.name} no WGEMU-COVERS-FEATURE line")
    for name, keys in sorted(marked.items()):
        it.check(name in known, f"feature={name} unknown (marked in {','.join(keys)})")
    return it.done()


def _expected_modules(handling: str, note: str) -> tuple[str, ...] | None:
    if handling in ("inline", "noop") or handling.startswith("lib:"):
        return ("ptxemu.rules_misc",)
    if handling == "rewrite:async":
        if note.startswith("rules_misc:"):
            return ("ptxemu.rules_async", "ptxemu.rules_misc")
        return ("ptxemu.rules_async",)
    if handling == "rewrite:wgmma":
        return ("ptxemu.rules_wgmma",)
    return None  # strip / rewrite:core: handled by the core


def item_routing(rows, show_all: bool) -> int:
    it = Item("misc.inventory.routing", show_all)
    mods = {}
    for name in RULE_MODULES:
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - reported as a mismatch
            it.check(False, f"module={name} import failed: {type(e).__name__}: {e}")
    misc = mods.get("ptxemu.rules_misc")
    egs = {r.pattern: r for r in gi.ROWS}
    for r in rows:
        want = _expected_modules(r.handling, r.note)
        if want is None:
            continue
        g = egs.get(r.pattern)
        if g is None or not g.eg:
            it.check(False, f"row={r.pattern} has no example in gen_inventory.ROWS")
            continue
        eg = g.eg[0]
        if r.kind == "operand":
            ok = misc is not None and bool(misc._SREG_RE.match(eg.strip()))
            it.check(
                ok,
                f"row={r.pattern} eg={eg} not recognised by rules_misc.rewrite_operand",
            )
            continue
        if r.kind != "opcode":
            continue
        try:
            stmt = core.parse_stmt(eg)
        except core.TranslateError as e:
            it.check(False, f"row={r.pattern} eg={eg!r} does not parse: {e}")
            continue
        got = next((n for n in RULE_MODULES if n in mods and mods[n].match(stmt)), None)
        it.check(
            got in want,
            f"row={r.pattern} eg={eg} routed to {got}, want {'|'.join(want)}",
        )
    return it.done()


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="print every mismatch line")
    ap.add_argument("--min-sm", type=int, default=86)
    a = ap.parse_args(list(argv) if argv is not None else None)
    rows = [
        r
        for r in check.load_inventory(INVENTORY)
        if r.min_sm >= a.min_sm and r.handling != "native"
    ]
    harness = _load_harness()
    try:
        entries = harness.load_manifests(None)
    except harness.ManifestError as e:
        print(f"WGEMU-MISMATCH name=misc.inventory.manifests {e}")
        print("WGEMU-TEST name=misc.inventory.manifests cases=1 mismatches=1")
        print("WGEMU-TEST-END")
        return 1
    bad = item_current(a.all)
    bad += item_covers(rows, entries, a.all)
    bad += item_features(entries, a.all)
    bad += item_routing(rows, a.all)
    print("WGEMU-TEST-END")
    return 0 if bad == 0 else 1


def test_inventory_coverage() -> None:
    assert main([]) == 0


if __name__ == "__main__":
    sys.exit(main())
