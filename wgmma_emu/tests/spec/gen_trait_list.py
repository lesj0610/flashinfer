#!/usr/bin/env python3
"""Generate tests/spec/trait_list.inc from the CuTe SM90 GMMA trait headers.

Every ``struct MMA_Traits<...>`` specialization in

  cute/atom/mma_traits_sm90_gmma.hpp
  cute/atom/mma_traits_sm90_gmma_ext.hpp
  cute/atom/mma_traits_sm90_gmma_sparse.hpp
  cute/atom/mma_traits_sm90_gmma_sparse_ext.hpp

becomes one line

  WGEMU_TRAIT("<name>", <A>, <B>, <D>, <N>, <K>, <sparse>, <rs>, <C++ type>)

where A/B/D are wgemu::EType enumerator names taken from the operation name,
N/K come from the name and are cross-checked against Shape_MNK, and the C++
type instantiates the operation with Major::K, ScaleIn::One and
SparseSel::Zero (the layouts do not depend on these parameters). The type is
the last macro argument so its template commas survive (__VA_ARGS__).

The script also checks the per-header specialization counts and that each
body declares the layout types the name implies, so a header update that adds
or reshapes atoms fails here instead of silently shrinking the test.

Usage:
  gen_trait_list.py [--cute-root DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_CUTE = Path("/nvme512g/wgmma_emu_0fff1f87/cutlass/include/cute")
HEADERS = {
    "atom/mma_traits_sm90_gmma.hpp": 320,
    "atom/mma_traits_sm90_gmma_ext.hpp": 736,
    "atom/mma_traits_sm90_gmma_sparse.hpp": 320,
    "atom/mma_traits_sm90_gmma_sparse_ext.hpp": 736,
}

TYPE_TOKENS = "BF16|F16|TF32|F32|E4M3|E5M2|S32|S8|U8"
NAME_RE = re.compile(
    r"^(?P<prefix>SM90::GMMA::SPARSE::GMMA_|SM90_)64x(?P<n>\d+)x(?P<k>\d+)_"
    rf"(?P<d>{TYPE_TOKENS})(?P<a>{TYPE_TOKENS})(?P<b>{TYPE_TOKENS})"
    r"_(?P<src>SS|RS)(?P<tn>_TN)?(?P<sat>_SATURATE)?$"
)
STRUCT_RE = re.compile(r"^struct MMA_Traits<(?P<inner>.+)>\s*$")
FIELD_RE = re.compile(r"^\s*using (?P<key>\w+)\s*=\s*(?P<val>.+?);\s*$")

ARG_MAP = {
    "tnspA": "cute::SM90::GMMA::Major::K",
    "tnspB": "cute::SM90::GMMA::Major::K",
    "scaleA": "cute::SM90::GMMA::ScaleIn::One",
    "scaleB": "cute::SM90::GMMA::ScaleIn::One",
    "spsel": "cute::SM90::GMMA::SparseSel::Zero",
}

# Logical K per input type (dense, sparse): PTX ISA 9.7.17 shape table.
K_OF = {
    "F16": (16, 32),
    "BF16": (16, 32),
    "TF32": (8, 16),
    "E4M3": (32, 64),
    "E5M2": (32, 64),
    "S8": (32, 64),
    "U8": (32, 64),
}


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def parse_header(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        m = STRUCT_RE.match(lines[i])
        if not m:
            i += 1
            continue
        inner = m.group("inner").strip()
        if "<" in inner:
            name, args = inner.split("<", 1)
            if not args.endswith(">"):
                raise SystemExit(f"{path}:{i + 1}: unbalanced template args: {inner}")
            args = [a.strip() for a in args[:-1].split(",")]
        else:
            name, args = inner, []
        fields = {}
        j = i + 1
        while j < len(lines) and not lines[j].startswith("};"):
            fm = FIELD_RE.match(lines[j])
            if fm:
                fields[fm.group("key")] = squash(fm.group("val"))
            j += 1
        if j >= len(lines):
            raise SystemExit(f"{path}:{i + 1}: struct body not closed")
        out.append(
            {"name": name, "args": args, "fields": fields, "line": i + 1, "path": path}
        )
        i = j + 1
    return out


def check_and_render(rec: dict) -> str:
    where = f"{rec['path']}:{rec['line']}"
    m = NAME_RE.match(rec["name"])
    if not m:
        raise SystemExit(f"{where}: unrecognized operation name {rec['name']}")
    n, k = int(m.group("n")), int(m.group("k"))
    a, b, d = m.group("a"), m.group("b"), m.group("d")
    sparse = m.group("prefix").startswith("SM90::GMMA::SPARSE")
    rs = m.group("src") == "RS"
    f = rec["fields"]

    if K_OF[a][1 if sparse else 0] != k:
        raise SystemExit(f"{where}: K={k} does not match input type {a}")
    if f.get("Shape_MNK") != f"Shape<_64,_{n},_{k}>":
        raise SystemExit(f"{where}: Shape_MNK {f.get('Shape_MNK')} vs name 64x{n}x{k}")
    if f.get("CLayout") != f"GMMA::CLayout_64x{n}":
        raise SystemExit(f"{where}: CLayout {f.get('CLayout')}")
    if f.get("BLayout") != f"GMMA::ABLayout<{n},{k}>":
        raise SystemExit(f"{where}: BLayout {f.get('BLayout')}")
    want_a = f"GMMA::ALayout_64x{k}" if rs else f"GMMA::ABLayout<64,{k}>"
    if f.get("ALayout") != want_a:
        raise SystemExit(f"{where}: ALayout {f.get('ALayout')} expected {want_a}")
    if sparse != ("ELayout" in f):
        raise SystemExit(f"{where}: ELayout presence does not match sparsity")
    if sparse and f["ELayout"] != f"GMMA::ELayout_64x{k}":
        raise SystemExit(f"{where}: ELayout {f['ELayout']}")

    targs = []
    for arg in rec["args"]:
        if arg not in ARG_MAP:
            raise SystemExit(f"{where}: unknown template parameter {arg}")
        targs.append(ARG_MAP[arg])
    ctype = "cute::" + rec["name"]
    if targs:
        ctype += "<" + ", ".join(targs) + ">"
    disp = rec["name"].replace("SM90::GMMA::SPARSE::", "SPARSE::")
    return (
        f'WGEMU_TRAIT("{disp}", {a}, {b}, {d}, {n}, {k}, {int(sparse)}, {int(rs)}, '
        f"{ctype})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cute-root", type=Path, default=DEFAULT_CUTE)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("trait_list.inc")
    )
    opts = ap.parse_args()

    body = []
    seen = set()
    totals = []
    for rel, expect in HEADERS.items():
        recs = parse_header(opts.cute_root / rel)
        if len(recs) != expect:
            raise SystemExit(f"{rel}: {len(recs)} specializations, expected {expect}")
        totals.append(f"// {rel}: {len(recs)}")
        for rec in recs:
            if rec["name"] in seen:
                raise SystemExit(f"{rel}:{rec['line']}: duplicate {rec['name']}")
            seen.add(rec["name"])
            body.append(check_and_render(rec))

    header = [
        "// Generated by tests/spec/gen_trait_list.py from the CuTe SM90 GMMA trait headers.",
        "// Do not edit by hand. Regenerate after a CUTLASS update.",
        *totals,
        f"// total: {len(body)}",
        "// clang-format off",
    ]
    opts.out.write_text("\n".join(header + body + ["// clang-format on", ""]))
    print(f"wrote {opts.out} ({len(body)} specializations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
