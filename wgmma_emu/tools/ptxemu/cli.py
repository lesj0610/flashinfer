"""Command line front end: ``python -m ptxemu in.ptx -o out.ptx --lib emu_lib.ptx --mode exact``.

Exit status: 0 success, 1 translation / input error, 2 usage error, 3 the translated PTX still
contains sm_86+ items listed in inventory.tsv (the output and report are still written so the
remaining items can be inspected).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import check, core
from . import lexer as lx

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_REMAINING = 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ptxemu",
        description="Translate sm_90a PTX into sm_80 PTX that calls the wgmma_emu library.",
    )
    ap.add_argument("input", help="input PTX file ('-' for stdin)")
    ap.add_argument(
        "-o", "--output", required=True, help="output PTX file ('-' for stdout)"
    )
    ap.add_argument(
        "--lib", help="emulation library PTX (emu_lib.ptx) merged after the header"
    )
    ap.add_argument("--mode", choices=core.MODES, default="exact")
    ap.add_argument(
        "--trap", action="store_true", help="check mode: trap on violations"
    )
    ap.add_argument("--report", help="write a JSON translation report here")
    ap.add_argument(
        "--inventory",
        default=str(check.DEFAULT_INVENTORY),
        help="inventory.tsv for the remaining-opcode check",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="skip the inventory check (debugging only)",
    )
    ap.add_argument(
        "--min-sm",
        type=int,
        default=check.DEFAULT_MIN_SM,
        help="report inventory rows with min_sm >= this",
    )
    ap.add_argument(
        "--allow-cluster",
        action="store_true",
        help="remove .reqnctapercluster with more than one CTA instead of failing (launch must use cluster 1)",
    )
    ap.add_argument(
        "--async-delay",
        action="store_true",
        help="E1: deliver asynchronous copies into shared memory when a wait observes them",
    )
    ap.add_argument(
        "--dl-slots",
        type=int,
        default=128,
        help="E1: pool slots (CTAs running at the same time with delayed copies)",
    )
    ap.add_argument(
        "--dl-records",
        type=int,
        default=4096,
        help="E1: pending copies per CTA",
    )
    ap.add_argument(
        "--count",
        action="store_true",
        help="E2: per-launch counters printed by host/shim.c (WGEMU-COUNT lines)",
    )
    return ap


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def _write(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    p = Path(path)
    tmp = p.with_name(p.name + ".wgemu_tmp")
    tmp.write_text(text)
    tmp.replace(p)


def run(args: argparse.Namespace) -> int:
    report: dict = {}
    rows = None
    if not args.no_check:
        try:
            rows = check.load_inventory(args.inventory)
        except check.InventoryError as e:
            print(
                f"ptxemu: error: {e} (pass --no-check only for debugging)",
                file=sys.stderr,
            )
            return EXIT_ERROR
    try:
        src = _read(args.input)
        lib = _read(args.lib) if args.lib else None
    except OSError as e:
        print(f"ptxemu: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    try:
        out = core.translate(
            src,
            mode=args.mode,
            lib_src=lib,
            trap=args.trap,
            filename=args.input,
            allow_cluster=args.allow_cluster,
            report=report,
            async_delay=args.async_delay,
            count=args.count,
            dl_slots=args.dl_slots,
            dl_records=args.dl_records,
        )
    except (core.TranslateError, lx.LexError) as e:
        print(f"ptxemu: error: {e}", file=sys.stderr)
        if args.report:
            report.update({"input": args.input, "error": str(e)})
            Path(args.report).write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n"
            )
        return EXIT_ERROR
    hits: list[check.Hit] = []
    if rows is not None:
        hits = check.find_remaining(out, rows, args.min_sm)
    report["output"] = args.output
    report["library_file"] = args.lib
    report["check"] = {
        "inventory": None if args.no_check else args.inventory,
        "min_sm": args.min_sm,
        "remaining": [
            {
                "line": h.lineno,
                "kind": h.kind,
                "token": h.token,
                "inventory": h.pattern,
                "min_sm": h.min_sm,
            }
            for h in hits
        ],
    }
    try:
        _write(args.output, out)
        if args.report:
            Path(args.report).write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n"
            )
    except OSError as e:
        print(f"ptxemu: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    if hits:
        print(check.format_hits(hits, args.output), file=sys.stderr)
        print(
            f"ptxemu: error: {len(hits)} sm_{args.min_sm}+ item(s) remain after translation",
            file=sys.stderr,
        )
        return EXIT_REMAINING
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    return run(args)
