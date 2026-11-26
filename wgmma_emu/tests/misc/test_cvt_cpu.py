#!/usr/bin/env python3
"""CPU check of include/wgmma_emu/cvt.h against the exact reference tests/misc/ref_cvt.py.

cvt_driver.cpp is built with the host C++ compiler, fed every case on stdin and compared bit
for bit with ref_cvt. 8- and 16-bit sources are exhaustive; f32 / f64 / 32- and 64-bit integer
sources use special values, every tie point and neighbour of each destination format (tf32:
every 16th code) and seeded random words. Output follows PLAN section 11:

    WGEMU-TEST name=misc.cvt_cpu.<op>.<params> cases=<n> mismatches=<m>
    WGEMU-MISMATCH name=... in=... got=... want=...      (at most 10 per item)
    WGEMU-TEST-END

Usage: python tests/misc/test_cvt_cpu.py [--build-dir DIR] [--cxx g++] [--quick] [--only SUBSTR]
or pytest tests/misc/test_cvt_cpu.py. Exit status 0 when every item has no mismatch.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ref_cvt as R  # noqa: E402

SEED = 0x5EED_C0DE
MAX_MISMATCH_LINES = 10


# ---------------------------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------------------------


def _f32_bits(x: Fraction) -> int | None:
    """f32 bits of an exactly representable non-negative value, else None."""
    try:
        return R.encode_exact(8, 23, False, x)
    except AssertionError:
        return None


def _f64_bits(x: Fraction) -> int | None:
    try:
        return R.encode_exact(11, 52, False, x)
    except AssertionError:
        return None


def _boundary_values(fmt: R.Format, stride: int) -> list[Fraction]:
    """Each stride-th value of fmt, the midpoints to its neighbours and MAX_NORM's upper
    neighbour, all as exact non-negative Fractions."""
    vals = fmt.values
    out: list[Fraction] = []
    codes = list(range(0, len(vals), stride))
    if codes[-1] != len(vals) - 1:
        codes.append(len(vals) - 1)
    for c in codes:
        v = vals[c]
        nxt = vals[c + 1] if c + 1 < len(vals) else fmt.above
        out += [v, (v + nxt) / 2, nxt]
    out += [fmt.above * 2, fmt.above * 3 / 2]
    return out


def _neighbours32(bits: int) -> list[int]:
    out = [bits]
    if bits & 0x7FFFFFFF:
        out.append(bits - 1)
    if (bits & 0x7FFFFFFF) < 0x7F800000:
        out.append(bits + 1)
    return out


F32_SPECIALS = [
    0x00000000,
    0x80000000,
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x00800000,
    0x00800001,
    0x3F800000,
    0x3F7FFFFF,
    0x3F800001,
    0xBF800000,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0x7FBFFFFF,
    0xFFFFFFFF,
    0x7FFFFFFF,
]


def f32_cases(n_random: int, half_stride: int, tf32_stride: int) -> list[int]:
    rng = random.Random(SEED)
    seen: set[int] = set()
    out: list[int] = []

    def add(b: int) -> None:
        b &= 0xFFFFFFFF
        if b not in seen:
            seen.add(b)
            out.append(b)

    for b in F32_SPECIALS:
        add(b)
    for fmt, stride in (
        (R.E4M3, 1),
        (R.E5M2, 1),
        (R.F16, half_stride),
        (R.BF16, half_stride),
        (R.TF32, tf32_stride),
    ):
        for v in _boundary_values(fmt, stride):
            b = _f32_bits(v)
            if b is None:
                continue
            for nb in _neighbours32(b):
                add(nb)
                add(nb | 0x80000000)
    for _ in range(n_random):
        add(rng.getrandbits(32))
        # exponents around the fp8 / f16 / bf16 ranges, random mantissa
        e = rng.randrange(127 - 30, 127 + 20)
        add((rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23))
    return out


def f64_cases(n_random: int) -> list[int]:
    rng = random.Random(SEED + 1)
    seen: set[int] = set()
    out: list[int] = []

    def add(b: int) -> None:
        b &= 0xFFFFFFFFFFFFFFFF
        if b not in seen:
            seen.add(b)
            out.append(b)

    for b in (
        0,
        1 << 63,
        1,
        0x000FFFFFFFFFFFFF,
        0x0010000000000000,
        0x3FF0000000000000,
        0x7FEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000000,
        0x7FF0000000000001,
        0xFFFFFFFFFFFFFFFF,
    ):
        add(b)
    for v in _boundary_values(R.BF16, 1):
        b = _f64_bits(v)
        if b is None:
            continue
        for nb in (b - 1, b, b + 1) if b else (b, b + 1):
            add(nb)
            add(nb | (1 << 63))
    for _ in range(n_random):
        add(rng.getrandbits(64))
        e = rng.randrange(1023 - 140, 1023 + 130)
        add((rng.getrandbits(1) << 63) | (e << 52) | rng.getrandbits(52))
    return out


def int_cases(bits: int, n_random: int) -> list[int]:
    if bits <= 16:
        return list(range(1 << bits))
    rng = random.Random(SEED + bits)
    mask = (1 << bits) - 1
    seen: set[int] = set()
    out: list[int] = []

    def add(v: int) -> None:
        v &= mask
        if v not in seen:
            seen.add(v)
            out.append(v)

    for k in range(bits):
        for d in (-2, -1, 0, 1, 2):
            add((1 << k) + d)
            add(-((1 << k) + d))
        # bf16 tie points: 2^k + 2^(k-8) (+-1), i.e. halfway between neighbours
        if k >= 8:
            for d in (-1, 0, 1):
                add((1 << k) + (1 << (k - 8)) + d)
                add((1 << k) + 3 * (1 << (k - 8)) + d)
                add(-((1 << k) + (1 << (k - 8)) + d))
    add(mask)
    add(mask >> 1)
    add(1 << (bits - 1))
    for _ in range(n_random):
        add(rng.getrandbits(bits))
        add(rng.getrandbits(rng.randrange(1, bits + 1)))
    return out


def rtf_cases(n_random: int) -> list[tuple[int, int, int]]:
    rng = random.Random(SEED + 99)
    out = [(0, 0, 0), (1, 0, 5), (0, 1, 0), (1, 1, -149), (0, (1 << 64) - 1, -200)]
    for _ in range(n_random):
        nb = rng.randrange(1, 65)
        mag = rng.getrandbits(nb) | (1 << (nb - 1))
        if rng.random() < 0.3:  # exact tie pattern: ...1 followed by zeros
            k = rng.randrange(0, max(1, nb - 1))
            mag = ((mag >> k) | 1) << k
        out.append((rng.getrandbits(1), mag, rng.randrange(-220, 140)))
    return out


# ---------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------


class Item:
    def __init__(
        self,
        name: str,
        argv: list[str],
        inputs: list[tuple[int, ...]],
        ref: Callable[..., int],
        signed_last: bool = False,
    ) -> None:
        self.name, self.argv, self.inputs, self.ref = name, argv, inputs, ref
        self.signed_last = signed_last

    def stdin(self) -> str:
        lines = []
        for t in self.inputs:
            if self.signed_last:
                lines.append(" ".join([*(f"{x:x}" for x in t[:-1]), str(t[-1])]))
            else:
                lines.append(" ".join(f"{x:x}" for x in t))
        return "\n".join(lines) + "\n"


def _flags(**kw: int) -> str:
    return ".".join(f"{k}={v}" for k, v in kw.items())


def build_items(quick: bool) -> list[Item]:
    nr = 2000 if quick else 20000
    f32 = [(x,) for x in f32_cases(nr, 64 if quick else 4, 2048 if quick else 256)]
    f64 = [(x,) for x in f64_cases(nr)]
    h16 = [(x,) for x in range(1 << 16)]
    b8 = [(x,) for x in range(1 << 8)]
    perm = list(range(1 << 16))
    random.Random(SEED + 7).shuffle(perm)
    h16x2 = [(x | (perm[x] << 16),) for x in range(1 << 16)]
    f32s = [t[0] for t in f32]
    f32x2 = [(f32s[i], f32s[(i * 7919 + 13) % len(f32s)]) for i in range(len(f32s))]
    items: list[Item] = []
    for e5 in (0, 1):
        for relu in (0, 1):
            for sat in (0, 1):
                p = [str(e5), str(relu), str(sat)]
                tag = _flags(e5m2=e5, relu=relu, sat=sat)
                items += [
                    Item(
                        f"misc.cvt_cpu.f32_to_fp8.{tag}",
                        ["f32_to_fp8", *p],
                        f32,
                        lambda x, e5=e5, relu=relu, sat=sat: R.f32_to_fp8(
                            x, e5, relu, sat
                        ),
                    ),
                    Item(
                        f"misc.cvt_cpu.f16_to_fp8.{tag}",
                        ["f16_to_fp8", *p],
                        h16,
                        lambda x, e5=e5, relu=relu, sat=sat: R.f16_to_fp8(
                            x, e5, relu, sat
                        ),
                    ),
                    Item(
                        f"misc.cvt_cpu.f32x2_to_fp8x2.{tag}",
                        ["f32x2_to_fp8x2", *p],
                        f32x2,
                        lambda a, b, e5=e5, relu=relu, sat=sat: R.f32x2_to_fp8x2(
                            a, b, e5, relu, sat
                        ),
                    ),
                    Item(
                        f"misc.cvt_cpu.f16x2_to_fp8x2.{tag}",
                        ["f16x2_to_fp8x2", *p],
                        h16x2,
                        lambda x, e5=e5, relu=relu, sat=sat: R.f16x2_to_fp8x2(
                            x, e5, relu, sat
                        ),
                    ),
                ]
            tag = _flags(e5m2=e5, relu=relu)
            items += [
                Item(
                    f"misc.cvt_cpu.fp8_to_f16.{tag}",
                    ["fp8_to_f16", str(e5), str(relu)],
                    b8,
                    lambda x, e5=e5, relu=relu: R.fp8_to_f16(x, e5, relu),
                ),
                Item(
                    f"misc.cvt_cpu.fp8x2_to_f16x2.{tag}",
                    ["fp8x2_to_f16x2", str(e5), str(relu)],
                    h16,
                    lambda x, e5=e5, relu=relu: R.fp8x2_to_f16x2(x, e5, relu),
                ),
            ]
    for rn, rv in (("rn", R.RN), ("rz", R.RZ), ("rna", R.RNA)):
        for relu in (0, 1):
            for sat in (0, 1):
                items.append(
                    Item(
                        f"misc.cvt_cpu.f32_to_tf32.{rn}.{_flags(relu=relu, sat=sat)}",
                        ["f32_to_tf32", str(rv), str(relu), str(sat)],
                        f32,
                        lambda x, rv=rv, relu=relu, sat=sat: R.f32_to_tf32(
                            x, rv, relu, sat
                        ),
                    )
                )
    for rn, rv in (("rn", R.RN), ("rz", R.RZ), ("rm", R.RM), ("rp", R.RP)):
        items += [
            Item(
                f"misc.cvt_cpu.f32_to_bf16.{rn}",
                ["f32_to_bf16", str(rv)],
                f32,
                lambda x, rv=rv: R.f32_to_bf16(x, rv),
            ),
            Item(
                f"misc.cvt_cpu.f16_to_bf16.{rn}",
                ["f16_to_bf16", str(rv)],
                h16,
                lambda x, rv=rv: R.f16_to_bf16(x, rv),
            ),
            Item(
                f"misc.cvt_cpu.f64_to_bf16.{rn}",
                ["f64_to_bf16", str(rv)],
                f64,
                lambda x, rv=rv: R.f64_to_bf16(x, rv),
            ),
            Item(
                f"misc.cvt_cpu.f32_to_f16.{rn}",
                ["f32_to_f16", str(rv)],
                f32,
                lambda x, rv=rv: R.f32_to_f16(x, rv),
            ),
            Item(
                f"misc.cvt_cpu.bf16_to_f16.{rn}",
                ["bf16_to_f16", str(rv)],
                h16,
                lambda x, rv=rv: R.bf16_to_f16(x, rv),
            ),
        ]
        for signed in (0, 1):
            for bits in (8, 16, 32, 64):
                ints = [(v,) for v in int_cases(bits, nr)]
                if signed:  # the driver gets the value sign-extended to 64 bits
                    ints = [
                        ((v - (1 << bits)) & 0xFFFFFFFFFFFFFFFF,)
                        if v >> (bits - 1)
                        else (v,)
                        for (v,) in ints
                    ]
                items.append(
                    Item(
                        f"misc.cvt_cpu.int_to_bf16.{'s' if signed else 'u'}{bits}.{rn}",
                        ["int_to_bf16", str(signed), str(bits), str(rv)],
                        ints,
                        lambda v, signed=signed, bits=bits, rv=rv: R.int_to_bf16(
                            v, bool(signed), bits, rv
                        ),
                    )
                )
    items += [
        Item("misc.cvt_cpu.f16_to_f32", ["f16_to_f32"], h16, R.f16_to_f32),
        Item("misc.cvt_cpu.bf16_to_f32", ["bf16_to_f32"], h16, R.bf16_to_f32),
        Item("misc.cvt_cpu.f32_ftz", ["f32_ftz"], f32, R.f32_ftz),
        Item("misc.cvt_cpu.f32_to_f64", ["f32_to_f64"], f32, R.f32_to_f64),
    ]
    rtf = rtf_cases(nr // 4)
    for fmt in ("tf32", "f16", "bf16", "e4m3", "e5m2"):
        for rn, rv in R.RND_NAMES.items():
            for sat in (0, 1):
                items.append(
                    Item(
                        f"misc.cvt_cpu.round_to_fmt.{fmt}.{rn}.sat={sat}",
                        ["round_to_fmt", fmt, str(rv), str(sat)],
                        rtf,
                        lambda neg, mag, e2, fmt=fmt, rv=rv, sat=sat: R.round_to_fmt(
                            fmt, bool(neg), mag, e2, rv, bool(sat)
                        ),
                        signed_last=True,
                    )
                )
    return items


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def build_driver(build_dir: Path, cxx: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "cvt_driver"
    cmd = [
        *shlex.split(cxx),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-I",
        str(ROOT / "include"),
        str(HERE / "cvt_driver.cpp"),
        "-o",
        str(exe),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{shlex.join(cmd)} failed:\n{r.stdout}{r.stderr}")
    return exe


def run_item(exe: Path, item: Item) -> tuple[int, int, list[str]]:
    r = subprocess.run(
        [str(exe), *item.argv],
        input=item.stdin(),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return (
            len(item.inputs),
            len(item.inputs),
            [f"driver rc={r.returncode}: {r.stderr.strip()}"],
        )
    got = r.stdout.split()
    if len(got) != len(item.inputs):
        return (
            len(item.inputs),
            len(item.inputs),
            [f"driver printed {len(got)} results for {len(item.inputs)} inputs"],
        )
    bad = 0
    notes: list[str] = []
    for t, g in zip(item.inputs, got, strict=True):
        want = item.ref(*t)
        gv = int(g, 16)
        if gv != want:
            bad += 1
            if len(notes) < MAX_MISMATCH_LINES:
                ins = ",".join(f"{x:x}" if x >= 0 else str(x) for x in t)
                notes.append(f"in={ins} got={gv:x} want={want:x}")
    return len(item.inputs), bad, notes


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--quick", action="store_true", help="fewer random cases")
    ap.add_argument("--only", default="", help="run items whose name contains this")
    args = ap.parse_args(list(argv) if argv is not None else None)
    with tempfile.TemporaryDirectory(prefix="wgemu_cvt_") as tmp:
        exe = build_driver(args.build_dir or Path(tmp), args.cxx)
        total_bad = 0
        n_items = 0
        for item in build_items(args.quick):
            if args.only and args.only not in item.name:
                continue
            cases, bad, notes = run_item(exe, item)
            for n in notes:
                print(f"WGEMU-MISMATCH name={item.name} {n}")
            print(
                f"WGEMU-TEST name={item.name} cases={cases} mismatches={bad}",
                flush=True,
            )
            total_bad += bad
            n_items += 1
        print("WGEMU-TEST-END")
    return 0 if (total_bad == 0 and n_items > 0) else 1


def test_cvt_cpu() -> None:
    assert main(["--quick"] if os.environ.get("WGEMU_CVT_QUICK") else []) == 0


if __name__ == "__main__":
    sys.exit(main())
