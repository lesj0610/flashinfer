#!/usr/bin/env python3
"""Build and run the end-to-end (nvcc_emu) tests listed in tests/<area>/e2e_manifest.json.

Manifest (PLAN section 11): a JSON list of objects with exactly the keys
``name`` (str), ``source`` (path relative to the area directory), ``nvcc_args`` (list of str;
an item ``area:<path>`` is an extra input file relative to the area directory, e.g. a second
-rdc translation unit),
``modes`` (non-empty subset of exact/fast/check/delay/count), ``expect_violation`` (false, or
the non-empty list of WGEMU-VIOLATION kinds a negative test must print; such a test has the
single mode "check" or "delay", and the pseudo-kind "mismatch" in the list means the run must
report result mismatches instead of clean results), ``shim`` (bool; run with LD_PRELOAD of the
host shim) and ``covers`` (list of opcode / operand / directive regexes the test exercises).
The optional key ``expect_count`` (required when modes contain "count", allowed only then) is
the list of WGEMU-COUNT lines the count-mode run must print, in launch order: objects with
``kernel`` (regex that must fullmatch the printed kernel name) and every other printed field
(``grid``, ``wgmma[<shape>.<D>.<A>.<B>]``, ``bulk_bytes``, ``tma_bytes``, ``cpasync_bytes``,
``ldg_bytes``, ``mbar_waits``, ``wgmma_overflow`` when printed) with its exact value.

For every (test, mode) the source is built with ``tools/nvcc_emu`` in parallel (--jobs): modes
exact/fast/check set WGEMU_MODE=<mode>; "delay" builds exact with WGEMU_ASYNC_DELAY=1 (E1,
asynchronous copies delivered when a wait observes them); "count" builds exact with
WGEMU_COUNT=1 (E2) and always runs with the host shim, which prints one WGEMU-COUNT line per
launch. Tests then run one at a time on --gpu with a timeout. Output lines

    WGEMU-TEST name=<id> cases=<n> mismatches=<m>
    WGEMU-TEST-END
    WGEMU-MISMATCH name=<id> ...
    WGEMU-VIOLATION <kind> ...
    WGEMU-COUNT kernel=<name> grid=<x,y,z> [wgmma[...]=<n> ...] bulk_bytes=<n> ...

are parsed. A positive test passes only if it exits 0, prints WGEMU-TEST-END, reports at least
one item, every item has cases > 0 and mismatches == 0, and prints no WGEMU-VIOLATION. A
negative test passes only if the set of printed WGEMU-VIOLATION kinds equals its
expect_violation list (without "mismatch"), and it prints WGEMU-TEST-END and reports at least
one item, every item with cases > 0; without "mismatch" it must exit 0 with mismatches == 0
(so the violating code path really ran and computed the right result), with "mismatch" it must
exit 1 with mismatches > 0. In modes delay and count the translation reports must show the
instrumentation (wgemu_dl_init in every module whose PTX has cp.async, resp. a counter call
in some module), and a count-mode run must print exactly the expect_count lines; WGEMU-COUNT
lines in other modes fail the test.
Each ``covers`` regex must match an opcode / operand / directive of the PTX the test compiled
(the original sm_90a PTX kept by nvcc_emu), otherwise the test fails.

Before each run the GPU is checked with nvidia-smi; if any compute process is present the
harness stops and reports it (exit 3). Results go to --out (TSV) and <out>.items.tsv.

Exit status: 0 all passed, 1 failures, 2 usage / manifest error, 3 GPU busy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools"
NVCC_EMU = TOOLS / "nvcc_emu"
MODES = ("exact", "fast", "check", "delay", "count")
# Build environment per mode (translate-time switches of tools/nvcc_emu).
MODE_ENV = {
    "exact": {"WGEMU_MODE": "exact"},
    "fast": {"WGEMU_MODE": "fast"},
    "check": {"WGEMU_MODE": "check"},
    "delay": {"WGEMU_MODE": "exact", "WGEMU_ASYNC_DELAY": "1"},
    "count": {"WGEMU_MODE": "exact", "WGEMU_COUNT": "1"},
}
NEG_MODES = ("check", "delay")
MISMATCH = "mismatch"  # pseudo violation kind: the negative test must report mismatches
KEYS = {"name", "source", "nvcc_args", "modes", "expect_violation", "shim", "covers"}
OPTIONAL_KEYS = {"expect_count"}

_TEST_RE = re.compile(
    r"^\s*WGEMU-TEST\s+name=(\S+)\s+cases=(\d+)\s+mismatches=(\d+)\s*$"
)
_END_RE = re.compile(r"^\s*WGEMU-TEST-END\s*$")
_TEST_PREFIX_RE = re.compile(r"^\s*WGEMU-TEST(?!-END)\b")
_VIOL_RE = re.compile(r"WGEMU-VIOLATION\b")
_VIOL_KIND_RE = re.compile(r"WGEMU-VIOLATION\s+(\S+)")
_MISMATCH_RE = re.compile(r"^\s*WGEMU-MISMATCH\b")
_NAME_RE = re.compile(r"[A-Za-z0-9_.+-]+\Z")
_COUNT_RE = re.compile(r"^\s*WGEMU-COUNT\s+(.*)$")
_COUNT_FIELD_RE = re.compile(r"(\S+?)=(\S+)")


class ManifestError(Exception):
    pass


@dataclass
class Entry:
    area: str
    name: str
    source: Path
    nvcc_args: list[str]
    modes: list[str]
    expect_violation: list[str]  # empty: positive test
    shim: bool
    covers: list[str]
    expect_count: list[dict[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.area}/{self.name}"


@dataclass
class Job:
    entry: Entry
    mode: str
    exe: Path
    build_rc: int | None = None
    build_note: str = ""
    run_rc: int | None = None
    timed_out: bool = False
    items: list[tuple[str, int, int]] = field(default_factory=list)
    ended: bool = False
    violations: int = 0
    violation_kinds: set[str] = field(default_factory=set)
    mismatch_lines: int = 0
    counts: list[dict[str, str]] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    covers_missing: list[str] = field(default_factory=list)
    instr_missing: str = ""
    verdict: str = "NOT_RUN"
    note: str = ""

    @property
    def tag(self) -> str:
        return f"{self.entry.key}.{self.mode}"

    @property
    def keep_dir(self) -> Path:
        return Path(str(self.exe) + ".wgemu.d")


# ---------------------------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------------------------


def load_manifests(areas: list[str] | None) -> list[Entry]:
    paths = sorted((ROOT / "tests").glob("*/e2e_manifest.json"))
    if areas:
        known = {p.parent.name for p in paths}
        unknown = [a for a in areas if a not in known]
        if unknown:
            raise ManifestError(
                f"no e2e_manifest.json for area(s) {unknown}; found {sorted(known)}"
            )
        paths = [p for p in paths if p.parent.name in areas]
    entries: list[Entry] = []
    for p in paths:
        entries.extend(parse_manifest(p))
    return entries


def parse_manifest(path: Path) -> list[Entry]:
    area = path.parent.name
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ManifestError(f"{path}: {e}") from None
    if not isinstance(data, list):
        raise ManifestError(f"{path}: top level must be a list")
    out: list[Entry] = []
    seen: set[str] = set()
    for i, d in enumerate(data):
        where = f"{path}[{i}]"
        if not isinstance(d, dict):
            raise ManifestError(f"{where}: entry must be an object")
        missing = KEYS - d.keys()
        extra = d.keys() - KEYS - OPTIONAL_KEYS
        if missing or extra:
            raise ManifestError(
                f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}"
            )
        name = d["name"]
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ManifestError(f"{where}: bad name {name!r}")
        if name in seen:
            raise ManifestError(f"{where}: duplicate name {name!r}")
        seen.add(name)
        if not isinstance(d["source"], str):
            raise ManifestError(f"{where}: source must be a string")
        src = (path.parent / d["source"]).resolve()
        if not src.is_file():
            raise ManifestError(f"{where}: source {src} does not exist")
        args = d["nvcc_args"]
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ManifestError(f"{where}: nvcc_args must be a list of strings")
        args = [_area_input(a, path.parent, where) for a in args]
        modes = d["modes"]
        if (
            not isinstance(modes, list)
            or not modes
            or not all(m in MODES for m in modes)
            or len(set(modes)) != len(modes)
        ):
            raise ManifestError(
                f"{where}: modes must be a non-empty subset of {list(MODES)}, got {modes!r}"
            )
        if not isinstance(d["shim"], bool):
            raise ManifestError(f"{where}: shim must be true or false")
        ev = d["expect_violation"]
        if ev is False:
            ev = []
        elif not (
            isinstance(ev, list)
            and ev
            and all(isinstance(x, str) and x and not x.isspace() for x in ev)
            and len(set(ev)) == len(ev)
        ):
            raise ManifestError(
                f"{where}: expect_violation must be false or a non-empty list of distinct "
                "violation kinds"
            )
        if ev and (len(modes) != 1 or modes[0] not in NEG_MODES):
            raise ManifestError(
                f"{where}: expect_violation tests must have one mode out of {list(NEG_MODES)}"
            )
        expect_count = _parse_expect_count(d.get("expect_count"), where)
        if ("count" in modes) != bool(expect_count):
            raise ManifestError(
                f"{where}: expect_count (non-empty) is required with mode 'count' and "
                "allowed only then"
            )
        covers = d["covers"]
        if not isinstance(covers, list) or not all(
            isinstance(c, str) and c for c in covers
        ):
            raise ManifestError(
                f"{where}: covers must be a list of non-empty regex strings"
            )
        for c in covers:
            try:
                re.compile(c)
            except re.error as e:
                raise ManifestError(f"{where}: bad covers regex {c!r}: {e}") from None
        out.append(
            Entry(
                area,
                name,
                src,
                list(args),
                list(modes),
                list(ev),
                d["shim"],
                list(covers),
                expect_count,
            )
        )
    return out


def _parse_expect_count(v: object, where: str) -> list[dict[str, str]]:
    """expect_count -> list of {field: value} (values as printed, kernel as a regex)."""
    if v is None:
        return []
    if not isinstance(v, list) or not v:
        raise ManifestError(
            f"{where}: expect_count must be a non-empty list of objects"
        )
    out: list[dict[str, str]] = []
    for k, line in enumerate(v):
        w = f"{where}.expect_count[{k}]"
        if not isinstance(line, dict) or "kernel" not in line or "grid" not in line:
            raise ManifestError(
                f"{w}: object with at least 'kernel' and 'grid' expected"
            )
        fields: dict[str, str] = {}
        for key, val in line.items():
            if not isinstance(key, str) or not key or "=" in key or key.isspace():
                raise ManifestError(f"{w}: bad field name {key!r}")
            if isinstance(val, bool) or not isinstance(val, (str, int)):
                raise ManifestError(f"{w}: {key} must be a string or an integer")
            if isinstance(val, int) and val < 0:
                raise ManifestError(f"{w}: {key} must not be negative")
            fields[key] = str(val)
        try:
            re.compile(fields["kernel"])
        except re.error as e:
            raise ManifestError(f"{w}: bad kernel regex: {e}") from None
        out.append(fields)
    return out


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


AREA_PREFIX = "area:"


def _area_input(arg: str, area_dir: Path, where: str) -> str:
    """``area:<path>`` -> absolute path of an existing file in the area directory."""
    if not arg.startswith(AREA_PREFIX):
        return arg
    f = (area_dir / arg[len(AREA_PREFIX) :]).resolve()
    if not f.is_file():
        raise ManifestError(f"{where}: nvcc_args input {f} does not exist")
    return str(f)


def build_cmd(job: Job) -> list[str]:
    args = job.entry.nvcc_args
    cmd = [sys.executable, str(NVCC_EMU), str(job.entry.source), "-o", str(job.exe)]
    if not any(a.startswith(("-std", "--std")) for a in args):
        cmd.append("-std=c++17")
    if not any(re.match(r"-O\d?\Z|--optimize", a) for a in args):
        cmd.append("-O3")
    cmd += ["-I", str(ROOT / "include"), *args]
    return cmd


def build_one(job: Job, timeout: int) -> None:
    job.exe.parent.mkdir(parents=True, exist_ok=True)
    kd = job.keep_dir
    if kd.is_dir() and kd.resolve().is_relative_to(job.exe.parent.resolve()):
        shutil.rmtree(kd)  # stale reports would leak into the coverage check
    if job.exe.exists():
        job.exe.unlink()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("WGEMU_ASYNC_DELAY", "WGEMU_COUNT")
    }
    env.update(MODE_ENV[job.mode])
    log = Path(str(job.exe) + ".build.log")
    cmd = build_cmd(job)
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            r = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=False,
            )
            job.build_rc = r.returncode
        except subprocess.TimeoutExpired:
            job.build_rc = -1
            job.build_note = f"build timeout {timeout}s"
    if job.build_rc == 0 and not job.exe.is_file():
        job.build_rc = -2
        job.build_note = "nvcc_emu returned 0 but produced no executable"
    if job.build_rc != 0 and not job.build_note:
        job.build_note = f"build rc={job.build_rc}, see {log}"


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------


def gpu_processes(gpu: str) -> list[str]:
    cmd = [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"cannot query GPU {gpu} with nvidia-smi: {e}") from None
    if r.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi -i {gpu} failed (rc={r.returncode}): {r.stderr.strip() or r.stdout.strip()}"
        )
    return [
        ln.strip()
        for ln in r.stdout.splitlines()
        if ln.strip() and "No running" not in ln
    ]


def ensure_shim(path: Path) -> Path:
    src = ROOT / "host" / "shim.c"
    if not src.is_file():
        raise RuntimeError(f"{src} not found; cannot build the host shim")
    if path.is_file() and path.stat().st_mtime >= src.stat().st_mtime:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gcc", "-O2", "-shared", "-fPIC", str(src), "-ldl", "-o", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"shim build failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return path


def parse_output(job: Job, text: str) -> None:
    names: set[str] = set()
    for line in text.splitlines():
        if _END_RE.match(line):
            job.ended = True
            continue
        m = _TEST_RE.match(line)
        if m:
            name, cases, mism = m.group(1), int(m.group(2)), int(m.group(3))
            if name in names:
                job.malformed.append(f"duplicate item {name}")
            names.add(name)
            job.items.append((name, cases, mism))
            continue
        if _TEST_PREFIX_RE.match(line):
            job.malformed.append(line.strip()[:120])
        if _VIOL_RE.search(line):
            job.violations += 1
            mk = _VIOL_KIND_RE.search(line)
            job.violation_kinds.add(mk.group(1) if mk else "<no-kind>")
        if _MISMATCH_RE.match(line):
            job.mismatch_lines += 1
        m = _COUNT_RE.match(line)
        if m:
            job.counts.append(dict(_COUNT_FIELD_RE.findall(m.group(1))))


def run_one(job: Job, gpu: str, timeout: int, shim: Path | None) -> None:
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu)
    # The shim prints WGEMU-COUNT only for modules translated with counters; keeping it on in
    # every mode shows instrumentation leaking into the other modes (judge fails on it).
    env["WGEMU_COUNT"] = "1"
    if needs_shim(job):
        assert shim is not None
        env["LD_PRELOAD"] = (
            f"{shim}:{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else str(shim)
        )
    log = Path(str(job.exe) + f".gpu{gpu}.run.log")
    proc = subprocess.Popen(
        [str(job.exe)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=job.exe.parent,
        start_new_session=True,
        text=True,
        errors="replace",
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # our own child's process group only
        out, _ = proc.communicate()
        job.timed_out = True
    job.run_rc = proc.returncode
    log.write_text(out)
    parse_output(job, out)


def needs_shim(job: Job) -> bool:
    return job.entry.shim or job.mode == "count"


def check_instrumented(job: Job) -> None:
    """Delay / count mode: the translation reports must show the E1 / E2 instrumentation."""
    if job.mode not in ("delay", "count") or job.build_rc != 0:
        return
    reports = sorted(job.keep_dir.glob("*.wgemu_report.json"))
    if not reports:
        job.instr_missing = "no *.wgemu_report.json in keep dir"
        return
    counted = False
    for rp in reports:
        try:
            called = json.loads(rp.read_text()).get("called") or {}
        except (OSError, ValueError) as e:
            job.instr_missing = f"{rp.name}: {e}"
            return
        if job.mode == "delay":
            orig = rp.with_name(
                rp.name.replace(".wgemu_report.json", ".wgemu_orig.ptx")
            )
            has_cp = orig.is_file() and "cp.async" in orig.read_text()
            if has_cp and "wgemu_dl_init" not in called:
                job.instr_missing = f"{rp.name}: cp.async in PTX but no wgemu_dl_init"
                return
        elif "wgemu_count_add" in called or "wgemu_count_wgmma" in called:
            counted = True
    if job.mode == "count" and not counted:
        job.instr_missing = "no module calls wgemu_count_add / wgemu_count_wgmma"


def _count_diff(want: list[dict[str, str]], got: list[dict[str, str]]) -> str:
    if len(want) != len(got):
        return f"WGEMU-COUNT lines: {len(got)} printed, {len(want)} expected"
    for k, (w, g) in enumerate(zip(want, got, strict=True)):
        if not re.fullmatch(w["kernel"], g.get("kernel", "")):
            return f"WGEMU-COUNT[{k}] kernel={g.get('kernel')} !~ {w['kernel']}"
        wf = {x: v for x, v in w.items() if x != "kernel"}
        gf = {x: v for x, v in g.items() if x != "kernel"}
        if wf != gf:
            diff = sorted(
                f"{x}={gf.get(x, '-')}(want {wf.get(x, '-')})"
                for x in wf.keys() | gf.keys()
                if wf.get(x) != gf.get(x)
            )
            return f"WGEMU-COUNT[{k}] " + ", ".join(diff[:4])
    return ""


def judge(job: Job) -> None:
    e = job.entry
    if job.build_rc is None:
        job.verdict, job.note = "NOT_BUILT", ""
        return
    if job.build_rc != 0:
        job.verdict, job.note = "BUILD_FAIL", job.build_note
        return
    if job.run_rc is None:
        if job.covers_missing:
            job.verdict, job.note = (
                "FAIL",
                "covers not in PTX: " + ", ".join(job.covers_missing),
            )
        elif job.instr_missing:
            job.verdict, job.note = (
                "FAIL",
                "instrumentation missing: " + job.instr_missing,
            )
        else:
            job.verdict, job.note = "BUILT", "not run"
        return
    if job.timed_out:
        job.verdict, job.note = "TIMEOUT", ""
        return
    total_mism = sum(m for _, _, m in job.items)
    expect_mism = MISMATCH in e.expect_violation
    notes: list[str] = []
    if job.covers_missing:
        notes.append("covers not in PTX: " + ", ".join(job.covers_missing))
    if job.instr_missing:
        notes.append("instrumentation missing: " + job.instr_missing)
    if job.mode == "count":
        d = _count_diff(e.expect_count, job.counts)
        if d:
            notes.append(d)
    elif job.counts:
        notes.append(f"WGEMU-COUNT printed in mode {job.mode}")
    if e.expect_violation:
        want = set(e.expect_violation) - {MISMATCH}
        missing = sorted(want - job.violation_kinds)
        extra = sorted(job.violation_kinds - want)
        if missing:
            notes.append("expected violation not printed: " + ",".join(missing))
        if extra:
            notes.append("unexpected violation: " + ",".join(extra))
    if job.run_rc != (1 if expect_mism else 0):
        notes.append(f"exit {job.run_rc}")
    if not job.ended:
        notes.append("no WGEMU-TEST-END")
    if not job.items:
        notes.append("no WGEMU-TEST items")
    zero = [n for n, c, _ in job.items if c == 0]
    if zero:
        notes.append("cases=0: " + ",".join(zero[:5]))
    if expect_mism and not total_mism:
        notes.append("expected mismatches, got none")
    if total_mism and not expect_mism:
        notes.append(f"mismatches={total_mism}")
    if job.violations and not e.expect_violation:
        notes.append(f"violations={job.violations}")
    if job.malformed:
        notes.append("malformed: " + "; ".join(job.malformed[:3]))
    job.verdict = "FAIL" if notes else "PASS"
    job.note = "; ".join(notes) or (
        f"violations={job.violations} kinds={','.join(sorted(job.violation_kinds))}"
        + (f" mismatches={total_mism}" if expect_mism else "")
        if e.expect_violation
        else ""
    )


# ---------------------------------------------------------------------------------------------
# Coverage (original PTX kept by nvcc_emu)
# ---------------------------------------------------------------------------------------------


def _ptx_tokens(keep_dirs: list[Path]) -> tuple[set[str], set[str], set[str]] | None:
    sys.path.insert(0, str(TOOLS))
    from ptxemu import check  # noqa: PLC0415
    from ptxemu import lexer as lx  # noqa: PLC0415

    opcodes: set[str] = set()
    operands: set[str] = set()
    directives: set[str] = set()
    found = False
    for kd in keep_dirs:
        for f in sorted(kd.glob("*.wgemu_orig.ptx")):
            found = True
            for el in lx.lex(f.read_text()):
                if el.kind in (lx.LINE_DIRECTIVE, lx.HEADER) or (
                    el.kind == lx.STATEMENT and el.text.startswith(".")
                ):
                    directives.update(re.findall(r"\.[A-Za-z_]\w*", el.text))
                    continue
                if el.kind != lx.STATEMENT:
                    continue
                parts = check._split_instr(el.text)
                if parts is None:
                    continue
                opcodes.add(parts[0])
                operands.update(check._OPERAND_TOKEN_RE.findall(parts[1]))
    return (opcodes, operands, directives) if found else None


def _matches(pattern: str, tokens: tuple[set[str], set[str], set[str]]) -> bool:
    opcodes, operands, directives = tokens
    if pattern.startswith(("%", "\\%")):
        rx = re.compile(f"(?:{pattern})(?![\\w$])")
        return any(rx.match(t) for t in operands)
    if pattern.startswith((".", "\\.")):
        rx = re.compile(f"(?:{pattern})(?![\\w$])")
        return any(rx.match(t) for t in directives)
    rx = re.compile(f"(?:{pattern})(?![^.])")
    return any(rx.match(t) for t in opcodes)


def check_covers(jobs: list[Job]) -> dict[str, tuple[set[str], set[str], set[str]]]:
    """Fill covers_missing per job; return PTX tokens per passing-build test key."""
    per_key: dict[str, list[Job]] = {}
    for j in jobs:
        if j.build_rc == 0:
            per_key.setdefault(j.entry.key, []).append(j)
    tokens_by_key: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for key, js in per_key.items():
        toks = _ptx_tokens([j.keep_dir for j in js])
        if toks is None:
            for j in js:
                j.covers_missing = ["<no *.wgemu_orig.ptx in keep dir>"]
            continue
        tokens_by_key[key] = toks
        missing = [c for c in js[0].entry.covers if not _matches(c, toks)]
        for j in js:
            j.covers_missing = missing
    return tokens_by_key


def coverage_table(
    jobs: list[Job], tokens_by_key: dict, inventory: Path, min_sm: int
) -> list[str]:
    sys.path.insert(0, str(TOOLS))
    from ptxemu import check  # noqa: PLC0415

    rows = [
        r
        for r in check.load_inventory(inventory)
        if r.min_sm >= min_sm and r.handling != "native"
    ]
    passing = {
        j.entry.key
        for j in jobs
        if j.verdict == "PASS" and not j.entry.expect_violation
    }
    uncovered = []
    for r in rows:
        if not any(
            _matches(r.pattern, tokens_by_key[k]) for k in passing if k in tokens_by_key
        ):
            uncovered.append(f"{r.pattern}\t{r.min_sm}\t{r.handling}\t{r.section}")
    return uncovered


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def print_table(jobs: list[Job]) -> None:
    hdr = ("test", "mode", "cases", "mismatches", "verdict", "note")
    rows = []
    for j in jobs:
        cases = sum(c for _, c, _ in j.items) if j.items else "-"
        mism = sum(m for _, _, m in j.items) if j.items else "-"
        neg = " (neg)" if j.entry.expect_violation else ""
        rows.append(
            (j.entry.key + neg, j.mode, str(cases), str(mism), j.verdict, j.note)
        )
    w = [
        max(len(hdr[i]), *(len(r[i]) for r in rows)) if rows else len(hdr[i])
        for i in range(5)
    ]
    fmt = "  ".join(f"{{:{x}}}" for x in w) + "  {}"
    print(fmt.format(*hdr))
    for r in rows:
        print(fmt.format(*r))


def write_tsv(path: Path, jobs: list[Job]) -> None:
    cols = [
        "area",
        "name",
        "mode",
        "expect_violation",
        "shim",
        "build_rc",
        "run_rc",
        "timed_out",
        "items",
        "cases",
        "mismatches",
        "violations",
        "ended",
        "verdict",
        "note",
        "exe",
    ]
    lines = ["\t".join(cols)]
    items = ["\t".join(["area", "name", "mode", "item", "cases", "mismatches"])]
    for j in jobs:
        e = j.entry
        vals = [
            e.area,
            e.name,
            j.mode,
            str(e.expect_violation),
            str(e.shim),
            str(j.build_rc),
            str(j.run_rc),
            str(j.timed_out),
            str(len(j.items)),
            str(sum(c for _, c, _ in j.items)),
            str(sum(m for _, _, m in j.items)),
            str(j.violations),
            str(j.ended),
            j.verdict,
            j.note.replace("\t", " "),
            str(j.exe),
        ]
        lines.append("\t".join(vals))
        items += [
            "\t".join([e.area, e.name, j.mode, n, str(c), str(m)])
            for n, c, m in j.items
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    Path(str(path) + ".items.tsv").write_text("\n".join(items) + "\n")


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpu", help="GPU index (nvidia-smi / PCI bus order) to run on")
    ap.add_argument("--jobs", type=int, default=4, help="parallel builds")
    ap.add_argument(
        "--areas", help="comma-separated area directories under tests/ (default: all)"
    )
    ap.add_argument(
        "--modes",
        default=",".join(MODES),
        help="comma-separated subset of " + ",".join(MODES),
    )
    ap.add_argument("--filter", help="regex on <area>/<name>")
    ap.add_argument(
        "--timeout", type=int, default=300, help="per-run timeout in seconds"
    )
    ap.add_argument("--build-timeout", type=int, default=3600)
    ap.add_argument("--build-dir", default=str(ROOT / "build" / "e2e"))
    ap.add_argument("--out", help="results TSV (default <build-dir>/results.tsv)")
    ap.add_argument("--shim", default=str(ROOT / "build" / "libwgemu_shim.so"))
    ap.add_argument("--inventory", default=str(TOOLS / "ptxemu" / "inventory.tsv"))
    ap.add_argument("--min-sm", type=int, default=86)
    ap.add_argument(
        "--require-coverage",
        action="store_true",
        help="fail if an inventory row has no passing test",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--build-only", action="store_true")
    g.add_argument("--run-only", action="store_true")
    g.add_argument("--list", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    modes = [m for m in a.modes.split(",") if m]
    if not modes or any(m not in MODES for m in modes):
        print(f"error: --modes must be a subset of {MODES}", file=sys.stderr)
        return 2
    if not (a.build_only or a.list) and a.gpu is None:
        print("error: --gpu is required to run tests", file=sys.stderr)
        return 2
    if a.gpu is not None and not re.fullmatch(r"\d+", a.gpu):
        print("error: --gpu takes one GPU index", file=sys.stderr)
        return 2
    try:
        entries = load_manifests(
            [x for x in a.areas.split(",") if x] if a.areas else None
        )
    except ManifestError as e:
        print(f"manifest error: {e}", file=sys.stderr)
        return 2
    if a.filter:
        rx = re.compile(a.filter)
        entries = [e for e in entries if rx.search(e.key)]
    build_dir = Path(a.build_dir).resolve()
    jobs = [
        Job(e, m, build_dir / e.area / f"{e.name}.{m}")
        for e in entries
        for m in e.modes
        if m in modes
    ]
    if not jobs:
        print("error: no tests selected", file=sys.stderr)
        return 2
    if a.list:
        for e in entries:
            print(
                f"{e.key}\tmodes={','.join(e.modes)}\texpect_violation={','.join(e.expect_violation) or False}\tshim={e.shim}\tcovers={e.covers}"
                + (f"\texpect_count={len(e.expect_count)}" if e.expect_count else "")
            )
        return 0

    t0 = time.time()
    if not a.run_only:
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            list(ex.map(lambda j: build_one(j, a.build_timeout), jobs))
    else:
        for j in jobs:
            j.build_rc = 0 if j.exe.is_file() else -3
            if j.build_rc:
                j.build_note = f"--run-only: {j.exe} missing"
    print(
        f"build: {sum(j.build_rc == 0 for j in jobs)}/{len(jobs)} ok ({time.time() - t0:.0f}s)"
    )
    tokens_by_key = check_covers(jobs)
    for j in jobs:
        check_instrumented(j)

    rc_busy = False
    if not a.build_only:
        shim = None
        if any(needs_shim(j) and j.build_rc == 0 for j in jobs):
            try:
                shim = ensure_shim(Path(a.shim))
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
        for j in jobs:
            if j.build_rc != 0:
                continue
            try:
                procs = gpu_processes(a.gpu)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                rc_busy = True
                break
            if procs:
                print(
                    f"GPU {a.gpu} is in use; stopping before {j.tag}:", file=sys.stderr
                )
                for ln in procs:
                    print(f"  {ln}", file=sys.stderr)
                rc_busy = True
                break
            run_one(j, a.gpu, a.timeout, shim)
    for j in jobs:
        judge(j)
    print_table(jobs)
    out = Path(a.out) if a.out else build_dir / "results.tsv"
    write_tsv(out, jobs)
    print(f"results: {out}")

    ok = ("PASS", "BUILT") if a.build_only else ("PASS",)
    failed = [j for j in jobs if j.verdict not in ok]
    uncovered: list[str] = []
    inv = Path(a.inventory)
    if inv.is_file():
        uncovered = (
            coverage_table(jobs, tokens_by_key, inv, a.min_sm)
            if not a.build_only
            else []
        )
        if uncovered:
            print(
                f"inventory rows (min_sm >= {a.min_sm}, not native) without a passing test: {len(uncovered)}"
            )
            for u in uncovered:
                print(f"  {u}")
    elif a.require_coverage:
        print(f"error: --require-coverage but {inv} does not exist", file=sys.stderr)
        return 2
    else:
        print(f"warning: {inv} not found; coverage not checked", file=sys.stderr)
    print(
        f"summary: {sum(j.verdict in ok for j in jobs)} {'built' if a.build_only else 'passed'}, {len(failed)} not passed, {len(jobs)} total"
    )
    if rc_busy:
        return 3
    if failed or (a.require_coverage and uncovered):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
