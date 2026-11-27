#!/usr/bin/env python3
"""Generate the wgmma end-to-end tests. Owner: W.

Writes ``tests/wgmma/e2e/*.cu`` and ``tests/wgmma/e2e_manifest.json`` (which also lists
``tests/wgmma/cute_gmma.cu``). The tests are built by ``tools/nvcc_emu`` for sm_90a through
``tests/e2e/common/build_and_run.py`` and run on SM80 in every mode they list.

Positive tests (one file per wgmma type combination, 17 dense + 16 sparse): every kernel holds
one ``wgmma.mma_async`` written as inline PTX in the CUTLASS form
(``{ .reg .pred p; setp.ne.b32 p, %s, 0; wgmma... p, ...; }``, immediates as literals) between
an operand fence, ``wgmma.fence``, ``wgmma.commit_group``, ``wgmma.wait_group 0`` and a second
operand fence. Per type combination:

* one kernel per valid N with the flags (SS/RS, imm-trans-a/b, imm-scale-a/b, .satfinite,
  sp-sel) cycled, two runs each (different swizzles, scale-d 1 and 0);
* at the smallest and the largest N a grid of kernels (f16/bf16: SS x trans-a x trans-b plus
  RS x trans-b; other types: SS and RS) whose flags take both values at both N, each run with
  every swizzle (none, 32B, 64B, 128B; the major is the kernel's imm-trans), scale-d 0 and 1,
  LBO/SBO gaps, K-major start offsets and base_offset 1..7 on part of the swizzled runs;
* a mode probe (``e2e_common.h``).

The host side (``e2e_common.h``) lays out the operands with ``ref_model.h`` and compares D bit
for bit with ``ref_exact`` (exact / check) or ``ref_fast`` (fast).

Sparse B in K-major with 32B swizzle is replaced by 64B (sparse B rows hold 64 bytes of K; see
``tests/wgmma/test_direct.cu``). Both runs of such a kernel name the swizzle actually used.

Negative tests (check mode only, ``expect_violation``):

* ``neg_acc_read_before_wait``: the accumulator is stored to global memory between
  ``commit_group`` and ``wait_group``;
* ``neg_smem_write_before_wait``: a byte of B in shared memory is changed between
  ``commit_group`` and ``wait_group``;
* ``neg_acc_write_without_fence``: inside one asm block, ``add.f32`` changes an accumulator
  register after ``wait_group`` and the next ``wgmma.mma_async`` follows without
  ``wgmma.fence``;
* ``neg_wait_group_too_few``: two groups, ``wait_group 1``, then the second group's
  accumulator is read;
* ``neg_rs_a_write_before_wait``: inside one asm block, ``xor.b32`` changes an RS A register
  right after the ``wgmma.mma_async`` that reads it, before ``wait_group``;
* ``neg_sparse_b_kmajor_sw32``: sparse f16 with K-major B whose descriptor says 32B swizzle
  (the host lays B out with 128B swizzle; the kernel rewrites descriptor bits 63-62 to 3).

Each negative test's manifest entry lists the one violation kind it must print
(``expect_violation``).

Usage (pure text generation, no build)::

    python3 tests/wgmma/gen_e2e.py           # (re)write the files
    python3 tests/wgmma/gen_e2e.py --check   # exit 1 if a file differs from the generated text
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "e2e"
MANIFEST = HERE / "e2e_manifest.json"
CUTLASS_INCLUDE = "/nvme512g/wgmma_emu_0fff1f87/cutlass/include"

BITS = {
    "f16": 16,
    "bf16": 16,
    "tf32": 32,
    "e4m3": 8,
    "e5m2": 8,
    "s8": 8,
    "u8": 8,
    "b1": 1,
}
CXX = {
    "f16": "F16",
    "bf16": "BF16",
    "tf32": "TF32",
    "e4m3": "E4M3",
    "e5m2": "E5M2",
    "s8": "S8",
    "u8": "U8",
    "b1": "B1",
    "f32": "F32",
    "s32": "S32",
}
FLOAT_IN = ("f16", "bf16", "tf32", "e4m3", "e5m2")
FP8 = ("e4m3", "e5m2")
INT8 = ("s8", "u8")
SW = (0, 32, 64, 128)
MODES_ALL = ["exact", "fast", "check"]
# E4 (shadow shared memory) variant of the generated positive tests. cute_gmma.cu is excluded: its
# PTX converts shared addresses with cvta.shared, which the E4 translation rejects.
MODES_SMEM = [*MODES_ALL, "smem"]
BASE_COVERS = [
    r"wgmma\.fence\.sync\.aligned",
    r"wgmma\.commit_group\.sync\.aligned",
    r"wgmma\.wait_group\.sync\.aligned",
]


def wg_k(at: str, sparse: bool) -> int:
    if at in ("f16", "bf16"):
        return 32 if sparse else 16
    if at == "tf32":
        return 16 if sparse else 8
    if at == "b1":
        return 256
    return 64 if sparse else 32


def valid_ns(at: str) -> list[int]:
    if at in INT8 or at == "b1":
        return [8, 16, 24, 32] + list(range(48, 257, 16))
    return list(range(8, 257, 8))


def nregs(n: int, dt: str) -> int:
    return n // 4 if dt == "f16" else n // 2


def mix32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


@dataclass(frozen=True)
class Combo:
    at: str
    bt: str
    dt: str
    sparse: bool

    @property
    def stem(self) -> str:
        return f"{'sp' if self.sparse else 'dense'}_{self.dt}_{self.at}_{self.bt}"

    @property
    def test_name(self) -> str:
        return f"wgmma.e2e.{'sp' if self.sparse else 'dense'}.{self.dt}.{self.at}.{self.bt}"


def combos() -> list[Combo]:
    out = []
    for sparse in (False, True):
        out.append(Combo("f16", "f16", "f16", sparse))
        out.append(Combo("f16", "f16", "f32", sparse))
        out.append(Combo("bf16", "bf16", "f32", sparse))
        out.append(Combo("tf32", "tf32", "f32", sparse))
        for a in FP8:
            for b in FP8:
                out.append(Combo(a, b, "f16", sparse))
                out.append(Combo(a, b, "f32", sparse))
        for a in INT8:
            for b in INT8:
                out.append(Combo(a, b, "s32", sparse))
        if not sparse:
            out.append(Combo("b1", "b1", "s32", False))
    return out


@dataclass
class Kern:
    n: int
    at: str
    bt: str
    dt: str
    sparse: bool = False
    rs: bool = False
    ta: bool = False
    tb: bool = False
    na: bool = False
    nb: bool = False
    sat: bool = False
    sel: int = 0
    tag: str = ""

    def make_tag(self, prefix: str) -> None:
        flags = "".join(
            s
            for s, on in (
                ("_ta", self.ta),
                ("_tb", self.tb),
                ("_na", self.na),
                ("_nb", self.nb),
                ("_sat", self.sat),
                ("_sel1", self.sel == 1),
            )
            if on
        )
        self.tag = f"{prefix}n{self.n}_{'rs' if self.rs else 'ss'}{flags}"


@dataclass
class Run:
    kernel: int
    sw_a: int
    sw_b: int
    order_a: int = 0
    order_b: int = 0
    kofs_a: int = 0
    kofs_b: int = 0
    gap_a: int = 0
    gap_b: int = 0
    scale_d: int = 1
    base_off: int = 0
    seed: int = 0


def opcode(k: Kern) -> str:
    parts = ["wgmma", "mma_async"] + (["sp"] if k.sparse else [])
    parts += ["sync", "aligned", f"m64n{k.n}k{wg_k(k.at, k.sparse)}", k.dt, k.at, k.bt]
    if k.at == "b1":
        parts += ["and", "popc"]
    if k.sat:
        parts.append("satfinite")
    return ".".join(parts)


def c_str_lines(text_lines: list[str], indent: str) -> list[str]:
    """PTX lines -> C string literal lines (one literal per PTX line)."""
    return [f'{indent}"{t}\\n"' for t in text_lines]


def wrap_list(items: list[str], indent: str, width: int = 96) -> list[str]:
    lines, cur = [], ""
    for it in items:
        piece = it if not cur else ", " + it
        if cur and len(indent) + len(cur) + len(piece) + 1 > width:
            lines.append(cur + ",")
            cur = it
        else:
            cur += piece
    if cur:
        lines.append(cur)
    return [indent + s for s in lines]


@dataclass
class Asm:
    """One wgmma.mma_async as an inline asm statement."""

    k: Kern
    acc: str = "d"
    a_inout: bool = False

    def operands(self) -> tuple[list[str], list[str], dict]:
        k = self.k
        nr = nregs(k.n, k.dt)
        cons = "+f" if k.dt == "f32" else "+r"
        outs = [f'"{cons}"({self.acc}[{i}])' for i in range(nr)]
        idx = {"d": list(range(nr))}
        ins: list[str] = []
        a_names = ["a0", "a1", "a2", "a3"]
        if k.rs and self.a_inout:
            idx["a"] = list(range(len(outs), len(outs) + 4))
            outs += [f'"+r"({a})' for a in a_names]
        n_out = len(outs)

        def add_in(text: str) -> int:
            ins.append(text)
            return n_out + len(ins) - 1

        if not k.rs:
            idx["adesc"] = add_in('"l"(adesc)')
        elif not self.a_inout:
            idx["a"] = [add_in(f'"r"({a})') for a in a_names]
        idx["bdesc"] = add_in('"l"(bdesc)')
        if k.sparse:
            idx["meta"] = add_in('"r"(meta)')
        idx["scale"] = add_in('"r"(g.scale_d)')
        return outs, ins, idx

    def ptx(self, idx: dict) -> str:
        k = self.k
        dvec = "{" + ", ".join(f"%{i}" for i in idx["d"]) + "}"
        if k.rs:
            a = "{" + ", ".join(f"%{i}" for i in idx["a"]) + "}"
        else:
            a = f"%{idx['adesc']}"
        ops = [dvec, a, f"%{idx['bdesc']}"]
        if k.sparse:
            ops += [f"%{idx['meta']}", str(k.sel)]
        ops.append("p")
        if k.at in FLOAT_IN:
            ops += ["-1" if k.na else "1", "-1" if k.nb else "1"]
        if k.at in ("f16", "bf16"):
            ops += [str(int(k.tb))] if k.rs else [str(int(k.ta)), str(int(k.tb))]
        return f"{opcode(k)} {', '.join(ops)};"

    def statement(self, indent: str, body: list[str] | None = None) -> list[str]:
        """``body`` replaces the single wgmma line; it may use {mma} and {d0}, {a0}."""
        outs, ins, idx = self.operands()
        mma = self.ptx(idx)
        lines_ptx = [
            "{",
            ".reg .pred p;",
            f"setp.ne.b32 p, %{idx['scale']}, 0;",
        ]
        fmt = {"mma": mma, "d0": f"%{idx['d'][0]}"}
        if "a" in idx:
            fmt["a0"] = f"%{idx['a'][0]}"
        lines_ptx += [b.format(**fmt) for b in body] if body else [mma]
        lines_ptx.append("}")
        out = [f"{indent}asm volatile("]
        out += c_str_lines(lines_ptx, indent + "    ")
        out.append(f"{indent}    :")
        out += wrap_list(outs, indent + "      ")
        out.append(f"{indent}    :")
        w = wrap_list(ins, indent + "      ")
        w[-1] += ");"
        out += w
        return out


def operand_fence(k: Kern, acc: str, indent: str) -> list[str]:
    nr = nregs(k.n, k.dt)
    cons = "+f" if k.dt == "f32" else "+r"
    ops = wrap_list(
        [f'"{cons}"({acc}[{i}])' for i in range(nr)], indent + "             "
    )
    ops[0] = indent + 'asm volatile("" : ' + ops[0].lstrip()
    ops[-1] += ' :: "memory");'
    return ops


FENCE = 'asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");'
COMMIT = 'asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");'


def wait_stmt(n: int) -> str:
    return f'asm volatile("wgmma.wait_group.sync.aligned {n};\\n" ::: "memory");'


def acc_decl(k: Kern, acc: str, indent: str) -> list[str]:
    nr = nregs(k.n, k.dt)
    if k.dt == "f32":
        return [
            f"{indent}float {acc}[{nr}];",
            f"{indent}#pragma unroll",
            f"{indent}for (int i = 0; i < {nr}; ++i) {acc}[i] = __uint_as_float(in[5 + i]);",
        ]
    return [
        f"{indent}uint32_t {acc}[{nr}];",
        f"{indent}#pragma unroll",
        f"{indent}for (int i = 0; i < {nr}; ++i) {acc}[i] = in[5 + i];",
    ]


def acc_store(k: Kern, acc: str, base: int, indent: str) -> list[str]:
    nr = nregs(k.n, k.dt)
    val = f"__float_as_uint({acc}[i])" if k.dt == "f32" else f"{acc}[i]"
    return [
        f"{indent}#pragma unroll",
        f"{indent}for (int i = 0; i < {nr}; ++i) out[{base} + i] = {val};",
    ]


def kernel_prologue(name: str, k: Kern, nout: int) -> list[str]:
    nr = nregs(k.n, k.dt)
    lines = [
        f"__global__ void __launch_bounds__(128) {name}(e2e::KArgs g) {{",
        "  uint64_t adesc, bdesc;",
        "  uint8_t* region = e2e::stage(g, &adesc, &bdesc);",
        "  (void)region;",
        f"  const uint32_t* in = g.rin + threadIdx.x * {5 + nr}u;",
        f"  uint32_t* out = g.rout + threadIdx.x * {nout}u;",
    ]
    if k.rs:
        lines.append("  uint32_t a0 = in[0], a1 = in[1], a2 = in[2], a3 = in[3];")
    if k.sparse:
        lines.append("  const uint32_t meta = in[4];")
    return lines


def positive_kernel(name: str, k: Kern) -> list[str]:
    nr = nregs(k.n, k.dt)
    ind = "  "
    lines = kernel_prologue(name, k, nr)
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(ind + FENCE)
    lines += Asm(k).statement(ind)
    lines.append(ind + COMMIT)
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines.append("}")
    return lines


def cxx_kern(fn: str, k: Kern) -> str:
    return (
        f'  {{{fn}, "{k.tag}", {k.n}, EType::{CXX[k.at]}, EType::{CXX[k.bt]}, '
        f"EType::{CXX[k.dt]}, {str(k.sparse).lower()}, {str(k.rs).lower()}, "
        f"{str(k.ta).lower()}, {str(k.tb).lower()}, {str(k.na).lower()}, "
        f"{str(k.nb).lower()}, {str(k.sat).lower()}, {k.sel}}},"
    )


def cxx_run(r: Run) -> str:
    return (
        f"  {{{r.kernel}, {r.sw_a}, {r.sw_b}, {r.order_a}, {r.order_b}, {r.kofs_a}, "
        f"{r.kofs_b}, {r.gap_a}, {r.gap_b}, {r.scale_d}, {r.base_off}, {r.seed}ull}},"
    )


HEADER = """\
// GENERATED by tests/wgmma/gen_e2e.py -- do not edit. Owner: W.
// {what}
// Build and run: python tests/e2e/common/build_and_run.py --area wgmma (see e2e_manifest.json).
#include <stdint.h>

#include "../e2e_common.h"

using wgemu::EType;
"""


def fix_sparse_b(c: Combo, k: Kern, sw_b: int) -> int:
    # Sparse B rows hold 64 bytes of K: K-major 32B swizzle cannot hold them (test_direct.cu).
    return 64 if (c.sparse and not k.tb and sw_b == 32) else sw_b


def build_positive(ci: int, c: Combo) -> tuple[str, list[str]]:
    fl = c.at in FLOAT_IN
    i8 = c.at in INT8
    trans_ok = c.at in ("f16", "bf16")
    sel_ok = c.sparse and c.at in ("f16", "bf16", "tf32")
    ns = valid_ns(c.at)
    kerns: list[Kern] = []
    runs: list[Run] = []
    seed = [1000003 * (ci + 1)]

    def next_seed() -> int:
        seed[0] += 1
        return seed[0]

    # N sweep, flags cycled.
    for idx, n in enumerate(ns):
        h = mix32(idx * 0x9E3779B1 + ci * 7919 + 1)
        rs = bool(idx & 1)
        k = Kern(n, c.at, c.bt, c.dt, c.sparse, rs)
        k.ta = trans_ok and not rs and bool((idx >> 1) & 1)
        k.tb = trans_ok and bool((idx >> 2) & 1)
        k.na = fl and bool(h & 1)
        k.nb = fl and bool((h >> 1) & 1)
        k.sat = i8 and bool((h >> 4) & 1)
        k.sel = int(sel_ok and bool((h >> 5) & 1))
        k.make_tag("sweep_")
        ki = len(kerns)
        kerns.append(k)
        for ri in range(2):
            runs.append(
                Run(
                    ki,
                    SW[(idx + 2 * ri) % 4],
                    fix_sparse_b(c, k, SW[(idx + 1 + 2 * ri) % 4]),
                    order_a=((h >> 6) ^ ri) & 1,
                    order_b=((h >> 7) ^ ri) & 1,
                    kofs_a=(h >> 8) & 3,
                    kofs_b=(h >> 10) & 3,
                    gap_a=(h >> 12) & 3,
                    gap_b=(h >> 14) & 3,
                    scale_d=(idx + ri + 1) & 1,
                    seed=next_seed(),
                )
            )
    # Grid at the smallest and the largest N.
    for gi, n in enumerate((ns[0], ns[-1])):
        if trans_ok:
            forms = [(False, ta, tb) for ta in (False, True) for tb in (False, True)]
            forms += [(True, False, tb) for tb in (False, True)]
        else:
            forms = [(False, False, False), (True, False, False)]
        for kk, (rs, ta, tb) in enumerate(forms):
            k = Kern(n, c.at, c.bt, c.dt, c.sparse, rs, ta, tb)
            if len(forms) == 2:
                k.na = fl and kk == 1
                k.nb = fl and kk == 0
                k.sat = i8 and kk == 1
                k.sel = int(sel_ok and kk == 1)
            else:
                k.na = fl and bool(kk & 1)
                k.nb = fl and bool(((kk >> 1) ^ kk) & 1)
                k.sel = int(sel_ok and bool((kk >> 1) & 1))
            k.make_tag("grid_")
            ki = len(kerns)
            kerns.append(k)
            for i, sw in enumerate(SW):
                boff = 1 + (3 * i + kk + gi) % 7 if sw and (i + kk) & 1 else 0
                runs.append(
                    Run(
                        ki,
                        sw,
                        fix_sparse_b(c, k, sw),
                        order_a=(i >> 1) & 1,
                        order_b=i & 1,
                        kofs_a=i + kk,
                        kofs_b=i + kk + 1,
                        gap_a=(i + kk) & 3,
                        gap_b=(i + 2 * kk) & 3,
                        scale_d=(i ^ kk) & 1,
                        base_off=boff,
                        seed=next_seed(),
                    )
                )
    probe = Kern(8, "f16", "f16", "f32", tag="probe")
    src = [
        HEADER.format(
            what=f"wgmma e2e: {c.test_name} (all N, flags, swizzles, majors, scale-d)"
        )
    ]
    src += positive_kernel("k_probe", probe)
    src.append("")
    for i, k in enumerate(kerns):
        src += positive_kernel(f"k{i}", k)
        src.append("")
    src.append("static const e2e::Kern kProbe = ")
    src[-1] += cxx_kern("k_probe", probe).strip().rstrip(",") + ";"
    src.append("static const e2e::Kern kKernels[] = {")
    src += [cxx_kern(f"k{i}", k) for i, k in enumerate(kerns)]
    src.append("};")
    src.append("static const e2e::Run kRuns[] = {")
    src += [cxx_run(r) for r in runs]
    src.append("};")
    src += [
        "",
        "int main() {",
        f'  e2e::Harness h("{c.test_name}");',
        "  h.probe(kProbe);",
        "  for (const e2e::Run& r : kRuns) h.run(kKernels[r.kernel], r);",
        "  return h.finish();",
        "}",
        "",
    ]
    k_name = rf"m64n\d+k{wg_k(c.at, c.sparse)}\.{c.dt}\.{c.at}\.{c.bt}"
    pre = (
        r"wgmma\.mma_async\.sp\.sync\.aligned\."
        if c.sparse
        else (r"wgmma\.mma_async\.sync\.aligned\.")
    )
    covers = [pre + k_name + (r"\.and\.popc" if c.at == "b1" else "")]
    for n in (ns[0], ns[-1]):
        covers.append(
            pre + f"m64n{n}k{wg_k(c.at, c.sparse)}" + rf"\.{c.dt}\.{c.at}\.{c.bt}"
        )
    if i8:
        covers.append(pre + k_name + r"\.satfinite")
    covers += BASE_COVERS
    return "\n".join(src), covers


# ---------------------------------------------------------------------------------------------
# Negative tests (check mode)
# ---------------------------------------------------------------------------------------------
NEG_K = Kern(16, "f16", "f16", "f32")
NEG_K_RS = Kern(16, "f16", "f16", "f32", rs=True)
NEG_K_SP = Kern(16, "f16", "f16", "f32", sparse=True)

NEG_MAIN = """\
int main() {{
  e2e::Harness h("{name}");
  wgref::Op o;
  o.n = 16;
  o.dt = EType::F32;
{setup}  const wgref::Problem P = wgref::make_problem(o, 0x5EED0001ull);
  wgref::LayoutSel la, lb;
  la.sw = lb.sw = 128;
  std::vector<uint32_t> raw;
  std::string why;
  wgref::Stat& st = h.stat("ran");
  ++st.cases;
  if (!h.launch(k_neg, P, {rs}, la, lb, 0, 0x5EED0001ull, {nout}, &raw, &why)) {{
    ++st.mismatches;
    printf("WGEMU-MISMATCH name={name}.ran layout-error %s\\n", why.c_str());
  }}
  // Pass / fail is decided by the build harness: this test must make check mode print
  // WGEMU-VIOLATION ({kind}).
  return h.finish();
}}
"""


def neg_acc_read_before_wait() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K, "  "
    lines = kernel_prologue("k_neg", k, 16)
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(ind + FENCE)
    lines += Asm(k).statement(ind)
    lines.append(ind + COMMIT)
    lines.append(ind + "// Undefined: the accumulator is read before wgmma.wait_group.")
    lines += acc_store(k, "d", 8, ind)
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines.append("}")
    return lines, 16, False, "acc-access-before-wait"


def neg_smem_write_before_wait() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K, "  "
    lines = kernel_prologue("k_neg", k, 8)
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(ind + FENCE)
    lines += Asm(k).statement(ind)
    lines.append(ind + COMMIT)
    lines += [
        ind + "// Undefined: B in shared memory changes before wgmma.wait_group.",
        ind + "if (threadIdx.x == 0) {",
        ind + "  volatile uint8_t* p = region + g.flip;",
        ind + "  *p = uint8_t(*p ^ 1u);",
        ind + "}",
        ind + "__syncthreads();",
    ]
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines.append("}")
    return lines, 8, False, "smem-modified-before-wait"


def neg_acc_write_without_fence() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K, "  "
    lines = kernel_prologue("k_neg", k, 8)
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(
        ind
        + "// Undefined: d0 is written after the wait and the second wgmma.mma_async"
    )
    lines.append(ind + "// that uses it is not preceded by wgmma.fence.")
    body = [
        "wgmma.fence.sync.aligned;",
        "{mma}",
        "wgmma.commit_group.sync.aligned;",
        "wgmma.wait_group.sync.aligned 0;",
        "add.f32 {d0}, {d0}, 0f3F800000;",
        "{mma}",
        "wgmma.commit_group.sync.aligned;",
        "wgmma.wait_group.sync.aligned 0;",
    ]
    lines += Asm(k).statement(ind, body)
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines.append("}")
    return lines, 8, False, "fence-missing"


def neg_wait_group_too_few() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K, "  "
    lines = kernel_prologue("k_neg", k, 16)
    lines += acc_decl(k, "d", ind)
    lines += acc_decl(k, "e", ind)
    lines += operand_fence(k, "d", ind)
    lines += operand_fence(k, "e", ind)
    lines.append(ind + FENCE)
    lines += Asm(k).statement(ind)
    lines.append(ind + COMMIT)
    lines += Asm(k, acc="e").statement(ind)
    lines.append(ind + COMMIT)
    lines.append(
        ind
        + "// Undefined: wait_group 1 leaves the second group (accumulator e) pending."
    )
    lines.append(ind + wait_stmt(1))
    lines += acc_store(k, "d", 0, ind)
    lines += acc_store(k, "e", 8, ind)
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += operand_fence(k, "e", ind)
    lines.append("}")
    return lines, 16, False, "acc-access-before-wait"


def neg_rs_a_write_before_wait() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K_RS, "  "
    lines = kernel_prologue("k_neg", k, 12)
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(ind + FENCE)
    lines.append(
        ind
        + "// Undefined: an A register of the RS wgmma changes before wgmma.wait_group."
    )
    lines += Asm(k, a_inout=True).statement(ind, ["{mma}", "xor.b32 {a0}, {a0}, 1;"])
    lines.append(ind + COMMIT)
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines += [
        ind + "out[8] = a0;",
        ind + "out[9] = a1;",
        ind + "out[10] = a2;",
        ind + "out[11] = a3;",
    ]
    lines.append("}")
    return lines, 12, True, "rs-a-modified-before-wait"


def neg_sparse_b_kmajor_sw32() -> tuple[list[str], int, bool, str]:
    k, ind = NEG_K_SP, "  "
    lines = kernel_prologue("k_neg", k, 8)
    lines += [
        ind
        + "// Undefined: sparse B rows hold 64 bytes of K; a K-major 32B-swizzle descriptor "
        "gives no",
        ind
        + "// address for bytes 32..63. The host lays B out with 128B swizzle; only the "
        "descriptor's",
        ind + "// swizzle field (bits 63-62) is changed to 32B.",
        ind + "bdesc = (bdesc & ~(3ull << 62)) | (3ull << 62);",
    ]
    lines += acc_decl(k, "d", ind)
    lines += operand_fence(k, "d", ind)
    lines.append(ind + FENCE)
    lines += Asm(k).statement(ind)
    lines.append(ind + COMMIT)
    lines.append(ind + wait_stmt(0))
    lines += operand_fence(k, "d", ind)
    lines += acc_store(k, "d", 0, ind)
    lines.append("}")
    return lines, 8, False, "desc-undefined"


NEGATIVES = {
    "neg_acc_read_before_wait": neg_acc_read_before_wait,
    "neg_smem_write_before_wait": neg_smem_write_before_wait,
    "neg_acc_write_without_fence": neg_acc_write_without_fence,
    "neg_wait_group_too_few": neg_wait_group_too_few,
    "neg_rs_a_write_before_wait": neg_rs_a_write_before_wait,
    "neg_sparse_b_kmajor_sw32": neg_sparse_b_kmajor_sw32,
}

# Extra host setup and opcode cover for negatives that are not dense m64n16k16 f32.f16.f16.
NEG_EXTRA = {
    "neg_sparse_b_kmajor_sw32": (
        "  o.sparse = true;\n",
        r"wgmma\.mma_async\.sp\.sync\.aligned\.m64n16k32\.f32\.f16\.f16",
    ),
}


def build_negative(stem: str) -> tuple[str, list[str], str]:
    kernel, nout, rs, kind = NEGATIVES[stem]()
    setup, op_cover = NEG_EXTRA.get(
        stem, ("", r"wgmma\.mma_async\.sync\.aligned\.m64n16k16\.f32\.f16\.f16")
    )
    name = f"wgmma.e2e.{stem.replace('_', '.', 1)}"
    src = [
        HEADER.format(what=f"wgmma e2e negative test (check mode): {stem}"),
        "#include <string>",
        "#include <vector>",
        "",
    ]
    src += kernel
    src.append("")
    src.append(
        NEG_MAIN.format(
            name=name, rs=str(rs).lower(), nout=nout, kind=kind, setup=setup
        )
    )
    covers = [op_cover, *BASE_COVERS]
    return "\n".join(src), covers, kind


# ---------------------------------------------------------------------------------------------
# Manifest and main
# ---------------------------------------------------------------------------------------------
CUTE_ENTRY = {
    "name": "wgmma.cute.gmma",
    "source": "cute_gmma.cu",
    "nvcc_args": ["-I", CUTLASS_INCLUDE, "--expt-relaxed-constexpr"],
    "modes": MODES_ALL,
    "expect_violation": False,
    "shim": False,
    "covers": [
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k16\.f32\.f16\.f16",
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k16\.f16\.f16\.f16",
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k16\.f32\.bf16\.bf16",
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k8\.f32\.tf32\.tf32",
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k32\.f32\.e4m3\.e4m3",
        r"wgmma\.mma_async\.sync\.aligned\.m64n\d+k32\.s32\.s8\.s8",
        *BASE_COVERS,
    ],
}


def generate() -> dict[Path, str]:
    files: dict[Path, str] = {}
    manifest = []
    for ci, c in enumerate(combos()):
        text, covers = build_positive(ci, c)
        rel = f"e2e/{c.stem}.cu"
        files[HERE / rel] = text
        manifest.append(
            {
                "name": c.test_name,
                "source": rel,
                "nvcc_args": [],
                "modes": MODES_SMEM,
                "expect_violation": False,
                "shim": False,
                "covers": covers,
            }
        )
    for stem in NEGATIVES:
        text, covers, kind = build_negative(stem)
        rel = f"e2e/{stem}.cu"
        files[HERE / rel] = text
        manifest.append(
            {
                "name": f"wgmma.e2e.{stem.replace('_', '.', 1)}",
                "source": rel,
                "nvcc_args": [],
                "modes": ["check"],
                "expect_violation": [kind],
                "shim": False,
                "covers": covers,
            }
        )
    manifest.append(CUTE_ENTRY)
    files[MANIFEST] = json.dumps(manifest, indent=2) + "\n"
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    args = ap.parse_args(argv)
    files = generate()
    stale = sorted(
        p
        for p in OUT_DIR.glob("*.cu")
        if p not in files  # generated files that no longer exist in the generator
    )
    if args.check:
        bad = [p for p, t in files.items() if not p.is_file() or p.read_text() != t]
        for p in bad + stale:
            print(f"out of date: {p.relative_to(HERE)}")
        return 1 if bad or stale else 0
    OUT_DIR.mkdir(exist_ok=True)
    for p, t in files.items():
        if not p.is_file() or p.read_text() != t:
            p.write_text(t)
    for p in stale:
        print(f"stale generated file (not removed): {p.relative_to(HERE)}")
    n_cu = sum(1 for p in files if p.suffix == ".cu")
    print(f"wrote {n_cu} .cu files and {MANIFEST.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
