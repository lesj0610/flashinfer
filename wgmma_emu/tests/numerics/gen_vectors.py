#!/usr/bin/env python3
"""Reference vectors for include/wgmma_emu/numerics.h (PLAN section 1, criterion 2a).

Expected results come from MMA-Sim (b953d74) running on the CPU:

  wgmma.hopper.*            nv_ptx/isa_wgmma.py Hopper list; N does not change a single dot
                            product, so one entry per (K, dtype, atype, btype)
  mma.{ampere,ada,hopper}.* nv_ptx/isa_mma.py lists of the three architectures (f64 included)
  mma.hopper-inferred.*     the sm_89 fp8 shapes, which MMA-Sim's Hopper list does not contain,
                            run through fdpa.MMA_T_FDPA with sim.py's Hopper table. These check
                            the port of numerics.h only; no hardware data backs them.
  int.*, b1.*               exact Python integer reference (MMA-Sim has no integer model)

Every combination gets random bit patterns plus structured cases: +-0, subnormals, inf, NaN,
largest exponents, cancellation, overflow, terms around the F truncation boundary, c-dominant
and c-tiny accumulators, and zero accumulators around e_zero (see FLOAT_CATS).

Output files in OUT_DIR, per combination <name>.bin and <name>.json, plus manifest.txt (one name
per line) and index.json. .bin layout, little endian: a 96-byte header (HEADER_FMT, mirrored in
test_cpu.cpp), then A[count][units], B[count][units], C[count], D[count] with element widths
a_bytes, b_bytes, cd_bytes. units is K, or K / 32 words for b1.

The script runs MMA-Sim on the CPU only: it sets CUDA_VISIBLE_DEVICES to an empty string before
importing torch. The first import of mmasim.nv_ptx JIT-builds MMA-Sim's fma_cpu.cpp through
torch.utils.cpp_extension (needs ninja and a C++ compiler); later runs reuse the cached build.

Example:
  python gen_vectors.py /path/to/vectors --count 1000000 --procs 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import re
import struct
import sys
import time
import zlib

import numpy as np

DEFAULT_MMASIM = "/nvme512g/wgmma_emu_0fff1f87/ext/mmasim/src/mmasim"
MMASIM_COMMIT = "b953d74"
MAGIC = b"WGEMUNV1"
VERSION = 1
# magic, version, kind, atype, btype, d_f16, arch, api, K, units, a_bytes, b_bytes, cd_bytes,
# flags, F, L_max, e_zero, rho, count, 12 reserved bytes
HEADER_FMT = "<8s13I3iIQ12x"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 96

# wgemu::EType values (types.h); 10 marks f64, which only these vector files use.
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
    "f64": 10,
}
KIND_FLOAT, KIND_F64, KIND_INT, KIND_B1 = 0, 1, 2, 3
ARCH_CODE = {"Ampere": 0, "Ada Lovelace": 1, "Hopper": 2, None: 3}
ARCH_TAG = {"Ampere": "ampere", "Ada Lovelace": "ada", "Hopper": "hopper"}
API_CODE = {"mma": 0, "wgmma": 1, "none": 2}
RHO_CODE = {"RZ-FP32": 0, "RZ-E8M13": 1, "RNE-FP16": 2, None: 255}
FLAG_SAT, FLAG_A_SIGNED, FLAG_B_SIGNED, FLAG_INFERRED = 1, 2, 4, 8
BYTES = {"f16": 2, "bf16": 2, "tf32": 4, "f32": 4, "e4m3": 1, "e5m2": 1, "f64": 8}

# Float categories (relative weights):
#   uniform   uniformly random bit patterns, as in MMA-Sim's own equivalence tests
#   narrow    exponents clustered per dot: alignment and cancellation inside a block
#   wide      exponents spread over the whole range, subnormals included
#   specials  +-0, subnormal extremes, min normal, max finite, 1, inf, NaN payloads
#   cancel    exact and near-exact cancellation, including c = -(a0 * b0)
#   cdom      c far above or far below the products, around the F boundary
#   fbound    one dominant term per block, the others F +- 3 binades below it
#   overflow  same-sign products and c at the largest exponents
#   ezero     c = +-0, products around e_zero - F .. e_zero + 4
FLOAT_CATS = [
    ("uniform", 30),
    ("narrow", 14),
    ("wide", 10),
    ("specials", 10),
    ("cancel", 10),
    ("cdom", 8),
    ("fbound", 8),
    ("overflow", 4),
    ("ezero", 6),
]
INT_CATS = [("uniform", 50), ("extremes", 20), ("boundary", 30)]
B1_CATS = [("uniform", 50), ("density", 25), ("boundary", 25)]
UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
SIGNED_VIEW = {1: np.uint8, 2: np.int16, 4: np.int32, 8: np.int64}


class Fmt:
    """Binary floating-point format; ieee_special=False is e4m3fn (no inf, S.1111.111 NaN)."""

    def __init__(self, name, nbits, ebits, mbits, bias, ieee_special=True):
        self.name = name
        self.nbits = nbits
        self.ebits = ebits
        self.mbits = mbits
        self.bias = bias
        self.ieee_special = ieee_special
        self.top = (1 << ebits) - 1
        self.emin = 1 - bias
        self.emax = (self.top - 1 - bias) if ieee_special else (self.top - bias)
        self.esub = self.emin - mbits  # exponent of the lowest subnormal bit
        self.sign_bit = 1 << (ebits + mbits)
        mmask = (1 << mbits) - 1
        if ieee_special:
            maxfin = ((self.top - 1) << mbits) | mmask
            nonfinite = [
                self.top << mbits,
                (self.top << mbits) | (1 << (mbits - 1)),
                (self.top << mbits) | 1,
                (self.top << mbits) | mmask,
            ]
        else:
            maxfin = (self.top << mbits) | (mmask - 1)
            nonfinite = [(self.top << mbits) | mmask]
        vals = [0, 1, mmask, 1 << mbits, maxfin, bias << mbits] + nonfinite
        self.specials = np.array(
            vals + [v | self.sign_bit for v in vals], dtype=np.uint64
        )

    def encode(self, rng, neg, e):
        """Finite values whose leading bit has exponent e, with random lower bits."""
        e = np.clip(np.asarray(e, dtype=np.int64), self.esub - 1, self.emax)
        neg = np.broadcast_to(np.asarray(neg, dtype=bool), e.shape)
        one = np.uint64(1)
        rnd = rng.integers(0, 1 << self.mbits, size=e.shape, dtype=np.uint64)
        normal = e >= self.emin
        p = np.clip(e - self.esub, 0, self.mbits).astype(np.uint64)
        sub = (one << p) | (rnd & ((one << p) - one))
        mant = np.where(normal, rnd, np.where(e >= self.esub, sub, np.uint64(0)))
        expf = np.where(normal, e + self.bias, 0).astype(np.uint64)
        if not self.ieee_special:
            nan = (expf == np.uint64(self.top)) & (
                mant == np.uint64((1 << self.mbits) - 1)
            )
            mant = np.where(nan, mant - one, mant)
        bits = (expf << np.uint64(self.mbits)) | mant
        return np.where(neg, bits | np.uint64(self.sign_bit), bits).astype(np.uint64)


FMTS = {
    "f16": Fmt("f16", 16, 5, 10, 15),
    "bf16": Fmt("bf16", 16, 8, 7, 127),
    # random low 13 bits of tf32 exercise the operand truncation
    "tf32": Fmt("tf32", 32, 8, 23, 127),
    "f32": Fmt("f32", 32, 8, 23, 127),
    "e4m3": Fmt("e4m3", 8, 4, 3, 7, ieee_special=False),
    "e5m2": Fmt("e5m2", 8, 5, 2, 15),
    "f64": Fmt("f64", 64, 11, 52, 1023),
}

_W: dict = {}  # per-process cache: torch module, MMA-Sim objects, fp8 decode tables


def rbits(rng, nbits, shape):
    return rng.integers(0, 1 << nbits, size=shape, dtype=np.uint64)


def signs(rng, shape):
    return rng.random(shape) < 0.5


def decode_f64(name, bits):
    """Exact float64 value of operand bits (tf32 truncated as MMA-Sim does)."""
    if name == "f16":
        return bits.astype(np.uint16).view(np.float16).astype(np.float64)
    if name == "bf16":
        return (
            (bits.astype(np.uint32) << np.uint32(16))
            .view(np.float32)
            .astype(np.float64)
        )
    if name == "tf32":
        return (
            (bits.astype(np.uint32) & np.uint32(0xFFFFE000))
            .view(np.float32)
            .astype(np.float64)
        )
    if name == "f32":
        return bits.astype(np.uint32).view(np.float32).astype(np.float64)
    if name == "f64":
        return bits.astype(np.uint64).view(np.float64)
    if name in ("e4m3", "e5m2"):
        key = "fp8tab_" + name
        if key not in _W:
            torch = _W["torch"]
            dt = torch.float8_e4m3fn if name == "e4m3" else torch.float8_e5m2
            _W[key] = (
                torch.arange(256, dtype=torch.uint8).view(dt).to(torch.float64).numpy()
            )
        return _W[key][bits.astype(np.int64)]
    raise ValueError(f"decode_f64: unsupported format {name}")


def encode_exact(name, v):
    """Bits of v in format name and a mask of values that are exactly representable."""
    with np.errstate(over="ignore", invalid="ignore"):
        if name == "f32":
            w = v.astype(np.float32)
            ok = np.isfinite(v) & (w.astype(np.float64) == v)
            return w.view(np.uint32).astype(np.uint64), ok
        if name == "f16":
            w = v.astype(np.float16)
            ok = np.isfinite(v) & (w.astype(np.float64) == v)
            return w.view(np.uint16).astype(np.uint64), ok
        if name == "f64":
            return v.view(np.uint64).astype(np.uint64), np.isfinite(v)
    raise ValueError(f"encode_exact: unsupported format {name}")


def split_target(rng, target, fa, fb):
    """Leading-bit exponents ea + eb ~= target inside both formats (subnormals included)."""
    ea = np.clip(target // 2 + rng.integers(-2, 3, size=target.shape), fa.esub, fa.emax)
    eb = np.clip(target - ea, fb.esub, fb.emax)
    return ea, eb


def gen_float_cat(rng, cat, n, ctx):
    fa, fb, fc = ctx["fa"], ctx["fb"], ctx["fc"]
    K, L, F, ez = ctx["K"], ctx["L"], ctx["F"], ctx["e_zero"]
    if cat == "uniform":
        return (
            rbits(rng, fa.nbits, (n, K)),
            rbits(rng, fb.nbits, (n, K)),
            rbits(rng, fc.nbits, n),
        )
    ca = rng.integers(fa.esub, fa.emax + 1, size=(n, 1))
    cb = rng.integers(fb.esub, fb.emax + 1, size=(n, 1))

    def narrow():
        a = fa.encode(rng, signs(rng, (n, K)), ca + rng.integers(-2, 3, size=(n, K)))
        b = fb.encode(rng, signs(rng, (n, K)), cb + rng.integers(-2, 3, size=(n, K)))
        c = fc.encode(
            rng, signs(rng, n), ca[:, 0] + cb[:, 0] + rng.integers(-4, 5, size=n)
        )
        return a, b, c

    if cat == "narrow":
        return narrow()
    if cat == "wide":
        a = fa.encode(
            rng, signs(rng, (n, K)), rng.integers(fa.esub, fa.emax + 1, size=(n, K))
        )
        b = fb.encode(
            rng, signs(rng, (n, K)), rng.integers(fb.esub, fb.emax + 1, size=(n, K))
        )
        c = fc.encode(rng, signs(rng, n), rng.integers(fc.esub, fc.emax + 1, size=n))
        return a, b, c
    if cat == "specials":
        a, b, c = narrow()
        for x, f, prob in ((a, fa, 0.12), (b, fb, 0.12), (c, fc, 0.25)):
            m = rng.random(x.shape) < prob
            x[m] = f.specials[rng.integers(0, len(f.specials), size=int(m.sum()))]
        return a, b, c
    if cat == "cancel":
        a, b, c = narrow()
        for j in range(1, K - 1, 2):  # pairs (1,2), (3,4), ...: a_j b_j + a_j (-b_j)
            a[:, j + 1] = a[:, j]
            b[:, j + 1] = b[:, j] ^ np.uint64(fb.sign_bit)
            tweak = rng.random(n) < 0.4  # off by one ulp (tf32: an ignored bit)
            b[tweak, j + 1] ^= np.uint64(1)
        a[rng.random(n) < 0.5, K - 1] = 0
        with np.errstate(over="ignore", invalid="ignore"):
            v = -(decode_f64(ctx["a"], a[:, 0]) * decode_f64(ctx["b"], b[:, 0]))
        enc, ok = encode_exact(ctx["c"], v)
        c = np.where(ok & (rng.random(n) < 0.6), enc, c)
        return a, b, c
    if cat == "cdom":
        a, b, _ = narrow()
        up = rng.random(n) < 0.5
        far = rng.choice(
            np.array([F - 2, F - 1, F, F + 1, F + 2, F + 3, 40, 80]), size=n
        )
        d = np.where(up, far, -rng.integers(F - 3, F + 4, size=n))
        c = fc.encode(rng, signs(rng, n), ca[:, 0] + cb[:, 0] + d)
        return a, b, c
    if cat == "fbound":
        lo, hi = fa.esub + fb.esub + F, fa.emax + fb.emax
        if lo > hi:
            lo = fa.esub + fb.esub
        top = rng.integers(lo, hi + 1, size=(n, 1))
        dom = (np.arange(K) % L == 0)[None, :]
        target = np.where(dom, top, top - F + rng.integers(-3, 4, size=(n, K)))
        ea, eb = split_target(rng, target, fa, fb)
        a = fa.encode(rng, signs(rng, (n, K)), ea)
        b = fb.encode(rng, signs(rng, (n, K)), eb)
        zero = np.where(signs(rng, n), np.uint64(fc.sign_bit), np.uint64(0))
        near = fc.encode(
            rng, signs(rng, n), top[:, 0] - F + rng.integers(-3, 4, size=n)
        )
        c = np.where(rng.random(n) < 0.5, zero, near)
        return a, b, c
    if cat == "overflow":
        neg = np.broadcast_to(rng.random((n, 1)) < 0.5, (n, K))
        a = fa.encode(rng, neg, fa.emax - rng.integers(0, 2, size=(n, K)))
        b = fb.encode(
            rng, np.zeros((n, K), dtype=bool), fb.emax - rng.integers(0, 2, size=(n, K))
        )
        big = fc.encode(rng, neg[:, 0], fc.emax - rng.integers(0, 2, size=n))
        other = fc.encode(
            rng, signs(rng, n), rng.integers(fc.esub, fc.emax + 1, size=n)
        )
        c = np.where(rng.random(n) < 0.5, big, other)
        return a, b, c
    if cat == "ezero":
        target = ez + rng.integers(-F - 4, 5, size=(n, K))
        ea, eb = split_target(rng, target, fa, fb)
        a = fa.encode(rng, signs(rng, (n, K)), ea)
        b = fb.encode(rng, signs(rng, (n, K)), eb)
        c = np.where(signs(rng, n), np.uint64(fc.sign_bit), np.uint64(0))
        return a, b, c
    raise ValueError(f"unknown float category {cat}")


def gen_int_cat(rng, cat, n, K, a_signed, b_signed):
    ext_s = np.array([0x80, 0x7F, 0xFF, 0x00, 0x01], dtype=np.uint64)  # -128 127 -1 0 1
    ext_u = np.array([0x00, 0xFF, 0x80, 0x7F, 0x01], dtype=np.uint64)
    big_s = np.array([0x80, 0x7F], dtype=np.uint64)
    big_u = np.array([0xFF, 0xFE], dtype=np.uint64)
    if cat == "uniform":
        a = rbits(rng, 8, (n, K))
        b = rbits(rng, 8, (n, K))
        c = rng.integers(-(2**31), 2**31, size=n, dtype=np.int64)
    elif cat == "extremes":
        a = rng.choice(ext_s if a_signed else ext_u, size=(n, K))
        b = rng.choice(ext_s if b_signed else ext_u, size=(n, K))
        c = rng.integers(-(2**31), 2**31, size=n, dtype=np.int64)
    elif cat == "boundary":  # c within 2^22 of a limit, |sum of products| up to ~2^21
        a = rng.choice(big_s if a_signed else big_u, size=(n, K))
        b = rng.choice(big_s if b_signed else big_u, size=(n, K))
        off = rng.integers(0, 1 << 22, size=n, dtype=np.int64)
        c = np.where(rng.random(n) < 0.5, 2**31 - 1 - off, -(2**31) + off)
    else:
        raise ValueError(f"unknown int category {cat}")
    return a.astype(np.uint64), b.astype(np.uint64), c


def ref_int(a, b, c, a_signed, b_signed, sat):
    av = a.astype(np.int64)
    bv = b.astype(np.int64)
    if a_signed:
        av = np.where(av >= 128, av - 256, av)
    if b_signed:
        bv = np.where(bv >= 128, bv - 256, bv)
    s = c + (av * bv).sum(axis=1)
    if sat:
        return np.clip(s, -(2**31), 2**31 - 1)
    return ((s + 2**31) % 2**32) - 2**31


def gen_b1_cat(rng, cat, n, words):
    if cat in ("uniform", "boundary"):
        a = rbits(rng, 32, (n, words))
        b = rbits(rng, 32, (n, words))
    elif cat == "density":
        dens = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
        weights = np.uint64(1) << np.arange(32, dtype=np.uint64)
        pa = dens[rng.integers(0, len(dens), size=n)][:, None, None]
        pb = dens[rng.integers(0, len(dens), size=n)][:, None, None]
        a = ((rng.random((n, words, 32)) < pa).astype(np.uint64) * weights).sum(axis=2)
        b = ((rng.random((n, words, 32)) < pb).astype(np.uint64) * weights).sum(axis=2)
    else:
        raise ValueError(f"unknown b1 category {cat}")
    if cat == "boundary":  # popcount sum <= 256
        off = rng.integers(0, 1 << 9, size=n, dtype=np.int64)
        c = np.where(rng.random(n) < 0.5, 2**31 - 1 - off, -(2**31) + off)
    else:
        c = rng.integers(-(2**31), 2**31, size=n, dtype=np.int64)
    return a.astype(np.uint64), b.astype(np.uint64), c


def ref_b1(a, b, c):
    s = c + np.bitwise_count(a & b).astype(np.int64).sum(axis=1)
    return ((s + 2**31) % 2**32) - 2**31


def split_counts(n, cats):
    total = sum(w for _, w in cats)
    counts = [n * w // total for _, w in cats]
    counts[0] += n - sum(counts)
    return counts


def build_op(spec):
    """MMA-Sim object for a float combination."""
    if spec["source"] == "mmasim":
        from mmasim.nv_ptx import MMA, WGMMA

        if spec["api"] == "wgmma":
            return WGMMA("Hopper", spec["shape_and_type"])
        return MMA(spec["arch"], spec["shape_and_type"])
    if spec["source"] == "mmasim-fdpa-inferred":
        # sim.py MMA.__init__ generic branch with the Hopper table (no Ada-only fp8 case).
        from mmasim.arithmetic import fdpa
        from mmasim.nv_ptx import sim

        f16_out = spec["c"] == "f16"
        F = sim._mma_F_bits["Hopper"]
        L_max = sim._mma_L_max_bytes["Hopper"] // BYTES[spec["a"]]
        e_zero = (sim._mma_e_zero_for_fp16 if f16_out else sim._mma_e_zero_for_fp32)[
            "Hopper"
        ]
        rho = "RNE-FP16" if f16_out else "RZ-FP32"
        return fdpa.MMA_T_FDPA(F, rho, L_max, e_zero, spec["a"], spec["b"])
    raise ValueError(f"build_op: no MMA-Sim object for source {spec['source']}")


def op_params(op):
    from mmasim.arithmetic import fdpa

    ar = getattr(op, "arithmetic_op", op)
    if isinstance(ar, fdpa.MMA_T_FDPA):
        return {"F": ar.F, "L_max": ar.L_max, "e_zero": ar.e_zero, "rho": ar.rho}
    return None  # fma.MMA_FMA (f64)


def to_torch(bits, ptx_type):
    from mmasim.nv_ptx.isa_common import nv_torch_dtype

    torch = _W["torch"]
    dt = nv_torch_dtype[ptx_type]
    nb = dt.itemsize
    arr = np.ascontiguousarray(bits.astype(UNSIGNED[nb]).view(SIGNED_VIEW[nb]))
    return torch.from_numpy(arr).view(dt)


def from_torch(t):
    torch = _W["torch"]
    nb = t.dtype.itemsize
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[nb]
    return t.contiguous().view(view).numpy().view(UNSIGNED[nb]).astype(np.uint64)


def run_mmasim(spec, a, b, c):
    torch = _W["torch"]
    key = "op_" + spec["name"]
    if key not in _W:
        _W[key] = build_op(spec)
    op = _W[key]
    with torch.no_grad():
        d = op.dpa(
            to_torch(a, spec["a"]), to_torch(b, spec["b"]), to_torch(c, spec["c"])
        )
    return from_torch(d)


def float_ctx(spec):
    p = spec["params"]
    if spec["kind"] == KIND_F64:
        # f64: the ezero category then probes the binary64 subnormal range
        F, L, ez = 53, spec["K"], -1040
    else:
        F, L, ez = p["F"], p["L_max"], p["e_zero"]
    return {
        "fa": FMTS[spec["a"]],
        "fb": FMTS[spec["b"]],
        "fc": FMTS[spec["c"]],
        "a": spec["a"],
        "b": spec["b"],
        "c": spec["c"],
        "K": spec["K"],
        "L": L,
        "F": F,
        "e_zero": ez,
    }


def init_worker(mmasim_path, threads):
    if mmasim_path not in sys.path:
        sys.path.insert(0, mmasim_path)
    import torch

    torch.set_num_threads(threads)
    _W["torch"] = torch


def work(task):
    spec, chunk_idx, row0, nrows, seed = task
    t0 = time.time()
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(spec["name"].encode()), chunk_idx])
    )
    kind = spec["kind"]
    cats = (
        FLOAT_CATS
        if kind in (KIND_FLOAT, KIND_F64)
        else INT_CATS
        if kind == KIND_INT
        else B1_CATS
    )
    parts = []
    cat_counts = {}
    for (cat, _), cnt in zip(cats, split_counts(nrows, cats), strict=True):
        cat_counts[cat] = cnt
        if cnt == 0:
            continue
        if kind in (KIND_FLOAT, KIND_F64):
            parts.append(gen_float_cat(rng, cat, cnt, float_ctx(spec)))
        elif kind == KIND_INT:
            parts.append(
                gen_int_cat(
                    rng, cat, cnt, spec["K"], spec["a_signed"], spec["b_signed"]
                )
            )
        else:
            parts.append(gen_b1_cat(rng, cat, cnt, spec["units"]))
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    c = np.concatenate([p[2] for p in parts])
    if kind in (KIND_FLOAT, KIND_F64):
        d = run_mmasim(spec, a, b, c)
        c_out, d_out = c, d
    elif kind == KIND_INT:
        d = ref_int(a, b, c, spec["a_signed"], spec["b_signed"], spec["sat"])
        c_out, d_out = (
            c.astype(np.int32).view(np.uint32),
            d.astype(np.int32).view(np.uint32),
        )
    else:
        d = ref_b1(a, b, c)
        c_out, d_out = (
            c.astype(np.int32).view(np.uint32),
            d.astype(np.int32).view(np.uint32),
        )
    ab, bb, cd = spec["a_bytes"], spec["b_bytes"], spec["cd_bytes"]
    out = (
        np.ascontiguousarray(a.astype(f"<u{ab}")).tobytes(),
        np.ascontiguousarray(b.astype(f"<u{bb}")).tobytes(),
        np.ascontiguousarray(c_out.astype(f"<u{cd}")).tobytes(),
        np.ascontiguousarray(d_out.astype(f"<u{cd}")).tobytes(),
    )
    return spec["name"], row0, nrows, out, cat_counts, time.time() - t0


def float_spec(name, kind, api, arch, st, k, a, b, c, source):
    return {
        "name": name,
        "kind": kind,
        "api": api,
        "arch": arch,
        "shape_and_type": st,
        "source": source,
        "K": k,
        "units": k,
        "a": a,
        "b": b,
        "c": c,
        "a_bytes": BYTES[a],
        "b_bytes": BYTES[b],
        "cd_bytes": BYTES[c],
        "a_signed": False,
        "b_signed": False,
        "sat": False,
        "flags": FLAG_INFERRED if source == "mmasim-fdpa-inferred" else 0,
    }


def enumerate_combos():
    from mmasim.nv_ptx import isa_mma, isa_wgmma
    from mmasim.nv_ptx.isa_common import nv_shape_to_mnk

    specs = []
    seen = set()
    for st in isa_wgmma.wgmma_shape_and_type["Hopper"]:
        shape, d, a, b = st.split(".")
        _, n, k = nv_shape_to_mnk(shape)
        if (k, d, a, b) in seen:
            continue
        seen.add((k, d, a, b))
        name = f"wgmma.hopper.k{k}.{d}.{a}.{b}"
        specs.append(
            float_spec(name, KIND_FLOAT, "wgmma", "Hopper", st, k, a, b, d, "mmasim")
        )
    for arch in ("Ampere", "Ada Lovelace", "Hopper"):
        for st in isa_mma.mma_shape_and_type[arch]:
            q = st.split(".")
            if len(q) != 5:
                raise RuntimeError(f"unexpected mma form for {arch}: {st}")
            shape, d, a, b, c = q
            if d != c:
                raise RuntimeError(
                    f"mixed C/D types are outside numerics.h: {arch} {st}"
                )
            _, _, k = nv_shape_to_mnk(shape)
            kind = KIND_F64 if a == "f64" else KIND_FLOAT
            name = f"mma.{ARCH_TAG[arch]}.{st}"
            specs.append(float_spec(name, kind, "mma", arch, st, k, a, b, d, "mmasim"))
    hopper = set(isa_mma.mma_shape_and_type["Hopper"])
    for st in isa_mma.mma_shape_and_type_from_sm_89:
        if st in hopper:
            continue
        shape, d, a, b, _ = st.split(".")
        _, _, k = nv_shape_to_mnk(shape)
        name = f"mma.hopper-inferred.{st}"
        specs.append(
            float_spec(
                name,
                KIND_FLOAT,
                "mma",
                "Hopper",
                st,
                k,
                a,
                b,
                d,
                "mmasim-fdpa-inferred",
            )
        )
    for k in (16, 32):
        for a in ("s8", "u8"):
            for b in ("s8", "u8"):
                for sat in (False, True):
                    name = f"int.k{k}.{a}.{b}.{'satfinite' if sat else 'wrap'}"
                    flags = (FLAG_SAT if sat else 0) | (
                        FLAG_A_SIGNED if a == "s8" else 0
                    )
                    flags |= FLAG_B_SIGNED if b == "s8" else 0
                    specs.append(
                        {
                            "name": name,
                            "kind": KIND_INT,
                            "api": "none",
                            "arch": None,
                            "shape_and_type": None,
                            "source": "python-exact",
                            "K": k,
                            "units": k,
                            "a": a,
                            "b": b,
                            "c": "s32",
                            "a_bytes": 1,
                            "b_bytes": 1,
                            "cd_bytes": 4,
                            "a_signed": a == "s8",
                            "b_signed": b == "s8",
                            "sat": sat,
                            "flags": flags,
                        }
                    )
    for k in (128, 256):
        specs.append(
            {
                "name": f"b1.k{k}.and.popc",
                "kind": KIND_B1,
                "api": "none",
                "arch": None,
                "shape_and_type": None,
                "source": "python-exact",
                "K": k,
                "units": k // 32,
                "a": "b1",
                "b": "b1",
                "c": "s32",
                "a_bytes": 4,
                "b_bytes": 4,
                "cd_bytes": 4,
                "a_signed": False,
                "b_signed": False,
                "sat": False,
                "flags": 0,
            }
        )
    return specs


def header_bytes(spec, count):
    p = spec.get("params") or {"F": 0, "L_max": 0, "e_zero": 0, "rho": None}
    return struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        spec["kind"],
        ETYPE[spec["a"]],
        ETYPE[spec["b"]],
        1 if spec["c"] == "f16" else 0,
        ARCH_CODE[spec["arch"]],
        API_CODE[spec["api"]],
        spec["K"],
        spec["units"],
        spec["a_bytes"],
        spec["b_bytes"],
        spec["cd_bytes"],
        spec["flags"],
        p["F"],
        p["L_max"],
        p["e_zero"],
        RHO_CODE[p["rho"]],
        count,
    )


def section_offsets(spec, count):
    a_off = HEADER_SIZE
    b_off = a_off + count * spec["units"] * spec["a_bytes"]
    c_off = b_off + count * spec["units"] * spec["b_bytes"]
    d_off = c_off + count * spec["cd_bytes"]
    end = d_off + count * spec["cd_bytes"]
    return a_off, b_off, c_off, d_off, end


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("out_dir")
    ap.add_argument(
        "--count", type=int, default=1_000_000, help="dot products per combination"
    )
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument(
        "--procs", type=int, default=min(8, os.cpu_count() or 1), help="max 8"
    )
    ap.add_argument(
        "--threads", type=int, default=1, help="torch/OpenMP threads per process"
    )
    ap.add_argument("--chunk", type=int, default=65536, help="rows per task")
    ap.add_argument("--only", default=None, help="regex on combination names")
    ap.add_argument(
        "--list", action="store_true", help="print the combinations and exit"
    )
    ap.add_argument("--mmasim-path", default=DEFAULT_MMASIM)
    args = ap.parse_args()
    if args.count < 1 or args.chunk < 1 or not 1 <= args.procs <= 8 or args.threads < 1:
        ap.error("--count, --chunk, --threads must be >= 1 and --procs in [1, 8]")

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        ""  # CPU-only MMA-Sim (inherited by the workers)
    )
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    init_worker(args.mmasim_path, args.threads)
    import mmasim.nv_ptx  # noqa: F401  (JIT-builds fma_cpu once, before the workers start)

    specs = enumerate_combos()
    if args.only:
        specs = [s for s in specs if re.search(args.only, s["name"])]
    if not specs:
        sys.exit("no combination selected")
    for s in specs:
        s["params"] = op_params(build_op(s)) if s["kind"] == KIND_FLOAT else None
    if args.list:
        for s in specs:
            print(s["name"], s["source"], s["params"])
        return

    os.makedirs(args.out_dir, exist_ok=True)
    with contextlib.ExitStack() as stack:
        cat_totals, rows_done = generate(args, specs, stack)
    write_metadata(args, specs, cat_totals, rows_done)


def pwrite_all(fd, blob, off):
    view = memoryview(blob)
    while view:
        n = os.pwrite(fd, view, off)
        view = view[n:]
        off += n


def generate(args, specs, stack):
    files = {}
    tasks = []
    for s in specs:
        path = os.path.join(args.out_dir, s["name"] + ".bin")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        stack.callback(os.close, fd)
        os.ftruncate(fd, section_offsets(s, args.count)[-1])
        pwrite_all(fd, header_bytes(s, args.count), 0)
        files[s["name"]] = fd
        for ci, row0 in enumerate(range(0, args.count, args.chunk)):
            tasks.append((s, ci, row0, min(args.chunk, args.count - row0), args.seed))
    by_name = {s["name"]: s for s in specs}
    cat_totals = {s["name"]: {} for s in specs}
    rows_done = {s["name"]: 0 for s in specs}
    total_rows = len(specs) * args.count
    done_rows = 0
    t_start = time.time()
    print(
        f"gen_vectors: {len(specs)} combinations x {args.count} cases, {len(tasks)} tasks, "
        f"{args.procs} processes",
        file=sys.stderr,
        flush=True,
    )
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.procs, initializer=init_worker, initargs=(args.mmasim_path, args.threads)
    ) as pool:
        for name, row0, nrows, out, cats, _ in pool.imap_unordered(work, tasks):
            s = by_name[name]
            fd = files[name]
            offs = section_offsets(s, args.count)
            widths = (
                s["units"] * s["a_bytes"],
                s["units"] * s["b_bytes"],
                s["cd_bytes"],
                s["cd_bytes"],
            )
            for off, width, blob in zip(offs[:4], widths, out, strict=True):
                if len(blob) != nrows * width:
                    raise RuntimeError(
                        f"{name}: section size {len(blob)} != {nrows * width}"
                    )
                pwrite_all(fd, blob, off + row0 * width)
            for k, v in cats.items():
                cat_totals[name][k] = cat_totals[name].get(k, 0) + v
            rows_done[name] += nrows
            done_rows += nrows
            if rows_done[name] == args.count:
                os.fsync(fd)
                el = time.time() - t_start
                eta = el / done_rows * (total_rows - done_rows)
                print(
                    f"gen_vectors: done {name} ({done_rows}/{total_rows} rows, "
                    f"{el:.0f}s elapsed, ~{eta:.0f}s left)",
                    file=sys.stderr,
                    flush=True,
                )
    return cat_totals, rows_done


def write_metadata(args, specs, cat_totals, rows_done):
    index = []
    for s in specs:
        if rows_done[s["name"]] != args.count:
            raise RuntimeError(
                f"{s['name']}: {rows_done[s['name']]} of {args.count} rows written"
            )
        meta = {
            "name": s["name"],
            "source": s["source"],
            "mmasim_commit": MMASIM_COMMIT,
            "arch": s["arch"],
            "api": s["api"],
            "shape_and_type": s["shape_and_type"],
            "kind": ["float", "f64", "int", "b1"][s["kind"]],
            "K": s["K"],
            "units": s["units"],
            "atype": s["a"],
            "btype": s["b"],
            "cdtype": s["c"],
            "a_bytes": s["a_bytes"],
            "b_bytes": s["b_bytes"],
            "cd_bytes": s["cd_bytes"],
            "a_signed": s["a_signed"],
            "b_signed": s["b_signed"],
            "satfinite": s["sat"],
            "params": s["params"],
            "params_inferred": bool(s["flags"] & FLAG_INFERRED),
            "count": args.count,
            "seed": args.seed,
            "chunk": args.chunk,
            "categories": cat_totals[s["name"]],
            "layout": "96-byte header, A[count][units], B[count][units], C[count], D[count]",
            "n_independent": s["api"] == "wgmma",
        }
        with open(os.path.join(args.out_dir, s["name"] + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        index.append(meta)
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("".join(s["name"] + "\n" for s in specs))
    print(
        f"gen_vectors: wrote {len(specs)} combinations to {args.out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
