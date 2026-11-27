"""Independent exact reference for include/wgmma_emu/cvt.h (fork-only). Owner: T2.

Every finite value of a destination format is enumerated as a ``fractions.Fraction``; a
source value is located among them with ``bisect`` and rounded by comparing exact distances.
Nothing here shares code or bit tricks with cvt.h (which rounds a shifted integer mantissa),
so agreement of the two is evidence for both.

Conventions (PTX ISA 9.5 cvt, 9.7.10.24, and the cvt.h header; [inference] where PTX is
silent, listed in the T2 report):

* NaN results are canonical: f32/tf32 0x7FFFFFFF, f16/bf16 0x7FFF, fp8 0x7F;
* ``relu``: NaN -> canonical NaN, any negative result including -0 -> +0;
* ``satfinite``: +-inf and overflow -> +-MAX_NORM, NaN stays NaN;
* overflow without ``satfinite``: formats without inf (e4m3) give NaN; otherwise .rz gives
  MAX_NORM, .rm gives -inf / +MAX_NORM, .rp gives +inf / -MAX_NORM, .rn / .rna give inf;
* rounding is done once with an unbounded exponent, so a value above MAX_NORM that rounds
  down to MAX_NORM is not an overflow (IEEE 754 7.4);
* ties: .rn to the even code, .rna away from zero.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from fractions import Fraction

RN, RZ, RM, RP, RNA = 0, 1, 2, 3, 4
RND_NAMES = {"rn": RN, "rz": RZ, "rm": RM, "rp": RP, "rna": RNA}


@dataclass
class Format:
    name: str
    ebits: int
    mbits: int
    ieee: bool  # all-ones exponent is inf / NaN; else only the all-ones code is NaN (e4m3)
    shift: int  # position of the magnitude inside the storage word (tf32: 13)
    sign: int
    nan: int
    enumerate: bool = True  # f32 is too large to enumerate; it is only an exact target
    values: list[Fraction] = field(default_factory=list, repr=False)
    maxcode: int = 0
    above: Fraction = Fraction(0)  # the value one quantum above MAX_NORM

    def __post_init__(self) -> None:
        if not self.enumerate:
            return
        bias = (1 << (self.ebits - 1)) - 1
        emax_field = (1 << self.ebits) - 1
        vals = []
        code = 0
        while True:
            e, m = code >> self.mbits, code & ((1 << self.mbits) - 1)
            if self.ieee and e == emax_field:
                break
            if not self.ieee and e == emax_field and m == (1 << self.mbits) - 1:
                break
            if e == 0:
                v = Fraction(m) * Fraction(2) ** (1 - bias - self.mbits)
            else:
                v = Fraction((1 << self.mbits) + m) * Fraction(2) ** (
                    e - bias - self.mbits
                )
            vals.append(v)
            code += 1
        self.values = vals
        self.maxcode = len(vals) - 1
        e_top = self.maxcode >> self.mbits
        self.above = vals[-1] + Fraction(2) ** (max(e_top, 1) - bias - self.mbits)

    @property
    def inf(self) -> int:
        return ((1 << self.ebits) - 1) << self.mbits


F32 = Format("f32", 8, 23, True, 0, 0x80000000, 0x7FFFFFFF, enumerate=False)
TF32 = Format("tf32", 8, 10, True, 13, 0x80000000, 0x7FFFFFFF)
F16 = Format("f16", 5, 10, True, 0, 0x8000, 0x7FFF)
BF16 = Format("bf16", 8, 7, True, 0, 0x8000, 0x7FFF)
E4M3 = Format("e4m3", 4, 3, False, 0, 0x80, 0x7F)
E5M2 = Format("e5m2", 5, 2, True, 0, 0x80, 0x7F)
FORMATS = {f.name: f for f in (TF32, F16, BF16, E4M3, E5M2)}  # enumerated targets


# ---------------------------------------------------------------------------------------------
# Decoding: ("nan" | "inf" | "fin", negative, Fraction magnitude)
# ---------------------------------------------------------------------------------------------


def decode(bits: int, ebits: int, mbits: int) -> tuple[str, bool, Fraction]:
    bias = (1 << (ebits - 1)) - 1
    neg = bool((bits >> (ebits + mbits)) & 1)
    e = (bits >> mbits) & ((1 << ebits) - 1)
    m = bits & ((1 << mbits) - 1)
    if e == (1 << ebits) - 1:
        return ("nan" if m else "inf"), neg, Fraction(0)
    if e == 0:
        return "fin", neg, Fraction(m) * Fraction(2) ** (1 - bias - mbits)
    return "fin", neg, Fraction((1 << mbits) | m) * Fraction(2) ** (e - bias - mbits)


def decode_f32(x: int) -> tuple[str, bool, Fraction]:
    return decode(x & 0xFFFFFFFF, 8, 23)


def decode_f16(h: int) -> tuple[str, bool, Fraction]:
    return decode(h & 0xFFFF, 5, 10)


def decode_bf16(b: int) -> tuple[str, bool, Fraction]:
    return decode(b & 0xFFFF, 8, 7)


def decode_f64(d: int) -> tuple[str, bool, Fraction]:
    return decode(d & 0xFFFFFFFFFFFFFFFF, 11, 52)


def decode_e4m3(v: int) -> tuple[str, bool, Fraction]:
    v &= 0xFF
    neg = bool(v & 0x80)
    if v & 0x7F == 0x7F:
        return "nan", neg, Fraction(0)
    return "fin", neg, E4M3.values[v & 0x7F]


def decode_e5m2(v: int) -> tuple[str, bool, Fraction]:
    return decode(v & 0xFF, 5, 2)


# ---------------------------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------------------------


def _overflow(f: Format, neg: bool, r: int, satfinite: bool) -> int:
    sgn = f.sign if neg else 0
    maxw = f.maxcode << f.shift
    if satfinite:
        return sgn | maxw
    if not f.ieee:
        return f.nan
    if r == RZ:
        to_inf = False
    elif r == RM:
        to_inf = neg
    elif r == RP:
        to_inf = not neg
    else:
        to_inf = True
    return sgn | ((f.inf << f.shift) if to_inf else maxw)


def round_value(f: Format, neg: bool, x: Fraction, r: int, satfinite: bool) -> int:
    """Round the exact value (-1)^neg * x (x >= 0) into format f once."""
    sgn = f.sign if neg else 0
    if x == 0:
        return sgn
    vals = f.values
    if x >= f.above:  # rounds (with an unbounded exponent) to at least f.above
        return _overflow(f, neg, r, satfinite)
    i = bisect.bisect_right(vals, x) - 1  # vals[i] <= x
    if vals[i] == x:
        return sgn | (i << f.shift)
    lo_code, hi_code = i, i + 1  # hi_code == maxcode + 1 means "f.above" (overflow)
    lo_v = vals[lo_code]
    hi_v = vals[hi_code] if hi_code <= f.maxcode else f.above
    toward_zero = r == RZ or (r == RM and not neg) or (r == RP and neg)
    away = (r == RM and neg) or (r == RP and not neg)
    if toward_zero:
        code = lo_code
    elif away:
        code = hi_code
    else:
        dl, dh = x - lo_v, hi_v - x
        if dl < dh:
            code = lo_code
        elif dh < dl or r == RNA:  # nearer to hi, or a tie under .rna
            code = hi_code
        else:
            code = lo_code if lo_code % 2 == 0 else hi_code
    if code > f.maxcode:
        return _overflow(f, neg, r, satfinite)
    return sgn | (code << f.shift)


def convert(
    dec: tuple[str, bool, Fraction], f: Format, r: int, relu: bool, satfinite: bool
) -> int:
    kind, neg, x = dec
    if kind == "nan":
        return f.nan
    if relu and neg:
        return 0
    if kind == "inf":
        if satfinite:
            return (f.sign if neg else 0) | (f.maxcode << f.shift)
        if not f.ieee:
            return f.nan
        return (f.sign if neg else 0) | (f.inf << f.shift)
    return round_value(f, neg, x, r, satfinite)


# ---------------------------------------------------------------------------------------------
# cvt.h entry points
# ---------------------------------------------------------------------------------------------


def _fp8(e5m2: bool) -> Format:
    return E5M2 if e5m2 else E4M3


def f32_to_fp8(x: int, e5m2: bool, relu: bool, satfinite: bool) -> int:
    return convert(decode_f32(x), _fp8(e5m2), RN, relu, satfinite)


def f16_to_fp8(h: int, e5m2: bool, relu: bool, satfinite: bool) -> int:
    return convert(decode_f16(h), _fp8(e5m2), RN, relu, satfinite)


def fp8_to_f16(v: int, e5m2: bool, relu: bool) -> int:
    dec = decode_e5m2(v) if e5m2 else decode_e4m3(v)
    return convert(dec, F16, RN, relu, False)


def f32x2_to_fp8x2(a: int, b: int, e5m2: bool, relu: bool, satfinite: bool) -> int:
    return (f32_to_fp8(a, e5m2, relu, satfinite) << 8) | f32_to_fp8(
        b, e5m2, relu, satfinite
    )


def f16x2_to_fp8x2(a: int, e5m2: bool, relu: bool, satfinite: bool) -> int:
    return (f16_to_fp8((a >> 16) & 0xFFFF, e5m2, relu, satfinite) << 8) | f16_to_fp8(
        a & 0xFFFF, e5m2, relu, satfinite
    )


def fp8x2_to_f16x2(a: int, e5m2: bool, relu: bool) -> int:
    return (fp8_to_f16((a >> 8) & 0xFF, e5m2, relu) << 16) | fp8_to_f16(
        a & 0xFF, e5m2, relu
    )


def f32_to_tf32(x: int, r: int, relu: bool, satfinite: bool) -> int:
    return convert(decode_f32(x), TF32, r, relu, satfinite)


def f32_to_bf16(x: int, r: int) -> int:
    return convert(decode_f32(x), BF16, r, False, False)


def f16_to_bf16(h: int, r: int) -> int:
    return convert(decode_f16(h), BF16, r, False, False)


def f64_to_bf16(d: int, r: int) -> int:
    return convert(decode_f64(d), BF16, r, False, False)


def int_to_bf16(v: int, is_signed: bool, bits: int, r: int) -> int:
    v &= (1 << bits) - 1
    if is_signed and v >> (bits - 1):
        v -= 1 << bits
    return round_value(BF16, v < 0, Fraction(abs(v)), r, False)


def f32_to_f16(x: int, r: int) -> int:
    return convert(decode_f32(x), F16, r, False, False)


def bf16_to_f16(b: int, r: int) -> int:
    return convert(decode_bf16(b), F16, r, False, False)


def _floor_log2(x: Fraction) -> int:
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return e


def encode_exact(ebits: int, mbits: int, neg: bool, x: Fraction) -> int:
    """Bits of the IEEE (ebits, mbits) value (-1)^neg * x, which must be representable."""
    bias = (1 << (ebits - 1)) - 1
    s = 1 << (ebits + mbits) if neg else 0
    if x == 0:
        return s
    e = _floor_log2(x)
    if e < 1 - bias:
        m = x / Fraction(2) ** (1 - bias - mbits)
        assert m.denominator == 1, "not representable"
        return s | int(m)
    m = x / Fraction(2) ** (e - mbits) - (1 << mbits)
    assert m.denominator == 1 and e + bias < (1 << ebits) - 1, "not representable"
    return s | ((e + bias) << mbits) | int(m)


def f16_to_f32(h: int) -> int:
    kind, neg, x = decode_f16(h)
    if kind == "nan":
        return F32.nan
    if kind == "inf":
        return (F32.sign if neg else 0) | F32.inf
    return encode_exact(8, 23, neg, x)


def bf16_to_f32(b: int) -> int:
    # Every bf16 value is an f32 value; NaN keeps its payload (cvt.h documents the shift).
    kind, neg, x = decode_bf16(b)
    if kind == "nan":
        return (b & 0xFFFF) << 16
    if kind == "inf":
        return (F32.sign if neg else 0) | F32.inf
    return encode_exact(8, 23, neg, x)


def f32_ftz(x: int) -> int:
    kind, neg, v = decode_f32(x)
    if kind == "fin" and v < Fraction(2) ** -126:  # below the smallest normal f32
        return F32.sign if neg else 0
    return x & 0xFFFFFFFF


def f32_to_f64(x: int) -> int:
    kind, neg, v = decode_f32(x)
    if kind == "nan":
        return 0x7FFFFFFFFFFFFFFF
    if kind == "inf":
        return ((1 << 63) if neg else 0) | 0x7FF0000000000000
    return encode_exact(11, 52, neg, v)


def round_to_fmt(
    fmt: str, neg: bool, mag: int, exp2: int, r: int, satfinite: bool
) -> int:
    """Reference for cvt::round_to_fmt: (-1)^neg * mag * 2^exp2 rounded into fmt."""
    x = Fraction(mag) * Fraction(2) ** exp2
    return round_value(FORMATS[fmt], neg, x, r, satfinite)
