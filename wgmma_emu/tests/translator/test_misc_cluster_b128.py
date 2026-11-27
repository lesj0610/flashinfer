"""rules_misc: %is_explicit_cluster only where the entry's directives decide it, and atom.sys.b128
rejected (CPU only, no library PTX needed)."""

from __future__ import annotations

import re

import pytest
from ptxemu import rules_misc
from ptxemu.core import TranslateError, translate

HEAD = ".version 8.0\n.target sm_90a\n.address_size 64\n\n"

CLUSTER = (
    HEAD + ".visible .entry k(\n\t.param .u64 k_param_0\n)\n{dirs}{{\n"
    "\t.reg .pred \t%p<2>;\n\t.reg .b32 \t%r<2>;\n\t.reg .b64 \t%rd<3>;\n"
    "\tld.param.u64 \t%rd1, [k_param_0];\n\tcvta.to.global.u64 \t%rd2, %rd1;\n"
    "\tmov.pred \t%p1, %is_explicit_cluster;\n\tselp.u32 \t%r1, 1, 0, %p1;\n"
    "\tst.global.u32 \t[%rd2], %r1;\n\tret;\n}}\n"
)


def run(src: str) -> str:
    return translate(src, rules=[rules_misc], warn_stream=None)


@pytest.mark.parametrize(
    "dirs",
    [".explicitcluster\n.reqnctapercluster 1, 1, 1\n", ".reqnctapercluster 1, 1, 1\n"],
)
def test_is_explicit_cluster_static(dirs):
    out = run(CLUSTER.format(dirs=dirs))
    assert "%is_explicit_cluster" not in out
    assert re.search(r"mov\.u32\s+\S+, 1;", out)


@pytest.mark.parametrize("dirs", ["", ".maxclusterrank 1\n", ".maxntid 128, 1, 1\n"])
def test_is_explicit_cluster_launch_dependent(dirs):
    # A runtime cluster attribute (even 1x1x1) would make it true; the shim drops it.
    with pytest.raises(TranslateError, match="is_explicit_cluster in an entry without"):
        run(CLUSTER.format(dirs=dirs))


def test_is_explicit_cluster_in_func():
    src = (
        HEAD
        + ".func (.param .b32 r) f()\n{\n\t.reg .pred \t%p<2>;\n\t.reg .b32 \t%r<2>;\n"
        "\tmov.pred \t%p1, %is_explicit_cluster;\n\tselp.u32 \t%r1, 1, 0, %p1;\n"
        "\tst.param.b32 \t[r], %r1;\n\tret;\n}\n"
    )
    with pytest.raises(TranslateError, match="outside an entry"):
        run(src)


@pytest.mark.parametrize(
    "op",
    [
        "atom.sys.global.cas.b128 \t%q1, [%rd1], %q2, %q3;",
        "atom.relaxed.sys.global.exch.b128 \t%q1, [%rd1], %q2;",
    ],
)
def test_atom_sys_b128_rejected(op):
    src = (
        HEAD + ".visible .entry k(\n\t.param .u64 k_param_0\n)\n{\n"
        "\t.reg .b64 \t%rd<2>;\n\t.reg .b128 \t%q<4>;\n"
        "\tld.param.u64 \t%rd1, [k_param_0];\n"
        "\tmov.b128 \t%q2, {%rd1, %rd1};\n\tmov.b128 \t%q3, {%rd1, %rd1};\n"
        f"\t{op}\n\tret;\n}}\n"
    )
    with pytest.raises(TranslateError, match="atom.sys.*b128.*unsupported"):
        run(src)
