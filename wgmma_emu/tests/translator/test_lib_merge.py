"""Merging the emulation library PTX (data/lib_sample.ptx, built like runtime/emu_lib.cu)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ptxemu.core import TranslateError, parse_module, translate

USER = """
.version 9.0
.target sm_90a
.address_size 64

.extern .func  (.param .b32 func_retval0) vprintf
(
\t.param .b64 vprintf_param_0,
\t.param .b64 vprintf_param_1
)
;
.global .align 1 .b8 $str[4] = {65, 66, 10, 0};

.visible .entry k(
\t.param .u64 k_param_0
)
{
\t.reg .b32 \t%r<4>;
\t.reg .b64 \t%rd<4>;

\tld.param.u64 \t%rd1, [k_param_0];
\tmov.u32 \t%r1, %tid.x;
\tfakecall.u32 \t%r2, %r1, %rd1;
\tmov.u64 \t%rd2, $str;
\tcvta.global.u64 \t%rd3, %rd2;
\t{ // callseq 0, 0
\t.reg .b32 temp_param_reg;
\t.param .b64 param0;
\tst.param.b64 \t[param0+0], %rd3;
\t.param .b64 param1;
\tst.param.b64 \t[param1+0], 0;
\t.param .b32 retval0;
\tcall.uni (retval0),
\tvprintf,
\t(
\tparam0,
\tparam1
\t);
\tld.param.b32 \t%r3, [retval0+0];
\t} // callseq 0
\tret;
}
"""

LIB_BEGIN = "// ---- wgemu emulation library (pruned) ----"
LIB_END = "// ---- end of wgemu emulation library ----"


def call_rule(func: str = "wgemu_sample_u32", arg_types=("b32", "b64")):
    def rewrite(stmt, ctx):
        d, a, p = stmt.operands
        return ctx.call(func, [(arg_types[0], a), (arg_types[1], p)], ("b32", d))

    return (
        "calls",
        SimpleNamespace(match=lambda s: s.opcode == "fakecall.u32", rewrite=rewrite),
    )


def lib_section(out: str) -> str:
    return out[out.index(LIB_BEGIN) : out.index(LIB_END)]


def test_merge_prunes_renames_and_internalizes(read_data):
    rep: dict = {}
    out = translate(
        USER,
        lib_src=read_data("lib_sample.ptx"),
        rules=[call_rule()],
        report=rep,
        warn_stream=None,
    )
    lib = lib_section(out)
    inner = "_ZN44_INTERNAL_7efb7be2_13_lib_sample_cu_23f9b67c9lib_innerEj"
    info = rep["library"]
    assert info["renamed"] == {"$str": "__wgemulib_$str"}
    assert info["functions_kept"] == [inner, "wgemu_sample_u32", "_Z8lib_tmplILi3EEjj"]
    assert info["functions_total"] == 5
    assert info["externs"] == ["vprintf"]
    assert info["dropped_input_declarations"] == ["vprintf"]
    assert info["version"] == "9.0"
    # unreachable library code is pruned
    for gone in ("wgemu_sample_unused", "wgemu_sample_void", "$str$1"):
        assert gone not in out
    # the library string was renamed away from the user's $str, everywhere in the library
    assert ".global .align 1 .b8 __wgemulib_$str[20]" in lib
    assert "mov.u64 \t%rd3, __wgemulib_$str;" in lib
    assert ".global .align 1 .b8 $str[4]" in out
    # library definitions lose .visible / .weak; vprintf stays the single .extern declaration
    assert ".visible" not in lib and ".weak" not in lib
    assert ".func  (.param .b32 func_retval0) wgemu_sample_u32(" in lib
    assert out.count(".extern .func") == 1
    assert "// wgemu: declaration of vprintf provided by the emulation library" in out
    # library code precedes the user module items, right after the module header
    assert (
        out.index(".address_size 64")
        < out.index(LIB_BEGIN)
        < out.index(".visible .entry k(")
    )
    assert ".version 9.0\n.target sm_80\n" in out
    assert rep["warnings"] == []
    parse_module(out)


def test_call_signature_must_match_library(read_data):
    with pytest.raises(TranslateError, match="call signature of wgemu_sample_u32"):
        translate(
            USER,
            lib_src=read_data("lib_sample.ptx"),
            rules=[call_rule(arg_types=("b32", "b32"))],
            warn_stream=None,
        )


def test_missing_library_function(read_data):
    with pytest.raises(TranslateError, match="no definition of wgemu_nosuch"):
        translate(
            USER,
            lib_src=read_data("lib_sample.ptx"),
            rules=[call_rule("wgemu_nosuch")],
            warn_stream=None,
        )


def test_user_definition_of_library_function_is_rejected(read_data):
    src = (
        USER
        + "\n.func wgemu_sample_void(\n\t.param .b64 wgemu_sample_void_param_0\n)\n{\n\tret;\n}\n"
    )
    with pytest.raises(TranslateError, match="defines wgemu_sample_void"):
        translate(
            src,
            lib_src=read_data("lib_sample.ptx"),
            rules=[call_rule()],
            warn_stream=None,
        )


def test_user_prototype_of_library_function(read_data):
    proto = "\n.extern .func  (.param .b32 func_retval0) wgemu_sample_u32\n(\n\t.param .b32 a,\n\t.param .b64 b\n)\n;\n"
    src = USER.replace("\n.visible .entry k(", proto + "\n.visible .entry k(")
    rep: dict = {}
    out = translate(
        src,
        lib_src=read_data("lib_sample.ptx"),
        rules=[call_rule()],
        report=rep,
        warn_stream=None,
    )
    assert rep["library"]["dropped_input_declarations"] == [
        "vprintf",
        "wgemu_sample_u32",
    ]
    assert (
        "// wgemu: declaration of wgemu_sample_u32 provided by the emulation library"
        in out
    )
    bad = src.replace("\t.param .b64 b\n", "\t.param .b32 b\n")
    with pytest.raises(TranslateError, match="does not match the library definition"):
        translate(
            bad,
            lib_src=read_data("lib_sample.ptx"),
            rules=[call_rule()],
            warn_stream=None,
        )


def test_library_extern_other_than_vprintf(read_data):
    lib = read_data("lib_sample.ptx").replace("vprintf", "wgemu_host_hook")
    with pytest.raises(TranslateError, match="external symbol wgemu_host_hook"):
        translate(USER, lib_src=lib, rules=[call_rule()], warn_stream=None)


def test_library_with_debug_info_is_rejected(read_data):
    lib = read_data("lib_sample.ptx").replace(
        ".address_size 64\n", '.address_size 64\n\t.file\t1 "lib.cu"\n', 1
    )
    with pytest.raises(TranslateError, match="unsupported module item"):
        translate(USER, lib_src=lib, rules=[call_rule()], warn_stream=None)


def test_library_version_raises_output_version(read_data):
    lib = read_data("lib_sample.ptx").replace(".version 9.0", ".version 9.1", 1)
    rep: dict = {}
    out = translate(
        USER, lib_src=lib, rules=[call_rule()], report=rep, warn_stream=None
    )
    assert ".version 9.1\n" in out
    assert rep["version"] == {"input": "9.0", "output": "9.1"}
    assert any(".version raised from 9.0 to 9.1" in w for w in rep["warnings"])
