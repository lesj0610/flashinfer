from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from ptxemu import lexer as lx
from ptxemu.core import (
    Ctx,
    Func,
    Stmt,
    TranslateError,
    load_rules,
    parse_address,
    parse_int,
    parse_module,
    parse_stmt,
    parse_tensor_address,
    parse_vector,
    split_dests,
    translate,
)

# ---------------------------------------------------------------------------------------------
# Operand helpers (PLAN section 5)
# ---------------------------------------------------------------------------------------------


def test_parse_vector():
    assert parse_vector("{%f1, %f2}") == ["%f1", "%f2"]
    assert parse_vector("{ %r1 ,%r2,\n %r3 }") == ["%r1", "%r2", "%r3"]
    assert parse_vector("{}") == []
    with pytest.raises(TranslateError):
        parse_vector("%f1")


@pytest.mark.parametrize(
    "op,want",
    [
        ("[%rd1+16]", ("%rd1", 16)),
        ("[sym]", ("sym", 0)),
        ("[%rd1+-8]", ("%rd1", -8)),
        ("[%rd1-8]", ("%rd1", -8)),
        ("[%r5+0x10]", ("%r5", 16)),
        ("[ %rd2 + 4 ]", ("%rd2", 4)),
        ("%rd1", ("%rd1", 0)),
        ("[64]", ("", 64)),
        ("[__wgemu_shared_bar+8]", ("__wgemu_shared_bar", 8)),
    ],
)
def test_parse_address(op, want):
    assert parse_address(op) == want


@pytest.mark.parametrize("op", ["[%rd5, {%r10, %r4}]", "[%rd1+%r2]", "[%rd1", "[]"])
def test_parse_address_rejects(op):
    with pytest.raises(TranslateError):
        parse_address(op)


def test_parse_tensor_address():
    assert parse_tensor_address("[%rd5, {%r10, %r4}]") == ("%rd5", ["%r10", "%r4"])
    with pytest.raises(TranslateError):
        parse_tensor_address("[%rd5]")


def test_split_dests():
    assert split_dests("%r1|%p1") == ["%r1", "%p1"]
    assert split_dests("%r1") == ["%r1"]
    with pytest.raises(TranslateError):
        split_dests("%r1|")


@pytest.mark.parametrize(
    "s,v", [("0x1F", 31), ("0b101", 5), ("017", 15), ("-3", -3), ("10U", 10), ("0", 0)]
)
def test_parse_int(s, v):
    assert parse_int(s) == v


def test_parse_stmt_shapes():
    st = parse_stmt(
        "@!%p2 wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%f1, %f2, %f3, %f4}, %rd3, %rd4, p, 1, 1, 0, 1",
        7,
    )
    assert st.guard == "@!%p2"
    assert st.opcode == "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"
    assert st.operands == [
        "{%f1, %f2, %f3, %f4}",
        "%rd3",
        "%rd4",
        "p",
        "1",
        "1",
        "0",
        "1",
    ]
    assert st.lineno == 7 and st.in_asm is False
    assert parse_stmt("ret").operands == []
    call = parse_stmt("call.uni (retval0), \n\t_Z9elect_onev, \n\t(\n\t)")
    assert call.opcode == "call.uni"
    assert call.operands == ["(retval0)", "_Z9elect_onev", "()"]
    # PLAN section 5 positional construction
    s2 = Stmt(None, "ret", [], "ret", 3)
    assert s2.in_asm is False


# ---------------------------------------------------------------------------------------------
# Module model
# ---------------------------------------------------------------------------------------------


def test_parse_module_sample(read_data):
    mod = parse_module(read_data("sample_sm90.ptx"))
    assert (mod.version, mod.target, mod.address_size) == ("9.0", "sm_90a", "64")
    funcs = {it.name: it.func for it in mod.items if it.kind == "func"}
    assert set(funcs) == {"_Z9elect_onev", "_Z13sample_kernelPfmmmi"}
    k = funcs["_Z13sample_kernelPfmmmi"]
    assert k.is_entry and k.linkage == ".visible"
    assert k.params[0] == (".u64", "_Z13sample_kernelPfmmmi_param_0")
    assert k.params[1] == (".align 8 .b8 [8]", "_Z13sample_kernelPfmmmi_param_1")
    assert len(k.params) == 5
    assert k.directives == [
        ".maxntid 128, 1, 1",
        ".minnctapersm 1",
        ".explicitcluster",
        ".reqnctapercluster 1, 1, 1",
    ]
    e = funcs["_Z9elect_onev"]
    assert not e.is_entry and e.linkage == "" and e.rets == [(".b32", "func_retval0")]
    wg = [s for s in k.body if isinstance(s, Stmt) and s.opcode.startswith("wgmma")]
    assert [s.opcode.split(".")[1] for s in wg] == [
        "fence",
        "mma_async",
        "commit_group",
        "wait_group",
    ]
    assert all(s.in_asm for s in wg)
    kinds = {it.name: it.kind for it in mod.items if it.name}
    assert (
        kinds["vprintf"] == "func_decl"
        and kinds["smem"] == "var"
        and kinds["$str"] == "var"
    )


def test_parse_module_requires_header():
    with pytest.raises(TranslateError):
        parse_module(".visible .entry k()\n{\n\tret;\n}\n")


# ---------------------------------------------------------------------------------------------
# Ctx helpers
# ---------------------------------------------------------------------------------------------


def make_ctx(mode: str = "exact") -> Ctx:
    body: list = [
        ".reg .pred \t%p<3>;",
        ".reg .b32 \t%r<8>;",
        ".reg .b64 \t%rd<8>;",
        ".shared .align 8 .u64 bar;",
        ".local .align 8 .b8 \tdepot[16];",
    ]
    func = Func(name="f", is_entry=True, params=[(".u64", "f_param_0")], body=body)
    return Ctx(mode=mode, src_target="sm_90a", func=func)


def test_ctx_registers_and_buffers():
    ctx = make_ctx()
    assert ctx.mode == "exact" and ctx.src_target == "sm_90a" and ctx.func.name == "f"
    assert ctx.new_reg("b32") == "%wgemu_b32_0"
    assert ctx.new_reg("b32") == "%wgemu_b32_1"
    assert ctx.new_reg("pred") == "%wgemu_pred_0"
    assert ctx.new_reg(".u64") == "%wgemu_u64_0"
    with pytest.raises(TranslateError):
        ctx.new_reg("x32")
    assert ctx.local_buf("acc", 64) == "__wgemu_local_acc"
    assert ctx.local_buf("acc", 128, 32) == "__wgemu_local_acc"
    assert ctx.shared_var("bar.x", 8, 8) == "__wgemu_shared_bar_x"
    with pytest.raises(TranslateError):
        ctx.local_buf("bad", 8, 3)
    decls = ctx._decl_lines()
    assert ".reg .b32 \t%wgemu_b32_<2>;" in decls
    assert ".reg .pred \t%wgemu_pred_<1>;" in decls
    assert ".local .align 32 .b8 \t__wgemu_local_acc[128];" in decls
    assert (
        ctx.reg_type("%r3") == "b32"
        and ctx.reg_bits("%rd7") == 64
        and ctx.reg_bits("%p1") == 1
    )
    assert ctx.reg_type("%rd8") is None  # outside %rd<8>
    assert ctx.reg_type("%wgemu_b32_1") == "b32"
    assert ctx.symbol_space("bar") == "shared"
    assert ctx.symbol_space("depot") == "local"
    assert ctx.symbol_space("f_param_0") == "param"
    assert ctx.symbol_space("__wgemu_shared_bar_x") == "shared"
    st = ctx.func_state()
    st["x"] = 1
    assert ctx.func_state() == {"x": 1}


def test_ctx_to_shared_u32():
    ctx = make_ctx()
    lines, r = ctx.to_shared_u32("[%rd1+16]", "shared::cta")
    assert lines == [
        "cvt.u32.u64 \t%wgemu_b32_0, %rd1;",
        "add.s32 \t%wgemu_b32_1, %wgemu_b32_0, 16;",
    ]
    assert r == "%wgemu_b32_1"
    assert ctx.to_shared_u32("[%r1]", "shared") == ([], "%r1")
    lines, r = ctx.to_shared_u32("[%rd2]", None)
    assert lines == [
        "cvta.to.shared.u64 \t%wgemu_b64_0, %rd2;",
        "cvt.u32.u64 \t%wgemu_b32_2, %wgemu_b64_0;",
    ]
    assert r == "%wgemu_b32_2"
    lines, r = ctx.to_shared_u32("[bar+8]", "shared::cluster")
    assert lines == [
        "mov.u32 \t%wgemu_b32_3, bar;",
        "add.s32 \t%wgemu_b32_4, %wgemu_b32_3, 8;",
    ]
    with pytest.raises(TranslateError):
        ctx.to_shared_u32("[%rd1]", "global")
    with pytest.raises(TranslateError):
        ctx.to_shared_u32("[nosuch]", "shared")
    with pytest.raises(TranslateError):
        ctx.to_shared_u32("[%r1]", None)  # generic addresses are 64-bit
    with pytest.raises(TranslateError):
        ctx.to_shared_u32("[%rd1]", "shared::weird")


def test_ctx_to_generic_u64_and_local():
    ctx = make_ctx()
    lines, r = ctx.to_generic_u64("[%r1+4]", "shared")
    assert lines == [
        "cvt.u64.u32 \t%wgemu_b64_0, %r1;",
        "cvta.shared.u64 \t%wgemu_b64_1, %wgemu_b64_0;",
        "add.s64 \t%wgemu_b64_2, %wgemu_b64_1, 4;",
    ]
    assert r == "%wgemu_b64_2"
    assert ctx.to_generic_u64("[%rd1]", None) == ([], "%rd1")
    lines, r = ctx.to_generic_u64("[%rd3]", "global")
    assert lines == ["cvta.global.u64 \t%wgemu_b64_3, %rd3;"] and r == "%wgemu_b64_3"
    lines, r = ctx.to_generic_u64("[depot]", "local")
    assert lines == [
        "mov.u64 \t%wgemu_b64_4, depot;",
        "cvta.local.u64 \t%wgemu_b64_5, %wgemu_b64_4;",
    ]
    with pytest.raises(TranslateError):
        ctx.to_generic_u64("[%r1]", "global")
    lines, r = ctx.generic_addr_of_local("depot")
    assert lines == [
        "mov.u64 \t%wgemu_b64_6, depot;",
        "cvta.local.u64 \t%wgemu_b64_7, %wgemu_b64_6;",
    ]
    assert r == "%wgemu_b64_7"


def test_ctx_error_is_translate_error():
    ctx = make_ctx()
    with pytest.raises(TranslateError, match="boom"):
        ctx.error("boom")
    with pytest.raises(TranslateError):
        Ctx(mode="bogus")


# ---------------------------------------------------------------------------------------------
# translate(): target, cluster directives, round trip
# ---------------------------------------------------------------------------------------------


def instr_texts(ptx: str) -> list[str]:
    return [
        " ".join(e.text.split())
        for e in lx.lex(ptx)
        if e.kind == lx.STATEMENT and not e.text.startswith(".")
    ]


def test_round_trip_without_rules(read_data):
    src = read_data("sample_sm90.ptx")
    rep: dict = {}
    out = translate(src, rules=[], report=rep, warn_stream=None)
    assert ".target sm_80\n" in out and "sm_90a" not in out
    assert ".explicitcluster" not in out and ".reqnctapercluster" not in out
    assert ".maxntid 128, 1, 1" in out and ".minnctapersm 1" in out
    assert instr_texts(out) == instr_texts(src)
    k = next(f for f in rep["functions"] if f["name"] == "_Z13sample_kernelPfmmmi")
    assert k["cluster"] == {"explicitcluster": True, "reqnctapercluster": [1, 1, 1]}
    assert rep["src_target"] == "sm_90a" and rep["dst_target"] == "sm_80"
    assert rep["rules"] == {"loaded": [], "missing": []}
    again = translate(out, rules=[], warn_stream=None)
    assert instr_texts(again) == instr_texts(src)


def test_round_trip_lineinfo(read_data):
    src = read_data("sample_sm90_rdc_lineinfo.ptx")
    out = translate(src, rules=[], warn_stream=None)
    loc = re.compile(r"^\s*\.loc\s", re.M)
    assert len(loc.findall(out)) == len(loc.findall(src)) == 30
    assert '.file\t1 "' in out
    assert ".section\t.debug_str" in out and "$L__info_string0:" in out
    assert instr_texts(out) == instr_texts(src)


def test_target_options_and_rejections(read_data):
    src = read_data("sample_sm90.ptx")
    out = translate(
        src.replace(".target sm_90a", ".target sm_90a, debug"),
        rules=[],
        warn_stream=None,
    )
    assert ".target sm_80, debug\n" in out
    with pytest.raises(TranslateError, match="newer than sm_90a"):
        translate(
            src.replace(".target sm_90a", ".target sm_100a"), rules=[], warn_stream=None
        )


def test_cluster_size_above_one(read_data):
    src = read_data("sample_sm90.ptx").replace(
        ".reqnctapercluster 1, 1, 1", ".reqnctapercluster 2, 1, 1"
    )
    with pytest.raises(TranslateError, match="reqnctapercluster"):
        translate(src, rules=[], warn_stream=None)
    rep: dict = {}
    out = translate(src, rules=[], allow_cluster=True, report=rep, warn_stream=None)
    assert ".reqnctapercluster" not in out
    assert any("reqnctapercluster" in w for w in rep["warnings"])


# ---------------------------------------------------------------------------------------------
# Rule driver
# ---------------------------------------------------------------------------------------------

SMALL = """
.version 8.7
.target sm_90a
.address_size 64

.visible .entry k(
\t.param .u64 k_param_0
)
{
\t.reg .pred \t%p<3>;
\t.reg .b32 \t%r<8>;
\t.reg .b64 \t%rd<4>;
\t.reg .b16 \tq;
\t.reg .f32 \t%f<5>;

\tld.param.u64 \t%rd1, [k_param_0];
\tmov.u32 \t%r1, %tid.x;
\tsetp.eq.s32 \t%p1, %r1, 0;
\t// begin inline asm
\t@%p1 fakeop.sync.aligned \t[%rd1+16], %r1;
\t// end inline asm
\t// begin inline asm
\t{
\t.reg .pred q;
\tfakeop.sync.aligned \t[%rd1], %r2;
\t}
\t// end inline asm
\tmov.u32 \t%r3, %cluster_ctarank;
\tmov.b64 \t%rd2, {%r4, %cluster_ctarank};
\t@!%p2 fakeop.sync.aligned \t[%rd1], %r5;
\tret;
}
"""


def fake_rules(record: dict):
    def scan(func, ctx):
        record.setdefault("scanned", []).append(func.name)

    def match(stmt):
        return stmt.opcode == "fakeop.sync.aligned"

    def rewrite(stmt, ctx):
        st = ctx.func_state()
        st["n"] = st.get("n", 0) + 1
        record.setdefault("q", []).append(ctx.reg_type("q"))
        record.setdefault("bits", []).append(ctx.reg_bits(stmt.operands[1]))
        r = ctx.new_reg("b32")
        buf = ctx.local_buf("fk", 32, 8)
        sh = ctx.shared_var("fk", 16, 8)
        return [f"mov.u32 \t{r}, {sh};", f"// fake {stmt.operands[0]} {buf}"]

    def prologue(ctx):
        return [f"// prologue n={ctx.func_state().get('n')}"]

    fake = SimpleNamespace(
        scan=scan, match=match, rewrite=rewrite, function_prologue=prologue
    )

    def rewrite_operand(op, stmt, ctx):
        if op != "%cluster_ctarank":
            return None
        r = ctx.new_reg("b32")
        return [f"mov.u32 \t{r}, 0;"], r

    ops = SimpleNamespace(
        match=lambda s: False, rewrite=lambda s, c: [], rewrite_operand=rewrite_operand
    )
    return [("fake", fake), ("ops", ops)]


def test_rule_driver_end_to_end():
    record: dict = {}
    rep: dict = {}
    out = translate(SMALL, rules=fake_rules(record), report=rep, warn_stream=None)
    assert record["scanned"] == ["k"]
    assert record["bits"] == [32, 32, 32]
    # the inline-asm scope of the second statement shadows the function-scope q
    assert record["q"] == ["b16", "pred", "b16"]
    # guard wrapping (core negates the guard)
    assert re.search(
        r"// wgemu fake: @%p1 fakeop\.sync\.aligned \[%rd1\+16\], %r1\n"
        r"\t@!%p1 bra \t\$WGEMU_SKIP_0;\n"
        r"\tmov\.u32 \t%wgemu_b32_0, __wgemu_shared_fk;\n"
        r"\t// fake \[%rd1\+16\] __wgemu_local_fk\n"
        r"\$WGEMU_SKIP_0:\n",
        out,
    )
    assert re.search(r"\t@%p2 bra \t\$WGEMU_SKIP_1;\n", out)
    # unguarded rewrite is not wrapped
    assert re.search(
        r"// wgemu fake: fakeop\.sync\.aligned \[%rd1\], %r2\n\tmov\.u32 \t%wgemu_b32_1, ",
        out,
    )
    # operand rewrite: top level and inside a vector operand
    assert "\tmov.u32 \t%wgemu_b32_2, 0;\n\tmov.u32 \t%r3, %wgemu_b32_2;\n" in out
    assert (
        "\tmov.u32 \t%wgemu_b32_3, 0;\n\tmov.b64 \t%rd2, {%r4, %wgemu_b32_3};\n" in out
    )
    assert "%cluster_ctarank" not in out
    # declarations and prologue after the leading declarations, before the first instruction
    i_decl = out.index("\t.reg .b32 \t%wgemu_b32_<5>;")
    i_local = out.index("\t.local .align 8 .b8 \t__wgemu_local_fk[32];")
    i_pro = out.index("\t// prologue n=3")
    i_first = out.index("\tld.param.u64")
    assert out.index("\t.reg .f32 \t%f<5>;") < i_decl < i_local < i_pro < i_first
    # module-scope shared variable before the function
    assert out.index(".shared .align 8 .b8 __wgemu_shared_fk[16];") < out.index(
        ".visible .entry k("
    )
    assert rep["rewritten_opcodes"] == {"fakeop.sync.aligned": 3}
    assert rep["rewritten_operands"] == {"%cluster_ctarank": 2}
    assert rep["shared_vars"] == {"__wgemu_shared_fk": {"bytes": 16, "align": 8}}
    assert rep["functions"][0]["local_bufs"] == {
        "__wgemu_local_fk": {"bytes": 32, "align": 8}
    }
    # the result is still a well-formed module
    parse_module(out)


def test_first_matching_rule_wins():
    a = SimpleNamespace(
        match=lambda s: s.opcode.startswith("fakeop"),
        rewrite=lambda s, c: ["// from a"],
    )
    b = SimpleNamespace(
        match=lambda s: s.opcode.startswith("fakeop"),
        rewrite=lambda s, c: ["// from b"],
    )
    out = translate(SMALL, rules=[("a", a), ("b", b)], warn_stream=None)
    assert "// from a" in out and "// from b" not in out


def test_rule_misbehaviour_is_an_error():
    bad_ret = SimpleNamespace(
        match=lambda s: s.opcode.startswith("fakeop"), rewrite=lambda s, c: None
    )
    with pytest.raises(TranslateError, match="list of PTX lines"):
        translate(SMALL, rules=[("bad", bad_ret)], warn_stream=None)

    def boom(stmt, ctx):
        raise ValueError("boom")

    crash = SimpleNamespace(match=lambda s: s.opcode.startswith("fakeop"), rewrite=boom)
    with pytest.raises(TranslateError, match="boom"):
        translate(SMALL, rules=[("crash", crash)], warn_stream=None)

    def refuse(stmt, ctx):
        ctx.error("unsupported variant")

    err = SimpleNamespace(match=lambda s: s.opcode.startswith("fakeop"), rewrite=refuse)
    with pytest.raises(TranslateError, match="unsupported variant"):
        translate(SMALL, rules=[("err", err)], warn_stream=None)
    with pytest.raises(TranslateError, match="match"):
        translate(
            SMALL,
            rules=[("nomatch", SimpleNamespace(rewrite=lambda s, c: []))],
            warn_stream=None,
        )


def test_load_rules_missing_and_invalid():
    warned: list[str] = []
    assert load_rules(["ptxemu.no_such_rules_module_xyz"], warn=warned.append) == []
    assert warned and "not found" in warned[0]
    with pytest.raises(TranslateError, match="match"):
        load_rules(["ptxemu.lexer"])  # importable, but not a rule module


def test_reserved_prefix_in_input_is_rejected():
    src = SMALL.replace(
        "\t.reg .f32 \t%f<5>;",
        "\t.reg .f32 \t%f<5>;\n\t.local .align 4 .b8 \t__wgemu_local_x[4];",
    )
    with pytest.raises(TranslateError, match="reserved"):
        translate(src, rules=[], warn_stream=None)
