"""Ctx.call must produce the callseq block nvcc 13.0 emits for the same call (data/callseq_sample.*)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from ptxemu.core import Ctx, Func, TranslateError, parse_module, translate

_BLOCK_RE = re.compile(
    r"^\t\{ // callseq (\w+), 0\n.*?^\t\} // callseq \1$", re.M | re.S
)


def _norm(text: str) -> str:
    return re.sub(r"callseq \w+", "callseq N", text)


def nvcc_blocks(read_data) -> list[str]:
    blocks = [m.group(0) for m in _BLOCK_RE.finditer(read_data("callseq_sample.ptx"))]
    assert len(blocks) == 3
    return blocks


def make_ctx() -> Ctx:
    body: list = [".reg .b32 \t%r<2>;", ".reg .b64 \t%rd<4>;"]
    return Ctx(
        func=Func(name="kc", is_entry=True, params=[(".u64", "kc_param_0")], body=body)
    )


def test_call_lines_match_nvcc(read_data):
    want = [_norm(b) for b in nvcc_blocks(read_data)]
    ctx = make_ctx()
    got = [
        ctx.call("wgemu_b", [("b64", "%rd1")]),
        ctx.call("wgemu_f", [("u32", "%r1"), ("b64", "5")]),
        ctx.call("wgemu_d", [("b32", "%r1")], ("b64", "%rd3")),
    ]
    assert [_norm("\n".join("\t" + ln for ln in lines)) for lines in got] == want
    # block numbers are module-unique and cannot collide with nvcc's own numbering
    assert [lines[0] for lines in got] == [
        f"{{ // callseq wgemu{i}, 0" for i in range(3)
    ]


def test_call_validation():
    ctx = make_ctx()
    ctx.call("wgemu_b", [("b64", "%rd1")])
    with pytest.raises(TranslateError, match="signature"):
        ctx.call("wgemu_b", [("b32", "%r1")])
    with pytest.raises(TranslateError, match="signature"):
        ctx.call("wgemu_b", [("b64", "%rd1")], ("b32", "%r1"))
    with pytest.raises(TranslateError, match="unsupported argument type"):
        ctx.call("wgemu_x", [("f32", "%f1")])
    with pytest.raises(TranslateError, match="neither a register nor an integer"):
        ctx.call("wgemu_x", [("b64", "[%rd1]")])
    with pytest.raises(TranslateError, match="not a \\(type, operand\\) pair"):
        ctx.call("wgemu_x", ["%rd1"])
    with pytest.raises(TranslateError, match="invalid function name"):
        ctx.call("1wgemu", [])
    with pytest.raises(TranslateError, match="invalid return"):
        ctx.call("wgemu_y", [], ("b64", "5"))


CALLER = """
.version 9.0
.target sm_90a
.address_size 64

.visible .entry kc(
\t.param .u64 kc_param_0
)
{
\t.reg .b32 \t%r<2>;
\t.reg .b64 \t%rd<4>;

\tld.param.u64 \t%rd1, [kc_param_0];
\tfakecall.b \t%rd1;
\tmov.u32 \t%r1, %tid.x;
\tfakecall.f \t%r1;
\tfakecall.d \t%rd3, %r1;
\tret;
}
"""


def _call_rule():
    def rewrite(stmt, ctx):
        kind = stmt.opcode.split(".")[1]
        ops = stmt.operands
        if kind == "b":
            return ctx.call("wgemu_b", [("b64", ops[0])])
        if kind == "f":
            return ctx.call("wgemu_f", [("u32", ops[0]), ("b64", "5")])
        return ctx.call("wgemu_d", [("b32", ops[1])], ("b64", ops[0]))

    return (
        "calls",
        SimpleNamespace(
            match=lambda s: s.opcode.startswith("fakecall."), rewrite=rewrite
        ),
    )


def test_translated_module_contains_nvcc_blocks(read_data):
    rep: dict = {}
    out = translate(CALLER, rules=[_call_rule()], report=rep, warn_stream=None)
    norm = _norm(out)
    for block in nvcc_blocks(read_data):
        assert _norm(block) in norm
    # without a library the callee prototypes are declared .extern
    assert ".extern .func wgemu_b(.param .b64 wgemu_b_param_0);" in out
    assert (
        ".extern .func wgemu_f(.param .b32 wgemu_f_param_0, .param .b64 wgemu_f_param_1);"
        in out
    )
    assert (
        ".extern .func (.param .b64 func_retval0) wgemu_d(.param .b32 wgemu_d_param_0);"
        in out
    )
    assert rep["called"] == {
        "wgemu_b": {"args": ["b64"], "ret": None},
        "wgemu_f": {"args": ["b32", "b64"], "ret": None},
        "wgemu_d": {"args": ["b32"], "ret": "b64"},
    }
    assert any("no emulation library given" in w for w in rep["warnings"])
    assert rep["library"] is None
    parse_module(out)
