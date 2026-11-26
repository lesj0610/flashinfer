from __future__ import annotations

import pytest
from ptxemu import lexer as lx
from ptxemu.core import parse_stmt, split_dests


def kinds(els):
    return [e.kind for e in els]


def test_module_and_body_elements():
    src = (
        ".version 8.7\n"
        ".target sm_90a\n"
        ".address_size 64\n"
        "\n"
        "// comment one\n"
        "/* block\n"
        "comment */\n"
        ".visible .entry k(\n"
        "\t.param .u64 k_param_0\n"
        ")\n"
        ".maxntid 128, 1, 1\n"
        "{\n"
        "\t.reg .b32 \t%r<4>;\n"
        '\t.pragma "nounroll";\n'
        "$L__BB0_1:\n"
        "\t@%p1 bra \t$L__BB0_1;\n"
        "\tret;\n"
        "}\n"
    )
    els = lx.lex(src)
    assert kinds(els) == [
        lx.LINE_DIRECTIVE,
        lx.LINE_DIRECTIVE,
        lx.LINE_DIRECTIVE,
        lx.COMMENT,
        lx.COMMENT,
        lx.HEADER,
        lx.LBRACE,
        lx.STATEMENT,
        lx.STATEMENT,
        lx.LABEL,
        lx.STATEMENT,
        lx.STATEMENT,
        lx.RBRACE,
    ]
    assert els[0].text == ".version 8.7"
    assert els[4].text == "/* block\ncomment */"
    assert els[5].text.startswith(".visible .entry k(")
    assert els[5].text.endswith(".maxntid 128, 1, 1")
    assert els[7].text == ".reg .b32 \t%r<4>"
    assert els[8].text == '.pragma "nounroll"'
    assert els[9].text == "$L__BB0_1"
    assert els[10].text == "@%p1 bra \t$L__BB0_1"
    assert els[1].lineno == 2
    assert els[10].lineno == 16


def test_inline_asm_region_and_scoped_labels():
    src = (
        "\t// begin inline asm\n"
        "\t{\n"
        ".reg .pred p;\n"
        "LAB: \n"
        "@p bra LAB;\n"
        "}\n"
        "\t// end inline asm\n"
        "\tmov.u32 \t%r1, 0;\n"
    )
    els = lx.lex(src)
    assert kinds(els) == [
        lx.ASM_BEGIN,
        lx.LBRACE,
        lx.STATEMENT,
        lx.LABEL,
        lx.STATEMENT,
        lx.RBRACE,
        lx.ASM_END,
        lx.STATEMENT,
    ]
    assert [e.in_asm for e in els] == [True] * 7 + [False]
    assert els[3].text == "LAB"


def test_multiline_statement_with_comment_and_tensor_address():
    src = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r8], // dst\n"
        "    [%rd5, {%r10, %r4}], [%r9];\n"
    )
    els = lx.lex(src)
    assert kinds(els) == [lx.STATEMENT, lx.COMMENT]
    assert els[1].text == "// dst"
    st = parse_stmt(els[0].text)
    assert (
        st.opcode
        == "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    )
    assert st.operands == ["[%r8]", "[%rd5, {%r10, %r4}]", "[%r9]"]


def test_dual_destination_and_guard():
    (el,) = lx.lex("@%px elect.sync %rx|%px, %r2;")
    st = parse_stmt(el.text)
    assert st.guard == "@%px"
    assert st.opcode == "elect.sync"
    assert st.operands == ["%rx|%px", "%r2"]
    assert split_dests(st.operands[0]) == ["%rx", "%px"]


def test_prototype_vs_definition():
    src = ".extern .func f(.param .b32 a);\n.func g()\n{\n\tret;\n}\n"
    els = lx.lex(src)
    assert kinds(els) == [lx.STATEMENT, lx.HEADER, lx.LBRACE, lx.STATEMENT, lx.RBRACE]
    assert lx.is_header_text(els[0].text)


def test_initializer_braces_are_not_blocks():
    (el,) = lx.lex(".global .align 1 .b8 $str[3] = {65, 10, 0};")
    assert el.kind == lx.STATEMENT
    assert el.text == ".global .align 1 .b8 $str[3] = {65, 10, 0}"


def test_debug_sections_and_line_directives():
    src = (
        '\t.file\t1 "/tmp/x.cu"\n'
        "\t.section\t.debug_str\n"
        "\t{\n"
        "$L__info_string0:\n"
        ".b8 95,90,0\n"
        "\n"
        "\t}\n"
        "\t.loc\t1 30 13, function_name $L__info_string0, inlined_at 1 54 3\n"
    )
    els = lx.lex(src)
    assert kinds(els) == [lx.LINE_DIRECTIVE, lx.BLOCK_DIRECTIVE, lx.LINE_DIRECTIVE]
    assert els[1].text.startswith(".section\t.debug_str") and els[1].text.endswith("}")
    assert els[2].text.startswith(".loc\t1 30 13")


def test_empty_statement_is_dropped():
    els = lx.lex("ret;;\n")
    assert kinds(els) == [lx.STATEMENT]


@pytest.mark.parametrize(
    "src",
    [
        "mov.u32 %r1, 0",  # missing ';'
        "mov.u32 %r1, [%rd1;",  # ';' inside '['
        "mov.u32 %r1, %r2];",  # unbalanced ']'
        "/* never closed",
        '.pragma "open;',
    ],
)
def test_malformed_input_raises(src):
    with pytest.raises(lx.LexError):
        lx.lex(src)


def test_nvcc_sample_structure(read_data):
    els = lx.lex(read_data("sample_sm90.ptx"))
    assert sum(e.kind == lx.HEADER for e in els) == 2
    begins = sum(e.kind == lx.ASM_BEGIN for e in els)
    assert begins == sum(e.kind == lx.ASM_END for e in els) == 10
    inside = False
    for e in els:
        if e.kind == lx.ASM_BEGIN:
            inside = True
        assert e.in_asm == inside, e
        if e.kind == lx.ASM_END:
            inside = False
    target = next(e for e in els if e.text.startswith(".target"))
    assert target.lineno == 10
    assert target.text == ".target sm_90a"
