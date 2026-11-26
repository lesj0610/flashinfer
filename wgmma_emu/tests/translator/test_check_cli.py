"""Inventory check (ptxemu.check) and the command line front end (ptxemu.cli)."""

from __future__ import annotations

import json

import pytest
from ptxemu import check, cli, core

INVENTORY = (
    "opcode_regex\tmin_sm\thandling\tptx_section\tnote\n"
    "# comment line\n"
    "\n"
    "cvt\t90\trewrite:test\t9.7.9.21\tmust not match cvta.*\n"
    "wgmma\\.fence\t90\trewrite:wgmma\t9.7.15.5.1\t\n"
    "elect\\.sync\t90\tinline\t9.7.13.12\t\n"
    "%cluster_ctarank\t90\trewrite:misc\t10.14\t\n"
    "\\.reqnctapercluster\t90\tstrip\t11.4.3\t\n"
    "mov\t20\tnative\t9.7.9.10\tbelow the threshold\n"
)

PTX = """.version 8.0
.target sm_80
.address_size 64

.visible .entry k(
\t.param .u64 k_param_0
)
.reqnctapercluster 1, 1, 1
{
\t.reg .pred \t%p<2>;
\t.reg .b32 \t%r<4>;
\t.reg .b64 \t%rd<4>;

\tld.param.u64 \t%rd1, [k_param_0];
\tcvta.to.shared.u64 \t%rd2, %rd1;
\twgmma.fence.sync.aligned;
\tmov.u32 \t%r1, %cluster_ctarank;
\t@%p1 elect.sync \t%r3|%p1, -1;
\tret;
}
"""


@pytest.fixture
def inventory(tmp_path):
    p = tmp_path / "inventory.tsv"
    p.write_text(INVENTORY)
    return p


def line_of(text: str, needle: str) -> int:
    return 1 + next(i for i, ln in enumerate(text.splitlines()) if needle in ln)


def test_load_inventory(inventory):
    rows = check.load_inventory(inventory)
    assert [r.pattern for r in rows] == [
        "cvt",
        "wgmma\\.fence",
        "elect\\.sync",
        "%cluster_ctarank",
        "\\.reqnctapercluster",
        "mov",
    ]
    assert [r.kind for r in rows] == [
        "opcode",
        "opcode",
        "opcode",
        "operand",
        "directive",
        "opcode",
    ]
    assert rows[0].note == "must not match cvta.*" and rows[0].lineno == 4
    assert rows[1].handling == "rewrite:wgmma" and rows[1].min_sm == 90


@pytest.mark.parametrize(
    "line,msg",
    [
        ("x\t90\tinline\n", "4 tab-separated columns"),
        ("x\tninety\tinline\ts\n", "not an integer"),
        ("x\t90\tsometimes\ts\n", "unknown handling"),
        ("x\t90\tlib:\ts\n", "unknown handling"),
        ("x\t90\tnative\ts\n", "cannot be 'native'"),
        ("x(\t90\tinline\ts\n", "bad regex"),
        ("\t90\tinline\ts\n", "empty opcode_regex"),
    ],
)
def test_load_inventory_rejects(tmp_path, line, msg):
    p = tmp_path / "inv.tsv"
    p.write_text("opcode_regex\tmin_sm\thandling\tptx_section\tnote\n" + line)
    with pytest.raises(check.InventoryError, match=msg):
        check.load_inventory(p)


def test_load_inventory_empty_or_missing(tmp_path):
    p = tmp_path / "inv.tsv"
    p.write_text("opcode_regex\tmin_sm\thandling\tptx_section\tnote\n# nothing\n")
    with pytest.raises(check.InventoryError, match="no inventory rows"):
        check.load_inventory(p)
    with pytest.raises(check.InventoryError, match="not found"):
        check.load_inventory(tmp_path / "missing.tsv")


def test_find_remaining(inventory):
    hits = check.find_remaining(PTX, check.load_inventory(inventory))
    assert [(h.kind, h.token, h.pattern) for h in hits] == [
        ("directive", ".reqnctapercluster", "\\.reqnctapercluster"),
        ("opcode", "wgmma.fence.sync.aligned", "wgmma\\.fence"),
        ("operand", "%cluster_ctarank", "%cluster_ctarank"),
        ("opcode", "elect.sync", "elect\\.sync"),
    ]
    assert [h.lineno for h in hits] == [
        line_of(PTX, ".visible .entry k("),
        line_of(PTX, "wgmma.fence"),
        line_of(PTX, "%cluster_ctarank"),
        line_of(PTX, "elect.sync"),
    ]
    # a lower threshold selects the sm_20 'mov' row as well
    low = check.find_remaining(PTX, check.load_inventory(inventory), min_sm=20)
    assert [h.token for h in low if h.pattern == "mov"] == ["mov.u32"]
    text = check.format_hits(hits, "x.ptx")
    assert text.splitlines()[1].startswith(
        f"x.ptx:{line_of(PTX, 'wgmma.fence')}: opcode wgmma.fence.sync.aligned"
    )


def test_find_remaining_target(inventory):
    rows = check.load_inventory(inventory)
    hits = check.find_remaining(PTX.replace(".target sm_80", ".target sm_90a"), rows)
    assert (
        hits[0].kind == "target" and hits[0].token == "sm_90a" and hits[0].lineno == 2
    )
    assert not [
        h
        for h in check.find_remaining(
            PTX.replace(".target sm_80", ".target sm_75"), rows
        )
        if h.kind == "target"
    ]


def test_check_main(tmp_path, inventory, capsys):
    bad = tmp_path / "bad.ptx"
    bad.write_text(PTX)
    good = tmp_path / "good.ptx"
    good.write_text(".version 8.0\n.target sm_80\n.address_size 64\n")
    assert check.main([str(bad), "--inventory", str(inventory)]) == 3
    assert "4 sm_86+ item(s) remain" in capsys.readouterr().err
    assert check.main([str(good), "--inventory", str(inventory)]) == 0
    assert check.main([str(good), "--inventory", str(tmp_path / "none.tsv")]) == 1
    assert check.main([str(tmp_path / "none.ptx"), "--inventory", str(inventory)]) == 1


# ---------------------------------------------------------------------------------------------
# CLI (rule modules of the other workers are not loaded here)
# ---------------------------------------------------------------------------------------------

SRC = """// header comment
.version 8.7
.target sm_90a
.address_size 64

.visible .entry k(
\t.param .u64 k_param_0
)
{
\t.reg .b32 \t%r<2>;

\tmov.u32 \t%r1, %tid.x;
\twgemutestfake.op \t%r1;
\tret;
}
"""


@pytest.fixture
def no_default_rules(monkeypatch):
    monkeypatch.setattr(core, "DEFAULT_RULE_MODULES", ())


@pytest.fixture
def cli_inventory(tmp_path):
    p = tmp_path / "inventory.tsv"
    p.write_text("wgemutestfake\\.op\t90\trewrite:none\t-\ttest-only opcode\n")
    return p


def run_cli(tmp_path, src, *extra):
    inp = tmp_path / "in.ptx"
    inp.write_text(src)
    out = tmp_path / "out.ptx"
    rep = tmp_path / "rep.json"
    rc = cli.main([str(inp), "-o", str(out), "--report", str(rep), *extra])
    return rc, out, rep


def test_cli_remaining_items_exit_3(tmp_path, no_default_rules, cli_inventory, capsys):
    rc, out, rep = run_cli(tmp_path, SRC, "--inventory", str(cli_inventory))
    assert rc == 3
    text = out.read_text()
    assert ".target sm_80\n" in text and text.startswith(
        "// header comment\n.version 8.7\n"
    )
    r = json.loads(rep.read_text())
    assert r["check"]["remaining"] == [
        {
            "line": line_of(text, "wgemutestfake.op"),
            "kind": "opcode",
            "token": "wgemutestfake.op",
            "inventory": "wgemutestfake\\.op",
            "min_sm": 90,
        }
    ]
    assert r["check"]["inventory"] == str(cli_inventory) and r["check"]["min_sm"] == 86
    assert r["output"] == str(out) and r["library_file"] is None
    assert r["rules"] == {"loaded": [], "missing": []}
    assert r["mode"] == "exact" and r["src_target"] == "sm_90a"
    assert "1 sm_86+ item(s) remain after translation" in capsys.readouterr().err
    assert not (tmp_path / "out.ptx.wgemu_tmp").exists()


def test_cli_clean_and_no_check(tmp_path, no_default_rules, cli_inventory):
    rc, out, rep = run_cli(
        tmp_path,
        SRC.replace("\twgemutestfake.op \t%r1;\n", ""),
        "--inventory",
        str(cli_inventory),
    )
    assert rc == 0 and json.loads(rep.read_text())["check"]["remaining"] == []
    rc, out, rep = run_cli(tmp_path, SRC, "--no-check", "--mode", "check", "--trap")
    assert rc == 0
    r = json.loads(rep.read_text())
    assert (
        r["check"]["inventory"] is None and r["mode"] == "check" and r["trap"] is True
    )


def test_cli_errors(tmp_path, no_default_rules, cli_inventory, capsys):
    rc, out, rep = run_cli(
        tmp_path,
        SRC.replace(".target sm_90a", ".target sm_100a"),
        "--inventory",
        str(cli_inventory),
    )
    assert rc == 1 and not out.exists()
    r = json.loads(rep.read_text())
    assert "newer than sm_90a" in r["error"]
    assert run_cli(tmp_path, SRC, "--inventory", str(tmp_path / "none.tsv"))[0] == 1
    assert (
        cli.main(
            [str(tmp_path / "missing.ptx"), "-o", str(tmp_path / "o.ptx"), "--no-check"]
        )
        == 1
    )
    assert (
        cli.main(
            [str(tmp_path / "in.ptx"), "-o", str(tmp_path / "o.ptx"), "--mode", "bogus"]
        )
        == 2
    )
    assert cli.main([str(tmp_path / "in.ptx")]) == 2
    capsys.readouterr()
