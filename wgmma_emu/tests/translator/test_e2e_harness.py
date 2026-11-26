"""tests/e2e/common/build_and_run.py logic that needs no GPU and no build."""

from __future__ import annotations

import json
import shutil
import sys

import pytest

GOOD_ENTRY = {
    "name": "t1",
    "source": "t1.cu",
    "nvcc_args": [],
    "modes": ["exact", "check"],
    "expect_violation": False,
    "shim": False,
    "covers": ["wgmma\\.mma_async"],
}


def write_area(root, entries, area="wg"):
    d = root / "tests" / area
    d.mkdir(parents=True, exist_ok=True)
    (d / "t1.cu").write_text("// test\n")
    (d / "e2e_manifest.json").write_text(json.dumps(entries))
    return d / "e2e_manifest.json"


def make_job(harness, tmp_path, *, expect_violation=(), covers=("x",), mode="exact"):
    e = harness.Entry(
        area="wg",
        name="t1",
        source=tmp_path / "t1.cu",
        nvcc_args=[],
        modes=[mode],
        expect_violation=list(expect_violation),
        shim=False,
        covers=list(covers),
    )
    return harness.Job(entry=e, mode=mode, exe=tmp_path / f"t1.{mode}")


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------


def test_parse_manifest(harness, tmp_path):
    path = write_area(tmp_path, [GOOD_ENTRY])
    (e,) = harness.parse_manifest(path)
    assert e.key == "wg/t1" and e.source == (path.parent / "t1.cu").resolve()
    assert e.modes == ["exact", "check"] and e.covers == ["wgmma\\.mma_async"]


DROP = object()


@pytest.mark.parametrize(
    "change,msg",
    [
        ({"shim": DROP}, "missing keys \\['shim'\\]"),
        ({"shim": 1}, "shim must be true or false"),
        ({"extra": 1}, "unknown keys \\['extra'\\]"),
        ({"name": "a b"}, "bad name"),
        ({"source": "missing.cu"}, "does not exist"),
        ({"nvcc_args": "-O3"}, "nvcc_args"),
        ({"nvcc_args": ["area:missing_tu2.cu"]}, "nvcc_args input .* does not exist"),
        ({"modes": []}, "modes"),
        ({"modes": ["exact", "exact"]}, "modes"),
        ({"modes": ["slow"]}, "modes"),
        (
            {"expect_violation": True},
            "expect_violation must be false or a non-empty list",
        ),
        (
            {"expect_violation": []},
            "expect_violation must be false or a non-empty list",
        ),
        (
            {"expect_violation": ["k", "k"]},
            "expect_violation must be false or a non-empty list",
        ),
        ({"expect_violation": ["k"]}, "expect_violation tests"),
        ({"covers": [""]}, "covers"),
        ({"covers": ["("]}, "bad covers regex"),
    ],
)
def test_parse_manifest_rejects(harness, tmp_path, change, msg):
    entry = {k: v for k, v in {**GOOD_ENTRY, **change}.items() if v is not DROP}
    with pytest.raises(harness.ManifestError, match=msg):
        harness.parse_manifest(write_area(tmp_path, [entry]))


def test_parse_manifest_area_input(harness, tmp_path):
    m = write_area(tmp_path, [{**GOOD_ENTRY, "nvcc_args": ["-rdc=true", "area:t1.cu"]}])
    (e,) = harness.parse_manifest(m)
    assert e.nvcc_args == ["-rdc=true", str((m.parent / "t1.cu").resolve())]


def test_parse_manifest_duplicates_and_shape(harness, tmp_path):
    with pytest.raises(harness.ManifestError, match="duplicate name"):
        harness.parse_manifest(write_area(tmp_path, [GOOD_ENTRY, GOOD_ENTRY]))
    with pytest.raises(harness.ManifestError, match="must be a list"):
        harness.parse_manifest(write_area(tmp_path, GOOD_ENTRY))
    neg = dict(GOOD_ENTRY, modes=["check"], expect_violation=["mbarrier-phase"])
    (e,) = harness.parse_manifest(write_area(tmp_path, [neg]))
    assert e.expect_violation == ["mbarrier-phase"]


def test_load_manifests_areas(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    write_area(tmp_path, [GOOD_ENTRY], "wg")
    write_area(tmp_path, [dict(GOOD_ENTRY, name="t2")], "async")
    assert [e.key for e in harness.load_manifests(None)] == ["async/t2", "wg/t1"]
    assert [e.key for e in harness.load_manifests(["wg"])] == ["wg/t1"]
    with pytest.raises(harness.ManifestError, match="no e2e_manifest.json"):
        harness.load_manifests(["nope"])


# ---------------------------------------------------------------------------------------------
# Output parsing and verdicts
# ---------------------------------------------------------------------------------------------

PASS_OUT = "hello\nWGEMU-TEST name=m64n8 cases=4 mismatches=0\nWGEMU-TEST name=m64n16 cases=2 mismatches=0\nWGEMU-TEST-END\n"


def judged(harness, tmp_path, out, *, run_rc=0, expect_violation=()):
    j = make_job(
        harness,
        tmp_path,
        expect_violation=expect_violation,
        mode="check" if expect_violation else "exact",
    )
    j.build_rc = 0
    j.run_rc = run_rc
    harness.parse_output(j, out)
    harness.judge(j)
    return j


def test_positive_verdicts(harness, tmp_path):
    j = judged(harness, tmp_path, PASS_OUT)
    assert (j.verdict, j.note) == ("PASS", "")
    assert j.items == [("m64n8", 4, 0), ("m64n16", 2, 0)] and j.ended
    cases = [
        (
            PASS_OUT.replace(
                "mismatches=0\nWGEMU-TEST name=m64n16",
                "mismatches=3\nWGEMU-TEST name=m64n16",
            ),
            0,
            "mismatches=3",
        ),
        (PASS_OUT.replace("cases=2", "cases=0"), 0, "cases=0: m64n16"),
        (PASS_OUT.replace("WGEMU-TEST-END\n", ""), 0, "no WGEMU-TEST-END"),
        ("WGEMU-TEST-END\n", 0, "no WGEMU-TEST items"),
        (PASS_OUT + "WGEMU-VIOLATION wgmma-fence missing\n", 0, "violations=1"),
        (
            PASS_OUT + "WGEMU-TEST name=x cases=many\n",
            0,
            "malformed: WGEMU-TEST name=x cases=many",
        ),
        (
            PASS_OUT + "WGEMU-TEST name=m64n8 cases=1 mismatches=0\n",
            0,
            "duplicate item m64n8",
        ),
        (PASS_OUT, 1, "exit 1"),
    ]
    for out, rc, note in cases:
        j = judged(harness, tmp_path, out, run_rc=rc)
        assert j.verdict == "FAIL" and note in j.note, (out, j.note)


def test_negative_verdicts(harness, tmp_path):
    kinds = ["mbarrier-phase"]
    viol = "WGEMU-VIOLATION mbarrier-phase tid=0\n"
    j = judged(harness, tmp_path, PASS_OUT + viol, expect_violation=kinds)
    assert (j.verdict, j.note) == ("PASS", "violations=1 kinds=mbarrier-phase")
    bad_items = PASS_OUT.replace("cases=4 mismatches=0", "cases=4 mismatches=2")
    cases = [
        # mismatches without the expected diagnostic (Codex emu-prebuild-review P1)
        (bad_items, 0, "expected violation not printed: mbarrier-phase"),
        (PASS_OUT, 0, "expected violation not printed: mbarrier-phase"),
        # abnormal exit without the completion marker after a violation (same review)
        (viol, 2, "exit 2"),
        (viol, 2, "no WGEMU-TEST-END"),
        (viol + "WGEMU-TEST-END\n", 0, "no WGEMU-TEST items"),
        (
            PASS_OUT + "WGEMU-VIOLATION other-kind\n",
            0,
            "expected violation not printed",
        ),
        (
            PASS_OUT + viol + "WGEMU-VIOLATION other-kind\n",
            0,
            "unexpected violation: other-kind",
        ),
        (bad_items + viol, 0, "mismatches=2"),
    ]
    for out, rc, note in cases:
        j = judged(harness, tmp_path, out, run_rc=rc, expect_violation=kinds)
        assert j.verdict == "FAIL" and note in j.note, (out, j.note)


def test_build_and_run_states(harness, tmp_path):
    j = make_job(harness, tmp_path)
    harness.judge(j)
    assert j.verdict == "NOT_BUILT"
    j.build_rc, j.build_note = 1, "build rc=1"
    harness.judge(j)
    assert (j.verdict, j.note) == ("BUILD_FAIL", "build rc=1")
    j.build_rc = 0
    harness.judge(j)
    assert (j.verdict, j.note) == ("BUILT", "not run")
    j.covers_missing = ["setmaxnreg"]
    harness.judge(j)
    assert j.verdict == "FAIL" and "setmaxnreg" in j.note
    j.covers_missing = []
    j.run_rc, j.timed_out = -9, True
    harness.judge(j)
    assert j.verdict == "TIMEOUT"
    j = judged(harness, tmp_path, PASS_OUT)
    j.covers_missing = ["setmaxnreg"]
    harness.judge(j)
    assert j.verdict == "FAIL" and "covers not in PTX: setmaxnreg" in j.note


def test_build_cmd_defaults(harness, tmp_path):
    j = make_job(harness, tmp_path)
    assert harness.build_cmd(j) == [
        sys.executable, str(harness.NVCC_EMU), str(tmp_path / "t1.cu"), "-o", str(tmp_path / "t1.exact"),
        "-std=c++17", "-O3", "-I", str(harness.ROOT / "include"),
    ]  # fmt: skip
    j.entry.nvcc_args = ["-std=c++20", "-O2", "-lineinfo"]
    cmd = harness.build_cmd(j)
    assert (
        "-std=c++17" not in cmd
        and "-O3" not in cmd
        and cmd[-3:] == ["-std=c++20", "-O2", "-lineinfo"]
    )
    assert j.keep_dir == tmp_path / "t1.exact.wgemu.d" and j.tag == "wg/t1.exact"


# ---------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------

TOKENS = (
    {"wgmma.fence.sync.aligned", "cvta.to.shared.u64", "elect.sync"},
    {"%cluster_ctarank", "%r1"},
    {".reqnctapercluster", ".maxntid"},
)


@pytest.mark.parametrize(
    "pattern,hit",
    [
        ("wgmma\\.fence", True),
        ("wgmma", True),
        ("cvt", False),
        ("cvta", True),
        ("elect\\.sync", True),
        ("%cluster_ctarank", True),
        ("%cluster", False),
        ("\\.reqnctapercluster", True),
        (".maxntid", True),
        ("\\.explicitcluster", False),
    ],
)
def test_matches(harness, pattern, hit):
    assert harness._matches(pattern, TOKENS) is hit


def test_check_covers_and_coverage_table(harness, tmp_path, data_dir):
    j = make_job(
        harness,
        tmp_path,
        covers=(
            "wgmma\\.mma_async",
            "%cluster_ctarank",
            "\\.reqnctapercluster",
            "setmaxnreg",
        ),
    )
    j.build_rc = 0
    j.keep_dir.mkdir()
    shutil.copy(data_dir / "sample_sm90.ptx", j.keep_dir / "sample_sm90.wgemu_orig.ptx")
    other = make_job(harness, tmp_path, mode="fast")
    other.entry.name = "t2"
    other.build_rc = 0
    tokens = harness.check_covers([j, other])
    assert j.covers_missing == ["setmaxnreg"]
    assert other.covers_missing == ["<no *.wgemu_orig.ptx in keep dir>"]
    assert set(tokens) == {"wg/t1"}

    inv = tmp_path / "inventory.tsv"
    inv.write_text(
        "wgmma\\.mma_async\t90\trewrite:wgmma\t9.7.15.5.2\t\n"
        "setmaxnreg\t90\tnoop\t9.7.19.5\t\n"
        "mov\t20\tnative\t9.7.9.10\t\n"
        "elect\\.sync\t90\tinline\t9.7.13.12\t\n"
    )
    j.covers_missing = []
    j.verdict = "PASS"
    assert harness.coverage_table([j], tokens, inv, 86) == [
        "setmaxnreg\t90\tnoop\t9.7.19.5"
    ]
    j.verdict = "FAIL"
    assert len(harness.coverage_table([j], tokens, inv, 86)) == 3


def test_write_tsv(harness, tmp_path):
    j = judged(harness, tmp_path, PASS_OUT)
    out = tmp_path / "res" / "results.tsv"
    harness.write_tsv(out, [j])
    rows = out.read_text().splitlines()
    assert rows[0].split("\t")[:3] == ["area", "name", "mode"] and len(rows) == 2
    assert rows[1].split("\t")[13] == "PASS"
    items = (tmp_path / "res" / "results.tsv.items.tsv").read_text().splitlines()
    assert items[1:] == ["wg\tt1\texact\tm64n8\t4\t0", "wg\tt1\texact\tm64n16\t2\t0"]


def test_main_usage_and_list(harness, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    write_area(tmp_path, [GOOD_ENTRY])
    assert harness.main(["--list"]) == 0
    assert "wg/t1\tmodes=exact,check\texpect_violation=False" in capsys.readouterr().out
    assert harness.main([]) == 2  # running needs --gpu
    assert harness.main(["--gpu", "0,1"]) == 2
    assert harness.main(["--build-only", "--modes", "slow"]) == 2
    assert harness.main(["--list", "--areas", "nope"]) == 2
    assert harness.main(["--list", "--filter", "nomatch"]) == 2
    assert harness.main(["--build-only", "--run-only"]) == 2
    capsys.readouterr()
