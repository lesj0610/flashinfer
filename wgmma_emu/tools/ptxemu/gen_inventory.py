"""Generate and audit ``tools/ptxemu/inventory.tsv`` (PLAN section 5). Owner: T2.

The inventory lists every sm_86 / sm_87 / sm_88 / sm_89 / sm_90 / sm_90a requirement of the PTX
ISA text (``ext/ptxisa/ptx.txt``) down to qualifiers, types, special registers and directives.
Forms that need sm_100 or later are not listed; the translation rules reject them.

``ROWS`` and ``FEATURES`` below are the hand-checked table. Running this module

1. checks each row against ptx.txt: every cited line names an sm_86..sm_99 target, the first
   cited line lies in the row's section, and the smallest target the cited lines name is the
   row's ``min_sm``;
2. audits ptx.txt: every line from chapter 1 on that names an sm_86..sm_99 target must be cited
   by a row or a feature (uncited lines are printed, exit 1);
3. runs ``check.find_remaining`` on probe modules: each ``eg`` of a row is reported by that row
   (the first matching row, as check.py reports it), and no row reports the sm_80 forms of
   ``SM80_NATIVE``;
4. writes inventory.tsv and reloads it with ``check.load_inventory``.

Requirements without an opcode / operand / directive token of their own (an operand count, a
qualifier on an address operand, the size of a static ``.shared`` variable) are ``FEATURES``:
``#feature`` comment lines of the TSV. check.py and the e2e harness skip comment lines;
``tests/misc/test_inventory_coverage.py`` requires a ``// WGEMU-COVERS-FEATURE: <name>`` line in
a test source of an e2e manifest for each of them.

CLI: ``python -m ptxemu.gen_inventory [--ptx PATH] [--out PATH] [--check-only]``; exit 0 when
every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import check

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PTX = Path(
    os.environ.get("WGEMU_PTX_ISA", str(ROOT.parents[1] / "ext" / "ptxisa" / "ptx.txt"))
)
DEFAULT_OUT = Path(__file__).with_name("inventory.tsv")

# sm_86 .. sm_99 (with or without '_' and an 'a' suffix), as named in requirement lines.
SM_RE = re.compile(r"\bsm_?(8[6-9]|9\d)a?\b")
_HEAD_RE = re.compile(r"^(\d+(?:\.\d+)*)\. (\S.*)$")

Q = r"(?:\.[\w:]+)*"  # any run of dotted qualifiers


@dataclass(frozen=True)
class Row:
    pattern: str
    min_sm: int
    handling: str
    section: str
    lines: tuple[int, ...]  # requirement lines of ptx.txt
    eg: tuple[str, ...]  # tokens / module snippets the row must report
    note: str = ""


@dataclass(frozen=True)
class Feature:
    name: str
    min_sm: int
    handling: str
    section: str
    lines: tuple[int, ...]
    eg: tuple[str, ...]  # PTX text showing the form (documentation only)
    note: str = ""


def _R(pattern, min_sm, handling, section, lines, eg, note=""):  # noqa: N802
    return Row(pattern, min_sm, handling, section, tuple(lines), tuple(eg), note)


# Rows of one kind are listed most specific first: check.py reports the first matching opcode /
# operand row, and every eg below must be reported by its own row.
ROWS: tuple[Row, ...] = (
    # ---- integer arithmetic (9.7.1) ----
    _R(r"add\.[us]16x2", 90, "inline", "9.7.1.1", [7443], ["add.u16x2", "add.s16x2"]),
    _R(
        r"min(?:\.relu\.(?:s16x2|s32)|\.[us]16x2)",
        90,
        "inline",
        "9.7.1.13",
        [7832],
        ["min.u16x2", "min.s16x2", "min.relu.s16x2", "min.relu.s32"],
    ),
    _R(
        r"max(?:\.relu\.(?:s16x2|s32)|\.[us]16x2)",
        90,
        "inline",
        "9.7.1.14",
        [7894],
        ["max.u16x2", "max.s16x2", "max.relu.s16x2", "max.relu.s32"],
    ),
    # ---- floating point min/max .xorsign.abs (9.7.3, 9.7.4) ----
    _R(
        r"min(?:\.ftz)?(?:\.NaN)?\.xorsign\.abs\.f32",
        86,
        "inline",
        "9.7.3.11",
        [9345],
        ["min.xorsign.abs.f32", "min.ftz.NaN.xorsign.abs.f32"],
    ),
    _R(
        r"max(?:\.ftz)?(?:\.NaN)?\.xorsign\.abs\.f32",
        86,
        "inline",
        "9.7.3.12",
        [9436],
        ["max.xorsign.abs.f32", "max.NaN.xorsign.abs.f32"],
    ),
    _R(
        r"min(?:\.ftz)?(?:\.NaN)?\.xorsign\.abs\.b?f16(?:x2)?",
        86,
        "inline",
        "9.7.4.7",
        [10592],
        [
            "min.xorsign.abs.f16",
            "min.NaN.xorsign.abs.bf16x2",
            "min.ftz.xorsign.abs.f16x2",
        ],
    ),
    _R(
        r"max(?:\.ftz)?(?:\.NaN)?\.xorsign\.abs\.b?f16(?:x2)?",
        86,
        "inline",
        "9.7.4.8",
        [10679],
        ["max.xorsign.abs.bf16", "max.NaN.xorsign.abs.f16x2"],
    ),
    # ---- half precision (9.7.4) ----
    _R(
        r"add(?:\.rn)?\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.4.1",
        [10161],
        ["add.bf16", "add.rn.bf16x2"],
    ),
    _R(
        r"sub(?:\.rn)?\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.4.2",
        [10236],
        ["sub.rn.bf16", "sub.bf16x2"],
    ),
    _R(
        r"mul(?:\.rn)?\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.4.3",
        [10312],
        ["mul.rn.bf16", "mul.bf16x2"],
    ),
    _R(
        rf"fma{Q}\.oob",
        90,
        "inline",
        "9.7.4.4",
        [10397],
        ["fma.rn.oob.bf16", "fma.rn.oob.relu.f16x2", "fma.rn.oob.f16"],
        "OOB NaN = 0x7FFF per half [inference]",
    ),
    _R(
        r"tanh\.approx\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.4.9",
        [10752],
        ["tanh.approx.bf16", "tanh.approx.bf16x2"],
    ),
    _R(
        r"ex2\.approx(?:\.ftz)?\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.4.10",
        [10853],
        ["ex2.approx.ftz.bf16", "ex2.approx.ftz.bf16x2"],
    ),
    # ---- comparison (9.7.8) ----
    _R(
        rf"set{Q}\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.8.1",
        [11792],
        [
            "set.eq.u32.bf16",
            "set.lt.bf16.f32",
            "set.ne.bf16x2.bf16x2",
            "set.gt.and.u16.bf16",
            "set.ge.bf16.s64",
        ],
    ),
    _R(
        rf"setp{Q}\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.8.2",
        [11857],
        ["setp.eq.bf16", "setp.lt.and.bf16x2"],
    ),
    # ---- ld / st (9.7.10.8, 9.7.10.11, 9.7.10.12) ----
    _R(
        rf"ld{Q}\.cluster",
        90,
        "inline",
        "9.7.10.8",
        [13255],
        ["ld.relaxed.cluster.global.u32", "ld.acquire.cluster.shared::cta.b64"],
        ".cluster -> .gpu",
    ),
    _R(
        rf"ld{Q}\.shared::cluster",
        90,
        "inline",
        "9.7.10.8",
        [13257, 931],
        ["ld.shared::cluster.u32", "ld.relaxed.cta.shared::cluster.v2.f32"],
        "shared::cluster -> shared (cluster of one CTA)",
    ),
    _R(
        rf"ld{Q}\.proxy::readonly",
        90,
        "inline",
        "9.7.10.8",
        [13263],
        ["ld.global.u32.proxy::readonly", "ld.global.v2.b32.proxy::readonly"],
        "qualifier dropped",
    ),
    _R(
        rf"st(?!\.async){Q}\.cluster",
        90,
        "inline",
        "9.7.10.11",
        [13559],
        ["st.relaxed.cluster.global.u32", "st.release.cluster.shared::cta.b32"],
        ".cluster -> .gpu",
    ),
    _R(
        rf"st(?!\.async){Q}\.shared::cluster",
        90,
        "inline",
        "9.7.10.11",
        [13561],
        ["st.shared::cluster.u32", "st.relaxed.cta.shared::cluster.v4.b32"],
    ),
    _R(
        r"st\.async",
        90,
        "rewrite:async",
        "9.7.10.12",
        [13671],
        ["st.async.shared::cluster.mbarrier::complete_tx::bytes.u32"],
    ),
    # ---- multimem (9.7.10.15) ----
    _R(
        r"multimem\.(?:ld_reduce|st|red)",
        90,
        "inline",
        "9.7.10.15",
        [13902],
        [
            "multimem.ld_reduce.relaxed.sys.global.add.f32",
            "multimem.st.global.v4.f32",
            "multimem.red.relaxed.gpu.global.add.u32",
        ],
        "one replica [inference]",
    ),
    # ---- prefetch (9.7.10.16) ----
    _R(
        rf"prefetch{Q}\.tensormap",
        90,
        "rewrite:async",
        "9.7.10.16",
        [13979],
        ["prefetch.tensormap", "prefetch.param.tensormap", "prefetch.const.tensormap"],
    ),
    _R(
        r"prefetch(?:\.global)?\.L1::32B\.valid_addr",
        90,
        "inline",
        "9.7.10.16",
        [13980],
        ["prefetch.global.L1::32B.valid_addr", "prefetch.L1::32B.valid_addr"],
        "-> prefetch{.global}.L1",
    ),
    # ---- isspacep / cvta (9.7.10.22, 9.7.10.23) ----
    _R(
        r"isspacep\.shared::cluster",
        90,
        "inline",
        "9.7.10.22",
        [14219],
        ["isspacep.shared::cluster"],
    ),
    _R(
        r"cvta(?:\.to)?\.shared::cluster",
        90,
        "inline",
        "9.7.10.23",
        [14278],
        ["cvta.shared::cluster.u64", "cvta.to.shared::cluster.u64"],
    ),
    # ---- cvt (9.7.10.24) ----
    _R(
        rf"cvt{Q}\.e(?:4m3|5m2)x2\.f32",
        89,
        "lib:wgemu_cvt_f32x2_to_fp8x2",
        "9.7.10.24",
        [14702, 14703, 14704, 14705, 14734, 14735, 38846],
        ["cvt.rn.satfinite.e4m3x2.f32", "cvt.rn.satfinite.relu.e5m2x2.f32"],
    ),
    _R(
        rf"cvt{Q}\.e(?:4m3|5m2)x2\.f16x2",
        89,
        "lib:wgemu_cvt_f16x2_to_fp8x2",
        "9.7.10.24",
        [14702, 14703, 14704, 14705, 14734, 14735],
        ["cvt.rn.satfinite.e4m3x2.f16x2", "cvt.rn.satfinite.relu.e5m2x2.f16x2"],
    ),
    _R(
        rf"cvt{Q}\.f16x2\.e(?:4m3|5m2)x2",
        89,
        "lib:wgemu_cvt_fp8x2_to_f16x2",
        "9.7.10.24",
        [14702, 14704, 14734],
        ["cvt.rn.f16x2.e4m3x2", "cvt.rn.relu.f16x2.e5m2x2"],
    ),
    _R(
        rf"cvt{Q}\.bf16\.(?:[us](?:8|16|32|64)|f16|f64)",
        90,
        "lib:wgemu_cvt_to_bf16",
        "9.7.10.24",
        [14732],
        [
            "cvt.rn.bf16.f16",
            "cvt.rn.bf16.s32",
            "cvt.rz.bf16.f64",
            "cvt.rm.bf16.u8",
            "cvt.rp.bf16.u64",
        ],
    ),
    _R(
        rf"cvt{Q}\.bf16\.bf16",
        90,
        "inline",
        "9.7.10.24",
        [14732],
        ["cvt.rni.bf16.bf16", "cvt.rzi.bf16.bf16"],
    ),
    _R(
        rf"cvt{Q}\.(?:[us](?:8|16|32|64)|f16|f64)\.bf16",
        90,
        "inline",
        "9.7.10.24",
        [14732],
        ["cvt.rzi.s32.bf16", "cvt.rn.f16.bf16", "cvt.f64.bf16", "cvt.rni.u8.bf16"],
        "via the exact f32 value",
    ),
    _R(
        rf"cvt{Q}\.ftz{Q}\.f32\.bf16",
        90,
        "inline",
        "9.7.10.24",
        [14733],
        ["cvt.ftz.f32.bf16"],
    ),
    _R(
        r"cvt\.r[nz](?:\.relu)?\.tf32\.f32",
        90,
        "lib:wgemu_cvt_f32_to_tf32",
        "9.7.10.24",
        [14732],
        ["cvt.rn.tf32.f32", "cvt.rz.relu.tf32.f32"],
    ),
    # ---- mapa / getctarank (9.7.10.26, 9.7.10.27) ----
    _R(
        r"mapa",
        90,
        "inline",
        "9.7.10.26",
        [14941],
        ["mapa.u64", "mapa.shared::cluster.u32"],
        "rank 0 only",
    ),
    _R(
        r"getctarank",
        90,
        "inline",
        "9.7.10.27",
        [14972],
        ["getctarank.u32", "getctarank.shared::cluster.u32"],
    ),
    # ---- bulk async copies, tensor copies, tensormap (A: rules_async) ----
    _R(
        r"cp\.async\.bulk(?:\.(?:weak|relaxed|cta|cluster|gpu|sys))*\.(?:shared::cluster|shared::cta|global)",
        90,
        "rewrite:async",
        "9.7.10.28.4.1",
        [15533, 15518, 15521, 15534, 15538, 38836, 38837],
        [
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes",
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes",
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes",
            "cp.async.bulk.global.shared::cta.bulk_group",
            "cp.async.bulk.relaxed.gpu.global.shared::cta.bulk_group.b128",
        ],
    ),
    _R(
        r"cp\.reduce\.async\.bulk(?:\.(?:relaxed|cta|cluster|gpu|sys))*\.(?:shared::cluster|global)",
        90,
        "rewrite:async",
        "9.7.10.28.4.2",
        [15738],
        [
            "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32",
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32",
        ],
    ),
    _R(
        r"cp\.async\.bulk\.prefetch\.L2",
        90,
        "rewrite:async",
        "9.7.10.28.4.3",
        [15778],
        ["cp.async.bulk.prefetch.L2.global"],
    ),
    _R(
        r"multimem\.cp\.async\.bulk",
        90,
        "rewrite:async",
        "9.7.10.28.4.4",
        [15832, 15835],
        [
            "multimem.cp.async.bulk.global.shared::cta.bulk_group",
            "multimem.cp.async.bulk.relaxed.cta.global.shared::cta.bulk_group.b128",
        ],
        "rules_misc: one replica, plain cp.async.bulk via rules_async",
    ),
    _R(
        r"multimem\.cp\.reduce\.async\.bulk",
        90,
        "rewrite:async",
        "9.7.10.28.4.5",
        [15938],
        [
            "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32",
            "multimem.cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16",
        ],
        "rules_misc: one replica, plain cp.reduce.async.bulk via rules_async",
    ),
    _R(
        r"cp\.async\.bulk\.tensor",
        90,
        "rewrite:async",
        "9.7.10.28.5.3",
        [16406, 16389, 16392, 16407],
        [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
            "cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group",
        ],
    ),
    _R(
        r"cp\.reduce\.async\.bulk\.tensor",
        90,
        "rewrite:async",
        "9.7.10.28.5.4",
        [16569],
        ["cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"],
    ),
    _R(
        r"cp\.async\.bulk\.prefetch\.tensor",
        90,
        "rewrite:async",
        "9.7.10.28.5.5",
        [16702],
        ["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"],
    ),
    _R(
        r"cp\.async\.bulk\.commit_group",
        90,
        "rewrite:async",
        "9.7.10.28.6.1",
        [16770],
        ["cp.async.bulk.commit_group"],
    ),
    _R(
        r"cp\.async\.bulk\.wait_group",
        90,
        "rewrite:async",
        "9.7.10.28.6.2",
        [16801],
        ["cp.async.bulk.wait_group", "cp.async.bulk.wait_group.read"],
    ),
    _R(
        r"tensormap\.replace",
        90,
        "rewrite:async",
        "9.7.10.29",
        [17083, 17118, 37575, 37633],
        ["tensormap.replace.tile.global_address.global.b1024.b64"],
    ),
    _R(
        r"tensormap\.query",
        90,
        "rewrite:async",
        "9.7.10.30",
        [17146, 37643],
        ["tensormap.query.alloc_class_large.shared::cta.b1024"],
    ),
    # ---- barrier.cluster (9.7.15.3) ----
    _R(
        r"barrier\.cluster",
        90,
        "inline",
        "9.7.15.3",
        [19604],
        [
            "barrier.cluster.arrive",
            "barrier.cluster.arrive.release.aligned",
            "barrier.cluster.wait.acquire",
        ],
        "software split barrier, cluster of one CTA",
    ),
    # ---- fence / membar (9.7.15.4, A: rules_async) ----
    _R(
        r"fence\.proxy\.tensormap::generic",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19756],
        [
            "fence.proxy.tensormap::generic.release.gpu",
            "fence.proxy.tensormap::generic.acquire.cta",
        ],
    ),
    _R(
        r"fence\.mbarrier_init",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19754],
        ["fence.mbarrier_init.release.cluster"],
    ),
    _R(
        r"fence\.alias",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19762],
        ["fence.alias.sync_restrict::read.release.sys"],
    ),
    _R(
        r"fence\.proxy\.alias\.[\w:]+",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19761],
        ["fence.proxy.alias.release.gpu", "fence.proxy.alias.acquire.sys"],
    ),
    _R(
        rf"fence{Q}\.sync_restrict(?:::\w+)*",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19758, 19760],
        [
            "fence.acquire.sync_restrict::shared::cluster.cluster",
            "fence.release.sync_restrict::shared::cta.cluster",
            "fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster",
        ],
    ),
    _R(
        r"(?:fence|membar)\.proxy\.async(?:::generic)?",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19755],
        [
            "fence.proxy.async",
            "fence.proxy.async.shared::cta",
            "fence.proxy.async.global",
            "membar.proxy.async",
        ],
    ),
    _R(
        r"fence\.(?:acquire|release)",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19757],
        ["fence.acquire.gpu", "fence.release.cta"],
    ),
    _R(
        rf"fence{Q}\.cluster",
        90,
        "rewrite:async",
        "9.7.15.4",
        [19753],
        ["fence.acq_rel.cluster", "fence.sc.cluster", "fence.cluster"],
    ),
    # ---- atom (9.7.15.5) ----
    _R(
        rf"atom{Q}\.b128",
        90,
        "lib:wgemu_atom_b128",
        "9.7.15.5",
        [20123],
        ["atom.global.cas.b128", "atom.relaxed.gpu.shared::cta.exch.b128"],
        "global lock table (Ctx.global_var)",
    ),
    _R(
        rf"atom{Q}\.add\.noftz\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.15.5",
        [20118],
        ["atom.global.add.noftz.bf16", "atom.add.noftz.bf16x2"],
    ),
    _R(
        rf"atom{Q}\.add\.noftz\.f32",
        90,
        "inline",
        "9.7.15.5",
        [20124],
        ["atom.global.add.noftz.f32"],
    ),
    _R(
        rf"atom{Q}\.v[248]",
        90,
        "inline",
        "9.7.15.5",
        [20122],
        [
            "atom.global.add.v2.f32",
            "atom.global.min.noftz.v8.f16",
            "atom.add.noftz.v4.f16x2",
        ],
        "one atom per element",
    ),
    _R(
        rf"atom{Q}\.cluster",
        90,
        "inline",
        "9.7.15.5",
        [20119],
        ["atom.relaxed.cluster.global.add.u32", "atom.cluster.global.cas.b32"],
    ),
    _R(
        rf"atom{Q}\.shared::cluster",
        90,
        "inline",
        "9.7.15.5",
        [20121],
        ["atom.shared::cluster.add.u32"],
    ),
    # ---- red (9.7.15.6, 9.7.15.7) ----
    _R(
        rf"red(?!\.async){Q}\.add\.noftz\.bf16(?:x2)?",
        90,
        "inline",
        "9.7.15.6",
        [20422],
        ["red.global.add.noftz.bf16", "red.add.noftz.bf16x2"],
    ),
    _R(
        rf"red(?!\.async){Q}\.add\.noftz\.f32",
        90,
        "inline",
        "9.7.15.6",
        [20427],
        ["red.global.add.noftz.f32"],
    ),
    _R(
        rf"red(?!\.async){Q}\.v[248]",
        90,
        "inline",
        "9.7.15.6",
        [20426],
        ["red.global.add.v4.f32", "red.global.max.noftz.v8.bf16"],
    ),
    _R(
        rf"red(?!\.async){Q}\.cluster",
        90,
        "inline",
        "9.7.15.6",
        [20423],
        [
            "red.relaxed.cluster.global.add.u32",
            "red.release.cluster.shared::cta.and.b32",
        ],
    ),
    _R(
        rf"red(?!\.async){Q}\.shared::cluster",
        90,
        "inline",
        "9.7.15.6",
        [20425],
        ["red.shared::cluster.add.u32"],
    ),
    _R(
        r"red\.async",
        90,
        "rewrite:async",
        "9.7.15.7",
        [20556],
        [
            "red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32"
        ],
    ),
    # ---- griddepcontrol / elect.sync (9.7.15.14, 9.7.15.15) ----
    _R(
        r"griddepcontrol",
        90,
        "noop",
        "9.7.15.14",
        [20825],
        ["griddepcontrol.launch_dependents", "griddepcontrol.wait"],
        "kernels run without programmatic dependent launch",
    ),
    _R(r"elect\.sync", 90, "inline", "9.7.15.15", [20848], ["elect.sync"]),
    # ---- mbarrier (9.7.15.16, A: rules_async) ----
    _R(
        r"mbarrier\.try_wait",
        90,
        "rewrite:async",
        "9.7.15.16.19",
        [21743],
        [
            "mbarrier.try_wait.shared::cta.b64",
            "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64",
        ],
    ),
    _R(
        r"mbarrier\.expect_tx",
        90,
        "rewrite:async",
        "9.7.15.16.14",
        [21281],
        ["mbarrier.expect_tx.relaxed.cta.shared::cta.b64"],
    ),
    _R(
        r"mbarrier\.complete_tx",
        90,
        "rewrite:async",
        "9.7.15.16.15",
        [21340],
        ["mbarrier.complete_tx.relaxed.cta.shared::cta.b64"],
    ),
    _R(
        r"mbarrier\.check_layout",
        90,
        "rewrite:async",
        "9.7.15.16.21",
        [21899],
        ["mbarrier.check_layout.layout::v0.shared::cta.b64"],
    ),
    _R(
        rf"mbarrier{Q}\.layout::v[01]",
        90,
        "rewrite:async",
        "9.7.15.16.12",
        [21168, 21872],
        [
            "mbarrier.init.layout::v0.shared::cta.b64",
            "mbarrier.pending_count.layout::v0.b64",
        ],
    ),
    _R(
        rf"mbarrier\.arrive(?:_drop)?{Q}\.expect_tx",
        90,
        "rewrite:async",
        "9.7.15.16.16",
        [21425, 21535],
        [
            "mbarrier.arrive.expect_tx.shared::cta.b64",
            "mbarrier.arrive_drop.expect_tx.shared::cta.b64",
        ],
    ),
    _R(
        rf"mbarrier{Q}\.phase_type::\w+",
        90,
        "rewrite:async",
        "9.7.15.16.19",
        [21746],
        ["mbarrier.test_wait.phase_type::primary.shared::cta.b64"],
    ),
    _R(
        rf"mbarrier{Q}\.relaxed",
        90,
        "rewrite:async",
        "9.7.15.16.16",
        [21428, 21538, 21745],
        [
            "mbarrier.arrive.relaxed.cta.shared::cta.b64",
            "mbarrier.arrive_drop.relaxed.cta.shared::cta.b64",
            "mbarrier.test_wait.relaxed.cta.shared::cta.b64",
        ],
    ),
    _R(
        rf"mbarrier{Q}\.shared::cluster",
        90,
        "rewrite:async",
        "9.7.15.16.16",
        [21426, 21536],
        [
            "mbarrier.arrive.release.cluster.shared::cluster.b64",
            "mbarrier.arrive_drop.shared::cluster.b64",
        ],
    ),
    _R(
        rf"mbarrier{Q}\.cluster",
        90,
        "rewrite:async",
        "9.7.15.16.16",
        [21427, 21537, 21744],
        [
            "mbarrier.arrive.release.cluster.shared::cta.b64",
            "mbarrier.arrive_drop.release.cluster.shared::cta.b64",
            "mbarrier.test_wait.acquire.cluster.shared::cta.b64",
        ],
    ),
    _R(
        r"tensormap\.cp_fenceproxy",
        90,
        "rewrite:async",
        "9.7.15.17",
        [21938],
        [
            "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"
        ],
    ),
    # ---- warp-level matrix instructions (9.7.16) ----
    _R(
        rf"mma\.sync\.aligned\.m16n8k(?:4|8|16)\.row\.col{Q}\.f64",
        90,
        "lib:wgemu_mma_f64",
        "9.7.16.5.14",
        [25165],
        [
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64",
            "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64.rn",
            "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64",
        ],
        "sequential IEEE fma per k (numerics.h model) with .rnd",
    ),
    _R(
        rf"mma\.sync\.aligned\.m16n8k(?:16|32)\.row\.col{Q}\.e(?:4m3|5m2)",
        89,
        "lib:wgemu_mma_fp8",
        "9.7.16.5.14",
        [25166],
        [
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32",
            "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16",
        ],
        "T-FDPA, Ada or Hopper parameters by .target",
    ),
    _R(
        rf"mma\.sp(?:::ordered_metadata)?\.sync\.aligned{Q}\.e(?:4m3|5m2)",
        89,
        "lib:wgemu_mma_sp_fp8",
        "9.7.16.6.3",
        [26570],
        [
            "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32",
            "mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32",
        ],
    ),
    _R(
        rf"ldmatrix\.sync\.aligned\.m8n16{Q}\.s8\.s4",
        90,
        "inline",
        "9.7.16.5.15",
        [25669, 37959],
        [
            "ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4",
            "ldmatrix.sync.aligned.m8n16.x4.s8.s4",
        ],
    ),
    _R(
        r"stmatrix\.sync\.aligned\.m8n8",
        90,
        "inline",
        "9.7.16.5.16",
        [25799],
        [
            "stmatrix.sync.aligned.m8n8.x4.shared.b16",
            "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16",
        ],
    ),
    # ---- wgmma (9.7.17, W: rules_wgmma) ----
    _R(
        r"wgmma\.mma_async\.sync",
        90,
        "rewrite:wgmma",
        "9.7.17.5.2",
        [27779],
        ["wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16"],
    ),
    _R(
        r"wgmma\.mma_async\.sp",
        90,
        "rewrite:wgmma",
        "9.7.17.6.3",
        [28275],
        ["wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.f16.f16"],
    ),
    _R(
        r"wgmma\.fence",
        90,
        "rewrite:wgmma",
        "9.7.17.7.1",
        [28316],
        ["wgmma.fence.sync.aligned"],
    ),
    _R(
        r"wgmma\.commit_group",
        90,
        "rewrite:wgmma",
        "9.7.17.7.2",
        [28358],
        ["wgmma.commit_group.sync.aligned"],
    ),
    _R(
        r"wgmma\.wait_group",
        90,
        "rewrite:wgmma",
        "9.7.17.7.3",
        [28384, 37656],
        ["wgmma.wait_group.sync.aligned"],
    ),
    # ---- setmaxnreg (9.7.21.5) ----
    _R(
        r"setmaxnreg",
        90,
        "noop",
        "9.7.21.5",
        [34467, 37664],
        ["setmaxnreg.inc.sync.aligned.u32", "setmaxnreg.dec.sync.aligned.u32"],
        "register count fixed at compile time",
    ),
    # ---- special registers (10.x) ----
    _R(
        r"%is_explicit_cluster",
        90,
        "inline",
        "10.11",
        [34864],
        ["%is_explicit_cluster"],
    ),
    _R(
        r"%clusterid",
        90,
        "inline",
        "10.12",
        [34886],
        ["%clusterid.x", "%clusterid.w"],
        "= %ctaid",
    ),
    _R(r"%nclusterid", 90, "inline", "10.13", [34908], ["%nclusterid.y"], "= %nctaid"),
    _R(r"%cluster_ctaid", 90, "inline", "10.14", [34933], ["%cluster_ctaid.x"], "= 0"),
    _R(
        r"%cluster_nctaid", 90, "inline", "10.15", [34956], ["%cluster_nctaid.z"], "= 1"
    ),
    _R(
        r"%cluster_ctarank", 90, "inline", "10.16", [34976], ["%cluster_ctarank"], "= 0"
    ),
    _R(
        r"%cluster_nctarank",
        90,
        "inline",
        "10.17",
        [34991],
        ["%cluster_nctarank"],
        "= 1",
    ),
    _R(
        r"%aggr_smem_size",
        90,
        "inline",
        "10.31",
        [35259],
        ["%aggr_smem_size"],
        "= %total_smem_size + %reserved_smem_offset_cap [inference]",
    ),
    # ---- directives (5.4.8, 11.x) ----
    _R(
        r"\.unified",
        90,
        "strip",
        "5.4.8",
        [3128, 36025],
        [
            ".global .attribute(.unified(0x1, 0x2)) .u32 wgemu_g;",
            ".visible .func .attribute(.unified(0x3, 0x4)) wgemu_f()\n{\n\tret;\n}",
        ],
        ".attribute(.unified(...)) removed by the core (T1 diff)",
    ),
    _R(
        r"\.reqnctapercluster",
        90,
        "strip",
        "11.8.1",
        [36899, 896, 917],
        [".visible .entry wgemu_k1() .reqnctapercluster 1, 1, 1\n{\n\tret;\n}"],
        "core: removed for a cluster of one CTA, error otherwise",
    ),
    _R(
        r"\.explicitcluster",
        90,
        "strip",
        "11.8.2",
        [36918],
        [".visible .entry wgemu_k2() .explicitcluster\n{\n\tret;\n}"],
    ),
    _R(
        r"\.maxclusterrank",
        90,
        "strip",
        "11.8.3",
        [36936],
        [".visible .entry wgemu_k3() .maxclusterrank 1\n{\n\tret;\n}"],
    ),
    _R(
        r"\.blocksareclusters",
        90,
        "strip",
        "11.9.1",
        [36962],
        [
            ".visible .entry wgemu_k4() .reqntid 32, 1, 1 .blocksareclusters\n{\n\tret;\n}"
        ],
    ),
    # check.py reports any .target above sm_80 itself; this row records the requirement.
    _R(
        r"\.target",
        86,
        "rewrite:core",
        "11.1.2",
        [
            35422,
            35423,
            35438,
            35578,
            35580,
            35582,
            35584,
            35594,
            35598,
            35600,
            35602,
            35604,
            35606,
            35608,
            35748,
            35749,
            35750,
            35751,
            35752,
            35753,
            35781,
            35782,
            37427,
            38549,
            38859,
            38885,
            38886,
            38955,
            38988,
        ],
        [".target sm_90a", ".target sm_86", ".target sm_89"],
        "the core retargets to sm_80",
    ),
)

FEATURES: tuple[Feature, ...] = (
    Feature(
        "static_smem_over_48k",
        90,
        "unsupported",
        "5.1.7",
        (2280, 2286),
        (".shared .align 16 .b8 wgemu_big[65536];",),
        "sm_90a static .shared above 48 KB per CTA; phase 2, not translated, ptxas "
        "-arch=sm_80 rejects it",
    ),
    Feature(
        "ld_unified_operand",
        90,
        "inline",
        "9.7.10.8",
        (13258,),
        ("ld.global.u32 %r1, [wgemu_g].unified;",),
        "address operand qualifier, dropped by rules_misc",
    ),
    Feature(
        "mbarrier_arrive_count_without_noComplete",
        90,
        "rewrite:async",
        "9.7.15.16.16",
        (21424, 21534),
        (
            "mbarrier.arrive.shared::cta.b64 %rd1, [bar], 2;",
            "mbarrier.arrive_drop.shared::cta.b64 %rd1, [bar], 2;",
        ),
        "the count operand is the only difference from the sm_80 form",
    ),
    Feature(
        "mbarrier_wait_report_operands",
        90,
        "rewrite:async",
        "9.7.15.16.19",
        (21746,),
        (
            "mbarrier.test_wait.phase_type::primary.shared::cta.b64 %p1|%p2, [bar], %rd1;",
        ),
        "reportPredicate / reportValue destinations",
    ),
)

# sm_80 forms no inventory row may report (statements, then module-level text).
SM80_NATIVE: tuple[str, ...] = (
    "add.u32",
    "add.sat.s32",
    "add.f16x2",
    "add.rn.f32",
    "add.rn.f16",
    "sub.f16x2",
    "mul.rn.f16",
    "min.s32",
    "max.u32",
    "min.f16x2",
    "max.NaN.bf16x2",
    "min.bf16",
    "min.ftz.NaN.f32",
    "max.f32",
    "fma.rn.bf16",
    "fma.rn.relu.bf16x2",
    "fma.rn.ftz.f16",
    "fma.rn.f32",
    "tanh.approx.f32",
    "tanh.approx.f16x2",
    "ex2.approx.ftz.f32",
    "ex2.approx.f16",
    "set.eq.u32.f32",
    "set.lt.u16.f16",
    "set.gt.u32.f16x2",
    "setp.eq.f16",
    "setp.lt.ftz.f16x2",
    "setp.ne.f32",
    "ld.global.u32",
    "ld.relaxed.gpu.global.u32",
    "ld.acquire.sys.global.b64",
    "ld.shared::cta.u32",
    "ld.global.nc.L1::no_allocate.v4.u32",
    "ld.global.L2::256B.v2.b64",
    "ldu.global.u32",
    "ld.volatile.shared.u16",
    "st.global.u32",
    "st.release.gpu.global.u32",
    "st.shared::cta.v4.b32",
    "prefetch.global.L1",
    "prefetch.global.L2::evict_last",
    "prefetchu.L1",
    "prefetch.L2",
    "isspacep.shared",
    "isspacep.shared::cta",
    "isspacep.global",
    "cvta.to.shared.u64",
    "cvta.shared::cta.u64",
    "cvta.global.u64",
    "cvt.rn.f16.f32",
    "cvt.rn.bf16.f32",
    "cvt.rz.relu.bf16.f32",
    "cvt.rn.bf16x2.f32",
    "cvt.rn.relu.satfinite.bf16x2.f32",
    "cvt.f32.bf16",
    "cvt.f32.f16",
    "cvt.rna.tf32.f32",
    "cvt.rna.satfinite.tf32.f32",
    "cvt.rzi.s32.f32",
    "cvt.rn.f16x2.f32",
    "cvt.u32.u64",
    "cp.async.ca.shared.global",
    "cp.async.cg.shared.global.L2::128B",
    "cp.async.commit_group",
    "cp.async.wait_group",
    "cp.async.wait_all",
    "cp.async.mbarrier.arrive.shared.b64",
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64",
    "barrier.sync",
    "barrier.sync.aligned",
    "barrier.arrive",
    "bar.sync",
    "bar.warp.sync",
    "fence.acq_rel.gpu",
    "fence.sc.sys",
    "fence.acq_rel.cta",
    "fence.sc.gpu",
    "fence.proxy.alias",
    "membar.proxy.alias",
    "membar.gl",
    "membar.cta",
    "membar.sys",
    "atom.global.add.f32",
    "atom.add.noftz.f16",
    "atom.global.add.noftz.f16x2",
    "atom.global.cas.b64",
    "atom.shared::cta.min.s32",
    "atom.relaxed.gpu.global.exch.b32",
    "atom.acq_rel.sys.global.add.u64",
    "atom.global.add.L2::cache_hint.f32",
    "red.global.add.f32",
    "red.global.add.noftz.f16x2",
    "red.relaxed.gpu.global.add.u32",
    "red.shared::cta.and.b32",
    "redux.sync.add.u32",
    "mbarrier.init.shared.b64",
    "mbarrier.init.shared::cta.b64",
    "mbarrier.arrive.shared.b64",
    "mbarrier.arrive.noComplete.shared.b64",
    "mbarrier.arrive.release.cta.shared::cta.b64",
    "mbarrier.arrive_drop.shared::cta.b64",
    "mbarrier.arrive_drop.noComplete.shared.b64",
    "mbarrier.test_wait.shared.b64",
    "mbarrier.test_wait.parity.shared::cta.b64",
    "mbarrier.test_wait.acquire.cta.shared::cta.b64",
    "mbarrier.inval.shared.b64",
    "mbarrier.pending_count.b64",
    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64",
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32",
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32",
    "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32",
    "wmma.mma.sync.aligned.row.col.m8n8k4.f64.f64.f64.f64",
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16",
    "movmatrix.sync.aligned.m8n8.trans.b16",
    "activemask.b32",
    "match.any.sync.b32",
    "vote.sync.ballot.b32",
    "shfl.sync.idx.b32",
    "nanosleep.u32",
    "createpolicy.fractional.L2::evict_last.b64",
    "discard.global.L2",
    "applypriority.global.L2::evict_normal",
    "mov.u32 \t%r1, %ctaid.x",
    "mov.u32 \t%r1, %nctaid.y",
    "mov.u32 \t%r1, %tid.x",
    "mov.u32 \t%r1, %total_smem_size",
    "mov.u32 \t%r1, %dynamic_smem_size",
    "mov.u32 \t%r1, %reserved_smem_offset_cap",
    "mov.u32 \t%r1, %smid",
    "mov.u64 \t%rd1, %globaltimer",
    "mov.u32 \t%r1, %lanemask_eq",
)
SM80_NATIVE_MODULE: tuple[str, ...] = (
    ".global .attribute(.managed) .s32 wgemu_m;",
    ".global .align 16 .b8 wgemu_x[16];",
    ".shared .align 16 .b8 wgemu_s[1024];",
    ".extern .shared .align 16 .b8 wgemu_dyn[];",
    ".visible .entry wgemu_n1() .maxntid 128, 1, 1 .minnctapersm 2\n{\n\tret;\n}",
    ".visible .entry wgemu_n2() .reqntid 64, 1, 1 .maxnreg 128\n{\n\tret;\n}",
)


# ---------------------------------------------------------------------------------------------
# ptx.txt
# ---------------------------------------------------------------------------------------------


def read_ptx(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(
            f"gen_inventory: PTX ISA text not found: {path} (set WGEMU_PTX_ISA)"
        )
    return path.read_text(encoding="utf-8").split("\n")


def sections(lines: list[str]) -> tuple[int, list[str | None]]:
    """(index of the first body line, section number of each line). The table of contents
    repeats the chapter headings; the body starts at the second '1.1. Scalable' heading."""
    idx = [i for i, ln in enumerate(lines) if ln.startswith("1.1. Scalable")]
    if len(idx) < 2:
        raise SystemExit("gen_inventory: cannot find the start of the PTX ISA body")
    start = idx[1] - 3
    out: list[str | None] = [None] * len(lines)
    cur = None
    for i in range(start, len(lines)):
        m = _HEAD_RE.match(lines[i])
        if m:
            cur = m.group(1)
        out[i] = cur
    return start, out


def requirement_lines(lines: list[str], start: int) -> list[int]:
    return [i + 1 for i in range(start, len(lines)) if SM_RE.search(lines[i])]


def _all_lines() -> list[tuple[str, tuple[int, ...], int, str]]:
    out = [(r.pattern, r.lines, r.min_sm, r.section) for r in ROWS]
    out += [(f"#feature {f.name}", f.lines, f.min_sm, f.section) for f in FEATURES]
    return out


def check_citations(lines: list[str], secs: list[str | None]) -> list[str]:
    errs: list[str] = []
    for name, cited, min_sm, section in _all_lines():
        if not cited:
            errs.append(f"{name}: no ptx.txt line cited")
            continue
        sms: list[int] = []
        for ln in cited:
            if not 1 <= ln <= len(lines):
                errs.append(f"{name}: line {ln} is outside ptx.txt")
                continue
            found = [int(x) for x in SM_RE.findall(lines[ln - 1])]
            if not found:
                errs.append(
                    f"{name}: line {ln} names no sm_86..sm_99 target: {lines[ln - 1]!r}"
                )
            sms += found
        if secs[cited[0] - 1] != section:
            errs.append(
                f"{name}: first cited line {cited[0]} is in section {secs[cited[0] - 1]}, not {section}"
            )
        if sms and min(sms) != min_sm:
            errs.append(
                f"{name}: cited lines name sm_{min(sms)} at least, row says min_sm {min_sm}"
            )
    return errs


def audit(lines: list[str], start: int, secs: list[str | None]) -> list[str]:
    cited = {ln for _, ls, _, _ in _all_lines() for ln in ls}
    return [
        f"ptx.txt:{ln} [{secs[ln - 1]}] not cited: {lines[ln - 1].strip()}"
        for ln in requirement_lines(lines, start)
        if ln not in cited
    ]


# ---------------------------------------------------------------------------------------------
# Probes through check.find_remaining
# ---------------------------------------------------------------------------------------------


def probe_module(
    body: list[str], module: tuple[str, ...] | list[str] = (), target: str = "sm_80"
) -> str:
    stmts = "\n".join(f"\t{s};" for s in body)
    mod = "\n\n".join(module)
    return (
        f".version 8.7\n.target {target}\n.address_size 64\n\n{mod}\n\n"
        f".visible .entry wgemu_inventory_probe()\n{{\n{stmts}\n\tret;\n}}\n"
    )


def _probe(row: Row, eg: str) -> tuple[str, str]:
    """(ptx module, hit kind) for one example of ``row``."""
    kind = check._row_kind(row.pattern)
    if row.pattern == r"\.target":
        return probe_module([], target=eg.split()[1]), "target"
    if kind == "operand":
        return probe_module([f"mov.u32 \t%r1, {eg}"]), kind
    if kind == "directive":
        return probe_module([], [eg]), kind
    return probe_module([eg]), kind


def check_probes(inv: list[check.InvRow]) -> list[str]:
    errs: list[str] = []
    for row in ROWS:
        if not row.eg:
            errs.append(f"{row.pattern}: no eg")
        for eg in row.eg:
            ptx, kind = _probe(row, eg)
            hits = check.find_remaining(ptx, inv)
            if kind == "target":
                ok = any(h.kind == "target" for h in hits)
            elif kind == "directive":
                ok = any(h.kind == kind and h.pattern == row.pattern for h in hits)
            else:
                mine = [h for h in hits if h.kind == kind]
                ok = bool(mine) and mine[0].pattern == row.pattern
            if not ok:
                got = ", ".join(f"{h.kind}:{h.pattern}" for h in hits) or "nothing"
                errs.append(f"{row.pattern}: eg {eg!r} reported as {got}")
    ptx = probe_module(list(SM80_NATIVE), list(SM80_NATIVE_MODULE))
    for h in check.find_remaining(ptx, inv):
        errs.append(f"sm_80 form reported: {h.kind} {h.token} by {h.pattern!r}")
    return errs


# ---------------------------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------------------------


def _cell(s: str) -> str:
    if "\t" in s or "\n" in s:
        raise ValueError(f"tab or newline in inventory cell {s!r}")
    return s


def _note(note: str, cited: tuple[int, ...], eg: tuple[str, ...]) -> str:
    refs = "ptx.txt:" + ",".join(str(x) for x in cited)
    first = eg[0].split("\n")[0] if eg else ""
    return " ".join(x for x in (note, refs, f"eg={first}" if first else "") if x)


def tsv_text(ptx_sha256: str) -> str:
    out = [
        "# wgmma_emu inventory of sm_86..sm_90a PTX requirements (PLAN section 5).",
        "# Generated by tools/ptxemu/gen_inventory.py from ext/ptxisa/ptx.txt",
        f"# (sha256 {ptx_sha256}); edit ROWS / FEATURES there, not this file.",
        "# Columns: opcode_regex, min_sm, handling, ptx_section, note (tab separated).",
        "# '#feature' lines: requirements without a token of their own; check.py skips them,",
        "# tests/misc/test_inventory_coverage.py checks them (WGEMU-COVERS-FEATURE markers).",
        "opcode_regex\tmin_sm\thandling\tptx_section\tnote",
    ]
    for r in ROWS:
        cols = (
            r.pattern,
            str(r.min_sm),
            r.handling,
            r.section,
            _note(r.note, r.lines, r.eg),
        )
        out.append("\t".join(_cell(c) for c in cols))
    for f in FEATURES:
        cols = (
            f"#feature\t{f.name}",
            str(f.min_sm),
            f.handling,
            f.section,
            _note(f.note, f.lines, f.eg),
        )
        out.append("\t".join(cols))
    return "\n".join(out) + "\n"


def load_features(path: str | Path = DEFAULT_OUT) -> list[tuple[str, int, str, str]]:
    """``#feature`` lines of inventory.tsv as (name, min_sm, handling, section)."""
    out = []
    for raw in Path(path).read_text().splitlines():
        if raw.startswith("#feature\t"):
            cols = raw.split("\t")
            out.append((cols[1], int(cols[2]), cols[3], cols[4]))
    return out


def load_inventory(path: str | Path = DEFAULT_OUT) -> list[check.InvRow]:
    return check.load_inventory(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ptxemu.gen_inventory", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--ptx", default=str(DEFAULT_PTX), help="PTX ISA text (ptx.txt)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="inventory.tsv to write")
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="compare with --out instead of writing it",
    )
    a = ap.parse_args(argv)
    ptx_path = Path(a.ptx)
    lines = read_ptx(ptx_path)
    start, secs = sections(lines)
    sha = hashlib.sha256(ptx_path.read_bytes()).hexdigest()
    text = tsv_text(sha)
    errs = check_citations(lines, secs) + audit(lines, start, secs)
    out = Path(a.out)
    if a.check_only:
        if not out.is_file() or out.read_text() != text:
            errs.append(
                f"{out} is not the output of gen_inventory.py (run it without --check-only)"
            )
        inv_path = out
    else:
        out.write_text(text)
        inv_path = out
    try:
        inv = check.load_inventory(inv_path)
    except check.InventoryError as e:
        errs.append(f"check.load_inventory: {e}")
        inv = []
    if inv:
        if len(inv) != len(ROWS):
            errs.append(
                f"check.load_inventory read {len(inv)} rows, ROWS has {len(ROWS)}"
            )
        errs += check_probes(inv)
    n_req = len(requirement_lines(lines, start))
    for e in errs:
        print(f"gen_inventory: {e}", file=sys.stderr)
    print(
        f"gen_inventory: {len(ROWS)} rows, {len(FEATURES)} features, {n_req} requirement lines, "
        f"{len(errs)} error(s); {'checked' if a.check_only else 'wrote'} {out}"
    )
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
