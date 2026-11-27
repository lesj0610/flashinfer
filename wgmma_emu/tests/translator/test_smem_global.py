"""E4 shadow shared memory (ptxemu.rules_smem): CTA shared window in a global pool.

CPU only. A synthetic library stands in for the real one: it defines the header variable and the
lib functions the translator calls; the checks are on the translated text.
"""

from __future__ import annotations

import re

import pytest
from ptxemu.core import TranslateError, translate

HDR = "__wgemu_e4_hdr_v1"

LIB = f"""
.version 9.0
.target sm_80
.address_size 64

.visible .shared .align 16 .b8 {HDR}[32];

.visible .func wgemu_e4_enter(
\t.param .b64 p0,
\t.param .b32 p1,
\t.param .b64 p2,
\t.param .b64 p3,
\t.param .b64 p4,
\t.param .b32 p5
)
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+24];
\tret;
}}

.visible .func wgemu_e4_exit()
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+24];
\tret;
}}

.visible .func wgemu_e4_cp_async(
\t.param .b32 p0,
\t.param .b64 p1,
\t.param .b32 p2,
\t.param .b32 p3
)
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+24];
\tret;
}}

.visible .func wgemu_e4_cp_async_arrive(
\t.param .b32 p0,
\t.param .b32 p1
)
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+24];
\tret;
}}

.visible .func (.param .b32 func_retval0) wgemu_e4_dyn_smem()
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+12];
\tst.param.b32 [func_retval0+0], %r1;
\tret;
}}

.visible .func (.param .b32 func_retval0) wgemu_e4_total_smem()
{{
\t.reg .b32 %r<2>;
\tld.shared.u32 %r1, [{HDR}+12];
\tst.param.b32 [func_retval0+0], %r1;
\tret;
}}
"""

PRE = """
.version 9.0
.target sm_90a
.address_size 64

"""

SHARED_KERNEL = (
    PRE
    + """
.extern .shared .align 16 .b8 dyn_buf[];
.shared .align 4 .b8 tile[256];

.visible .entry kshared(
\t.param .u64 kshared_param_0
)
{
\t.reg .b32 \t%r<6>;
\t.reg .b64 \t%rd<4>;

\tld.param.u64 \t%rd1, [kshared_param_0];
\tmov.u32 \t%r1, %tid.x;
\tmov.u32 \t%r2, tile;
\tshl.b32 \t%r3, %r1, 2;
\tadd.s32 \t%r4, %r2, %r3;
\tst.shared.u32 \t[%r4], %r1;
\tbar.sync \t0;
\tld.shared.u32 \t%r5, [%r4];
\tst.global.u32 \t[%rd1], %r5;
\tret;
}
"""
)

PLAIN_KERNEL = (
    PRE
    + """
.visible .entry kplain(
\t.param .u64 kplain_param_0
)
{
\t.reg .b32 \t%r<3>;
\t.reg .b64 \t%rd<2>;

\tld.param.u64 \t%rd1, [kplain_param_0];
\tmov.u32 \t%r1, %tid.x;
\tst.global.u32 \t[%rd1], %r1;
\tret;
}
"""
)


def e4(src: str, **kw):
    rep: dict = {}
    out = translate(
        src, lib_src=LIB, smem_global=True, report=rep, warn_stream=None, **kw
    )
    return out, rep


def user_part(out: str) -> str:
    begin = "// ---- wgemu emulation library (pruned) ----"
    end = "// ---- end of wgemu emulation library ----"
    if begin in out:
        a = out.index(begin)
        b = out.index(end) + len(end)
        return out[:a] + out[b:]
    return out


def test_e4_off_has_no_e4_symbols():
    out = translate(SHARED_KERNEL, lib_src=LIB, warn_stream=None)
    assert "__wgemu_e4_pool_v1" not in out
    assert "__wgemu_e4_meta" not in out
    assert ".shared .align 4 .b8 tile[256]" in out


def test_e4_rewrites_shared_to_global_window():
    out, rep = e4(SHARED_KERNEL)
    user = user_part(out)
    assert "__wgemu_e4_meta" in user
    assert "__wgemu_e4_pool_v1" in user and "__wgemu_e4_lease_v1" in user
    assert "__wgemu_e4_stats_v1" in user
    assert re.search(r"__wgemu_e4f_[0-9a-f]{16}", user)
    assert "wgemu_e4_enter" in user and "wgemu_e4_exit" in user
    # no shared access of the user kernel survives
    shared = [
        ln for ln in re.sub(r"//[^\n]*", "", user).splitlines() if ".shared" in ln
    ]
    assert all(HDR in ln for ln in shared), shared
    assert "st.global.u32" in user and "ld.global.u32" in user
    e = rep["e4"]
    assert e["slots"] == 512
    assert len(e["entries"]) == 1 and e["non_e4_entries"] == []
    assert "tile" in e["removed_shared"] and "dyn_buf" in e["removed_shared"]


@pytest.mark.parametrize(
    "line",
    [
        "atom.shared.add.f32 \t%r5, [%r4], %r1;",
        "atom.shared.add.noftz.f32 \t%r5, [%r4], %r1;",
        "red.shared.add.f32 \t[%r4], %r1;",
    ],
)
def test_e4_shared_add_f32_keeps_subnormals(line):
    # Shared .add.f32 keeps subnormals, global .add.f32 flushes them: the relocated access must
    # be a CAS loop, never the native global form.
    src = SHARED_KERNEL.replace("\tbar.sync \t0;\n", f"\t{line}\n\tbar.sync \t0;\n")
    user = re.sub(r"//[^\n]*", "", user_part(e4(src)[0]))
    assert not re.search(r"\b(atom|red)\.global\.add\.(noftz\.)?f32", user), user
    assert re.search(r"atom\.global\.cas\.b32", user)
    assert re.search(r"add\.rn\.f32", user)
    off = translate(src, lib_src=LIB, warn_stream=None)
    assert re.search(r"\b(atom|red)\.shared\.add\.f32", off)


def test_e4_shared_add_u32_stays_native():
    src = SHARED_KERNEL.replace(
        "\tbar.sync \t0;\n", "\tatom.shared.add.u32 \t%r5, [%r4], 1;\n\tbar.sync \t0;\n"
    )
    user = user_part(e4(src)[0])
    assert "atom.global.add.u32" in user and "atom.global.cas" not in user


def test_e4_classification_leaves_plain_entry_alone():
    out, rep = e4(PLAIN_KERNEL)
    e = rep["e4"]
    assert e["entries"] == [] and len(e["non_e4_entries"]) == 1
    assert "__wgemu_e4_meta" not in user_part(out)
    assert "wgemu_e4_enter" not in user_part(out)


def test_e4_slots_changes_pool_size():
    out, rep = e4(SHARED_KERNEL, e4_slots=8)
    assert rep["e4"]["slots"] == 8
    assert re.search(r"__wgemu_e4_pool_v1\s*\[\s*%d\s*\]" % (8 * 0x40000), out)


@pytest.mark.parametrize(
    "body, why",
    [
        ("\tcvta.shared.u64 \t%rd2, %rd1;\n", "cvta"),
        ("\tldmatrix.sync.aligned.m8n8.x1.shared.b16 \t{%r2}, [%r1];\n", "ldmatrix"),
        ("\tisspacep.shared \t%p1, %rd1;\n", "isspacep"),
    ],
)
def test_e4_fails_closed_on_generic_shared_forms(body, why):
    src = SHARED_KERNEL.replace(
        "\tbar.sync \t0;\n", "\t.reg .pred %p<2>;\n" + body + "\tbar.sync \t0;\n"
    )
    with pytest.raises(TranslateError):
        translate(src, lib_src=LIB, smem_global=True, warn_stream=None)


def test_e4_fails_closed_on_initialized_shared():
    src = SHARED_KERNEL.replace(
        ".shared .align 4 .b8 tile[256];", ".shared .align 4 .b32 tile[1] = {5};"
    )
    with pytest.raises(TranslateError):
        e4(src)


def test_e4_refuses_delayed_delivery_and_counters_and_missing_lib():
    with pytest.raises(TranslateError, match="E1"):
        translate(
            SHARED_KERNEL,
            lib_src=LIB,
            smem_global=True,
            async_delay=True,
            warn_stream=None,
        )
    with pytest.raises(TranslateError, match="E2"):
        translate(
            SHARED_KERNEL, lib_src=LIB, smem_global=True, count=True, warn_stream=None
        )
    with pytest.raises(TranslateError, match="--lib"):
        translate(SHARED_KERNEL, smem_global=True, warn_stream=None)


def test_e4_needs_e4_library_variant():
    lib = LIB.replace(f".visible .shared .align 16 .b8 {HDR}[32];", "")
    lib = lib.replace(f"ld.shared.u32 %r1, [{HDR}+24];", "mov.u32 %r1, 0;")
    lib = lib.replace(f"ld.shared.u32 %r1, [{HDR}+12];", "mov.u32 %r1, 0;")
    with pytest.raises(TranslateError, match="WGEMU_SMEM_GLOBAL"):
        translate(SHARED_KERNEL, lib_src=lib, smem_global=True, warn_stream=None)


# ---------------------------------------------------------------------------------------------
# Explicit entry selection (--smem-global-entries / WGEMU_E4_ENTRIES)
# ---------------------------------------------------------------------------------------------

OTHER_KERNEL = """
.shared .align 4 .b8 tile2[128];

.visible .entry kother(
\t.param .u64 kother_param_0
)
{
\t.reg .b32 \t%r<6>;
\t.reg .b64 \t%rd<4>;

\tld.param.u64 \t%rd1, [kother_param_0];
\tmov.u32 \t%r1, %tid.x;
\tmov.u32 \t%r2, tile2;
\tshl.b32 \t%r3, %r1, 2;
\tadd.s32 \t%r4, %r2, %r3;
\tst.shared.u32 \t[%r4], %r1;
\tbar.sync \t0;
\tld.shared.u32 \t%r5, [%r4];
\tst.global.u32 \t[%rd1], %r5;
\tret;
}
"""

MIXED = SHARED_KERNEL + OTHER_KERNEL
CVTA_BODY = "\t.reg .pred %p<2>;\n\tcvta.shared.u64 \t%rd2, %rd1;\n"


def sel(src: str, regex, **kw):
    rep: dict = {}
    out = translate(
        src,
        lib_src=LIB,
        smem_global=True,
        smem_global_entries=regex,
        report=rep,
        warn_stream=None,
        **kw,
    )
    return out, rep


def func_text(out: str, name: str) -> str:
    m = re.search(r"\.entry %s\(.*?\n\}\n" % re.escape(name), out, re.S)
    assert m, name
    return m.group(0)


def shared_lines(text: str) -> list[str]:
    return [ln for ln in re.sub(r"//[^\n]*", "", text).splitlines() if ".shared" in ln]


def helper_module(selected_calls: bool, excluded_calls: bool) -> str:
    helper = """
.shared .align 4 .b8 hbuf[64];

.func helper(
\t.param .b64 helper_param_0
)
{
\t.reg .b32 \t%r<3>;
\t.reg .b64 \t%rd<2>;

\tmov.u32 \t%r1, hbuf;
\tmov.u32 \t%r2, 7;
\tst.shared.u32 \t[%r1], %r2;
\tret;
}
"""
    call = "\t{\n\t.param .b64 param0;\n\tst.param.b64 [param0+0], %rd1;\n\tcall.uni helper, (param0);\n\t}\n"
    s = SHARED_KERNEL.replace(
        "\tbar.sync \t0;\n", (call if selected_calls else "") + "\tbar.sync \t0;\n"
    )
    o = OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n", (call if excluded_calls else "") + "\tbar.sync \t0;\n"
    )
    return PRE + helper + s.replace(PRE, "") + o


def test_select_matching_entry_e4_other_entry_real_shared():
    out, rep = sel(MIXED, r"^kshared$")
    k = func_text(out, "kshared")
    assert "__wgemu_e4_meta" in k and "wgemu_e4_enter" in k and "wgemu_e4_exit" in k
    assert re.search(r"__wgemu_e4f_[0-9a-f]{16}", out)
    assert all(HDR in ln for ln in shared_lines(k)), shared_lines(k)
    # the excluded entry keeps its real shared memory and takes no E4 part
    o = func_text(out, "kother")
    assert "__wgemu_e4" not in o and "wgemu_e4_" not in o
    assert "st.shared.u32" in o and "ld.shared.u32" in o
    assert ".shared .align 4 .b8 tile2[128];" in out
    assert "tile[256]" not in re.sub(
        r"//[^\n]*", "", user_part(out)
    )  # E4 variable removed
    # text identical to the translation without E4
    off_out = translate(MIXED, lib_src=LIB, warn_stream=None)
    assert func_text(off_out, "kother") == o
    e = rep["e4"]
    assert e["entries"] == e["e4_entries"] == ["kshared"]
    assert e["e4_excluded"] == [
        {"entry": "kother", "reason": "not selected by WGEMU_E4_ENTRIES"}
    ]
    assert e["non_e4_entries"] == ["kother"] and e["entry_regex"] == r"^kshared$"
    assert "tile2" in e["kept_shared"] and "tile" in e["removed_shared"]
    assert "tile2" not in e["removed_shared"]


def test_select_regex_is_search_on_mangled_name():
    _, rep = sel(MIXED, "shar")
    assert rep["e4"]["e4_entries"] == ["kshared"]
    _, rep = sel(MIXED, "k(shared|other)")
    assert rep["e4"]["e4_entries"] == ["kshared", "kother"]
    assert rep["e4"]["e4_excluded"] == []


def test_select_cvta_shared_in_excluded_entry_translates_selected_fails_closed():
    src = SHARED_KERNEL + OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n", CVTA_BODY + "\tbar.sync \t0;\n"
    )
    out, _ = sel(src, r"^kshared$")
    assert "cvta.shared.u64" in func_text(out, "kother")
    # the same entry selected: the existing fail-closed message
    with pytest.raises(
        TranslateError, match=r"unsupported shared form \.shared in cvta\.shared\.u64"
    ):
        sel(src, r"kother")
    with pytest.raises(
        TranslateError, match=r"unsupported shared form \.shared in cvta\.shared\.u64"
    ):
        e4(src)  # no selection: as before


def test_select_function_reachable_from_both_modes_fails_closed():
    with pytest.raises(TranslateError) as ei:
        sel(helper_module(True, True), r"^kshared$")
    msg = str(ei.value)
    assert "helper" in msg and "kshared" in msg and "kother" in msg
    assert "WGEMU_E4_ENTRIES" in msg


def test_select_function_reachable_from_one_mode_follows_that_mode():
    # only the excluded entry calls the helper: it stays in real shared memory
    out, rep = sel(helper_module(False, True), r"^kshared$")
    h = re.search(r"\.func helper\(.*?\n\}\n", out, re.S).group(0)
    assert "st.shared.u32" in h and "__wgemu_e4" not in h
    assert "hbuf" in rep["e4"]["kept_shared"]
    assert rep["e4"]["functions"] == []
    # only the selected entry calls it: E4
    out, rep = sel(helper_module(True, False), r"^kshared$")
    h = re.search(r"\.func helper\(.*?\n\}\n", out, re.S).group(0)
    assert "st.shared" not in h and "st.global.u32" in h
    assert rep["e4"]["functions"] == ["helper"]


def test_select_excluded_entry_cannot_use_the_e4_library():
    src = SHARED_KERNEL + OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n",
        "\tcp.async.ca.shared.global [%r4], [%rd1], 16;\n\tbar.sync \t0;\n",
    )
    # excluded: cp.async is the sm_80 instruction, not the E4 library call
    out, _ = sel(src, r"^kshared$")
    assert "cp.async.ca.shared.global" in func_text(out, "kother")
    assert "wgemu_e4_cp_async" not in func_text(out, "kother")
    # a library function that uses the E4 header called from a real shared function fails closed
    call = "\t{\n\t.param .b32 param0;\n\tst.param.b32 [param0+0], %r1;\n\tcall.uni wgemu_e4_cp_async_arrive, (param0, param0);\n\t}\n"
    bad = SHARED_KERNEL + OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n", call + "\tbar.sync \t0;\n"
    )
    with pytest.raises(TranslateError, match="kother.*E4 window|E4 window.*kother"):
        sel(bad, r"^kshared$")


def test_select_unset_or_empty_is_identical_to_no_selection():
    for src in (SHARED_KERNEL, PLAIN_KERNEL, MIXED):
        base, _ = e4(src)
        assert sel(src, None)[0] == base
        assert sel(src, "")[0] == base
    # every shared entry is E4, nothing is excluded
    _, rep = sel(MIXED, None)
    assert rep["e4"]["e4_excluded"] == [] and rep["e4"]["entry_regex"] is None
    assert rep["e4"]["entries"] == ["kshared", "kother"]


def test_select_invalid_regex_is_an_error():
    with pytest.raises(TranslateError, match="invalid E4 entry regex"):
        sel(MIXED, "(unclosed")


def test_select_needs_e4():
    with pytest.raises(TranslateError, match="needs E4"):
        translate(MIXED, lib_src=LIB, smem_global_entries="k", warn_stream=None)
    # empty selection with E4 off is the ordinary call
    assert translate(
        MIXED, lib_src=LIB, smem_global_entries="", warn_stream=None
    ) == translate(MIXED, lib_src=LIB, warn_stream=None)


def test_select_nothing_equals_e4_off_translation():
    off = translate(MIXED, lib_src=LIB, warn_stream=None)
    out, rep = sel(MIXED, r"^nomatch$")
    assert out == off
    for sym in (
        "__wgemu_e4_pool_v1",
        "__wgemu_e4_lease_v1",
        "__wgemu_e4_stats_v1",
        "__wgemu_e4_meta",
        "__wgemu_e4f_",
    ):
        assert sym not in user_part(out)
    e = rep["e4"]
    assert e["e4_entries"] == [] and len(e["e4_excluded"]) == 2
    # a regex that matches only a kernel without shared memory selects no shared-using entry either
    both = MIXED + PLAIN_KERNEL.replace(PRE, "")
    assert sel(both, r"^kplain$")[0] == translate(both, lib_src=LIB, warn_stream=None)


def test_select_with_cluster_style_module_without_selected_shared_entry():
    # the excluded entry with cvta.shared must not stop a module whose selection matches nothing
    src = SHARED_KERNEL + OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n", CVTA_BODY + "\tbar.sync \t0;\n"
    )
    out, _ = sel(src, r"^nomatch$")
    assert out == translate(src, lib_src=LIB, warn_stream=None)


def test_nvcc_emu_e4_entries_env(nvcc_emu, monkeypatch):
    for k in ("WGEMU_SMEM_GLOBAL", "WGEMU_E4_ENTRIES", "WGEMU_ASYNC_DELAY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WGEMU_E4_ENTRIES", "^k")
    with pytest.raises(nvcc_emu.EmuError, match="WGEMU_E4_ENTRIES.*WGEMU_SMEM_GLOBAL"):
        nvcc_emu.Driver("nvcc", None, None)
    monkeypatch.setenv("WGEMU_SMEM_GLOBAL", "1")
    monkeypatch.setenv("WGEMU_E4_ENTRIES", "(bad")
    with pytest.raises(nvcc_emu.EmuError, match="not a regex"):
        nvcc_emu.Driver("nvcc", None, None)
    monkeypatch.setenv("WGEMU_E4_ENTRIES", "^k")
    assert nvcc_emu.Driver("nvcc", None, None).e4_entries == "^k"
    monkeypatch.setenv("WGEMU_E4_ENTRIES", "")
    monkeypatch.delenv("WGEMU_SMEM_GLOBAL")
    assert nvcc_emu.Driver("nvcc", None, None).e4_entries == ""  # empty: unset


# ---- nothing selected: the library of E4 off ---------------------------------------------------

LIBFN_CALL = (
    "\t{\n\t.param .b32 param0;\n\tst.param.b32 [param0+0], %r1;\n"
    "\tcall.uni libfn, (param0);\n\t}\n"
)


def _libfn(body: str) -> str:
    return (
        "\n.visible .func libfn(\n\t.param .b32 p0\n)\n{\n\t.reg .b32 %r<2>;\n"
        + body
        + "\tret;\n}\n"
    )


LIB_E4 = LIB + _libfn(
    f"\tld.shared.u32 %r1, [{HDR}+24];\n"
)  # E4 variant: through the header
LIB_OFF = LIB + _libfn("\tmov.u32 %r1, 4242;\n")  # E4-off variant: no E4 window


def libcall_module() -> str:
    return SHARED_KERNEL + OTHER_KERNEL.replace(
        "\tbar.sync \t0;\n", LIBFN_CALL + "\tbar.sync \t0;\n"
    )


def sel_plain(src: str, regex, plain=LIB_OFF, lib=LIB_E4):
    rep: dict = {}
    out = translate(
        src,
        lib_src=lib,
        plain_lib_src=plain,
        smem_global=True,
        smem_global_entries=regex,
        report=rep,
        warn_stream=None,
    )
    return out, rep


def test_select_nothing_uses_the_e4_off_library():
    src = libcall_module()
    off = translate(src, lib_src=LIB_OFF, warn_stream=None)
    out, rep = sel_plain(src, r"^nomatch$")
    assert out == off  # same bytes as E4 off, with the E4-off library
    assert "4242" in out and HDR not in out
    assert rep["e4"]["e4_entries"] == [] and rep["e4"]["entry_regex"] == "^nomatch$"
    assert [e["entry"] for e in rep["e4"]["e4_excluded"]] == ["kshared", "kother"]
    assert all("not selected" in e["reason"] for e in rep["e4"]["e4_excluded"])
    # the E4 library alone cannot serve real shared memory: still fails closed without the plain one
    with pytest.raises(TranslateError, match="kother.*E4 window|E4 window.*kother"):
        sel(
            src.replace("call.uni libfn", "call.uni wgemu_e4_cp_async_arrive"),
            r"^nomatch$",
        )
    with pytest.raises(TranslateError, match="E4 window"):
        translate(
            src,
            lib_src=LIB_E4,
            smem_global=True,
            smem_global_entries=r"^nomatch$",
            warn_stream=None,
        )


def test_select_one_keeps_the_e4_library_and_fails_closed_on_lib_calls():
    with pytest.raises(TranslateError, match="kother.*E4 window|E4 window.*kother"):
        sel_plain(libcall_module(), r"^kshared$")
    # the plain library does not change a selection that has entries: same output with or without it
    src = SHARED_KERNEL + OTHER_KERNEL
    assert sel_plain(src, r"^kshared$")[0] == sel(src, r"^kshared$")[0]


def test_select_nothing_without_library_calls_is_unchanged():
    # no library call: the E4 library is as good as the E4-off one; output equals E4 off either way
    assert sel_plain(MIXED, r"^nomatch$", plain=LIB, lib=LIB)[0] == translate(
        MIXED, lib_src=LIB, warn_stream=None
    )
    assert sel(MIXED, r"^nomatch$")[0] == translate(
        MIXED, lib_src=LIB, warn_stream=None
    )


def test_plain_library_needs_a_selection():
    with pytest.raises(TranslateError, match="--lib-plain"):
        translate(
            MIXED, lib_src=LIB, plain_lib_src=LIB, smem_global=True, warn_stream=None
        )


def test_nvcc_emu_plain_library_and_cli_pair(nvcc_emu, monkeypatch, tmp_path):
    from ptxemu import cli

    for k in (
        "WGEMU_SMEM_GLOBAL",
        "WGEMU_E4_ENTRIES",
        "WGEMU_ASYNC_DELAY",
        "WGEMU_LIB",
    ):
        monkeypatch.delenv(k, raising=False)
    plans = []

    def plan(real, smem_global=False):
        plans.append(smem_global)
        return tmp_path / ("lib_e4.ptx" if smem_global else "lib_off.ptx"), None

    monkeypatch.setattr(nvcc_emu, "library_plan", plan)
    monkeypatch.setenv("WGEMU_SMEM_GLOBAL", "1")
    monkeypatch.setenv("WGEMU_E4_ENTRIES", "^nomatch$")
    d = nvcc_emu.Driver("nvcc", None, None)
    assert d.library_plain() == tmp_path / "lib_off.ptx" and plans == [False]
    monkeypatch.setenv("WGEMU_LIB", str(tmp_path / "given.ptx"))
    assert (
        nvcc_emu.Driver("nvcc", None, None).library_plain() is None
    )  # one given library
    # the argv nvcc_emu builds, run through the CLI: E4 selection of nothing equals the E4-off run
    (tmp_path / "in.ptx").write_text(libcall_module())
    (tmp_path / "lib_e4.ptx").write_text(LIB_E4)
    (tmp_path / "lib_off.ptx").write_text(LIB_OFF)
    base = [str(tmp_path / "in.ptx"), "--no-check"]
    assert (
        cli.main(
            [
                *base,
                "-o",
                str(tmp_path / "off.ptx"),
                "--lib",
                str(tmp_path / "lib_off.ptx"),
            ]
        )
        == 0
    )
    rc = cli.main(
        [
            *base,
            "-o",
            str(tmp_path / "e4.ptx"),
            "--lib",
            str(tmp_path / "lib_e4.ptx"),
            "--smem-global",
            "--smem-global-entries",
            "^nomatch$",
            "--lib-plain",
            str(tmp_path / "lib_off.ptx"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "e4.ptx").read_bytes() == (tmp_path / "off.ptx").read_bytes()
