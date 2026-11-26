"""tools/nvcc_emu and tools/make_cuda_home_shadow.sh without compiling CUDA code.

The nvcc command shapes below are the ones nvcc 13.0 prints with -dryrun. The optional
test at the end runs the real ``nvcc -dryrun`` (no compilation) when WGEMU_TEST_NVCC=1
(WGEMU_TEST_REAL_NVCC overrides /usr/local/cuda/bin/nvcc).
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
SHADOW_SH = TOOLS / "make_cuda_home_shadow.sh"
ENV_KEYS = (
    "WGEMU_REAL_NVCC",
    "WGEMU_LIB",
    "WGEMU_MODE",
    "WGEMU_TRAP",
    "WGEMU_ALLOW_CLUSTER",
    "WGEMU_INVENTORY",
    "WGEMU_SRC_ARCH",
    "WGEMU_CACHE_DIR",
    "CUDA_HOME",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def write_exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------------------------
# Command rewriting
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,want",
    [
        (
            'ptxas -arch=sm_90a -m64 "a.ptx" -o "a.sm_90a.cubin"',
            'ptxas -arch=sm_80 -m64 "a.ptx" -o "a.sm_90a.cubin"',
        ),
        (
            'ptxas -arch=compute_90a -m64  "a2.ptx" ',
            'ptxas -arch=compute_80 -m64  "a2.ptx" ',
        ),
        ("ptxas -arch sm_90 x.ptx", "ptxas -arch sm_80 x.ptx"),
        (
            'nvlink --arch=sm_90a -m64 -cpu-arch=X86_64 "a.o"',
            'nvlink --arch=sm_80 -m64 -cpu-arch=X86_64 "a.o"',
        ),
        ('nvlink "--arch=sm_90a" x.o', 'nvlink "--arch=sm_80" x.o'),
        ("ptxas -arch=sm_75 x.ptx", "ptxas -arch=sm_75 x.ptx"),
    ],
)
def test_rewrite_arch(nvcc_emu, cmd, want):
    assert nvcc_emu._rewrite_arch(cmd) == want


def test_rewrite_fatbinary(nvcc_emu):
    cmd = (
        'fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " '
        '"--image3=kind=elf,sm=90a,file=/k/a.sm_90a.cubin" "--image3=kind=ptx,sm=90a,file=/k/a.ptx" '
        '--embedded-fatbin="/k/a.fatbin.c"'
    )
    assert nvcc_emu.rewrite_fatbinary(cmd) == (
        'fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " '
        '"--image3=kind=elf,sm=80,file=/k/a.sm_90a.cubin" --embedded-fatbin="/k/a.fatbin.c"'
    )
    old = 'fatbinary --create="a.fatbin" -64 --image=profile=sm_90a,file=a.cubin --image=profile=compute_90a,file=a.ptx'
    assert (
        nvcc_emu.rewrite_fatbinary(old)
        == 'fatbinary --create="a.fatbin" -64 --image=profile=sm_80,file=a.cubin'
    )
    keep = '"--image3=kind=elf,sm=75,file=b.cubin"'
    assert nvcc_emu.rewrite_fatbinary("fatbinary " + keep) == "fatbinary " + keep
    with pytest.raises(nvcc_emu.EmuError, match="kind=lto"):
        nvcc_emu.rewrite_fatbinary('fatbinary "--image3=kind=lto,sm=90a,file=a.ltoir"')


def test_cicc_output(nvcc_emu):
    cmd = (
        '"$CICC_PATH/cicc" --c++17 --orig_src_file_name "a.cu" --orig_src_path_name "/s/a.cu" '
        '-arch compute_90a -show-src -m64 "/k/a.cpp1.ii" -o "/k/a.ptx"'
    )
    assert nvcc_emu.cicc_output(cmd) == (90, "/k/a.ptx")
    assert nvcc_emu.cicc_output(cmd.replace("compute_90a", "compute_80")) is None
    assert nvcc_emu.cicc_output(cmd.replace('-o "/k/a.ptx"', "-o a2.ptx")) == (
        90,
        "a2.ptx",
    )
    with pytest.raises(nvcc_emu.EmuError, match="is not PTX"):
        nvcc_emu.cicc_output(cmd.replace("a.ptx", "a.ltoir"))
    with pytest.raises(nvcc_emu.EmuError, match="cannot find the cicc output"):
        nvcc_emu.cicc_output('"$CICC_PATH/cicc" -arch compute_90a x.ii')


def test_ptxas_inputs_and_replace_path(nvcc_emu):
    assert nvcc_emu.ptxas_inputs('ptxas -arch=sm_80 -m64 "k/a.ptx" -o "k/a.cubin"') == [
        "k/a.ptx"
    ]
    assert nvcc_emu.ptxas_inputs(
        'ptxas -arch=sm_80 -m64 --compile-only "u.ptx" -o "u.cubin"'
    ) == ["u.ptx"]
    q = nvcc_emu.replace_path(
        'ptxas "a.ptx" -o "a.cubin"', "a.ptx", "/k/a.wgemu_input.ptx"
    )
    assert q == 'ptxas "/k/a.wgemu_input.ptx" -o "a.cubin"'
    u = nvcc_emu.replace_path("ptxas a.ptx -o a.cubin", "a.ptx", "/k/x y.ptx")
    assert u == "ptxas '/k/x y.ptx' -o a.cubin"
    with pytest.raises(nvcc_emu.EmuError, match="cannot locate"):
        nvcc_emu.replace_path("ptxas b.ptx", "a.ptx", "/k/a.ptx")


@pytest.mark.parametrize(
    "raw,want",
    [
        (
            '"-isystem" "/usr/local/cuda/include/cccl"  ',
            '"-isystem" "/usr/local/cuda/include/cccl"',
        ),
        ('"/usr/local/cuda/bin"', "/usr/local/cuda/bin"),
        (" ", ""),
        ('""', ""),
        ("nvvm/bin", "nvvm/bin"),
    ],
)
def test_parse_env_value(nvcc_emu, raw, want):
    assert nvcc_emu.parse_env_value(raw) == want


@pytest.mark.parametrize(
    "cmd,tool",
    [
        ('"$CICC_PATH/cicc" -arch compute_90a', "cicc"),
        ("rm /k/a_dlink.reg.c", "rm"),
        ("ptxas -arch=sm_90a", "ptxas"),
        ('fatbinary "--image3=kind=elf', "fatbinary"),
        ("", ""),
    ],
)
def test_tool_of(nvcc_emu, cmd, tool):
    assert nvcc_emu.tool_of(cmd) == tool


# ---------------------------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------------------------


def test_parse_args(nvcc_emu):
    p = nvcc_emu.parse_args(
        [
            "-c", "a.cu", "-o", "a.o", "-Iinc1", "-I", "inc2,inc3", "-arch=sm_90a", "-gencode",
            "arch=compute_90a,code=sm_90a", "--wgemu-shadow", "cutlass/arch/arch.h:foo\\:bar:baz",
            "-Xcompiler", "-fPIC", "-keep-dir", "kd", "-std=c++17", "-v", "-dryrun", "-keep",
        ]
    )  # fmt: skip
    assert p.forward == [
        "-c",
        "a.cu",
        "-o",
        "a.o",
        "-Iinc1",
        "-I",
        "inc2,inc3",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
    ]
    assert p.inputs == ["a.cu"] and p.output == "a.o"
    assert p.include_dirs == ["inc1", "inc2", "inc3"]
    assert p.shadow == [("cutlass/arch/arch.h", "foo:bar", "baz")]
    assert (p.keep_dir, p.keep, p.dryrun, p.verbose) == ("kd", True, True, True)


@pytest.mark.parametrize(
    "args,msg",
    [
        (["a.cu", "-dlto"], "not supported"),
        (["a.cu", "--split-compile=4"], "not supported"),
        (["a.cu", "-o"], "needs a value"),
        (["a.cu", "--wgemu-shadow", "only:two"], "relpath"),
        (["a.cu", "--wgemu-shadow", ":rx:repl"], "relpath"),
    ],
)
def test_parse_args_rejects(nvcc_emu, args, msg):
    with pytest.raises(nvcc_emu.EmuError, match=msg):
        nvcc_emu.parse_args(args)


def test_expand_options_files(nvcc_emu, tmp_path):
    f1 = tmp_path / "f1.opt"
    f1.write_text("-O3 -I inc4\n-DFOO=1\n")
    f2 = tmp_path / "f2.opt"
    f2.write_text("-lineinfo")
    f3 = tmp_path / "f3.opt"
    f3.write_text("-std=c++17 '-DMSG=a b'")
    got = nvcc_emu.expand_options_files(
        ["a.cu", "-optf", f"{f1},{f2}", f"--options-file={f3}"]
    )
    assert got == [
        "a.cu",
        "-O3",
        "-I",
        "inc4",
        "-DFOO=1",
        "-lineinfo",
        "-std=c++17",
        "-DMSG=a b",
    ]
    with pytest.raises(nvcc_emu.EmuError, match="needs a value"):
        nvcc_emu.expand_options_files(["a.cu", "-optf"])


def test_default_keep_dir(nvcc_emu, tmp_path, monkeypatch):
    P = nvcc_emu.Parsed
    p = P()
    p.keep_dir = str(tmp_path / "kd")
    assert nvcc_emu.default_keep_dir(p, tmp_path) == (
        (tmp_path / "kd").resolve(),
        False,
    )
    p = P()
    p.keep = True
    assert nvcc_emu.default_keep_dir(p, tmp_path) == (tmp_path, False)
    p = P()
    p.output = str(tmp_path / "a.o")
    assert nvcc_emu.default_keep_dir(p, tmp_path) == (
        Path(str(tmp_path / "a.o") + ".wgemu.d"),
        False,
    )
    p = P()
    p.inputs = ["src/a.cu"]
    assert nvcc_emu.default_keep_dir(p, tmp_path) == (tmp_path / "a.cu.wgemu.d", False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    p = P()
    p.output = "/dev/null"
    d, temp = nvcc_emu.default_keep_dir(p, tmp_path)
    assert (
        temp and d.parent == tmp_path and d.name.startswith("nvcc_emu.") and d.is_dir()
    )


def test_normalize_arch(nvcc_emu, monkeypatch):
    assert nvcc_emu.normalize_arch() == "-gencode=arch=compute_90a,code=sm_90a"
    monkeypatch.setenv("WGEMU_SRC_ARCH", "90")
    assert nvcc_emu.normalize_arch() == "-gencode=arch=compute_90,code=sm_90"
    monkeypatch.setenv("WGEMU_SRC_ARCH", "100a")
    with pytest.raises(nvcc_emu.EmuError, match="WGEMU_SRC_ARCH"):
        nvcc_emu.normalize_arch()


def test_shadow_headers(nvcc_emu, tmp_path):
    inc = tmp_path / "inc"
    hdr = inc / "cutlass" / "arch" / "arch.h"
    hdr.parent.mkdir(parents=True)
    (hdr.parent / "config.h").write_text("// config\n")
    hdr.write_text(
        "#pragma once\n"
        '#include "config.h"\n'
        '#include "not_there.h"\n'
        "#include <cuda.h>\n"
        "#define CUTLASS_ARCH_MMA_SM90A_ENABLED 1\n"
        "#define OTHER 1\n"
    )
    p = nvcc_emu.Parsed()
    p.include_dirs = [str(tmp_path / "none"), str(inc)]
    p.shadow = [
        (
            "cutlass/arch/arch.h",
            r"^#define CUTLASS_ARCH_MMA_SM90A_ENABLED 1$",
            "#define CUTLASS_ARCH_MMA_SM90A_ENABLED 0",
        ),
        ("cutlass/arch/arch.h", r"^#define OTHER 1$", "#define OTHER 2"),
    ]
    kd = tmp_path / "kd"
    root = kd / "wgemu_shadow"
    assert nvcc_emu.make_shadow_headers(p, kd) == [f"-I{root}"]
    text = (root / "cutlass" / "arch" / "arch.h").read_text()
    assert f'#include "{(hdr.parent / "config.h").resolve()}"\n' in text
    assert '#include "not_there.h"\n' in text and "#include <cuda.h>\n" in text
    assert (
        "#define CUTLASS_ARCH_MMA_SM90A_ENABLED 0\n" in text
        and "#define OTHER 2\n" in text
    )
    assert hdr.read_text().count(" 1\n") == 2  # the original is untouched

    p.shadow = [("cutlass/arch/arch.h", r"^#define NOPE$", "x")]
    with pytest.raises(nvcc_emu.EmuError, match="matched nothing"):
        nvcc_emu.make_shadow_headers(p, kd)
    p.shadow = [("cutlass/arch/arch.h", r"(", "x")]
    with pytest.raises(nvcc_emu.EmuError, match="bad regex"):
        nvcc_emu.make_shadow_headers(p, kd)
    p.shadow = [("cutlass/arch/missing.h", r"x", "y")]
    with pytest.raises(nvcc_emu.EmuError, match="not found in the include path"):
        nvcc_emu.make_shadow_headers(p, kd)
    p.shadow = []
    assert nvcc_emu.make_shadow_headers(p, kd) == []


# ---------------------------------------------------------------------------------------------
# Real nvcc, emulation library
# ---------------------------------------------------------------------------------------------


def test_find_real_nvcc_from_env(nvcc_emu, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "elsewhere" / "bin" / "nvcc")])
    fake = write_exe(tmp_path / "cuda" / "bin" / "nvcc", "exit 0\n")
    monkeypatch.setenv("WGEMU_REAL_NVCC", str(fake))
    assert nvcc_emu.find_real_nvcc() == str(fake)
    monkeypatch.setenv("WGEMU_REAL_NVCC", str(tmp_path / "nope"))
    with pytest.raises(nvcc_emu.EmuError, match="not executable"):
        nvcc_emu.find_real_nvcc()
    monkeypatch.setenv("WGEMU_REAL_NVCC", str(TOOLS / "nvcc_emu"))
    with pytest.raises(nvcc_emu.EmuError, match="nvcc_emu itself"):
        nvcc_emu.find_real_nvcc()
    monkeypatch.delenv("WGEMU_REAL_NVCC")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    assert nvcc_emu.find_real_nvcc() == str(fake)


def make_real_home(base: Path) -> Path:
    write_exe(base / "bin" / "nvcc", "exit 0\n")
    write_exe(base / "bin" / "ptxas", "exit 0\n")
    (base / "include").mkdir(parents=True)
    (base / "include" / "cuda.h").write_text("// cuda\n")
    (base / "lib64").mkdir()
    return base


def run_shadow(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SHADOW_SH), *args], capture_output=True, text=True, check=False
    )


def test_cuda_home_shadow(nvcc_emu, tmp_path, monkeypatch):
    real = make_real_home(tmp_path / "cuda")
    shadow = tmp_path / "shadow"
    r = run_shadow(str(shadow), str(real))
    assert r.returncode == 0, r.stderr
    assert "export CUDA_HOME=" in r.stdout
    real_r = real.resolve()
    assert os.path.realpath(shadow / "bin" / "nvcc") == os.path.realpath(
        TOOLS / "nvcc_emu"
    )
    assert os.path.realpath(shadow / "bin" / "ptxas") == str(real_r / "bin" / "ptxas")
    assert (shadow / "include").is_symlink() and (
        shadow / "include" / "cuda.h"
    ).is_file()
    assert (shadow / "lib64").is_symlink()
    assert (shadow / ".wgemu_real_cuda_home").read_text() == f"{real_r}\n"
    assert run_shadow(str(shadow), str(real)).returncode == 0  # refresh is allowed

    # nvcc_emu finds the real nvcc through the shadow, by CUDA_HOME or by its own location
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "elsewhere" / "bin" / "nvcc")])
    monkeypatch.setenv("CUDA_HOME", str(shadow))
    assert nvcc_emu.find_real_nvcc() == str(real_r / "bin" / "nvcc")
    monkeypatch.delenv("CUDA_HOME")
    monkeypatch.setattr(sys, "argv", [str(shadow / "bin" / "nvcc")])
    assert nvcc_emu.find_real_nvcc() == str(real_r / "bin" / "nvcc")

    other = make_real_home(tmp_path / "cuda2")
    r = run_shadow(str(shadow), str(other))
    assert r.returncode == 1 and "use a new directory" in r.stderr
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x")
    r = run_shadow(str(busy), str(real))
    assert r.returncode == 1 and "not a wgemu shadow" in r.stderr
    r = run_shadow(str(tmp_path / "shadow2"), str(shadow))
    assert r.returncode == 1 and "itself a wgemu shadow" in r.stderr
    assert run_shadow().returncode == 2


def test_library_plan(nvcc_emu, tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setattr(nvcc_emu, "ROOT", root)
    fake = write_exe(tmp_path / "bin" / "nvcc", 'echo "fake nvcc release 13.0"\n')
    monkeypatch.setenv("WGEMU_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(nvcc_emu.EmuError, match="emu_lib.cu not found"):
        nvcc_emu.library_plan(str(fake))
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "emu_lib.cu").write_text("// lib\n")
    (root / "include" / "wgmma_emu").mkdir(parents=True)
    hdr = root / "include" / "wgmma_emu" / "numerics.h"
    hdr.write_text("// v1\n")
    path, cmd = nvcc_emu.library_plan(str(fake))
    assert path.parent == tmp_path / "cache" and re.fullmatch(
        r"emu_lib\.[0-9a-f]{20}\.ptx", path.name
    )
    assert cmd == [
        str(fake), "-arch=compute_80", "-rdc=true", "--ptx", "-O3", "-std=c++17",
        "-I", str(root / "include"), str(root / "runtime" / "emu_lib.cu"), "-o",
    ]  # fmt: skip
    assert nvcc_emu.library_plan(str(fake)) == (path, cmd)
    hdr.write_text("// v2\n")
    path2, _ = nvcc_emu.library_plan(str(fake))
    assert path2 != path
    path2.parent.mkdir(parents=True, exist_ok=True)
    path2.write_text("// cached\n")
    assert nvcc_emu.library_plan(str(fake)) == (path2, None)

    given = tmp_path / "given.ptx"
    given.write_text("// given\n")
    monkeypatch.setenv("WGEMU_LIB", str(given))
    assert nvcc_emu.library_plan(str(fake)) == (given, None)
    monkeypatch.setenv("WGEMU_LIB", str(tmp_path / "none.ptx"))
    with pytest.raises(nvcc_emu.EmuError, match="does not exist"):
        nvcc_emu.library_plan(str(fake))


def test_ensure_library(nvcc_emu, tmp_path):
    ok = write_exe(
        tmp_path / "ok_nvcc",
        'for a in "$@"; do last=$a; done\nprintf "// built\\n" > "$last"\n',
    )
    lib = tmp_path / "cache" / "emu_lib.0123456789abcdef0123.ptx"
    nvcc_emu.ensure_library(lib, [str(ok), "-o"])
    assert lib.read_text() == "// built\n"
    nvcc_emu.ensure_library(lib, None)  # nothing to build
    bad = write_exe(tmp_path / "bad_nvcc", "echo boom >&2\nexit 3\n")
    lib2 = tmp_path / "cache" / "emu_lib.ffffffffffffffffffff.ptx"
    with pytest.raises(nvcc_emu.EmuError, match=r"(?s)failed \(rc=3\).*boom"):
        nvcc_emu.ensure_library(lib2, [str(bad), "-o"])
    assert not lib2.exists()
    assert not list((tmp_path / "cache").glob("*.tmp.ptx"))


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def test_driver_dry_run_rewrites(nvcc_emu, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lib = tmp_path / "emu_lib.ptx"
    lib.write_text("// lib\n")
    monkeypatch.setenv("WGEMU_LIB", str(lib))

    def no_run(*a, **k):
        raise AssertionError("a dry run must not execute commands")

    monkeypatch.setattr(nvcc_emu.subprocess, "run", no_run)
    kd = tmp_path / "kd"
    kd.mkdir()
    p = nvcc_emu.Parsed()
    p.dryrun = True
    drv = nvcc_emu.Driver("/fake/nvcc", p, kd)
    cmds = [
        f'rm "{kd}/a_dlink.reg.c"',
        f'"$CICC_PATH/cicc" --c++17 --orig_src_file_name "a.cu" -arch compute_90a -m64 "{kd}/a.cpp1.ii" -o "{kd}/a.ptx"',
        f'ptxas -arch=sm_90a -m64  "{kd}/a.ptx"  -o "{kd}/a.sm_90a.cubin"',
        f'fatbinary -64 "--image3=kind=elf,sm=90a,file={kd}/a.sm_90a.cubin" '
        f'"--image3=kind=ptx,sm=90a,file={kd}/a.ptx" --embedded-fatbin="{kd}/a.fatbin.c"',
        'ptxas -arch=sm_90a -m64 --compile-only "user.ptx" -o "user.cubin"',
        f'nvlink -m64 --arch=sm_90a "{kd}/a.o" -lcudadevrt -o "{kd}/a_dlink.sm_90a.cubin" -cpu-arch=X86_64',
    ]
    for c in cmds:
        drv.run_cmd(c, {})
    s = drv.script
    assert s[:3] == ["#!/usr/bin/env bash", "set -e", f"cd {shlex.quote(os.getcwd())}"]
    assert s[3:5] == cmds[:2]
    tr_a = s[5]
    assert tr_a.startswith("cp ") and " -m ptxemu " in tr_a
    assert (
        f"{kd}/a.wgemu_orig.ptx -o {kd}/a.ptx --lib {lib} --mode exact --report {kd}/a.wgemu_report.json"
        in tr_a
    )
    assert (
        s[6] == f'ptxas -arch=sm_80 -m64  "{kd}/a.ptx"  -o "{kd}/a.sm_90a.cubin"'
    )  # translated once
    assert (
        s[7]
        == f'fatbinary -64 "--image3=kind=elf,sm=80,file={kd}/a.sm_90a.cubin" --embedded-fatbin="{kd}/a.fatbin.c"'
    )
    assert f"{kd}/user.wgemu_orig.ptx -o {kd}/user.wgemu_input.ptx" in s[8]
    assert (
        s[9]
        == f'ptxas -arch=sm_80 -m64 --compile-only "{kd}/user.wgemu_input.ptx" -o "user.cubin"'
    )
    assert (
        s[10]
        == f'nvlink -m64 --arch=sm_80 "{kd}/a.o" -lcudadevrt -o "{kd}/a_dlink.sm_90a.cubin" -cpu-arch=X86_64'
    )
    assert len(s) == 11
    assert "#$ ptxas -arch=sm_80" in capsys.readouterr().err

    monkeypatch.setenv("WGEMU_MODE", "check")
    monkeypatch.setenv("WGEMU_TRAP", "1")
    monkeypatch.setenv("WGEMU_ALLOW_CLUSTER", "1")
    drv = nvcc_emu.Driver("/fake/nvcc", p, kd)
    drv.run_cmd(cmds[1], {})
    assert "--mode check --report" in drv.script[-1] and drv.script[-1].endswith(
        "--trap --allow-cluster"
    )
    monkeypatch.setenv("WGEMU_MODE", "bogus")
    with pytest.raises(nvcc_emu.EmuError, match="WGEMU_MODE"):
        nvcc_emu.Driver("/fake/nvcc", p, kd)


def test_driver_runs_commands(nvcc_emu, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drv = nvcc_emu.Driver("/fake/nvcc", nvcc_emu.Parsed(), tmp_path)
    env = dict(os.environ)
    drv.run_cmd(f"rm {tmp_path}/does_not_exist.c", env)  # nvcc's stale-file rm may fail
    drv.run_cmd(f"echo hi > {tmp_path}/made.txt", env)
    assert (tmp_path / "made.txt").read_text() == "hi\n"
    with pytest.raises(nvcc_emu.EmuError) as ei:
        drv.run_cmd("exit 7", env)
    assert ei.value.args[1] == 7


def test_cleanup(nvcc_emu, tmp_path):
    kd = tmp_path / "kd"
    kd.mkdir()
    for name in (
        "old.txt",
        "a.cpp1.ii",
        "a.ptx",
        "a.wgemu_report.json",
        "a.sm_90a.cubin",
    ):
        (kd / name).write_text("x")
    (kd / "sub").mkdir()
    nvcc_emu.cleanup(kd, False, {"old.txt"})
    assert sorted(f.name for f in kd.iterdir()) == [
        "a.ptx",
        "a.wgemu_report.json",
        "old.txt",
        "sub",
    ]
    nvcc_emu.cleanup(kd, True, set())
    assert not kd.exists()


# ---------------------------------------------------------------------------------------------
# Optional: real nvcc -dryrun (prints the steps, compiles nothing)
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("WGEMU_TEST_NVCC") != "1",
    reason="set WGEMU_TEST_NVCC=1 to run nvcc -dryrun",
)
def test_real_nvcc_dryrun(nvcc_emu, tmp_path, monkeypatch, data_dir):
    real = os.environ.get("WGEMU_TEST_REAL_NVCC", "/usr/local/cuda/bin/nvcc")
    monkeypatch.setenv("WGEMU_REAL_NVCC", real)
    lib = tmp_path / "emu_lib.ptx"
    lib.write_text("// lib\n")
    monkeypatch.setenv("WGEMU_LIB", str(lib))
    monkeypatch.chdir(tmp_path)
    shutil.copy(data_dir / "sample_sm90.cu", tmp_path / "s.cu")
    assert nvcc_emu.main(["s.cu", "-c", "-o", "s.o", "-arch=sm_90a", "-dryrun"]) == 0
    script = (tmp_path / "s.o.wgemu.d" / "wgemu_cmds.sh").read_text()
    lines = script.splitlines()
    cicc = [ln for ln in lines if nvcc_emu.tool_of(ln) == "cicc"]
    ptxas = [ln for ln in lines if nvcc_emu.tool_of(ln) == "ptxas"]
    fatbin = [ln for ln in lines if nvcc_emu.tool_of(ln) == "fatbinary"]
    assert cicc and all(re.search(r"-arch\s+compute_90a", ln) for ln in cicc)
    assert ptxas and all(re.search(r"-arch=sm_80\b", ln) for ln in ptxas)
    assert fatbin and all("sm=80" in ln and "kind=ptx" not in ln for ln in fatbin)
    assert sum(" -m ptxemu " in ln for ln in lines) == 1
    assert not (tmp_path / "s.o").exists()
