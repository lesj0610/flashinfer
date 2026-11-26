"""Shared fixtures for the translator tests (CPU only, no nvcc needed by default)."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
DATA = Path(__file__).resolve().parent / "data"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_script(name: str, path: Path):
    """Import a script without a .py suffix (tools/nvcc_emu) or outside a package."""
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def read_data():
    def _read(name: str) -> str:
        return (DATA / name).read_text()

    return _read


@pytest.fixture(scope="session")
def nvcc_emu():
    return load_script("nvcc_emu_under_test", TOOLS / "nvcc_emu")


@pytest.fixture(scope="session")
def harness():
    return load_script(
        "build_and_run_under_test",
        ROOT / "tests" / "e2e" / "common" / "build_and_run.py",
    )
