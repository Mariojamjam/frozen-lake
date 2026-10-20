"""Fixtures shared by the suite: one small experiment run, made once per session."""

import shutil
from pathlib import Path

import pytest

from frozen_lake.experiments import load_config, run_experiment

TINY = Path(__file__).parent / "data" / "tiny.toml"


@pytest.fixture(scope="session")
def tiny_config():
    return load_config(TINY)


@pytest.fixture(scope="session")
def tiny_run(tiny_config, tmp_path_factory) -> Path:
    """A finished run of the small configuration. Read it; copy it before changing anything."""
    return run_experiment(tiny_config, output_root=tmp_path_factory.mktemp("runs"))


@pytest.fixture
def run_copy(tiny_run, tmp_path) -> Path:
    """A private copy of the small run, for tests that damage or extend it."""
    copy = tmp_path / "run"
    shutil.copytree(tiny_run, copy)
    return copy
