import subprocess
import sys

import pytest
from typer.testing import CliRunner

from frozen_lake import __version__
from frozen_lake.cli import app

runner = CliRunner()


def test_help_lists_planned_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("tui", "episode", "benchmark", "validate", "plot"):
        assert command in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("command", ["episode", "benchmark", "validate", "plot"])
def test_placeholder_commands_fail_clearly(command):
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_tui_builds_the_explorer_from_its_options(monkeypatch):
    from frozen_lake.tui.app import FrozenLakeApp

    launched = []
    monkeypatch.setattr(FrozenLakeApp, "run", lambda self: launched.append(self))
    result = runner.invoke(app, ["tui", "--size", "5", "--map-seed", "7", "--episode-seed", "3"])
    assert result.exit_code == 0, result.output
    (explorer,) = launched
    assert len(explorer.desc) == 5
    assert explorer.session.max_steps == 4 * 5 * 5
    assert explorer.session.seed == 3


@pytest.mark.parametrize(
    "args", [["--size", "1"], ["--frozen-probability", "0"], ["--success-rate", "2"]]
)
def test_tui_rejects_invalid_options(args):
    result = runner.invoke(app, ["tui", *args])
    assert result.exit_code == 2


def test_non_tui_commands_never_import_textual():
    code = (
        "import sys; from typer.testing import CliRunner; from frozen_lake.cli import app; "
        "import frozen_lake.episodes; CliRunner().invoke(app, ['--help']); "
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
