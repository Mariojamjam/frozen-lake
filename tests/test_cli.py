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


@pytest.mark.parametrize("command", ["tui", "episode", "benchmark", "validate", "plot"])
def test_placeholder_commands_fail_clearly(command):
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output
