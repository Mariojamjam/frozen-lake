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


@pytest.mark.parametrize("command", ["benchmark", "validate", "plot"])
def test_placeholder_commands_fail_clearly(command):
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_tui_builds_the_explorer_from_its_options(monkeypatch):
    from frozen_lake.tui.app import FrozenLakeApp

    launched = []
    monkeypatch.setattr(FrozenLakeApp, "run", lambda self: launched.append(self))
    result = runner.invoke(
        app,
        ["tui", "--size", "5", "--map-seed", "7", "--episode-seed", "3", "--depth", "4"],
    )
    assert result.exit_code == 0, result.output
    (explorer,) = launched
    assert len(explorer.desc) == 5
    assert explorer.session.max_steps == 4 * 5 * 5
    assert explorer.session.seed == 3
    assert (explorer.depth, explorer.beta, explorer.gamma) == (4, 0.9, 1.0)
    assert explorer.make_map is not None
    assert explorer.make_map(7) == explorer.desc  # the same options rebuild the same lake


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


def test_episode_prints_every_step_and_a_summary():
    result = runner.invoke(app, ["episode", "--size", "6", "--episode-seed", "7"])
    assert result.exit_code == 0, result.output
    assert "strategy BFS" in result.output
    assert "step   1:" in result.output
    assert "result:" in result.output


def test_episode_quiet_prints_only_the_summary():
    result = runner.invoke(app, ["episode", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "step   1:" not in result.output
    assert "result:" in result.output


def test_episode_plays_with_expectimax():
    result = runner.invoke(app, ["episode", "--strategy", "EXP", "--depth", "3", "--beta", "0.85"])
    assert result.exit_code == 0, result.output
    assert "strategy EXP" in result.output
    assert "depth 3, beta 0.85" in result.output
    assert "subproblems" in result.output
    assert "result:" in result.output


@pytest.mark.parametrize("args", [["--beta", "1.5"], ["--beta", "0"], ["--depth", "0"]])
def test_episode_rejects_invalid_expectimax_settings(args):
    assert runner.invoke(app, ["episode", "--strategy", "EXP", *args]).exit_code == 2


def test_episode_rejects_unknown_and_unavailable_strategies():
    assert runner.invoke(app, ["episode", "--strategy", "XYZ"]).exit_code == 2
    unavailable = runner.invoke(app, ["episode", "--strategy", "BFS-RO"])
    assert unavailable.exit_code == 2
    assert "not implemented yet" in unavailable.output
