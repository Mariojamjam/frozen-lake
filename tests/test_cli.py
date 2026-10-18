import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("command", ["plot"])
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
        [
            "tui",
            "--size",
            "5",
            "--map-seed",
            "7",
            "--episode-seed",
            "3",
            "--depth",
            "4",
            "--rollouts",
            "11",
            "--horizon",
            "13",
        ],
    )
    assert result.exit_code == 0, result.output
    (explorer,) = launched
    assert len(explorer.desc) == 5
    assert explorer.session.max_steps == 4 * 5 * 5
    assert explorer.session.seed == 3
    assert (explorer.depth, explorer.beta, explorer.gamma) == (4, 0.9, 1.0)
    assert (explorer.rollouts, explorer.horizon) == (11, 13)
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


def test_episode_rejects_an_unknown_strategy():
    result = runner.invoke(app, ["episode", "--strategy", "XYZ"])
    assert result.exit_code == 2
    assert "BFS-RO" in result.output  # the message lists the valid strategies


@pytest.mark.parametrize("strategy", ["BFS-RO", "EXP-RO"])
def test_episode_plays_with_rollout(strategy):
    result = runner.invoke(
        app, ["episode", "--strategy", strategy, "--rollouts", "5", "--horizon", "20"]
    )
    assert result.exit_code == 0, result.output
    assert f"strategy {strategy}" in result.output
    assert "rollout: 5 trajectories per action, horizon 20" in result.output
    assert "estimates L" in result.output
    assert "simulated transitions" in result.output


def test_rollout_episodes_are_reproducible_and_seedable():
    args = ["episode", "--strategy", "BFS-RO", "--rollouts", "5", "--quiet"]
    first, second = runner.invoke(app, args), runner.invoke(app, args)
    assert (
        first.output.splitlines()[-1].split(" · ")[:2]
        == second.output.splitlines()[-1].split(" · ")[:2]
    )
    assert runner.invoke(app, [*args, "--rollout-seed", "3"]).exit_code == 0


@pytest.mark.parametrize("args", [["--rollouts", "0"], ["--horizon", "0"]])
def test_episode_rejects_invalid_rollout_settings(args):
    assert runner.invoke(app, ["episode", "--strategy", "BFS-RO", *args]).exit_code == 2


TINY = "tests/data/tiny.toml"


def test_validate_accepts_a_good_configuration_and_describes_it():
    result = runner.invoke(app, ["validate", "--config", "configs/experiment.toml"])
    assert result.exit_code == 0, result.output
    assert "13500 episodes" in result.output
    assert "configuration is valid" in result.output


def test_validate_rejects_a_bad_configuration(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(Path(TINY).read_text().replace("sizes = [3, 4]", "sizes = []"))
    result = runner.invoke(app, ["validate", "--config", str(bad)])
    assert result.exit_code == 2
    assert "environment.sizes must not be empty" in result.output


def test_validate_needs_exactly_one_target():
    assert runner.invoke(app, ["validate"]).exit_code == 2
    both = runner.invoke(app, ["validate", "--config", TINY, "--run", "x"])
    assert both.exit_code == 2


def test_benchmark_dry_run_describes_without_running(tmp_path):
    result = runner.invoke(
        app, ["benchmark", "--config", TINY, "--output-dir", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "64 episodes" in result.output
    assert list(tmp_path.iterdir()) == []


def test_benchmark_runs_and_its_results_validate(tmp_path):
    result = runner.invoke(app, ["benchmark", "--config", TINY, "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    (run_dir,) = tmp_path.iterdir()
    assert "results written to" in result.output
    checked = runner.invoke(app, ["validate", "--run", str(run_dir)])
    assert checked.exit_code == 0, checked.output
    assert "complete and consistent" in checked.output


def test_validate_reports_damaged_results(tmp_path):
    runner.invoke(app, ["benchmark", "--config", TINY, "--output-dir", str(tmp_path)])
    (run_dir,) = tmp_path.iterdir()
    (run_dir / "summary.csv").unlink()
    result = runner.invoke(app, ["validate", "--run", str(run_dir)])
    assert result.exit_code == 1
    assert "missing files: summary.csv" in result.output
