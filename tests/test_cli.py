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


def test_validate_checks_the_study_configuration_by_default():
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0, result.output
    assert "13500 episodes" in result.output
    assert "configuration is valid" in result.output


def test_validate_rejects_a_bad_configuration(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(Path(TINY).read_text().replace("sizes = [3, 4]", "sizes = []"))
    result = runner.invoke(app, ["validate", "-i", str(bad)])
    assert result.exit_code == 2
    assert "environment.sizes must not be empty" in result.output


def test_validate_reports_a_missing_file(tmp_path):
    result = runner.invoke(app, ["validate", "-i", str(tmp_path / "nope.toml")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_benchmark_dry_run_describes_without_running(tmp_path):
    result = runner.invoke(app, ["benchmark", "-i", TINY, "-o", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "64 episodes" in result.output
    assert list(tmp_path.iterdir()) == []


def test_benchmark_takes_a_folder_with_an_experiment_toml(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "experiment.toml").write_text(Path(TINY).read_text())
    result = runner.invoke(app, ["benchmark", "-i", str(folder), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "64 episodes" in result.output


def test_benchmark_asks_before_running_and_stops_on_no(tmp_path):
    result = runner.invoke(app, ["benchmark", "-i", TINY, "-o", str(tmp_path)], input="n\n")
    assert result.exit_code == 1
    assert "Run 64 episodes now?" in result.output
    assert list(tmp_path.iterdir()) == []


def test_benchmark_runs_after_yes_and_its_results_validate(tmp_path):
    result = runner.invoke(app, ["benchmark", "-i", TINY, "-o", str(tmp_path)], input="y\n")
    assert result.exit_code == 0, result.output
    (run_dir,) = tmp_path.iterdir()
    assert "results written to" in result.output
    assert f"frozen-lake plot -i {run_dir}" in result.output  # tells what to do next
    checked = runner.invoke(app, ["validate", "-i", str(run_dir)])
    assert checked.exit_code == 0, checked.output
    assert "complete and consistent" in checked.output


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    result = runner.invoke(app, ["benchmark", "-i", TINY, "-o", str(root), "-y"])
    assert result.exit_code == 0, result.output
    (run_dir,) = root.iterdir()
    return run_dir


def test_validate_reports_damaged_results(tiny_run, tmp_path):
    import shutil

    damaged = tmp_path / "damaged"
    shutil.copytree(tiny_run, damaged)
    (damaged / "summary.csv").unlink()
    result = runner.invoke(app, ["validate", "-i", str(damaged)])
    assert result.exit_code == 1
    assert "missing files: summary.csv" in result.output


def test_plot_analyzes_the_run_it_is_given(tiny_run, tmp_path):
    result = runner.invoke(
        app, ["plot", "-i", str(tiny_run), "-o", str(tmp_path / "out"), "--resamples", "100"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "analysis.json").is_file()


def test_plot_without_a_run_lists_the_runs_it_found(tiny_run, tmp_path, monkeypatch):
    import shutil

    import frozen_lake.cli as cli

    shutil.copytree(tiny_run, tmp_path / "canonical" / "experiment-1")
    (tmp_path / "generated" / "not-a-run").mkdir(parents=True)
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    result = runner.invoke(app, ["plot"])
    assert result.exit_code == 2  # no terminal to ask in
    assert "1. " + str(tmp_path / "canonical" / "experiment-1") in result.output
    assert "not-a-run" not in result.output


def test_plot_asks_which_run_in_a_terminal(tiny_run, tmp_path, monkeypatch):
    import shutil

    import frozen_lake.cli as cli

    shutil.copytree(tiny_run, tmp_path / "generated" / "a")
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    analyzed = []
    monkeypatch.setattr(
        "frozen_lake.analysis.analyze_run",
        lambda run, *args, **kwargs: analyzed.append(run) or (_ for _ in ()).throw(SystemExit(0)),
    )
    runner.invoke(app, ["plot"], input="1\n")
    assert analyzed == [tmp_path / "generated" / "a"]


def test_plot_without_any_run_says_how_to_make_one(tmp_path, monkeypatch):
    import frozen_lake.cli as cli

    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    result = runner.invoke(app, ["plot"])
    assert result.exit_code == 2
    assert "frozen-lake benchmark" in result.output


# --- help and ergonomics -----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["tui", "episode", "benchmark", "validate", "plot"])
def test_every_command_explains_itself_with_examples(command):
    for flag in ("-h", "--help"):
        result = runner.invoke(app, [command, flag])
        assert result.exit_code == 0, result.output
        assert "Examples" in result.output
        assert f"frozen-lake {command}" in result.output


def test_the_overview_shows_the_workflow():
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "Typical workflow" in result.output
    assert "frozen-lake plot -i <run>" in result.output


def test_tui_without_options_shows_its_help_instead_of_opening():
    result = runner.invoke(app, ["tui"])
    assert "Examples" in result.output
    assert "Keys inside the explorer" in result.output


def test_tui_needs_the_lake_size():
    result = runner.invoke(app, ["tui", "-m", "3"])
    assert result.exit_code == 2
    assert "Missing option '--size'" in result.output


def test_tui_accepts_short_options(monkeypatch):
    from frozen_lake.tui.app import FrozenLakeApp

    launched = []
    monkeypatch.setattr(FrozenLakeApp, "run", lambda self: launched.append(self))
    result = runner.invoke(
        app, ["tui", "-s", "6", "-m", "7", "-e", "3", "-d", "4", "-b", "0.8", "-n", "50"]
    )
    assert result.exit_code == 0, result.output
    (explorer,) = launched
    assert (len(explorer.desc), explorer.map_seed, explorer.episode_seed) == (6, 7, 3)
    assert (explorer.depth, explorer.beta, explorer.rollouts) == (4, 0.8, 50)
    assert explorer.rollout_seed is None


def test_tui_passes_the_rollout_seed(monkeypatch):
    from frozen_lake.tui.app import FrozenLakeApp

    launched = []
    monkeypatch.setattr(FrozenLakeApp, "run", lambda self: launched.append(self))
    assert runner.invoke(app, ["tui", "-s", "4", "--rollout-seed", "9"]).exit_code == 0
    assert launched[0].rollout_seed == 9


def test_episode_short_options_match_the_long_ones():
    short = runner.invoke(app, ["episode", "-S", "EXP", "-s", "6", "-m", "3", "-e", "7", "-q"])
    long = runner.invoke(
        app,
        [
            "episode", "--strategy", "EXP", "--size", "6", "--map-seed", "3",
            "--episode-seed", "7", "--quiet",
        ],
    )  # fmt: skip
    assert short.exit_code == long.exit_code == 0, short.output
    assert short.output.splitlines()[0] == long.output.splitlines()[0]
    assert (
        short.output.splitlines()[-1].split(" · ")[:2]
        == long.output.splitlines()[-1].split(" · ")[:2]
    )


def test_episode_prints_its_effective_settings():
    result = runner.invoke(app, ["episode", "-s", "6", "-q"])
    assert "step limit 144 (4 × 6²)" in result.output
    assert "as intended 80% of the time, 10% to each side" in result.output


def test_episode_without_options_outside_a_terminal_plays_the_defaults():
    result = runner.invoke(app, ["episode"])
    assert result.exit_code == 0, result.output
    assert "strategy BFS" in result.output
    assert "result:" in result.output


def test_episode_without_options_in_a_terminal_opens_the_form(monkeypatch):
    import frozen_lake.cli as cli
    from frozen_lake.models import StrategyId
    from frozen_lake.tui.episode_form import EpisodeForm

    monkeypatch.setattr(cli, "_interactive", lambda: True)

    def choose(form):
        return {**form.initial, "strategy": StrategyId.EXP, "size": 5, "quiet": True}

    monkeypatch.setattr(EpisodeForm, "run", choose)
    result = runner.invoke(app, ["episode"])
    assert result.exit_code == 0, result.output
    assert "$ frozen-lake episode -S EXP -s 5 -q" in result.output
    assert "map 5×5" in result.output and "strategy EXP" in result.output


def test_cancelling_the_form_plays_nothing(monkeypatch):
    import frozen_lake.cli as cli
    from frozen_lake.tui.episode_form import EpisodeForm

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(EpisodeForm, "run", lambda form: None)
    result = runner.invoke(app, ["episode"])
    assert result.exit_code == 0
    assert "cancelled" in result.output
    assert "result:" not in result.output


def test_invalid_settings_explain_what_they_mean():
    result = runner.invoke(app, ["episode", "-S", "EXP", "-b", "1.5"])
    assert result.exit_code == 2
    assert "leaf heuristic" in " ".join(result.output.replace("│", " ").split())


def test_episode_with_options_never_imports_textual():
    code = (
        "import sys; from typer.testing import CliRunner; from frozen_lake.cli import app; "
        "result = CliRunner().invoke(app, ['episode', '-s', '4', '-q']); "
        "assert result.exit_code == 0, result.output; "
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_package_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "frozen_lake", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
