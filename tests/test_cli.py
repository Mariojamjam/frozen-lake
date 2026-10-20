"""The command line: options reach the right calls, defaults and prompts hold, exit codes mean
what they say. Nothing here reads printed text, which Typer may color or wrap."""

import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import frozen_lake.cli as cli
from frozen_lake import __version__
from frozen_lake.cli import app
from frozen_lake.models import StrategyId

TINY = Path(__file__).parent / "data" / "tiny.toml"
COMMANDS = {"tui", "episode", "benchmark", "validate", "plot"}
runner = CliRunner()


@pytest.fixture
def launched(monkeypatch):
    """Explorers that would have opened."""
    from frozen_lake.tui.app import FrozenLakeApp

    opened = []
    monkeypatch.setattr(FrozenLakeApp, "run", lambda self: opened.append(self))
    return opened


@pytest.fixture
def played(monkeypatch):
    """The settings each episode would have been played with."""
    calls = []
    monkeypatch.setattr(cli, "_play_episode", lambda **settings: calls.append(settings))
    return calls


@pytest.fixture
def episodes(monkeypatch):
    """Every real episode `episode` plays: its on_step callback and its result."""
    import frozen_lake.episodes as engine

    real, seen = engine.run_episode, []

    def spy(session, decide, **kwargs):
        result = real(session, decide, **kwargs)
        seen.append((kwargs["on_step"], result))
        return result

    monkeypatch.setattr(engine, "run_episode", spy)
    return seen


def record_calls(monkeypatch, module, name):
    """Replace `module.name` by a wrapper that records its arguments."""
    real, calls = getattr(module, name), []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


# --- the command set and help ------------------------------------------------------------------


def test_the_cli_offers_exactly_the_five_commands():
    assert set(typer.main.get_command(app).commands) == COMMANDS


@pytest.mark.parametrize("command", [[], *([c] for c in sorted(COMMANDS))])
def test_every_command_answers_short_and_long_help(command):
    for flag in ("-h", "--help"):
        assert runner.invoke(app, [*command, flag]).exit_code == 0


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output


def test_the_package_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "frozen_lake", "--version"], capture_output=True, text=True
    )
    assert result.returncode == 0 and __version__ in result.stdout


@pytest.mark.parametrize(
    "args",
    [["--help"], ["episode", "-s", "4", "-q"], ["validate"], ["benchmark", "--dry-run"]],
    ids=["help", "episode", "validate", "benchmark"],
)
def test_commands_without_an_interface_never_import_textual(args):
    code = (
        "import sys; from typer.testing import CliRunner; from frozen_lake.cli import app; "
        f"result = CliRunner().invoke(app, {args!r}); "
        "assert result.exit_code == 0, result.output; "
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- tui ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--size", "6", "--map-seed", "7", "--episode-seed", "3", "--depth", "4", "--beta",
         "0.8", "--rollouts", "50", "--horizon", "13", "--rollout-seed", "9"],
        ["-s", "6", "-m", "7", "-e", "3", "-d", "4", "-b", "0.8", "-n", "50", "--horizon", "13",
         "--rollout-seed", "9"],
    ],
    ids=["long", "short"],
)  # fmt: skip
def test_tui_options_reach_the_explorer(launched, args):
    assert runner.invoke(app, ["tui", *args]).exit_code == 0
    (explorer,) = launched
    assert (len(explorer.desc), explorer.map_seed, explorer.episode_seed) == (6, 7, 3)
    assert explorer.session.max_steps == 4 * 6 * 6  # the default step limit, 4 × size²
    assert (explorer.depth, explorer.beta, explorer.gamma) == (4, 0.8, cli.GAMMA)
    assert (explorer.rollouts, explorer.horizon, explorer.rollout_seed) == (50, 13, 9)
    assert explorer.make_map(7) == explorer.desc  # the same options rebuild the same lake


@pytest.mark.parametrize("args", [[], ["-m", "3"]], ids=["no-options", "no-size"])
def test_tui_never_opens_without_the_lake_size(launched, args):
    assert runner.invoke(app, ["tui", *args]).exit_code != 0
    assert launched == []


@pytest.mark.parametrize(
    "args",
    [
        ["tui", "-s", "1"],
        ["tui", "-s", "4", "--frozen-probability", "0"],
        ["tui", "-s", "4", "--success-rate", "2"],
        ["episode", "-S", "XYZ"],
        ["episode", "-S", "EXP", "-b", "1.5"],
        ["episode", "-S", "EXP", "-d", "0"],
        ["episode", "-S", "BFS-RO", "-n", "0"],
        ["episode", "-S", "BFS-RO", "--horizon", "0"],
        ["episode", "--max-steps", "0"],
    ],
    ids=["size", "frozen", "success", "strategy", "beta", "depth", "rollouts", "horizon", "steps"],
)
def test_invalid_values_are_rejected_before_anything_runs(launched, args):
    assert runner.invoke(app, args).exit_code == 2
    assert launched == []


# --- episode -----------------------------------------------------------------------------------


def test_short_and_long_episode_options_are_the_same_settings(played):
    runner.invoke(app, ["episode", "-S", "EXP-RO", "-s", "6", "-m", "3", "-e", "7", "-d", "4",
                        "-b", "0.8", "-n", "40", "-q"])  # fmt: skip
    runner.invoke(
        app,
        ["episode", "--strategy", "EXP-RO", "--size", "6", "--map-seed", "3", "--episode-seed",
         "7", "--depth", "4", "--beta", "0.8", "--rollouts", "40", "--quiet"],
    )  # fmt: skip
    short, long = played
    assert short == long
    assert StrategyId(short["strategy"]) is StrategyId.EXP_RO and short["rollouts"] == 40


def test_episode_builds_the_chosen_policy_with_its_settings(monkeypatch):
    import frozen_lake.policies as policies

    builds = record_calls(monkeypatch, policies, "build_base_policy")
    result = runner.invoke(app, ["episode", "-S", "EXP", "-d", "3", "-b", "0.85", "-q"])
    assert result.exit_code == 0
    ((strategy, *_), settings) = builds[0]
    assert strategy is StrategyId.EXP and (settings["depth"], settings["beta"]) == (3, 0.85)


@pytest.mark.parametrize("strategy", [StrategyId.BFS_RO, StrategyId.EXP_RO])
def test_rollout_episodes_sample_with_the_given_budget(monkeypatch, episodes, strategy):
    import frozen_lake.rollout as rollout

    deciders = record_calls(monkeypatch, rollout, "rollout_decider")
    args = ["episode", "-S", str(strategy), "-n", "5", "--horizon", "20", "-q"]
    assert runner.invoke(app, args).exit_code == 0
    ((_, settings),) = deciders
    assert (settings["samples"], settings["horizon"]) == (5, 20)
    ((_, result),) = episodes
    assert result.strategy is strategy and result.simulated_transitions > 0


def test_quiet_episodes_report_no_steps(episodes):
    runner.invoke(app, ["episode", "-s", "4"])
    runner.invoke(app, ["episode", "-s", "4", "-q"])
    (verbose, _), (quiet, _) = episodes
    assert verbose is not None and quiet is None


def test_episodes_are_reproducible_and_the_rollout_seed_changes_only_the_simulations(episodes):
    base = ["episode", "-S", "BFS-RO", "-n", "5", "-s", "6", "-q"]
    for args in (base, base, [*base, "--rollout-seed", "3"], [*base, "--rollout-seed", "3"]):
        assert runner.invoke(app, args).exit_code == 0
    first, again, seeded, seeded_again = (result for _, result in episodes)

    def choices(result):
        return [(d.state, d.action, d.estimated_action_values) for d in result.decisions]

    assert choices(first) == choices(again)
    assert choices(seeded) == choices(seeded_again)
    assert choices(seeded) != choices(first)


def test_episode_without_options_outside_a_terminal_plays_the_defaults(monkeypatch, played):
    from frozen_lake.tui.episode_form import EpisodeForm

    def no_form(form):
        raise AssertionError("the form must not open without a terminal")

    monkeypatch.setattr(EpisodeForm, "run", no_form)
    assert runner.invoke(app, ["episode"]).exit_code == 0
    (settings,) = played
    assert StrategyId(settings["strategy"]) is StrategyId.BFS and settings["quiet"] is False


def test_episode_without_options_in_a_terminal_plays_what_the_form_returns(monkeypatch, played):
    from frozen_lake.tui.episode_form import EpisodeForm

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    chosen = {}

    def choose(form):
        chosen.update({**form.initial, "strategy": StrategyId.EXP, "size": 5, "quiet": True})
        return dict(chosen)

    monkeypatch.setattr(EpisodeForm, "run", choose)
    assert runner.invoke(app, ["episode"]).exit_code == 0
    assert played == [chosen]


def test_cancelling_the_form_plays_nothing(monkeypatch, played):
    from frozen_lake.tui.episode_form import EpisodeForm

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(EpisodeForm, "run", lambda form: None)
    assert runner.invoke(app, ["episode"]).exit_code == 0
    assert played == []


# --- validate ----------------------------------------------------------------------------------


def test_validate_checks_files_as_configurations_and_folders_as_runs(tiny_run, run_copy):
    assert runner.invoke(app, ["validate"]).exit_code == 0  # the study's configuration
    assert runner.invoke(app, ["validate", "-i", str(TINY)]).exit_code == 0
    assert runner.invoke(app, ["validate", "-i", str(tiny_run)]).exit_code == 0
    (run_copy / "summary.csv").unlink()
    assert runner.invoke(app, ["validate", "-i", str(run_copy)]).exit_code == 1


def test_validate_rejects_bad_or_missing_configurations(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(TINY.read_text(encoding="utf-8").replace("sizes = [3, 4]", "sizes = []"))
    assert runner.invoke(app, ["validate", "-i", str(bad)]).exit_code == 2
    assert runner.invoke(app, ["validate", "-i", str(tmp_path / "nope.toml")]).exit_code == 2


# --- benchmark ---------------------------------------------------------------------------------


@pytest.fixture
def experiments(monkeypatch, tmp_path):
    """Runs `benchmark` would have started, with the folder it was told to use."""
    import frozen_lake.experiments as experiments

    calls = []

    def fake_run(config, *, output_root=None, config_path=None, progress=None):
        calls.append({"config": config, "output_root": output_root, "config_path": config_path})
        return tmp_path / "run"

    monkeypatch.setattr(experiments, "run_experiment", fake_run)
    return calls


def test_benchmark_reads_the_given_configuration_and_output_folder(experiments, tmp_path):
    args = ["benchmark", "-i", str(TINY), "-o", str(tmp_path), "-y"]
    assert runner.invoke(app, args).exit_code == 0
    (call,) = experiments
    assert (call["config_path"], call["output_root"]) == (TINY, tmp_path)


def test_benchmark_defaults_to_the_study_configuration(experiments):
    assert runner.invoke(app, ["benchmark", "-y"]).exit_code == 0
    (call,) = experiments
    assert call["config_path"] == cli.DEFAULT_CONFIG and call["output_root"] is None


def test_benchmark_takes_a_folder_that_holds_an_experiment_toml(experiments, tmp_path):
    (tmp_path / "experiment.toml").write_text(TINY.read_text(encoding="utf-8"))
    assert runner.invoke(app, ["benchmark", "-i", str(tmp_path), "-y"]).exit_code == 0
    assert experiments[0]["config_path"] == tmp_path / "experiment.toml"


@pytest.mark.parametrize(
    ("args", "answer", "code"),
    [(["--dry-run"], None, 0), ([], "n\n", 1)],
    ids=["dry-run", "declined"],
)
def test_benchmark_runs_nothing_when_dry_or_declined(experiments, args, answer, code):
    result = runner.invoke(app, ["benchmark", "-i", str(TINY), *args], input=answer)
    assert result.exit_code == code
    assert experiments == []


def test_benchmark_runs_once_confirmed(experiments):
    assert runner.invoke(app, ["benchmark", "-i", str(TINY)], input="y\n").exit_code == 0
    assert len(experiments) == 1


def test_a_benchmark_from_the_cli_writes_a_valid_run(tmp_path):
    args = ["benchmark", "-i", str(TINY), "-o", str(tmp_path), "-y"]
    assert runner.invoke(app, args).exit_code == 0
    (run_dir,) = tmp_path.iterdir()
    assert runner.invoke(app, ["validate", "-i", str(run_dir)]).exit_code == 0


# --- plot --------------------------------------------------------------------------------------


@pytest.fixture
def analyses(monkeypatch):
    """Runs `plot` would have analyzed, with where it was told to write."""
    import frozen_lake.analysis as analysis

    calls = []

    def fake_analyze(run, output, **kwargs):
        calls.append((run, output, kwargs))
        raise typer.Exit()

    monkeypatch.setattr(analysis, "analyze_run", fake_analyze)
    return calls


def make_runs(root, *names):
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / "metadata.json").write_text("{}")


def test_plot_analyzes_the_given_run_where_it_is_told(analyses, tmp_path):
    runner.invoke(app, ["plot", "-i", "some/run", "-o", str(tmp_path), "--resamples", "500"])
    ((run, output, settings),) = analyses
    assert (run, output, settings["resamples"]) == (Path("some/run"), tmp_path, 500)


def test_plot_writes_inside_the_run_by_default(run_copy):
    assert runner.invoke(app, ["plot", "-i", str(run_copy), "--resamples", "100"]).exit_code == 0
    assert (run_copy / "analysis" / "analysis.json").is_file()


def test_plot_finds_only_folders_that_hold_a_run(monkeypatch, tmp_path):
    make_runs(tmp_path, "canonical/experiment-1", "generated/b", "generated/a")
    (tmp_path / "generated" / "not-a-run").mkdir()
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    # The study's runs come first, then the generated ones.
    assert cli._runs() == [
        tmp_path / "canonical" / "experiment-1",
        tmp_path / "generated" / "a",
        tmp_path / "generated" / "b",
    ]


def test_plot_asks_which_run_in_a_terminal(monkeypatch, analyses, tmp_path):
    make_runs(tmp_path, "generated/a", "generated/b")
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    runner.invoke(app, ["plot"], input="2\n")
    assert analyses[0][0] == tmp_path / "generated" / "b"


@pytest.mark.parametrize("has_runs", [True, False], ids=["no-terminal", "no-runs"])
def test_plot_analyzes_nothing_without_a_run_it_can_choose(
    monkeypatch, analyses, tmp_path, has_runs
):
    if has_runs:
        make_runs(tmp_path, "generated/a")
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    assert runner.invoke(app, ["plot"]).exit_code == 2
    assert analyses == []


def test_plot_needs_enough_resamples(analyses):
    assert runner.invoke(app, ["plot", "-i", "run", "--resamples", "10"]).exit_code == 2
    assert analyses == []
