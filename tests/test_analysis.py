import csv
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from frozen_lake.analysis import InvalidResults, analyze_run, analyze_size, pareto_frontier
from frozen_lake.analysis.bootstrap import amortization_sensitivity
from frozen_lake.analysis.loading import BASES, SizeData, load_decisions, load_run
from frozen_lake.analysis.summaries import build_summaries, shared_state_times
from frozen_lake.cli import app
from frozen_lake.experiments import SAMPLED, Arm, load_config, parse_config, run_experiment
from frozen_lake.models import StrategyId

TINY = Path(__file__).parent / "data" / "tiny.toml"
RESAMPLES = 200
PNG = b"\x89PNG\r\n\x1a\n"
runner = CliRunner()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return run_experiment(load_config(TINY), output_root=tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="module")
def analysis(run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis")
    return analyze_run(run_dir, out, resamples=RESAMPLES)


def synthetic(success: np.ndarray, *, build_bfs: float = 1.0, build_exp: float = 3.0) -> SizeData:
    """Four arms over 3 maps and 4 episodes, with chosen successes and constant times."""
    arms = (
        Arm(StrategyId.BFS),
        Arm(StrategyId.EXP),
        Arm(StrategyId.BFS_RO, SAMPLED, 5),
        Arm(StrategyId.EXP_RO, SAMPLED, 5),
    )
    build = np.empty((len(BASES), 3, 2))
    build[0], build[1] = build_bfs, build_exp
    return SizeData(
        size=4,
        map_ids=("4-0", "4-1", "4-2"),
        arms=arms,
        success=success,
        agent_ms=np.full(success.shape, 2.0),
        transitions=np.zeros(success.shape),
        build_ms=build,
    )


# --- the Pareto frontier ---------------------------------------------------------------------


def test_pareto_frontier_drops_dominated_points():
    points = [(1.0, 0.5), (2.0, 0.9), (3.0, 0.9), (2.0, 0.4)]
    assert pareto_frontier(points) == {0, 1}


def test_pareto_frontier_keeps_identical_points_and_handles_none():
    assert pareto_frontier([(1.0, 1.0), (1.0, 1.0)]) == {0, 1}
    assert pareto_frontier([]) == frozenset()


# --- the bootstrap ---------------------------------------------------------------------------


def test_constant_data_gives_degenerate_intervals_and_exact_amortized_time():
    result = analyze_size(synthetic(np.ones((4, 3, 4))), resamples=RESAMPLES, seed=1)
    assert all((i.estimate, i.low, i.high) == (1.0, 1.0, 1.0) for i in result.success)
    # amortized time = median build / R + agent time, with R = E = 4
    times = [t.estimate for t in result.amortized_ms if t is not None]
    assert times == pytest.approx([1 / 4 + 2, 3 / 4 + 2, 1 / 4 + 2, 3 / 4 + 2])
    assert result.frontier == {0, 2}  # BFS and BFS-RO tie on cost; Expectimax arms are slower
    for diff in result.differences:
        assert diff.success.estimate == 0.0
        assert diff.amortized_ms is not None
        assert diff.amortized_ms.estimate == pytest.approx(0.5)


def test_interval_reflects_the_variation_between_maps():
    success = np.ones((4, 3, 4))
    success[2, 0] = 0.0  # BFS-RO fails every episode on one map out of three
    result = analyze_size(synthetic(success), resamples=1000, seed=3)
    rollout = result.success[2]
    assert rollout.estimate == pytest.approx(2 / 3)
    assert rollout.low == 0.0 and rollout.high == 1.0  # a resample can pick one map thrice


def test_same_seed_gives_the_same_intervals():
    success = np.random.default_rng(0).integers(0, 2, size=(4, 3, 4)).astype(float)
    first = analyze_size(synthetic(success), resamples=RESAMPLES, seed=9)
    second = analyze_size(synthetic(success), resamples=RESAMPLES, seed=9)
    assert first == second


def test_paired_difference_uses_the_same_episodes_of_both_arms():
    success = np.zeros((4, 3, 4))
    success[2] = 1.0  # BFS-RO always succeeds, EXP-RO never does
    diff = analyze_size(synthetic(success), resamples=RESAMPLES, seed=2).differences[0]
    assert (diff.success.estimate, diff.success.low, diff.success.high) == (-1.0, -1.0, -1.0)


def test_amortization_curve_agrees_with_the_estimate_at_r_equal_e():
    data = synthetic(np.ones((4, 3, 4)))
    curve = amortization_sensitivity(data, [1, 4, 1000])[5]
    diff = analyze_size(data, resamples=RESAMPLES, seed=1).differences[0]
    assert diff.amortized_ms is not None
    assert curve[1] == pytest.approx(diff.amortized_ms.estimate)
    assert curve[0] > curve[1] > curve[2]  # the build gap matters less as R grows


# --- reading a run ---------------------------------------------------------------------------


def test_load_run_fills_every_array(run_dir):
    run = load_run(run_dir)
    config = run.config
    assert sorted(run.sizes) == sorted(config.environment.sizes)
    for data in run.sizes.values():
        shape = (len(data.arms), len(data.map_ids), config.experiment.episodes_per_map)
        assert data.success.shape == shape
        assert not np.isnan(data.success).any()
        assert set(np.unique(data.success)) <= {0.0, 1.0}
        assert not np.isnan(data.agent_ms).any()
        assert not np.isnan(data.build_ms).any()
        assert data.build_ms.shape[2] == config.policies.build_repeats


def test_point_estimates_match_the_runs_own_summary(run_dir, analysis):
    summary = {
        (r["size"], r["strategy"], r["estimator"], r["rollout_n"]): float(r["success_rate"])
        for r in read_csv(run_dir / "summary.csv")
    }
    for row in read_csv(analysis.output_dir / "tables" / "arms.csv"):
        key = (row["size"], row["strategy"], row["estimator"], row["rollout_n"])
        assert float(row["success_rate"]) == pytest.approx(summary[key])


def test_paired_difference_matches_a_hand_computation(run_dir, analysis):
    episodes = read_csv(run_dir / "episode_metrics.csv")

    def rate(size: str, strategy: str, n: str) -> float:
        rows = [
            float(e["success"])
            for e in episodes
            if (e["size"], e["strategy"], e["rollout_n"], e["estimator"])
            == (size, strategy, n, SAMPLED)
        ]
        return sum(rows) / len(rows)

    for row in read_csv(analysis.output_dir / "tables" / "paired_differences.csv"):
        if row["rollout_n"]:
            expected = rate(row["size"], "EXP-RO", row["rollout_n"]) - rate(
                row["size"], "BFS-RO", row["rollout_n"]
            )
            assert float(row["delta_success"]) == pytest.approx(expected)


def test_build_summaries_report_each_algorithms_own_counters(run_dir):
    run = load_run(run_dir)
    by_base = {}
    for summary in build_summaries(run):
        by_base.setdefault(summary.base, summary)
        assert summary.builds == summary.maps * run.config.policies.build_repeats
        assert summary.q1_ms <= summary.median_ms <= summary.q3_ms
    assert "peak_queue" in by_base["BFS"].counters
    assert "cache_entries" in by_base["EXP"].counters
    assert "cache_entries" not in by_base["BFS"].counters


def test_shared_state_ratios_are_positive_and_averaged(run_dir):
    run = load_run(run_dir)
    decisions = load_decisions(run_dir)
    assert decisions is not None
    times = shared_state_times(run, decisions)
    assert times
    for entry in times:
        assert all(r > 0 for r in entry.ratios)
        assert entry.mean_ratio == pytest.approx(sum(entry.ratios) / len(entry.ratios))


# --- the whole analysis ----------------------------------------------------------------------


def test_analysis_writes_figures_tables_and_a_record(analysis):
    assert len(analysis.figures) == 8
    for path in analysis.figures:
        assert path.read_bytes().startswith(PNG)
    names = {path.name for path in analysis.tables}
    assert names == {
        "arms.csv",
        "paired_differences.csv",
        "amortization_sensitivity.csv",
        "policy_builds.csv",
        "decision_times.csv",
        "shared_state_decision_time.csv",
    }
    record = json.loads((analysis.output_dir / "analysis.json").read_text())
    assert record["bootstrap"]["resamples"] == RESAMPLES
    assert record["bootstrap"]["seed"] == load_config(TINY).experiment.master_seed
    assert record["decisions_logged"] is True
    assert len(record["figures"]) == 8


def test_exact_reference_has_no_cost_in_the_tables(analysis):
    rows = read_csv(analysis.output_dir / "tables" / "arms.csv")
    exact = [r for r in rows if r["estimator"] == "exact"]
    assert exact
    assert all(r["amortized_ms"] == "" and r["on_frontier"] == "" for r in exact)
    assert all(r["success_rate"] != "" for r in exact)


def test_the_same_run_and_seed_give_identical_tables(run_dir, analysis, tmp_path):
    again = analyze_run(run_dir, tmp_path, resamples=RESAMPLES)
    for path in analysis.tables:
        assert (again.output_dir / "tables" / path.name).read_bytes() == path.read_bytes()


def test_a_run_without_logged_decisions_skips_what_needs_them(tmp_path):
    raw = tomllib.loads(TINY.read_text())
    raw["experiment"]["write_decisions"] = False
    run_dir = run_experiment(parse_config(raw), output_root=tmp_path / "runs")
    result = analyze_run(run_dir, tmp_path / "out", resamples=RESAMPLES)
    figure_names = {p.name for p in result.figures}
    assert "figure_c_cost_vs_samples.png" in figure_names
    assert not any("figure_g" in name for name in figure_names)
    assert not any(p.name == "decision_times.csv" for p in result.tables)


def test_damaged_results_are_not_analyzed(run_dir, tmp_path):
    damaged = tmp_path / "damaged"
    shutil.copytree(run_dir, damaged)
    (damaged / "summary.csv").unlink()
    with pytest.raises(InvalidResults) as error:
        analyze_run(damaged, tmp_path / "out", resamples=RESAMPLES)
    assert error.value.problems
    assert not (tmp_path / "out").exists()


# --- the command -----------------------------------------------------------------------------


def test_plot_command_writes_the_analysis(run_dir, tmp_path):
    result = runner.invoke(
        app,
        ["plot", "-i", str(run_dir), "-o", str(tmp_path), "--resamples", "100"],
    )
    assert result.exit_code == 0, result.output
    assert "analysis written to" in result.output
    assert (tmp_path / "figures" / "figure_a_success_vs_amortized_time.png").exists()


def test_plot_defaults_to_an_analysis_folder_inside_the_run(run_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    result = runner.invoke(app, ["plot", "-i", str(copy), "--resamples", "100"])
    assert result.exit_code == 0, result.output
    assert (copy / "analysis" / "analysis.json").exists()


def test_plot_reports_damaged_results(run_dir, tmp_path):
    damaged = tmp_path / "damaged"
    shutil.copytree(run_dir, damaged)
    (damaged / "episode_metrics.csv").unlink()
    result = runner.invoke(app, ["plot", "-i", str(damaged)])
    assert result.exit_code == 1
    assert "failed validation" in result.output


def test_plot_needs_a_run_and_enough_resamples(run_dir):
    assert runner.invoke(app, ["plot"]).exit_code == 2
    assert runner.invoke(app, ["plot", "-i", str(run_dir), "--resamples", "10"]).exit_code == 2
