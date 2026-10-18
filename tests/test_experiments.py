import csv
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from frozen_lake.experiments import (
    DECISION_COLUMNS,
    EPISODE_COLUMNS,
    POLICY_COLUMNS,
    SUMMARY_COLUMNS,
    ConfigError,
    arms,
    derive_seed,
    generate_maps,
    load_config,
    parse_config,
    run_experiment,
    validate_run,
    workload,
)
from frozen_lake.models import StrategyId

TINY = Path(__file__).parent / "data" / "tiny.toml"
EXPERIMENT = Path(__file__).parent.parent / "configs" / "experiment.toml"
TIME_COLUMNS = {"agent_compute_ns", "wall_ns", "decision_time_ns"}


def raw_tiny() -> dict:
    return tomllib.loads(TINY.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> Path:
    return run_experiment(load_config(TINY), output_root=tmp_path_factory.mktemp("runs"))


# --- configuration -----------------------------------------------------------------------------


def test_the_shipped_experiment_configuration_is_valid():
    config = load_config(EXPERIMENT)
    assert config.environment.sizes == (4, 6, 8, 10, 12)
    assert config.rollout.samples_per_action == (10, 25, 50, 100, 200, 400, 800)
    assert workload(config).episodes == 5 * 5 * 30 * 18


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("rollout", "sampels_per_action", [1], "unknown key rollout.sampels_per_action"),
        ("environment", "sizes", [], "environment.sizes must not be empty"),
        ("environment", "sizes", [4, 4], "must not repeat"),
        ("environment", "sizes", [1], "integers >= 2"),
        ("policies", "heuristic_beta", 1.5, "beta must be in (0, 1)"),
        ("policies", "build_repeats", 0, "build_repeats must be at least 1"),
        ("environment", "success_rate", "high", "must be float"),
        ("experiment", "write_decisions", 1, "must be bool"),
    ],
)
def test_invalid_configurations_are_rejected_with_a_clear_message(section, key, value, message):
    raw = raw_tiny()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_config(raw)


def test_a_missing_key_and_an_unknown_section_are_both_reported():
    raw = raw_tiny()
    del raw["experiment"]["master_seed"]
    raw["extras"] = {}
    with pytest.raises(ConfigError) as error:
        parse_config(raw)
    assert "missing key experiment.master_seed" in str(error.value)
    assert "unknown section [extras]" in str(error.value)


def test_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_equal_settings_have_equal_fingerprints():
    assert load_config(TINY).fingerprint() == parse_config(raw_tiny()).fingerprint()
    raw = raw_tiny()
    raw["experiment"]["master_seed"] = 8
    assert parse_config(raw).fingerprint() != load_config(TINY).fingerprint()


# --- seeds, maps and arms ----------------------------------------------------------------------


def test_seeds_are_stable_and_distinct_per_role():
    assert derive_seed(2026, "map", 6, 0) == derive_seed(2026, "map", 6, 0)
    assert derive_seed(2026, "map", 6, 0) == 6299549060498745003  # pinned: never changes silently
    assert derive_seed(2026, "map", 6, 0) != derive_seed(2026, "map", 6, 1)
    assert derive_seed(2026, "map", 6, 0) != derive_seed(2026, "episode", 6, 0)
    assert 0 <= derive_seed(2026, "x") < 2**63


def test_maps_are_reproducible_and_reachable():
    config = load_config(TINY)
    first, second = generate_maps(config), generate_maps(config)
    assert first == second
    assert [m.map_id for m in first] == ["3-0", "3-1", "4-0", "4-1"]
    assert all(m.reachable for m in first)


def test_base_strategies_get_one_arm_each_and_rollout_one_per_n_plus_exact():
    labels = [arm.label for arm in arms(load_config(TINY))]
    assert labels == [
        "BFS", "EXP",
        "BFS-RO N=2", "EXP-RO N=2", "BFS-RO N=4", "EXP-RO N=4",
        "BFS-RO exact", "EXP-RO exact",
    ]  # fmt: skip


# --- a whole run -------------------------------------------------------------------------------


def test_a_run_writes_every_file_with_the_documented_columns(tiny_run):
    names = sorted(p.name for p in tiny_run.iterdir())
    assert names == [
        "configuration.json", "decision_metrics.csv", "episode_metrics.csv", "maps.json",
        "metadata.json", "policy_metrics.csv", "summary.csv",
    ]  # fmt: skip
    for name, columns in [
        ("policy_metrics.csv", POLICY_COLUMNS),
        ("episode_metrics.csv", EPISODE_COLUMNS),
        ("decision_metrics.csv", DECISION_COLUMNS),
        ("summary.csv", SUMMARY_COLUMNS),
    ]:
        with (tiny_run / name).open(newline="") as handle:
            assert next(csv.reader(handle)) == columns


def test_the_written_run_passes_validation(tiny_run):
    assert validate_run(tiny_run) == []


def test_every_arm_plays_the_same_maps_and_episode_seeds(tiny_run):
    episodes = read_csv(tiny_run / "episode_metrics.csv")
    by_arm = {}
    for row in episodes:
        arm = (row["strategy"], row["estimator"], row["rollout_n"])
        by_arm.setdefault(arm, set()).add(
            (row["map_id"], row["episode_index"], row["episode_seed"])
        )
    assert len(by_arm) == 8
    assert len({frozenset(value) for value in by_arm.values()}) == 1


def test_base_only_strategies_run_once_per_episode_not_once_per_n(tiny_run):
    episodes = read_csv(tiny_run / "episode_metrics.csv")
    bfs = [r for r in episodes if r["strategy"] == "BFS"]
    assert len(bfs) == 4 * 2  # 4 maps × 2 episodes
    assert {r["estimator"] for r in bfs} == {""}
    assert {r["rollout_n"] for r in bfs} == {""}


def test_exact_reference_rows_have_no_n_and_sample_nothing(tiny_run):
    episodes = read_csv(tiny_run / "episode_metrics.csv")
    exact = [r for r in episodes if r["estimator"] == "exact"]
    assert exact and {r["strategy"] for r in exact} == {"BFS-RO", "EXP-RO"}
    assert {r["rollout_n"] for r in exact} == {""}
    assert {r["simulated_transitions"] for r in exact} == {"0"}
    summary = read_csv(tiny_run / "summary.csv")
    exact_summary = [r for r in summary if r["estimator"] == "exact"]
    assert all(r["amortized_ms_per_episode"] == "" for r in exact_summary)  # not a cost to compare


def test_each_policy_build_is_repeated_and_recorded(tiny_run):
    builds = read_csv(tiny_run / "policy_metrics.csv")
    assert len(builds) == 4 * 2 * 3  # maps × base policies × build_repeats
    exp = [r for r in builds if r["base_policy"] == "EXP"]
    assert {r["depth"] for r in exp} == {"3"}
    assert all(int(r["build_time_ns"]) > 0 for r in builds)


def test_decisions_add_up_to_each_episode(tiny_run):
    decisions = read_csv(tiny_run / "decision_metrics.csv")
    episodes = read_csv(tiny_run / "episode_metrics.csv")
    assert len(decisions) == sum(int(r["steps"]) for r in episodes)
    sampled = [d for d in decisions if d["estimator"] == "sampled"]
    assert all(d["q_left"] != "" for d in sampled)
    assert all(d["q_left"] == "" for d in decisions if d["strategy"] in ("BFS", "EXP"))


def test_metadata_records_provenance(tiny_run):
    meta = json.loads((tiny_run / "metadata.json").read_text())
    for key in ("config_sha256", "maps_sha256", "seed_scheme", "python", "packages", "units"):
        assert key in meta
    assert tiny_run.name.endswith(meta["config_sha256"][:8])


def test_the_same_configuration_reproduces_the_same_outcomes(tmp_path, tiny_run):
    again = run_experiment(load_config(TINY), output_root=tmp_path)

    def outcomes(run_dir, name):
        return [
            {k: v for k, v in row.items() if k not in TIME_COLUMNS}
            for row in read_csv(run_dir / name)
        ]

    assert outcomes(again, "episode_metrics.csv") == outcomes(tiny_run, "episode_metrics.csv")
    assert outcomes(again, "decision_metrics.csv") == outcomes(tiny_run, "decision_metrics.csv")


def test_decisions_can_be_turned_off(tmp_path):
    raw = raw_tiny()
    raw["experiment"]["write_decisions"] = False
    run_dir = run_experiment(parse_config(raw), output_root=tmp_path)
    assert not (run_dir / "decision_metrics.csv").exists()
    assert validate_run(run_dir) == []


def test_an_existing_results_folder_is_never_overwritten(tmp_path, monkeypatch):
    import frozen_lake.experiments.runner as runner

    monkeypatch.setattr(runner, "run_folder_name", lambda config, now=None: "fixed")
    run_experiment(load_config(TINY), output_root=tmp_path)
    with pytest.raises(FileExistsError):
        run_experiment(load_config(TINY), output_root=tmp_path)


# --- validation catches damage -----------------------------------------------------------------


def test_validation_detects_a_missing_episode(tmp_path):
    run_dir = run_experiment(load_config(TINY), output_root=tmp_path)
    path = run_dir / "episode_metrics.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))
    assert any("missing" in problem for problem in validate_run(run_dir))


def test_validation_detects_a_duplicated_episode(tmp_path):
    run_dir = run_experiment(load_config(TINY), output_root=tmp_path)
    path = run_dir / "episode_metrics.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join([*lines, lines[1]]))
    assert any("duplicate" in problem for problem in validate_run(run_dir))


def test_validation_detects_a_tampered_map(tmp_path):
    run_dir = run_experiment(load_config(TINY), output_root=tmp_path)
    path = run_dir / "maps.json"
    maps = json.loads(path.read_text())
    maps[0]["rows"][0] = "SHG"
    path.write_text(json.dumps(maps))
    assert any("does not match its seed" in problem for problem in validate_run(run_dir))


def test_the_runner_never_imports_textual():
    code = (
        "import sys; import frozen_lake.experiments; "
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rollout_strategies_follow_their_base_policy():
    assert {arm.strategy.base for arm in arms(load_config(TINY))} == {
        StrategyId.BFS,
        StrategyId.EXP,
    }
