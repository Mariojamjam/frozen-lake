import json
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest
from helpers import read_csv, write_csv

from frozen_lake.experiments import (
    DECISION_COLUMNS,
    EPISODE_COLUMNS,
    EXACT,
    POLICY_COLUMNS,
    SAMPLED,
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

ROOT = Path(__file__).parent.parent
TINY = Path(__file__).parent / "data" / "tiny.toml"
EXPERIMENT = ROOT / "configs" / "experiment.toml"
CANONICAL = ROOT / "results" / "canonical" / "experiment-1"
TIME_COLUMNS = {"agent_compute_ns", "wall_ns", "decision_time_ns"}


def raw_tiny() -> dict:
    return tomllib.loads(TINY.read_text(encoding="utf-8"))


def arm_key(row: dict) -> tuple:
    return row["strategy"], row["estimator"], row["rollout_n"]


# --- configuration -----------------------------------------------------------------------------


def test_the_shipped_configuration_is_the_one_the_canonical_run_used():
    # The study's protocol must not drift from the run it reports. Only the output folder of
    # new runs moved (results/ to results/generated/), which affects no measurement.
    shipped = json.loads(json.dumps(load_config(EXPERIMENT).as_dict()))  # tuples as lists
    recorded = json.loads((CANONICAL / "configuration.json").read_text(encoding="utf-8"))
    shipped["experiment"].pop("output_dir")
    recorded["experiment"].pop("output_dir")
    assert shipped == recorded


def test_the_workload_counts_every_arm_on_every_map_and_episode(tiny_config):
    config = tiny_config
    load = workload(config)
    maps = len(config.environment.sizes) * config.experiment.maps_per_size
    assert load.episodes == maps * config.experiment.episodes_per_map * len(arms(config))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("rollout", "sampels_per_action", [1]),  # a misspelled key is an error, not ignored
        ("environment", "sizes", []),
        ("environment", "sizes", [4, 4]),
        ("environment", "sizes", [1]),
        ("policies", "heuristic_beta", 1.5),
        ("policies", "build_repeats", 0),
        ("environment", "success_rate", "high"),
        ("experiment", "write_decisions", 1),
    ],
    ids=["unknown-key", "empty", "repeated", "too-small", "beta", "repeats", "type", "bool"],
)
def test_invalid_configurations_are_rejected(section, key, value):
    raw = raw_tiny()
    raw[section][key] = value
    with pytest.raises(ConfigError):
        parse_config(raw)


def test_every_problem_of_a_configuration_is_reported_at_once():
    raw = raw_tiny()
    del raw["experiment"]["master_seed"]
    raw["extras"] = {}
    with pytest.raises(ConfigError) as error:
        parse_config(raw)
    assert "experiment.master_seed" in str(error.value) and "extras" in str(error.value)


def test_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")


def test_equal_settings_have_equal_fingerprints(tiny_config):
    assert tiny_config.fingerprint() == parse_config(raw_tiny()).fingerprint()
    raw = raw_tiny()
    raw["experiment"]["master_seed"] += 1
    assert parse_config(raw).fingerprint() != tiny_config.fingerprint()


# --- seeds, maps and arms ----------------------------------------------------------------------


def test_seeds_are_stable_and_distinct_per_role():
    # Pinned on purpose: the canonical run's maps and episodes come from this seed scheme, so a
    # change to it must break this test rather than silently change every seed.
    assert derive_seed(2026, "map", 6, 0) == 6299549060498745003
    assert derive_seed(2026, "map", 6, 0) != derive_seed(2026, "map", 6, 1)
    assert derive_seed(2026, "map", 6, 0) != derive_seed(2026, "episode", 6, 0)
    assert 0 <= derive_seed(2026, "x") < 2**63


def test_maps_are_reproducible_reachable_and_one_set_per_size(tiny_config):
    first, second = generate_maps(tiny_config), generate_maps(tiny_config)
    assert first == second
    assert all(m.reachable for m in first)
    per_size = defaultdict(int)
    for record in first:
        per_size[len(record.rows)] += 1
    assert per_size == dict.fromkeys(tiny_config.environment.sizes, 2)


def test_each_base_strategy_runs_once_and_each_rollout_once_per_n_plus_exact(tiny_config):
    expected = {(StrategyId.BFS, None, None), (StrategyId.EXP, None, None)}
    for n in tiny_config.rollout.samples_per_action:
        expected |= {(StrategyId.BFS_RO, SAMPLED, n), (StrategyId.EXP_RO, SAMPLED, n)}
    expected |= {(StrategyId.BFS_RO, EXACT, None), (StrategyId.EXP_RO, EXACT, None)}
    found = [(arm.strategy, arm.estimator, arm.rollout_n) for arm in arms(tiny_config)]
    assert len(found) == len(set(found)) and set(found) == expected


# --- a whole run -------------------------------------------------------------------------------


def test_a_run_writes_every_file_with_the_documented_columns(tiny_run):
    for name, columns in [
        ("policy_metrics.csv", POLICY_COLUMNS),
        ("episode_metrics.csv", EPISODE_COLUMNS),
        ("decision_metrics.csv", DECISION_COLUMNS),
        ("summary.csv", SUMMARY_COLUMNS),
    ]:
        rows = read_csv(tiny_run / name)
        assert rows and list(rows[0]) == columns
    for name in ("configuration.json", "maps.json", "metadata.json"):
        assert (tiny_run / name).is_file()


def test_the_written_run_passes_validation(tiny_run):
    assert validate_run(tiny_run) == []


def test_every_arm_plays_the_same_maps_and_episode_seeds(tiny_run, tiny_config):
    by_arm = defaultdict(set)
    for row in read_csv(tiny_run / "episode_metrics.csv"):
        by_arm[arm_key(row)].add((row["map_id"], row["episode_index"], row["episode_seed"]))
    assert len(by_arm) == len(arms(tiny_config))
    assert len({frozenset(value) for value in by_arm.values()}) == 1


def test_base_only_strategies_run_once_per_episode_not_once_per_n(tiny_run, tiny_config):
    episodes = workload(tiny_config).episodes // len(arms(tiny_config))  # per arm
    for strategy in ("BFS", "EXP"):
        rows = [r for r in read_csv(tiny_run / "episode_metrics.csv") if r["strategy"] == strategy]
        assert len(rows) == episodes
        assert {r["estimator"] for r in rows} == {r["rollout_n"] for r in rows} == {""}


def test_the_exact_reference_samples_nothing_and_has_no_cost(tiny_run):
    exact = [r for r in read_csv(tiny_run / "episode_metrics.csv") if r["estimator"] == EXACT]
    assert exact
    assert {r["rollout_n"] for r in exact} == {""}
    assert {r["simulated_transitions"] for r in exact} == {"0"}
    summary = [r for r in read_csv(tiny_run / "summary.csv") if r["estimator"] == EXACT]
    assert summary and all(r["amortized_ms_per_episode"] == "" for r in summary)


def test_each_policy_build_is_repeated_and_recorded(tiny_run, tiny_config):
    builds = read_csv(tiny_run / "policy_metrics.csv")
    maps = len(tiny_config.environment.sizes) * tiny_config.experiment.maps_per_size
    assert len(builds) == maps * 2 * tiny_config.policies.build_repeats  # BFS and EXP
    assert {r["depth"] for r in builds if r["base_policy"] == "EXP"} == {
        str(tiny_config.policies.expectimax_depth)
    }
    assert all(int(r["build_time_ns"]) > 0 for r in builds)


def test_decisions_add_up_to_each_episode_and_carry_rollout_values(tiny_run):
    decisions = read_csv(tiny_run / "decision_metrics.csv")
    episodes = read_csv(tiny_run / "episode_metrics.csv")
    assert len(decisions) == sum(int(r["steps"]) for r in episodes)
    assert all(d["q_left"] != "" for d in decisions if d["estimator"] == SAMPLED)
    assert all(d["q_left"] == "" for d in decisions if d["strategy"] in ("BFS", "EXP"))


def test_metadata_records_provenance(tiny_run, tiny_config):
    meta = json.loads((tiny_run / "metadata.json").read_text(encoding="utf-8"))
    for key in ("config_sha256", "maps_sha256", "seed_scheme", "python", "packages", "units"):
        assert key in meta
    assert meta["config_sha256"] == tiny_config.fingerprint()
    assert tiny_run.name.endswith(meta["config_sha256"][:8])


def test_the_same_configuration_reproduces_the_same_outcomes(tmp_path, tiny_run, tiny_config):
    again = run_experiment(tiny_config, output_root=tmp_path)

    def outcomes(run_dir, name):
        rows = read_csv(run_dir / name)
        return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in rows]

    for name in ("episode_metrics.csv", "decision_metrics.csv"):
        assert outcomes(again, name) == outcomes(tiny_run, name)


def test_decisions_can_be_turned_off(tmp_path):
    raw = raw_tiny()
    raw["experiment"]["write_decisions"] = False
    run_dir = run_experiment(parse_config(raw), output_root=tmp_path)
    assert not (run_dir / "decision_metrics.csv").exists()
    assert validate_run(run_dir) == []


def test_an_existing_results_folder_is_never_overwritten(tmp_path, monkeypatch, tiny_config):
    import frozen_lake.experiments.runner as runner

    monkeypatch.setattr(runner, "run_folder_name", lambda config, now=None: "fixed")
    run_experiment(tiny_config, output_root=tmp_path)
    with pytest.raises(FileExistsError):
        run_experiment(tiny_config, output_root=tmp_path)


def test_headless_code_never_imports_textual():
    code = (
        "import sys; import frozen_lake.experiments, frozen_lake.analysis; "
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- validation catches damage -----------------------------------------------------------------


def damage(path: Path, change) -> None:
    rows = read_csv(path)
    write_csv(path, change(rows))


def test_validation_detects_a_missing_episode(run_copy):
    damage(run_copy / "episode_metrics.csv", lambda rows: rows[:-1])
    assert validate_run(run_copy)


def test_validation_detects_a_duplicated_episode(run_copy):
    damage(run_copy / "episode_metrics.csv", lambda rows: [*rows, rows[0]])
    assert validate_run(run_copy)


def test_validation_detects_arms_that_did_not_share_an_episode_seed(run_copy):
    def change(rows):
        rows[0]["episode_seed"] = str(int(rows[0]["episode_seed"]) + 1)
        return rows

    damage(run_copy / "episode_metrics.csv", change)
    assert validate_run(run_copy)


def test_validation_detects_a_missing_decision(run_copy):
    damage(run_copy / "decision_metrics.csv", lambda rows: rows[1:])
    assert validate_run(run_copy)


def test_validation_detects_a_tampered_map(run_copy):
    path = run_copy / "maps.json"
    maps = json.loads(path.read_text(encoding="utf-8"))
    maps[0]["rows"][0] = "S" + "H" * (len(maps[0]["rows"][0]) - 1)
    path.write_text(json.dumps(maps), encoding="utf-8")
    assert validate_run(run_copy)


def test_validation_detects_a_missing_file(run_copy):
    (run_copy / "summary.csv").unlink()
    assert validate_run(run_copy)
