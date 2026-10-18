"""Checking a written results folder for completeness and consistency."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from frozen_lake.experiments.config import parse_config
from frozen_lake.experiments.matrix import arms, generate_maps
from frozen_lake.policies import BASE_STRATEGIES


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_run(run_dir: str | Path) -> list[str]:
    """Check a results folder for completeness and consistency; return the problems found."""
    run_dir = Path(run_dir)
    problems: list[str] = []
    required = ["configuration.json", "maps.json", "metadata.json", "policy_metrics.csv",
                "episode_metrics.csv", "summary.csv"]  # fmt: skip
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return [f"missing files: {', '.join(missing)}"]

    config = parse_config(json.loads((run_dir / "configuration.json").read_text()))
    maps = json.loads((run_dir / "maps.json").read_text())
    expected_maps = {record.map_id: record for record in generate_maps(config)}
    if [m["map_id"] for m in maps] != list(expected_maps):
        problems.append("maps.json does not list the expected maps")
    for saved in maps:
        record = expected_maps.get(saved["map_id"])
        if record is not None and tuple(saved["rows"]) != record.rows:
            problems.append(f"map {saved['map_id']} does not match its seed")

    builds = _read_csv(run_dir / "policy_metrics.csv")
    expected_builds = len(expected_maps) * len(BASE_STRATEGIES) * config.policies.build_repeats
    if len(builds) != expected_builds:
        problems.append(f"policy_metrics.csv has {len(builds)} rows, expected {expected_builds}")

    episodes = _read_csv(run_dir / "episode_metrics.csv")
    expected_keys = {
        (map_id, arm.strategy.value, arm.estimator or "", str(arm.rollout_n or ""), str(e))
        for map_id in expected_maps
        for arm in arms(config)
        for e in range(config.experiment.episodes_per_map)
    }
    keys = [
        (r["map_id"], r["strategy"], r["estimator"], r["rollout_n"], r["episode_index"])
        for r in episodes
    ]
    if len(keys) != len(set(keys)):
        problems.append("episode_metrics.csv has duplicate episodes")
    if set(keys) != expected_keys:
        absent = len(expected_keys - set(keys))
        extra = len(set(keys) - expected_keys)
        problems.append(
            f"episode_metrics.csv: {absent} expected episodes missing, {extra} unexpected"
        )
    for row in episodes:
        if row["success"] not in ("0", "1"):
            problems.append(f"invalid success value {row['success']!r}")
            break
        if int(row["agent_compute_ns"]) < 0 or int(row["wall_ns"]) <= 0:
            problems.append("negative or zero times in episode_metrics.csv")
            break
    seeds = defaultdict(set)
    for row in episodes:
        seeds[(row["map_id"], row["episode_index"])].add(row["episode_seed"])
    if any(len(values) != 1 for values in seeds.values()):
        problems.append("arms of one episode did not share the same episode seed")

    decisions_path = run_dir / "decision_metrics.csv"
    if config.experiment.write_decisions:
        if not decisions_path.exists():
            problems.append("decision_metrics.csv is missing")
        else:
            counts = defaultdict(int)
            for row in _read_csv(decisions_path):
                counts[(row["map_id"], row["strategy"], row["estimator"], row["rollout_n"],
                        row["episode_index"])] += 1  # fmt: skip
            for row, key in zip(episodes, keys, strict=True):
                if counts.get(key, 0) != int(row["steps"]):
                    problems.append(
                        f"episode {key} has {counts.get(key, 0)} decisions for {row['steps']} steps"
                    )
                    break
    return problems
