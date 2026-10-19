"""Reading a results folder into the arrays the analysis works on.

The loader expects a folder that already passed `validate_run`; it never repairs data. A value
that is missing would show up as NaN rather than as a silent zero.
"""

import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frozen_lake.experiments import SAMPLED, Arm, ExperimentConfig, arms, parse_config
from frozen_lake.models import StrategyId

# The two base policies, in the order used for the `build_ms` axis below.
BASES = (StrategyId.BFS, StrategyId.EXP)


@dataclass(frozen=True)
class SizeData:
    """Every measurement of one map size, as arrays indexed [arm, map, episode].

    All arms play the same episode seeds on the same maps, so index [:, m, e] is one paired
    episode across arms.
    """

    size: int
    map_ids: tuple[str, ...]
    arms: tuple[Arm, ...]
    success: np.ndarray  # 1.0 if the goal was reached, else 0.0
    agent_ms: np.ndarray  # agent computation time of the episode
    transitions: np.ndarray  # simulated transitions of the episode
    build_ms: np.ndarray  # [base, map, repeat]: timed policy builds, in BASES order


@dataclass(frozen=True)
class BuildRow:
    """One timed policy build with the search counters it reported (None: not applicable)."""

    map_id: str
    size: int
    base: str
    repeat: int
    build_ms: float
    states_evaluated: int
    successors_examined: int
    peak_queue: int | None
    peak_recursion: int | None
    cache_hits: int | None
    cache_entries: int | None


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    config: ExperimentConfig
    metadata: dict
    sizes: dict[int, SizeData]
    builds: list[BuildRow]


@dataclass(frozen=True)
class DecisionData:
    """Decision times of a run, read from `decision_metrics.csv`.

    Keys use the strategy, estimator and sample count exactly as written in the file
    (blank strings where they do not apply).
    """

    times_ms: dict[tuple[int, str, str, str], list[float]]  # (size, strategy, estimator, n)
    by_state: dict[tuple[str, str, str, str], dict[int, list[float]]]  # (map_id, strategy, ...)


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def _optional_int(value: str) -> int | None:
    return int(float(value)) if value != "" else None


def load_run(run_dir: str | Path) -> RunData:
    """Read the configuration, the episodes and the policy builds of a results folder."""
    run_dir = Path(run_dir)
    config = parse_config(json.loads((run_dir / "configuration.json").read_text()))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    maps = json.loads((run_dir / "maps.json").read_text())

    all_arms = tuple(arms(config))
    arm_position = {(a.strategy.value, a.estimator or "", str(a.rollout_n or "")): i
                    for i, a in enumerate(all_arms)}  # fmt: skip
    n_episodes = config.experiment.episodes_per_map
    n_repeats = config.policies.build_repeats

    map_ids: dict[int, list[str]] = defaultdict(list)
    for record in maps:
        map_ids[record["size"]].append(record["map_id"])
    position = {map_id: i for ids in map_ids.values() for i, map_id in enumerate(ids)}

    def empty(*shape: int) -> np.ndarray:
        return np.full(shape, np.nan)

    shape = {size: (len(ids), n_episodes) for size, ids in map_ids.items()}
    success = {s: empty(len(all_arms), *shape[s]) for s in map_ids}
    agent_ms = {s: empty(len(all_arms), *shape[s]) for s in map_ids}
    transitions = {s: empty(len(all_arms), *shape[s]) for s in map_ids}
    build_ms = {s: empty(len(BASES), len(map_ids[s]), n_repeats) for s in map_ids}

    for row in _rows(run_dir / "episode_metrics.csv"):
        size = int(row["size"])
        index = (
            arm_position[(row["strategy"], row["estimator"], row["rollout_n"])],
            position[row["map_id"]],
            int(row["episode_index"]),
        )
        success[size][index] = float(row["success"])
        agent_ms[size][index] = int(row["agent_compute_ns"]) / 1e6
        transitions[size][index] = int(row["simulated_transitions"])

    builds = []
    for row in _rows(run_dir / "policy_metrics.csv"):
        build = BuildRow(
            map_id=row["map_id"],
            size=int(row["size"]),
            base=row["base_policy"],
            repeat=int(row["build_repeat"]),
            build_ms=int(row["build_time_ns"]) / 1e6,
            states_evaluated=int(row["states_evaluated"]),
            successors_examined=int(row["successors_examined"]),
            peak_queue=_optional_int(row["peak_queue"]),
            peak_recursion=_optional_int(row["peak_recursion"]),
            cache_hits=_optional_int(row["cache_hits"]),
            cache_entries=_optional_int(row["cache_entries"]),
        )
        builds.append(build)
        base = BASES.index(StrategyId(build.base))
        build_ms[build.size][base, position[build.map_id], build.repeat] = build.build_ms

    sizes = {
        size: SizeData(
            size=size,
            map_ids=tuple(map_ids[size]),
            arms=all_arms,
            success=success[size],
            agent_ms=agent_ms[size],
            transitions=transitions[size],
            build_ms=build_ms[size],
        )
        for size in sorted(map_ids)
    }
    return RunData(run_dir, config, metadata, sizes, builds)


def load_decisions(run_dir: str | Path) -> DecisionData | None:
    """Read `decision_metrics.csv`, or return None if the run did not write it."""
    path = Path(run_dir) / "decision_metrics.csv"
    if not path.exists():
        return None
    times: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    by_state: dict[tuple[str, str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _rows(path):
        arm = (row["strategy"], row["estimator"], row["rollout_n"])
        milliseconds = int(row["decision_time_ns"]) / 1e6
        times[(int(row["size"]), *arm)].append(milliseconds)
        if row["estimator"] == SAMPLED:
            by_state[(row["map_id"], *arm)][int(row["state"])].append(milliseconds)
    return DecisionData(dict(times), {key: dict(states) for key, states in by_state.items()})
