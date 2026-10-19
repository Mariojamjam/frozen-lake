"""Descriptive summaries of builds and decisions: medians and spread, not means (§11.3)."""

import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from frozen_lake.analysis.loading import BuildRow, DecisionData, RunData
from frozen_lake.experiments import SAMPLED
from frozen_lake.models import StrategyId

COUNTERS = (
    "states_evaluated",
    "successors_examined",
    "peak_queue",
    "peak_recursion",
    "cache_hits",
    "cache_entries",
)
_STRATEGY_ORDER = {s.value: i for i, s in enumerate(StrategyId)}


@dataclass(frozen=True)
class BuildSummary:
    """Timed builds of one base policy at one map size."""

    size: int
    base: str
    maps: int
    builds: int
    median_ms: float
    q1_ms: float
    q3_ms: float
    per_map_median_ms: tuple[float, ...]
    # Search counters, one value per map. They do not change between repeats of a build, so
    # the first repeat stands for the map. Counters a policy does not report are left out.
    counters: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class DecisionSummary:
    """Decision times of one arm at one map size."""

    size: int
    strategy: str
    rollout_n: int | None
    decisions: int
    median_ms: float
    q1_ms: float
    q3_ms: float


@dataclass(frozen=True)
class SharedStateTime:
    """EXP-RO decision time over BFS-RO, on states both visited (see `shared_state_times`)."""

    size: int
    rollout_n: int
    ratios: tuple[float, ...]  # one per map that has states in common

    @property
    def mean_ratio(self) -> float:
        return statistics.fmean(self.ratios)


def build_summaries(run: RunData) -> list[BuildSummary]:
    groups: dict[tuple[int, str], list[BuildRow]] = defaultdict(list)
    for row in run.builds:
        groups[(row.size, row.base)].append(row)

    summaries = []
    for (size, base), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] != "BFS")):
        times = np.array([row.build_ms for row in rows])
        by_map: dict[str, list[BuildRow]] = defaultdict(list)
        for row in rows:
            by_map[row.map_id].append(row)
        first = [min(group, key=lambda row: row.repeat) for group in by_map.values()]
        counters = {}
        for name in COUNTERS:
            values = [getattr(row, name) for row in first]
            if all(value is not None for value in values):
                counters[name] = tuple(float(value) for value in values)
        q1, q3 = np.percentile(times, [25, 75])
        summaries.append(
            BuildSummary(
                size=size,
                base=base,
                maps=len(by_map),
                builds=len(rows),
                median_ms=float(np.median(times)),
                q1_ms=float(q1),
                q3_ms=float(q3),
                per_map_median_ms=tuple(
                    statistics.median(row.build_ms for row in group) for group in by_map.values()
                ),
                counters=counters,
            )
        )
    return summaries


def decision_summaries(decisions: DecisionData) -> list[DecisionSummary]:
    """Median and interquartile range of the decision times of every arm but the exact one."""
    summaries = []
    for (size, strategy, estimator, n), times in decisions.times_ms.items():
        if estimator not in ("", SAMPLED):
            continue
        q1, q3 = np.percentile(times, [25, 75])
        summaries.append(
            DecisionSummary(
                size=size,
                strategy=strategy,
                rollout_n=int(n) if n else None,
                decisions=len(times),
                median_ms=float(np.median(times)),
                q1_ms=float(q1),
                q3_ms=float(q3),
            )
        )
    return sorted(
        summaries,
        key=lambda s: (s.size, _STRATEGY_ORDER[s.strategy], s.rollout_n or 0),
    )


def shared_state_times(run: RunData, decisions: DecisionData) -> list[SharedStateTime]:
    """Compare EXP-RO with BFS-RO decision time on the states both visited, per map and N.

    Episodes of different strategies end at different times, so decision time per episode mixes
    cost per decision with episode length. Looking only at states that both strategies decided
    in removes most of that. This is not the dedicated paired-state measurement of the design
    doc (§8.1), which times both strategies on a prespecified set of states: it reuses the
    decisions already logged, so the states are those the agents happened to visit.
    """
    results = []
    for size, data in run.sizes.items():
        for n in run.config.rollout.samples_per_action:
            ratios = []
            for map_id in data.map_ids:
                bfs = decisions.by_state.get((map_id, "BFS-RO", "sampled", str(n)), {})
                exp = decisions.by_state.get((map_id, "EXP-RO", "sampled", str(n)), {})
                shared = bfs.keys() & exp.keys()
                if shared:
                    bfs_ms = statistics.median(t for s in shared for t in bfs[s])
                    exp_ms = statistics.median(t for s in shared for t in exp[s])
                    ratios.append(exp_ms / bfs_ms)
            if ratios:
                results.append(SharedStateTime(size, n, tuple(ratios)))
    return results
