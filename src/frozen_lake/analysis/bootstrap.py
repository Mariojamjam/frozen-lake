"""Success rate, amortized time and paired differences, with cluster-bootstrap intervals.

This follows the analysis plan of the design doc (§7 and §11):

- Everything is estimated per map first and then averaged over the maps of a size, so each map
  counts once however many episodes it played.
- Uncertainty comes from a cluster bootstrap: resample maps with replacement, then resample
  episodes within each selected map, recompute the estimate, repeat, and report percentile
  intervals. One resample is applied to every arm at once, which keeps the pairing between arms
  (they share maps and episode seeds) and so also gives intervals for paired differences.
- The amortized time of a map is its median policy build time divided by R, plus the mean agent
  time of its episodes (§7.3). R defaults to E, the episodes each strategy plays on a map.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from frozen_lake.analysis.loading import BASES, SizeData
from frozen_lake.experiments import EXACT, Arm
from frozen_lake.models import StrategyId

LEVEL = 0.95  # coverage of the percentile intervals
CHUNK = 250  # resamples drawn at once, to bound memory


@dataclass(frozen=True)
class Interval:
    """A point estimate with the percentile interval of its bootstrap distribution."""

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedDifference:
    """EXP-RO minus BFS-RO at one sample count (None: the exact reference).

    `amortized_ms` is None for the exact reference, whose time is not a cost to compare.
    """

    rollout_n: int | None
    success: Interval
    amortized_ms: Interval | None


@dataclass(frozen=True)
class SizeResults:
    """The estimates of every arm at one map size. Tuples follow the order of `arms`."""

    size: int
    maps: int
    episodes: int  # per map and arm
    amortization_r: int
    arms: tuple[Arm, ...]
    success: tuple[Interval, ...]
    amortized_ms: tuple[Interval | None, ...]  # None for the exact reference
    agent_ms: tuple[float | None, ...]  # mean agent time per episode
    build_ms: tuple[float, ...]  # mean over maps of the per-map median build time
    transitions: tuple[float, ...]  # mean simulated transitions per episode
    frontier: frozenset[int]  # arms on the observed Pareto frontier
    differences: tuple[PairedDifference, ...]


def pareto_frontier(points: Sequence[tuple[float, float]]) -> frozenset[int]:
    """Indices of the (time, success) points that no other point dominates (§7.5).

    A point dominates another if it is at least as fast and at least as successful, and
    strictly better on one of the two.
    """

    def dominates(x: tuple[float, float], y: tuple[float, float]) -> bool:
        return x[0] <= y[0] and x[1] >= y[1] and (x[0] < y[0] or x[1] > y[1])

    return frozenset(
        i
        for i, point in enumerate(points)
        if not any(dominates(other, point) for j, other in enumerate(points) if j != i)
    )


def _build_per_arm(data: SizeData) -> np.ndarray:
    """Median build time of each map, for the base policy each arm uses: [arm, map]."""
    base_of_arm = np.array([BASES.index(arm.strategy.base) for arm in data.arms])
    return np.median(data.build_ms, axis=2)[base_of_arm]


def _pairs(data: SizeData) -> list[tuple[int | None, int, int]]:
    """(sample count, BFS-RO arm, EXP-RO arm) for every Rollout arm; None is the exact one."""
    position = {arm: i for i, arm in enumerate(data.arms)}
    return [
        (arm.rollout_n, i, position[replace(arm, strategy=StrategyId.EXP_RO)])
        for i, arm in enumerate(data.arms)
        if arm.strategy is StrategyId.BFS_RO
    ]


def _estimates(
    data: SizeData,
    build_arm: np.ndarray,
    r: int,
    pairs: list[tuple[int | None, int, int]],
    map_idx: np.ndarray,
    episode_idx: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Estimates for K resamples; map_idx is (K, M) and episode_idx is (K, M, E).

    Returns success and amortized time as (arm, K), the mean agent time as (arm, K), and the
    paired differences of success and time as (pair, K).
    """
    index = map_idx[:, :, None]
    success_map = data.success[:, index, episode_idx].mean(axis=3)  # (arm, K, map)
    agent_map = data.agent_ms[:, index, episode_idx].mean(axis=3)
    time_map = build_arm[:, map_idx] / r + agent_map
    k = map_idx.shape[0]
    d_success = np.empty((len(pairs), k))
    d_time = np.empty((len(pairs), k))
    for p, (_, bfs, exp) in enumerate(pairs):
        d_success[p] = (success_map[exp] - success_map[bfs]).mean(axis=1)
        d_time[p] = (time_map[exp] - time_map[bfs]).mean(axis=1)
    return (
        success_map.mean(axis=2),
        time_map.mean(axis=2),
        agent_map.mean(axis=2),
        d_success,
        d_time,
    )


def _interval(estimate: float, samples: np.ndarray, level: float) -> Interval:
    low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return Interval(float(estimate), float(low), float(high))


def analyze_size(
    data: SizeData,
    *,
    resamples: int,
    seed: int,
    amortization_r: int | None = None,
    level: float = LEVEL,
) -> SizeResults:
    """Estimate every arm of one map size, with cluster-bootstrap intervals."""
    n_maps, n_episodes = data.success.shape[1:]
    r = amortization_r or n_episodes
    rng = np.random.default_rng(seed)
    build_arm = _build_per_arm(data)
    pairs = _pairs(data)

    # The point estimate is the same computation on the data as it is, without resampling.
    identity_maps = np.arange(n_maps)[None, :]
    identity_episodes = np.broadcast_to(np.arange(n_episodes), (1, n_maps, n_episodes))
    point = _estimates(data, build_arm, r, pairs, identity_maps, identity_episodes)

    drawn: list[tuple[np.ndarray, ...]] = []
    for start in range(0, resamples, CHUNK):
        k = min(CHUNK, resamples - start)
        drawn.append(
            _estimates(
                data,
                build_arm,
                r,
                pairs,
                rng.integers(0, n_maps, size=(k, n_maps)),
                rng.integers(0, n_episodes, size=(k, n_maps, n_episodes)),
            )
        )
    boot = [np.concatenate([chunk[i] for chunk in drawn], axis=1) for i in range(5)]

    exact = [arm.estimator == EXACT for arm in data.arms]
    success = tuple(_interval(point[0][a, 0], boot[0][a], level) for a in range(len(data.arms)))
    amortized = tuple(
        None if exact[a] else _interval(point[1][a, 0], boot[1][a], level)
        for a in range(len(data.arms))
    )
    # Exact arms are left out of the frontier: their cost is not comparable (§5.7).
    compared = [a for a in range(len(data.arms)) if not exact[a]]
    on_frontier = pareto_frontier(
        [(point[1][a, 0], point[0][a, 0]) for a in compared]  # (time, success)
    )
    differences = tuple(
        PairedDifference(
            rollout_n=n,
            success=_interval(point[3][p, 0], boot[3][p], level),
            amortized_ms=None if n is None else _interval(point[4][p, 0], boot[4][p], level),
        )
        for p, (n, _, _) in enumerate(pairs)
    )
    return SizeResults(
        size=data.size,
        maps=n_maps,
        episodes=n_episodes,
        amortization_r=r,
        arms=data.arms,
        success=success,
        amortized_ms=amortized,
        agent_ms=tuple(None if exact[a] else float(point[2][a, 0]) for a in range(len(data.arms))),
        build_ms=tuple(float(build_arm[a].mean()) for a in range(len(data.arms))),
        transitions=tuple(float(data.transitions[a].mean()) for a in range(len(data.arms))),
        frontier=frozenset(compared[i] for i in on_frontier),
        differences=differences,
    )


def amortization_sensitivity(data: SizeData, r_values: Sequence[int]) -> dict[int, list[float]]:
    """How the EXP-RO minus BFS-RO amortized time changes with R, per sample count (§7.3).

    Time per episode and build time are recorded separately, so any R can be applied
    afterwards. These are point estimates, without intervals.
    """
    build_arm = _build_per_arm(data)
    agent = data.agent_ms.mean(axis=2)  # (arm, map)
    curves: dict[int, list[float]] = {}
    for n, bfs, exp in _pairs(data):
        if n is None:
            continue
        agent_gap = (agent[exp] - agent[bfs]).mean()
        build_gap = (build_arm[exp] - build_arm[bfs]).mean()
        curves[n] = [float(agent_gap + build_gap / r) for r in r_values]
    return curves
