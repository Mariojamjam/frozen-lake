"""What the experiment consists of: its maps, and the arms that play on every map."""

import math
from dataclasses import dataclass

from frozen_lake.environment import MapDesc, find_tile, generate_map
from frozen_lake.experiments.config import ExperimentConfig
from frozen_lake.experiments.seeds import derive_seed
from frozen_lake.models import StrategyId, Tile
from frozen_lake.policies import BASE_STRATEGIES
from frozen_lake.policies.bfs import bfs_distances

# Estimators of the Rollout strategies: sampled trajectories, or exact values (N = infinity).
SAMPLED, EXACT = "sampled", "exact"


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    size: int
    index: int
    seed: int
    frozen_probability: float
    rows: MapDesc
    reachable: bool


def generate_maps(config: ExperimentConfig) -> list[MapRecord]:
    """Generate every map of the experiment, before any strategy runs."""
    records = []
    for size in config.environment.sizes:
        for index in range(config.experiment.maps_per_size):
            seed = derive_seed(config.experiment.master_seed, "map", size, index)
            rows = generate_map(size, config.environment.frozen_probability, seed=seed)
            start = find_tile(rows, Tile.START)
            records.append(
                MapRecord(
                    map_id=f"{size}-{index}",
                    size=size,
                    index=index,
                    seed=seed,
                    frozen_probability=config.environment.frozen_probability,
                    rows=rows,
                    reachable=bfs_distances(rows)[start] != math.inf,
                )
            )
    return records


@dataclass(frozen=True)
class Arm:
    """One way of choosing actions: a strategy, and for Rollout its estimator and sample count."""

    strategy: StrategyId
    estimator: str | None = None
    rollout_n: int | None = None

    @property
    def label(self) -> str:
        if self.estimator is None:
            return self.strategy.value
        if self.estimator == EXACT:
            return f"{self.strategy} exact"
        return f"{self.strategy} N={self.rollout_n}"


def arms(config: ExperimentConfig) -> list[Arm]:
    """Every arm, in a fixed order. Base-only strategies do not depend on N: one arm each."""
    result = [Arm(StrategyId.BFS), Arm(StrategyId.EXP)]
    for n in config.rollout.samples_per_action:
        result += [Arm(StrategyId.BFS_RO, SAMPLED, n), Arm(StrategyId.EXP_RO, SAMPLED, n)]
    if config.rollout.exact_reference:
        result += [Arm(StrategyId.BFS_RO, EXACT), Arm(StrategyId.EXP_RO, EXACT)]
    return result


@dataclass(frozen=True)
class Workload:
    maps: int
    arms: int
    episodes: int
    policy_builds: int


def workload(config: ExperimentConfig) -> Workload:
    n_maps = len(config.environment.sizes) * config.experiment.maps_per_size
    n_arms = len(arms(config))
    builds = (
        n_maps
        * len(BASE_STRATEGIES)
        * (config.policies.build_repeats + config.policies.build_warmup)
    )
    return Workload(
        maps=n_maps,
        arms=n_arms,
        episodes=n_maps * config.experiment.episodes_per_map * n_arms,
        policy_builds=builds,
    )
