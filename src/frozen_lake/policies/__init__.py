"""Base policies: complete state-to-action tables built once per map."""

from frozen_lake.environment import MapDesc
from frozen_lake.models import PolicyBuildResult, StrategyId, TransitionModel
from frozen_lake.policies.bfs import build_bfs_policy
from frozen_lake.policies.expectimax import build_expectimax_policy

# Strategies whose decisions are table lookups in a precomputed base policy.
BASE_STRATEGIES = (StrategyId.BFS, StrategyId.EXP)


def build_base_policy(
    strategy: StrategyId,
    desc: MapDesc,
    model: TransitionModel,
    *,
    depth: int,
    beta: float,
    gamma: float,
) -> PolicyBuildResult:
    """Build the base policy a strategy uses; Expectimax settings are ignored by BFS."""
    if strategy is StrategyId.BFS:
        return build_bfs_policy(desc)
    if strategy is StrategyId.EXP:
        return build_expectimax_policy(desc, model, depth=depth, beta=beta, gamma=gamma)
    raise ValueError(f"{strategy} does not use a precomputed base policy")
