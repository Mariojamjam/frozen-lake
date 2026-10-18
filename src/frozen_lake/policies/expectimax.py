"""Depth-limited Expectimax base policy: plan over the slip probabilities, a few moves ahead.

MAX nodes pick the agent's best intended action; CHANCE nodes average that action's outcomes,
weighted by their probabilities. Where the look-ahead runs out on a nonterminal tile, the
tile is scored by the BFS-distance heuristic beta ** d_BFS (design doc, section 4.2). Values
are memoized by (state, remaining depth), and the best action is computed for every
nonterminal tile before any episode, so episodes only look actions up.
"""

import math
import time

from frozen_lake.environment import MapDesc, map_size, tile_at
from frozen_lake.models import (
    DEFAULT_ACTION_PRIORITY,
    Action,
    PolicyBuildResult,
    StateId,
    Tile,
    TransitionModel,
)
from frozen_lake.policies.bfs import bfs_distances
from frozen_lake.policies.utils import fallback_action

# Expected values computed along different paths can differ by rounding error alone. Actions
# whose values are this close are treated as tied, and the action priority decides.
TIE_TOLERANCE = 1e-12


def bfs_heuristic(distance: float, beta: float) -> float:
    """Progress score of a nonterminal tile at BFS `distance` from the goal: beta ** distance.

    Closer tiles score higher. A tile with no route to the goal (infinite distance) scores 0;
    the infinity is handled explicitly rather than computing beta ** inf. This is a score of
    progress, not a probability of reaching the goal.
    """
    return 0.0 if distance == math.inf else beta**distance


def validate_parameters(depth: int, beta: float, gamma: float) -> None:
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")


class ExpectimaxSearch:
    """Memoized MAX/CHANCE evaluation on one map, with the counters the study records.

    Counting conventions: a subproblem is one (state, remaining depth) value actually computed;
    an outcome is one probability-weighted transition record a CHANCE node processed; a cache
    hit is a value reused instead of recomputed; recursion depth counts active MAX nodes.
    """

    def __init__(
        self,
        model: TransitionModel,
        distances: tuple[float, ...],
        *,
        beta: float,
        gamma: float,
    ) -> None:
        self.model = model
        self.distances = distances
        self.beta = beta
        self.gamma = gamma
        # The remaining depth is part of the key: V_3(s) and V_5(s) generally differ.
        self.cache: dict[tuple[StateId, int], float] = {}
        self.subproblems_computed = 0
        self.outcomes_examined = 0
        self.cache_hits = 0
        self.peak_recursion = 0
        self._active = 0

    def value(self, state: StateId, depth: int) -> float:
        """MAX node: V_d(s), the best expected score of a nonterminal state with `depth` left."""
        key = (state, depth)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self._active += 1
        self.peak_recursion = max(self.peak_recursion, self._active)
        if depth == 0:
            result = bfs_heuristic(self.distances[state], self.beta)
        else:
            result = max(self.action_value(state, action, depth) for action in Action)
        self._active -= 1
        self.cache[key] = result
        self.subproblems_computed += 1
        return result

    def action_value(self, state: StateId, action: Action, depth: int) -> float:
        """CHANCE node: Q_d(s, a), the probability-weighted value of choosing `action`.

        An outcome that ends the episode contributes only its reward: the goal pays 1 once, on
        entry, and neither a hole nor the goal has any continuation value.
        """
        total = 0.0
        for outcome in self.model[state][action]:
            self.outcomes_examined += 1
            continuation = 0.0
            if not outcome.terminated:
                continuation = self.gamma * self.value(outcome.next_state, depth - 1)
            total += outcome.probability * (outcome.reward + continuation)
        return total


def expectimax_action_values(
    desc: MapDesc,
    model: TransitionModel,
    state: StateId,
    *,
    depth: int,
    beta: float,
    gamma: float,
) -> dict[Action, float]:
    """Q_depth(state, a) for every action: the values the policy chooses between."""
    validate_parameters(depth, beta, gamma)
    search = ExpectimaxSearch(model, bfs_distances(desc), beta=beta, gamma=gamma)
    return {action: search.action_value(state, action, depth) for action in Action}


def build_expectimax_policy(
    desc: MapDesc,
    model: TransitionModel,
    *,
    depth: int,
    beta: float,
    gamma: float,
    action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY,
) -> PolicyBuildResult:
    """Build the complete Expectimax policy table for `desc`, timing the whole build.

    Every nonterminal tile intends the action with the highest Q_depth; actions within
    TIE_TOLERANCE of the best are tied and the first one in `action_priority` wins. When every
    action is worth zero (no route to the goal within reach of the heuristic), the BFS safe
    fallback is used, so the tile does not walk into a hole merely because a hole is also worth
    zero. Terminal tiles get None.

    The timed build includes the BFS distances the heuristic needs: that preprocessing is part
    of what this strategy costs. Counters follow ExpectimaxSearch and include the root
    evaluations of every tile.
    """
    validate_parameters(depth, beta, gamma)
    if sorted(action_priority) != sorted(Action):
        raise ValueError("action_priority must list every action exactly once")

    start_ns = time.perf_counter_ns()
    n = map_size(desc)
    search = ExpectimaxSearch(model, bfs_distances(desc), beta=beta, gamma=gamma)
    policy: list[Action | None] = []
    fallback_states = 0
    for state in range(n * n):
        if tile_at(desc, state) in (Tile.HOLE, Tile.GOAL):
            policy.append(None)
            continue
        values = {action: search.action_value(state, action, depth) for action in Action}
        best = max(values.values())
        if best <= TIE_TOLERANCE:
            policy.append(fallback_action(desc, state, n, action_priority))
            fallback_states += 1
            continue
        policy.append(next(a for a in action_priority if values[a] >= best - TIE_TOLERANCE))
    build_time_ns = time.perf_counter_ns() - start_ns

    return PolicyBuildResult(
        policy=tuple(policy),
        build_time_ns=build_time_ns,
        states_evaluated=search.subproblems_computed,
        successors_examined=search.outcomes_examined,
        diagnostics={
            "cache_hits": search.cache_hits,
            "cache_entries": len(search.cache),
            "peak_recursion": search.peak_recursion,
            "fallback_states": fallback_states,
        },
    )
