"""Monte Carlo Rollout: improve any base policy by simulating what follows each first action.

Before each real step, every candidate action is tried in N simulated trajectories: the
candidate first, then the base policy at every simulated tile, until a hole, the goal or the
horizon. The action whose trajectories did best on average is taken (design doc, section 5).
This module never builds or re-plans a policy: it only looks actions up in the table it is
given, so the same code serves BFS and Expectimax alike.
"""

import random
import time

from frozen_lake.environment import sample_transition
from frozen_lake.episodes import Decide
from frozen_lake.models import (
    DEFAULT_ACTION_PRIORITY,
    Action,
    DecisionResult,
    Policy,
    StateId,
    TransitionModel,
)

# Estimates that differ by less than this are tied. With 0/1 returns they are multiples of 1/N,
# so real differences are far larger; this only absorbs rounding.
TIE_TOLERANCE = 1e-12


def rollout_rng(episode_seed: int) -> random.Random:
    """Simulation random stream for an episode, derived from its seed but separate from it.

    Seeding with a string is stable across runs and processes (it goes through SHA-512), so the
    same episode always simulates the same trajectories, without touching the real episode's
    random stream.
    """
    return random.Random(f"rollout:{episode_seed}")


def simulate_trajectory(
    model: TransitionModel,
    base_policy: Policy,
    state: StateId,
    first_action: Action,
    *,
    horizon: int,
    gamma: float,
    rng: random.Random,
) -> tuple[float, int]:
    """Simulate one trajectory: `first_action` now, then the base policy, for at most `horizon`.

    Returns the (discounted) return and the number of transitions simulated. The trajectory
    stops at the first transition that ends the episode, so it never asks the policy for an
    action on a hole or the goal.
    """
    total = 0.0
    discount = 1.0
    current, action = state, first_action
    steps = 0
    while steps < horizon:
        outcome = sample_transition(model, current, action, rng)
        steps += 1
        total += discount * outcome.reward
        if outcome.terminated:
            break
        discount *= gamma
        current = outcome.next_state
        action = base_policy[current]
    return total, steps


def rollout_decider(
    base_policy: Policy,
    model: TransitionModel,
    *,
    samples: int,
    horizon: int,
    gamma: float,
    rng: random.Random,
    action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY,
) -> Decide:
    """Decide by Monte Carlo Rollout over `base_policy` (the BFS-RO and EXP-RO strategies).

    Each candidate action gets `samples` trajectories. Simulations are capped by the horizon
    and by the real steps the episode has left. The highest estimate wins; if the base policy's
    action is among the best, it is kept, otherwise the first best action in `action_priority`.
    `rng` must be a stream of its own, never the real environment's.
    """
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    def decide(state: StateId, remaining_steps: int) -> DecisionResult:
        base_action = base_policy[state]
        if base_action is None:
            raise ValueError(f"policy has no action for state {state}, a terminal tile")
        if remaining_steps < 1:
            raise ValueError("no real steps left to decide for")
        effective_horizon = min(horizon, remaining_steps)

        start_ns = time.perf_counter_ns()
        estimates: dict[Action, float] = {}
        simulated = 0
        for candidate in Action:
            total = 0.0
            for _ in range(samples):
                trajectory_return, steps = simulate_trajectory(
                    model,
                    base_policy,
                    state,
                    candidate,
                    horizon=effective_horizon,
                    gamma=gamma,
                    rng=rng,
                )
                total += trajectory_return
                simulated += steps
            estimates[candidate] = total / samples

        best = max(estimates.values())
        if estimates[base_action] >= best - TIE_TOLERANCE:
            action = base_action  # with no evidence against it, keep the base policy's choice
        else:
            action = next(a for a in action_priority if estimates[a] >= best - TIE_TOLERANCE)
        compute_time_ns = time.perf_counter_ns() - start_ns

        return DecisionResult(
            state=state,
            base_action=base_action,
            action=action,
            estimated_action_values=estimates,
            compute_time_ns=compute_time_ns,
            simulated_transitions=simulated,
            trajectories_evaluated=len(Action) * samples,
        )

    return decide
