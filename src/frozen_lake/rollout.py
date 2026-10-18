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


def choose_action(
    estimates: dict[Action, float], base_action: Action, action_priority: tuple[Action, ...]
) -> Action:
    """The action with the highest value; the base policy's action wins any tie it is part of.

    With no evidence against it, the base policy's choice is kept; otherwise the first best
    action in `action_priority` is taken.
    """
    best = max(estimates.values())
    if estimates[base_action] >= best - TIE_TOLERANCE:
        return base_action
    return next(a for a in action_priority if estimates[a] >= best - TIE_TOLERANCE)


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

        action = choose_action(estimates, base_action, action_priority)
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


def exact_values(
    model: TransitionModel, base_policy: Policy, *, horizon: int, gamma: float
) -> list[list[float]]:
    """V[h][s]: the exact expected return of following `base_policy` for h steps from s.

    Computed by dynamic programming over the known model, for every h up to `horizon`, once:
    the base policy is fixed, so the table serves every decision on the map. With rewards of 1
    for the goal only and gamma = 1, V[h][s] is the probability of reaching the goal within h
    steps. Terminal tiles have value 0: there is nothing left to collect from them.
    """
    n_states = len(model)
    values = [[0.0] * n_states]
    for _ in range(horizon):
        previous = values[-1]
        current = [0.0] * n_states
        for state in range(n_states):
            action = base_policy[state]
            if action is None:
                continue
            current[state] = sum(
                t.probability
                * (t.reward + (0.0 if t.terminated else gamma * previous[t.next_state]))
                for t in model[state][action]
            )
        values.append(current)
    return values


def exact_rollout_decider(
    base_policy: Policy,
    model: TransitionModel,
    *,
    horizon: int,
    gamma: float,
    action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY,
) -> Decide:
    """Rollout with exact action values instead of sampled estimates: the N = infinity limit.

    Each candidate action's value is computed exactly (take the action, then follow the base
    policy) instead of being estimated from simulated trajectories, with the same horizon cap
    and the same tie rule as rollout_decider. It samples nothing, so it shows how well Rollout
    over this base policy can do once sampling noise is gone. Its time is not comparable with
    the sampled strategies: the value table is computed once, when the decider is created.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    values = exact_values(model, base_policy, horizon=horizon, gamma=gamma)

    def decide(state: StateId, remaining_steps: int) -> DecisionResult:
        base_action = base_policy[state]
        if base_action is None:
            raise ValueError(f"policy has no action for state {state}, a terminal tile")
        if remaining_steps < 1:
            raise ValueError("no real steps left to decide for")
        steps = min(horizon, remaining_steps)

        start_ns = time.perf_counter_ns()
        future = values[steps - 1]
        estimates = {
            candidate: sum(
                t.probability * (t.reward + (0.0 if t.terminated else gamma * future[t.next_state]))
                for t in model[state][candidate]
            )
            for candidate in Action
        }
        action = choose_action(estimates, base_action, action_priority)
        compute_time_ns = time.perf_counter_ns() - start_ns

        return DecisionResult(
            state=state,
            base_action=base_action,
            action=action,
            estimated_action_values=estimates,
            compute_time_ns=compute_time_ns,
            simulated_transitions=0,
            trajectories_evaluated=0,
        )

    return decide
