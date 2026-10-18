"""Real FrozenLake episodes: the only code that steps the real environment."""

import time
from collections.abc import Callable

from frozen_lake.environment import (
    MapDesc,
    create_environment,
    deterministic_move,
    find_tile,
    map_size,
    tile_at,
)
from frozen_lake.models import (
    Action,
    DecisionResult,
    EpisodeResult,
    Policy,
    StateId,
    StepEvent,
    StrategyId,
    TerminationReason,
    Tile,
)

# Chooses the real action for an observed state, given how many real steps the episode has left.
# Base policies and Rollout share this shape, so the episode loop never needs to know which one
# is deciding. Table policies ignore the remaining steps; Rollout caps its simulations by them.
type Decide = Callable[[StateId, int], DecisionResult]


class EpisodeFinishedError(RuntimeError):
    """Raised when stepping an episode that already reached the goal, a hole, or its step limit."""


def classify_end(
    desc: MapDesc, state: StateId, *, terminated: bool, truncated: bool
) -> TerminationReason | None:
    """Map Gymnasium's end flags to a termination reason, or None if the episode continues.

    Both flags are set when the goal is entered on the last allowed step. Termination wins:
    the goal was reached, so that episode is a success, not a timeout.
    """
    if terminated:
        return (
            TerminationReason.GOAL if tile_at(desc, state) == Tile.GOAL else TerminationReason.HOLE
        )
    if truncated:
        return TerminationReason.TIMEOUT
    return None


class EpisodeSession:
    """One real episode at a time on a fixed map, advanced by exactly one env.step per step()."""

    def __init__(self, desc: MapDesc, *, max_steps: int, success_rate: float) -> None:
        self.desc = desc
        self.max_steps = max_steps
        self.success_rate = success_rate
        self._env = create_environment(desc, max_steps=max_steps, success_rate=success_rate)
        self._seed: int | None = None
        self._state: StateId = find_tile(desc, Tile.START)
        self._steps = 0
        self._total_reward = 0.0
        self._termination: TerminationReason | None = None

    def reset(self, seed: int) -> StateId:
        """Start a new episode whose real slips are determined by `seed`."""
        state, _ = self._env.reset(seed=seed)
        self._seed = seed
        self._state = int(state)
        self._steps = 0
        self._total_reward = 0.0
        self._termination = None
        return self._state

    def step(self, action: Action) -> StepEvent:
        """Commit one intended action to the real environment and report what happened."""
        if self._seed is None:
            raise RuntimeError("call reset(seed) before step()")
        if self.finished:
            raise EpisodeFinishedError(f"episode already ended: {self._termination}")

        state = self._state
        intended_state = deterministic_move(state, action, map_size(self.desc))
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        next_state = int(next_state)

        self._state = next_state
        self._steps += 1
        self._total_reward += float(reward)
        self._termination = classify_end(
            self.desc, next_state, terminated=terminated, truncated=truncated
        )
        return StepEvent(
            step=self._steps,
            state=state,
            action=action,
            intended_state=intended_state,
            next_state=next_state,
            reward=float(reward),
            termination=self._termination,
        )

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def state(self) -> StateId:
        return self._state

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def termination(self) -> TerminationReason | None:
        return self._termination

    @property
    def finished(self) -> bool:
        return self._termination is not None

    def close(self) -> None:
        self._env.close()


def policy_decider(policy: Policy) -> Decide:
    """Decide by looking up a precomputed policy table (the BFS and EXP strategies)."""

    def decide(state: StateId, remaining_steps: int) -> DecisionResult:
        start_ns = time.perf_counter_ns()
        action = policy[state]
        compute_time_ns = time.perf_counter_ns() - start_ns
        if action is None:
            raise ValueError(f"policy has no action for state {state}, a terminal tile")
        return DecisionResult(
            state=state,
            base_action=action,
            action=action,  # without Rollout the base policy's action is the one taken
            estimated_action_values=None,
            compute_time_ns=compute_time_ns,
            simulated_transitions=0,
            trajectories_evaluated=0,
        )

    return decide


def run_episode(
    session: EpisodeSession,
    decide: Decide,
    *,
    seed: int,
    strategy: StrategyId,
    on_step: Callable[[DecisionResult, StepEvent], None] | None = None,
) -> EpisodeResult:
    """Play one whole real episode: decide, take one real step, repeat until it ends.

    `agent_compute_ns` adds up only the time spent deciding; `wall_ns` covers the whole
    episode, environment included. `on_step`, if given, sees every decision and its outcome
    (e.g. to print them); keep it out of anything being timed for research.
    """
    wall_start_ns = time.perf_counter_ns()
    state = session.reset(seed)
    decisions: list[DecisionResult] = []
    while not session.finished:
        decision = decide(state, session.max_steps - session.steps)
        event = session.step(decision.action)
        decisions.append(decision)
        if on_step is not None:
            on_step(decision, event)
        state = event.next_state
    wall_ns = time.perf_counter_ns() - wall_start_ns

    assert session.termination is not None  # the loop only exits once the episode has ended
    return EpisodeResult(
        strategy=strategy,
        termination_reason=session.termination,
        steps=session.steps,
        total_reward=session.total_reward,
        agent_compute_ns=sum(d.compute_time_ns for d in decisions),
        wall_ns=wall_ns,
        simulated_transitions=sum(d.simulated_transitions for d in decisions),
        decisions=tuple(decisions),
    )
