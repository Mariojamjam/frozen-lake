"""Real FrozenLake episodes: the only code that steps the real environment."""

from frozen_lake.environment import (
    MapDesc,
    create_environment,
    deterministic_destination,
    find_tile,
    tile_at,
)
from frozen_lake.models import Action, StateId, StepEvent, TerminationReason, Tile


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
        intended_state = deterministic_destination(self.desc, state, action)
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
