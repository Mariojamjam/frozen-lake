"""Shared types for the FrozenLake decision problem: actions, states, transitions, and results."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

type StateId = int


class Tile(StrEnum):
    START = "S"
    FROZEN = "F"
    HOLE = "H"
    GOAL = "G"


class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


# Fixed order used to break ties between equally good actions, for BFS and for Rollout.
# Frozen study setting: changing it changes which actions the policies choose.
DEFAULT_ACTION_PRIORITY: tuple[Action, ...] = (
    Action.RIGHT,
    Action.DOWN,
    Action.LEFT,
    Action.UP,
)


@dataclass(frozen=True, slots=True)
class Transition:
    probability: float
    next_state: StateId
    reward: float
    terminated: bool


# model[state][action] -> every possible outcome of choosing `action` in `state`.
type TransitionModel = tuple[tuple[tuple[Transition, ...], ...], ...]

# policy[state] -> action to take there; None only on terminal tiles (hole or goal).
type Policy = tuple[Action | None, ...]


class StrategyId(StrEnum):
    BFS = "BFS"
    EXP = "EXP"
    BFS_RO = "BFS-RO"
    EXP_RO = "EXP-RO"

    @property
    def uses_rollout(self) -> bool:
        return self in (StrategyId.BFS_RO, StrategyId.EXP_RO)

    @property
    def base(self) -> "StrategyId":
        """The base policy this strategy follows or improves: BFS or EXP."""
        return StrategyId.BFS if self in (StrategyId.BFS, StrategyId.BFS_RO) else StrategyId.EXP


class TerminationReason(StrEnum):
    GOAL = "goal"
    HOLE = "hole"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class StepEvent:
    """What happened in one real environment step."""

    step: int  # number of real steps taken so far, including this one
    state: StateId
    action: Action
    # Where the action would have led without slipping. At a border a slip can land on the
    # same tile, so equal states do not prove that no slip happened.
    intended_state: StateId
    next_state: StateId
    reward: float
    termination: TerminationReason | None  # None while the episode continues

    @property
    def moved_as_intended(self) -> bool:
        return self.next_state == self.intended_state


@dataclass(frozen=True, slots=True)
class PolicyBuildResult:
    policy: Policy
    build_time_ns: int
    states_evaluated: int
    successors_examined: int
    # Algorithm-specific counters (e.g. BFS peak queue, Expectimax cache hits). Their units
    # differ between planners, so they are kept by name instead of forced into shared fields.
    diagnostics: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    state: StateId
    base_action: Action
    action: Action
    # Monte Carlo estimated returns per candidate action; None for base-only strategies.
    # These are sampled estimates, not exact success probabilities.
    estimated_action_values: Mapping[Action, float] | None
    compute_time_ns: int
    simulated_transitions: int
    trajectories_evaluated: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    strategy: StrategyId
    termination_reason: TerminationReason
    steps: int
    total_reward: float
    agent_compute_ns: int
    wall_ns: int
    simulated_transitions: int
    decisions: tuple[DecisionResult, ...]

    @property
    def success(self) -> bool:
        return self.termination_reason is TerminationReason.GOAL
