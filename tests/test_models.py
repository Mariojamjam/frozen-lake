from dataclasses import FrozenInstanceError

import pytest
from gymnasium.envs.toy_text import frozen_lake as gym_frozen_lake

from frozen_lake.models import (
    Action,
    EpisodeResult,
    StrategyId,
    TerminationReason,
    Tile,
    Transition,
)


def test_action_values_match_gymnasium():
    assert Action.LEFT == gym_frozen_lake.LEFT
    assert Action.DOWN == gym_frozen_lake.DOWN
    assert Action.RIGHT == gym_frozen_lake.RIGHT
    assert Action.UP == gym_frozen_lake.UP


def test_action_iterates_in_gymnasium_order():
    assert list(Action) == [0, 1, 2, 3]


def test_tile_letters():
    assert [tile.value for tile in Tile] == ["S", "F", "H", "G"]


def test_transition_is_immutable():
    transition = Transition(0.8, 1, 0.0, False)
    with pytest.raises(FrozenInstanceError):
        transition.reward = 1.0  # type: ignore[misc]


def test_transition_rejects_unknown_attributes():
    transition = Transition(0.8, 1, 0.0, False)
    with pytest.raises((AttributeError, TypeError)):
        transition.probabilty = 0.5  # type: ignore[attr-defined]


def test_transition_equality_is_by_value():
    assert Transition(0.8, 1, 0.0, False) == Transition(0.8, 1, 0.0, False)
    assert Transition(0.8, 1, 0.0, False) != Transition(0.8, 2, 0.0, False)


def test_strategy_ids_are_the_public_identifiers():
    assert [s.value for s in StrategyId] == ["BFS", "EXP", "BFS-RO", "EXP-RO"]
    assert StrategyId("BFS-RO") is StrategyId.BFS_RO


def test_only_rollout_strategies_use_rollout():
    assert {s for s in StrategyId if s.uses_rollout} == {StrategyId.BFS_RO, StrategyId.EXP_RO}


@pytest.mark.parametrize(
    ("reason", "success"),
    [
        (TerminationReason.GOAL, True),
        (TerminationReason.HOLE, False),
        (TerminationReason.TIMEOUT, False),
    ],
)
def test_episode_success_is_derived_from_termination_reason(reason, success):
    episode = EpisodeResult(
        strategy=StrategyId.BFS,
        termination_reason=reason,
        steps=3,
        total_reward=1.0 if success else 0.0,
        agent_compute_ns=0,
        wall_ns=0,
        simulated_transitions=0,
        decisions=(),
    )
    assert episode.success is success
