import pytest
from gymnasium.envs.toy_text import frozen_lake as gym_frozen_lake

from frozen_lake.models import Action, EpisodeResult, StrategyId, TerminationReason


def test_action_values_match_gymnasium():
    # States and actions are exchanged with Gymnasium as integers, so the encoding must agree.
    assert Action.LEFT == gym_frozen_lake.LEFT
    assert Action.DOWN == gym_frozen_lake.DOWN
    assert Action.RIGHT == gym_frozen_lake.RIGHT
    assert Action.UP == gym_frozen_lake.UP


def test_strategies_know_their_base_policy_and_whether_they_sample():
    assert {s for s in StrategyId if s.uses_rollout} == {StrategyId.BFS_RO, StrategyId.EXP_RO}
    assert {s: s.base for s in StrategyId} == {
        StrategyId.BFS: StrategyId.BFS,
        StrategyId.EXP: StrategyId.EXP,
        StrategyId.BFS_RO: StrategyId.BFS,
        StrategyId.EXP_RO: StrategyId.EXP,
    }


@pytest.mark.parametrize("reason", list(TerminationReason))
def test_success_means_reaching_the_goal(reason):
    episode = EpisodeResult(
        strategy=StrategyId.BFS,
        termination_reason=reason,
        steps=3,
        total_reward=0.0,
        agent_compute_ns=0,
        wall_ns=0,
        simulated_transitions=0,
        decisions=(),
    )
    assert episode.success is (reason is TerminationReason.GOAL)
