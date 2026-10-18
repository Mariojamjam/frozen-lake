import pytest

from frozen_lake.environment import generate_map, parse_map
from frozen_lake.episodes import (
    EpisodeFinishedError,
    EpisodeSession,
    classify_end,
    policy_decider,
    run_episode,
)
from frozen_lake.models import Action, StrategyId, TerminationReason
from frozen_lake.policies.bfs import bfs_distances, build_bfs_policy

# With success_rate=1.0 every move goes exactly where intended, which makes outcomes testable.
HOLE_TO_THE_RIGHT = parse_map("SH\nFG")
GOAL_TO_THE_RIGHT = parse_map("SG\nFF")
OPEN_4X4 = parse_map("SFFF\nFFFF\nFFFF\nFFFG")


def session(desc, *, max_steps=50, success_rate=1.0, seed=0):
    episode = EpisodeSession(desc, max_steps=max_steps, success_rate=success_rate)
    episode.reset(seed)
    return episode


def test_step_before_reset_is_rejected():
    episode = EpisodeSession(OPEN_4X4, max_steps=10, success_rate=0.8)
    with pytest.raises(RuntimeError, match="reset"):
        episode.step(Action.RIGHT)


def test_one_step_advances_exactly_one_transition():
    episode = session(OPEN_4X4)
    event = episode.step(Action.RIGHT)
    assert (event.step, event.state, event.action) == (1, 0, Action.RIGHT)
    assert (event.intended_state, event.next_state) == (1, 1)
    assert event.moved_as_intended
    assert event.termination is None
    assert (episode.steps, episode.state, episode.finished) == (1, 1, False)


def test_entering_a_hole_ends_the_episode_as_hole():
    episode = session(HOLE_TO_THE_RIGHT)
    event = episode.step(Action.RIGHT)
    assert event.termination is TerminationReason.HOLE
    assert event.reward == 0.0
    assert episode.finished


def test_entering_the_goal_ends_the_episode_with_reward_once():
    episode = session(GOAL_TO_THE_RIGHT)
    event = episode.step(Action.RIGHT)
    assert event.termination is TerminationReason.GOAL
    assert event.reward == 1.0
    assert episode.total_reward == 1.0


def test_running_out_of_steps_ends_the_episode_as_timeout():
    episode = session(OPEN_4X4, max_steps=2)
    assert episode.step(Action.LEFT).termination is None
    assert episode.step(Action.LEFT).termination is TerminationReason.TIMEOUT


def test_goal_on_the_last_allowed_step_is_a_success_not_a_timeout():
    episode = session(GOAL_TO_THE_RIGHT, max_steps=1)
    assert episode.step(Action.RIGHT).termination is TerminationReason.GOAL


@pytest.mark.parametrize(
    ("desc", "max_steps", "action"),
    [
        (HOLE_TO_THE_RIGHT, 50, Action.RIGHT),
        (GOAL_TO_THE_RIGHT, 50, Action.RIGHT),
        (OPEN_4X4, 1, Action.LEFT),
    ],
    ids=["hole", "goal", "timeout"],
)
def test_no_step_is_allowed_after_the_episode_ends(desc, max_steps, action):
    episode = session(desc, max_steps=max_steps)
    episode.step(action)
    with pytest.raises(EpisodeFinishedError):
        episode.step(action)
    assert episode.steps == 1


def test_reset_starts_over():
    episode = session(HOLE_TO_THE_RIGHT)
    episode.step(Action.RIGHT)
    assert episode.reset(seed=1) == 0
    assert (episode.steps, episode.total_reward, episode.termination) == (0, 0.0, None)


def test_same_seed_and_actions_reproduce_the_same_slips():
    desc = generate_map(8, 0.9, seed=3)
    actions = [Action.RIGHT, Action.DOWN] * 20

    def play(seed):
        episode = session(desc, max_steps=100, success_rate=0.8, seed=seed)
        states = []
        for action in actions:
            states.append(episode.step(action).next_state)
            if episode.finished:
                break
        return states

    assert play(11) == play(11)


def test_slips_are_reported_as_not_moving_as_intended():
    # From the start corner, DOWN aims at state 4; a slip goes left (bumps, stays at 0) or
    # right (to 1). Either way the agent does not land where it intended.
    outcomes = set()
    for seed in range(200):
        event = session(OPEN_4X4, success_rate=0.8, seed=seed).step(Action.DOWN)
        outcomes.add((event.next_state, event.moved_as_intended))
    assert outcomes == {(4, True), (0, False), (1, False)}


@pytest.mark.parametrize(
    ("state", "terminated", "truncated", "expected"),
    [
        (3, True, False, TerminationReason.GOAL),
        (3, True, True, TerminationReason.GOAL),
        (1, True, False, TerminationReason.HOLE),
        (2, False, True, TerminationReason.TIMEOUT),
        (2, False, False, None),
    ],
)
def test_classify_end(state, terminated, truncated, expected):
    desc = parse_map("SH\nFG")
    assert classify_end(desc, state, terminated=terminated, truncated=truncated) is expected


# --- whole episodes driven by a policy --------------------------------------------------------

RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")


def play_bfs(desc, *, seed, max_steps=64, success_rate=0.8):
    episode = EpisodeSession(desc, max_steps=max_steps, success_rate=success_rate)
    decide = policy_decider(build_bfs_policy(desc).policy)
    return run_episode(episode, decide, seed=seed, strategy=StrategyId.BFS)


def test_bfs_without_slipping_reaches_the_goal_in_its_shortest_distance():
    result = play_bfs(RUNNING_MAP, seed=0, success_rate=1.0)
    assert result.termination_reason is TerminationReason.GOAL
    assert result.success
    assert result.steps == bfs_distances(RUNNING_MAP)[0] == 6
    assert result.total_reward == 1.0


def test_every_step_has_one_base_policy_decision():
    result = play_bfs(generate_map(8, 0.9, seed=3), seed=5)
    assert len(result.decisions) == result.steps
    for decision in result.decisions:
        assert decision.action is decision.base_action
        assert decision.estimated_action_values is None
        assert decision.simulated_transitions == decision.trajectories_evaluated == 0
    assert result.agent_compute_ns == sum(d.compute_time_ns for d in result.decisions)
    assert result.wall_ns >= result.agent_compute_ns


def test_decisions_follow_the_policy_at_every_observed_state():
    desc = generate_map(8, 0.9, seed=3)
    policy = build_bfs_policy(desc).policy
    result = play_bfs(desc, seed=11)
    assert all(d.action is policy[d.state] for d in result.decisions)


def test_same_seed_reproduces_the_same_policy_episode():
    desc = generate_map(10, 0.9, seed=2026)
    first, second = play_bfs(desc, seed=9, max_steps=400), play_bfs(desc, seed=9, max_steps=400)
    assert [d.state for d in first.decisions] == [d.state for d in second.decisions]
    assert first.termination_reason is second.termination_reason


def test_a_tile_without_a_route_does_not_break_the_episode():
    # The start is walled in; the safe fallback keeps bumping into the border until timeout.
    result = play_bfs(parse_map("SHF\nHHF\nFFG"), seed=0, max_steps=5, success_rate=1.0)
    assert result.termination_reason is TerminationReason.TIMEOUT
    assert result.steps == 5


def test_policy_decider_refuses_terminal_tiles():
    decide = policy_decider(build_bfs_policy(RUNNING_MAP).policy)
    with pytest.raises(ValueError, match="terminal"):
        decide(15)


def test_on_step_sees_each_decision_and_its_outcome():
    seen = []
    episode = EpisodeSession(RUNNING_MAP, max_steps=64, success_rate=1.0)
    decide = policy_decider(build_bfs_policy(RUNNING_MAP).policy)
    run_episode(
        episode, decide, seed=0, strategy=StrategyId.BFS, on_step=lambda d, e: seen.append((d, e))
    )
    assert [event.step for _, event in seen] == [1, 2, 3, 4, 5, 6]
    assert all(decision.action is event.action for decision, event in seen)
