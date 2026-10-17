import pytest

from frozen_lake.environment import generate_map, parse_map
from frozen_lake.episodes import EpisodeFinishedError, EpisodeSession, classify_end
from frozen_lake.models import Action, TerminationReason

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
