import copy
import inspect
import random

import pytest

import frozen_lake.rollout as rollout_module
from frozen_lake.environment import create_environment, extract_transitions, generate_map, parse_map
from frozen_lake.episodes import EpisodeSession, run_episode
from frozen_lake.models import Action, StrategyId, TerminationReason
from frozen_lake.policies.bfs import build_bfs_policy
from frozen_lake.policies.expectimax import build_expectimax_policy
from frozen_lake.rollout import (
    exact_rollout_decider,
    exact_values,
    rollout_decider,
    rollout_rng,
    simulate_trajectory,
)

RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
OPEN_4X4 = parse_map("SFFF\nFFFF\nFFFF\nFFFG")


def model_of(desc, success_rate=0.8):
    env = create_environment(desc, max_steps=4 * len(desc) ** 2, success_rate=success_rate)
    return extract_transitions(env)


class SpyPolicy:
    """A policy table that records every state it is asked about."""

    def __init__(self, actions):
        self.actions = tuple(actions)
        self.asked = []

    def __getitem__(self, state):
        self.asked.append(state)
        return self.actions[state]


def decider(policy, model, *, samples=20, horizon=20, seed=0, **kwargs):
    return rollout_decider(
        policy,
        model,
        samples=samples,
        horizon=horizon,
        gamma=1.0,
        rng=random.Random(seed),
        **kwargs,
    )


# --- one trajectory ----------------------------------------------------------------------------


def test_the_candidate_action_comes_first_and_the_base_policy_after():
    # Without slipping: DOWN from 0 lands on 4, where the base policy is asked for the first time.
    model = model_of(RUNNING_MAP, success_rate=1.0)
    spy = SpyPolicy(build_bfs_policy(RUNNING_MAP).policy)
    simulate_trajectory(model, spy, 0, Action.DOWN, horizon=2, gamma=1.0, rng=random.Random(0))
    assert spy.asked[0] == 4  # never asked about the starting state: the candidate decided it


def test_every_later_action_comes_from_the_base_policy():
    model = model_of(RUNNING_MAP, success_rate=1.0)
    policy = build_bfs_policy(RUNNING_MAP).policy
    spy = SpyPolicy(policy)
    reward, steps = simulate_trajectory(
        model, spy, 0, Action.RIGHT, horizon=50, gamma=1.0, rng=random.Random(0)
    )
    # Without slips, RIGHT then BFS follows the shortest path: 0→1, then 1→2→6→10→14→15.
    assert (reward, steps) == (1.0, 6)
    assert spy.asked == [1, 2, 6, 10, 14]


def test_a_trajectory_stops_at_the_goal_without_asking_the_policy():
    model = model_of(RUNNING_MAP, success_rate=1.0)
    spy = SpyPolicy(build_bfs_policy(RUNNING_MAP).policy)
    reward, steps = simulate_trajectory(
        model, spy, 14, Action.RIGHT, horizon=10, gamma=1.0, rng=random.Random(0)
    )
    assert (reward, steps) == (1.0, 1)
    assert spy.asked == []


def test_a_trajectory_stops_at_a_hole_with_no_reward():
    model = model_of(RUNNING_MAP, success_rate=1.0)
    reward, steps = simulate_trajectory(
        model,
        build_bfs_policy(RUNNING_MAP).policy,
        6,
        Action.LEFT,
        horizon=10,
        gamma=1.0,
        rng=random.Random(0),
    )
    assert (reward, steps) == (0.0, 1)


def test_a_trajectory_stops_at_the_horizon():
    model = model_of(OPEN_4X4, success_rate=1.0)
    always_left = tuple([Action.LEFT] * 15 + [None])  # never reaches the goal
    reward, steps = simulate_trajectory(
        model, always_left, 0, Action.LEFT, horizon=5, gamma=1.0, rng=random.Random(0)
    )
    assert (reward, steps) == (0.0, 5)


def test_discounting_applies_per_simulated_step():
    model = model_of(RUNNING_MAP, success_rate=1.0)
    reward, steps = simulate_trajectory(
        model,
        build_bfs_policy(RUNNING_MAP).policy,
        0,
        Action.RIGHT,
        horizon=50,
        gamma=0.5,
        rng=random.Random(0),
    )
    assert steps == 6
    assert reward == pytest.approx(0.5**5)  # the goal reward arrives on the sixth transition


# --- one decision ------------------------------------------------------------------------------


def test_a_decision_estimates_every_action_with_n_trajectories():
    decide = decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP), samples=20)
    result = decide(0, 64)
    assert set(result.estimated_action_values) == set(Action)
    for value in result.estimated_action_values.values():
        assert 0.0 <= value <= 1.0
        assert (value * 20) == pytest.approx(round(value * 20))  # a count of successes over 20
    assert result.trajectories_evaluated == 4 * 20
    assert result.base_action is build_bfs_policy(RUNNING_MAP).policy[0]
    assert result.compute_time_ns > 0


def test_simulated_transitions_count_what_was_actually_sampled(monkeypatch):
    calls = []
    real_sample = rollout_module.sample_transition

    def counting_sample(*args, **kwargs):
        calls.append(1)
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "sample_transition", counting_sample)
    result = decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP))(0, 64)
    assert result.simulated_transitions == len(calls)


def test_simulations_respect_the_real_steps_left():
    decide = decider(build_bfs_policy(OPEN_4X4).policy, model_of(OPEN_4X4), samples=10, horizon=50)
    result = decide(0, 2)
    assert result.simulated_transitions <= 4 * 10 * 2


def test_ties_keep_the_base_policy_action():
    # Every tile is cut off from the goal, so every estimate is 0: no evidence against the base.
    isolated = parse_map("SHF\nHFH\nFHG")
    policy = build_bfs_policy(isolated).policy
    result = decider(policy, model_of(isolated))(0, 36)
    assert set(result.estimated_action_values.values()) == {0.0}
    assert result.action is policy[0]


def test_when_the_base_action_is_worse_the_priority_breaks_ties_among_the_best():
    # Without slips, RIGHT and DOWN from 0 both reach the goal for sure; the base UP never does.
    lake = parse_map("SF\nFG")
    model = model_of(lake, success_rate=1.0)
    base = (Action.UP, Action.DOWN, Action.RIGHT, None)
    result = decider(base, model, samples=3)(0, 16)
    assert result.estimated_action_values[Action.RIGHT] == 1.0
    assert result.estimated_action_values[Action.DOWN] == 1.0
    assert result.estimated_action_values[Action.UP] == 0.0
    assert result.action is Action.RIGHT  # RIGHT comes before DOWN in the priority


def test_rollout_overrules_a_base_action_that_never_reaches_the_goal():
    lake = parse_map("SF\nFG")
    model = model_of(lake, success_rate=1.0)
    result = decider((Action.UP, Action.DOWN, Action.RIGHT, None), model, samples=3)(0, 16)
    assert result.base_action is Action.UP
    assert result.action is not result.base_action


def test_same_seed_gives_the_same_decision():
    policy, model = build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP)
    first = decider(policy, model, seed=7)(0, 64)
    second = decider(policy, model, seed=7)(0, 64)
    assert first.action is second.action
    assert first.estimated_action_values == second.estimated_action_values
    assert first.simulated_transitions == second.simulated_transitions


def test_simulations_never_touch_the_real_environment():
    session = EpisodeSession(RUNNING_MAP, max_steps=64, success_rate=0.8)
    session.reset(seed=123)
    session.step(Action.DOWN)
    env = session._env.unwrapped
    state_before = env.s
    rng_before = copy.deepcopy(env.np_random.bit_generator.state)

    decide = decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP), samples=200)
    decide(session.state, 63)

    assert env.s == state_before
    assert env.np_random.bit_generator.state == rng_before


@pytest.mark.parametrize(("samples", "horizon"), [(0, 10), (10, 0)])
def test_invalid_settings_are_rejected(samples, horizon):
    with pytest.raises(ValueError):
        decider(
            build_bfs_policy(RUNNING_MAP).policy,
            model_of(RUNNING_MAP),
            samples=samples,
            horizon=horizon,
        )


def test_terminal_tiles_and_finished_episodes_are_refused():
    decide = decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP))
    with pytest.raises(ValueError, match="terminal"):
        decide(15, 10)
    with pytest.raises(ValueError, match="no real steps"):
        decide(0, 0)


# --- shared implementation and whole episodes --------------------------------------------------


def test_rollout_never_imports_a_planner():
    source = inspect.getsource(rollout_module)
    assert "frozen_lake.policies" not in source


def test_both_rollout_strategies_use_the_same_function_with_different_tables():
    model = model_of(RUNNING_MAP)
    bfs = build_bfs_policy(RUNNING_MAP).policy
    exp = build_expectimax_policy(RUNNING_MAP, model, depth=6, beta=0.9, gamma=1.0).policy
    assert bfs != exp  # the tables differ...
    for policy in (bfs, exp):  # ...and the same code consults whichever it is given
        spy = SpyPolicy(policy)
        decider(spy, model)(0, 64)
        assert spy.asked


@pytest.mark.parametrize("strategy", [StrategyId.BFS_RO, StrategyId.EXP_RO])
def test_a_whole_rollout_episode(strategy):
    desc = generate_map(6, 0.9, seed=42)
    model = model_of(desc)
    if strategy.base is StrategyId.BFS:
        policy = build_bfs_policy(desc).policy
    else:
        policy = build_expectimax_policy(desc, model, depth=4, beta=0.9, gamma=1.0).policy
    session = EpisodeSession(desc, max_steps=144, success_rate=0.8)
    decide = rollout_decider(policy, model, samples=10, horizon=72, gamma=1.0, rng=rollout_rng(7))
    result = run_episode(session, decide, seed=7, strategy=strategy)
    assert result.termination_reason in set(TerminationReason)
    assert len(result.decisions) == result.steps
    assert all(d.estimated_action_values is not None for d in result.decisions)
    assert result.simulated_transitions == sum(d.simulated_transitions for d in result.decisions)


def test_the_simulation_stream_is_derived_from_the_episode_seed():
    assert rollout_rng(5).random() == rollout_rng(5).random()
    assert rollout_rng(5).random() != rollout_rng(6).random()


# --- exact reference (N = infinity) ------------------------------------------------------------


def test_exact_values_without_slipping_count_the_steps_to_the_goal():
    # Without slips, BFS reaches the goal from the start in exactly 6 steps.
    model = model_of(RUNNING_MAP, success_rate=1.0)
    values = exact_values(model, build_bfs_policy(RUNNING_MAP).policy, horizon=10, gamma=1.0)
    assert [values[h][0] for h in range(10)] == [0.0] * 6 + [1.0] * 4
    assert all(values[h][15] == 0.0 for h in range(10))  # the goal itself has nothing left to give


def test_exact_action_values_match_a_hand_computation():
    # From 14, RIGHT enters the goal with 0.8; with one step left nothing follows the slips.
    decide = exact_rollout_decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP),
                                   horizon=1, gamma=1.0)  # fmt: skip
    values = decide(14, 10).estimated_action_values
    assert values[Action.RIGHT] == pytest.approx(0.8)
    assert values[Action.UP] == pytest.approx(0.1)  # UP slips right into the goal with 0.1


def test_sampled_rollout_converges_to_the_exact_values():
    policy, model = build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP)
    exact = exact_rollout_decider(policy, model, horizon=32, gamma=1.0)(6, 64)
    sampled = decider(policy, model, samples=4000, horizon=32, seed=3)(6, 64)
    for action in Action:
        assert sampled.estimated_action_values[action] == pytest.approx(
            exact.estimated_action_values[action], abs=0.03
        )


def test_exact_reference_respects_the_real_steps_left():
    decide = exact_rollout_decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP),
                                   horizon=50, gamma=1.0)  # fmt: skip
    capped = decide(14, 1).estimated_action_values
    assert capped[Action.RIGHT] == pytest.approx(0.8)  # one step left: only the direct entry


def test_exact_reference_keeps_the_base_action_on_ties_and_samples_nothing():
    isolated = parse_map("SHF\nHFH\nFHG")
    policy = build_bfs_policy(isolated).policy
    result = exact_rollout_decider(policy, model_of(isolated), horizon=9, gamma=1.0)(0, 36)
    assert set(result.estimated_action_values.values()) == {0.0}
    assert result.action is policy[0]
    assert (result.simulated_transitions, result.trajectories_evaluated) == (0, 0)


@pytest.mark.parametrize(("horizon", "gamma"), [(0, 1.0), (5, 0.0)])
def test_exact_reference_rejects_invalid_settings(horizon, gamma):
    with pytest.raises(ValueError):
        exact_rollout_decider(build_bfs_policy(RUNNING_MAP).policy, model_of(RUNNING_MAP),
                              horizon=horizon, gamma=gamma)  # fmt: skip
