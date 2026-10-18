import math

import pytest

from frozen_lake.environment import (
    create_environment,
    deterministic_move,
    extract_transitions,
    generate_map,
    parse_map,
    tile_at,
)
from frozen_lake.models import Action, Tile
from frozen_lake.policies.bfs import bfs_distances, build_bfs_policy
from frozen_lake.policies.expectimax import (
    ExpectimaxSearch,
    bfs_heuristic,
    build_expectimax_policy,
    expectimax_action_values,
)

# The running 4x4 map of the design doc.
RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
BETA, GAMMA = 0.9, 1.0


def model_of(desc, success_rate=0.8):
    env = create_environment(desc, max_steps=4 * len(desc) ** 2, success_rate=success_rate)
    return extract_transitions(env)


RUNNING_MODEL = model_of(RUNNING_MAP)


def uncached_value(desc, model, state, depth, beta=BETA, gamma=GAMMA):
    """Independent oracle: the same recursion written plainly, with no cache at all."""
    if depth == 0:
        distance = bfs_distances(desc)[state]
        return 0.0 if distance == math.inf else beta**distance
    return max(uncached_q(desc, model, state, a, depth, beta, gamma) for a in Action)


def uncached_q(desc, model, state, action, depth, beta=BETA, gamma=GAMMA):
    total = 0.0
    for t in model[state][action]:
        future = (
            0.0 if t.terminated else gamma * uncached_value(desc, model, t.next_state, depth - 1)
        )
        total += t.probability * (t.reward + future)
    return total


# --- heuristic ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distance", "expected"), [(1, 0.9), (2, 0.81), (3, 0.729), (5, 0.59049), (math.inf, 0.0)]
)
def test_heuristic_values(distance, expected):
    assert bfs_heuristic(distance, 0.9) == pytest.approx(expected)


def test_heuristic_decreases_with_distance():
    scores = [bfs_heuristic(d, 0.85) for d in range(20)]
    assert all(a > b for a, b in zip(scores, scores[1:], strict=False))


@pytest.mark.parametrize(
    ("depth", "beta", "gamma"), [(0, 0.9, 1.0), (6, 0.0, 1.0), (6, 1.0, 1.0), (6, 0.9, 0.0)]
)
def test_invalid_parameters_are_rejected(depth, beta, gamma):
    with pytest.raises(ValueError):
        build_expectimax_policy(RUNNING_MAP, RUNNING_MODEL, depth=depth, beta=beta, gamma=gamma)


# --- recursion ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (14, {Action.LEFT: 0.819, Action.DOWN: 0.901, Action.RIGHT: 0.971, Action.UP: 0.829}),
        (6, {Action.LEFT: 0.14661, Action.DOWN: 0.648, Action.RIGHT: 0.14661, Action.UP: 0.52488}),
    ],
)
def test_one_step_values_match_the_design_doc(state, expected):
    values = expectimax_action_values(
        RUNNING_MAP, RUNNING_MODEL, state, depth=1, beta=BETA, gamma=GAMMA
    )
    assert values == pytest.approx(expected)


def test_goal_reward_is_counted_once_with_no_continuation():
    # From 14, RIGHT enters the goal with probability 0.8: it contributes exactly 0.8 * 1, not
    # 0.8 * (1 + h(goal)). The slips land on 14 and 10, scored by the heuristic at depth 1.
    q = expectimax_action_values(RUNNING_MAP, RUNNING_MODEL, 14, depth=1, beta=BETA, gamma=GAMMA)
    assert q[Action.RIGHT] == pytest.approx(0.8 * 1 + 0.1 * 0.9 + 0.1 * 0.81)


def test_holes_contribute_nothing():
    # From 6, LEFT falls into hole 5 with probability 0.8; only the two slips score.
    q = expectimax_action_values(RUNNING_MAP, RUNNING_MODEL, 6, depth=1, beta=BETA, gamma=GAMMA)
    assert q[Action.LEFT] == pytest.approx(0.1 * 0.9**4 + 0.1 * 0.9**2)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_memoized_values_match_the_uncached_oracle(depth):
    search = ExpectimaxSearch(RUNNING_MODEL, bfs_distances(RUNNING_MAP), beta=BETA, gamma=GAMMA)
    for state in range(16):
        if tile_at(RUNNING_MAP, state) in (Tile.HOLE, Tile.GOAL):
            continue
        for action in Action:
            expected = uncached_q(RUNNING_MAP, RUNNING_MODEL, state, action, depth)
            assert search.action_value(state, action, depth) == pytest.approx(expected)


def test_remaining_depth_is_part_of_the_cache_key():
    search = ExpectimaxSearch(RUNNING_MODEL, bfs_distances(RUNNING_MAP), beta=BETA, gamma=GAMMA)
    shallow, deep = search.value(0, 1), search.value(0, 5)
    assert shallow != pytest.approx(deep)
    assert search.cache[(0, 1)] == shallow and search.cache[(0, 5)] == deep


def test_the_heuristic_is_used_only_where_the_depth_runs_out():
    search = ExpectimaxSearch(RUNNING_MODEL, bfs_distances(RUNNING_MAP), beta=BETA, gamma=GAMMA)
    assert search.value(9, 0) == pytest.approx(0.9**3)  # depth 0: the heuristic itself
    assert search.value(9, 2) != pytest.approx(0.9**3)  # depth 2: planned, not the heuristic


# --- policy ------------------------------------------------------------------------------------


def build(desc=RUNNING_MAP, model=RUNNING_MODEL, depth=6, **kwargs):
    return build_expectimax_policy(desc, model, depth=depth, beta=BETA, gamma=GAMMA, **kwargs)


def test_every_nonterminal_tile_has_an_action_and_terminals_have_none():
    policy = build().policy
    for state, action in enumerate(policy):
        terminal = tile_at(RUNNING_MAP, state) in (Tile.HOLE, Tile.GOAL)
        assert (action is None) == terminal


def test_policy_picks_the_highest_value_action():
    policy = build(depth=1).policy
    assert policy[14] is Action.RIGHT
    assert policy[6] is Action.DOWN


def test_policy_matches_the_argmax_of_the_oracle_at_depth_3():
    policy = build(depth=3).policy
    for state, action in enumerate(policy):
        if action is None:
            continue
        values = {a: uncached_q(RUNNING_MAP, RUNNING_MODEL, state, a, 3) for a in Action}
        assert values[action] == pytest.approx(max(values.values()))


def test_ties_follow_the_action_priority():
    # On an open lake, RIGHT and DOWN are mirror images from the start, so they tie.
    open_map = parse_map("SFFF\nFFFF\nFFFF\nFFFG")
    model = model_of(open_map)
    assert build(open_map, model).policy[0] is Action.RIGHT
    down_first = (Action.DOWN, Action.RIGHT, Action.LEFT, Action.UP)
    assert build(open_map, model, action_priority=down_first).policy[0] is Action.DOWN


def test_zero_value_tiles_use_the_safe_bfs_fallback():
    # Holes isolate every tile, so every action is worth zero everywhere.
    isolated = parse_map("SHF\nHFH\nFHG")
    result = build(isolated, model_of(isolated))
    assert result.policy == build_bfs_policy(isolated).policy
    assert result.diagnostics["fallback_states"] == 4


def test_counters_follow_the_memoization():
    result = build(depth=6)
    n_states = 16
    assert result.states_evaluated == result.diagnostics["cache_entries"]
    assert result.states_evaluated <= n_states * (6 + 1)  # one value per (state, depth)
    assert result.diagnostics["cache_hits"] > 0
    assert result.diagnostics["peak_recursion"] == 6  # MAX nodes from depth 5 down to 0
    assert result.successors_examined > 0
    assert result.build_time_ns > 0


def test_larger_maps_build_a_complete_policy():
    desc = generate_map(10, 0.9, seed=2026)
    policy = build(desc, model_of(desc)).policy
    assert len(policy) == 100
    assert all(
        (policy[s] is None) == (tile_at(desc, s) in (Tile.HOLE, Tile.GOAL)) for s in range(100)
    )


def test_without_slipping_and_depth_1_the_heuristic_picks_the_closer_tile():
    # One step ahead, without slips, each action leads to one tile whose heuristic is
    # beta ** distance, so the best action is the one that gets one step closer.
    model = model_of(RUNNING_MAP, success_rate=1.0)
    policy = build(model=model, depth=1).policy
    distances = bfs_distances(RUNNING_MAP)
    for state, action in enumerate(policy):
        if action is not None:
            assert distances[deterministic_move(state, action, 4)] == distances[state] - 1


def test_with_gamma_1_every_route_reaching_the_goal_within_the_horizon_is_worth_1():
    # From state 4 (distance 5), without slips, DOWN gets closer and UP goes back to the start
    # (distance 6). Both still reach the goal within 7 moves, so with gamma = 1 both are worth
    # exactly 1: the planner has no preference for arriving sooner.
    model = model_of(RUNNING_MAP, success_rate=1.0)
    q = expectimax_action_values(RUNNING_MAP, model, 4, depth=7, beta=BETA, gamma=GAMMA)
    assert q[Action.DOWN] == pytest.approx(1.0)
    assert q[Action.UP] == pytest.approx(1.0)
