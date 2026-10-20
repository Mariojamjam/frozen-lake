import math
import time

import pytest

import frozen_lake.policies.expectimax as expectimax_module
from frozen_lake.environment import (
    create_environment,
    deterministic_move,
    extract_transitions,
    generate_map,
    map_size,
    parse_map,
    tile_at,
)
from frozen_lake.models import DEFAULT_ACTION_PRIORITY, Action, Tile
from frozen_lake.policies.bfs import bfs_distances, build_bfs_policy
from frozen_lake.policies.expectimax import (
    ExpectimaxSearch,
    bfs_heuristic,
    build_expectimax_policy,
    expectimax_action_values,
)

# The running 4x4 map of the design doc.
RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
GENERATED_6X6 = generate_map(6, 0.9, seed=3)
BETA, GAMMA = 0.9, 1.0


def model_of(desc, success_rate=0.8):
    env = create_environment(desc, max_steps=4 * len(desc) ** 2, success_rate=success_rate)
    return extract_transitions(env)


RUNNING_MODEL = model_of(RUNNING_MAP)


def nonterminal(desc, state):
    return tile_at(desc, state) not in (Tile.HOLE, Tile.GOAL)


def uncached_q(model, distances, state, action, depth, beta=BETA, gamma=GAMMA):
    """Independent oracle: the same recursion written plainly, with no cache at all."""

    def value(s, d):
        if d == 0:
            return bfs_heuristic(distances[s], beta)
        return max(q(s, a, d) for a in Action)

    def q(s, a, d):
        return sum(
            t.probability
            * (t.reward + (0.0 if t.terminated else gamma * value(t.next_state, d - 1)))
            for t in model[s][a]
        )

    return q(state, action, depth)


def build(desc=RUNNING_MAP, model=RUNNING_MODEL, depth=6, **kwargs):
    return build_expectimax_policy(desc, model, depth=depth, beta=BETA, gamma=GAMMA, **kwargs)


# --- heuristic and parameters ------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.5, 0.85, 0.9, 0.95])
def test_the_heuristic_is_beta_to_the_distance_and_zero_without_a_route(beta):
    for distance in range(25):
        assert bfs_heuristic(distance, beta) == pytest.approx(beta**distance)
    assert bfs_heuristic(math.inf, beta) == 0.0


@pytest.mark.parametrize(
    ("depth", "beta", "gamma"), [(0, 0.9, 1.0), (6, 0.0, 1.0), (6, 1.0, 1.0), (6, 0.9, 0.0)]
)
def test_invalid_parameters_are_rejected(depth, beta, gamma):
    with pytest.raises(ValueError):
        build_expectimax_policy(RUNNING_MAP, RUNNING_MODEL, depth=depth, beta=beta, gamma=gamma)


# --- recursion ---------------------------------------------------------------------------------


def test_goal_reward_is_counted_once_with_no_continuation():
    # From 14, RIGHT enters the goal with probability 0.8: it contributes exactly 0.8 * 1, not
    # 0.8 * (1 + h(goal)). The slips land on 14 and 10, scored by the heuristic at depth 1.
    q = expectimax_action_values(RUNNING_MAP, RUNNING_MODEL, 14, depth=1, beta=BETA, gamma=GAMMA)
    assert q[Action.RIGHT] == pytest.approx(0.8 * 1 + 0.1 * BETA + 0.1 * BETA**2)


def test_holes_contribute_nothing():
    # From 6, LEFT falls into hole 5 with probability 0.8; only the two slips score.
    q = expectimax_action_values(RUNNING_MAP, RUNNING_MODEL, 6, depth=1, beta=BETA, gamma=GAMMA)
    assert q[Action.LEFT] == pytest.approx(0.1 * BETA**4 + 0.1 * BETA**2)


@pytest.mark.parametrize("gamma", [1.0, 0.9])
@pytest.mark.parametrize(
    ("desc", "depth"),
    [(RUNNING_MAP, 1), (RUNNING_MAP, 2), (RUNNING_MAP, 4), (GENERATED_6X6, 3)],
    ids=["running-d1", "running-d2", "running-d4", "generated-6x6-d3"],
)
def test_memoized_values_match_the_uncached_oracle(desc, depth, gamma):
    model, distances = model_of(desc), bfs_distances(desc)
    search = ExpectimaxSearch(model, distances, beta=BETA, gamma=gamma)
    for state in range(map_size(desc) ** 2):
        if nonterminal(desc, state):
            for action in Action:
                expected = uncached_q(model, distances, state, action, depth, gamma=gamma)
                assert search.action_value(state, action, depth) == pytest.approx(expected)


def test_one_search_keeps_values_of_different_depths_apart():
    # The remaining depth is part of every memoized value: asking for several depths in any
    # order on one search must give each depth its own value.
    distances = bfs_distances(RUNNING_MAP)
    search = ExpectimaxSearch(RUNNING_MODEL, distances, beta=BETA, gamma=GAMMA)
    for depth in (5, 1, 3, 2, 4, 1, 5):
        for action in Action:
            expected = uncached_q(RUNNING_MODEL, distances, 0, action, depth)
            assert search.action_value(0, action, depth) == pytest.approx(expected)


def test_counters_count_each_subproblem_outcome_and_reuse_once():
    depth = 4
    result = build(depth=depth)
    model = RUNNING_MODEL
    roots = [s for s in range(16) if nonterminal(RUNNING_MAP, s)]

    # Recount independently: the (state, depth) values the search needs, the outcome records it
    # reads, and how many times a value is asked for.
    subproblems, frontier, outcomes, requests = set(), [], 0, 0
    for state in roots:
        for action in Action:
            for t in model[state][action]:
                outcomes += 1
                if not t.terminated:
                    requests += 1
                    frontier.append((t.next_state, depth - 1))
    while frontier:
        key = frontier.pop()
        if key in subproblems:
            continue
        subproblems.add(key)
        state, remaining = key
        if remaining == 0:
            continue
        for action in Action:
            for t in model[state][action]:
                outcomes += 1
                if not t.terminated:
                    requests += 1
                    frontier.append((t.next_state, remaining - 1))

    assert result.states_evaluated == len(subproblems)
    assert result.successors_examined == outcomes
    assert result.diagnostics["cache_hits"] == requests - len(subproblems)
    assert result.diagnostics["cache_entries"] == len(subproblems)


def test_the_timed_build_includes_the_bfs_distances(monkeypatch):
    # Expectimax is charged for the BFS distances its heuristic needs (standalone cost).
    delay_ns = 30_000_000
    real = expectimax_module.bfs_distances

    def slow_bfs(desc):
        time.sleep(delay_ns / 1e9)
        return real(desc)

    monkeypatch.setattr(expectimax_module, "bfs_distances", slow_bfs)
    assert build(depth=1).build_time_ns >= delay_ns


# --- policy ------------------------------------------------------------------------------------


@pytest.mark.parametrize("desc", [RUNNING_MAP, GENERATED_6X6], ids=["running-4x4", "generated-6x6"])
def test_policy_takes_the_first_best_action_in_priority_order(desc):
    model, distances = model_of(desc), bfs_distances(desc)
    policy = build(desc, model, depth=3).policy
    for state, action in enumerate(policy):
        if not nonterminal(desc, state):
            assert action is None
            continue
        values = {a: uncached_q(model, distances, state, a, 3) for a in Action}
        best = max(values.values())
        if best > 1e-12:
            tied = [a for a in DEFAULT_ACTION_PRIORITY if values[a] >= best - 1e-9]
            assert action is tied[0]


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


def test_larger_maps_build_a_complete_policy():
    desc = generate_map(10, 0.9, seed=2026)
    policy = build(desc, model_of(desc)).policy
    assert all((policy[s] is None) == (not nonterminal(desc, s)) for s in range(100))


def test_without_slipping_and_depth_1_expectimax_moves_like_bfs():
    # One step ahead, without slips, each action leads to one tile scored beta ** distance,
    # so the best action is the one that gets one step closer, as in BFS.
    model = model_of(RUNNING_MAP, success_rate=1.0)
    policy = build(model=model, depth=1).policy
    distances = bfs_distances(RUNNING_MAP)
    for state, action in enumerate(policy):
        if action is not None:
            assert distances[deterministic_move(state, action, 4)] == distances[state] - 1
