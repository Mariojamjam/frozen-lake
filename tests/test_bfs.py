import math

import pytest

from frozen_lake.environment import (
    create_environment,
    deterministic_move,
    extract_transitions,
    parse_map,
)
from frozen_lake.models import Action
from frozen_lake.policies.bfs import bfs_distances, build_bfs_policy, policy_from_distances

# The running 4x4 map of the design doc, with its published distance grid.
RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
INF = math.inf
RUNNING_DISTANCES = (6, 5, 4, 5, 5, INF, 3, INF, 4, 3, 2, INF, INF, 2, 1, 0)
HOLES_AND_GOAL = (5, 7, 11, 12, 15)


def test_distances_match_the_design_doc_example():
    assert bfs_distances(RUNNING_MAP) == RUNNING_DISTANCES


def test_policy_matches_the_design_doc_example():
    policy = build_bfs_policy(RUNNING_MAP).policy
    assert len(policy) == 16
    assert policy[6] is Action.DOWN  # the worked example in the design doc
    assert all(policy[s] is None for s in HOLES_AND_GOAL)
    assert all(policy[s] is not None for s in range(16) if s not in HOLES_AND_GOAL)


def test_every_move_gets_one_step_closer_to_the_goal():
    distances = bfs_distances(RUNNING_MAP)
    policy = build_bfs_policy(RUNNING_MAP).policy
    for state, action in enumerate(policy):
        if action is not None:
            assert distances[deterministic_move(state, action, 4)] == distances[state] - 1


def test_ties_follow_the_action_priority():
    # From the start, RIGHT (state 1) and DOWN (state 4) are both at distance 5.
    assert build_bfs_policy(RUNNING_MAP).policy[0] is Action.RIGHT
    down_first = (Action.DOWN, Action.RIGHT, Action.LEFT, Action.UP)
    assert build_bfs_policy(RUNNING_MAP, action_priority=down_first).policy[0] is Action.DOWN


def test_action_priority_must_list_every_action_once():
    with pytest.raises(ValueError, match="action_priority"):
        build_bfs_policy(RUNNING_MAP, action_priority=(Action.RIGHT,))


def test_tile_without_a_route_gets_a_safe_fallback_and_is_counted():
    cut_off = parse_map("SHF\nHHF\nFFG")  # the start is walled in by holes
    result = build_bfs_policy(cut_off)
    assert bfs_distances(cut_off)[0] == INF
    # RIGHT and DOWN lead into holes; LEFT bumps into the border and keeps the agent alive.
    assert result.policy[0] is Action.LEFT
    assert result.diagnostics["fallback_states"] == 1
    assert build_bfs_policy(RUNNING_MAP).diagnostics["fallback_states"] == 0


def test_fallback_uses_the_first_action_when_every_move_leads_into_a_hole():
    # Holes at 1, 3, 5 and 7 isolate every tile, so none of 0, 2, 4, 6 has a route to the goal.
    isolated = parse_map("SHF\nHFH\nFHG")
    result = build_bfs_policy(isolated)
    assert result.policy[4] is Action.RIGHT  # center: all four moves lead into holes
    assert result.policy[0] is Action.LEFT  # corners prefer a safe border bump
    assert result.policy[2] is Action.RIGHT
    assert result.policy[6] is Action.DOWN
    assert result.diagnostics["fallback_states"] == 4


def test_counters_on_a_tiny_map():
    result = build_bfs_policy(parse_map("SF\nFG"))
    assert result.policy == (Action.RIGHT, Action.DOWN, Action.RIGHT, None)
    assert result.states_evaluated == 4
    assert result.successors_examined == 8  # each of the 4 tiles has 2 in-grid neighbors
    assert result.diagnostics["peak_queue_length"] == 2
    assert result.build_time_ns > 0


def test_deterministic_move_bumps_into_borders():
    assert deterministic_move(0, Action.LEFT, 4) == 0
    assert deterministic_move(0, Action.UP, 4) == 0
    assert deterministic_move(15, Action.RIGHT, 4) == 15
    assert deterministic_move(0, Action.RIGHT, 4) == 1
    assert deterministic_move(0, Action.DOWN, 4) == 4


def test_deterministic_move_agrees_with_the_environment_without_slipping():
    env = create_environment(RUNNING_MAP, max_steps=16, success_rate=1.0)
    model = extract_transitions(env)
    for state in range(16):
        if state in HOLES_AND_GOAL:
            continue
        for action in Action:
            (outcome,) = model[state][action]
            assert outcome.next_state == deterministic_move(state, action, 4)


def test_policy_from_distances_matches_the_full_build():
    for desc in (RUNNING_MAP, parse_map("SHF\nHHF\nFFG"), parse_map("SHF\nHFH\nFHG")):
        assert policy_from_distances(desc, bfs_distances(desc)) == build_bfs_policy(desc).policy


def test_policy_from_distances_follows_a_hand_made_table():
    # A fabricated table puts the start's right and down neighbors at the same distance,
    # so the choice there is decided by the action priority alone.
    open_map = parse_map("SF\nFG")
    table = (2, 1, 1, 0)
    assert policy_from_distances(open_map, table)[0] is Action.RIGHT
    down_first = (Action.DOWN, Action.RIGHT, Action.LEFT, Action.UP)
    assert policy_from_distances(open_map, table, down_first)[0] is Action.DOWN


def test_policy_from_distances_rejects_a_table_of_the_wrong_size():
    with pytest.raises(ValueError, match="expected 16"):
        policy_from_distances(RUNNING_MAP, (0, 1, 2))
