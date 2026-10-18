"""BFS base policy: shortest deterministic routes to the goal, ignoring slipping.

A single reverse BFS from the goal gives the distance of every tile to the goal; each nonterminal
tile then intends the move toward its closest neighbor (design doc, section 4.1). The result is a
complete policy table, so the agent still has a sensible action after slipping onto any tile.
"""

import math
import time
from collections import deque

from frozen_lake.environment import (
    MapDesc,
    deterministic_move,
    find_tile,
    map_size,
    tile_at,
)
from frozen_lake.models import (
    DEFAULT_ACTION_PRIORITY,
    Action,
    Policy,
    PolicyBuildResult,
    Tile,
)


def bfs_distances(desc: MapDesc) -> tuple[float, ...]:
    """Return the BFS distance of every state to the goal; holes and cut-off tiles are `inf`."""
    distances, _, _, _ = _reverse_bfs(desc)
    return tuple(distances)


def policy_from_distances(
    desc: MapDesc,
    distances: tuple[float, ...],
    action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY,
) -> Policy:
    """Turn a BFS distance table into a complete policy table. Not timed and not counted.

    Every nonterminal tile intends the move whose deterministic target has the smallest BFS
    distance; ties go to the first action in `action_priority`. Terminal tiles (hole, goal) get
    None. A nonterminal tile with no route to the goal gets a deterministic fallback: the first
    action in priority order whose deterministic target is not a hole (bumping into a border
    counts as safe), or the first action overall if every move leads into a hole.

    Use this to reuse a distance table already computed for another purpose (e.g. the
    Expectimax heuristic or a display). To measure what a BFS policy costs, use
    build_bfs_policy, which always runs the search itself.
    """
    _check_priority(action_priority)
    n = map_size(desc)
    if len(distances) != n * n:
        raise ValueError(f"distances has {len(distances)} entries, expected {n * n}")

    policy: list[Action | None] = []
    for state in range(n * n):
        if tile_at(desc, state) in (Tile.HOLE, Tile.GOAL):
            policy.append(None)
            continue
        best_action: Action | None = None
        best_distance = math.inf
        for action in action_priority:
            distance = distances[deterministic_move(state, action, n)]
            if distance < best_distance:  # strict: earlier actions win ties
                best_action, best_distance = action, distance
        if best_action is None:
            best_action = _fallback_action(desc, state, n, action_priority)
        policy.append(best_action)
    return tuple(policy)


def build_bfs_policy(
    desc: MapDesc, action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY
) -> PolicyBuildResult:
    """Build the BFS policy from scratch, timing the search and the extraction together.

    This is the standalone cost of the BFS strategy: it never accepts a precomputed distance
    table, so a benchmark cannot skip the search by accident.

    Counters: `states_evaluated` is the number of states the search dequeued (the goal
    included), `successors_examined` the in-grid neighbors it checked. The extraction only
    reads the finished distance table and is not counted. Diagnostics record the peak queue
    length and how many tiles needed the fallback (see policy_from_distances).
    """
    _check_priority(action_priority)
    start_ns = time.perf_counter_ns()
    distances, visited, neighbor_checks, peak_queue = _reverse_bfs(desc)
    policy = policy_from_distances(desc, tuple(distances), action_priority)
    build_time_ns = time.perf_counter_ns() - start_ns

    # A nonterminal tile needs the fallback exactly when it has no route: a reachable tile
    # always has a neighbor one step closer to the goal.
    fallback_states = sum(
        1
        for state, distance in enumerate(distances)
        if distance == math.inf and tile_at(desc, state) not in (Tile.HOLE, Tile.GOAL)
    )
    return PolicyBuildResult(
        policy=policy,
        build_time_ns=build_time_ns,
        states_evaluated=visited,
        successors_examined=neighbor_checks,
        diagnostics={"peak_queue_length": peak_queue, "fallback_states": fallback_states},
    )


def _check_priority(action_priority: tuple[Action, ...]) -> None:
    if sorted(action_priority) != sorted(Action):
        raise ValueError("action_priority must list every action exactly once")


def _fallback_action(
    desc: MapDesc, state: int, n: int, action_priority: tuple[Action, ...]
) -> Action:
    """Pick an action for a tile with no route to the goal, avoiding holes when possible.

    No action can reach the goal from such a tile, so this is not a good decision, only a
    deterministic one that does not walk into a hole on purpose.
    """
    for action in action_priority:
        if tile_at(desc, deterministic_move(state, action, n)) is not Tile.HOLE:
            return action
    return action_priority[0]


def _reverse_bfs(desc: MapDesc) -> tuple[list[float], int, int, int]:
    """Run BFS outward from the goal; return distances, states dequeued, checks, peak queue."""
    n = map_size(desc)
    goal = find_tile(desc, Tile.GOAL)
    distances = [math.inf] * (n * n)
    distances[goal] = 0
    queue = deque([goal])
    visited = 0
    neighbor_checks = 0
    peak_queue = 1
    while queue:
        current = queue.popleft()
        visited += 1
        for action in Action:
            neighbor = deterministic_move(current, action, n)
            if neighbor == current:
                continue  # a border bump is not a neighbor
            neighbor_checks += 1
            if distances[neighbor] == math.inf and tile_at(desc, neighbor) is not Tile.HOLE:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
        peak_queue = max(peak_queue, len(queue))
    return distances, visited, neighbor_checks, peak_queue
