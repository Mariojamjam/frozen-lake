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
    PolicyBuildResult,
    Tile,
)


def bfs_distances(desc: MapDesc) -> tuple[float, ...]:
    """Return the BFS distance of every state to the goal; holes and cut-off tiles are `inf`."""
    distances, _, _, _ = _reverse_bfs(desc)
    return tuple(distances)


def build_bfs_policy(
    desc: MapDesc, action_priority: tuple[Action, ...] = DEFAULT_ACTION_PRIORITY
) -> PolicyBuildResult:
    """Build the complete BFS policy table for `desc`, timing the whole build.

    Every nonterminal tile intends the move whose deterministic target has the smallest BFS
    distance; ties go to the first action in `action_priority`. Terminal tiles (hole, goal) get
    None. A nonterminal tile with no route to the goal gets the first action in the priority
    order as a deterministic fallback, counted in `diagnostics["fallback_states"]`.

    Counters: `states_evaluated` is the number of states the search dequeued (the goal
    included), `successors_examined` the in-grid neighbors it checked. The policy extraction
    only reads the finished distance table and is not counted. Diagnostics also record the peak
    queue length.
    """
    if sorted(action_priority) != sorted(Action):
        raise ValueError("action_priority must list every action exactly once")

    start_ns = time.perf_counter_ns()
    n = map_size(desc)
    distances, visited, neighbor_checks, peak_queue = _reverse_bfs(desc)

    policy: list[Action | None] = []
    fallback_states = 0
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
            best_action = action_priority[0]
            fallback_states += 1
        policy.append(best_action)

    return PolicyBuildResult(
        policy=tuple(policy),
        build_time_ns=time.perf_counter_ns() - start_ns,
        states_evaluated=visited,
        successors_examined=neighbor_checks,
        diagnostics={"peak_queue_length": peak_queue, "fallback_states": fallback_states},
    )


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
