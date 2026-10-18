"""Helpers shared by the base policies."""

from frozen_lake.environment import MapDesc, deterministic_move, tile_at
from frozen_lake.models import Action, Tile


def fallback_action(
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
