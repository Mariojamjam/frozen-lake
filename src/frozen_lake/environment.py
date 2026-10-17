import math
import random
from collections import Counter

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from frozen_lake.models import Action, StateId, Tile, Transition, TransitionModel

# Float sums of probabilities may differ from 1 by rounding error; never compare them with ==.
PROBABILITY_TOLERANCE = 1e-9

# Rows of tile letters, top to bottom, e.g. ("SFFF", "FHFH", "FFFH", "HFFG").
type MapDesc = tuple[str, ...]

# (row, col) displacement of each intended move.
MOVES = {Action.LEFT: (0, -1), Action.DOWN: (1, 0), Action.RIGHT: (0, 1), Action.UP: (-1, 0)}


def validate_map(rows: list[str] | tuple[str, ...]) -> MapDesc:
    """Return `rows` as a canonical MapDesc, or raise ValueError if it is not a valid map."""
    desc = tuple(rows)
    n = len(desc)
    if n < 2:
        raise ValueError(f"map must have at least 2 rows, got {n}")
    if any(len(row) != n for row in desc):
        raise ValueError(f"map must be square ({n}x{n}); row lengths are {[len(r) for r in desc]}")

    counts = Counter("".join(desc))
    unknown = set(counts) - set(Tile)
    if unknown:
        raise ValueError(f"unknown tile letters {sorted(unknown)}; allowed are S, F, H, G")
    for tile in (Tile.START, Tile.GOAL):
        if counts[tile] != 1:
            raise ValueError(f"map must contain exactly one '{tile}', found {counts[tile]}")
    return desc


def parse_map(text: str) -> MapDesc:
    """Parse a map written one row per line; blank lines and surrounding spaces are ignored."""
    return validate_map([line.strip() for line in text.splitlines() if line.strip()])


def generate_map(size: int, frozen_probability: float, seed: int) -> MapDesc:
    """Generate a reproducible random map with a deterministic path from start to goal.

    Delegates to Gymnasium, which places S top-left and G bottom-right and redraws until a
    hole-free path exists. That path ignores slipping, so it says nothing about how likely
    an agent is to actually arrive.
    """
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    if not 0.0 < frozen_probability <= 1.0:
        raise ValueError(f"frozen_probability must be in (0, 1], got {frozen_probability}")
    return validate_map(generate_random_map(size=size, p=frozen_probability, seed=seed))


def create_environment(desc: MapDesc, *, max_steps: int, success_rate: float) -> gym.Env:
    """Create the real slippery FrozenLake-v1 environment in which measured episodes run."""
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    if not 0.0 < success_rate <= 1.0:
        raise ValueError(f"success_rate must be in (0, 1], got {success_rate}")
    return gym.make(
        "FrozenLake-v1",
        desc=list(desc),
        is_slippery=True,
        success_rate=success_rate,
        # Goal, hole, frozen. Pinned so a library default change cannot alter the experiment.
        reward_schedule=(1, 0, 0),
        # The registered preset silently caps every map at 100 steps; always set it per map.
        max_episode_steps=max_steps,
    )


def extract_transitions(env: gym.Env) -> TransitionModel:
    """Convert the environment's transition table into an immutable, validated TransitionModel.

    Records are kept as Gymnasium lists them: one per slip direction, so bumping into a border
    can yield two records with the same next_state, and terminal tiles keep their
    (1.0, s, 0, True) self-loop. Zero-probability records (success_rate=1.0) are dropped
    because they are impossible outcomes.
    """
    table = env.unwrapped.P
    if sorted(table) != list(range(len(table))):
        raise ValueError("transition table states must be numbered 0..n-1")
    for state, actions in table.items():
        if sorted(actions) != sorted(Action):
            raise ValueError(f"state {state} must define exactly the actions {list(Action)}")

    model = tuple(
        tuple(
            tuple(
                Transition(float(probability), int(next_state), float(reward), bool(terminated))
                for probability, next_state, reward, terminated in table[state][action]
                if probability > 0.0
            )
            for action in Action
        )
        for state in range(len(table))
    )
    validate_transition_model(model)
    return model


def validate_transition_model(model: TransitionModel) -> None:
    """Raise ValueError unless every state-action lists a valid probability distribution."""
    n_states = len(model)
    for state, actions in enumerate(model):
        if len(actions) != len(Action):
            raise ValueError(f"state {state} has {len(actions)} actions, expected {len(Action)}")
        for action, outcomes in zip(Action, actions, strict=True):
            if not outcomes:
                raise ValueError(f"state {state}, {action.name}: no outcomes")
            for t in outcomes:
                if not 0 <= t.next_state < n_states:
                    raise ValueError(
                        f"state {state}, {action.name}: next_state {t.next_state} out of range"
                    )
            total = math.fsum(t.probability for t in outcomes)
            if abs(total - 1.0) > PROBABILITY_TOLERANCE:
                raise ValueError(
                    f"state {state}, {action.name}: probabilities sum to {total}, not 1"
                )


def sample_transition(
    model: TransitionModel, state: StateId, action: Action, rng: random.Random
) -> Transition:
    """Draw one outcome of `action` in `state` from the model, never touching a real environment.

    Same cumulative-probability rule as Gymnasium: the first outcome whose running total
    exceeds a uniform draw. Simulations must pass their own `rng`, separate from the real
    episode's, so that planning never changes which slips happen in real episodes.
    """
    outcomes = model[state][action]
    draw = rng.random()
    cumulative = 0.0
    for outcome in outcomes:
        cumulative += outcome.probability
        if draw < cumulative:
            return outcome
    # Rounding can leave the total a hair below 1 and the draw above it.
    return outcomes[-1]


def map_size(desc: MapDesc) -> int:
    return len(desc)


def to_state(row: int, col: int, n: int) -> StateId:
    return row * n + col


def to_coordinates(state: StateId, n: int) -> tuple[int, int]:
    """Return (row, col) of a row-major state on an n x n grid."""
    return divmod(state, n)


def tile_at(desc: MapDesc, state: StateId) -> Tile:
    row, col = to_coordinates(state, map_size(desc))
    return Tile(desc[row][col])


def find_tile(desc: MapDesc, tile: Tile) -> StateId:
    """Return the state of the first `tile` in row-major order (S and G are unique)."""
    n = map_size(desc)
    flat = "".join(desc)
    index = flat.find(tile)
    if index < 0:
        raise ValueError(f"map has no '{tile}' tile")
    return to_state(*divmod(index, n), n)


def deterministic_destination(desc: MapDesc, state: StateId, action: Action) -> StateId:
    """Return where `action` leads from `state` without slipping; borders keep the agent put."""
    n = map_size(desc)
    row, col = to_coordinates(state, n)
    d_row, d_col = MOVES[action]
    return to_state(min(max(row + d_row, 0), n - 1), min(max(col + d_col, 0), n - 1), n)
