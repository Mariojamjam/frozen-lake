import copy
import random
from collections import Counter

import pytest
from gymnasium.envs.toy_text.utils import categorical_sample

from frozen_lake.environment import (
    create_environment,
    deterministic_destination,
    extract_transitions,
    find_tile,
    generate_map,
    parse_map,
    sample_transition,
    tile_at,
    to_coordinates,
    to_state,
    validate_map,
    validate_transition_model,
)
from frozen_lake.models import Action, Tile, Transition

SUCCESS_RATE = 0.8

LAKE_4X4 = parse_map(
    """
    SFFF
    FHFH
    FFFH
    HFFG
    """
)
NO_HOLES_4X4 = parse_map(
    """
    SFFF
    FFFF
    FFFF
    FFFG
    """
)
LAKE_10X10 = generate_map(10, 0.9, seed=2026)

MOVES = {Action.LEFT: (0, -1), Action.DOWN: (1, 0), Action.RIGHT: (0, 1), Action.UP: (-1, 0)}


def make_env(desc, *, max_steps=100, success_rate=SUCCESS_RATE):
    return create_environment(desc, max_steps=max_steps, success_rate=success_rate)


def expected_outcomes(desc, state, action, success_rate):
    """Independent oracle: the slip physics recomputed from grid geometry, not from Gymnasium."""
    n = len(desc)
    if tile_at(desc, state) in (Tile.HOLE, Tile.GOAL):
        return [(1.0, state, 0.0, True)]
    slip = (1.0 - success_rate) / 2.0
    row, col = to_coordinates(state, n)
    outcomes = []
    # Gymnasium lists the left-rotated slip, the intended move, then the right-rotated slip.
    for direction in (Action((action - 1) % 4), action, Action((action + 1) % 4)):
        d_row, d_col = MOVES[direction]
        next_row = min(max(row + d_row, 0), n - 1)  # bumping into a border means staying put
        next_col = min(max(col + d_col, 0), n - 1)
        next_state = to_state(next_row, next_col, n)
        tile = tile_at(desc, next_state)
        probability = success_rate if direction == action else slip
        if probability > 0.0:
            reward = 1.0 if tile == Tile.GOAL else 0.0
            outcomes.append((probability, next_state, reward, tile in (Tile.HOLE, Tile.GOAL)))
    return outcomes


def as_tuples(outcomes):
    return [(t.probability, t.next_state, t.reward, t.terminated) for t in outcomes]


class FixedDraw:
    """Stands in for an RNG whose next uniform draw is known."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


# --- maps -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["S"], "at least 2 rows"),
        (["SFF", "FFG"], "must be square"),
        (["SX", "FG"], "unknown tile letters"),
        (["SS", "FG"], "exactly one 'S'"),
        (["SF", "FF"], "exactly one 'G'"),
    ],
)
def test_validate_map_rejects_invalid_maps(rows, message):
    with pytest.raises(ValueError, match=message):
        validate_map(rows)


def test_validate_map_returns_an_immutable_tuple():
    assert validate_map(["SF", "FG"]) == ("SF", "FG")


def test_parse_map_ignores_indentation_and_blank_lines():
    assert LAKE_4X4 == ("SFFF", "FHFH", "FFFH", "HFFG")


def test_generate_map_is_reproducible_from_its_seed():
    assert generate_map(8, 0.9, seed=7) == generate_map(8, 0.9, seed=7)


@pytest.mark.parametrize("size", [4, 6, 10])
def test_generated_maps_start_top_left_and_end_bottom_right(size):
    desc = generate_map(size, 0.9, seed=1)
    assert len(desc) == size
    assert find_tile(desc, Tile.START) == 0
    assert find_tile(desc, Tile.GOAL) == size * size - 1


@pytest.mark.parametrize(("size", "frozen_probability"), [(1, 0.9), (4, 0.0), (4, 1.5), (4, -0.1)])
def test_generate_map_rejects_invalid_parameters(size, frozen_probability):
    with pytest.raises(ValueError):
        generate_map(size, frozen_probability, seed=0)


@pytest.mark.parametrize("n", [2, 4, 7, 10])
def test_state_and_coordinates_are_inverse_row_major(n):
    for state in range(n * n):
        row, col = to_coordinates(state, n)
        assert (row, col) == (state // n, state % n)
        assert to_state(row, col, n) == state


def test_tile_lookup():
    assert tile_at(LAKE_4X4, 0) == Tile.START
    assert tile_at(LAKE_4X4, 5) == Tile.HOLE
    assert tile_at(LAKE_4X4, 15) == Tile.GOAL
    assert find_tile(LAKE_4X4, Tile.GOAL) == 15


@pytest.mark.parametrize(
    ("state", "action", "expected"),
    [
        (5, Action.LEFT, 4),
        (5, Action.DOWN, 9),
        (5, Action.RIGHT, 6),
        (5, Action.UP, 1),
        (0, Action.LEFT, 0),  # top-left corner: left and up bump into the border
        (0, Action.UP, 0),
        (15, Action.RIGHT, 15),  # bottom-right corner: right and down bump into the border
        (15, Action.DOWN, 15),
    ],
)
def test_deterministic_destination_moves_one_tile_or_stays_at_borders(state, action, expected):
    assert deterministic_destination(LAKE_4X4, state, action) == expected


def test_deterministic_destination_is_the_intended_outcome_of_the_model():
    model = extract_transitions(make_env(LAKE_10X10))
    for state in range(len(model)):
        if tile_at(LAKE_10X10, state) in (Tile.HOLE, Tile.GOAL):
            continue
        for action in Action:
            intended = max(model[state][action], key=lambda t: t.probability)
            assert deterministic_destination(LAKE_10X10, state, action) == intended.next_state


# --- real environment -------------------------------------------------------------------------


@pytest.mark.parametrize(("max_steps", "success_rate"), [(0, 0.8), (10, 0.0), (10, 1.2)])
def test_create_environment_rejects_invalid_parameters(max_steps, success_rate):
    with pytest.raises(ValueError):
        create_environment(LAKE_4X4, max_steps=max_steps, success_rate=success_rate)


def test_create_environment_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        create_environment(LAKE_4X4, 10, 0.8)  # type: ignore[misc]


@pytest.mark.parametrize("max_steps", [3, 64, 250])
def test_episode_truncates_exactly_at_max_steps_not_the_preset_100(max_steps):
    # Always choosing LEFT keeps the agent in column 0: it can slip up or down but never reach G.
    env = make_env(NO_HOLES_4X4, max_steps=max_steps)
    env.reset(seed=0)
    for step in range(1, max_steps + 1):
        _, reward, terminated, truncated, _ = env.step(Action.LEFT)
        assert reward == 0 and not terminated
        assert truncated == (step == max_steps)


def test_same_reset_seed_and_actions_reproduce_the_same_episode():
    def trajectory(seed):
        env = make_env(LAKE_10X10, max_steps=200)
        states = [env.reset(seed=seed)[0]]
        for _ in range(50):
            state, _, terminated, truncated, _ = env.step(Action.RIGHT)
            states.append(state)
            if terminated or truncated:
                break
        return states

    assert trajectory(42) == trajectory(42)


# --- transition model -------------------------------------------------------------------------


@pytest.mark.parametrize("desc", [LAKE_4X4, NO_HOLES_4X4, LAKE_10X10], ids=["4x4", "open", "10x10"])
def test_model_matches_gymnasium_record_by_record(desc):
    env = make_env(desc)
    model = extract_transitions(env)
    table = env.unwrapped.P
    assert len(model) == len(desc) ** 2
    for state in range(len(model)):
        for action in Action:
            assert as_tuples(model[state][action]) == list(table[state][action])


@pytest.mark.parametrize("success_rate", [0.8, 0.6, 1.0])
@pytest.mark.parametrize("desc", [LAKE_4X4, LAKE_10X10], ids=["4x4", "10x10"])
def test_model_matches_independent_slip_geometry(desc, success_rate):
    model = extract_transitions(make_env(desc, success_rate=success_rate))
    for state in range(len(model)):
        for action in Action:
            expected = expected_outcomes(desc, state, action, success_rate)
            assert as_tuples(model[state][action]) == pytest.approx(expected)


def test_border_bump_keeps_duplicate_destinations_with_full_mass():
    outcomes = extract_transitions(make_env(LAKE_4X4))[0][Action.LEFT]
    mass_by_state = Counter()
    for t in outcomes:
        mass_by_state[t.next_state] += t.probability
    assert len(outcomes) == 3
    assert mass_by_state[0] == pytest.approx(0.9)
    assert mass_by_state[4] == pytest.approx(0.1)


def test_goal_reward_is_paid_once_on_entry():
    model = extract_transitions(make_env(LAKE_4X4))
    goal = find_tile(LAKE_4X4, Tile.GOAL)
    for state in range(len(model)):
        for action in Action:
            for t in model[state][action]:
                entering_goal = t.next_state == goal and state != goal
                assert t.reward == (1.0 if entering_goal else 0.0)
    assert model[goal][Action.UP] == (Transition(1.0, goal, 0.0, True),)


def test_holes_and_goal_terminate_and_frozen_tiles_do_not():
    model = extract_transitions(make_env(LAKE_4X4))
    for state in range(len(model)):
        for action in Action:
            for t in model[state][action]:
                tile = tile_at(LAKE_4X4, t.next_state)
                assert t.terminated == (tile in (Tile.HOLE, Tile.GOAL))


def test_deterministic_movement_drops_zero_probability_records():
    model = extract_transitions(make_env(LAKE_4X4, success_rate=1.0))
    assert model[0][Action.RIGHT] == (Transition(1.0, 1, 0.0, False),)
    assert all(t.probability > 0.0 for actions in model for outcomes in actions for t in outcomes)


def test_model_is_immutable():
    model = extract_transitions(make_env(LAKE_4X4))
    assert isinstance(model, tuple)
    assert isinstance(model[0], tuple)
    assert isinstance(model[0][0], tuple)


def test_extraction_rejects_a_corrupted_gymnasium_table():
    env = make_env(LAKE_4X4)
    env.unwrapped.P[0][Action.LEFT][0] = (0.5, 0, 0, False)
    with pytest.raises(ValueError, match="sum to 1.4"):
        extract_transitions(env)


VALID_OUTCOMES = (Transition(1.0, 0, 0.0, False),)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (((VALID_OUTCOMES,) * 3,), "3 actions"),
        (((), VALID_OUTCOMES, VALID_OUTCOMES, VALID_OUTCOMES), "no outcomes"),
        (((Transition(1.0, 9, 0.0, False),),) + (VALID_OUTCOMES,) * 3, "out of range"),
        (((Transition(0.5, 0, 0.0, False),),) + (VALID_OUTCOMES,) * 3, "sum to 0.5"),
    ],
    ids=["missing-action", "empty", "bad-state", "bad-sum"],
)
def test_validate_transition_model_rejects_invalid_models(model, message):
    if len(model) == 4:  # the last three cases describe the actions of one single state
        model = (model,)
    with pytest.raises(ValueError, match=message):
        validate_transition_model(model)


# --- sampler ----------------------------------------------------------------------------------


def test_sampler_returns_only_listed_outcomes():
    model = extract_transitions(make_env(LAKE_10X10))
    rng = random.Random(0)
    for state in range(len(model)):
        for action in Action:
            for _ in range(20):
                assert sample_transition(model, state, action, rng) in model[state][action]


def test_sampler_frequencies_match_probabilities():
    # From state 6, RIGHT aims at the hole 7 (0.8) and slips to 2 or 10 (0.1 each).
    model = extract_transitions(make_env(LAKE_4X4))
    draws = 100_000
    rng = random.Random(2026)
    counts = Counter(
        sample_transition(model, 6, Action.RIGHT, rng).next_state for _ in range(draws)
    )
    for next_state, probability in {7: 0.8, 2: 0.1, 10: 0.1}.items():
        std = (probability * (1 - probability) / draws) ** 0.5
        assert counts[next_state] / draws == pytest.approx(probability, abs=5 * std)


def test_sampler_is_reproducible_from_its_seed():
    model = extract_transitions(make_env(LAKE_4X4))

    def sequence(seed):
        rng = random.Random(seed)
        return [sample_transition(model, 6, Action.RIGHT, rng) for _ in range(1_000)]

    assert sequence(7) == sequence(7)


@pytest.mark.parametrize("draw", [0.0, 0.05, 0.1, 0.5, 0.899, 0.9, 0.95, 0.999999])
def test_sampler_picks_the_same_outcome_as_gymnasium_for_the_same_draw(draw):
    model = extract_transitions(make_env(LAKE_4X4))
    outcomes = model[6][Action.RIGHT]
    gym_index = categorical_sample([t.probability for t in outcomes], FixedDraw(draw))
    assert sample_transition(model, 6, Action.RIGHT, FixedDraw(draw)) == outcomes[gym_index]


def test_sampler_falls_back_to_last_outcome_when_rounding_leaves_a_gap():
    outcomes = (Transition(0.5, 0, 0.0, False), Transition(0.4999999999, 1, 0.0, False))
    model = ((outcomes,) * 4,)
    assert sample_transition(model, 0, Action.LEFT, FixedDraw(0.99999999999)) == outcomes[-1]


def test_sampling_never_touches_the_real_environment():
    env = make_env(LAKE_4X4)
    env.reset(seed=123)
    env.step(Action.DOWN)
    model = extract_transitions(env)
    state_before = env.unwrapped.s
    rng_before = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)

    rng = random.Random(0)
    for _ in range(100_000):
        sample_transition(model, 6, Action.RIGHT, rng)

    assert env.unwrapped.s == state_before
    assert env.unwrapped.np_random.bit_generator.state == rng_before
