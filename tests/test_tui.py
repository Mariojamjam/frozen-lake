import pytest
from textual.widgets import Static

from frozen_lake.environment import (
    create_environment,
    extract_transitions,
    generate_map,
    parse_map,
)
from frozen_lake.models import Action, StrategyId, TerminationReason
from frozen_lake.policies.bfs import build_bfs_policy
from frozen_lake.policies.expectimax import build_expectimax_policy
from frozen_lake.tui.app import (
    ARROWS,
    CELL_SIZES,
    FrozenLakeApp,
    LakeView,
    cell_size,
    lake_lines,
    render_lake,
)

OPEN_4X4 = parse_map("SFFF\nFFFF\nFFFF\nFFFG")
RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
HOLE_TO_THE_RIGHT = parse_map("SH\nFG")


def make_app(desc=OPEN_4X4, *, max_steps=64, success_rate=0.8, episode_seed=42, **kwargs):
    return FrozenLakeApp(
        desc,
        map_seed=0,
        episode_seed=episode_seed,
        max_steps=max_steps,
        success_rate=success_rate,
        **kwargs,
    )


def app_model(app):
    env = create_environment(app.desc, max_steps=app.max_steps, success_rate=app.success_rate)
    return extract_transitions(env)


def lake_text(app) -> str:
    return app.query_one(LakeView).render().plain


def status(app) -> str:
    return str(app.query_one("#status", Static).content)


async def test_initial_screen_shows_the_manual_placeholder_and_start_state():
    app = make_app()
    async with app.run_test(size=(100, 30)):
        assert app.session.steps == 0
        assert "MANUAL" in status(app)
        assert "running" in status(app)


async def test_space_takes_exactly_one_real_step():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space")
        assert app.session.steps == 1
        assert app.last_event is not None and app.last_event.step == 1
        await pilot.press("space")
        assert app.session.steps == 2


async def test_arrows_choose_the_intended_action_without_stepping():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("down")
        assert app.chosen_action is Action.DOWN
        assert app.session.steps == 0
        await pilot.press("space")
        assert app.last_event.action is Action.DOWN


async def test_no_step_after_the_episode_ends():
    app = make_app(HOLE_TO_THE_RIGHT, success_rate=1.0)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("right", "space")
        assert app.session.termination is TerminationReason.HOLE
        assert "HOLE" in status(app)
        await pilot.press("space")
        assert app.session.steps == 1


async def test_timeout_is_shown():
    app = make_app(max_steps=1)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("left", "space")
        assert app.session.termination is TerminationReason.TIMEOUT
        assert "TIMEOUT" in status(app)


async def test_replay_restarts_with_the_same_seed_and_reproduces_the_slips():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space", "space", "space")
        first = app.session.state
        await pilot.press("r")
        assert (app.session.steps, app.session.seed) == (0, 42)
        await pilot.press("space", "space", "space")
        assert app.session.state == first


async def test_new_episode_uses_the_next_seed():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space", "n")
        assert (app.session.steps, app.session.seed) == (0, 43)


@pytest.mark.parametrize("size", [(100, 30), (60, 20)])
async def test_layout_fits_common_and_small_terminals(size):
    app = make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.press("space")
        lake = app.query_one(LakeView)
        assert lake.region.width > 0 and lake.region.height > 0


async def test_q_quits():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("q")
    assert app.return_code == 0


async def test_bfs_mode_follows_the_policy_and_restarts_the_episode():
    app = make_app(RUNNING_MAP, success_rate=1.0)
    policy = build_bfs_policy(RUNNING_MAP).policy
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space")  # one manual step
        await pilot.press("1")
        assert app.strategy is StrategyId.BFS
        assert app.session.steps == 0  # switching mode restarted the episode
        assert "BFS policy" in status(app)
        await pilot.press("space")
        assert app.last_event.action is policy[0]


async def test_bfs_mode_reaches_the_goal_without_slipping():
    app = make_app(RUNNING_MAP, success_rate=1.0)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", *["space"] * 6)
        assert app.session.termination is TerminationReason.GOAL


async def test_arrows_do_not_override_the_policy_in_bfs_mode():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "left")
        assert app.chosen_action is Action.RIGHT  # unchanged default


async def test_m_returns_to_manual_mode():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "m")
        assert app.strategy is None
        assert "MANUAL" in status(app)


async def test_status_shows_both_recommendations_in_manual_mode():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)):
        recommends = next(line for line in status(app).splitlines() if "Recommends" in line)
        assert "BFS → RIGHT" in recommends  # from the start, BFS ties go to RIGHT
        exp_action = app.builds[StrategyId.EXP].policy[0]
        assert f"EXP {ARROWS[exp_action]} {exp_action.name}" in recommends


async def test_p_cycles_the_policy_arrows_off_bfs_exp():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.arrows is None
        assert "↓" not in lake_text(app)
        await pilot.press("p")
        assert app.arrows is StrategyId.BFS
        assert "↓" in lake_text(app)
        assert "Arrows         BFS" in status(app)
        await pilot.press("p")
        assert app.arrows is StrategyId.EXP
        await pilot.press("p")
        assert app.arrows is None


async def test_exp_mode_follows_the_expectimax_policy():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("2")
        assert app.strategy is StrategyId.EXP
        assert "EXP policy" in status(app)
        policy = app.builds[StrategyId.EXP].policy
        for _ in range(5):
            if app.session.finished:
                break
            state = app.session.state
            await pilot.press("space")
            assert app.last_event.action is policy[state]


async def test_expectimax_settings_reach_the_policy():
    app = make_app(RUNNING_MAP, depth=1, beta=0.9)
    async with app.run_test(size=(100, 30)):
        assert "depth 1" in status(app)
        expected = build_expectimax_policy(
            RUNNING_MAP, app_model(app), depth=1, beta=0.9, gamma=1.0
        ).policy
        assert app.builds[StrategyId.EXP].policy == expected


def make_varying_app(**kwargs):
    def make_map(seed):
        return generate_map(6, 0.8, seed=seed)

    return FrozenLakeApp(
        make_map(42),
        map_seed=42,
        episode_seed=7,
        max_steps=144,
        success_rate=0.8,
        make_map=make_map,
        **kwargs,
    )


async def test_g_generates_the_next_map_and_restarts_on_it():
    app = make_varying_app()
    async with app.run_test(size=(100, 30)) as pilot:
        first_lake = app.desc
        await pilot.press("space", "g")
        assert app.map_seed == 43
        assert app.desc == generate_map(6, 0.8, seed=43) != first_lake
        assert app.session.desc == app.desc  # the real environment uses the new lake
        assert app.session.steps == 0
        assert app.session.seed == 7  # the episode seed is kept
        assert "map 43" in status(app)


async def test_the_bfs_policy_is_rebuilt_for_the_new_map():
    app = make_varying_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("g")
        assert app.builds[StrategyId.BFS].policy == build_bfs_policy(app.desc).policy


async def test_new_map_keeps_the_current_mode():
    app = make_varying_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "g")
        assert app.strategy is StrategyId.BFS


async def test_a_fixed_map_cannot_be_varied():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("g")
        assert app.desc == OPEN_4X4
        assert app.map_seed == 0


@pytest.mark.parametrize("key", ["N", "n"])
async def test_letter_keys_work_in_both_cases(key):
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("space", key)
        assert (app.session.steps, app.session.seed) == (0, 43)


@pytest.mark.parametrize("n", [4, 6, 10, 12])
def test_cell_size_is_the_largest_that_fits(n):
    for width, height in [(30, 12), (52, 18), (120, 50), (8, 4)]:
        cell = cell_size(n, width, height)
        fits = n * cell[0] <= width and lake_lines(n, cell[1]) <= height
        assert fits or cell == CELL_SIZES[-1]
        larger = CELL_SIZES[: CELL_SIZES.index(cell)]
        assert not any(n * w <= width and lake_lines(n, h) <= height for w, h in larger)


@pytest.mark.parametrize("size", [(120, 40), (100, 30), (80, 24), (70, 20)])
async def test_the_whole_lake_fits_its_panel_at_any_window_size(size):
    app = make_varying_app()  # a 6×6 lake
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        lake = app.query_one(LakeView)
        lines = lake.render().plain.split("\n")
        assert len(lines) <= lake.size.height
        assert all(len(line) <= lake.size.width for line in lines)
        # The whole lake is drawn, not cropped: n tile rows and n tile columns.
        cell = cell_size(6, lake.size.width, lake.size.height)
        assert len(lines) >= 6 * cell[1]
        assert max(len(line) for line in lines) >= 6 * cell[0]


async def test_resizing_redraws_the_lake_with_new_tile_sizes():
    app = make_varying_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        big = len(lake_text(app).split("\n"))
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        small = len(lake_text(app).split("\n"))
        assert small < big


@pytest.mark.parametrize("cell", CELL_SIZES)
def test_glyphs_sit_at_the_exact_center_of_their_tile(cell):
    width, height = cell
    lake = parse_map("SH\nFG")  # the hole "○" is the tile at row 0, column 1
    lines = render_lake(lake, agent=0, previous=None, cell=cell).plain.split("\n")
    assert len(lines) == lake_lines(2, height)

    (glyph_line,) = [i for i, line in enumerate(lines) if "○" in line]
    # The tile's own lines run from its first full line to its last; edge lines ("▀") are
    # shared halves. Count, in half-lines, how much of the tile lies above and below the glyph.
    shift = 1 if height % 2 == 0 else 0
    tile_top, tile_bottom = shift, shift + 2 * height  # half-lines covered by tile row 0
    above = 2 * glyph_line - tile_top
    below = tile_bottom - (2 * glyph_line + 2)
    assert above == below

    column = lines[glyph_line].index("○")
    assert column - width == width // 2  # middle column of the second tile


def test_tiles_of_even_height_get_one_extra_line_for_the_shared_edges():
    assert lake_lines(6, 3) == 18
    assert lake_lines(6, 2) == 13
