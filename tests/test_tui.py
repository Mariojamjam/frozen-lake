import pytest
from textual.widgets import DataTable, Static

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


def lake_subtitle(app) -> str:
    return str(app.query_one("#lake-panel").border_subtitle)


def estimate_rows(app) -> list[list[str]]:
    table = app.query_one("#estimates", DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


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
        assert "arrows BFS" in lake_subtitle(app)
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
        assert "depth 1" in lake_subtitle(app)
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


# --- rollout modes, estimates and autoplay ---------------------------------------------------


@pytest.mark.parametrize(("key", "strategy"), [("3", StrategyId.BFS_RO), ("4", StrategyId.EXP_RO)])
async def test_rollout_modes_follow_the_rollout_decision(key, strategy):
    app = make_app(RUNNING_MAP, rollouts=10)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(key)
        assert app.strategy is strategy
        assert f"{strategy} (Rollout over {strategy.base})" in status(app)
        state = app.session.state
        await pilot.press("space")
        decision = app.last_decision
        assert app.last_event.action is decision.action
        assert decision.base_action is app.builds[strategy.base].policy[state]
        assert decision.estimated_action_values is not None
        assert decision.trajectories_evaluated == 4 * 10


async def test_the_estimates_table_shows_each_action_with_base_and_chosen_marks():
    app = make_app(RUNNING_MAP, rollouts=10)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("3", "space")
        rows = estimate_rows(app)
        assert len(rows) == 4
        decision = app.last_decision
        marks = {row[0].split()[1]: row[2] for row in rows}
        assert "chosen" in marks[decision.action.name]
        assert "base" in marks[decision.base_action.name]


async def test_the_estimates_table_is_not_applicable_outside_rollout_modes():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "space")
        assert estimate_rows(app) == [["—", "Rollout modes only (3, 4)", ""]]


async def test_replaying_a_rollout_episode_repeats_the_same_choices():
    app = make_app(RUNNING_MAP, rollouts=10)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("3")
        first = []
        for _ in range(4):
            if app.session.finished:
                break
            await pilot.press("space")
            first.append(app.last_event.action)
        await pilot.press("r")
        second = []
        for _ in range(len(first)):
            await pilot.press("space")
            second.append(app.last_event.action)
        assert first == second


async def test_rollout_settings_reach_the_explorer():
    app = make_app(RUNNING_MAP, rollouts=7, horizon=9)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("4")
        assert "N 7 · horizon 9" in status(app)


async def test_autoplay_runs_until_the_episode_ends_and_then_stops():
    app = make_app(OPEN_4X4, max_steps=5)
    app.AUTOPLAY_INTERVAL = 0.01
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "a")  # no Space pressed: every step comes from the timer
        for _ in range(100):
            if app.session.finished:
                break
            await pilot.pause(0.02)
        await pilot.pause(0.05)
        assert app.session.finished
        assert app.session.steps >= 1
        assert app.autoplay_timer is None


async def test_autoplay_can_be_paused():
    app = make_app(OPEN_4X4, max_steps=500)
    app.AUTOPLAY_INTERVAL = 0.01
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "a")
        await pilot.pause(0.05)
        await pilot.press("a")
        assert app.autoplay_timer is None
        paused_at = app.session.steps
        await pilot.pause(0.05)
        assert app.session.steps == paused_at


async def test_autoplay_needs_a_strategy():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("a")
        assert app.autoplay_timer is None


async def test_switching_mode_stops_autoplay():
    app = make_app(OPEN_4X4, max_steps=500)
    app.AUTOPLAY_INTERVAL = 0.05
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("1", "a", "2")
        assert app.autoplay_timer is None
        assert app.session.steps == 0


# --- tuning N and the Expectimax depth inside the explorer -------------------------------------


async def test_plus_and_minus_step_n_along_the_experiment_budgets_and_restart():
    app = make_app(rollouts=25)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("4", "space")
        await pilot.press("plus")
        assert app.rollouts == 50
        assert app.session.steps == 0 and app.session.seed == 42  # same seed, new decisions
        await pilot.press("minus", "minus")
        assert app.rollouts == 10
        await pilot.press("minus")  # already the smallest step
        assert app.rollouts == 10
        assert "N 10" in status(app)


async def test_n_off_the_budget_steps_moves_to_the_nearest_one():
    app = make_app(rollouts=30)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("plus")
        assert app.rollouts == 50
    app = make_app(rollouts=30)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("minus")
        assert app.rollouts == 25


async def test_brackets_change_the_depth_and_rebuild_expectimax():
    app = make_app(RUNNING_MAP, depth=6)
    async with app.run_test(size=(100, 30)) as pilot:
        bfs_build = app.builds[StrategyId.BFS]
        await pilot.press("left_square_bracket", "left_square_bracket")
        assert app.depth == 4
        assert app.builds[StrategyId.BFS] is bfs_build  # BFS does not depend on the depth
        assert "EXP depth 4" in lake_subtitle(app)
        model = app_model(app)
        expected = build_expectimax_policy(app.desc, model, depth=4, beta=0.9, gamma=1.0)
        assert app.builds[StrategyId.EXP].policy == expected.policy
        await pilot.press("right_square_bracket")
        assert app.depth == 5


async def test_the_depth_keys_stop_at_one():
    app = make_app(depth=1)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("left_square_bracket")
        assert app.depth == 1


async def test_settings_apply_typed_values():
    from textual.widgets import Input

    from frozen_lake.tui.app import SettingsScreen

    app = make_app(RUNNING_MAP)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("s")
        assert isinstance(app.screen, SettingsScreen)
        app.screen.query_one("#rollouts", Input).value = "333"
        app.screen.query_one("#depth", Input).value = "3"
        app.screen.query_one("#beta", Input).value = "0.95"
        app.screen.query_one("#horizon", Input).value = "7"
        await pilot.press("enter")
        assert not isinstance(app.screen, SettingsScreen)
        assert (app.rollouts, app.depth, app.beta, app.horizon) == (333, 3, 0.95, 7)
        assert "N 333 · horizon 7" in status(app)
        assert "EXP depth 3 · beta 0.95" in lake_subtitle(app)


async def test_invalid_settings_cannot_be_applied_and_cancel_keeps_everything():
    from textual.widgets import Button, Input

    from frozen_lake.tui.app import SettingsScreen

    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("s")
        app.screen.query_one("#beta", Input).value = "1"
        await pilot.pause()
        assert app.screen.query_one("#apply", Button).disabled
        assert "beta" in str(app.screen.query_one("#problems", Static).content)
        await pilot.press("enter")
        assert isinstance(app.screen, SettingsScreen)  # not applied
        await pilot.press("escape")
        assert not isinstance(app.screen, SettingsScreen)
        assert app.beta == 0.9


async def test_space_and_arrows_do_not_play_while_the_settings_are_open():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("s", "space", "left", "space")
        assert app.session.steps == 0


async def test_a_rollout_seed_fixes_the_simulations_across_episode_seeds():
    def first_estimates(episode_seed):
        return make_app(RUNNING_MAP, rollouts=5, rollout_seed=3, episode_seed=episode_seed)

    first, second = first_estimates(1), first_estimates(2)
    async with first.run_test(size=(100, 30)) as pilot:
        await pilot.press("3", "space")
        one = first.last_decision.estimated_action_values
    async with second.run_test(size=(100, 30)) as pilot:
        await pilot.press("3", "space")
        two = second.last_decision.estimated_action_values
    assert one == two  # the first decision is taken from the start, with the same simulations


# --- the episode form ----------------------------------------------------------------------------


def form_defaults():
    return dict(
        strategy=StrategyId.BFS, quiet=False, size=4, map_seed=42, episode_seed=42,
        frozen_probability=0.9, success_rate=0.8, max_steps=None, depth=6, beta=0.9,
        rollouts=25, horizon=None, rollout_seed=None,
    )  # fmt: skip


def test_the_equivalent_command_lists_only_what_changed_and_matters():
    from frozen_lake.tui.episode_form import episode_command

    defaults = form_defaults()
    assert episode_command(defaults, defaults) == "frozen-lake episode"
    changed = {**defaults, "strategy": StrategyId.EXP_RO, "size": 8, "rollouts": 100, "quiet": True}
    assert episode_command(changed, defaults) == "frozen-lake episode -S EXP-RO -s 8 -n 100 -q"
    # BFS ignores the Expectimax and Rollout settings, so they are left out.
    ignored = {**defaults, "depth": 3, "rollouts": 100}
    assert episode_command(ignored, defaults) == "frozen-lake episode"
    assert episode_command({**defaults, "beta": 0.95, "strategy": StrategyId.EXP}, defaults) == (
        "frozen-lake episode -S EXP -b 0.95"
    )


async def test_the_form_enables_only_the_settings_of_the_chosen_strategy():
    from textual.widgets import Input

    from frozen_lake.tui.episode_form import EpisodeForm

    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        assert form.query_one("#depth", Input).disabled
        assert form.query_one("#rollouts", Input).disabled
        await pilot.click("#EXP")
        assert not form.query_one("#depth", Input).disabled
        assert form.query_one("#rollouts", Input).disabled
        await pilot.click("#BFS_RO")
        assert form.query_one("#depth", Input).disabled
        assert not form.query_one("#rollouts", Input).disabled


async def test_the_form_blocks_invalid_values_and_returns_the_chosen_ones():
    from textual.widgets import Button, Input

    from frozen_lake.tui.episode_form import EpisodeForm

    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.click("#EXP_RO")
        form.query_one("#size", Input).value = "1"
        await pilot.pause()
        assert form.query_one("#run", Button).disabled
        assert "Lake size" in str(form.query_one("#problems", Static).content)
        form.query_one("#size", Input).value = "6"
        form.query_one("#rollouts", Input).value = "40"
        await pilot.pause()
        assert "-S EXP-RO -s 6 -n 40" in str(form.query_one("#command", Static).content)
        await pilot.click("#run")
    chosen = form.return_value
    assert chosen["strategy"] is StrategyId.EXP_RO
    assert (chosen["size"], chosen["rollouts"], chosen["horizon"]) == (6, 40, None)


async def test_escape_cancels_the_form():
    from frozen_lake.tui.episode_form import EpisodeForm

    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.press("escape")
    assert form.return_value is None


async def test_enter_plays_from_the_strategy_buttons():
    from frozen_lake.tui.episode_form import EpisodeForm

    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.press("enter")
    assert form.return_value is not None
    assert form.return_value["strategy"] is StrategyId.BFS


# --- lakes larger than the window --------------------------------------------------------------


def lake_app(n):
    return make_app(generate_map(n, 0.9, seed=42), max_steps=4 * n * n)


@pytest.mark.parametrize("n", [20, 24, 30])
async def test_a_lake_too_large_for_glyphs_is_drawn_whole_in_the_compact_view(n):
    from frozen_lake.tui.app import compact_lines
    from frozen_lake.tui.colors import AURORA

    app = lake_app(n)
    async with app.run_test(size=(100, 30)):
        view = app.query_one(LakeView)
        drawing = view.render()
        lines = drawing.plain.split("\n")
        assert len(lines) <= view.size.height  # nothing is cropped
        assert sum(line.count("▀") for line in lines) == n * compact_lines(n)
        assert "compact view" in drawing.plain
        # The goal, bottom right, keeps its color: it is the last tile of the last tile row.
        assert any(AURORA.lower() in str(span.style).lower() for span in drawing.spans)


async def test_a_lake_too_large_even_for_the_compact_view_says_so_instead_of_cropping():
    app = lake_app(40)
    async with app.run_test(size=(100, 30)):
        text = app.query_one(LakeView).render().plain
        assert "This 40×40 lake needs at least 40 × 20 characters" in text
        assert "▀" not in text


async def test_enlarging_the_window_brings_the_glyphs_back():
    app = lake_app(24)
    async with app.run_test(size=(100, 30)) as pilot:
        assert "★" not in lake_text(app)
        await pilot.resize_terminal(160, 50)
        await pilot.pause()
        assert "★" in lake_text(app)
