"""The explorer and the episode form, driven by key presses without a screen.

Every check reads the application's state (the session, the policies, the settings, how the
lake was drawn), never the text, glyphs or colors on screen.
"""

import pytest
from textual.widgets import Button, Input
from typer.testing import CliRunner

import frozen_lake.cli as cli
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
    CELL_SIZES,
    MAX_QUICK_DEPTH,
    ROLLOUT_STEPS,
    FrozenLakeApp,
    LakeView,
    SettingsScreen,
    cell_size,
    compact_fits,
    compact_lines,
    lake_lines,
    render_compact_lake,
)
from frozen_lake.tui.episode_form import EpisodeForm, episode_command

OPEN_4X4 = parse_map("SFFF\nFFFF\nFFFF\nFFFG")
RUNNING_MAP = parse_map("SFFF\nFHFH\nFFFH\nHFFG")
HOLE_TO_THE_RIGHT = parse_map("SH\nFG")
SCREEN = (100, 30)


def make_app(desc=OPEN_4X4, *, max_steps=64, success_rate=0.8, episode_seed=42, **kwargs):
    return FrozenLakeApp(
        desc,
        map_seed=0,
        episode_seed=episode_seed,
        max_steps=max_steps,
        success_rate=success_rate,
        **kwargs,
    )


def make_varying_app(n=6, **kwargs):
    def make_map(seed):
        return generate_map(n, 0.8, seed=seed)

    return FrozenLakeApp(
        make_map(42),
        map_seed=42,
        episode_seed=7,
        max_steps=4 * n * n,
        success_rate=0.8,
        make_map=make_map,
        **kwargs,
    )


def expectimax_policy(app):
    model = extract_transitions(
        create_environment(app.desc, max_steps=app.max_steps, success_rate=app.success_rate)
    )
    return build_expectimax_policy(
        app.desc, model, depth=app.depth, beta=app.beta, gamma=app.gamma
    ).policy


# --- real steps --------------------------------------------------------------------------------


async def test_space_takes_exactly_one_real_step():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        assert app.strategy is None and app.session.steps == 0  # starts in manual mode
        await pilot.press("space")
        assert app.session.steps == 1 and app.last_event.step == 1
        await pilot.press("space")
        assert app.session.steps == 2


async def test_arrows_choose_the_intended_action_without_stepping():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("down")
        assert app.chosen_action is Action.DOWN and app.session.steps == 0
        await pilot.press("space")
        assert app.last_event.action is Action.DOWN


async def test_no_step_after_the_episode_ends():
    app = make_app(HOLE_TO_THE_RIGHT, success_rate=1.0)
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("right", "space")
        assert app.session.termination is TerminationReason.HOLE
        await pilot.press("space")
        assert app.session.steps == 1


async def test_the_step_limit_ends_the_episode_in_a_timeout():
    app = make_app(max_steps=1)
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("left", "space")
        assert app.session.termination is TerminationReason.TIMEOUT


# --- seeds -------------------------------------------------------------------------------------


async def test_replay_restarts_with_the_same_seed_and_reproduces_the_slips():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("space", "space", "space")
        first = app.session.state
        await pilot.press("r")
        assert (app.session.steps, app.session.seed) == (0, 42)
        await pilot.press("space", "space", "space")
        assert app.session.state == first


async def test_new_slips_use_the_next_episode_seed():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("space", "n")
        assert (app.session.steps, app.session.seed) == (0, 43)


async def test_replaying_a_rollout_episode_repeats_the_same_choices():
    app = make_app(RUNNING_MAP, rollouts=10)
    async with app.run_test(size=SCREEN) as pilot:
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


async def test_a_rollout_seed_fixes_the_simulations_across_episode_seeds():
    async def first_estimates(episode_seed):
        app = make_app(RUNNING_MAP, rollouts=5, rollout_seed=3, episode_seed=episode_seed)
        async with app.run_test(size=SCREEN) as pilot:
            await pilot.press("3", "space")
            return app.last_decision.estimated_action_values

    # The first decision is taken from the start, before any slip, with the same simulations.
    assert await first_estimates(1) == await first_estimates(2)


# --- who decides -------------------------------------------------------------------------------


async def test_bfs_mode_follows_the_policy_and_restarts_the_episode():
    app = make_app(RUNNING_MAP, success_rate=1.0)
    policy = build_bfs_policy(RUNNING_MAP).policy
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("space", "1")
        assert app.strategy is StrategyId.BFS
        assert app.session.steps == 0  # one episode, one decision maker
        await pilot.press("space")
        assert app.last_event.action is policy[0]
        await pilot.press("left")  # the arrows do not override a strategy
        assert app.chosen_action is Action.RIGHT


async def test_exp_mode_follows_the_expectimax_policy():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("2")
        policy = app.builds[StrategyId.EXP].policy
        assert policy == expectimax_policy(app)
        for _ in range(5):
            if app.session.finished:
                break
            state = app.session.state
            await pilot.press("space")
            assert app.last_event.action is policy[state]


@pytest.mark.parametrize(("key", "strategy"), [("3", StrategyId.BFS_RO), ("4", StrategyId.EXP_RO)])
async def test_rollout_modes_follow_the_rollout_decision(key, strategy):
    app = make_app(RUNNING_MAP, rollouts=7, horizon=9)
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press(key)
        assert app.strategy is strategy
        state = app.session.state
        await pilot.press("space")
        decision = app.last_decision
        assert app.last_event.action is decision.action
        assert decision.base_action is app.builds[strategy.base].policy[state]
        assert decision.trajectories_evaluated == 4 * 7
        assert app._horizon() == 9


async def test_m_returns_to_manual_mode():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("1", "m")
        assert app.strategy is None


async def test_p_draws_each_base_policy_then_none():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=SCREEN) as pilot:
        view = app.query_one(LakeView)
        drawn = []
        for _ in range(3):
            await pilot.press("p")
            drawn.append(view.policy)
        assert drawn == [
            app.builds[StrategyId.BFS].policy,
            app.builds[StrategyId.EXP].policy,
            None,
        ]


# --- a new lake --------------------------------------------------------------------------------


async def test_g_moves_to_the_next_lake_with_rebuilt_policies_and_the_same_mode():
    app = make_varying_app()
    async with app.run_test(size=SCREEN) as pilot:
        first_lake = app.desc
        await pilot.press("1", "space", "g")
        assert app.map_seed == 43
        assert app.desc == generate_map(6, 0.8, seed=43) != first_lake
        assert app.session.desc == app.desc  # the real environment uses the new lake
        assert (app.session.steps, app.session.seed) == (0, 7)  # restarted, same episode seed
        assert app.strategy is StrategyId.BFS
        assert app.builds[StrategyId.BFS].policy == build_bfs_policy(app.desc).policy
        assert app.builds[StrategyId.EXP].policy == expectimax_policy(app)


async def test_a_fixed_lake_cannot_be_varied():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("g")
        assert (app.desc, app.map_seed) == (OPEN_4X4, 0)


# --- tuning N and the Expectimax depth ---------------------------------------------------------


async def test_plus_and_minus_move_n_along_the_experiment_budgets_and_restart():
    app = make_app(rollouts=ROLLOUT_STEPS[1])
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("4", "space", "plus")
        assert app.rollouts == ROLLOUT_STEPS[2]
        assert (app.session.steps, app.session.seed) == (0, 42)  # same slips, new decisions
        await pilot.press("minus", "minus", "minus")
        assert app.rollouts == ROLLOUT_STEPS[0]  # stays at the smallest budget
        await pilot.press("space")
        assert app.last_decision.trajectories_evaluated == 4 * ROLLOUT_STEPS[0]


async def test_brackets_change_the_depth_and_rebuild_only_expectimax():
    app = make_app(RUNNING_MAP, depth=2)
    async with app.run_test(size=SCREEN) as pilot:
        bfs = app.builds[StrategyId.BFS]
        await pilot.press("space", "right_square_bracket")
        assert app.depth == 3 and app.session.steps == 0
        assert app.builds[StrategyId.BFS] is bfs
        assert app.builds[StrategyId.EXP].policy == expectimax_policy(app)
        for _ in range(5):
            await pilot.press("left_square_bracket")
        assert app.depth == 1
        for _ in range(MAX_QUICK_DEPTH + 2):
            await pilot.press("right_square_bracket")
        assert app.depth == MAX_QUICK_DEPTH


async def test_the_settings_window_applies_typed_values_and_restarts_on_the_same_seed():
    app = make_app(RUNNING_MAP)
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("4", "space", "s")
        assert isinstance(app.screen, SettingsScreen)
        for field, value in {
            "rollouts": "333",
            "depth": "3",
            "beta": "0.95",
            "horizon": "7",
        }.items():
            app.screen.query_one(f"#{field}", Input).value = value
        await pilot.press("enter")
        assert not isinstance(app.screen, SettingsScreen)
        assert (app.rollouts, app.depth, app.beta, app.horizon) == (333, 3, 0.95, 7)
        assert (app.session.steps, app.session.seed) == (0, 42)
        assert app.builds[StrategyId.EXP].policy == expectimax_policy(app)


async def test_invalid_settings_cannot_be_applied_and_cancelling_changes_nothing():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("s")
        app.screen.query_one("#beta", Input).value = "1"
        await pilot.pause()
        assert app.screen.query_one("#apply", Button).disabled
        await pilot.press("enter")
        assert isinstance(app.screen, SettingsScreen)  # not applied
        await pilot.press("escape")
        assert not isinstance(app.screen, SettingsScreen)
        assert (app.rollouts, app.depth, app.beta, app.horizon) == (25, 6, 0.9, None)


async def test_space_and_arrows_do_not_play_while_the_settings_are_open():
    app = make_app()
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("s", "space", "left", "space")
        assert app.session.steps == 0


# --- autoplay ----------------------------------------------------------------------------------


async def test_autoplay_runs_until_the_episode_ends_and_then_stops():
    app = make_app(max_steps=5)
    app.AUTOPLAY_INTERVAL = 0.01
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("1", "a")  # no Space: every step comes from the timer
        for _ in range(100):
            if app.session.finished:
                break
            await pilot.pause(0.02)
        await pilot.pause(0.05)
        assert app.session.finished and app.session.steps >= 1
        assert app.autoplay_timer is None


async def test_autoplay_can_be_paused():
    app = make_app(max_steps=500)
    app.AUTOPLAY_INTERVAL = 0.01
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("1", "a")
        await pilot.pause(0.05)
        await pilot.press("a")
        assert app.autoplay_timer is None
        paused_at = app.session.steps
        await pilot.pause(0.05)
        assert app.session.steps == paused_at


async def test_autoplay_needs_a_strategy_and_stops_when_the_mode_changes():
    app = make_app(max_steps=500)
    app.AUTOPLAY_INTERVAL = 0.05
    async with app.run_test(size=SCREEN) as pilot:
        await pilot.press("a")
        assert app.autoplay_timer is None  # manual mode has nothing to play
        await pilot.press("1", "a", "2")
        assert app.autoplay_timer is None and app.session.steps == 0


# --- the lake is never cropped -----------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 10, 12, 24])
def test_cell_size_is_the_largest_that_fits(n):
    for width, height in [(30, 12), (52, 18), (120, 50), (8, 4)]:
        cell = cell_size(n, width, height)
        fits = n * cell[0] <= width and lake_lines(n, cell[1]) <= height
        assert fits or cell == CELL_SIZES[-1]
        larger = CELL_SIZES[: CELL_SIZES.index(cell)]
        assert not any(n * w <= width and lake_lines(n, h) <= height for w, h in larger)


@pytest.mark.parametrize("n", [6, 20, 24, 40])
@pytest.mark.parametrize("window", [(160, 50), (100, 30), (70, 20)])
async def test_the_lake_is_drawn_whole_or_not_at_all(n, window):
    # The goal sits in the bottom-right corner, so a cropped lake would hide it.
    app = make_varying_app(n)
    async with app.run_test(size=window) as pilot:
        await pilot.pause()
        view = app.query_one(LakeView)
        width, height = view.size.width, view.size.height
        lines = view.render().plain.split("\n")
        cell = cell_size(n, width, height)
        if n * cell[0] <= width and lake_lines(n, cell[1]) <= height:
            assert view.mode == "tiles"
        elif compact_fits(n, width, height):
            assert view.mode == "compact"
        else:
            assert view.mode == "too-small"  # a message instead of a cropped lake
            return
        assert len(lines) <= height and all(len(line) <= width for line in lines)


@pytest.mark.parametrize("n", [5, 20, 24])
def test_the_compact_view_draws_every_tile_row_and_column(n):
    lines = render_compact_lake(generate_map(n, 0.9, seed=1), agent=0).plain.split("\n")
    assert len(lines) == compact_lines(n)  # two tile rows per line, the last one included
    assert all(len(line) == n for line in lines)


async def test_resizing_redraws_the_lake_at_the_size_that_fits():
    app = make_varying_app(24)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(LakeView)
        assert view.mode == "compact"
        await pilot.resize_terminal(160, 50)
        await pilot.pause()
        view.render()
        assert view.mode == "tiles"


# --- the episode form --------------------------------------------------------------------------


def form_defaults():
    return dict(
        strategy=StrategyId.BFS, quiet=False, size=4, map_seed=42, episode_seed=42,
        frozen_probability=0.9, success_rate=0.8, max_steps=None, depth=6, beta=0.9,
        rollouts=25, horizon=None, rollout_seed=None,
    )  # fmt: skip


@pytest.mark.parametrize(
    "values",
    [
        {"strategy": StrategyId.EXP_RO, "size": 8, "rollouts": 100, "beta": 0.95, "quiet": True},
        {"strategy": StrategyId.EXP, "depth": 3, "max_steps": 20, "episode_seed": 7},
        {"strategy": StrategyId.BFS_RO, "horizon": 12, "rollout_seed": 5, "success_rate": 1.0},
        {},
    ],
    ids=["exp-ro", "exp", "bfs-ro", "defaults"],
)
def test_the_equivalent_command_plays_exactly_the_chosen_settings(monkeypatch, values):
    defaults = form_defaults()
    chosen = {**defaults, **values}
    played = []
    monkeypatch.setattr(cli, "_play_episode", lambda **settings: played.append(settings))
    args = episode_command(chosen, defaults).split()[1:]  # without the program name
    if args == ["episode"]:
        args.append("-q")  # with no option at all the CLI would open the form in a terminal
        chosen["quiet"] = True
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    (settings,) = played
    assert {**settings, "strategy": StrategyId(settings["strategy"])} == chosen


async def test_the_form_enables_only_the_settings_of_the_chosen_strategy():
    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        enabled = {}
        for strategy in StrategyId:
            await pilot.click(f"#{strategy.name}")
            enabled[strategy] = (
                not form.query_one("#depth", Input).disabled,
                not form.query_one("#rollouts", Input).disabled,
            )
    assert enabled == {
        StrategyId.BFS: (False, False),
        StrategyId.EXP: (True, False),
        StrategyId.BFS_RO: (False, True),
        StrategyId.EXP_RO: (True, True),
    }


async def test_the_form_blocks_invalid_values_and_returns_the_chosen_ones():
    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.click("#EXP_RO")
        form.query_one("#size", Input).value = "1"
        await pilot.pause()
        assert form.query_one("#run", Button).disabled
        await pilot.press("enter")
        assert form.return_value is None  # still open
        form.query_one("#size", Input).value = "6"
        form.query_one("#rollouts", Input).value = "40"
        await pilot.pause()
        await pilot.click("#run")
    assert form.return_value == {**form_defaults(), "strategy": StrategyId.EXP_RO, "size": 6,
                                 "rollouts": 40}  # fmt: skip


async def test_enter_plays_from_anywhere_and_escape_cancels():
    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.press("enter")  # the focus starts on the strategy buttons
    assert form.return_value == form_defaults()

    form = EpisodeForm(form_defaults())
    async with form.run_test(size=(90, 32)) as pilot:
        await pilot.press("escape")
    assert form.return_value is None
