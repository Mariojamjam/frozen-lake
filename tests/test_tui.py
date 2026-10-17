import pytest
from textual.widgets import Static

from frozen_lake.environment import parse_map
from frozen_lake.models import Action, TerminationReason
from frozen_lake.tui.app import FrozenLakeApp

OPEN_4X4 = parse_map("SFFF\nFFFF\nFFFF\nFFFG")
HOLE_TO_THE_RIGHT = parse_map("SH\nFG")


def make_app(desc=OPEN_4X4, *, max_steps=64, success_rate=0.8, episode_seed=42):
    return FrozenLakeApp(
        desc,
        map_seed=0,
        episode_seed=episode_seed,
        max_steps=max_steps,
        success_rate=success_rate,
    )


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
        lake = app.query_one("#lake", Static)
        assert lake.region.width > 0 and lake.region.height > 0


async def test_q_quits():
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("q")
    assert app.return_code == 0
