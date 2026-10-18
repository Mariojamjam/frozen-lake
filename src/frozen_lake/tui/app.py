"""Single-screen Textual explorer: step through real episodes by hand or with a base policy."""

from collections.abc import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from frozen_lake.environment import (
    MapDesc,
    create_environment,
    extract_transitions,
    map_size,
    tile_at,
    to_coordinates,
)
from frozen_lake.episodes import Decide, EpisodeSession, policy_decider
from frozen_lake.models import (
    Action,
    DecisionResult,
    Policy,
    StateId,
    StepEvent,
    StrategyId,
    TerminationReason,
    Tile,
)
from frozen_lake.policies import BASE_STRATEGIES, build_base_policy
from frozen_lake.rollout import rollout_decider, rollout_rng
from frozen_lake.tui.colors import (
    AMBER,
    AURORA,
    DEEP_WATER,
    FROST_THEME,
    ICE,
    ICE_MARK,
    LILAC,
    MINT,
    NIGHT,
    ORCHID,
    ROSE,
    SNOW,
    VIOLET,
)

TILE_GLYPHS = {Tile.START: "S", Tile.FROZEN: "·", Tile.HOLE: "○", Tile.GOAL: "★"}
AGENT_GLYPH = "●"
PREVIOUS_GLYPH = "∘"

END_MESSAGES = {
    TerminationReason.GOAL: ("GOAL reached", f"bold {AURORA}"),
    TerminationReason.HOLE: ("fell into a HOLE", f"bold {ROSE}"),
    TerminationReason.TIMEOUT: ("TIMEOUT (step limit reached)", f"bold {AMBER}"),
}

ARROWS = {Action.LEFT: "←", Action.DOWN: "↓", Action.RIGHT: "→", Action.UP: "↑"}

HELP = (
    "M: manual mode, where arrows choose the intended action. 1 BFS, 2 Expectimax, 3 Rollout "
    "over BFS, 4 Rollout over Expectimax: that strategy chooses. Space takes one real step; the "
    "ice may slip the agent sideways (10% each way). A plays the episode automatically. "
    "P cycles the policy arrows on the lake: off, BFS, Expectimax. "
    "R replays the episode seed, N tries the next episode "
    "seed (new slips, same lake), G generates the next map (new lake). Switching mode restarts "
    "the episode."
)


def tile_background(tile: Tile, row: int, col: int) -> str:
    if tile == Tile.START:
        return VIOLET
    if tile == Tile.HOLE:
        return DEEP_WATER
    if tile == Tile.GOAL:
        return AURORA
    return ICE[(row + col) % 2]  # checkered so neighboring ice tiles stay distinguishable


def tile_glyph(tile: Tile) -> tuple[str, str]:
    """The glyph drawn at a tile's center and its color."""
    if tile == Tile.START:
        return TILE_GLYPHS[tile], SNOW
    if tile == Tile.HOLE:
        return TILE_GLYPHS[tile], ROSE
    if tile == Tile.GOAL:
        return TILE_GLYPHS[tile], NIGHT
    return TILE_GLYPHS[tile], ICE_MARK


# Tile sizes in characters (width, height), largest first. Terminal cells are about twice as
# tall as they are wide, so width ≈ 2 × height keeps tiles roughly square.
CELL_SIZES = ((11, 5), (9, 4), (7, 3), (5, 2), (3, 1), (1, 1))


def lake_lines(n: int, cell_height: int) -> int:
    """Lines an n×n lake takes: tiles of even height need one extra line (see render_lake)."""
    return n * cell_height + (1 if cell_height % 2 == 0 else 0)


def cell_size(n: int, width: int, height: int) -> tuple[int, int]:
    """Largest tile size at which an n×n lake fits in width×height characters."""
    for cell_width, cell_height in CELL_SIZES:
        if n * cell_width <= width and lake_lines(n, cell_height) <= height:
            return cell_width, cell_height
    return CELL_SIZES[-1]


def render_lake(
    desc: MapDesc,
    agent: StateId,
    previous: StateId | None,
    policy: Policy | None = None,
    *,
    cell: tuple[int, int] = (7, 3),
) -> Text:
    """Draw the grid as colored tiles of `cell` characters, with a glyph at each tile's center.

    A glyph fills one whole line, so it is centered only when the tile has a middle line. A tile
    with an odd number of lines has one. A tile with an even number of lines does not: its center
    is the boundary between two lines. Those tiles are therefore shifted down by half a line, so
    that their middle falls on a whole line again. Tile edges then fall in the middle of a line
    and are drawn with "▀", colored as the tile above (foreground) over the tile below
    (background). The lake takes one extra line for this.

    With `policy`, every tile without another mark shows its action as an arrow, so the whole
    table is visible at once: one recommended move for every tile, not a single path.
    """
    n = map_size(desc)
    width, height = cell
    shift = 1 if height % 2 == 0 else 0  # in half-lines

    def tile_row_at(half_line: int) -> int | None:
        position = half_line - shift
        return position // (2 * height) if 0 <= position < 2 * n * height else None

    def glyph_line(row: int) -> int:
        return (shift + 2 * height * row + height - 1) // 2

    def colors(state: int) -> tuple[str, str, str]:
        """Background, glyph and glyph color of a tile, with the agent and policy applied."""
        row, col = divmod(state, n)
        tile = tile_at(desc, state)
        background = tile_background(tile, row, col)
        glyph, glyph_color = tile_glyph(tile)
        if state == agent:
            # Where the episode ends, the agent takes that ending's color, so a fall stands out.
            background = {Tile.HOLE: ROSE, Tile.GOAL: AURORA}.get(tile, ORCHID)
            glyph, glyph_color = AGENT_GLYPH, NIGHT
        elif state == previous and tile == Tile.FROZEN:
            glyph, glyph_color = PREVIOUS_GLYPH, ORCHID
        elif policy is not None and policy[state] is not None and tile == Tile.FROZEN:
            glyph, glyph_color = ARROWS[policy[state]], VIOLET
        return background, glyph, glyph_color

    tiles = [colors(state) for state in range(n * n)]
    # Never wrap: a wrapped line would push the rest of a tile row onto the next line.
    text = Text(no_wrap=True, overflow="crop")
    for line in range(lake_lines(n, height)):
        if line:
            text.append("\n")
        upper, lower = tile_row_at(2 * line), tile_row_at(2 * line + 1)
        for col in range(n):
            if upper is not None and upper == lower:
                background, glyph, glyph_color = tiles[upper * n + col]
                content = glyph if line == glyph_line(upper) else " "
                text.append(content.center(width), style=f"bold {glyph_color} on {background}")
            else:
                top = NIGHT if upper is None else tiles[upper * n + col][0]
                bottom = NIGHT if lower is None else tiles[lower * n + col][0]
                text.append("▀" * width, style=f"{top} on {bottom}")
    return text


class LakeView(Widget):
    """The lake, redrawn at the largest tile size that fits whenever the widget is resized."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.desc: MapDesc = ()
        self.agent: StateId = 0
        self.previous: StateId | None = None
        self.policy: Policy | None = None

    def show(
        self, desc: MapDesc, agent: StateId, previous: StateId | None, policy: Policy | None
    ) -> None:
        self.desc, self.agent, self.previous, self.policy = desc, agent, previous, policy
        self.refresh()

    def render(self) -> Text:
        if not self.desc:
            return Text()
        n = map_size(self.desc)
        available_width, available_height = self.size.width, self.size.height
        cell = cell_size(n, available_width, available_height)
        lake = render_lake(self.desc, self.agent, self.previous, self.policy, cell=cell)
        # Center the drawing in the space the tiles leave free.
        left = " " * max((available_width - n * cell[0]) // 2, 0)
        top = max((available_height - lake_lines(n, cell[1])) // 2, 0)
        centered = Text(no_wrap=True, overflow="crop")
        centered.append("\n" * top)
        for index, line in enumerate(lake.split("\n")):
            if index:
                centered.append("\n")
            centered.append(left)
            centered.append_text(line)
        return centered


def render_legend() -> Text:
    text = Text()
    entries = [
        (AGENT_GLYPH, f"bold {NIGHT} on {ORCHID}", "agent"),
        (PREVIOUS_GLYPH, f"bold {ORCHID} on {ICE[0]}", "previous"),
        (TILE_GLYPHS[Tile.START], f"bold {SNOW} on {VIOLET}", "start"),
        (TILE_GLYPHS[Tile.FROZEN], f"bold {ICE_MARK} on {ICE[0]}", "ice"),
        (TILE_GLYPHS[Tile.HOLE], f"bold {ROSE} on {DEEP_WATER}", "hole"),
        (TILE_GLYPHS[Tile.GOAL], f"bold {NIGHT} on {AURORA}", "goal"),
    ]
    for index, (glyph, style, label) in enumerate(entries):
        text.append(f" {glyph} ", style=style)
        text.append(f" {label:<10}", style=LILAC)
        if index % 3 == 2 and index < len(entries) - 1:
            text.append("\n")
    text.append("\nModes  ", style="bold")
    text.append("m manual · 1 BFS · 2 EXP\n", style=LILAC)
    text.append("       3 BFS-RO · 4 EXP-RO", style=LILAC)
    return text


class FrozenLakeApp(App):
    """Explorer for one map: actions come from the keyboard or from one of the four strategies."""

    CSS_PATH = "app.tcss"
    TITLE = "frozen-lake"
    BINDINGS = [
        Binding("left", "choose('LEFT')", "Left", show=False, priority=True),
        Binding("down", "choose('DOWN')", "Down", show=False, priority=True),
        Binding("right", "choose('RIGHT')", "Right", show=False, priority=True),
        Binding("up", "choose('UP')", "Up", show=False, priority=True),
        Binding("space", "step", "Step", priority=True),
        # Letter keys accept both cases, so Caps Lock or Shift does not silently disable them.
        # Mode keys are listed in the legend panel, which leaves the footer room for the rest.
        Binding("m,M", "set_mode('MANUAL')", "Manual", show=False),
        Binding("1", "set_mode('BFS')", "BFS", show=False),
        Binding("2", "set_mode('EXP')", "EXP", show=False),
        Binding("3", "set_mode('BFS-RO')", "BFS-RO", show=False),
        Binding("4", "set_mode('EXP-RO')", "EXP-RO", show=False),
        Binding("a,A", "toggle_autoplay", "Autoplay", key_display="a"),
        Binding("r,R", "replay", "Replay", key_display="r"),
        Binding("n,N", "new_episode", "New slips", key_display="n"),
        Binding("g,G", "new_map", "New map", key_display="g"),
        Binding("p,P", "cycle_policy_arrows", "Arrows", key_display="p"),
        Binding("question_mark", "help", "Help"),
        Binding("q,Q", "quit", "Quit", key_display="q"),
    ]

    # Seconds between steps in autoplay. Only pacing for the eye: decision times are measured
    # inside the decision functions, so this delay never enters any reported time.
    AUTOPLAY_INTERVAL = 0.25

    chosen_action: reactive[Action] = reactive(Action.RIGHT)
    # Which base policy is drawn on the lake as arrows; None draws none.
    arrows: reactive[StrategyId | None] = reactive(None)

    def __init__(
        self,
        desc: MapDesc,
        *,
        map_seed: int,
        episode_seed: int,
        max_steps: int,
        success_rate: float,
        depth: int = 6,
        beta: float = 0.9,
        gamma: float = 1.0,
        rollouts: int = 25,
        horizon: int | None = None,
        make_map: Callable[[int], MapDesc] | None = None,
    ) -> None:
        """`make_map(seed)` builds the map for a seed; without it the map cannot be changed.

        `depth`, `beta` and `gamma` configure Expectimax; `rollouts` (trajectories per action)
        and `horizon` (2 × size² by default) configure the Rollout strategies.
        """
        super().__init__()
        self.register_theme(FROST_THEME)
        self.theme = "frost"
        self.max_steps = max_steps
        self.success_rate = success_rate
        self.depth, self.beta, self.gamma = depth, beta, gamma
        self.rollouts = rollouts
        self.horizon = horizon
        self.make_map = make_map
        self.map_seed = map_seed
        self.episode_seed = episode_seed
        self.last_event: StepEvent | None = None
        self.last_decision: DecisionResult | None = None
        self.strategy: StrategyId | None = None  # None means manual: the keyboard decides
        self.autoplay_timer: Timer | None = None
        self._set_map(desc)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="lake-panel"):
                yield LakeView(id="lake")
            with Vertical(id="side"):
                yield Static(id="status")
                yield Static(render_legend(), id="legend")
        with Horizontal(id="bottom"):
            yield RichLog(id="log", max_lines=200, markup=False, wrap=True, min_width=20)
            yield DataTable(id="estimates", show_cursor=False, zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self._show_map_titles()
        self.query_one("#status", Static).border_title = "Status"
        self.query_one("#legend", Static).border_title = "Legend"
        log = self.query_one("#log", RichLog)
        log.border_title = "Event log"
        log.can_focus = False  # keep arrow keys for choosing actions, not scrolling
        table = self.query_one("#estimates", DataTable)
        table.border_title = "Rollout estimates"
        table.can_focus = False
        table.add_columns("Action", "Estimate", "")
        self._log_start()
        self.refresh_view()

    # --- actions ------------------------------------------------------------------------------

    def action_choose(self, name: str) -> None:
        if self.strategy is not None:
            self.notify(f"{self.strategy} mode: the strategy chooses. Press M for manual.")
            return
        self.chosen_action = Action[name]

    def action_set_mode(self, name: str) -> None:
        strategy = None if name == "MANUAL" else StrategyId(name)
        if strategy is self.strategy:
            return
        # One episode, one decision maker: mixing them would not be an episode of either.
        self.strategy = strategy
        self._restart(self.episode_seed)

    def action_cycle_policy_arrows(self) -> None:
        order = [None, *BASE_STRATEGIES]
        self.arrows = order[(order.index(self.arrows) + 1) % len(order)]

    def action_step(self) -> None:
        if self.session.finished:
            self._stop_autoplay()
            self.notify("Episode finished. Press R to replay or N for a new seed.")
            return
        if self.strategy is None:
            action = self.chosen_action
            self.last_decision = None
        else:
            remaining = self.session.max_steps - self.session.steps
            self.last_decision = self._decider(self.strategy)(self.session.state, remaining)
            action = self.last_decision.action
        event = self.session.step(action)
        self.last_event = event
        self._log_step(event)
        if self.session.finished:
            self._stop_autoplay()
        self.refresh_view()

    def action_toggle_autoplay(self) -> None:
        if self.autoplay_timer is not None:
            self._stop_autoplay()
            self.refresh_view()
            return
        if self.strategy is None:
            self.notify("Autoplay needs a strategy: press 1, 2, 3 or 4 first.")
            return
        if self.session.finished:
            self.notify("Episode finished. Press R to replay or N for a new seed.")
            return
        self.autoplay_timer = self.set_interval(self.AUTOPLAY_INTERVAL, self.action_step)
        self.refresh_view()

    def action_replay(self) -> None:
        self._restart(self.episode_seed)

    def action_new_episode(self) -> None:
        self._restart(self.episode_seed + 1)

    def action_new_map(self) -> None:
        if self.make_map is None:
            self.notify("This lake was given as a fixed map; there is no seed to vary.")
            return
        self.map_seed += 1
        self.session.close()
        self._set_map(self.make_map(self.map_seed))
        self._show_map_titles()
        self.query_one("#log", RichLog).write(f"— new map, seed {self.map_seed} —")
        self._restart(self.episode_seed)

    def action_help(self) -> None:
        self.notify(HELP, title="How to explore", timeout=10)

    def watch_chosen_action(self) -> None:
        if self.is_mounted:
            self.refresh_view()

    def watch_arrows(self) -> None:
        if self.is_mounted:
            self.refresh_view()

    # --- rendering ----------------------------------------------------------------------------

    def refresh_view(self) -> None:
        previous = self.last_event.state if self.last_event else None
        policy = None if self.arrows is None else self.builds[self.arrows].policy
        self.query_one("#lake", LakeView).show(self.desc, self.session.state, previous, policy)
        self.query_one("#status", Static).update(self.status_text())
        self.query_one(
            "#lake-panel", Vertical
        ).border_subtitle = (
            f"EXP depth {self.depth} · beta {self.beta:g} · arrows {self.arrows or 'off'}"
        )
        self._show_estimates()

    def status_text(self) -> Text:
        session = self.session
        n = map_size(self.desc)
        row, col = to_coordinates(session.state, n)
        text = Text()
        text.append("Status         ", style="bold")
        if session.termination is None:
            text.append("running", style="bold")
            if self.autoplay_timer is not None:
                text.append(" · autoplay", style=f"bold {ORCHID}")
            text.append("\n")
        else:
            message, style = END_MESSAGES[session.termination]
            text.append(f"{message}\n", style=style)

        text.append("Action source  ", style="bold")
        if self.strategy is None:
            text.append("MANUAL · keyboard\n", style=f"italic {LILAC}")
            text.append("Chosen action  ", style="bold")
            text.append(
                f"{ARROWS[self.chosen_action]} {self.chosen_action.name}\n",
                style=f"bold {ORCHID}",
            )
        elif self.strategy.uses_rollout:
            text.append(
                f"{self.strategy} (Rollout over {self.strategy.base})\n", style=f"bold {ORCHID}"
            )
            text.append(
                f"Rollout        N {self.rollouts} · horizon {self._horizon()}\n", style=LILAC
            )
        else:
            text.append(f"{self.strategy} policy\n", style=f"bold {ORCHID}")

        text.append("Recommends     ", style="bold")
        text.append(
            " · ".join(
                f"{strategy} {ARROWS[action]} {action.name}" if action else f"{strategy} —"
                for strategy in BASE_STRATEGIES
                for action in [self.builds[strategy].policy[session.state]]
            )
            + "\n"
        )
        text.append(
            "Builds         "
            + " · ".join(
                f"{strategy} {self.builds[strategy].build_time_ns / 1e6:.2f} ms"
                for strategy in BASE_STRATEGIES
            )
            + "\n",
            style=LILAC,
        )
        text.append(f"Seeds          map {self.map_seed} · episode {session.seed}\n")
        text.append(
            f"Step           {session.steps}/{session.max_steps} · "
            f"state {session.state} ({row},{col})\n"
        )

        decision = self.last_decision
        if decision is not None:
            text.append(f"Decision       {decision.compute_time_ns / 1e6:.2f} ms", style=LILAC)
            if decision.simulated_transitions:
                text.append(f" · {decision.simulated_transitions} sims", style=LILAC)
            text.append("\n")

        event = self.last_event
        if event is None:
            text.append("Last step      none yet\n")
        else:
            text.append(f"Last step      {event.action.name} from {event.state}")
            if decision is not None and decision.action is not decision.base_action:
                text.append(f" (overrode {decision.base_action.name})", style=f"bold {ORCHID}")
            text.append(f"\nIntended       {event.intended_state}\n")
            reached_style = MINT if event.moved_as_intended else f"bold {AMBER}"
            text.append("Reached        ")
            text.append(f"{event.next_state}", style=reached_style)
            if not event.moved_as_intended:
                text.append(" (slipped)", style=reached_style)
            text.append("\n")
        return text

    def _show_estimates(self) -> None:
        """Fill the table with the last Rollout decision's estimate for each action."""
        table = self.query_one("#estimates", DataTable)
        table.clear()
        decision = self.last_decision
        if decision is None or decision.estimated_action_values is None:
            reason = (
                "Rollout modes only (3, 4)"
                if not (self.strategy and self.strategy.uses_rollout)
                else "after the first step"
            )
            table.add_row("—", Text(reason, style=LILAC), "")
            return
        best = max(decision.estimated_action_values.values())
        for action, value in decision.estimated_action_values.items():
            marks = []
            if action is decision.base_action:
                marks.append("base")
            if action is decision.action:
                marks.append("chosen")
            style = f"bold {ORCHID}" if action is decision.action else ""
            bar = "█" * round(6 * value)
            table.add_row(
                Text(f"{ARROWS[action]} {action.name}", style=style),
                Text(f"{value:.2f} {bar}", style=MINT if value == best else ""),
                Text("+".join(marks), style=style),
            )

    # --- helpers ------------------------------------------------------------------------------

    def _horizon(self) -> int:
        n = map_size(self.desc)
        return self.horizon if self.horizon is not None else 2 * n * n

    def _decider(self, strategy: StrategyId) -> Decide:
        if strategy.uses_rollout:
            return self.rollout_deciders[strategy]
        return self.deciders[strategy]

    def _new_rollout_deciders(self) -> None:
        """Rollout deciders for this episode, each with a simulation stream from its seed.

        Recreated at every restart, so replaying an episode seed replays the same simulations
        and therefore the same choices.
        """
        self.rollout_deciders: dict[StrategyId, Decide] = {
            strategy: rollout_decider(
                self.builds[strategy.base].policy,
                self.model,
                samples=self.rollouts,
                horizon=self._horizon(),
                gamma=self.gamma,
                rng=rollout_rng(self.episode_seed),
            )
            for strategy in (StrategyId.BFS_RO, StrategyId.EXP_RO)
        }

    def _set_map(self, desc: MapDesc) -> None:
        """Make `desc` the current lake: a fresh real environment and its base policies."""
        self.desc = desc
        self.session = EpisodeSession(
            desc, max_steps=self.max_steps, success_rate=self.success_rate
        )
        self.session.reset(self.episode_seed)
        self.last_event = None
        self.last_decision = None
        # Built once per map; every base-policy decision afterwards is a table lookup.
        self.model = extract_transitions(
            create_environment(desc, max_steps=self.max_steps, success_rate=self.success_rate)
        )
        self.builds = {
            strategy: build_base_policy(
                strategy, desc, self.model, depth=self.depth, beta=self.beta, gamma=self.gamma
            )
            for strategy in BASE_STRATEGIES
        }
        self.deciders: dict[StrategyId, Decide] = {
            strategy: policy_decider(build.policy) for strategy, build in self.builds.items()
        }
        self._new_rollout_deciders()

    def _show_map_titles(self) -> None:
        n = map_size(self.desc)
        self.query_one("#lake-panel", Vertical).border_title = f"Lake · {n}×{n}"
        self.sub_title = f"{n}×{n} · map seed {self.map_seed}"

    def _restart(self, seed: int) -> None:
        self._stop_autoplay()
        self.episode_seed = seed
        self.session.reset(seed)
        self.last_event = None
        self.last_decision = None
        self._new_rollout_deciders()
        self._log_start()
        self.refresh_view()

    def _stop_autoplay(self) -> None:
        if self.autoplay_timer is not None:
            self.autoplay_timer.stop()
            self.autoplay_timer = None

    def _log_start(self) -> None:
        source = "manual" if self.strategy is None else str(self.strategy)
        self.query_one("#log", RichLog).write(
            f"— new episode, seed {self.episode_seed}, {source} —"
        )

    def _log_step(self, event: StepEvent) -> None:
        outcome = "as intended" if event.moved_as_intended else "SLIPPED"
        line = (
            f"step {event.step}: {event.action.name} from {event.state} → "
            f"intended {event.intended_state}, reached {event.next_state} ({outcome}), "
            f"reward {event.reward:g}"
        )
        if event.termination is not None:
            line += f" · {END_MESSAGES[event.termination][0]}"
        self.query_one("#log", RichLog).write(line)

    def on_unmount(self) -> None:
        self._stop_autoplay()
        self.session.close()
