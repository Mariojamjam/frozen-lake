"""Single-screen Textual explorer: pick an intended action, take one real step, watch the ice."""

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static

from frozen_lake.environment import MapDesc, map_size, tile_at, to_coordinates
from frozen_lake.episodes import EpisodeSession
from frozen_lake.models import Action, StateId, StepEvent, TerminationReason, Tile

TILE_STYLES = {
    Tile.START: ("S", "bold white on dark_blue"),
    Tile.FROZEN: ("·", "bright_white on grey23"),
    Tile.HOLE: ("H", "bold white on dark_red"),
    Tile.GOAL: ("G", "bold black on green3"),
}
AGENT_STYLE = "bold black on gold1"
# The agent marker keeps the tile's color where the episode ends, so a fall or a win stays visible.
AGENT_ON_TERMINAL_STYLES = {Tile.HOLE: "bold gold1 on dark_red", Tile.GOAL: "bold black on green3"}
PREVIOUS_STYLE = "bold gold1 on grey23"

END_MESSAGES = {
    TerminationReason.GOAL: ("GOAL reached", "bold green3"),
    TerminationReason.HOLE: ("fell into a HOLE", "bold red"),
    TerminationReason.TIMEOUT: ("TIMEOUT (step limit reached)", "bold yellow"),
}

LEGEND = "@ agent   ○ previous tile\nS start   · frozen   H hole   G goal"
HELP = (
    "Arrows choose the intended action; Space takes one real step with it. The ice may slip "
    "the agent sideways (10% each way). R replays the same episode seed, N starts a new seed."
)


def render_lake(desc: MapDesc, agent: StateId, previous: StateId | None) -> Text:
    """Draw the grid as colored three-character cells."""
    n = map_size(desc)
    text = Text()
    for row in range(n):
        for col in range(n):
            state = row * n + col
            tile = tile_at(desc, state)
            letter, style = TILE_STYLES[tile]
            if state == agent:
                letter, style = "@", AGENT_ON_TERMINAL_STYLES.get(tile, AGENT_STYLE)
            elif state == previous and tile == Tile.FROZEN:
                letter, style = "○", PREVIOUS_STYLE
            text.append(f" {letter} ", style=style)
        text.append("\n")
    text.append(f"\n{LEGEND}", style="dim")
    return text


class FrozenLakeApp(App):
    """Manual explorer for one map. Actions come from the keyboard, not from a planner."""

    CSS_PATH = "app.tcss"
    TITLE = "frozen-lake"
    BINDINGS = [
        Binding("left", "choose('LEFT')", "Left", show=False, priority=True),
        Binding("down", "choose('DOWN')", "Down", show=False, priority=True),
        Binding("right", "choose('RIGHT')", "Right", show=False, priority=True),
        Binding("up", "choose('UP')", "Up", show=False, priority=True),
        Binding("space", "step", "Step", priority=True),
        Binding("r", "replay", "Replay seed"),
        Binding("n", "new_episode", "New seed"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    chosen_action: reactive[Action] = reactive(Action.RIGHT)

    def __init__(
        self,
        desc: MapDesc,
        *,
        map_seed: int,
        episode_seed: int,
        max_steps: int,
        success_rate: float,
    ) -> None:
        super().__init__()
        self.desc = desc
        self.map_seed = map_seed
        self.episode_seed = episode_seed
        self.session = EpisodeSession(desc, max_steps=max_steps, success_rate=success_rate)
        self.session.reset(episode_seed)
        self.last_event: StepEvent | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield Static(id="lake")
            yield Static(id="status")
        yield RichLog(id="log", max_lines=200, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        n = map_size(self.desc)
        self.query_one("#lake", Static).border_title = "Lake"
        self.query_one("#status", Static).border_title = "Status"
        log = self.query_one("#log", RichLog)
        log.border_title = "Event log"
        log.can_focus = False  # keep arrow keys for choosing actions, not scrolling
        self.sub_title = f"{n}×{n} · map seed {self.map_seed}"
        self._log_start()
        self.refresh_view()

    # --- actions ------------------------------------------------------------------------------

    def action_choose(self, name: str) -> None:
        self.chosen_action = Action[name]

    def action_step(self) -> None:
        if self.session.finished:
            self.notify("Episode finished. Press R to replay or N for a new seed.")
            return
        event = self.session.step(self.chosen_action)
        self.last_event = event
        self._log_step(event)
        self.refresh_view()

    def action_replay(self) -> None:
        self._restart(self.episode_seed)

    def action_new_episode(self) -> None:
        self._restart(self.episode_seed + 1)

    def action_help(self) -> None:
        self.notify(HELP, title="How to explore", timeout=8)

    def watch_chosen_action(self) -> None:
        if self.is_mounted:
            self.refresh_view()

    # --- rendering ----------------------------------------------------------------------------

    def refresh_view(self) -> None:
        previous = self.last_event.state if self.last_event else None
        self.query_one("#lake", Static).update(render_lake(self.desc, self.session.state, previous))
        self.query_one("#status", Static).update(self.status_text())

    def status_text(self) -> Text:
        session = self.session
        n = map_size(self.desc)
        row, col = to_coordinates(session.state, n)
        text = Text()
        text.append("Action source  ", style="bold")
        text.append("MANUAL (keyboard, no planner)\n", style="italic")
        text.append("Chosen action  ", style="bold")
        text.append(f"{self.chosen_action.name}\n", style="bold cyan")
        text.append(f"Episode seed   {session.seed}\n")
        text.append(f"Step           {session.steps} / {session.max_steps}\n")
        text.append(f"State          {session.state} (row {row}, col {col})\n")
        text.append(f"Total reward   {session.total_reward:g}\n\n")

        event = self.last_event
        if event is None:
            text.append("Last step      none yet\n")
        else:
            text.append(f"Last step      {event.action.name} from {event.state}\n")
            text.append(f"Intended       {event.intended_state}\n")
            reached_style = "green3" if event.moved_as_intended else "bold yellow"
            text.append("Reached        ")
            text.append(f"{event.next_state}", style=reached_style)
            if not event.moved_as_intended:
                text.append(" (slipped)", style=reached_style)
            text.append("\n")

        text.append("\nStatus         ", style="bold")
        if session.termination is None:
            text.append("running", style="bold")
        else:
            message, style = END_MESSAGES[session.termination]
            text.append(message, style=style)
        return text

    # --- helpers ------------------------------------------------------------------------------

    def _restart(self, seed: int) -> None:
        self.episode_seed = seed
        self.session.reset(seed)
        self.last_event = None
        self._log_start()
        self.refresh_view()

    def _log_start(self) -> None:
        self.query_one("#log", RichLog).write(f"— new episode, seed {self.episode_seed} —")

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
        self.session.close()
