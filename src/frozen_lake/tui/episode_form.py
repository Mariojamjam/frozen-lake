"""A one-screen form to choose the options of `frozen-lake episode` before playing it.

The form shows only the settings the chosen strategy uses, says what is wrong with any value,
and keeps the equivalent command line up to date, so the options can be learned by using it.
"""

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, RadioButton, RadioSet, Static

from frozen_lake.models import StrategyId
from frozen_lake.tui.colors import FROST_THEME, LILAC, ROSE
from frozen_lake.tui.fields import Field, show

LAKE_FIELDS = (
    Field("size", "Lake size (tiles per side)", int, low=2),
    Field("map_seed", "Map seed", int, low=0),
    Field("episode_seed", "Episode seed (slips)", int, low=0),
    Field("frozen_probability", "Frozen-tile probability", float, low=0, high=1, open_low=True),
    Field("success_rate", "Intended-move probability", float, low=0, high=1, open_low=True),
    Field("max_steps", "Step limit", int, low=1, optional=True, blank="4 × size²"),
)
EXPECTIMAX_FIELDS = (
    Field("depth", "Depth (decisions ahead)", int, low=1),
    Field("beta", "Heuristic beta", float, low=0, high=1, open_low=True, open_high=True),
)
ROLLOUT_FIELDS = (
    Field("rollouts", "N (trajectories per action)", int, low=1),
    Field("horizon", "Horizon (steps)", int, low=1, optional=True, blank="2 × size²"),
    Field("rollout_seed", "Rollout seed", int, low=0, optional=True, blank="from episode seed"),
)
SECTIONS = (
    ("Lake", LAKE_FIELDS),
    ("Expectimax · EXP and EXP-RO", EXPECTIMAX_FIELDS),
    ("Rollout · BFS-RO and EXP-RO", ROLLOUT_FIELDS),
)

# How each setting is written on the command line, shortest form first.
FLAGS = {
    "strategy": "-S",
    "size": "-s",
    "map_seed": "-m",
    "episode_seed": "-e",
    "frozen_probability": "--frozen-probability",
    "success_rate": "--success-rate",
    "max_steps": "--max-steps",
    "depth": "-d",
    "beta": "-b",
    "rollouts": "-n",
    "horizon": "--horizon",
    "rollout_seed": "--rollout-seed",
}


def relevant(key: str, strategy: StrategyId) -> bool:
    """Whether a setting affects an episode of `strategy`."""
    if any(field.key == key for field in EXPECTIMAX_FIELDS):
        return strategy.base is StrategyId.EXP
    if any(field.key == key for field in ROLLOUT_FIELDS):
        return strategy.uses_rollout
    return True


def episode_command(values: dict, defaults: dict) -> str:
    """The `frozen-lake episode` command that plays `values`: only what differs from `defaults`."""
    strategy = StrategyId(values["strategy"])
    parts = ["frozen-lake", "episode"]
    for key, flag in FLAGS.items():
        value = values[key]
        if value is None or value == defaults[key] or not relevant(key, strategy):
            continue
        parts += [flag, str(value) if key == "strategy" else show(value)]
    if values["quiet"]:
        parts.append("-q")
    return " ".join(parts)


class EpisodeForm(App[dict | None]):
    """Choose a strategy and its settings; returns them, or None if cancelled."""

    TITLE = "frozen-lake episode"
    CSS = """
    Screen { align: center middle; background: $background; }
    #form {
        width: 76; height: auto; max-height: 100%; padding: 0 2;
        border: heavy $primary; border-title-color: $accent; border-title-style: bold;
        background: $surface; overflow-y: auto;
    }
    .section { color: $accent; text-style: bold; margin-top: 1; }
    RadioSet { width: 100%; layout: horizontal; border: none; padding: 0; background: $surface; }
    .row { height: 1; margin-bottom: 0; }
    .row Label { width: 30; }
    .row Input { width: 1fr; height: 1; border: none; padding: 0 1; background: $boost; }
    .row Input:focus { background: $primary 35%; }
    .row Input.-invalid { background: $error 45%; }
    .row Input:disabled { background: $surface; color: $text-disabled; }
    #quiet { margin-top: 1; border: none; padding: 0; background: $surface; }
    #command { margin-top: 1; padding: 0 1; background: $panel; height: auto; }
    #problems { height: auto; }
    .buttons { height: 1; align: right middle; margin: 1 0; }
    Button { margin-left: 2; min-width: 10; height: 1; border: none; }
    """
    # Enter plays from any field, including the strategy buttons, which would otherwise take it.
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Run", priority=True),
    ]

    def __init__(self, initial: dict) -> None:
        super().__init__()
        self.initial = dict(initial)
        self.register_theme(FROST_THEME)
        self.theme = "frost"

    def compose(self) -> ComposeResult:
        with Vertical(id="form"):
            yield Static("Strategy", classes="section")
            with RadioSet(id="strategy"):
                for strategy in StrategyId:
                    yield RadioButton(
                        str(strategy), value=strategy == self.initial["strategy"], id=strategy.name
                    )
            for title, fields in SECTIONS:
                yield Static(title, classes="section")
                for field in fields:
                    with Horizontal(classes="row"):
                        yield Label(field.label)
                        yield Input(
                            show(self.initial[field.key]),
                            id=field.key,
                            type="integer" if field.kind is int else "number",
                            placeholder=field.blank,
                        )
            yield Checkbox(
                "Summary only (no step-by-step output)", self.initial["quiet"], id="quiet"
            )
            yield Static(id="command")
            yield Static(id="problems")
            with Horizontal(classes="buttons"):
                yield Button("Run", id="run", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#form").border_title = "Play one episode"
        self.refresh_form()

    def chosen_strategy(self) -> StrategyId:
        pressed = self.query_one("#strategy", RadioSet).pressed_button
        return (
            StrategyId[pressed.id] if pressed is not None else StrategyId(self.initial["strategy"])
        )

    def read(self) -> tuple[dict, list[str]]:
        """The values in the form and what is wrong with them, for the chosen strategy."""
        strategy = self.chosen_strategy()
        values = {"strategy": strategy, "quiet": self.query_one("#quiet", Checkbox).value}
        problems = []
        for _, fields in SECTIONS:
            for field in fields:
                entry = self.query_one(f"#{field.key}", Input)
                used = relevant(field.key, strategy)
                entry.disabled = not used
                value, problem = field.parse(entry.value)
                if not used:
                    # A setting the strategy ignores keeps its starting value and cannot block Run.
                    value, problem = self.initial[field.key], None
                entry.set_class(problem is not None, "-invalid")
                values[field.key] = value
                if problem:
                    problems.append(problem)
        return values, problems

    def refresh_form(self) -> bool:
        values, problems = self.read()
        command = Text("$ ", style=LILAC)
        command.append(episode_command(values, self.initial), style="bold")
        self.query_one("#command", Static).update(command)
        self.query_one("#problems", Static).update(
            Text("\n".join(problems), style=ROSE)
            if problems
            else Text("Enter or Run plays the episode · Esc cancels", style=LILAC)
        )
        self.query_one("#run", Button).disabled = bool(problems)
        return not problems

    def on_input_changed(self) -> None:
        self.refresh_form()

    def on_radio_set_changed(self) -> None:
        self.refresh_form()

    def on_checkbox_changed(self) -> None:
        self.refresh_form()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.action_run()
        else:
            self.action_cancel()

    def action_submit(self) -> None:
        focused = self.focused
        if isinstance(focused, Button) and focused.id == "cancel":
            self.action_cancel()
        else:
            self.action_run()

    def action_run(self) -> None:
        if self.refresh_form():
            self.exit(self.read()[0])

    def action_cancel(self) -> None:
        self.exit(None)
