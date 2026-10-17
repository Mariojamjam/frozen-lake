"""Typer entry point for the `frozen-lake` command.

Commands other than `--version` and `tui` are placeholders until they are implemented.
"""

from typing import Annotated

import typer

from frozen_lake import __version__

app = typer.Typer(
    name="frozen-lake",
    help="BFS vs Expectimax as base policies for Monte Carlo Rollout in stochastic FrozenLake.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"frozen-lake {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """BFS vs Expectimax as base policies for Monte Carlo Rollout in stochastic FrozenLake."""


def _not_implemented(command: str) -> None:
    typer.echo(f"`{command}` is not implemented yet.", err=True)
    raise typer.Exit(code=2)


@app.command()
def tui(
    size: Annotated[int, typer.Option(min=2, help="Side length of the generated square map.")] = 4,
    map_seed: Annotated[int, typer.Option(help="Seed that generates the map layout.")] = 42,
    episode_seed: Annotated[int, typer.Option(help="Seed of the real episode's slips.")] = 42,
    frozen_probability: Annotated[
        float, typer.Option(help="Probability that a generated tile is frozen (not a hole).")
    ] = 0.9,
    max_steps: Annotated[
        int | None, typer.Option(min=1, help="Real-step limit per episode; defaults to 4 × size².")
    ] = None,
    success_rate: Annotated[
        float, typer.Option(help="Probability that a move goes in the intended direction.")
    ] = 0.8,
) -> None:
    """Explore a slippery lake step by step, choosing each intended action by hand."""
    from frozen_lake.environment import generate_map

    try:
        desc = generate_map(size, frozen_probability, seed=map_seed)
        steps = max_steps if max_steps is not None else 4 * size * size
        # Imported here so that commands other than `tui` never load Textual.
        from frozen_lake.tui.app import FrozenLakeApp

        explorer = FrozenLakeApp(
            desc,
            map_seed=map_seed,
            episode_seed=episode_seed,
            max_steps=steps,
            success_rate=success_rate,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    explorer.run()


@app.command()
def episode() -> None:
    """Run a single headless episode."""
    _not_implemented("episode")


@app.command()
def benchmark() -> None:
    """Run a headless experiment from a TOML config."""
    _not_implemented("benchmark")


@app.command()
def validate() -> None:
    """Validate an experiment config and its maps."""
    _not_implemented("validate")


@app.command()
def plot() -> None:
    """Render figures from a persisted results directory."""
    _not_implemented("plot")
