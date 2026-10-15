"""Typer entry point for the `frozen-lake` command.

Commands other than `--version` are placeholders until they are implemented.
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
def tui() -> None:
    """Interactive Textual demonstrator."""
    _not_implemented("tui")


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
