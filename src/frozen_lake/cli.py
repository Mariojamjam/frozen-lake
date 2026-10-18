"""Typer entry point for the `frozen-lake` command.

Commands other than `--version`, `tui` and `episode` are placeholders until implemented.
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
            make_map=lambda seed: generate_map(size, frozen_probability, seed=seed),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    explorer.run()


@app.command()
def episode(
    strategy: Annotated[
        str, typer.Option(help="Strategy ID; only BFS is available so far.")
    ] = "BFS",
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
    quiet: Annotated[bool, typer.Option("--quiet", help="Print only the summary.")] = False,
) -> None:
    """Play one headless episode with a strategy and print what happened."""
    from frozen_lake.environment import generate_map
    from frozen_lake.episodes import EpisodeSession, policy_decider, run_episode
    from frozen_lake.models import StrategyId
    from frozen_lake.policies.bfs import build_bfs_policy

    try:
        strategy_id = StrategyId(strategy)
    except ValueError as error:
        choices = ", ".join(s.value for s in StrategyId)
        raise typer.BadParameter(f"unknown strategy {strategy!r}; choose from {choices}") from error
    if strategy_id is not StrategyId.BFS:
        typer.echo(f"Strategy {strategy_id} is not implemented yet.", err=True)
        raise typer.Exit(code=2)

    try:
        desc = generate_map(size, frozen_probability, seed=map_seed)
        steps = max_steps if max_steps is not None else 4 * size * size
        session = EpisodeSession(desc, max_steps=steps, success_rate=success_rate)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    build = build_bfs_policy(desc)
    typer.echo(
        f"map {size}×{size} (seed {map_seed}) · strategy {strategy_id} · "
        f"episode seed {episode_seed} · step limit {steps}"
    )
    typer.echo(
        f"policy built in {build.build_time_ns / 1e6:.3f} ms "
        f"({build.states_evaluated} states visited, "
        f"{build.diagnostics['fallback_states']} fallbacks)"
    )

    def print_step(_decision, event) -> None:
        slip = "" if event.moved_as_intended else "  (slipped)"
        typer.echo(
            f"step {event.step:>3}: state {event.state:>3} → {event.action.name:<5} → "
            f"reached {event.next_state:>3}{slip}"
        )

    result = run_episode(
        session,
        policy_decider(build.policy),
        seed=episode_seed,
        strategy=strategy_id,
        on_step=None if quiet else print_step,
    )
    session.close()
    typer.echo(
        f"result: {result.termination_reason} in {result.steps} steps · "
        f"reward {result.total_reward:g} · agent compute {result.agent_compute_ns / 1e6:.3f} ms"
    )


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
