"""Typer entry point for the `frozen-lake` command."""

from pathlib import Path
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


# Discount factor of the study: rewards count fully whenever they come within the horizon.
GAMMA = 1.0

DepthOption = Annotated[
    int, typer.Option(min=1, help="Expectimax look-ahead depth, in agent decisions.")
]
BetaOption = Annotated[
    float, typer.Option(help="Expectimax leaf heuristic: beta ** BFS distance, 0 < beta < 1.")
]


RolloutsOption = Annotated[
    int, typer.Option(min=1, help="Rollout strategies: simulated trajectories per action.")
]
HorizonOption = Annotated[
    int | None,
    typer.Option(
        min=1,
        help="Rollout strategies: simulated steps per trajectory; defaults to 2 × size², "
        "and never more than the real steps left.",
    ),
]


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
    depth: DepthOption = 6,
    beta: BetaOption = 0.9,
    rollouts: RolloutsOption = 25,
    horizon: HorizonOption = None,
) -> None:
    """Explore a slippery lake step by step, by hand or with any of the four strategies."""
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
            depth=depth,
            beta=beta,
            gamma=GAMMA,
            rollouts=rollouts,
            horizon=horizon,
            make_map=lambda seed: generate_map(size, frozen_probability, seed=seed),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    explorer.run()


@app.command()
def episode(
    strategy: Annotated[str, typer.Option(help="Strategy ID: BFS, EXP, BFS-RO or EXP-RO.")] = "BFS",
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
    depth: DepthOption = 6,
    beta: BetaOption = 0.9,
    rollouts: RolloutsOption = 25,
    horizon: HorizonOption = None,
    rollout_seed: Annotated[
        int | None,
        typer.Option(
            help="Rollout strategies: seed of the simulations; derived from the "
            "episode seed by default."
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Print only the summary.")] = False,
) -> None:
    """Play one headless episode with a strategy and print what happened."""
    import random

    from frozen_lake.environment import create_environment, extract_transitions, generate_map
    from frozen_lake.episodes import EpisodeSession, policy_decider, run_episode
    from frozen_lake.models import StrategyId
    from frozen_lake.policies import build_base_policy
    from frozen_lake.rollout import rollout_decider, rollout_rng

    try:
        strategy_id = StrategyId(strategy)
    except ValueError as error:
        choices = ", ".join(s.value for s in StrategyId)
        raise typer.BadParameter(f"unknown strategy {strategy!r}; choose from {choices}") from error
    try:
        desc = generate_map(size, frozen_probability, seed=map_seed)
        steps = max_steps if max_steps is not None else 4 * size * size
        session = EpisodeSession(desc, max_steps=steps, success_rate=success_rate)
        model = extract_transitions(
            create_environment(desc, max_steps=steps, success_rate=success_rate)
        )
        build = build_base_policy(
            strategy_id.base, desc, model, depth=depth, beta=beta, gamma=GAMMA
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    if strategy_id.uses_rollout:
        simulation_horizon = horizon if horizon is not None else 2 * size * size
        rng = random.Random(rollout_seed) if rollout_seed is not None else rollout_rng(episode_seed)
        decide = rollout_decider(
            build.policy, model, samples=rollouts, horizon=simulation_horizon, gamma=GAMMA, rng=rng
        )
    else:
        decide = policy_decider(build.policy)

    typer.echo(
        f"map {size}×{size} (seed {map_seed}) · strategy {strategy_id} · "
        f"episode seed {episode_seed} · step limit {steps}"
    )
    fallbacks = build.diagnostics["fallback_states"]
    if strategy_id.base is StrategyId.EXP:
        work = (
            f"depth {depth}, beta {beta}: {build.states_evaluated} subproblems, "
            f"{build.diagnostics['cache_hits']} cache hits, {fallbacks} fallbacks"
        )
    else:
        work = f"{build.states_evaluated} states visited, {fallbacks} fallbacks"
    typer.echo(f"{strategy_id.base} policy built in {build.build_time_ns / 1e6:.3f} ms ({work})")
    if strategy_id.uses_rollout:
        typer.echo(
            f"rollout: {rollouts} trajectories per action, horizon {simulation_horizon} "
            "(capped by the real steps left)"
        )

    def print_step(decision, event) -> None:
        slip = "" if event.moved_as_intended else "  (slipped)"
        line = (
            f"step {event.step:>3}: state {event.state:>3} → {event.action.name:<5} → "
            f"reached {event.next_state:>3}{slip}"
        )
        if decision.estimated_action_values is not None:
            estimates = " ".join(
                f"{action.name[0]} {value:.2f}"
                for action, value in decision.estimated_action_values.items()
            )
            overrode = (
                f" · overrode {strategy_id.base} {decision.base_action.name}"
                if decision.action is not decision.base_action
                else ""
            )
            line += f"\n           estimates {estimates}{overrode}"
        typer.echo(line)

    result = run_episode(
        session,
        decide,
        seed=episode_seed,
        strategy=strategy_id,
        on_step=None if quiet else print_step,
    )
    session.close()
    simulated = (
        f" · {result.simulated_transitions} simulated transitions"
        if strategy_id.uses_rollout
        else ""
    )
    typer.echo(
        f"result: {result.termination_reason} in {result.steps} steps · "
        f"reward {result.total_reward:g} · agent compute {result.agent_compute_ns / 1e6:.3f} ms"
        f"{simulated}"
    )


ConfigOption = Annotated[
    Path, typer.Option("--config", help="Experiment configuration (TOML).", show_default=False)
]


def _load_experiment_config(path: Path):
    from frozen_lake.experiments import ConfigError, load_config

    try:
        return load_config(path)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error


def _describe_workload(config) -> None:
    from frozen_lake.experiments import arms, workload

    load = workload(config)
    typer.echo(
        f"{len(config.environment.sizes)} sizes {list(config.environment.sizes)} × "
        f"{config.experiment.maps_per_size} maps × {config.experiment.episodes_per_map} episodes "
        f"× {load.arms} arms"
    )
    typer.echo("arms: " + ", ".join(arm.label for arm in arms(config)))
    typer.echo(f"{load.episodes} episodes · {load.policy_builds} policy builds")


@app.command()
def benchmark(
    config: ConfigOption,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Folder for the results; overrides experiment.output_dir."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and describe the experiment; run nothing.")
    ] = False,
) -> None:
    """Run the whole experiment headlessly and write its raw results."""
    experiment_config = _load_experiment_config(config)
    _describe_workload(experiment_config)
    if dry_run:
        return

    from frozen_lake.experiments import run_experiment

    try:
        run_dir = run_experiment(
            experiment_config,
            output_root=output_dir,
            config_path=config,
            progress=lambda message: typer.echo(message, err=True),
        )
    except FileExistsError as error:
        typer.echo(f"Results folder already exists, not overwriting: {error.filename}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"results written to {run_dir}")


@app.command()
def validate(
    config: Annotated[
        Path | None, typer.Option("--config", help="Experiment configuration (TOML) to check.")
    ] = None,
    run: Annotated[
        Path | None, typer.Option("--run", help="Results folder to check for completeness.")
    ] = None,
) -> None:
    """Check an experiment configuration, or the results a run wrote."""
    if (config is None) == (run is None):
        typer.echo("Give exactly one of --config or --run.", err=True)
        raise typer.Exit(code=2)
    if config is not None:
        _describe_workload(_load_experiment_config(config))
        typer.echo("configuration is valid")
        return

    from frozen_lake.experiments import validate_run

    problems = validate_run(run)
    if problems:
        typer.echo("results failed validation:\n- " + "\n- ".join(problems), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"results in {run} are complete and consistent")


@app.command()
def plot(
    run: Annotated[Path, typer.Option("--run", help="Results folder of a finished benchmark.")],
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Folder for the figures and tables; defaults to <run>/analysis."),
    ] = None,
    resamples: Annotated[
        int, typer.Option(min=100, help="Bootstrap resamples for every interval.")
    ] = 2000,
    seed: Annotated[
        int | None,
        typer.Option(help="Bootstrap seed; defaults to one derived from the master seed."),
    ] = None,
) -> None:
    """Analyze a finished run: intervals, tables and figures, from its saved results."""
    from frozen_lake.analysis import InvalidResults, analyze_run

    try:
        result = analyze_run(
            run,
            output_dir,
            resamples=resamples,
            seed=seed,
            progress=lambda message: typer.echo(message, err=True),
        )
    except InvalidResults as error:
        typer.echo("results failed validation:\n- " + "\n- ".join(error.problems), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{len(result.figures)} figures and {len(result.tables)} tables")
    typer.echo(f"analysis written to {result.output_dir}")
