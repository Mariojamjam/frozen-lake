"""Typer entry point for the `frozen-lake` command.

Every command explains itself with `-h`: what it does, its options grouped by topic, and
examples that can be copied as they are.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from frozen_lake import __version__
from frozen_lake.models import StrategyId


def _examples(rows: list[tuple[str, str]], *, title: str = "Examples", note: str = "") -> str:
    """Help text for a table of commands and what each one does, for the end of a help page."""
    width = max(len(command) for command, _ in rows)
    lines = [f"[bold]{title}[/bold]"]
    lines += [f"[cyan]{command.ljust(width)}[/cyan]   {what}" for command, what in rows]
    return "\n".join(lines) + (f"\n\n{note}" if note else "")


WORKFLOW = _examples(
    [
        ("frozen-lake tui -s 6", "1. explore a lake, by hand or with a strategy"),
        ("frozen-lake episode", "2. play one episode and read every step"),
        ("frozen-lake validate", "3. check the experiment configuration"),
        ("frozen-lake benchmark", "4. run the experiment"),
        ("frozen-lake validate -i <run>", "5. check that a run is complete"),
        ("frozen-lake plot -i <run>", "6. analyze it: tables and figures"),
    ],
    title="Typical workflow",
    note="Every command explains its options and shows examples with [bold]-h[/bold], "
    "for example [bold]frozen-lake episode -h[/bold].",
)

app = typer.Typer(
    name="frozen-lake",
    help="BFS vs Expectimax as base policies for Monte Carlo Rollout in stochastic FrozenLake.",
    epilog=WORKFLOW,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
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

DEFAULT_CONFIG = Path("configs/experiment.toml")
RESULTS = Path("results")

LAKE, STRATEGY = "Lake", "Strategy"
EXPECTIMAX, ROLLOUT = "Expectimax (EXP, EXP-RO)", "Rollout (BFS-RO, EXP-RO)"
INPUT, OUTPUT, ANALYSIS = "Input", "Output", "Analysis"

SizeOption = Annotated[
    int,
    typer.Option(
        "--size",
        "-s",
        min=2,
        help="Side of the square lake, in tiles: the lake has size × size tiles.",
        rich_help_panel=LAKE,
    ),
]
MapSeedOption = Annotated[
    int,
    typer.Option(
        "--map-seed",
        "-m",
        min=0,
        help="Seed of the lake layout: the same seed always draws the same lake.",
        rich_help_panel=LAKE,
    ),
]
EpisodeSeedOption = Annotated[
    int,
    typer.Option(
        "--episode-seed",
        "-e",
        min=0,
        help="Seed of the real episode's slips: the same seed replays the same slips.",
        rich_help_panel=LAKE,
    ),
]
FrozenProbabilityOption = Annotated[
    float,
    typer.Option(
        "--frozen-probability",
        help="Chance that each generated tile is ice rather than a hole, in (0, 1].",
        rich_help_panel=LAKE,
    ),
]
SuccessRateOption = Annotated[
    float,
    typer.Option(
        "--success-rate",
        help="Chance that a move goes in the intended direction, in (0, 1]; the rest is split "
        "evenly between the two sideways slips.",
        rich_help_panel=LAKE,
    ),
]
MaxStepsOption = Annotated[
    int | None,
    typer.Option(
        "--max-steps",
        min=1,
        show_default="4 × size²",
        help="Real steps before the episode ends in a timeout.",
        rich_help_panel=LAKE,
    ),
]
DepthOption = Annotated[
    int,
    typer.Option(
        "--depth",
        "-d",
        min=1,
        help="How many decisions Expectimax looks ahead.",
        rich_help_panel=EXPECTIMAX,
    ),
]
BetaOption = Annotated[
    float,
    typer.Option(
        "--beta",
        "-b",
        help="Leaf heuristic beta ** (BFS distance to the goal), with 0 < beta < 1. "
        "Closer to 1, distance to the goal matters less.",
        rich_help_panel=EXPECTIMAX,
    ),
]
RolloutsOption = Annotated[
    int,
    typer.Option(
        "--rollouts",
        "-n",
        min=1,
        help="N: simulated trajectories per action at each decision. More is more precise "
        "and slower.",
        rich_help_panel=ROLLOUT,
    ),
]
HorizonOption = Annotated[
    int | None,
    typer.Option(
        "--horizon",
        min=1,
        show_default="2 × size²",
        help="Longest simulated trajectory, in steps; never more than the real steps left.",
        rich_help_panel=ROLLOUT,
    ),
]
RolloutSeedOption = Annotated[
    int | None,
    typer.Option(
        "--rollout-seed",
        min=0,
        show_default="derived from the episode seed",
        help="Seed of the simulations, kept apart from the real episode's slips.",
        rich_help_panel=ROLLOUT,
    ),
]

# Appended to messages about invalid settings, to say what the setting means.
_HINTS = {
    "beta": "it is the base of the leaf heuristic beta ** distance to the goal",
    "frozen_probability": "it is the chance that a tile is ice; 1 draws a lake without holes",
    "success_rate": "it is the chance that a move is not a slip; 1 turns slipping off",
}


def _bad_parameter(error: ValueError) -> typer.BadParameter:
    message = str(error)
    hint = next((text for name, text in _HINTS.items() if message.startswith(name)), None)
    return typer.BadParameter(f"{message} ({hint})" if hint else message)


def _interactive() -> bool:
    """True in a terminal where someone can answer: never in CI, a pipe or a test runner."""
    return sys.stdin.isatty() and sys.stdout.isatty()


TUI_EXAMPLES = _examples(
    [
        ("frozen-lake tui -s 6", "a 6×6 lake with the default seeds"),
        ("frozen-lake tui -s 8 -m 7 -e 3", "another lake (map seed 7) and other slips (seed 3)"),
        ("frozen-lake tui -s 10 -n 100 -d 4", "Rollout with N = 100, Expectimax depth 4"),
        ("frozen-lake tui -s 4 --max-steps 10", "a short step limit, to see a timeout"),
    ],
    note="[bold]Keys inside the explorer[/bold]\n"
    "arrows choose a direction · Space takes one step · a plays the episode automatically\n"
    "1 BFS · 2 EXP · 3 BFS-RO · 4 EXP-RO · m manual (switching restarts the episode)\n"
    "+ / - change N · ] / [ change the Expectimax depth · s settings: N, depth, beta, horizon\n"
    "p policy arrows · r replay · n new slips · g new lake · ? help · q quit",
)


@app.command(no_args_is_help=True, epilog=TUI_EXAMPLES)
def tui(
    size: SizeOption,
    map_seed: MapSeedOption = 42,
    episode_seed: EpisodeSeedOption = 42,
    frozen_probability: FrozenProbabilityOption = 0.9,
    success_rate: SuccessRateOption = 0.8,
    max_steps: MaxStepsOption = None,
    depth: DepthOption = 6,
    beta: BetaOption = 0.9,
    rollouts: RolloutsOption = 25,
    horizon: HorizonOption = None,
    rollout_seed: RolloutSeedOption = None,
) -> None:
    """Explore a slippery lake step by step, by hand or with any of the four strategies.

    Opens an interactive explorer on a generated lake. Choose a direction with the arrow keys
    and press Space to take one real step, or let a strategy decide: BFS, Expectimax, or
    Rollout over either of them. The lake size is required; every other option has a default.
    """
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
            rollout_seed=rollout_seed,
            make_map=lambda seed: generate_map(size, frozen_probability, seed=seed),
        )
    except ValueError as error:
        raise _bad_parameter(error) from error
    explorer.run()


EPISODE_EXAMPLES = _examples(
    [
        ("frozen-lake episode", "in a terminal: choose every option in a form"),
        ("frozen-lake episode -s 6 -e 7", "BFS on a 6×6 lake, episode seed 7"),
        ("frozen-lake episode -S EXP -s 8 -d 4", "Expectimax with depth 4 on an 8×8 lake"),
        ("frozen-lake episode -S EXP-RO -n 100 -q", "Rollout over Expectimax, N = 100, summary"),
        ("frozen-lake episode -S BFS-RO --rollout-seed 3", "other simulations, same real slips"),
    ]
)


@app.command(epilog=EPISODE_EXAMPLES)
def episode(
    ctx: typer.Context,
    strategy: Annotated[
        StrategyId,
        typer.Option(
            "--strategy",
            "-S",
            help="Who decides: BFS or EXP follow their policy table; BFS-RO or EXP-RO run "
            "Rollout over it.",
            rich_help_panel=STRATEGY,
        ),
    ] = StrategyId.BFS,
    size: SizeOption = 4,
    map_seed: MapSeedOption = 42,
    episode_seed: EpisodeSeedOption = 42,
    frozen_probability: FrozenProbabilityOption = 0.9,
    success_rate: SuccessRateOption = 0.8,
    max_steps: MaxStepsOption = None,
    depth: DepthOption = 6,
    beta: BetaOption = 0.9,
    rollouts: RolloutsOption = 25,
    horizon: HorizonOption = None,
    rollout_seed: RolloutSeedOption = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet", "-q", help="Print only the summary, not every step.", rich_help_panel=OUTPUT
        ),
    ] = False,
) -> None:
    """Play one episode without the explorer and print every step and the result.

    Each step shows the action, where the agent meant to go and where it landed; Rollout
    strategies also show their estimate for every action. With no options in a terminal, a
    short form opens to choose them, and the equivalent command is printed before the episode.
    """
    values = {name: value for name, value in ctx.params.items()}
    if _interactive() and all(
        ctx.get_parameter_source(name).name == "DEFAULT" for name in ctx.params
    ):
        from frozen_lake.tui.episode_form import EpisodeForm, episode_command

        chosen = EpisodeForm(values).run()
        if chosen is None:
            typer.echo("cancelled")
            return
        typer.echo(f"$ {episode_command(chosen, values)}")
        values = chosen
    _play_episode(**values)


def _play_episode(
    *,
    strategy: StrategyId,
    quiet: bool,
    size: int,
    map_seed: int,
    episode_seed: int,
    frozen_probability: float,
    success_rate: float,
    max_steps: int | None,
    depth: int,
    beta: float,
    rollouts: int,
    horizon: int | None,
    rollout_seed: int | None,
) -> None:
    import random

    from frozen_lake.environment import create_environment, extract_transitions, generate_map
    from frozen_lake.episodes import EpisodeSession, policy_decider, run_episode
    from frozen_lake.policies import build_base_policy
    from frozen_lake.rollout import rollout_decider, rollout_rng

    strategy = StrategyId(strategy)
    try:
        desc = generate_map(size, frozen_probability, seed=map_seed)
        steps = max_steps if max_steps is not None else 4 * size * size
        session = EpisodeSession(desc, max_steps=steps, success_rate=success_rate)
        model = extract_transitions(
            create_environment(desc, max_steps=steps, success_rate=success_rate)
        )
        build = build_base_policy(strategy.base, desc, model, depth=depth, beta=beta, gamma=GAMMA)
    except ValueError as error:
        raise _bad_parameter(error) from error

    if strategy.uses_rollout:
        simulation_horizon = horizon if horizon is not None else 2 * size * size
        rng = random.Random(rollout_seed) if rollout_seed is not None else rollout_rng(episode_seed)
        decide = rollout_decider(
            build.policy, model, samples=rollouts, horizon=simulation_horizon, gamma=GAMMA, rng=rng
        )
    else:
        decide = policy_decider(build.policy)

    limit = f"{steps}" if max_steps is not None else f"{steps} (4 × {size}²)"
    typer.echo(
        f"map {size}×{size} (seed {map_seed}) · strategy {strategy} · "
        f"episode seed {episode_seed} · step limit {limit}"
    )
    side = (1 - success_rate) / 2
    typer.echo(
        f"moves go as intended {success_rate:.0%} of the time, {side:.0%} to each side · "
        f"tiles are ice with probability {frozen_probability:g}"
    )
    fallbacks = build.diagnostics["fallback_states"]
    if strategy.base is StrategyId.EXP:
        work = (
            f"depth {depth}, beta {beta}: {build.states_evaluated} subproblems, "
            f"{build.diagnostics['cache_hits']} cache hits, {fallbacks} fallbacks"
        )
    else:
        work = f"{build.states_evaluated} states visited, {fallbacks} fallbacks"
    typer.echo(f"{strategy.base} policy built in {build.build_time_ns / 1e6:.3f} ms ({work})")
    if strategy.uses_rollout:
        seed_note = "" if rollout_seed is None else f", rollout seed {rollout_seed}"
        typer.echo(
            f"rollout: {rollouts} trajectories per action, horizon {simulation_horizon} "
            f"(capped by the real steps left){seed_note}"
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
                f" · overrode {strategy.base} {decision.base_action.name}"
                if decision.action is not decision.base_action
                else ""
            )
            line += f"\n           estimates {estimates}{overrode}"
        typer.echo(line)

    result = run_episode(
        session,
        decide,
        seed=episode_seed,
        strategy=strategy,
        on_step=None if quiet else print_step,
    )
    session.close()
    simulated = (
        f" · {result.simulated_transitions} simulated transitions" if strategy.uses_rollout else ""
    )
    typer.echo(
        f"result: {result.termination_reason} in {result.steps} steps · "
        f"reward {result.total_reward:g} · agent compute {result.agent_compute_ns / 1e6:.3f} ms"
        f"{simulated}"
    )


def _config_path(path: Path) -> Path:
    """A configuration file, or the `experiment.toml` inside a folder."""
    return path / "experiment.toml" if path.is_dir() else path


def _load_experiment_config(path: Path):
    from frozen_lake.experiments import ConfigError, load_config

    try:
        return load_config(path)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        if path == DEFAULT_CONFIG and not path.exists():
            typer.echo(
                "Run the command from the repository folder, or give a configuration with -i.",
                err=True,
            )
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


def _next_steps(run_dir: Path) -> None:
    typer.echo("next:")
    typer.echo(f"  frozen-lake validate -i {run_dir}    check that the run is complete")
    typer.echo(f"  frozen-lake plot -i {run_dir}        tables and figures with intervals")


BENCHMARK_EXAMPLES = _examples(
    [
        ("frozen-lake benchmark --dry-run", "describe the study's experiment, run nothing"),
        ("frozen-lake benchmark", "run it (asks before starting)"),
        ("frozen-lake benchmark -y", "run it without asking"),
        ("frozen-lake benchmark -i my.toml -o runs/", "another configuration, results in runs/"),
        ("frozen-lake benchmark -i configs/", "a folder: uses the experiment.toml inside it"),
    ],
    note="The study's experiment takes about half an hour on a laptop. Every run goes to a new "
    "folder named after its start time and configuration hash; an existing folder is never "
    "overwritten.",
)


@app.command(epilog=BENCHMARK_EXAMPLES)
def benchmark(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Experiment configuration (TOML), or a folder that contains experiment.toml.",
            rich_help_panel=INPUT,
        ),
    ] = DEFAULT_CONFIG,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            show_default="experiment.output_dir of the configuration",
            help="Folder where the run's results folder is created.",
            rich_help_panel=OUTPUT,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Check and describe the experiment; run nothing."),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Start without asking for confirmation.")
    ] = False,
) -> None:
    """Run the whole experiment headlessly and write its raw results.

    Generates every map, builds and times each base policy, plays every strategy on the same
    maps and episode seeds, writes the raw CSV results, a summary and the run's metadata, and
    checks them for completeness. The configuration fixes every setting of the experiment.
    """
    config_path = _config_path(input_path)
    experiment_config = _load_experiment_config(config_path)
    _describe_workload(experiment_config)
    if dry_run:
        return

    from frozen_lake.experiments import run_experiment, workload

    if not yes:
        episodes = workload(experiment_config).episodes
        typer.confirm(f"Run {episodes} episodes now?", abort=True)
    try:
        run_dir = run_experiment(
            experiment_config,
            output_root=output,
            config_path=config_path,
            progress=lambda message: typer.echo(message, err=True),
        )
    except FileExistsError as error:
        typer.echo(f"Results folder already exists, not overwriting: {error.filename}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"results written to {run_dir}")
    _next_steps(run_dir)


VALIDATE_EXAMPLES = _examples(
    [
        ("frozen-lake validate", "the study's configuration"),
        ("frozen-lake validate -i my.toml", "another configuration"),
        ("frozen-lake validate -i results/canonical/experiment-1", "the results of a run"),
    ],
    note="A file is checked as a configuration; a folder, as the results of a run. Exit code 1 "
    "means the results have problems, and each one is listed.",
)


@app.command(epilog=VALIDATE_EXAMPLES)
def validate(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="A configuration file (TOML) to check, or a results folder to check for "
            "completeness.",
            rich_help_panel=INPUT,
        ),
    ] = DEFAULT_CONFIG,
) -> None:
    """Check an experiment configuration, or the results a run wrote.

    A configuration is checked key by key, and its workload is described. A results folder is
    checked for completeness: every map, episode and strategy present once, shared seeds, and
    maps that match their seeds.
    """
    if not input_path.is_dir():
        _describe_workload(_load_experiment_config(input_path))
        typer.echo("configuration is valid")
        return

    from frozen_lake.experiments import validate_run

    problems = validate_run(input_path)
    if problems:
        typer.echo("results failed validation:\n- " + "\n- ".join(problems), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"results in {input_path} are complete and consistent")
    typer.echo(f"next:\n  frozen-lake plot -i {input_path}        tables and figures")


def _runs() -> list[Path]:
    """Results folders under results/, the study's runs first, then the generated ones."""
    found = []
    for group in ("canonical", "generated"):
        folder = RESULTS / group
        if folder.is_dir():
            found += sorted(p for p in folder.iterdir() if (p / "metadata.json").is_file())
    return found


def _choose_run() -> Path:
    runs = _runs()
    if not runs:
        typer.echo("No results found under results/. Run frozen-lake benchmark first.", err=True)
        raise typer.Exit(code=2)
    listing = "\n".join(f"  {number}. {run}" for number, run in enumerate(runs, 1))
    if not _interactive():
        typer.echo(f"Choose a run with -i:\n{listing}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Runs found:\n{listing}")
    while True:
        number = typer.prompt("Run to analyze (number)", type=int, default=1)
        if 1 <= number <= len(runs):
            return runs[number - 1]
        typer.echo(f"Choose a number from 1 to {len(runs)}.")


PLOT_EXAMPLES = _examples(
    [
        ("frozen-lake plot", "choose a run from a list"),
        ("frozen-lake plot -i results/canonical/experiment-1", "the study's run"),
        ("frozen-lake plot -i <run> -o figures/ --resamples 5000", "elsewhere, more resamples"),
    ],
    note="Reads only the saved results and never runs an agent again; with the same seed it "
    "writes the same tables. Damaged results are refused (exit code 1).",
)


@app.command(epilog=PLOT_EXAMPLES)
def plot(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            show_default="choose from the runs under results/",
            help="Results folder of a finished benchmark.",
            rich_help_panel=INPUT,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            show_default="<run>/analysis",
            help="Folder for the figures and tables.",
            rich_help_panel=OUTPUT,
        ),
    ] = None,
    resamples: Annotated[
        int,
        typer.Option(
            min=100,
            help="Bootstrap resamples behind every interval.",
            rich_help_panel=ANALYSIS,
        ),
    ] = 2000,
    seed: Annotated[
        int | None,
        typer.Option(
            min=0,
            show_default="derived from the run's master seed",
            help="Seed of the bootstrap.",
            rich_help_panel=ANALYSIS,
        ),
    ] = None,
) -> None:
    """Analyze a finished run: success, cost and their intervals, as tables and figures.

    Validates the run, estimates every strategy per map and per size with cluster-bootstrap
    intervals, compares EXP-RO with BFS-RO map by map, marks the Pareto frontier, and writes
    CSV tables and PNG figures.
    """
    from frozen_lake.analysis import InvalidResults, analyze_run

    run = input_path if input_path is not None else _choose_run()
    try:
        result = analyze_run(
            run,
            output,
            resamples=resamples,
            seed=seed,
            progress=lambda message: typer.echo(message, err=True),
        )
    except InvalidResults as error:
        typer.echo("results failed validation:\n- " + "\n- ".join(error.problems), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{len(result.figures)} figures and {len(result.tables)} tables")
    typer.echo(f"analysis written to {result.output_dir}")
