"""Running the analysis of one results folder: estimates, tables and figures."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

from frozen_lake import __version__
from frozen_lake.analysis import figures, tables
from frozen_lake.analysis.bootstrap import (
    LEVEL,
    SizeResults,
    amortization_sensitivity,
    analyze_size,
)
from frozen_lake.analysis.loading import load_decisions, load_run
from frozen_lake.analysis.summaries import (
    build_summaries,
    decision_summaries,
    shared_state_times,
)
from frozen_lake.experiments import validate_run
from frozen_lake.experiments.records import write_json
from frozen_lake.experiments.seeds import derive_seed

DEFAULT_RESAMPLES = 2000
# Values of R shown in figure H; the run's own E is always added.
R_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIGURE_DPI = 150


class InvalidResults(ValueError):
    """The results folder failed validation, so it is not analyzed."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    figures: tuple[Path, ...]
    tables: tuple[Path, ...]
    results: tuple[SizeResults, ...]


def analyze_run(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Analyze a finished run: write the tables and figures of the analysis plan.

    Everything is computed from the saved raw results; no agent is run again. The run is
    validated first. Output goes to `<run>/analysis` unless `output_dir` is given; files of an
    earlier analysis with the same names are replaced, since they are derived data.

    `seed` seeds the bootstrap; by default it derives from the experiment's master seed, so
    the same run always gives the same intervals.
    """
    run_dir = Path(run_dir)
    problems = validate_run(run_dir)
    if problems:
        raise InvalidResults(problems)
    say = progress or (lambda message: None)

    out = Path(output_dir) if output_dir is not None else run_dir / "analysis"
    figure_dir, table_dir = out / "figures", out / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    run = load_run(run_dir)
    decisions = load_decisions(run_dir)
    base_seed = run.config.experiment.master_seed if seed is None else seed
    episodes = run.config.experiment.episodes_per_map

    results = []
    for size, data in run.sizes.items():
        say(f"bootstrapping size {size} ({resamples} resamples)")
        results.append(
            analyze_size(data, resamples=resamples, seed=derive_seed(base_seed, "bootstrap", size))
        )
    r_values = sorted(set(R_GRID) | {episodes})
    curves = {size: amortization_sensitivity(data, r_values) for size, data in run.sizes.items()}
    builds = build_summaries(run)
    decision_rows = decision_summaries(decisions) if decisions else None
    shared = shared_state_times(run, decisions) if decisions else []

    say("writing tables")
    written_tables = [
        tables.write_arm_table(table_dir / "arms.csv", results),
        tables.write_difference_table(table_dir / "paired_differences.csv", results),
        tables.write_sensitivity_table(
            table_dir / "amortization_sensitivity.csv", curves, r_values
        ),
        tables.write_build_table(table_dir / "policy_builds.csv", builds),
    ]
    if decision_rows is not None:
        written_tables.append(
            tables.write_decision_table(table_dir / "decision_times.csv", decision_rows)
        )
        written_tables.append(
            tables.write_shared_state_table(table_dir / "shared_state_decision_time.csv", shared)
        )

    say("drawing figures")
    drawings: list[tuple[str, Figure]] = [
        ("figure_a_success_vs_amortized_time", figures.success_vs_time(results)),
        ("figure_b_success_vs_samples", figures.success_vs_samples(results)),
        ("figure_c_cost_vs_samples", figures.cost_vs_samples(results, decision_rows)),
        ("figure_d_build_time_vs_size", figures.build_time_vs_size(builds)),
        ("figure_e_search_work_vs_size", figures.search_work_vs_size(builds)),
        ("figure_f_memory_diagnostics", figures.memory_diagnostics(builds)),
    ]
    if shared:
        drawings.append(("figure_g_shared_state_decision_time", figures.shared_state_time(shared)))
    drawings.append(
        (
            "figure_h_amortization_sensitivity",
            figures.amortization_sensitivity(results, curves, r_values),
        )
    )
    written_figures = []
    for name, figure in drawings:
        path = figure_dir / f"{name}.png"
        figure.savefig(path, dpi=FIGURE_DPI)
        written_figures.append(path)

    write_json(
        out / "analysis.json",
        {
            "run": str(run_dir),
            "config_sha256": run.metadata.get("config_sha256"),
            "maps_sha256": run.metadata.get("maps_sha256"),
            "frozen_lake_version": __version__,
            "bootstrap": {
                "method": "cluster bootstrap: resample maps, then episodes within each map",
                "resamples": resamples,
                "confidence_level": LEVEL,
                "interval": "percentile",
                "seed": base_seed,
            },
            "amortization_r": episodes,
            "decisions_logged": decisions is not None,
            "figures": [path.name for path in written_figures],
            "tables": [path.name for path in written_tables],
        },
    )
    return AnalysisResult(out, tuple(written_figures), tuple(written_tables), tuple(results))
