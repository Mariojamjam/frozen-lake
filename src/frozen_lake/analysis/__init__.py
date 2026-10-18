"""Analysis of a finished experiment: estimates with uncertainty, tables and figures.

This is the last stage of the pipeline (design doc §12). It reads the raw results a `benchmark`
run saved and never runs an agent again.

Modules:
- loading: reading a results folder into arrays;
- bootstrap: success, amortized time and paired differences, with cluster-bootstrap intervals,
  and the observed Pareto frontier;
- summaries: medians and spread of builds and decisions;
- tables: the numbers behind the figures, as CSV;
- figures: figures A to H;
- runner: the whole analysis of one results folder.

Matplotlib is only needed by `figures` and `runner`, and is imported lazily by the CLI.
"""

from frozen_lake.analysis.bootstrap import (
    Interval,
    PairedDifference,
    SizeResults,
    analyze_size,
    pareto_frontier,
)
from frozen_lake.analysis.runner import (
    DEFAULT_RESAMPLES,
    AnalysisResult,
    InvalidResults,
    analyze_run,
)

__all__ = [
    "DEFAULT_RESAMPLES",
    "AnalysisResult",
    "InvalidResults",
    "Interval",
    "PairedDifference",
    "SizeResults",
    "analyze_run",
    "analyze_size",
    "pareto_frontier",
]
