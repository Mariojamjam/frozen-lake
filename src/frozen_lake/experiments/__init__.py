"""Headless experiment runner: maps, policy builds, paired episodes and raw result files.

One TOML configuration describes the whole experiment. The runner generates every map before
any strategy plays, builds each base policy from scratch several times to time it, then plays
the same episode seeds on the same maps with every strategy, Rollout sample count and the exact
Rollout reference. Everything measured is written as raw rows (CSV) next to the configuration,
the maps and the run metadata, and the written data are checked for completeness.

Modules:
- config: reading and validating the TOML configuration;
- seeds: stable seed derivation;
- matrix: the maps and the arms the experiment consists of;
- runner: playing the experiment;
- records: result file schemas, rows, summary and metadata;
- validation: checking a written results folder.

Nothing here imports Textual or prints inside a timed section.
"""

from frozen_lake.experiments.config import ConfigError, ExperimentConfig, load_config, parse_config
from frozen_lake.experiments.matrix import (
    EXACT,
    SAMPLED,
    Arm,
    MapRecord,
    Workload,
    arms,
    generate_maps,
    workload,
)
from frozen_lake.experiments.records import (
    DECISION_COLUMNS,
    EPISODE_COLUMNS,
    POLICY_COLUMNS,
    SUMMARY_COLUMNS,
)
from frozen_lake.experiments.runner import run_experiment, run_folder_name
from frozen_lake.experiments.seeds import SEED_SCHEME, derive_seed
from frozen_lake.experiments.validation import validate_run

__all__ = [
    "DECISION_COLUMNS",
    "EPISODE_COLUMNS",
    "EXACT",
    "POLICY_COLUMNS",
    "SAMPLED",
    "SEED_SCHEME",
    "SUMMARY_COLUMNS",
    "Arm",
    "ConfigError",
    "ExperimentConfig",
    "MapRecord",
    "Workload",
    "arms",
    "derive_seed",
    "generate_maps",
    "load_config",
    "parse_config",
    "run_experiment",
    "run_folder_name",
    "validate_run",
    "workload",
]
