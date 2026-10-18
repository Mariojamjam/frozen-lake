"""The experiment configuration: one TOML file, read and validated in full before any run."""

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from frozen_lake.policies.expectimax import validate_parameters


class ConfigError(ValueError):
    """An experiment configuration that cannot be run, with every problem found."""


@dataclass(frozen=True)
class EnvironmentSettings:
    sizes: tuple[int, ...]
    frozen_probability: float
    success_rate: float
    max_steps_multiplier: int


@dataclass(frozen=True)
class PolicySettings:
    expectimax_depth: int
    heuristic_beta: float
    discount_factor: float
    build_repeats: int
    build_warmup: int


@dataclass(frozen=True)
class RolloutSettings:
    samples_per_action: tuple[int, ...]
    horizon_multiplier: int
    exact_reference: bool


@dataclass(frozen=True)
class RunSettings:
    maps_per_size: int
    episodes_per_map: int
    master_seed: int
    output_dir: str
    write_decisions: bool


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentSettings
    policies: PolicySettings
    rollout: RolloutSettings
    experiment: RunSettings

    def max_steps(self, size: int) -> int:
        return self.environment.max_steps_multiplier * size * size

    def horizon(self, size: int) -> int:
        return self.rollout.horizon_multiplier * size * size

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of the effective configuration: equal hashes mean identical settings."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


_SECTIONS = {
    "environment": (EnvironmentSettings, {
        "sizes": list, "frozen_probability": float, "success_rate": float,
        "max_steps_multiplier": int,
    }),
    "policies": (PolicySettings, {
        "expectimax_depth": int, "heuristic_beta": float, "discount_factor": float,
        "build_repeats": int, "build_warmup": int,
    }),
    "rollout": (RolloutSettings, {
        "samples_per_action": list, "horizon_multiplier": int, "exact_reference": bool,
    }),
    "experiment": (RunSettings, {
        "maps_per_size": int, "episodes_per_map": int, "master_seed": int,
        "output_dir": str, "write_decisions": bool,
    }),
}  # fmt: skip


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate an experiment configuration; raise ConfigError listing every problem."""
    try:
        raw = tomllib.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {path}: {error}") from error
    return parse_config(raw)


def parse_config(raw: dict) -> ExperimentConfig:
    problems: list[str] = []
    for unknown in sorted(set(raw) - set(_SECTIONS)):
        problems.append(f"unknown section [{unknown}]")

    sections = {}
    for name, (settings_class, fields) in _SECTIONS.items():
        table = raw.get(name)
        if not isinstance(table, dict):
            problems.append(f"missing section [{name}]")
            continue
        for unknown in sorted(set(table) - set(fields)):
            problems.append(f"unknown key {name}.{unknown}")
        values = {}
        for key, expected in fields.items():
            if key not in table:
                problems.append(f"missing key {name}.{key}")
                continue
            value = table[key]
            # TOML integers are acceptable where a float is expected; booleans never are numbers.
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                problems.append(f"{name}.{key} must be {expected.__name__}, got {value!r}")
                continue
            values[key] = tuple(value) if expected is list else value
        if len(values) == len(fields):
            sections[name] = settings_class(**values)

    if problems:
        raise ConfigError("invalid experiment configuration:\n- " + "\n- ".join(problems))
    config = ExperimentConfig(**sections)
    problems = _value_problems(config)
    if problems:
        raise ConfigError("invalid experiment configuration:\n- " + "\n- ".join(problems))
    return config


def _value_problems(config: ExperimentConfig) -> list[str]:
    problems = []
    env, pol, ro, run = config.environment, config.policies, config.rollout, config.experiment

    def int_list(name: str, values: tuple, minimum: int) -> None:
        if not values:
            problems.append(f"{name} must not be empty")
        elif any(not isinstance(v, int) or isinstance(v, bool) or v < minimum for v in values):
            problems.append(f"{name} must hold integers >= {minimum}, got {list(values)}")
        elif len(set(values)) != len(values):
            problems.append(f"{name} must not repeat values, got {list(values)}")

    int_list("environment.sizes", env.sizes, 2)
    int_list("rollout.samples_per_action", ro.samples_per_action, 1)
    if not 0.0 < env.frozen_probability <= 1.0:
        problems.append("environment.frozen_probability must be in (0, 1]")
    if not 0.0 < env.success_rate <= 1.0:
        problems.append("environment.success_rate must be in (0, 1]")
    for name, value in [
        ("environment.max_steps_multiplier", env.max_steps_multiplier),
        ("rollout.horizon_multiplier", ro.horizon_multiplier),
        ("policies.build_repeats", pol.build_repeats),
        ("experiment.maps_per_size", run.maps_per_size),
        ("experiment.episodes_per_map", run.episodes_per_map),
    ]:
        if value < 1:
            problems.append(f"{name} must be at least 1, got {value}")
    if pol.build_warmup < 0:
        problems.append(f"policies.build_warmup must be at least 0, got {pol.build_warmup}")
    if run.master_seed < 0:
        problems.append(f"experiment.master_seed must be at least 0, got {run.master_seed}")
    try:
        validate_parameters(pol.expectimax_depth, pol.heuristic_beta, pol.discount_factor)
    except ValueError as error:
        problems.append(f"policies: {error}")
    return problems
