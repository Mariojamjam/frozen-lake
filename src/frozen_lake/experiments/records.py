"""Result files: their columns, how rows are built, the summary and the run metadata."""

import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

from frozen_lake.experiments.config import ExperimentConfig, PolicySettings
from frozen_lake.experiments.matrix import EXACT, MapRecord, workload
from frozen_lake.experiments.seeds import SEED_SCHEME
from frozen_lake.models import Action, PolicyBuildResult, StrategyId

POLICY_COLUMNS = [
    "map_id", "size", "base_policy", "depth", "beta", "build_repeat", "build_time_ns",
    "states_evaluated", "successors_examined", "peak_queue", "peak_recursion",
    "cache_hits", "cache_entries", "fallback_states",
]  # fmt: skip
EPISODE_COLUMNS = [
    "map_id", "size", "strategy", "estimator", "rollout_n", "episode_index", "episode_seed",
    "rollout_seed", "success", "termination_reason", "steps", "total_reward",
    "agent_compute_ns", "wall_ns", "simulated_transitions",
]  # fmt: skip
DECISION_COLUMNS = [
    "map_id", "size", "strategy", "estimator", "rollout_n", "episode_index", "step", "state",
    "base_action", "chosen_action", "overrode", "decision_time_ns", "simulated_transitions",
    "trajectories", "q_left", "q_down", "q_right", "q_up",
]  # fmt: skip
SUMMARY_COLUMNS = [
    "size", "strategy", "estimator", "rollout_n", "maps", "episodes", "success_rate",
    "median_decision_ms", "mean_episode_agent_ms", "median_build_ms",
    "amortized_ms_per_episode", "mean_simulated_transitions",
]  # fmt: skip

UNITS = {
    "*_ns": "nanoseconds",
    "*_ms": "milliseconds",
    "success": "1 if the goal was reached, else 0",
    "rollout_n": "simulated trajectories per candidate action; blank where it does not apply",
    "estimator": "sampled or exact for Rollout strategies; blank for base-only strategies",
    "amortized_ms_per_episode": "median build time / episodes per map + mean agent time",
}


def policy_row(
    record: MapRecord, base: StrategyId, repeat: int, build: PolicyBuildResult, pol: PolicySettings
) -> dict:
    diag = build.diagnostics
    is_exp = base is StrategyId.EXP
    return {
        "map_id": record.map_id,
        "size": record.size,
        "base_policy": base.value,
        "depth": pol.expectimax_depth if is_exp else "",
        "beta": pol.heuristic_beta if is_exp else "",
        "build_repeat": repeat,
        "build_time_ns": build.build_time_ns,
        "states_evaluated": build.states_evaluated,
        "successors_examined": build.successors_examined,
        "peak_queue": diag.get("peak_queue_length", ""),
        "peak_recursion": diag.get("peak_recursion", ""),
        "cache_hits": diag.get("cache_hits", ""),
        "cache_entries": diag.get("cache_entries", ""),
        "fallback_states": diag.get("fallback_states", ""),
    }


def blank(value: object) -> object:
    return "" if value is None else value


def decision_row(episode_row: dict, step: int, decision) -> dict:
    values = decision.estimated_action_values or {}
    return {
        **{
            key: episode_row[key]
            for key in ("map_id", "size", "strategy", "estimator", "rollout_n", "episode_index")
        },
        "step": step,
        "state": decision.state,
        "base_action": decision.base_action.name,
        "chosen_action": decision.action.name,
        "overrode": int(decision.action is not decision.base_action),
        "decision_time_ns": decision.compute_time_ns,
        "simulated_transitions": decision.simulated_transitions,
        "trajectories": decision.trajectories_evaluated,
        "q_left": blank(values.get(Action.LEFT)),
        "q_down": blank(values.get(Action.DOWN)),
        "q_right": blank(values.get(Action.RIGHT)),
        "q_up": blank(values.get(Action.UP)),
    }


def write_summary(
    run_dir: Path,
    config: ExperimentConfig,
    episodes_rows: list[dict],
    build_times: dict[tuple[int, str], list[int]],
    decision_times: dict[tuple, list[int]],
) -> None:
    """One row per size, strategy, estimator and N. Uncertainty is left to the analysis.

    The exact reference samples nothing, so its cost columns are left blank: they are not
    comparable with the strategies' costs.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in episodes_rows:
        groups[(row["size"], row["strategy"], row["estimator"], row["rollout_n"])].append(row)

    episodes_per_map = config.experiment.episodes_per_map
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for key, rows in groups.items():
            size, strategy, estimator, rollout_n = key
            exact = estimator == EXACT
            build_ms = statistics.median(build_times[(size, StrategyId(strategy).base.value)]) / 1e6
            agent_ms = statistics.fmean(r["agent_compute_ns"] for r in rows) / 1e6
            writer.writerow({
                "size": size,
                "strategy": strategy,
                "estimator": estimator,
                "rollout_n": rollout_n,
                "maps": len({r["map_id"] for r in rows}),
                "episodes": len(rows),
                "success_rate": statistics.fmean(r["success"] for r in rows),
                "median_decision_ms": "" if exact else statistics.median(decision_times[key]) / 1e6,
                "mean_episode_agent_ms": "" if exact else agent_ms,
                "median_build_ms": build_ms,
                "amortized_ms_per_episode": "" if exact else build_ms / episodes_per_map + agent_ms,
                "mean_simulated_transitions": statistics.fmean(
                    r["simulated_transitions"] for r in rows
                ),
            })  # fmt: skip


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def map_dict(record: MapRecord) -> dict:
    return {**asdict(record), "rows": list(record.rows)}


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_metadata(
    config: ExperimentConfig,
    run_dir: Path,
    started: datetime,
    finished: datetime,
    config_path: str | Path | None,
) -> dict:
    status = _git("status", "--porcelain")
    return {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "command": sys.argv,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config.fingerprint(),
        "maps_sha256": hashlib.sha256((run_dir / "maps.json").read_bytes()).hexdigest(),
        "seed_scheme": SEED_SCHEME,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "packages": {
            name: _package_version(name)
            for name in ("frozen-lake", "gymnasium", "numpy", "typer", "textual", "matplotlib")
        },
        "units": UNITS,
        "workload": asdict(workload(config)),
    }
