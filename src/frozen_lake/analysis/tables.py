"""The numbers behind the figures, written as CSV. A blank value means "does not apply"."""

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from frozen_lake.analysis.bootstrap import SizeResults
from frozen_lake.analysis.summaries import (
    COUNTERS,
    BuildSummary,
    DecisionSummary,
    SharedStateTime,
)
from frozen_lake.experiments import EXACT

ARM_COLUMNS = [
    "size", "strategy", "estimator", "rollout_n", "maps", "episodes",
    "success_rate", "success_low", "success_high",
    "amortized_ms", "amortized_low", "amortized_high",
    "mean_agent_ms_per_episode", "build_ms", "mean_simulated_transitions", "on_frontier",
]  # fmt: skip
DIFFERENCE_COLUMNS = [
    "size", "rollout_n", "maps", "delta_success", "delta_success_low", "delta_success_high",
    "delta_amortized_ms", "delta_amortized_low", "delta_amortized_high",
]  # fmt: skip
SENSITIVITY_COLUMNS = ["size", "rollout_n", "amortization_r", "delta_amortized_ms"]
BUILD_COLUMNS = [
    "size", "base_policy", "maps", "builds", "median_build_ms", "q1_build_ms", "q3_build_ms",
    *(f"mean_{name}" for name in COUNTERS),
]  # fmt: skip
DECISION_COLUMNS = [
    "size", "strategy", "rollout_n", "decisions", "median_decision_ms", "q1_decision_ms",
    "q3_decision_ms",
]  # fmt: skip
SHARED_COLUMNS = [
    "size", "rollout_n", "maps", "mean_time_ratio", "min_time_ratio", "max_time_ratio",
]  # fmt: skip


def _blank(value: object) -> object:
    return "" if value is None else value


def _write(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _blank(row.get(key)) for key in columns})
    return path


def write_arm_table(path: Path, results: Sequence[SizeResults]) -> Path:
    rows = []
    for res in results:
        for i, arm in enumerate(res.arms):
            exact = arm.estimator == EXACT
            success = res.success[i]
            amortized = res.amortized_ms[i]
            rows.append({
                "size": res.size,
                "strategy": arm.strategy.value,
                "estimator": arm.estimator,
                "rollout_n": arm.rollout_n,
                "maps": res.maps,
                "episodes": res.episodes,
                "success_rate": success.estimate,
                "success_low": success.low,
                "success_high": success.high,
                "amortized_ms": amortized.estimate if amortized else None,
                "amortized_low": amortized.low if amortized else None,
                "amortized_high": amortized.high if amortized else None,
                "mean_agent_ms_per_episode": res.agent_ms[i],
                "build_ms": res.build_ms[i],
                "mean_simulated_transitions": res.transitions[i],
                "on_frontier": None if exact else int(i in res.frontier),
            })  # fmt: skip
    return _write(path, ARM_COLUMNS, rows)


def write_difference_table(path: Path, results: Sequence[SizeResults]) -> Path:
    rows = []
    for res in results:
        for diff in res.differences:
            time = diff.amortized_ms
            rows.append({
                "size": res.size,
                "rollout_n": diff.rollout_n,
                "maps": res.maps,
                "delta_success": diff.success.estimate,
                "delta_success_low": diff.success.low,
                "delta_success_high": diff.success.high,
                "delta_amortized_ms": time.estimate if time else None,
                "delta_amortized_low": time.low if time else None,
                "delta_amortized_high": time.high if time else None,
            })  # fmt: skip
    return _write(path, DIFFERENCE_COLUMNS, rows)


def write_sensitivity_table(
    path: Path, curves: dict[int, dict[int, list[float]]], r_values: Sequence[int]
) -> Path:
    """`curves` maps a size to {N: Δ amortized time for each value in r_values}."""
    rows = [
        {"size": size, "rollout_n": n, "amortization_r": r, "delta_amortized_ms": delta}
        for size, by_n in curves.items()
        for n, deltas in by_n.items()
        for r, delta in zip(r_values, deltas, strict=True)
    ]
    return _write(path, SENSITIVITY_COLUMNS, rows)


def write_build_table(path: Path, summaries: Sequence[BuildSummary]) -> Path:
    rows = []
    for s in summaries:
        row = {
            "size": s.size,
            "base_policy": s.base,
            "maps": s.maps,
            "builds": s.builds,
            "median_build_ms": s.median_ms,
            "q1_build_ms": s.q1_ms,
            "q3_build_ms": s.q3_ms,
        }
        for name, values in s.counters.items():
            row[f"mean_{name}"] = sum(values) / len(values)
        rows.append(row)
    return _write(path, BUILD_COLUMNS, rows)


def write_decision_table(path: Path, summaries: Sequence[DecisionSummary]) -> Path:
    rows = [
        {
            "size": s.size,
            "strategy": s.strategy,
            "rollout_n": s.rollout_n,
            "decisions": s.decisions,
            "median_decision_ms": s.median_ms,
            "q1_decision_ms": s.q1_ms,
            "q3_decision_ms": s.q3_ms,
        }
        for s in summaries
    ]
    return _write(path, DECISION_COLUMNS, rows)


def write_shared_state_table(path: Path, times: Sequence[SharedStateTime]) -> Path:
    rows = [
        {
            "size": t.size,
            "rollout_n": t.rollout_n,
            "maps": len(t.ratios),
            "mean_time_ratio": t.mean_ratio,
            "min_time_ratio": min(t.ratios),
            "max_time_ratio": max(t.ratios),
        }
        for t in times
    ]
    return _write(path, SHARED_COLUMNS, rows)
