"""Playing the experiment: timed policy builds, then every arm on the same maps and seeds."""

import csv
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from frozen_lake.environment import create_environment, extract_transitions
from frozen_lake.episodes import Decide, EpisodeSession, policy_decider, run_episode
from frozen_lake.experiments.config import ExperimentConfig
from frozen_lake.experiments.matrix import EXACT, SAMPLED, MapRecord, arms, generate_maps
from frozen_lake.experiments.records import (
    DECISION_COLUMNS,
    EPISODE_COLUMNS,
    POLICY_COLUMNS,
    blank,
    decision_row,
    map_dict,
    policy_row,
    run_metadata,
    write_json,
    write_summary,
)
from frozen_lake.experiments.seeds import derive_seed
from frozen_lake.experiments.validation import validate_run
from frozen_lake.models import Policy, StrategyId
from frozen_lake.policies import BASE_STRATEGIES, build_base_policy
from frozen_lake.rollout import exact_rollout_decider, rollout_decider


def run_folder_name(config: ExperimentConfig, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{config.fingerprint()[:8]}"


def build_timed_policies(
    config: ExperimentConfig, record: MapRecord, model, policy_writer: "_TimingTap"
) -> dict[StrategyId, Policy]:
    """Build each base policy from scratch warm-up + K times, write every timed build, return it.

    Builds are deterministic, so every repeat must produce the same table; a difference would
    be a bug, and the run stops.
    """
    pol = config.policies
    policies: dict[StrategyId, Policy] = {}
    for base in BASE_STRATEGIES:
        build = None
        for _ in range(pol.build_warmup):
            build_base_policy(
                base, record.rows, model,
                depth=pol.expectimax_depth, beta=pol.heuristic_beta, gamma=pol.discount_factor,
            )  # fmt: skip
        for repeat in range(pol.build_repeats):
            result = build_base_policy(
                base, record.rows, model,
                depth=pol.expectimax_depth, beta=pol.heuristic_beta, gamma=pol.discount_factor,
            )  # fmt: skip
            if build is not None and result.policy != build.policy:
                raise RuntimeError(f"{base} policy for map {record.map_id} differs between builds")
            build = result
            policy_writer.writerow(policy_row(record, base, repeat, result, pol))
        policies[base] = build.policy
    return policies


def run_experiment(
    config: ExperimentConfig,
    *,
    output_root: str | Path | None = None,
    config_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run the whole experiment and return the folder holding its results.

    The folder is new (named after the start time and the configuration hash); an existing
    folder is never overwritten. Progress messages are emitted between maps, never while an
    episode or a build is being timed.
    """
    started = datetime.now(UTC)
    root = Path(output_root if output_root is not None else config.experiment.output_dir)
    run_dir = root / run_folder_name(config, started.astimezone())
    run_dir.mkdir(parents=True, exist_ok=False)
    say = progress or (lambda _message: None)

    maps = generate_maps(config)
    for record in maps:
        if not record.reachable:
            raise RuntimeError(f"map {record.map_id} has no path from start to goal")
    write_json(run_dir / "configuration.json", config.as_dict())
    write_json(run_dir / "maps.json", [map_dict(record) for record in maps])

    gamma = config.policies.discount_factor
    master = config.experiment.master_seed
    experiment_arms = arms(config)
    episodes_rows: list[dict] = []
    build_times: dict[tuple[int, str], list[int]] = defaultdict(list)
    decision_times: dict[tuple, list[int]] = defaultdict(list)

    files = [run_dir / "policy_metrics.csv", run_dir / "episode_metrics.csv"]
    if config.experiment.write_decisions:
        files.append(run_dir / "decision_metrics.csv")
    handles = [path.open("w", newline="", encoding="utf-8") for path in files]
    try:
        policy_writer = csv.DictWriter(handles[0], fieldnames=POLICY_COLUMNS)
        episode_writer = csv.DictWriter(handles[1], fieldnames=EPISODE_COLUMNS)
        decision_writer = (
            csv.DictWriter(handles[2], fieldnames=DECISION_COLUMNS)
            if config.experiment.write_decisions
            else None
        )
        for writer in (policy_writer, episode_writer, decision_writer):
            if writer is not None:
                writer.writeheader()

        for position, record in enumerate(maps, start=1):
            max_steps = config.max_steps(record.size)
            horizon = config.horizon(record.size)
            model = extract_transitions(
                create_environment(
                    record.rows, max_steps=max_steps, success_rate=config.environment.success_rate
                )
            )
            # Policy builds are timed here, before any episode, with nothing printed meanwhile.
            policies = build_timed_policies(
                config, record, model, _TimingTap(policy_writer, build_times, record.size)
            )
            exact = {}
            if config.rollout.exact_reference:
                for base in BASE_STRATEGIES:
                    exact[base] = exact_rollout_decider(
                        policies[base], model, horizon=horizon, gamma=gamma
                    )
            session = EpisodeSession(
                record.rows, max_steps=max_steps, success_rate=config.environment.success_rate
            )
            for episode_index in range(config.experiment.episodes_per_map):
                episode_seed = derive_seed(
                    master, "episode", record.size, record.index, episode_index
                )
                for arm in experiment_arms:
                    policy = policies[arm.strategy.base]
                    rollout_seed = None
                    decide: Decide
                    if arm.estimator == SAMPLED:
                        rollout_seed = derive_seed(
                            master, "rollout", record.size, record.index, episode_index,
                            arm.strategy.value, arm.rollout_n,
                        )  # fmt: skip
                        decide = rollout_decider(
                            policy, model, samples=arm.rollout_n, horizon=horizon, gamma=gamma,
                            rng=random.Random(rollout_seed),
                        )  # fmt: skip
                    elif arm.estimator == EXACT:
                        decide = exact[arm.strategy.base]
                    else:
                        decide = policy_decider(policy)
                    result = run_episode(session, decide, seed=episode_seed, strategy=arm.strategy)

                    key = (
                        record.size,
                        arm.strategy.value,
                        blank(arm.estimator),
                        blank(arm.rollout_n),
                    )
                    row = {
                        "map_id": record.map_id,
                        "size": record.size,
                        "strategy": arm.strategy.value,
                        "estimator": blank(arm.estimator),
                        "rollout_n": blank(arm.rollout_n),
                        "episode_index": episode_index,
                        "episode_seed": episode_seed,
                        "rollout_seed": blank(rollout_seed),
                        "success": int(result.success),
                        "termination_reason": result.termination_reason.value,
                        "steps": result.steps,
                        "total_reward": result.total_reward,
                        "agent_compute_ns": result.agent_compute_ns,
                        "wall_ns": result.wall_ns,
                        "simulated_transitions": result.simulated_transitions,
                    }
                    episode_writer.writerow(row)
                    episodes_rows.append(row)
                    for step, decision in enumerate(result.decisions, start=1):
                        decision_times[key].append(decision.compute_time_ns)
                        if decision_writer is not None:
                            decision_writer.writerow(decision_row(row, step, decision))
            session.close()
            say(
                f"[{position}/{len(maps)}] map {record.map_id} done "
                f"({config.experiment.episodes_per_map * len(experiment_arms)} episodes)"
            )
    finally:
        for handle in handles:
            handle.close()

    write_summary(run_dir, config, episodes_rows, build_times, decision_times)
    finished = datetime.now(UTC)
    write_json(
        run_dir / "metadata.json", run_metadata(config, run_dir, started, finished, config_path)
    )
    problems = validate_run(run_dir)
    if problems:
        raise RuntimeError("written results failed validation:\n- " + "\n- ".join(problems))
    return run_dir


class _TimingTap:
    """Writes policy rows and keeps their build times for the summary."""

    def __init__(self, writer: csv.DictWriter, times: dict, size: int) -> None:
        self.writer, self.times, self.size = writer, times, size

    def writerow(self, row: dict) -> None:
        self.writer.writerow(row)
        self.times[(self.size, row["base_policy"])].append(row["build_time_ns"])
