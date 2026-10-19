"""The figures of the analysis plan (design doc §11.4), drawn from the saved results.

Figures are built on bare matplotlib `Figure` objects, without pyplot: no global state, no
display and no backend to choose, so they render the same on a server as on a laptop. They are
never produced by running agents again (§11.4).

    A  success rate vs amortized time, per size          the research question
    B  success rate vs N, with base-only and exact lines  H1, H3
    C  decision time and simulated transitions vs N       H3, cost mechanism
    D  policy build time vs map size                      H2
    E  search work vs map size                            H2, H5
    F  algorithm-specific memory diagnostics              H2
    G  decision time on states both strategies visited    cost without the episode-length confound
    H  amortized-time gap vs R                            how the comparison depends on R (§7.3)
"""

import math
from collections.abc import Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from frozen_lake.analysis.bootstrap import LEVEL, Interval, SizeResults
from frozen_lake.analysis.summaries import BuildSummary, DecisionSummary, SharedStateTime
from frozen_lake.experiments import EXACT, SAMPLED
from frozen_lake.models import StrategyId

# Colorblind-safe pair: blue for everything built on BFS, vermillion for Expectimax.
COLORS = {StrategyId.BFS: "#0072B2", StrategyId.EXP: "#D55E00"}
MARKERS = {StrategyId.BFS: "s", StrategyId.EXP: "D"}
DOTTED = (0, (1, 1.5))
UNCERTAINTY_NOTE = f"Bars and bands: {LEVEL:.0%} cluster-bootstrap intervals (maps, then episodes)"


def _grid(count: int, *, width: float = 4.3, height: float = 3.7, max_cols: int = 3):
    cols = min(count, max_cols)
    rows = math.ceil(count / cols)
    fig = Figure(figsize=(width * cols, height * rows + 0.9), layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False).ravel().tolist()
    for ax in axes[count:]:
        ax.set_visible(False)
    return fig, axes[:count]


def _legend(fig: Figure, ax: Axes, *, ncol: int = 4) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="outside lower center",
        ncol=min(len(unique), ncol),
        fontsize=8,
        frameon=False,
    )


def _arms(res: SizeResults, strategy: StrategyId, estimator: str | None) -> list[int]:
    """Arm indices of a strategy and estimator; sampled Rollout arms come sorted by N."""
    found = [
        i
        for i, arm in enumerate(res.arms)
        if arm.strategy is strategy and arm.estimator == estimator
    ]
    return sorted(found, key=lambda i: res.arms[i].rollout_n or 0)


def _error(intervals: Sequence[Interval]) -> np.ndarray:
    """Asymmetric error bars (2, n); a percentile interval can miss its own estimate."""
    estimate = np.array([i.estimate for i in intervals])
    return np.array([
        np.maximum(estimate - [i.low for i in intervals], 0),
        np.maximum([i.high for i in intervals] - estimate, 0),
    ])  # fmt: skip


def _time(res: SizeResults, arm: int) -> Interval:
    """Amortized time of an arm that has one (every arm but the exact reference)."""
    interval = res.amortized_ms[arm]
    assert interval is not None
    return interval


def _size_title(res: SizeResults) -> str:
    return f"{res.size}×{res.size} · {res.maps} maps × {res.episodes} episodes"


def _sample_ticks(ax: Axes, values: Sequence[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(values), labels=[str(v) for v in values])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7)


def success_vs_time(results: Sequence[SizeResults]) -> Figure:
    """Figure A: success rate against amortized time per episode, one point per arm."""
    fig, axes = _grid(len(results))
    for ax, res in zip(axes, results, strict=True):
        for strategy in (StrategyId.BFS_RO, StrategyId.EXP_RO):
            color = COLORS[strategy.base]
            sampled = _arms(res, strategy, SAMPLED)
            if sampled:
                time = [_time(res, i) for i in sampled]
                success = [res.success[i] for i in sampled]
                ax.errorbar(
                    [t.estimate for t in time],
                    [s.estimate for s in success],
                    xerr=_error(time),
                    yerr=_error(success),
                    marker="o", ms=4, lw=1.2, elinewidth=0.7, color=color, label=strategy.value,
                )  # fmt: skip
                for k in (0, -1):
                    ax.annotate(
                        str(res.arms[sampled[k]].rollout_n),
                        (time[k].estimate, success[k].estimate),
                        textcoords="offset points", xytext=(4, -10), fontsize=6, color=color,
                    )  # fmt: skip
            for i in _arms(res, strategy, EXACT):
                ax.axhline(
                    res.success[i].estimate, color=color, ls="--", lw=1,
                    label=f"{strategy.value} exact (N = ∞)",
                )  # fmt: skip
        for strategy in (StrategyId.BFS, StrategyId.EXP):
            for i in _arms(res, strategy, None):
                time, success = _time(res, i), res.success[i]
                ax.errorbar(
                    [time.estimate],
                    [success.estimate],
                    xerr=_error([time]),
                    yerr=_error([success]),
                    marker=MARKERS[strategy],
                    ms=6,
                    elinewidth=0.7,
                    ls="none",
                    color=COLORS[strategy],
                    label=strategy.value,
                )
        for i in res.frontier:
            time, success = _time(res, i), res.success[i]
            ax.scatter(
                [time.estimate], [success.estimate], s=140, facecolors="none",
                edgecolors="black", linewidths=1.1, zorder=5, label="observed Pareto frontier",
            )  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(_size_title(res), fontsize=9)
        ax.set_xlabel("amortized time per episode (ms)", fontsize=8)
        ax.set_ylabel("success rate", fontsize=8)
        ax.grid(alpha=0.25)
    _legend(fig, axes[0])
    fig.suptitle(
        f"A · Success rate vs amortized time per episode (R = E)\n{UNCERTAINTY_NOTE}", fontsize=10
    )
    return fig


def success_vs_samples(results: Sequence[SizeResults]) -> Figure:
    """Figure B: success rate against N, with base-only and exact lines for reference."""
    fig, axes = _grid(len(results))
    for ax, res in zip(axes, results, strict=True):
        for strategy in (StrategyId.BFS_RO, StrategyId.EXP_RO):
            color = COLORS[strategy.base]
            sampled = _arms(res, strategy, SAMPLED)
            ns = [res.arms[i].rollout_n for i in sampled]
            success = [res.success[i] for i in sampled]
            ax.plot(ns, [s.estimate for s in success], marker="o", ms=4, color=color,
                    label=strategy.value)  # fmt: skip
            ax.fill_between(ns, [s.low for s in success], [s.high for s in success],
                            color=color, alpha=0.18, lw=0)  # fmt: skip
            for i in _arms(res, strategy, EXACT):
                ax.axhline(res.success[i].estimate, color=color, ls="--", lw=1,
                           label=f"{strategy.value} exact (N = ∞)")  # fmt: skip
        for strategy in (StrategyId.BFS, StrategyId.EXP):
            for i in _arms(res, strategy, None):
                ax.axhline(res.success[i].estimate, color=COLORS[strategy], ls=DOTTED, lw=1.3,
                           label=f"{strategy.value} (base only)")  # fmt: skip
        _sample_ticks(ax, sorted({n for n in (a.rollout_n for a in res.arms) if n}))
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(_size_title(res), fontsize=9)
        ax.set_xlabel("N, simulated trajectories per action", fontsize=8)
        ax.set_ylabel("success rate", fontsize=8)
        ax.grid(alpha=0.25)
    _legend(fig, axes[0], ncol=3)
    fig.suptitle(f"B · Success rate vs sampling budget N\n{UNCERTAINTY_NOTE}", fontsize=10)
    return fig


def cost_vs_samples(
    results: Sequence[SizeResults], decisions: Sequence[DecisionSummary] | None
) -> Figure:
    """Figure C: decision time and simulated transitions against N.

    The top row shows the median decision time with its interquartile range when the run logged
    its decisions. Otherwise it shows the mean agent time per episode, which also depends on how
    long the episodes last.
    """
    cols = len(results)
    fig = Figure(figsize=(min(3.8 * cols, 19), 8.2), layout="constrained")
    axes = fig.subplots(2, cols, squeeze=False)
    by_key = {(d.size, d.strategy, d.rollout_n): d for d in decisions or []}
    for col, res in enumerate(results):
        top, bottom = axes[0, col], axes[1, col]
        ns = sorted({a.rollout_n for a in res.arms if a.estimator == SAMPLED and a.rollout_n})
        for strategy in (StrategyId.BFS_RO, StrategyId.EXP_RO):
            color = COLORS[strategy.base]
            sampled = _arms(res, strategy, SAMPLED)
            if decisions:
                rows = [by_key[(res.size, strategy.value, n)] for n in ns]
                top.plot(ns, [d.median_ms for d in rows], marker="o", ms=4, color=color,
                         label=strategy.value)  # fmt: skip
                top.fill_between(ns, [d.q1_ms for d in rows], [d.q3_ms for d in rows],
                                 color=color, alpha=0.18, lw=0)  # fmt: skip
            else:
                top.plot(ns, [res.agent_ms[i] for i in sampled], marker="o", ms=4, color=color,
                         label=strategy.value)  # fmt: skip
            bottom.plot(ns, [res.transitions[i] for i in sampled], marker="o", ms=4, color=color,
                        label=strategy.value)  # fmt: skip
        if decisions:
            for strategy in (StrategyId.BFS, StrategyId.EXP):
                row = by_key.get((res.size, strategy.value, None))
                if row:
                    top.axhline(row.median_ms, color=COLORS[strategy], ls=DOTTED, lw=1.3,
                                label=f"{strategy.value} (table lookup)")  # fmt: skip
        for ax in (top, bottom):
            _sample_ticks(ax, ns)
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
        top.set_title(f"{res.size}×{res.size}", fontsize=9)
        top.set_ylabel(
            "median decision time (ms)" if decisions else "agent time per episode (ms)", fontsize=8
        )
        bottom.set_ylabel("simulated transitions per episode", fontsize=8)
        bottom.set_xlabel("N, simulated trajectories per action", fontsize=8)
    _legend(fig, axes[0, 0], ncol=4)
    note = "Bands: interquartile range of decisions" if decisions else "Decisions were not logged"
    fig.suptitle(f"C · Cost against sampling budget N\n{note}", fontsize=10)
    return fig


def _positions(sizes: Sequence[int]) -> list[int]:
    return sorted(set(sizes))


def build_time_vs_size(builds: Sequence[BuildSummary]) -> Figure:
    """Figure D: how long building each base policy takes as the map grows."""
    fig = Figure(figsize=(6.6, 4.8), layout="constrained")
    ax = fig.subplots()
    sizes = _positions([b.size for b in builds])
    for base, shift in ((StrategyId.BFS, -0.12), (StrategyId.EXP, 0.12)):
        rows = [b for b in builds if b.base == base.value]
        if not rows:
            continue
        color = COLORS[base]
        x = [b.size + shift for b in rows]
        median = np.array([b.median_ms for b in rows])
        ax.errorbar(
            x, median,
            yerr=[median - [b.q1_ms for b in rows], [b.q3_ms for b in rows] - median],
            marker=MARKERS[base], ms=6, capsize=3, lw=1.3, color=color,
            label=f"{base.value}: median and IQR of all timed builds",
        )  # fmt: skip
        # One scatter for every size at once, so the legend gets one entry per base policy.
        ax.scatter(
            [b.size + shift for b in rows for _ in b.per_map_median_ms],
            [ms for b in rows for ms in b.per_map_median_ms],
            s=14,
            color=color,
            alpha=0.35,
            linewidths=0,
            label=f"{base.value}: map medians",
        )
    ax.set_yscale("log")
    ax.set_xticks(sizes, labels=[f"{n}×{n}" for n in sizes])
    ax.set_xlabel("map size", fontsize=9)
    ax.set_ylabel("policy build time (ms)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("D · Policy build time vs map size", fontsize=10)
    return fig


def _counter_line(
    ax: Axes, rows: Sequence[BuildSummary], name: str, color: str, marker: str, label: str
) -> bool:
    with_counter = [row for row in rows if name in row.counters]
    if not with_counter:
        return False
    x = [row.size for row in with_counter]
    values = [row.counters[name] for row in with_counter]
    ax.plot(x, [float(np.mean(v)) for v in values], marker=marker, ms=4, color=color, label=label)
    ax.fill_between(x, [min(v) for v in values], [max(v) for v in values], color=color,
                    alpha=0.18, lw=0)  # fmt: skip
    return True


def _size_axis(ax: Axes, builds: Sequence[BuildSummary]) -> None:
    sizes = _positions([b.size for b in builds])
    ax.set_xticks(sizes, labels=[f"{n}×{n}" for n in sizes])
    ax.set_xlabel("map size", fontsize=8)
    ax.grid(alpha=0.25)


def search_work_vs_size(builds: Sequence[BuildSummary]) -> Figure:
    """Figure E: how much each search does as the map grows, per algorithm and counter."""
    fig = Figure(figsize=(10.5, 4.6), layout="constrained")
    axes = fig.subplots(1, 2)
    panels = (
        ("BFS", "BFS", (("states_evaluated", "#009E73", "o", "states evaluated"),
                        ("successors_examined", "#CC79A7", "s", "successors examined"))),
        ("EXP", "Expectimax", (("states_evaluated", "#009E73", "o", "states evaluated"),
                               ("successors_examined", "#CC79A7", "s", "successors examined"),
                               ("cache_hits", "#E69F00", "^", "cache hits"))),
    )  # fmt: skip
    for ax, (base, title, counters) in zip(axes, panels, strict=True):
        rows = [b for b in builds if b.base == base]
        for name, color, marker, label in counters:
            _counter_line(ax, rows, name, color, marker, label)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("count per build (mean over maps)", fontsize=8)
        _size_axis(ax, builds)
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("E · Search work vs map size\nBands: minimum to maximum over maps", fontsize=10)
    return fig


def memory_diagnostics(builds: Sequence[BuildSummary]) -> Figure:
    """Figure F: each algorithm's own memory measure. They are never combined into one number."""
    fig = Figure(figsize=(14.5, 4.4), layout="constrained")
    axes = fig.subplots(1, 3)
    panels = (
        ("BFS", "peak_queue", "BFS · peak queue length", "#0072B2", False),
        ("EXP", "peak_recursion", "Expectimax · peak recursion depth", "#D55E00", False),
        ("EXP", "cache_entries", "Expectimax · cache entries", "#D55E00", True),
    )
    for ax, (base, name, title, color, log) in zip(axes, panels, strict=True):
        rows = [b for b in builds if b.base == base]
        if not _counter_line(ax, rows, name, color, "o", title):
            ax.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=ax.transAxes)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        _size_axis(ax, builds)
    fig.suptitle(
        "F · Memory diagnostics (different quantities: never compared as one size)\n"
        "Bands: minimum to maximum over maps",
        fontsize=10,
    )
    return fig


def shared_state_time(times: Sequence[SharedStateTime]) -> Figure:
    """Figure G: EXP-RO decision time over BFS-RO decision time, on states both visited."""
    fig = Figure(figsize=(7.0, 4.8), layout="constrained")
    ax = fig.subplots()
    sizes = sorted({t.size for t in times})
    cmap = colormaps["viridis"]
    for k, size in enumerate(sizes):
        color = cmap(k / max(len(sizes) - 1, 1) * 0.85)
        rows = sorted((t for t in times if t.size == size), key=lambda t: t.rollout_n)
        ns = [t.rollout_n for t in rows]
        ax.plot(ns, [t.mean_ratio for t in rows], marker="o", ms=4, color=color,
                label=f"{size}×{size}")  # fmt: skip
        for t in rows:
            ax.scatter([t.rollout_n] * len(t.ratios), t.ratios, s=10, color=color, alpha=0.35,
                       linewidths=0)  # fmt: skip
    ax.axhline(1.0, color="gray", lw=0.9)
    _sample_ticks(ax, sorted({t.rollout_n for t in times}))
    ax.set_yscale("log")
    ax.set_xlabel("N, simulated trajectories per action", fontsize=9)
    ax.set_ylabel("EXP-RO ÷ BFS-RO decision time", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, title="map size", title_fontsize=8)
    fig.suptitle(
        "G · Decision time on states both strategies visited\n"
        "Line: mean of the per-map ratios; dots: one map each. Above 1: EXP-RO is slower",
        fontsize=10,
    )
    return fig


def amortization_sensitivity(
    results: Sequence[SizeResults],
    curves: dict[int, dict[int, list[float]]],
    r_values: Sequence[int],
) -> Figure:
    """Figure H: how the EXP-RO minus BFS-RO amortized time changes with R."""
    fig, axes = _grid(len(results))
    cmap = colormaps["viridis"]
    for ax, res in zip(axes, results, strict=True):
        by_n = curves[res.size]
        for k, n in enumerate(sorted(by_n)):
            ax.plot(r_values, by_n[n], color=cmap(k / max(len(by_n) - 1, 1) * 0.85), lw=1.2,
                    label=f"N = {n}")  # fmt: skip
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(res.amortization_r, color="black", ls=":", lw=1, label=f"R = E = {res.episodes}")
        ax.set_xscale("log")
        ax.set_title(f"{res.size}×{res.size}", fontsize=9)
        ax.set_xlabel("R, episodes sharing one policy build", fontsize=8)
        ax.set_ylabel("EXP-RO − BFS-RO amortized time (ms)", fontsize=8)
        ax.grid(alpha=0.25)
    _legend(fig, axes[0], ncol=4)
    fig.suptitle(
        "H · Amortized-time gap vs R (point estimates)\nAbove 0: EXP-RO costs more per episode",
        fontsize=10,
    )
    return fig
