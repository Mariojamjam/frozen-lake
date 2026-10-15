# frozen-lake

**Evaluating the Time-Performance Trade-off of BFS and Expectimax as Base Policies for Monte Carlo Rollout in Stochastic FrozenLake**

This research investigates whether investing more computation in an agent's *initial* decision rule pays off once that rule is refined by Monte Carlo Rollout under uncertainty. The testbed is the slippery Gymnasium [`FrozenLake-v1`](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment, where every intended move can go wrong.

> **Status:** early development. The project shell and tooling are in place; the algorithms, the Textual UI and the benchmark runner are **not implemented yet**, and the CLI subcommands are placeholders.

## Motivation

Rollout is a simple and widely used policy-improvement technique: before acting, simulate what would happen after each possible action if a *base policy* took over afterwards, then pick the most promising action. The quality of those simulations depends on the base policy, and so does their cost.

A cheap base policy, such as one derived from deterministic shortest paths, ignores the risk of slipping. A more sophisticated one, such as depth-limited Expectimax, reasons about stochastic outcomes but costs more to build. It is not obvious whether that extra reasoning still matters once Rollout's own sampling corrects for risk, or whether the cheaper policy delivers comparable success for a fraction of the computation.

## Research question

> How does the choice of base policy (BFS or Expectimax) affect the trade-off between computational time and success rate in Monte Carlo Rollout across stochastic FrozenLake environments of increasing size?

**Objective:** determine whether the potential performance gains of a more sophisticated base policy justify its additional computational cost.

### Hypotheses

These are open questions to be tested, not expected results.

- **H1 — Quality:** with the same number of rollout samples, Rollout over Expectimax may reach higher success rates than Rollout over BFS on some maps, because its base policy accounts for transition risk.
- **H2 — Cost:** building the Expectimax policy requires more work than building the BFS policy; once both are stored, looking up an action costs about the same.
- **H3 — Sampling budget:** more rollout samples per action may improve value estimates at a higher decision cost, but success need not increase monotonically.
- **H4 — Trade-off:** the simpler BFS base may give competitive success per unit of amortized computation, even where Expectimax improves absolute success.
- **H5 — Scaling:** map size, hole layout, horizon and reward sparsity may substantially change the observed differences.

Conclusions will be limited to the measured settings: environment parameters, maps, budgets, heuristic and hardware.

## Approach

The agent moves on an `n × n` grid of frozen tiles (`F`), holes (`H`), a start (`S`) and a goal (`G`). Movement is stochastic: the intended direction succeeds with probability **0.80**, and the agent slips to each perpendicular direction with probability **0.10**. Reaching the goal yields reward 1; every other transition yields 0.

Two **base policies** are precomputed once per map as state → action tables:

| Base policy | Idea | Models slipping? |
|---|---|---|
| **BFS** | Reverse breadth-first search from the goal gives shortest deterministic distances; each state moves toward a neighbor with the smallest distance. | No |
| **Depth-limited Expectimax** | MAX nodes (agent actions) alternate with CHANCE nodes (probability-weighted outcomes), memoized by `(state, remaining_depth)`. Non-terminal leaves are scored with the heuristic `h(s) = β^d_BFS(s, G)`. | Yes |

A single shared **Monte Carlo Rollout** then improves either policy online: at every real state, it tries each of the four actions once, follows the base policy for the rest of `N` simulated trajectories, and picks the action with the highest estimated return. When estimates tie, it keeps the base policy's action.

This yields four experimental strategies:

| ID | Real action rule |
|---|---|
| `BFS` | BFS base policy |
| `EXP` | Expectimax base policy |
| `BFS-RO` | Rollout over the BFS policy |
| `EXP-RO` | Rollout over the Expectimax policy |

The primary metric is **success rate versus amortized compute time per episode**. Supporting metrics are policy build time, paired-state decision time, simulated transitions, and search-specific counters such as visited states, queue peak, cache hits and recursion depth.

## Planned features

- BFS and Expectimax planners written from scratch, not imported from a search library.
- One reusable, planner-agnostic Monte Carlo Rollout implementation.
- A **Textual** terminal UI to step through episodes, compare strategies and inspect per-action estimates.
- A headless, reproducible benchmark runner driven by TOML configs, with seeded maps.
- Raw CSV/JSON results, Matplotlib figures and a short Markdown research report.

## Requirements

- [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 automatically if needed.
- Optional: [just](https://just.systems/) for short task aliases.
- Optional: Docker and VS Code with the Dev Containers extension.

No database, web server, GPU or browser is required.

## Quickstart

```bash
git clone https://github.com/Mariojamjam/frozen-lake.git
cd frozen-lake
uv sync --locked
uv run frozen-lake --help
```

### Planned CLI

The commands below are **targets**. They currently exit with a "not implemented yet" message.

```bash
uv run frozen-lake tui --size 6 --strategy EXP-RO --rollouts 50 --seed 42
uv run frozen-lake episode --size 4 --strategy BFS --seed 42
uv run frozen-lake validate --config configs/pilot.toml
uv run frozen-lake benchmark --config configs/pilot.toml
uv run frozen-lake plot --input results/<run_id>
```

## Development

```bash
uv run pytest -q              # tests
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run ruff format .          # apply formatting
```

The same commands are available as `just` recipes (`just test`, `just lint`, `just format`, `just check`). If `just` is not installed, you can run it without installing:

```bash
uvx --from rust-just just check
```

### Dev Container

Open the folder in VS Code and choose **Dev Containers: Reopen in Container**. The image provides Python 3.12, a pinned `uv` and `just`, and runs `uv sync --locked` on creation.

### Continuous integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request. It installs the locked dependencies, then runs Ruff lint and format checks, pytest, and a CLI smoke test. CI never runs the full scientific benchmark.

## Repository layout

```text
frozen-lake/
├── .devcontainer/        # Dev Container (Dockerfile + devcontainer.json)
├── .github/workflows/    # CI
├── docs/                 # Supplementary documentation
├── src/frozen_lake/      # Python package (import name: frozen_lake)
│   └── cli.py            # Typer entry point for the `frozen-lake` command
├── tests/                # pytest suite
├── pyproject.toml        # Project metadata, dependencies, Ruff/pytest config
├── uv.lock               # Locked dependency versions
└── justfile              # Optional task aliases
```

Modules for the environment model, the planners, rollout, episodes, experiments, plots and the TUI will be added as they are implemented.

## References

- Gymnasium FrozenLake: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>
- Textual: <https://textual.textualize.io/>
- AIMA Python (study reference): <https://github.com/aimacode/aima-python>
