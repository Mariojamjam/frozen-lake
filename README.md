# frozen-lake

**Evaluating the Time-Performance Trade-off of BFS and Expectimax as Base Policies for Monte Carlo Rollout in Stochastic FrozenLake**

This research investigates whether investing more computation in an agent's *initial* decision rule pays off once that rule is refined by Monte Carlo Rollout under uncertainty. The testbed is the slippery Gymnasium [`FrozenLake-v1`](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment, where every intended move can go wrong.

> **Status:** early development. The validated environment model, the BFS and depth-limited Expectimax base policies, Monte Carlo Rollout over either of them, headless episodes, the terminal explorer and the experiment runner are in place.

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
- Raw CSV/JSON results, and an analysis step (`plot`) with bootstrap intervals, tables and Matplotlib figures.
- A short Markdown research report.

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

Every command, option, key and shortcut is documented in [`docs/commands.md`](docs/commands.md). The design of the study is described in [`docs/experiment-design.md`](docs/experiment-design.md).

### Explore the lake

```bash
uv run frozen-lake tui --size 6 --map-seed 42 --episode-seed 42
```

Choose an intended direction with the arrow keys and press **Space** to take one real step. The ice may slip the agent sideways (10% to each side), and the screen shows where the agent meant to go and where it actually landed. **R** replays the same episode seed, so the same actions reproduce the same slips; **N** tries the next episode seed (new slips on the same lake); **G** generates the next map seed (a new lake, with its own BFS policy); **?** shows help; **Q** quits. Letter keys work in either case. Run `uv run frozen-lake tui --help` for every option (map size, seeds, frozen-tile probability, step limit, intended-move probability).

The explorer starts in **manual** mode, where the arrow keys choose. Press **1** for the **BFS policy**, **2** for the **Expectimax policy**, **3** for **Rollout over BFS** and **4** for **Rollout over Expectimax**, and **M** to return to manual; switching mode restarts the episode. **A** plays the episode automatically. In every mode the status panel shows what each base policy recommends from the current tile; in Rollout modes a table shows every action's estimate and whether Rollout overrode its base policy. **P** cycles the arrows drawn on the lake between off, the BFS policy and the Expectimax policy: one recommended move for every tile, not a single path.

### Play a headless episode

```bash
uv run frozen-lake episode --strategy BFS --size 6 --map-seed 42 --episode-seed 7
```

Prints every real step (state, action, where the agent landed, whether it slipped) and a summary with the ending (`goal`, `hole` or `timeout`) and the agent's computation time. Add `--quiet` for the summary only. Use `--strategy EXP` for Expectimax (configured by `--depth` and `--beta`), and `--strategy BFS-RO` or `EXP-RO` for Monte Carlo Rollout over either policy (configured by `--rollouts` and `--horizon`).

### Run the experiment

```bash
uv run frozen-lake validate --config configs/experiment.toml             # check the configuration
uv run frozen-lake benchmark --config configs/experiment.toml --dry-run  # describe the run
uv run frozen-lake benchmark --config configs/experiment.toml            # run it
```

`configs/experiment.toml` describes the whole experiment: lake sizes, maps, episodes, Rollout sample counts and the Expectimax settings. `benchmark` generates every map, builds and times each base policy, plays every strategy on the same maps and episode seeds, and writes raw CSV results, a summary and the run's metadata to a new folder under `results/generated/`, then checks them for completeness.

The study's run is kept in [`results/canonical/experiment-1/`](results/canonical/experiment-1/): its configuration, maps, raw CSV results, metadata, and the tables and figures in `analysis/`. To check it and regenerate the analysis from the saved data:

```bash
uv run frozen-lake validate --run results/canonical/experiment-1
uv run frozen-lake plot --run results/canonical/experiment-1
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
├── configs/              # The experiment configuration (experiment.toml)
├── docs/                 # Supplementary documentation
├── src/frozen_lake/      # Python package (import name: frozen_lake)
│   ├── cli.py            # Typer entry point for the `frozen-lake` command
│   ├── models.py         # Shared types: actions, tiles, transitions, results
│   ├── environment.py    # Maps, the real environment, transition model, sampler
│   ├── episodes.py       # Real episodes: one environment step per decision
│   ├── policies/         # Base policies: bfs.py, expectimax.py, shared utils.py
│   ├── rollout.py        # Monte Carlo Rollout over any base policy, and its exact reference
│   ├── experiments/      # The experiment runner: config, seeds, matrix, runner, records, validation
│   ├── analysis/         # Analysis of a finished run: bootstrap, tables, figures (the `plot` command)
│   └── tui/              # Textual explorer (app.py, app.tcss, colors.py)
├── tests/                # pytest suite
├── pyproject.toml        # Project metadata, dependencies, Ruff/pytest config
├── uv.lock               # Locked dependency versions
└── justfile              # Optional task aliases
```

## References

- Gymnasium FrozenLake: <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>
- Textual: <https://textual.textualize.io/>
- AIMA Python (study reference): <https://github.com/aimacode/aima-python>
