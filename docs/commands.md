# Commands

Everything you can run in this project: what each command does, when to use it, and every option it accepts.

---

## Contents

1. [Quick reference](#1-quick-reference)
2. [How to run commands](#2-how-to-run-commands)
3. [Setup](#3-setup)
4. [The `frozen-lake` CLI](#4-the-frozen-lake-cli)
5. [Development: tests, lint, CI](#5-development-tests-lint-ci)
6. [`just` recipes](#6-just-recipes)
7. [Common workflows](#7-common-workflows)
8. [Using the package from Python](#8-using-the-package-from-python)

---

## 1. Quick reference

| I want to… | Command | `just` shortcut |
|---|---|---|
| Install the dependencies | `uv sync --locked` | `just setup` |
| Explore a lake interactively | `uv run frozen-lake tui --size 6` | `just tui` |
| Watch BFS play one episode | `uv run frozen-lake episode --size 6 --episode-seed 7` | `just episode` |
| Watch Expectimax play one episode | `uv run frozen-lake episode --strategy EXP --size 6 --episode-seed 7` | `just episode 6 7 --strategy EXP` |
| Watch Rollout over BFS play one episode | `uv run frozen-lake episode --strategy BFS-RO --size 6 --episode-seed 7` | `just episode 6 7 --strategy BFS-RO` |
| See only an episode's outcome | `uv run frozen-lake episode --quiet` | `just episode 4 42 --quiet` |
| List every command and option | `uv run frozen-lake --help` | — |
| Run the tests | `uv run pytest -q` | `just test` |
| Run one test file | `uv run pytest -q tests/test_bfs.py` | `just test tests/test_bfs.py` |
| Check lint and formatting | `uv run ruff check . && uv run ruff format --check .` | `just lint` |
| Fix lint and formatting | `uv run ruff check --fix . && uv run ruff format .` | `just format` |
| Check what CI will check | see [§5.4](#54-reproduce-ci-locally) | `just ci` |
| Remove caches | — | `just clean` |
| Check the experiment configuration | `uv run frozen-lake validate --config configs/experiment.toml` | `just validate` |
| Run the whole experiment | `uv run frozen-lake benchmark --config configs/experiment.toml` | `just benchmark` |
| Check a finished run | `uv run frozen-lake validate --run results/<folder>` | — |
| See the installed version | `uv run frozen-lake --version` | — |

---

## 2. How to run commands

Run every command **from the project root** (the folder that contains `pyproject.toml`).

### With `uv run` (recommended)

```bash
uv run frozen-lake --help
```

`uv run` executes the command inside the project's environment, where the package and its dependencies are installed. A plain `python` or `frozen-lake` outside that environment may fail with `ModuleNotFoundError` or `command not found`.

The `frozen-lake` command exists because `pyproject.toml` declares it:

```toml
[project.scripts]
frozen-lake = "frozen_lake.cli:app"
```

### With the environment activated

To skip typing `uv run`, activate the environment once per terminal:

```bash
source .venv/bin/activate
frozen-lake --help
deactivate            # leave the environment
```

In the Dev Container the environment is already active; the prompt starts with `(frozen-lake)`.

### With `just`

[`just`](https://just.systems/) provides short aliases for the commands on this page (see [§6](#6-just-recipes)). It is optional: every recipe is a plain `uv run …` command. If `just` is not installed, run any recipe through uv without installing it:

```bash
uvx --from rust-just just <recipe>
```

---

## 3. Setup

### Local machine

Requires [uv](https://docs.astral.sh/uv/), which downloads Python 3.12 automatically if needed.

```bash
git clone https://github.com/Mariojamjam/frozen-lake.git
cd frozen-lake
uv sync --locked          # or: just setup
```

`--locked` installs exactly the versions recorded in `uv.lock` and fails if `pyproject.toml` and the lockfile disagree, so everyone runs the same dependencies.

### Dev Container

Open the folder in VS Code and run **Dev Containers: Reopen in Container**. The container provides Python 3.12, a pinned `uv` and `just`, and runs `uv sync --locked` on creation.

---

## 4. The `frozen-lake` CLI

```text
frozen-lake [--version] [--help] COMMAND [OPTIONS]
```

| Command | Purpose |
|---|---|
| [`tui`](#43-tui-interactive-explorer) | Explore a lake step by step in the terminal |
| [`episode`](#44-episode-one-headless-episode) | Play one episode without an interface and print it |
| [`validate`](#45-validate-check-a-configuration-or-a-run) | Check an experiment configuration, or the results of a run |
| [`benchmark`](#46-benchmark-run-the-experiment) | Run the whole experiment headless and write its raw results |

Every command accepts `--help`, which lists its options and their defaults.

### 4.1 Global options

| Option | Effect |
|---|---|
| `--version` | Print the installed version and exit |
| `--help` | List the commands and exit |

Running `frozen-lake` with no command shows the help.

### 4.2 Options shared by `tui` and `episode`

Both commands generate a lake and play episodes on it, so they share these options.

**Map options** decide *what the lake looks like*. The same three values always produce the same lake.

| Option | Default | Meaning |
|---|---|---|
| `--size N` | `4` | Side length of the square map (`N ≥ 2`) |
| `--map-seed S` | `42` | Seed of the random layout: which tiles are holes |
| `--frozen-probability P` | `0.9` | Probability that each tile is ice rather than a hole, in `(0, 1]`. `0.9` gives about 10% holes |

**Episode options** decide *how an episode plays out on that lake*. They never change the map.

| Option | Default | Meaning |
|---|---|---|
| `--episode-seed S` | `42` | Seed of the real slips: when the agent slips, and to which side |
| `--max-steps K` | `4 × size²` | Real steps allowed before the episode ends in a timeout (`K ≥ 1`) |
| `--success-rate P` | `0.8` | Probability that a move goes in the intended direction, in `(0, 1]`. The rest splits evenly between the two perpendicular directions. `1.0` turns slipping off |

**Expectimax options** configure the Expectimax policy. BFS ignores them.

| Option | Default | Meaning |
|---|---|---|
| `--depth D` | `6` | How many agent decisions Expectimax looks ahead (`D ≥ 1`) |
| `--beta B` | `0.9` | Leaf heuristic `B ** distance`, scoring tiles where the look-ahead ends by their BFS distance to the goal, in `(0, 1)`. A smaller value penalizes distance more |

**Rollout options** configure the two Rollout strategies (`BFS-RO`, `EXP-RO`). The other strategies ignore them.

| Option | Default | Meaning |
|---|---|---|
| `--rollouts N` | `25` | Simulated trajectories per candidate action (`N ≥ 1`) |
| `--horizon H` | `2 × size²` | Simulated steps per trajectory (`H ≥ 1`), never more than the real steps the episode has left |

> **Two seeds, two roles.** `--map-seed` fixes the lake; `--episode-seed` fixes the luck on it. Keeping the lake and changing only the episode seed is how the same strategy is evaluated over many episodes on one map.

Generated maps always place the start (`S`) top-left and the goal (`G`) bottom-right, and contain at least one hole-free path between them. That path ignores slipping, so it says nothing about how likely an agent is to arrive.

### 4.3 `tui`: interactive explorer

```bash
uv run frozen-lake tui [map options] [episode options] [Expectimax options] [Rollout options]
```

**What it does.** Opens a full-screen terminal interface showing the lake, the agent, a status panel, a legend, an event log and a table of Rollout estimates. You take one real step at a time, by hand or with any of the four strategies, and watch where the ice actually sends the agent.

**When to use it.**
- To build intuition about slipping: intended move versus actual landing.
- To watch the BFS policy play, and see where ignoring risk leads it into holes.
- To watch Expectimax play, and compare where it avoids the risks BFS takes.
- To watch Rollout decide: the estimate of every action, and when it overrides its base policy.
- To inspect a policy over a whole lake at once (policy arrows).
- To demonstrate the project.

It is not a measurement tool: its timings include drawing the screen. Use `episode` to see an episode's computation time.

#### Options

All options from [§4.2](#42-options-shared-by-tui-and-episode). `tui` has no options of its own. In the explorer, each Rollout episode draws its simulations from a stream derived from the episode seed, so replaying an episode replays the same choices.

#### Keys

| Key | Action |
|---|---|
| `←` `↓` `→` `↑` | Manual mode: choose the intended action (does not move) |
| `Space` | Take one real step with the current decision maker |
| `m` | Manual mode: you choose with the arrow keys |
| `1` | BFS mode: the BFS policy chooses |
| `2` | Expectimax mode: the Expectimax policy chooses |
| `3` | BFS-RO mode: Rollout over the BFS policy chooses |
| `4` | EXP-RO mode: Rollout over the Expectimax policy chooses |
| `a` | Autoplay: take a step every quarter of a second until the episode ends; press again to pause. Needs a strategy (`1`–`4`) |
| `p` | Cycle the policy arrows on every tile: off (at start), BFS, Expectimax |
| `r` | Replay: restart with the same episode seed (same actions give the same slips) |
| `n` | New slips: restart with the next episode seed, on the same lake |
| `g` | New map: generate the next map seed, rebuild both policies, restart |
| `?` | Show help |
| `q` | Quit |

Letter keys work in upper or lower case. The footer lists the most used keys; the mode keys are listed in the legend panel. Switching mode (`m`, `1`–`4`) restarts the episode, because an episode mixing two decision makers would not be an episode of either. Restarting stops autoplay. `g` keeps the current mode and episode seed.

#### Reading the screen

- **Lake.** Lavender checkered tiles are ice, dark tiles with `○` are holes, the teal tile with `★` is the goal, the violet tile with `S` is the start. The agent is `●`; `∘` marks the tile it just left. Where the episode ends, the agent's tile takes that ending's color: rose for a hole, teal for the goal. With `p`, ice tiles show a policy as arrows. The line under the lake shows the Expectimax depth and beta and which policy the arrows show.
- **Status.** From top to bottom:
  - the outcome (`running`, `GOAL reached`, `fell into a HOLE`, `TIMEOUT`);
  - who decides: the chosen action in manual mode, or `N` and the horizon in Rollout modes;
  - what each base policy recommends from the current tile (`Recommends`), in every mode;
  - how long each policy took to build;
  - both seeds, the step count and limit, and the current state;
  - the last decision's computation time and, in Rollout modes, how many transitions it simulated;
  - the last step: intended tile versus reached tile, highlighted when the agent slipped, and `(overrode …)` when Rollout chose a different action from its base policy.
- **Event log.** One line per step and per restart.
- **Rollout estimates.** After each step in a Rollout mode: every action's estimate (the fraction of its simulated trajectories that reached the goal), a bar for its size, and which action was the base policy's (`base`) and which was taken (`chosen`). These are sampled estimates, not exact probabilities. Outside Rollout modes the table says it does not apply.

The lake is drawn with the largest tiles that fit the window and is redrawn whenever the window is resized, so it stays whole and aligned at any size. Tiles range from 11×5 characters down to a single character.

#### Examples

```bash
uv run frozen-lake tui                                  # 4×4, seeds 42
uv run frozen-lake tui --size 6                         # bigger tiles, more room to slip
uv run frozen-lake tui --size 10 --map-seed 7           # a different 10×10 lake
uv run frozen-lake tui --size 4 --max-steps 5           # reach a TIMEOUT quickly
uv run frozen-lake tui --success-rate 1.0               # no slipping: moves always go as intended
uv run frozen-lake tui --frozen-probability 0.75        # denser holes
uv run frozen-lake tui --size 6 --rollouts 100          # steadier Rollout estimates, press 3 or 4
```

### 4.4 `episode`: one headless episode

```bash
uv run frozen-lake episode [--strategy ID] [map options] [episode options]
                           [Expectimax options] [Rollout options] [--quiet]
```

**What it does.** Builds the strategy's policy, plays one whole real episode without any interface, and prints every step followed by a summary.

**When to use it.**
- To see quickly what a strategy does on a given lake and seed.
- To reproduce a specific episode exactly (same options, same output).
- To check a change from the command line, faster than opening the TUI.

#### Options

Map and episode options from [§4.2](#42-options-shared-by-tui-and-episode), plus:

| Option | Default | Meaning |
|---|---|---|
| `--strategy ID` | `BFS` | Which strategy decides (see below) |
| `--depth D`, `--beta B` | `6`, `0.9` | Expectimax options, see [§4.2](#42-options-shared-by-tui-and-episode); used by `EXP` and `EXP-RO` |
| `--rollouts N`, `--horizon H` | `25`, `2 × size²` | Rollout options, see [§4.2](#42-options-shared-by-tui-and-episode) |
| `--rollout-seed S` | derived from `--episode-seed` | Rollout strategies: seed of the simulations, kept separate from the real slips |
| `--quiet` | off | Print only the header and the summary, not every step |

| Strategy | How each real action is chosen |
|---|---|
| `BFS` | Look up the BFS policy |
| `EXP` | Look up the Expectimax policy |
| `BFS-RO` | Monte Carlo Rollout over the BFS policy: for each of the 4 actions, simulate `N` trajectories that take the action and then follow BFS; take the action with the best average |
| `EXP-RO` | The same Rollout over the Expectimax policy |

The two Rollout strategies run the same code; only the policy followed inside the simulations differs. When estimates tie, Rollout keeps the base policy's action.

#### Reading the output

```text
map 4×4 (seed 42) · strategy BFS · episode seed 42 · step limit 64
BFS policy built in 0.313 ms (14 states visited, 0 fallbacks)
step   1: state   0 → RIGHT → reached   1
step   2: state   1 → RIGHT → reached   2
step   3: state   2 → DOWN  → reached   6
step   4: state   6 → DOWN  → reached   5  (slipped)
result: hole in 4 steps · reward 0 · agent compute 0.009 ms
```

| Line | Meaning |
|---|---|
| `map …` | The lake, the strategy, the episode seed and the step limit |
| `… policy built in …` | Which base policy was built, the time it took and how much work it took. For BFS: states the search visited. For Expectimax: depth and beta, subproblems computed and cache hits. Both report tiles with no route to the goal (fallbacks) |
| `step k: state s → ACTION → reached t` | Step number, tile before the step, intended action, tile after the step. `(slipped)` means the agent did not land where the action aimed |
| `result: …` | How the episode ended (`goal`, `hole`, `timeout`), how many real steps it took, the reward (1 only for the goal), and the total time spent deciding |

States are numbered row by row from the top-left: on a 4×4 lake, the first row is 0–3 and the goal is 15.

**Reproducibility.** The same options always produce the same steps and the same result. The two times (`policy built in`, `agent compute`) change slightly between runs, because they depend on what else the computer is doing.

#### Examples

```bash
uv run frozen-lake episode                                   # 4×4, seeds 42, every step
uv run frozen-lake episode --size 8 --episode-seed 3         # an 8×8 episode
uv run frozen-lake episode --episode-seed 1 --quiet          # summary only
uv run frozen-lake episode --success-rate 1.0                # no slipping: BFS follows its shortest path
for s in 1 2 3 4 5; do uv run frozen-lake episode --episode-seed $s --quiet | tail -1; done
```

The last line plays five episodes on the same lake with different luck and prints each outcome.

With Expectimax:

```bash
uv run frozen-lake episode --strategy EXP                    # depth 6, beta 0.9
uv run frozen-lake episode --strategy EXP --depth 3 --beta 0.95
```

```text
map 4×4 (seed 42) · strategy EXP · episode seed 42 · step limit 64
EXP policy built in 1.506 ms (depth 6, beta 0.9: 78 subproblems, 732 cache hits, 0 fallbacks)
```

With Rollout:

```bash
uv run frozen-lake episode --strategy BFS-RO                 # 25 trajectories per action
uv run frozen-lake episode --strategy EXP-RO --rollouts 50 --horizon 32
```

```bash
uv run frozen-lake episode --strategy BFS-RO --rollouts 10 --horizon 32
```

```text
map 4×4 (seed 42) · strategy BFS-RO · episode seed 42 · step limit 64
BFS policy built in 0.163 ms (14 states visited, 0 fallbacks)
rollout: 10 trajectories per action, horizon 32 (capped by the real steps left)
step   1: state   0 → RIGHT → reached   1
           estimates L 0.50 D 0.30 R 0.70 U 0.70
...
result: hole in 17 steps · reward 0 · agent compute 1.524 ms · 3622 simulated transitions
```

Under each step, `estimates` lists the average return of each candidate action (`L`eft, `D`own, `R`ight, `U`p) over its simulated trajectories: the fraction that reached the goal within the horizon. These are sampled estimates, not exact probabilities. `overrode BFS RIGHT` appears when Rollout chose a different action from the base policy. The summary adds how many transitions were simulated in the whole episode; decision time grows with `--rollouts` and `--horizon`.

### 4.5 `validate`: check a configuration or a run

```bash
uv run frozen-lake validate --config configs/experiment.toml    # a configuration
uv run frozen-lake validate --run results/<folder>               # the results of a run
```

Give exactly one of the two options.

**`--config FILE`** reads the configuration and reports every problem at once: missing or unknown sections and keys (a misspelled key is an error, never silently ignored), wrong types and out-of-range values. When the configuration is valid, it describes the experiment:

```text
5 sizes [4, 6, 8, 10, 12] × 5 maps × 30 episodes × 18 arms
arms: BFS, EXP, BFS-RO N=10, EXP-RO N=10, …, BFS-RO N=800, EXP-RO N=800, BFS-RO exact, EXP-RO exact
13500 episodes · 1550 policy builds
configuration is valid
```

**`--run FOLDER`** checks a finished run: every file is present; every map, episode and arm appears exactly once; every arm of an episode shares the same episode seed; each episode's decisions add up to its steps; build rows are complete; and every saved map matches its seed. `benchmark` runs the same check at the end of every run.

### 4.6 `benchmark`: run the experiment

```bash
uv run frozen-lake benchmark --config configs/experiment.toml [--output-dir DIR] [--dry-run]
```

**What it does.**
1. Generates and saves every map, before any strategy plays.
2. Builds each base policy (BFS and Expectimax) from scratch on each map: one untimed warm-up, then the configured number of timed builds, each recorded.
3. Plays every **arm** on every map and episode, all with the same episode seeds: `BFS` and `EXP` once each, `BFS-RO` and `EXP-RO` once per sample count `N`, and the two exact references.
4. Writes the raw results, a summary and the run's metadata, then validates them.

It prints the experiment it is about to run, then one progress line per finished map (on standard error, between episodes, so it never enters a measured time), then the results folder.

**When to use it.** To produce the data the analysis uses. It is headless: nothing is drawn and Textual is never loaded.

| Option | Meaning |
|---|---|
| `--config FILE` | The experiment configuration (required) |
| `--output-dir DIR` | Folder for the results, overriding `experiment.output_dir` |
| `--dry-run` | Validate and describe the experiment, then stop without running |

Results go to a new folder named after the start time and the first 8 characters of the configuration hash, for example `results/20261018-125816-7307f7ed/`. If that folder already exists, `benchmark` stops instead of overwriting it.

#### The configuration file

`configs/experiment.toml` describes the whole experiment. Every key is required.

| Key | Meaning |
|---|---|
| `environment.sizes` | Lake sizes `n` (each `≥ 2`, no repeats) |
| `environment.frozen_probability` | Chance that a generated tile is ice, in `(0, 1]` |
| `environment.success_rate` | Chance that a move goes as intended, in `(0, 1]` |
| `environment.max_steps_multiplier` | Real-step limit per episode = multiplier × n² |
| `policies.expectimax_depth` | Expectimax look-ahead depth (`≥ 1`) |
| `policies.heuristic_beta` | Expectimax leaf heuristic, in `(0, 1)` |
| `policies.discount_factor` | Discount γ, in `(0, 1]` |
| `policies.build_repeats` | Timed builds of each base policy per map (`≥ 1`) |
| `policies.build_warmup` | Untimed builds before them (`≥ 0`) |
| `rollout.samples_per_action` | The sample counts `N` to run (each `≥ 1`, no repeats) |
| `rollout.horizon_multiplier` | Rollout horizon = multiplier × n², capped by the real steps left |
| `rollout.exact_reference` | Also play Rollout with exact values (`true` or `false`) |
| `experiment.maps_per_size` | Maps generated for each size |
| `experiment.episodes_per_map` | Episodes each arm plays on each map |
| `experiment.master_seed` | Every map, episode and simulation seed derives from it |
| `experiment.output_dir` | Where result folders are created |
| `experiment.write_decisions` | Write `decision_metrics.csv` (`true` or `false`) |

#### The results folder

| File | One row per | Holds |
|---|---|---|
| `configuration.json` | — | The configuration that was run |
| `maps.json` | map | Its id (`size-index`), seed and exact rows of tiles |
| `policy_metrics.csv` | timed policy build | Build time and the search's counters |
| `episode_metrics.csv` | real episode | Strategy, estimator, `N`, seeds, ending, steps, agent time, wall time, simulated transitions |
| `decision_metrics.csv` | real decision | State, base and chosen action, whether Rollout overrode the base policy, time, and each action's value |
| `summary.csv` | size × strategy × estimator × `N` | Success rate, median decision time, mean agent time per episode, median build time, amortized time per episode |
| `metadata.json` | — | Start and end times, command, configuration and maps hashes, seed scheme, git commit, Python, package versions, machine and units |

Conventions used in every file:
- times are in nanoseconds (`_ns`) or milliseconds (`_ms`);
- `estimator` is `sampled` or `exact` for Rollout strategies, and blank for `BFS` and `EXP`;
- `rollout_n` is blank where it does not apply: for `BFS`, `EXP` and the exact reference, which samples nothing;
- a blank value always means "does not apply", never zero;
- in `summary.csv`, the exact reference's cost columns are blank, because it simulates nothing and its time is not a cost to compare;
- `amortized_ms_per_episode` is the median build time divided by the episodes per map, plus the mean agent time per episode.

#### Examples

```bash
uv run frozen-lake benchmark --config configs/experiment.toml --dry-run    # describe only
uv run frozen-lake benchmark --config configs/experiment.toml              # the full experiment
uv run frozen-lake benchmark --config tests/data/tiny.toml --output-dir /tmp/runs   # a few seconds
```

### 4.7 Exit codes and errors

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | `validate --run` found problems in the results |
| `2` | Invalid option value, unknown strategy, invalid configuration, or an existing results folder |

Invalid values are rejected before anything runs, with a message naming the problem:

```text
Invalid value: frozen_probability must be in (0, 1], got 1.5
Invalid value for '--max-steps': 0 is not in the range x>=1.
```

---

## 5. Development: tests, lint, CI

### 5.1 Tests

```bash
uv run pytest -q                                   # the whole suite
uv run pytest -q tests/test_bfs.py                 # one file
uv run pytest -q -k fallback                       # tests whose name contains "fallback"
uv run pytest -q tests/test_bfs.py::test_ties_follow_the_action_priority   # one test
uv run pytest -x                                   # stop at the first failure
uv run pytest -v                                   # one line per test
```

| File | Covers |
|---|---|
| `tests/test_models.py` | Shared types: actions, tiles, transitions, strategy IDs, results |
| `tests/test_environment.py` | Maps, the real environment, the transition model and its parity with Gymnasium, the sampler |
| `tests/test_bfs.py` | BFS distances, policy extraction, ties, fallbacks, counters |
| `tests/test_expectimax.py` | Expectimax heuristic, MAX/CHANCE values, memoization, ties, fallbacks, counters |
| `tests/test_rollout.py` | Simulated trajectories, Rollout decisions, ties, horizon limits, reproducibility, isolation from the real environment, the exact reference |
| `tests/test_experiments.py` | Configuration validation, seeds, maps, arms, whole runs, pairing, reproducibility, result validation |
| `tests/test_episodes.py` | The episode engine and whole episodes played by a policy |
| `tests/test_cli.py` | Command-line options, output and errors |
| `tests/test_tui.py` | The terminal interface, driven by key presses without a screen |

The interface tests use Textual's headless `run_test()`, so they run anywhere, including CI, without opening a window.

### 5.2 Lint and formatting

```bash
uv run ruff check .              # report lint problems
uv run ruff check --fix .        # fix what can be fixed automatically
uv run ruff format --check .     # report files that are not formatted
uv run ruff format .             # format the files
```

Ruff's settings live in `pyproject.toml` (line length 100, Python 3.12 target).

### 5.3 What CI checks

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

1. `uv sync --locked`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run pytest -q`
5. `uv run frozen-lake --help` (smoke test)

### 5.4 Reproduce CI locally

```bash
just ci
```

`just ci` runs the same five steps. If it passes locally, CI should pass too.

---

## 6. `just` recipes

Run `just` with no arguments to list them. Parameters are positional and have defaults; anything after them is passed through to the underlying command.

| Recipe | Runs | Notes |
|---|---|---|
| `just setup` | `uv sync --locked` | Install locked dependencies |
| `just clean` | Removes `.pytest_cache`, `.ruff_cache`, `__pycache__` | Leaves `.venv` untouched |
| `just tui [size] [episode_seed] [map_seed] [args…]` | `frozen-lake tui --size {size} --episode-seed {episode_seed} --map-seed {map_seed} {args}` | Defaults: size 6, episode seed 42, map seed 42 |
| `just episode [size] [episode_seed] [args…]` | `frozen-lake episode --size {size} --episode-seed {episode_seed} {args}` | Defaults: size 6, episode seed 7 |
| `just validate` | `frozen-lake validate --config configs/experiment.toml` | Check the experiment configuration |
| `just benchmark [config] [output_dir] [args…]` | `frozen-lake benchmark --config {config} --output-dir {output_dir} {args}` | Defaults: `configs/experiment.toml`, and the config's own `output_dir` when `output_dir` is empty or omitted |
| `just test [args…]` | `pytest -q {args}` | e.g. `just test -k fallback -x` |
| `just lint` | `ruff check .` and `ruff format --check .` | Changes nothing |
| `just format` | `ruff check --fix .` and `ruff format .` | Changes files |
| `just ci` | The five CI steps | Run before pushing |

To see the command a recipe would run without running it: `just --dry-run <recipe> [args…]`.

Examples:

```bash
just tui 10                    # 10×10 lake, episode seed 42
just tui 8 12 7                # 8×8, episode seed 12, map seed 7
just tui 4 1 --max-steps 5     # 4×4, episode seed 1, quick timeout
just episode 8 3 --quiet       # 8×8, episode seed 3, summary only
just benchmark                 # the experiment, from configs/experiment.toml
just benchmark my.toml runs    # another config, results under runs/
just benchmark my.toml "" --dry-run   # another config, its own output_dir, only print the workload
```

---

## 7. Common workflows

**First time on a new machine**
```bash
uv sync --locked
uv run frozen-lake --help
uv run pytest -q
```

**Before pushing**
```bash
just format        # fix what can be fixed
just ci            # check exactly what CI checks
```

**Investigate a failing test**
```bash
uv run pytest -x -v tests/test_episodes.py      # stop at the first failure, show test names
```

---

## 8. Using the package from Python

Open an interactive Python session in the project environment:

```bash
uv run python
```

Then, for example:

```python
from frozen_lake.environment import generate_map
from frozen_lake.episodes import EpisodeSession, policy_decider, run_episode
from frozen_lake.models import StrategyId
from frozen_lake.policies.bfs import bfs_distances, build_bfs_policy

lake = generate_map(6, 0.9, seed=42)          # a 6×6 lake
print("\n".join(lake))

build = build_bfs_policy(lake)                # the full BFS policy, timed
build.policy[0], build.build_time_ns, build.diagnostics
bfs_distances(lake)[0]                        # moves from the start to the goal, ignoring slips

session = EpisodeSession(lake, max_steps=144, success_rate=0.8)
result = run_episode(session, policy_decider(build.policy), seed=7, strategy=StrategyId.BFS)
result.termination_reason, result.steps, result.success
```

Leave the session with `exit()` or `Ctrl+D`.
