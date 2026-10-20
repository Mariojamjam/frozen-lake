# Commands

Everything you can run in this project: how to install the `frozen-lake` command, what each command does, when to use it, and every option it accepts. Every command also explains itself: `frozen-lake <command> -h` shows its options, grouped by topic, and examples to copy.

---

## Contents

1. [Quick reference](#1-quick-reference)
2. [Install the `frozen-lake` command](#2-install-the-frozen-lake-command)
3. [The `frozen-lake` CLI](#3-the-frozen-lake-cli)
4. [Development: tests, lint, CI](#4-development-tests-lint-ci)
5. [Common workflows](#5-common-workflows)
6. [Using the package from Python](#6-using-the-package-from-python)

---

## 1. Quick reference

| I want to… | Command |
|---|---|
| See every command and the typical workflow | `frozen-lake` |
| See a command's options and examples | `frozen-lake episode -h` |
| Explore a 6×6 lake interactively | `frozen-lake tui -s 6` |
| Choose an episode's options in a form, then play it | `frozen-lake episode` |
| Watch Expectimax play one episode | `frozen-lake episode -S EXP -s 6 -e 7` |
| Watch Rollout over BFS with N = 100 | `frozen-lake episode -S BFS-RO -s 6 -n 100` |
| See only an episode's outcome | `frozen-lake episode -s 6 -q` |
| Check the experiment configuration | `frozen-lake validate` |
| Describe the experiment without running it | `frozen-lake benchmark --dry-run` |
| Run the whole experiment | `frozen-lake benchmark` |
| Check a finished run | `frozen-lake validate -i results/canonical/experiment-1` |
| Analyze a run: tables and figures | `frozen-lake plot -i results/canonical/experiment-1` |
| Choose a run to analyze from a list | `frozen-lake plot` |
| See the installed version | `frozen-lake --version` |
| Run the tests | `uv run pytest -q` |
| Check lint and formatting | `uv run ruff check . && uv run ruff format --check .` |

---

## 2. Install the `frozen-lake` command

The commands work the same on Linux, macOS and Windows. They need Python 3.12 or newer, which [uv](https://docs.astral.sh/uv/) downloads automatically if needed. Run them **from the project folder** (the one that contains `pyproject.toml`): the experiment's configuration and results are found relative to it.

### Once: put `frozen-lake` on the PATH (recommended)

```bash
uv tool install --editable .
```

uv installs the command in its tools folder (`~/.local/bin` on Linux and macOS, `%USERPROFILE%\.local\bin` on Windows, where it is `frozen-lake.exe`). If uv warns that this folder is not on the PATH, run `uv tool update-shell` once and open a new terminal. From then on, in any terminal:

```text
frozen-lake tui -s 6
```

`--editable` links the command to this folder, so changes to the code take effect without reinstalling. To remove it: `uv tool uninstall frozen-lake`.

`uv tool install` chooses compatible versions of the dependencies on its own and does not read `uv.lock`. That is enough to explore lakes and play episodes. To reproduce the experiment with exactly the locked versions, use `uv run` below.

### Without installing: `uv run`

```bash
uv sync --locked                 # once: the exact versions recorded in uv.lock
uv run frozen-lake tui -s 6      # any command, inside the project's environment
```

`--locked` installs exactly the versions in `uv.lock` and fails if `pyproject.toml` and the lockfile disagree, so everyone runs the same dependencies. Every command on this page also works as `uv run frozen-lake …`.

### With the project's environment activated

After `uv sync --locked`, activate the environment once per terminal:

```bash
source .venv/bin/activate        # Linux and macOS
.venv\Scripts\activate           # Windows (PowerShell or cmd)
frozen-lake tui -s 6
deactivate                       # leave the environment
```

If PowerShell refuses to run the activation script, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### If `frozen-lake` is not found

Inside the project's environment, `python -m frozen_lake` is the same command:

```bash
uv run python -m frozen_lake episode -s 6
```

### The terminal

The explorer (`tui`) and the episode form need a terminal with colors and Unicode, which every current terminal has: the default terminals of Linux and macOS, and Windows Terminal (the default on Windows 11). The legacy Windows console may draw the lake's symbols and colors incorrectly.

### Dev Container

Open the folder in VS Code and run **Dev Containers: Reopen in Container**. The container provides Python 3.12 and a pinned `uv`, runs `uv sync --locked` on creation, and puts the project's environment on the PATH, so `frozen-lake` works in its terminal right away.

---

## 3. The `frozen-lake` CLI

```text
frozen-lake [--version] [-h] COMMAND [OPTIONS]
```

| Command | Purpose |
|---|---|
| [`tui`](#33-tui-interactive-explorer) | Explore a lake step by step in the terminal |
| [`episode`](#34-episode-one-episode-step-by-step) | Play one episode without the explorer and print it |
| [`validate`](#35-validate-check-a-configuration-or-a-run) | Check an experiment configuration, or the results of a run |
| [`benchmark`](#36-benchmark-run-the-experiment) | Run the whole experiment headless and write its raw results |
| [`plot`](#37-plot-analyze-a-finished-run) | Turn a finished run into intervals, tables and figures |

### 3.1 Getting help

| Command | Shows |
|---|---|
| `frozen-lake` | Every command and the typical workflow, from exploring a lake to analyzing a run |
| `frozen-lake <command> -h` | What the command does, its options grouped by topic (lake, strategy, Expectimax, Rollout, input, output), the default of each, and examples to copy |
| `frozen-lake --version` | The installed version |

`-h` and `--help` are the same. Defaults that depend on other options are shown as formulas, for example `4 × size²`.

### 3.2 Options shared by `tui` and `episode`

Both commands generate a lake and play episodes on it, so they share these options. Most have a one-letter form.

**Lake options** decide *what the lake looks like* and *how an episode plays out on it*. The size, map seed and frozen-tile probability always produce the same lake.

| Option | Default | Meaning |
|---|---|---|
| `-s`, `--size N` | `4` (`tui`: required) | Side of the square lake, in tiles (`N ≥ 2`) |
| `-m`, `--map-seed S` | `42` | Seed of the random layout: which tiles are holes |
| `-e`, `--episode-seed S` | `42` | Seed of the real slips: when the agent slips, and to which side |
| `--frozen-probability P` | `0.9` | Chance that each tile is ice rather than a hole, in `(0, 1]`. `0.9` gives about 10% holes |
| `--success-rate P` | `0.8` | Chance that a move goes in the intended direction, in `(0, 1]`. The rest splits evenly between the two sideways slips. `1` turns slipping off |
| `--max-steps K` | `4 × size²` | Real steps allowed before the episode ends in a timeout (`K ≥ 1`) |

**Expectimax options** configure the Expectimax policy, used by `EXP` and `EXP-RO`. BFS ignores them.

| Option | Default | Meaning |
|---|---|---|
| `-d`, `--depth D` | `6` | How many agent decisions Expectimax looks ahead (`D ≥ 1`) |
| `-b`, `--beta B` | `0.9` | Leaf heuristic `B ** distance`, scoring tiles where the look-ahead ends by their BFS distance to the goal, in `(0, 1)`. Closer to 1, distance matters less |

**Rollout options** configure the two Rollout strategies (`BFS-RO`, `EXP-RO`). The other strategies ignore them.

| Option | Default | Meaning |
|---|---|---|
| `-n`, `--rollouts N` | `25` | N: simulated trajectories per candidate action (`N ≥ 1`) |
| `--horizon H` | `2 × size²` | Longest simulated trajectory, in steps (`H ≥ 1`), never more than the real steps the episode has left |
| `--rollout-seed S` | derived from the episode seed | Seed of the simulations, kept apart from the real slips |

> **Two seeds, two roles.** `-m` fixes the lake; `-e` fixes the luck on it. Keeping the lake and changing only the episode seed is how the same strategy is evaluated over many episodes on one map.

Generated maps always place the start (`S`) top-left and the goal (`G`) bottom-right, and contain at least one hole-free path between them. That path ignores slipping, so it says nothing about how likely an agent is to arrive.

### 3.3 `tui`: interactive explorer

```bash
frozen-lake tui -s SIZE [lake options] [Expectimax options] [Rollout options]
```

**What it does.** Opens a full-screen terminal interface showing the lake, the agent, a status panel, a legend, an event log and a table of Rollout estimates. You take one real step at a time, by hand or with any of the four strategies, and watch where the ice actually sends the agent. The lake size is required: `frozen-lake tui` alone shows the help with examples.

**When to use it.**
- To build intuition about slipping: intended move versus actual landing.
- To watch the BFS policy play, and see where ignoring risk leads it into holes.
- To watch Expectimax play, and compare where it avoids the risks BFS takes.
- To watch Rollout decide: the estimate of every action, and when it overrides its base policy.
- To change N or the Expectimax depth and replay the same episode, to see what changes.
- To inspect a policy over a whole lake at once (policy arrows).
- To demonstrate the project.

It is not a measurement tool: its timings include drawing the screen. Use `episode` to see an episode's computation time.

#### Options

All options from [§3.2](#32-options-shared-by-tui-and-episode); `-s` is required. Each Rollout episode draws its simulations from a stream derived from the episode seed (or from `--rollout-seed`), so replaying an episode replays the same choices.

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
| `+` / `-` | N one step up or down the experiment's budgets: 10, 25, 50, 100, 200, 400, 800 |
| `]` / `[` | Expectimax depth one up or down, from 1 to 12 |
| `s` | Settings: type exact values for N, the Expectimax depth and beta, and the Rollout horizon |
| `p` | Cycle the policy arrows on every tile: off (at start), BFS, Expectimax |
| `r` | Replay: restart with the same episode seed (same actions give the same slips) |
| `n` | New slips: restart with the next episode seed, on the same lake |
| `g` | New map: generate the next map seed, rebuild both policies, restart |
| `?` | Show help |
| `q` | Quit |

Letter keys work in upper or lower case. The footer lists the most used keys; the mode and tuning keys are listed in the legend panel.

**Restarts.** Switching mode (`m`, `1`–`4`) restarts the episode, because an episode mixing two decision makers would not be an episode of either. Changing N, the depth, beta or the horizon also restarts it, **on the same episode seed**: the same slips meet the new decisions, so the difference between the two episodes is the setting you changed. A new depth or beta rebuilds the Expectimax policy, and its arrows (`p`) update at once. Restarting stops autoplay. `g` keeps the current mode, settings and episode seed.

**Settings.** `s` opens a small window with four fields. A value outside its range is marked with what a valid value is, and Apply stays disabled until every field is valid. A blank horizon means the default, `2 × size²`. Enter applies; Esc closes without changing anything. While the window is open, the arrow keys and Space edit the fields instead of playing.

#### Reading the screen

- **Lake.** Lavender checkered tiles are ice, dark tiles with `○` are holes, the teal tile with `★` is the goal, the violet tile with `S` is the start. The agent is `●`; `∘` marks the tile it just left. Where the episode ends, the agent's tile takes that ending's color: rose for a hole, teal for the goal. With `p`, ice tiles show a policy as arrows. The line under the lake shows the Expectimax depth and beta and which policy the arrows show.
- **Status.** From top to bottom:
  - the outcome (`running`, `GOAL reached`, `fell into a HOLE`, `TIMEOUT`);
  - who decides, and the chosen action in manual mode;
  - the Rollout settings, `N` and the horizon, in every mode;
  - what each base policy recommends from the current tile (`Recommends`);
  - how long each policy took to build;
  - both seeds, the step count and limit, and the current state;
  - the last decision's computation time and, in Rollout modes, how many transitions it simulated;
  - the last step: intended tile versus reached tile, highlighted when the agent slipped, and `(overrode …)` when Rollout chose a different action from its base policy.
- **Event log.** One line per step, per restart and per change of settings.
- **Rollout estimates.** After each step in a Rollout mode: every action's estimate (the fraction of its simulated trajectories that reached the goal), a bar for its size, and which action was the base policy's (`base`) and which was taken (`chosen`). These are sampled estimates, not exact probabilities. Outside Rollout modes the table says it does not apply.

The lake is drawn with the largest tiles that fit the window and is redrawn whenever the window is resized, so it stays whole and aligned at any size. Tiles range from 11×5 characters down to a single character. A lake too large even for one character per tile, such as 24×24 in a small window, switches to a **compact view**: two tile rows per line, each tile half a character tall and still about square, keeping its colors (agent, start, holes, goal) but not its symbols or the policy arrows. If even that does not fit, the panel says how many characters the lake needs instead of cropping it; enlarge the window, make the font smaller or choose a smaller lake.

#### Examples

```bash
frozen-lake tui -s 4                          # 4×4, seeds 42
frozen-lake tui -s 6                          # more room to slip
frozen-lake tui -s 10 -m 7                    # a different 10×10 lake
frozen-lake tui -s 8 -m 7 -e 3                # another lake and other slips
frozen-lake tui -s 4 --max-steps 5            # reach a TIMEOUT quickly
frozen-lake tui -s 6 --success-rate 1         # no slipping: moves always go as intended
frozen-lake tui -s 6 --frozen-probability 0.75    # denser holes
frozen-lake tui -s 10 -n 100 -d 4             # Rollout with N = 100, Expectimax depth 4; press 3 or 4
```

### 3.4 `episode`: one episode, step by step

```bash
frozen-lake episode [-S STRATEGY] [lake options] [Expectimax options] [Rollout options] [-q]
```

**What it does.** Builds the strategy's policy, plays one whole real episode without the explorer, and prints every step followed by a summary.

**When to use it.**
- To see quickly what a strategy does on a given lake and seed.
- To reproduce a specific episode exactly (same options, same output).
- To check a change from the command line, faster than opening the explorer.

#### Choosing the options in a form

Run with no options in a terminal, `frozen-lake episode` opens a short form instead of playing the defaults:

- the four strategies, then the lake, Expectimax and Rollout settings, each filled with its default;
- only the settings the chosen strategy uses are enabled: Expectimax for `EXP` and `EXP-RO`, Rollout for `BFS-RO` and `EXP-RO`;
- an invalid value is marked, with the reason, and **Run** stays disabled until every value is valid;
- the **equivalent command** is updated with every change, with only the options that differ from the defaults.

Enter or **Run** closes the form, prints the equivalent command and plays the episode in the terminal as usual; Esc cancels. Outside a terminal (a script, a pipe, CI), `frozen-lake episode` plays the defaults without asking.

#### Options

All options from [§3.2](#32-options-shared-by-tui-and-episode), plus:

| Option | Default | Meaning |
|---|---|---|
| `-S`, `--strategy ID` | `BFS` | Which strategy decides: `BFS`, `EXP`, `BFS-RO` or `EXP-RO` (see below) |
| `-q`, `--quiet` | off | Print only the header and the summary, not every step |

| Strategy | How each real action is chosen |
|---|---|
| `BFS` | Look up the BFS policy |
| `EXP` | Look up the Expectimax policy |
| `BFS-RO` | Monte Carlo Rollout over the BFS policy: for each of the 4 actions, simulate `N` trajectories that take the action and then follow BFS; take the action with the best average |
| `EXP-RO` | The same Rollout over the Expectimax policy |

The two Rollout strategies run the same code; only the policy followed inside the simulations differs. When estimates tie, Rollout keeps the base policy's action.

#### Reading the output

```text
map 4×4 (seed 42) · strategy BFS · episode seed 42 · step limit 64 (4 × 4²)
moves go as intended 80% of the time, 10% to each side · tiles are ice with probability 0.9
BFS policy built in 0.229 ms (14 states visited, 0 fallbacks)
step   1: state   0 → RIGHT → reached   1
step   2: state   1 → RIGHT → reached   2
step   3: state   2 → DOWN  → reached   6
step   4: state   6 → DOWN  → reached   5  (slipped)
result: hole in 4 steps · reward 0 · agent compute 0.002 ms
```

| Line | Meaning |
|---|---|
| `map …` | The lake, the strategy, the episode seed and the step limit, with how it was computed when it is the default |
| `moves go as intended …` | The slipping and the share of ice that generated the lake |
| `… policy built in …` | Which base policy was built, the time it took and how much work it took. For BFS: states the search visited. For Expectimax: depth and beta, subproblems computed and cache hits. Both report tiles with no route to the goal (fallbacks) |
| `step k: state s → ACTION → reached t` | Step number, tile before the step, intended action, tile after the step. `(slipped)` means the agent did not land where the action aimed |
| `result: …` | How the episode ended (`goal`, `hole`, `timeout`), how many real steps it took, the reward (1 only for the goal), and the total time spent deciding |

States are numbered row by row from the top-left: on a 4×4 lake, the first row is 0–3 and the goal is 15.

**Reproducibility.** The same options always produce the same steps and the same result. The two times (`policy built in`, `agent compute`) change slightly between runs, because they depend on what else the computer is doing.

#### Examples

```bash
frozen-lake episode                          # in a terminal: choose the options in a form
frozen-lake episode -s 4                     # 4×4, seeds 42, every step
frozen-lake episode -s 8 -e 3                # an 8×8 episode
frozen-lake episode -s 4 -e 1 -q             # summary only
frozen-lake episode -s 4 --success-rate 1    # no slipping: BFS follows its shortest path
```

On Linux and macOS, this plays five episodes on the same lake with different luck and prints each outcome:

```bash
for s in 1 2 3 4 5; do frozen-lake episode -s 6 -e $s -q | tail -1; done
```

With Expectimax:

```bash
frozen-lake episode -S EXP                   # depth 6, beta 0.9
frozen-lake episode -S EXP -d 3 -b 0.95
```

```text
map 4×4 (seed 42) · strategy EXP · episode seed 42 · step limit 64 (4 × 4²)
moves go as intended 80% of the time, 10% to each side · tiles are ice with probability 0.9
EXP policy built in 0.968 ms (depth 6, beta 0.9: 78 subproblems, 732 cache hits, 0 fallbacks)
result: goal in 9 steps · reward 1 · agent compute 0.002 ms
```

With Rollout:

```bash
frozen-lake episode -S BFS-RO                # 25 trajectories per action
frozen-lake episode -S EXP-RO -n 50 --horizon 32
frozen-lake episode -S BFS-RO -n 10 --horizon 32
```

```text
map 4×4 (seed 42) · strategy BFS-RO · episode seed 42 · step limit 64 (4 × 4²)
moves go as intended 80% of the time, 10% to each side · tiles are ice with probability 0.9
BFS policy built in 0.227 ms (14 states visited, 0 fallbacks)
rollout: 10 trajectories per action, horizon 32 (capped by the real steps left)
step   1: state   0 → RIGHT → reached   1
           estimates L 0.50 D 0.30 R 0.70 U 0.70
...
result: hole in 17 steps · reward 0 · agent compute 2.171 ms · 3622 simulated transitions
```

Under each step, `estimates` lists the average return of each candidate action (`L`eft, `D`own, `R`ight, `U`p) over its simulated trajectories: the fraction that reached the goal within the horizon. These are sampled estimates, not exact probabilities. `overrode BFS RIGHT` appears when Rollout chose a different action from the base policy. The summary adds how many transitions were simulated in the whole episode; decision time grows with `-n` and `--horizon`.

### 3.5 `validate`: check a configuration or a run

```bash
frozen-lake validate [-i PATH]
```

| Option | Default | Meaning |
|---|---|---|
| `-i`, `--input PATH` | `configs/experiment.toml` | A configuration file to check, or a results folder to check for completeness |

A **file** is checked as a configuration. Every problem is reported at once: missing or unknown sections and keys (a misspelled key is an error, never silently ignored), wrong types and out-of-range values. When the configuration is valid, it describes the experiment:

```text
5 sizes [4, 6, 8, 10, 12] × 5 maps × 30 episodes × 18 arms
arms: BFS, EXP, BFS-RO N=10, EXP-RO N=10, …, BFS-RO N=800, EXP-RO N=800, BFS-RO exact, EXP-RO exact
13500 episodes · 1550 policy builds
configuration is valid
```

A **folder** is checked as the results of a finished run: every file is present; every map, episode and arm appears exactly once; every arm of an episode shares the same episode seed; each episode's decisions add up to its steps; build rows are complete; and every saved map matches its seed. `benchmark` runs the same check at the end of every run. When the run is complete, the command suggests `plot` as the next step.

```bash
frozen-lake validate                                     # the study's configuration
frozen-lake validate -i my.toml                          # another configuration
frozen-lake validate -i results/canonical/experiment-1   # the results of a run
```

### 3.6 `benchmark`: run the experiment

```bash
frozen-lake benchmark [-i CONFIG] [-o FOLDER] [--dry-run] [-y]
```

**What it does.**
1. Generates and saves every map, before any strategy plays.
2. Builds each base policy (BFS and Expectimax) from scratch on each map: one untimed warm-up, then the configured number of timed builds, each recorded.
3. Plays every **arm** on every map and episode, all with the same episode seeds: `BFS` and `EXP` once each, `BFS-RO` and `EXP-RO` once per sample count `N`, and the two exact references.
4. Writes the raw results, a summary and the run's metadata, then validates them.

It prints the experiment it is about to run and asks before starting, since the study's experiment takes about half an hour on a laptop. While it runs, it prints one progress line per finished map (on standard error, between episodes, so it never enters a measured time). At the end it prints the results folder and the next steps: `validate` and `plot` on that folder.

**When to use it.** To produce the data the analysis uses. It is headless: nothing is drawn and Textual is never loaded.

| Option | Default | Meaning |
|---|---|---|
| `-i`, `--input PATH` | `configs/experiment.toml` | The experiment configuration, or a folder that contains `experiment.toml` |
| `-o`, `--output FOLDER` | the configuration's `experiment.output_dir` | Folder where the run's results folder is created |
| `--dry-run` | off | Check and describe the experiment, then stop without running |
| `-y`, `--yes` | off | Start without asking. Needed when no one can answer, as in a script |

Results go to a new folder named after the start time and the first 8 characters of the configuration hash, for example `results/generated/20261018-125816-7307f7ed/`. If that folder already exists, `benchmark` stops instead of overwriting it.

`results/` has two parts:

| Folder | Contents | In git |
|---|---|---|
| `results/generated/` | Every new run, named automatically | No (ignored) |
| `results/canonical/` | The runs the study reports, under a chosen name, for example `results/canonical/experiment-1/` | Yes |

To make a run canonical, move its folder from `generated/` to `canonical/` under its new name, then regenerate its analysis with `plot`. Renaming is safe: `validate` and `plot` do not depend on the folder name, and the run's provenance (configuration hash, commit, map hashes) stays in its `metadata.json`.

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
frozen-lake benchmark --dry-run                         # describe the study's experiment only
frozen-lake benchmark                                   # the full experiment (asks first)
frozen-lake benchmark -y                                # the same, without asking
frozen-lake benchmark -i tests/data/tiny.toml -o runs/ -y   # a few seconds, results in runs/
frozen-lake benchmark -i configs/                       # a folder: uses its experiment.toml
```

### 3.7 `plot`: analyze a finished run

```bash
frozen-lake plot [-i RUN] [-o FOLDER] [--resamples N] [--seed S]
```

**What it does.** Reads the raw results of a finished `benchmark` and turns them into estimates with uncertainty, tables and figures. It first runs the same checks as `validate` on a results folder and refuses a damaged one. It never runs an agent again and never times anything, so it is safe to repeat and to change.

**When to use it.** After `benchmark`, to get the results the research question needs: success rate against amortized compute time, with intervals.

| Option | Default | Meaning |
|---|---|---|
| `-i`, `--input RUN` | choose from a list | The results folder of a finished benchmark |
| `-o`, `--output FOLDER` | `<run>/analysis` | Where to write. Files of an earlier analysis with the same names are replaced |
| `--resamples N` | `2000` | Bootstrap resamples for every interval (`≥ 100`) |
| `--seed S` | derived from the run's master seed | Bootstrap seed, so the same run always gives the same intervals |

Without `-i`, `plot` lists the runs it finds, those in `results/canonical/` first and then those in `results/generated/`, and asks which one to analyze. Outside a terminal it prints the list and stops, so a script must name the run with `-i`.

#### How the numbers are made

- **Per map first.** Each quantity is computed on every map and then averaged over the maps of a size, so a map counts once however many episodes it played.
- **Uncertainty.** A cluster bootstrap: resample maps with replacement, then resample episodes within each chosen map, recompute, repeat, and report the 95 % percentile interval. One resample is applied to every arm at once, so the pairing between arms (same maps, same episode seeds) is kept, and differences between arms get honest intervals too.
- **Amortized time per episode.** The median build time of a map's base policy divided by `R`, plus the mean agent time of that map's episodes. `R` is the number of episodes that share a build; the analysis uses `R = E` (`experiment.episodes_per_map`) and figure H shows other values.
- **Paired difference.** `EXP-RO` minus `BFS-RO` at the same `N`, for success rate and for amortized time.
- **Pareto frontier.** An arm is on the observed frontier of a size if no other arm is at least as fast and at least as successful, and better on one of the two. The exact reference is left out: its cost is not comparable.

#### What it writes

```text
<run>/analysis/
├── analysis.json       resamples, level, seed, R and hashes of the run it analyzed
├── figures/            figure_a_….png to figure_h_….png
└── tables/             the numbers behind the figures
```

| Figure | Shows |
|---|---|
| A | Success rate against amortized time per episode, one panel per size, with intervals and the observed frontier |
| B | Success rate against `N`, with base-only and exact-reference lines |
| C | Decision time (median and interquartile range) and simulated transitions against `N` |
| D | Policy build time against map size |
| E | Search work against map size: states evaluated, successors examined, cache hits |
| F | Each algorithm's own memory measure: BFS queue peak, Expectimax recursion depth and cache entries |
| G | `EXP-RO` over `BFS-RO` decision time on states both strategies visited (needs logged decisions) |
| H | The amortized-time gap between `EXP-RO` and `BFS-RO` as `R` varies |

| Table | One row per |
|---|---|
| `arms.csv` | size × arm: success, amortized time and their intervals, frontier membership |
| `paired_differences.csv` | size × `N`: `EXP-RO` minus `BFS-RO`, with intervals |
| `amortization_sensitivity.csv` | size × `N` × `R`: the gap plotted in figure H |
| `policy_builds.csv` | size × base policy: build-time quartiles and mean search counters |
| `decision_times.csv` | size × arm: decision-time quartiles (needs logged decisions) |
| `shared_state_decision_time.csv` | size × `N`: the ratio plotted in figure G (needs logged decisions) |

Blank still means "does not apply". Two things to keep in mind when reading the figures:
- Figure G reuses the decisions already logged, so the states are those the agents happened to visit. It is not the dedicated paired-state timing on a prespecified set of states that the design document describes.
- Without logged decisions (`experiment.write_decisions = false`) figure G and `decision_times.csv` are skipped, and the top row of figure C shows mean agent time per episode instead, which also depends on episode length.

#### Examples

```bash
frozen-lake plot                                                       # choose a run from a list
frozen-lake plot -i results/canonical/experiment-1                     # the study's run
frozen-lake plot -i results/canonical/experiment-1 -o figures/ --resamples 5000
```

### 3.8 Exit codes and errors

| Code | Meaning |
|---|---|
| `0` | Success, or the episode form was cancelled |
| `1` | `validate` or `plot` found problems in the results, or `benchmark` was not confirmed |
| `2` | Invalid option value, unknown strategy, invalid or missing configuration, an existing results folder, no run to analyze, or a help page shown because a required option was missing |

Invalid values are rejected before anything runs, with a message naming the problem and, where it helps, what the setting means:

```text
Invalid value: beta must be in (0, 1), got 1.5 (it is the base of the leaf heuristic beta ** distance to the goal)
Invalid value for '--max-steps': 0 is not in the range x>=1.
Invalid value for '--strategy' / '-S': 'XYZ' is not one of 'BFS', 'EXP', 'BFS-RO', 'EXP-RO'.
```

---

## 4. Development: tests, lint, CI

These commands use the locked environment (`uv sync --locked`, see [§2](#without-installing-uv-run)).

### 4.1 Tests

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
| `tests/test_analysis.py` | Loading runs, the cluster bootstrap, summaries, tables, figures and the `plot` command |
| `tests/test_episodes.py` | The episode engine and whole episodes played by a policy |
| `tests/test_cli.py` | Command-line options, short forms, help pages, defaults, prompts, output and errors |
| `tests/test_tui.py` | The explorer, its settings and the episode form, driven by key presses without a screen |

The interface tests use Textual's headless `run_test()`, so they run anywhere, including CI, without opening a window.

### 4.2 Lint and formatting

```bash
uv run ruff check .              # report lint problems
uv run ruff check --fix .        # fix what can be fixed automatically
uv run ruff format --check .     # report files that are not formatted
uv run ruff format .             # format the files
```

Ruff's settings live in `pyproject.toml` (line length 100, Python 3.12 target).

### 4.3 What CI checks

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

1. `uv sync --locked`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run pytest -q`
5. `uv run frozen-lake --help` (smoke test)

To check locally what CI checks before pushing, run the same five commands in that order. If they pass locally, CI should pass too.

---

## 5. Common workflows

**First time on a new machine**
```bash
git clone https://github.com/Mariojamjam/frozen-lake.git
cd frozen-lake
uv tool install --editable .     # the frozen-lake command
frozen-lake                      # every command and the typical workflow
```

**From the experiment to figures**
```bash
frozen-lake validate                       # check the configuration
frozen-lake benchmark                      # run it; prints the results folder
frozen-lake validate -i results/generated/<folder>
frozen-lake plot -i results/generated/<folder>    # writes <folder>/analysis/
```

**Before pushing**
```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest -q
```

**Investigate a failing test**
```bash
uv run pytest -x -v tests/test_episodes.py      # stop at the first failure, show test names
```

---

## 6. Using the package from Python

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
