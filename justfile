# Shortcuts for common tasks. Every recipe wraps a documented `uv run` command,
# so `just` is optional: the commands also work on their own.

# List the available recipes
default:
    @just --list

# --- setup ---------------------------------------------------------------------

# Install the exact locked dependencies
[group('setup')]
setup:
    uv sync --locked

# Remove tool caches and bytecode
[group('setup')]
clean:
    rm -rf .pytest_cache .ruff_cache
    find . -path ./.venv -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +

# --- run -----------------------------------------------------------------------

# Open the terminal explorer (e.g. `just tui 10`, `just tui 8 12 7`)
[group('run')]
tui size="6" episode_seed="42" map_seed="42" *args:
    uv run frozen-lake tui --size {{size}} --episode-seed {{episode_seed}} --map-seed {{map_seed}} {{args}}

# Play one headless episode (e.g. `just episode 8 3`, `just episode 6 7 --quiet`)
[group('run')]
episode size="6" episode_seed="7" *args:
    uv run frozen-lake episode --size {{size}} --episode-seed {{episode_seed}} {{args}}

# Check the experiment configuration
[group('experiment')]
validate:
    uv run frozen-lake validate --config configs/experiment.toml

# Run the experiment, from configs/experiment.toml unless another config is given (e.g. `just benchmark my.toml runs`)
[group('experiment')]
benchmark config="configs/experiment.toml" output_dir="" *args:
    uv run frozen-lake benchmark --config {{quote(config)}} {{ if output_dir != "" { "--output-dir " + quote(output_dir) } else { "" } }} {{args}}

# --- quality -------------------------------------------------------------------

# Run tests; extra arguments go to pytest (e.g. `just test tests/test_bfs.py -k fallback`)
[group('quality')]
test *args:
    uv run pytest -q {{args}}

# Lint and check formatting without changing files
[group('quality')]
lint:
    uv run ruff check .
    uv run ruff format --check .

# Fix lint issues and format the code
[group('quality')]
format:
    uv run ruff check --fix .
    uv run ruff format .

# Run exactly what CI runs, before you push
[group('quality')]
ci:
    uv sync --locked
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q
    uv run frozen-lake --help > /dev/null
