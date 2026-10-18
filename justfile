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

# Open the terminal explorer (e.g. `just tui 10`, `just tui 6 9 --map-seed 7`)
[group('run')]
tui size="6" episode_seed="42" *args:
    uv run frozen-lake tui --size {{size}} --episode-seed {{episode_seed}} {{args}}

# Play one headless episode (e.g. `just episode 8 3`, `just episode 6 7 --quiet`)
[group('run')]
episode size="6" episode_seed="7" *args:
    uv run frozen-lake episode --size {{size}} --episode-seed {{episode_seed}} {{args}}

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
