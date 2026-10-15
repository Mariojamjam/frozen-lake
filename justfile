# Thin wrappers around documented `uv run` commands. Optional for contributors.

default:
    @just --list

setup:
    uv sync --locked

test:
    uv run pytest -q

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff format .
    uv run ruff check --fix .

check: lint test

ui:
    uv run frozen-lake tui --size 4 --strategy BFS --seed 42

pilot:
    uv run frozen-lake benchmark --config configs/pilot.toml

main:
    uv run frozen-lake benchmark --config configs/main.toml

plots run_dir:
    uv run frozen-lake plot --input {{run_dir}}
