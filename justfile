set shell := ["bash", "-cu"]

# Synchronize monorepo workspace dependencies
sync:
    uv sync --all-packages --all-extras --dev

# Run all checks (lint, format, tests)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q

# Fix linting and formatting errors
lint:
    uv run ruff check --fix .
    uv run ruff format .


# Run test suite
test *args:
    uv run pytest {{args}}

# Run repro CLI
repro *args:
    uv run repro {{args}}
