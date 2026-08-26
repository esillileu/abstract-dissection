set shell := ["bash", "-cu"]

# Synchronize monorepo workspace dependencies
sync:
    uv sync --all-packages --all-extras --dev

# Run all checks (lint, format, tests)
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q

# Run test suite
test *args:
    uv run pytest {{args}}

# Run repro CLI
repro *args:
    uv run repro {{args}}

# Local MLflow Server Management
mlflow-up:
    docker compose -f infra/mlflow/compose.yaml up -d

mlflow-down:
    docker compose -f infra/mlflow/compose.yaml down

mlflow-logs:
    docker compose -f infra/mlflow/compose.yaml logs -f
