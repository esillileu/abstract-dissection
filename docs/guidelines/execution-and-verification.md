# Execution, CLI & Verification Workflows

This document outlines command-line workflows, automation recipes via `just`, and testing methodologies.

---

## 1. Unified CLI Workflow (`repro`)

All monorepo capabilities are accessible via the unified `repro` CLI:

```bash
# Display top-level CLI help and discovered study plugins
uv run repro --help

# 1. Plan Study (dry-run inspect study execution parameters)
uv run repro dlfs plan ds1 -e 01
uv run repro dlfs plan ds2 -e 01

# 2. Run Experiments (local or tracked)
uv run repro dlfs run ds1 -e 01 --seeds 1,2,3 --device cpu
uv run repro dlfs run ds2 -e 01 --device cuda:0

# 3. Analyze Results (generates summary.md and publication figures)
uv run repro dlfs analyze ds1 -e 01
uv run repro dlfs analyze ds2 -e 01 --output-dir artifacts/analysis/dlfs/ds2

# 4. Inspect Status
uv run repro dlfs status
```

---

## 2. Justfile Automation Recipes

The repository root includes a [`justfile`](file:///home/esillileu/abstract-dissection/justfile) for standardized developer workflows:

| Command | Action |
| :--- | :--- |
| **`just check`** | Run full verification suite (`ruff check` + `ruff format --check` + `pytest -q`). |
| **`just test`** | Run full test suite with verbose output (`uv run pytest -v`). |
| **`just lint`** | Run linter and auto-fix formatting (`uv run ruff check --fix` + `uv run ruff format`). |
| **`just mlflow-up`** | Start local Docker MLflow tracking server and UI. |
| **`just mlflow-down`** | Stop local MLflow tracking server. |

---

## 3. Comprehensive Verification Strategy

The repository maintains three levels of automated verification:

1. **Unit & Mathematical Precision Tests (`packages/*/tests`):**
   * Verifies analytical gradients against numerical gradients (`numerical_gradient`).
   * Validates layer tensor shapes, forward/backward passes, and optimizer updates.
2. **Architecture Boundary Tests (`tests/test_deepscratch_architecture.py`):**
   * Enforces zero dependencies between `deepscratch` and `repro-core`.
   * Ensures unidirectional imports across packages and studies.
3. **Catalog 1-Update Smoke Execution Tests:**
   * Automatically iterates through all 27 experiment YAML specs in `studies/dlfs/` (15 in DS1, 12 in DS2).
   * Executes 1 update in memory to ensure forward, backward, loss, and optimizer steps work without runtime errors.
