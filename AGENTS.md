# AI Agent Guidelines & Monorepo Operations

Welcome to the `abstract-dissection` reproduction monorepo. This document defines the mandatory operating conditions, architectural invariants, and documentation map for all AI coding agents working on this codebase.

---

## 1. Mandatory Operating Conditions

All AI agents working in this repository MUST adhere to the following 6 golden rules:

### 1) Strict Hermetic Python (`uv run`)
* **Rule:** NEVER execute bare system commands such as `python`, `pip`, `pytest`, or `ruff`.
* **Standard:** ALWAYS prefix Python executions and tools with `uv run` (e.g., `uv run pytest`, `uv run repro ...`, `uv run python -c "..."`).
* **Environment:** Python 3.11.11 managed via root [`pyproject.toml`](file:///home/esillileu/abstract-dissection/pyproject.toml) and [`uv.lock`](file:///home/esillileu/abstract-dissection/uv.lock).

### 2) Zero-Dependency Invariants & Package Isolation
* **`deepscratch`:** 100% standalone deep learning library. NEVER import `repro_core`, `repro_mlflow`, or studies (`dlfs`). Allowed: `numpy`, `psutil`, (optional: `cupy`).
* **`repro-core`:** Tracking-neutral experiment orchestration library. NEVER import `repro_mlflow`, `deepscratch`, or studies. Orchestrates executions via injected delegates and module-scoped executor dispatch (`ExecutionDefinition.executor_module`).
* **Directionality:** Unidirectional dependency flow: $\text{studies} \rightarrow \text{repro-mlflow} \rightarrow \text{repro-core} \rightarrow \text{standard library / third-party}$.


### 3) 3-Tier Dataset Resolution & Path Injection
* **Rule:** NEVER hardcode paths to dataset files (e.g. `./mnist`, `/tmp/data`).
* **Inside `deepscratch`:** Rely on 3-tier fallback (`data_dir` arg $\rightarrow$ `DEEPSCRATCH_DATA_DIR` env $\rightarrow$ `./data/<name>`).
* **Inside `studies/`:** Always inject paths via translation adapters using `RuntimePaths.from_environment().dataset("<name>")`.

### 4) Storage Lifecycle & Cache Discipline
* **`.staging/`:** Ephemeral in-memory/scratch buffer during runs. Can be wiped at any time.
* **`.cache/`:** Reconstructible caches (MLflow download cache, analysis cache).
* **`artifacts/analysis/`:** Only human-readable deliverables (paper plots `.png`/`.pdf`, `summary.md`, `summary.csv`).
* **MLflow:** Single source of truth for all production run checkpoints, manifests, and metrics.

### 5) Immutable References
* **Rule:** Never modify files inside [`references/`](file:///home/esillileu/abstract-dissection/references).
* **Policy:** Reference implementations are vendored snapshots with exact upstream commit hashes tracked in `provenance.json`. They are read-only.

### 6) Verification Gate
* **Rule:** Before completing any user request or committing code, always execute:
  ```bash
  just check
  ```
  Ensure all 500+ tests pass and linter/formatter have 0 errors.

---

## 2. Documentation Map

Before implementing features or refactoring, read the relevant detailed documentation in [`docs/`](file:///home/esillileu/abstract-dissection/docs):

```
docs/
├── architecture/
│   ├── overview.md                       # Monorepo 4-tier layer topology and dependency rules
│   ├── packages.md                       # Responsibilities and strict boundaries of packages/
│   ├── studies.md                        # Structure of studies/ and dynamic repro CLI plugin discovery
│   └── adapters.md                       # Adapter boundary & representation translation contracts
└── guidelines/
    ├── path-and-storage.md               # Storage tiers, MLflow upload rules, and 3-tier path precedence
    ├── reproducibility-and-references.md # Reference vendoring, provenance.json, and seed streams
    └── execution-and-verification.md     # CLI usage, just recipes, and testing/smoke test workflows
```

* **For Package & Core Work:** Read [`docs/architecture/packages.md`](file:///home/esillileu/abstract-dissection/docs/architecture/packages.md).
* **For Study & Experiment Work:** Read [`docs/architecture/studies.md`](file:///home/esillileu/abstract-dissection/docs/architecture/studies.md).
* **For Adapter & Representation Translation:** Read [`docs/architecture/adapters.md`](file:///home/esillileu/abstract-dissection/docs/architecture/adapters.md).
* **For Paths, Datasets & Storage:** Read [`docs/guidelines/path-and-storage.md`](file:///home/esillileu/abstract-dissection/docs/guidelines/path-and-storage.md).
* **For Reproducibility & RNG:** Read [`docs/guidelines/reproducibility-and-references.md`](file:///home/esillileu/abstract-dissection/docs/guidelines/reproducibility-and-references.md).
* **For Running, Testing & CLI:** Read [`docs/guidelines/execution-and-verification.md`](file:///home/esillileu/abstract-dissection/docs/guidelines/execution-and-verification.md).

---

## 3. Quick Reference Commands

```bash
# Verify workspace integrity (Linting + Formatting + Pytest)
just check

# Run tests with verbose output
just test

# Fix formatting and linter errors
just lint

# Inspect CLI and study commands
uv run repro --help
uv run repro dlfs plan ds1 -e 01
uv run repro dlfs plan ds2 -e 01
```
