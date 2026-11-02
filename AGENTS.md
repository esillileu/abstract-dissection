# AI Agent Guidelines & Monorepo Operations

Welcome to the `abstract-dissection` reproduction monorepo. This document defines the mandatory operating conditions, architectural invariants, and documentation map for all AI coding agents working on this codebase.

---

## Required Documentation by Work Type

| Work type | Read first |
| :--- | :--- |
| Packages and core | `docs/architecture/packages.md` |
| Studies and experiments | `docs/architecture/studies.md` |
| F2 execution & storage architecture | `docs/architecture/f2-experiments.md` (and design spec `ref.md`) |
| Adapters and representation translation | `docs/architecture/adapters.md` |
| Corpus pipeline | `docs/architecture/corpus-pipeline.md` |
| Paths, storage, MLflow, and databases | `docs/guidelines/path-and-storage.md` |
| Reproducibility and RNG | `docs/guidelines/reproducibility-and-references.md` |
| CLI, tests, and verification | `docs/guidelines/execution-and-verification.md` |

`docs/guidelines/path-and-storage.md` is the canonical source for service
connection ownership and environment-variable contracts.

---

## 1. Mandatory Operating Conditions

All AI agents working in this repository MUST adhere to the following 6 golden rules:

### 1) Strict Hermetic Python (`uv run`)
* **Rule:** NEVER execute bare system commands such as `python`, `pip`, `pytest`, or `ruff`.
* **Standard:** ALWAYS prefix Python executions and tools with `uv run` (e.g., `uv run pytest`, `uv run repro ...`, `uv run python -c "..."`).
* **Environment:** Python 3.12.14 managed via root [`pyproject.toml`](file:///home/esillileu/abstract-dissection/pyproject.toml) and [`uv.lock`](file:///home/esillileu/abstract-dissection/uv.lock).

### 2) Zero-Dependency Invariants & Package Isolation
* **`deepscratch`:** 100% standalone deep learning library. NEVER import `repro_core`, `repro_mlflow`, `repro_io`, or studies (`dlfs`, `f2`, `f2_cc`). Allowed: `numpy`, `psutil`, (optional: `cupy`).
* **`w2v`:** Standalone high-performance Rust computation library (PyO3). Pure computational algorithms (CBOW/SG, HS/NEG/NCE, NNLM, Downpour SGD). NEVER import MLflow, study catalogs, or F2 storage paths.
* **`repro-io`:** Dedicated, tracking-neutral I/O and protocol library (HTTP range/streaming, S3 SigV4, SHA-256, Common Crawl protocol/ARC). NEVER import workspace studies, MLflow, or study-specific business logic.
* **`repro-core`:** Tracking-neutral experiment orchestration library. NEVER import `repro_mlflow`, `deepscratch`, or studies. Orchestrates executions via injected delegates and module-scoped executor dispatch (`ExecutionDefinition.executor_module`).
* **Directionality:** Unidirectional dependency flow: $\text{studies} \rightarrow \text{repro-mlflow} \rightarrow \text{repro-core} \rightarrow \text{standard library / third-party}$.
* **Study & Execution Independence:**
  * **F2 Execution & Storage Architecture:** F2 employs a single, unified execution lifecycle managed by its Execution Coordinator and Storage layer. Completion is judged solely by native MLflow `FINISHED`. F2 does NOT share run lifecycle / execution policy with DLFS.
  * **DLFS Legacy Isolation:** `studies/dlfs` maintains its legacy runner/locator execution convention. DLFS is NOT required to adopt F2's new execution contracts, and F2 must not force new interfaces onto DLFS.
  * **F2-CC Study:** `studies/f2_cc` is an independent Common Crawl producer study with its own database (`f2_cc`); it is not an executor for F2 paper experiments.


### 3) 3-Tier Dataset Resolution & Path Injection
* **Rule:** NEVER hardcode paths to dataset files (e.g. `./mnist`, `/tmp/data`).
* **Inside `deepscratch`:** Rely on 3-tier fallback (`data_dir` arg $\rightarrow$ `DEEPSCRATCH_DATA_DIR` env $\rightarrow$ `./data/<name>`).
* **Inside `studies/`:** Always inject paths via translation adapters using `RuntimePaths.from_environment().dataset("<name>")`.

### 4) Storage Lifecycle, Cache Discipline & Execution Boundaries
* **Single Workspace Root:** All local runtime output, checkpoints, datasets, and analysis caches are managed under `REPRO_WORKSPACE_ROOT` (`workspace/`):
  * `workspace/inputs/`: Local input corpora, benchmarks, and reusable prepared/preprocessed data (`workspace/inputs/f2/prepared/corpus/`). Shared across runs and suites if source revision and transformation policies match.
  * `workspace/runs/`: Per-run local results, checkpoints, and intermediate files (`workspace/runs/f2/<suite>/<slot>/<attempt>/`). Preserved on override (never wiped automatically; re-runs allocate new Attempt IDs or indexed subdirectories).
  * `workspace/analysis/cache/mlflow/`: Downloaded MLflow artifacts isolated per tracking server and run ID (`root / <server_key> / <run_id> /`).
  * `workspace/analysis/cache/results/`: Signature-checked intermediate statistical analysis and evaluation cache (`eval_*.json`). Fully reconstructible from MLflow artifacts and benchmarks.
  * `workspace/analysis/results/`: Final human-readable deliverables (paper plots `.png`/`.pdf`, `summary.md`, `summary.csv`, JSON tables). Each table/filter has a **stable, fixed path**; updates overwrite the stable location rather than creating unbounded duplicate version files.
* **MLflow as Single Source of Truth:** Published production run checkpoints, manifests, and metrics are authoritative in MLflow.
  * In F2, execution completion is judged exclusively by MLflow native `FINISHED` status (no redundant `durable_complete`, `trial.status`, or external completion registries).
  * Prior to marking `FINISHED`, all required artifacts must be verified and durably uploaded.
* **Separation of Policy and Mechanism (F2 Architecture):**
  * **Engine (`packages/w2v`, `packages/deepscratch`):** Pure computation and numerical routines (no storage paths, no MLflow access, no execution policy).
  * **Coordinator (`studies/f2/execution`):** Sole policy decision maker for the execution lifecycle (skip on existing FINISHED / resume from valid checkpoint / publish unposted local complete / new attempt).
  * **Storage & Publication (`studies/f2/storage`):** Local attempt layout, checkpoint search/retention, atomic MLflow publication, and download cache management.
  * **Suite Executor (`studies/f2/suites/`):** Pure execution sequencing, numerical configuration, and adapter invocation. Does not manage MLflow lifecycle or local retry policies.
  * **Analyzer (`studies/f2/common/analysis` & suite analyzers):** Independent analysis; reads MLflow download cache or local result cache, produces reports. Never trains, resumes checkpoints, or touches raw corpora. Missing inputs are marked `unavailable` (never masked with zeros).
* **Zero Legacy Shim / No Artificial Fallback:** Never add permanent wrappers, shims, or fallback aliases for obsolete APIs or deprecated storage layouts. Required migrations must be executed once cleanly.

### 5) Immutable References
* **Rule:** Never modify files inside [`references/`](file:///home/esillileu/abstract-dissection/references).
* **Policy:** Reference implementations are vendored snapshots with exact upstream commit hashes tracked in `provenance.json`. They are read-only.

### 6) Verification Gate
* **Rule:** Before completing any user request or committing code, always execute:
  ```bash
  just check
  ```
  Ensure all 500+ tests pass and linter/formatter have 0 errors.
* **Elevated execution:** Always run the root `just check` outside the filesystem
  sandbox with elevated permissions. Its verification suite may start a disposable
  rootless Podman PostgreSQL instance and must be able to write under the user's
  runtime directory (for example, `/run/user/<uid>/libpod`). Request approval for
  the narrow `just check` command prefix when no persistent approval exists; do not
  broaden writable roots or grant general shell access merely to run this gate.
* **Environment-failure stop rule:** If the verification command cannot start or
  complete because of an environment or infrastructure problem rather than a
  code/test failure (for example: a missing executable, broken virtual
  environment, dependency-sync problem, unavailable service, credentials,
  permissions, network, GPU, or production database configuration), report the
  failure and **stop immediately**.
  * Do **not** run `uv sync`, install/reinstall packages, mutate `.venv`, change
    credentials or environment variables, start services, apply migrations, or
    retry against another environment unless the user explicitly authorizes it.
  * Do **not** run tests when the active environment points at production or
    shared external services. Treat this as an environment failure and stop.
  * Never attempt to repair the environment merely to satisfy the verification
    gate. Record the command, the environmental cause, and which checks did or
    did not run.

---

## 2. Documentation Map

Before implementing features or refactoring, read the relevant detailed documentation in [`docs/`](file:///home/esillileu/abstract-dissection/docs):

```
docs/
├── architecture/
│   ├── overview.md                       # Monorepo 4-tier layer topology and dependency rules
│   ├── packages.md                       # Responsibilities and strict boundaries of packages/
│   ├── studies.md                        # Structure of studies/ and dynamic repro CLI plugin discovery
│   ├── f2-experiments.md                 # F2 Word2Vec execution, storage & analysis contracts
│   ├── adapters.md                       # Adapter boundary & representation translation contracts
│   └── corpus-pipeline.md                # Common Crawl sampling, auditing, two-phase estimation, corpus extraction & lifecycle lineage
└── guidelines/
    ├── path-and-storage.md               # Storage tiers, MLflow upload rules, and 3-tier path precedence
    ├── reproducibility-and-references.md # Reference vendoring, provenance.json, and seed streams
    └── execution-and-verification.md     # CLI usage, just recipes, and testing/smoke test workflows
```

* **For Package & Core Work:** Read [`docs/architecture/packages.md`](file:///home/esillileu/abstract-dissection/docs/architecture/packages.md).
* **For Study & Experiment Work:** Read [`docs/architecture/studies.md`](file:///home/esillileu/abstract-dissection/docs/architecture/studies.md).
* **For F2 Execution & Storage Architecture:** Read [`docs/architecture/f2-experiments.md`](file:///home/esillileu/abstract-dissection/docs/architecture/f2-experiments.md) and [`ref.md`](file:///home/esillileu/abstract-dissection/ref.md).
* **For Adapter & Representation Translation:** Read [`docs/architecture/adapters.md`](file:///home/esillileu/abstract-dissection/docs/architecture/adapters.md).
* **For Corpus Sampling, Auditing & Pipeline Work:** Read [`docs/architecture/corpus-pipeline.md`](file:///home/esillileu/abstract-dissection/docs/architecture/corpus-pipeline.md).
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
uv run repro f2 corpus analyze
uv run repro f2 corpus calibrate
```


## Acquisition and semantic boundaries

- Component ownership mapping follows implementation boundaries.
- `repro-io` owns HTTP, S3, checksum, Common Crawl protocol and ARC parsing only.
  No workspace/study imports, F2 environment interpretation or research policy.
- `w2v` owns core algorithmic computation (CBOW/SG, HS/NEG/NCE, NNLM Downpour SGD); it has no knowledge of MLflow or study catalogs.
- Core analysis is tracking-neutral; DLFS owns run selection, book smoothing and
  parameter reports. MLflow result loading belongs to repro-mlflow.
- F2 SQL/DBML and migration ledgers stay in the study. Never modify existing SQL
  during ownership moves. Infra specifies service deployment, not domain tables.
- DB tests use only F2_TEST_DATABASE_URL or disposable PostgreSQL 18; no production
  dotenv fallback. Unavailable DB verification fails. See infra/ for the external
  service handoff, not for repository-owned operations.
