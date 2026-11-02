# Abstract Dissection: Deep Learning & NLP Paper Reproduction Monorepo

`abstract-dissection` is a modular, multi-package research monorepo designed for independent reproductions of foundational deep learning and NLP papers.

---

## 1. Architecture Overview

```text
abstract-dissection/
├── pyproject.toml              # Unified root workspace configuration
├── uv.lock                     # Monorepo lockfile
├── .python-version             # Hermetic Python version (3.12.14)
├── justfile                    # Top-level orchestration tasks
│
├── packages/                   # Reusable Libraries & Infrastructure
│   ├── repro-core/             # Central CLI, path resolver, config parser, tracking-neutral runner
│   ├── repro-mlflow/           # MLflow tracking client, artifact caching, result loading
│   ├── repro-io/               # Zero-dependency HTTP/S3 streaming, SHA-256, Common Crawl protocol
│   ├── w2v/                    # High-performance Rust computation library (PyO3)
│   └── deepscratch/            # Standalone scratch deep learning library (Tensor, Autograd, Layers, Optim)
│
├── studies/                    # Independent Study Packages
│   ├── dlfs/                   # Deep Learning from Scratch (Vol 1 & 2) reproductions (Legacy runner)
│   ├── f2/                     # Word2Vec (2013) reproductions (W2V1, W2V2, unified lifecycle)
│   └── f2_cc/                  # Independent Common Crawl candidate sampling & audit pipeline
│
├── references/                 # Read-only Reference Code & Papers
│   ├── dlfs1-book/             # Book 1 reference implementation
│   ├── dlfs2-book/             # Book 2 reference implementation
│   └── papers/                 # Original paper PDFs
│
├── workspace/                  # Unified Local Workspace Root (REPRO_WORKSPACE_ROOT)
│   ├── inputs/                 # Corpora, benchmarks, prepared/preprocessed data
│   ├── runs/                   # Per-run execution state, checkpoints, observations
│   └── analysis/               # Download cache (mlflow), computation cache (results), final reports
│
└── tests/                      # Monorepo integration & architecture boundary tests
```

---

## 2. Quick Start

### 2.1 Workspace Setup (Hermetic uv Environment)

No global Python installation is required. `uv` handles Python downloading, virtual environment creation, and dependency syncing:

```bash
# Sync all packages, dev tools, and extras in a single step
just sync
# Or using uv directly:
uv sync --all-packages --all-extras --dev
uv run maturin develop --release --manifest-path packages/w2v/Cargo.toml
```

### 2.2 CLI Usage (`repro`)

All study executions and analyses run through the unified `repro` entrypoint:

```bash
# List registered reproduction studies
uv run repro list

# Inspect repository runtime paths and backend status
uv run repro info

# --- DLFS Reproduction Suite ---
uv run repro dlfs plan ds1 -e 01 --seed 1
uv run repro dlfs run ds1 -e 01 --seed 1
uv run repro dlfs analyze ds1 -e 01

# --- F2 Word2Vec Reproduction Suite ---
# 1. Preflight service check
uv run repro f2 preflight

# 2. Inspect experiment plan
uv run repro f2 suites
uv run repro plan f2 w2v1 -e 01
uv run repro plan f2 w2v2 -e 01

# 3. Analyze results (offline or via MLflow cache)
uv run repro f2 analyze
uv run repro f2 w2v2-report --offline
```

Or using `just`:

```bash
just check
just test
just lint
```

---

## 3. Reproduction Study Guidelines

When adding or maintaining paper reproductions in `studies/`:

1. **Follow Layer Invariants:**
   - Packages in `packages/` provide reusable mechanisms and numerical routines. They never depend on studies.
   - Studies define scientific policies, specifications, and analysis.
2. **F2 Execution & Storage Architecture:**
   - Single lifecycle coordinator decides skip/resume/run/publish.
   - Native MLflow `FINISHED` is the sole execution completion authority.
   - Local runs and checkpoints are preserved under `workspace/runs/`.
   - Deliverables have stable, fixed paths under `workspace/analysis/results/`.
   - See [`ref.md`](ref.md) and [`docs/architecture/f2-experiments.md`](docs/architecture/f2-experiments.md) for details.
3. **Hermetic Verification Gate:**
   - Always run `just check` before committing changes. Ensure all tests pass with zero lint/format errors.
