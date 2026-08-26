# Abstract Dissection: Deep Learning & NLP Paper Reproduction Monorepo

`abstract-dissection` is a modular, multi-package research monorepo designed for independent reproductions of foundational deep learning and NLP papers.

---

## 1. Architecture Overview

```text
abstract-dissection/
├── pyproject.toml              # Unified root workspace configuration
├── uv.lock                     # Monorepo lockfile
├── .python-version             # Hermetic Python version (3.11.11)
├── justfile                    # Top-level orchestration tasks
│
├── packages/                   # Reusable Libraries & Infrastructure
│   ├── repro-core/             # Central CLI, path resolver, config parser, runner
│   ├── repro-mlflow/           # MLflow tracking, checkpoint schema, transfer tools
│   └── deepscratch/            # Standalone scratch deep learning library (Tensor, Autograd, Layers, Optim)
│
├── studies/                    # Independent Study Packages
│   └── dlfs/                   # Deep Learning from Scratch (Vol 1 & 2) reproductions
│
├── references/                 # Read-only Reference Code & Papers
│   ├── dlfs1-book/             # Book 1 reference implementation
│   ├── dlfs2-book/             # Book 2 reference implementation
│   └── papers/                 # Original paper PDFs
│
├── data/                       # Central datasets (MNIST, PTB, Sequence) [.gitignore]
├── artifacts/                  # Runs, metrics, checkpoints, and figures [.gitignore]
├── infra/                      # Services (Local MLflow compose)
└── tests/                      # Monorepo integration & regression tests
```

---

## 2. Quick Start

### 2.1 Workspace Setup (Hermetic uv Environment)

No global Python installation is required. `uv` handles Python downloading, virtual environment creation, and dependency syncing:

```bash
# Sync all packages, dev tools, and extras in a single step
uv sync --all-packages --all-extras --dev
```

### 2.2 CLI Usage (`repro`)

All study executions and analyses run through the unified `repro` entrypoint:

```bash
# List registered reproduction studies
uv run repro list

# Inspect repository runtime paths and backend status
uv run repro info

# Inspect experiment plan
uv run repro dlfs plan ds1 -e 01 --seed 1

# Execute a reproduction experiment
uv run repro dlfs run ds1 -e 01 --seed 1

# Generate analysis reports and comparative figures
uv run repro dlfs analyze ds1 -e 01

# Profile GPU / update performance
uv run repro dlfs profile ds2 -e 10
```

Or using `just`:

```bash
just repro dlfs plan ds1 -e 01
just check
```

---

## 3. Adding a New Paper Reproduction

To add a new reproduction (e.g. `word2vec-2013`):

1. Create a package directory in `studies/word2vec_2013/`:
   ```text
   studies/word2vec_2013/
   ├── pyproject.toml
   └── src/word2vec_2013/
       ├── __init__.py
       ├── plugin.py
       └── ...
   ```
2. In `studies/word2vec_2013/pyproject.toml`, declare `repro-core` (and optionally `deepscratch` or `torch`) as dependencies and register the study entry point:
   ```toml
   [project]
   name = "word2vec-2013"
   version = "0.1.0"
   dependencies = [
       "repro-core",
       "deepscratch",  # or "torch>=2.0.0"
   ]

   [project.entry-points."repro.studies"]
   word2vec-2013 = "word2vec_2013.plugin:PLUGIN"
   ```
3. Implement `StudyPlugin` in `plugin.py`.
4. Run `uv sync`. The new study is instantly discoverable via `repro word2vec-2013` without modifying any global code.
