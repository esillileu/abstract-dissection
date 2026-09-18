# Execution, CLI & Verification Workflows

This document outlines command-line workflows, automation recipes via `just`, and testing methodologies.

---

## 1. Unified CLI Workflow (`repro`)

All monorepo capabilities are accessible via the unified `repro` CLI:

```bash
# Display top-level CLI help and discovered study plugins
uv run repro --help

# -------------------------------------------------------------
# A. DLFS Reproduction Suite (Vision & NLP Experiments)
# -------------------------------------------------------------
# 1. Plan Study (dry-run inspect study execution parameters)
uv run repro dlfs plan ds1 -e 01
uv run repro dlfs plan ds2 -e 01

# 2. Run Experiments (local or tracked)
uv run repro dlfs run ds1 -e 01 --seed 1 --device cpu
uv run repro dlfs run ds2 -e 01 --device cuda:0

# 3. Analyze Results (generates summary.md and publication figures)
uv run repro dlfs analyze ds1 -e 01
uv run repro dlfs analyze ds2 -e 01

# 4. Inspect Status / Completeness
uv run repro dlfs check ds1 -e 01
uv run repro dlfs check ds2 -e 01

# -------------------------------------------------------------
# B. F2 Campaign Suite (Corpus Pipeline & Reproduction Catalog)
# -------------------------------------------------------------
# 1. Independent Common Crawl producer
uv run repro f2-cc migrate
uv run repro f2-cc corpus sample --run-id <run-id>
uv run repro f2-cc corpus analyze --manifest <provenance.parquet>
uv run repro f2-cc corpus calibrate --manifest <provenance.parquet> --audit-file <audit.jsonl>

# 2. F2 reproduction catalog and completed-source preparation
uv run repro f2 catalog migrate
uv run repro f2 catalog load-manifest studies/f2/catalog/w2v.json
uv run repro f2 catalog status
uv run repro f2 catalog matrix
uv run repro f2 corpus sources status

# 3. Word2Vec execution readiness and fixture smoke runs
uv run repro plan f2 w2v1 -e 01 -a local-smoke --seed 1
uv run repro run f2 w2v1 -e 01 -a local-smoke --seed 1 --progress line
uv run repro run f2 w2v2 -e 02 -a local-smoke --seed 1 --progress line

# 4. Canonical training (after preflight and explicit cost approval)
uv run repro f2 preflight
uv run repro run f2 w2v1 -e 01 -a wmt--d50-w24m --seed 1 \
  --tracking-uri "$F2_MLFLOW_TRACKING_URI" --approve-large-run --progress auto
```

F2 Word2Vec progress covers the selected-run count, service preflight, corpus
shards and lexical-token materialization, phrase passes/documents, completed
training epochs with loss and throughput, and artifact publication. Canonical
runs are rejected unless `--approve-large-run` is explicit; `--dry-run` and the
`local-smoke` fixture do not require approval.

Tracked DLFS commands require `F1_MLFLOW_TRACKING_URI` unless `--tracking-uri`
is supplied. The complete service connection contract is defined in
[`path-and-storage.md`](path-and-storage.md). DLFS analysis outputs resolve through `RuntimePaths` under
`artifacts/analysis/dlfs/<volume>/` unless `--output-dir` is explicitly supplied.

---

## 2. Justfile Automation Recipes

The repository root includes a [`justfile`](file:///home/esillileu/abstract-dissection/justfile) for standardized developer workflows:

| Command | Action |
| :--- | :--- |
| **`just check`** | Run full verification suite (`ruff check` + `ruff format --check` + `pytest -q`). |
| **`just test`** | Run full test suite with verbose output (`uv run pytest -v`). |
| **`just lint`** | Run linter and auto-fix formatting (`uv run ruff check --fix` + `uv run ruff format`). |

---

## 3. Comprehensive Verification Strategy

The repository maintains four levels of automated verification (500+ tests):

1. **Unit & Mathematical Precision Tests (`packages/*/tests`):**
   * Verifies analytical gradients against numerical gradients (`numerical_gradient`).
   * Validates layer tensor shapes, forward/backward passes, and optimizer updates.
2. **Representation & Checkpoint Adapter Tests (`studies/dlfs/src/dlfs/ds*/tests/`):**
   * Verifies model, objective, optimizer, and batch adapter construction from YAML configurations.
   * Verifies full roundtrip serialization and restoration of model parameters, buffers, and backend RNG states.
3. **Architecture Boundary & Multi-Study Isolation Tests (`tests/`):**
   * Enforces zero dependencies between `deepscratch`, `repro-core`, and `repro-mlflow` (`test_deepscratch_architecture.py`).
   * Enforces that `repro_core.context.checkpoint` contains zero deep-learning coupling tokens.
   * Verifies module-scoped executor resolution and plugin error diagnostics (`test_study_isolation_and_discovery.py`).
4. **Catalog 1-Update Smoke Execution Tests:**
   * Automatically iterates through all 27 experiment YAML specs in `studies/dlfs/` (15 in DS1, 12 in DS2).
   * Executes 1 update in memory to ensure forward, backward, loss, and optimizer steps work without runtime errors.


Database integration tests are marked `database` and are excluded from the default
`just check` gate. Run them explicitly with `just test-db`; they require the study's
dedicated test URL or an available rootless Podman/Docker runtime for a disposable
PostgreSQL 18 instance. Ordinary `.env` database settings are never used by fixtures.
Network smoke tests remain separately marked `network` and excluded by default.
Independent repro-io verification is documented in its package README.
