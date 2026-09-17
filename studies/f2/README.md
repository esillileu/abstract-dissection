# F2 Word2Vec (2013) Reproduction Catalog and Corpus Preparation

This study package owns the Word2Vec reproduction catalog, completed-corpus
preparation, and downstream experiments. Common Crawl acquisition and feasibility
analysis belong to the independent `f2_cc` producer.

---

## 1. Study Overview

* **Primary Objective**: Track the two Word2Vec papers' reproduction targets,
  resources, specifications, and execution plans, then prepare verified corpus
  releases for downstream experiments.
* **Subsystems & Architecture**:
  * **Common Infrastructure (`f2/common/`)**: Shared research statistics,
    analysis declarations, and study adapters.
  * **Corpus Preparation (`f2/corpus/`)**: Acquisition and normalization of
    completed source releases, with lifecycle, lineage, and validation metadata
    in the `corpus` schema.
  * **Reproduction Catalog (`f2/catalog/`)**: PostgreSQL catalog tracking paper targets, experiment specs, resource identity SSOT (`catalog.resources`), resource lineage/substitutions, execution plan revisions, and planned run slots (`schema: catalog`, `schema.dbml`, `CatalogPlanMaterializer`, `CatalogRepository`).
  * **Experimental Suites (`f2/suites/`)**: Modular volumes for downstream Word2Vec pretraining, vocabulary scaling, and embedding evaluation benchmarks.

---

## 2. CLI Usage (`repro f2`)

### A. Word2Vec Corpus Source Commands (`repro f2 corpus sources`)
```bash
# Inspect inventory and acquisition/readiness status across all sources
uv run repro f2 corpus sources status

# Acquire raw source archives (HTTP download -> S3 raw)
uv run repro f2 corpus sources acquire --source wmt
uv run repro f2 corpus sources acquire --source lm1b
uv run repro f2 corpus sources acquire --source umbc

# Process into 10M-word shards (Raw -> Canonical -> Word2Vec Normalized)
uv run repro f2 corpus sources process --source wmt
uv run repro f2 corpus sources process --source lm1b
uv run repro f2 corpus sources process --source umbc

# Validate S3 shards, SHA-256 integrity, and statistics
uv run repro f2 corpus sources validate --source wmt
uv run repro f2 corpus sources validate --source lm1b
uv run repro f2 corpus sources validate --source umbc
```

### B. Reproduction Catalog Commands
```bash
# Apply pending reproduction catalog DB migrations
uv run repro f2 catalog migrate

# Inspect canonical execution plan progress and resource inventory
uv run repro f2 catalog status

# Validate the W2V1/W2V2 research catalog without committing database changes
uv run repro f2 catalog load-manifest studies/f2/catalog/w2v.json

# Inspect expected run slots with MLflow execution pointers
uv run repro f2 catalog matrix
```
