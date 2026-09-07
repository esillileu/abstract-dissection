# F2 Word2Vec (2013) Paper Reproduction & Corpus Feasibility Study

This study package implements the Common Crawl (2009–2012) corpus acquisition, feasibility study, and word-embedding reproduction pipeline for Mikolov et al. (2013, *Distributed Representations of Words and Phrases and their Compositionality*).

---

## 1. Study Overview

* **Primary Objective**: Determine whether Common Crawl snapshots from 2009–2012 contain enough usable English news/article text to build reproducible training corpora up to 1B, 6B, and 33B words, and track reproduction resources, specifications, and execution plans for Word2Vec models.
* **Subsystems & Architecture**:
  * **Common Infrastructure (`f2/common/`)**: Shared rate-limited HTTP range fetching (`common/network/`), tabular Parquet/JSONL persistence (`common/storage/`), bootstrap variance & two-phase difference estimators (`common/stats/`), and publication plot styling (`common/analysis/`).
  * **Corpus Pipeline (`f2/corpus/`)**: Two-stage crawl-stratified sampling, rate-limited range fetcher, news classifier, Two-Phase 8-Stratum estimator, and full corpus lifecycle, multi-hop lineage DAG & validation metadata store (`schema: corpus`).
  * **Reproduction Catalog (`f2/catalog/`)**: PostgreSQL catalog tracking paper targets, experiment specs, resource identity SSOT (`catalog.resources`), resource lineage/substitutions, execution plan revisions, and planned run slots (`schema: catalog`, `schema.dbml`, `CatalogPlanMaterializer`, `CatalogRepository`).
  * **Experimental Suites (`f2/suites/`)**: Modular volumes for downstream Word2Vec pretraining, vocabulary scaling, and embedding evaluation benchmarks.

---

## 2. CLI Usage (`repro f2`)

### A. Corpus Pipeline Commands
```bash
# Apply pending corpus operational DB migrations
uv run repro f2 corpus migrate

# Inspect sampling plan & crawl strata
uv run repro f2 corpus plan --crawl CC-MAIN-2012 --sample-size 100 --seed 42

# Execute bounded feasibility sample (safe for shared research network)
uv run repro f2 corpus sample \
    --crawls CC-MAIN-2009-2010,CC-MAIN-2012 \
    --sample-size 10000 \
    --seed 42 \
    --bandwidth-limit 20 \
    --concurrency 2 \
    --output-dir .staging/exp/f2/sample_10k

# Generate feasibility analytics report
uv run repro f2 corpus analyze \
    --manifest .staging/exp/f2/sample_10k/provenance.jsonl \
    --output-dir artifacts/analysis/f2/corpus
```

### B. Word2Vec Generic Sources Commands (`repro f2 corpus sources`)
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

### C. Reproduction Catalog Commands
```bash
# Apply pending reproduction catalog DB migrations
uv run repro f2 catalog migrate

# Inspect canonical execution plan progress and resource inventory
uv run repro f2 catalog status

# Inspect expected run slots with MLflow execution pointers
uv run repro f2 catalog matrix
```
