# Common Crawl Corpus Pipeline & Auditing Architecture

This document describes the Common Crawl corpus sampling, auditing, calibration, and ingestion subsystem in `studies/f2`. This pipeline serves as the **foundational Phase-0 data infrastructure** for the broader `f2` research campaign (a suite of 8~10 studies comprising 2 corpus-dependent Word2Vec training studies and 5 independent benchmark studies).

---

## 1. Subsystem Architecture & Module Map

```text
studies/f2/
├── pyproject.toml                     # Workspace dependencies (repro-core, duckdb, psycopg, typer)
└── src/f2/
    ├── plugin.py                      # Repro CLI Plugin Entrypoint (discoverable by repro_core)
    ├── cli.py                         # Typer CLI: migrate, sample, process, audit-plan, audit-label, analyze, calibrate, build
    ├── cdx.py                         # CDX binary cluster index reader & block locator (O(log N) byte-range lookup)
    ├── discovery.py                   # 2-stage Horvitz-Thompson probability sampler & sequential audit priority sampler
    ├── fetcher.py                     # HTTP Range fetcher against Common Crawl ARC/WARC archives with retry/backoff
    ├── pipeline.py                    # Content extraction, language ID, validity filtering, and news scoring
    ├── storage.py                     # DuckDB / Parquet / JSONL provenance storage & clean text shard writers
    ├── analysis.py                    # Two-Phase Stratified Difference Estimator & B=10,000 bootstrap variance engine
    ├── calibration.py                 # Offline classifier calibration, pre-fetch filter ablation, & 10k reject-side FN validation
    └── db/                            # PostgreSQL migrations, schema, session, and repository layer
        ├── migrations/
        │   ├── 001_initial_corpus_schema.sql
        │   ├── 002_add_audit_sample_tracking.sql
        │   └── runner.py
        ├── repository.py              # CorpusStateRepository (bulk insertion, transaction-safe audit logging)
        └── session.py                 # Connection pooling and environment configuration
```

---

## 2. Core Invariants & Methodological Protocols

### 1) Two-Phase Stratified Difference Estimator
The total true-news word yield $\hat{W}_{\text{true}}$ across Common Crawl crawls $c \in \{\text{2009-2010}, \text{2012}\}$ is estimated without assuming 100% classifier precision or recall:

$$\hat{W}_{\text{true}, c} = \hat{W}_{\text{proxy}, c} + \hat{E}_c = \sum_{i \in s_{1, c}} w_{1, cki} y_i^{\text{proxy}} + \sum_{h \in \{0, 1\}} \frac{N_{c, h}^{(1)}}{n_{c, h}^{(2)}} \sum_{i \in s_{2, c, h}} w_{1, cki} (y_i^{\text{gold}} - y_i^{\text{proxy}})$$

* **Phase 1 Sample ($s_1$):** Large-scale probability sample ($N=10,000$) drawn from CDX cluster index blocks using inclusion probabilities $\pi_{1, i} = \pi_{\text{block}} \times \pi_{\text{record}|\text{block}}$.
* **Phase 2 Audit Sample ($s_2$):** Stratified subsample ($n=400$) drawn from Phase 1, allocated equally across Stratum 0 (predicted non-news) and Stratum 1 (predicted news), with deterministic `priority_order` ranking for sequential waves ($n=200 \to 300 \to 400$).
* **Variance Engine:** $B=10,000$ Two-Phase Stratified Residual Bootstrap resampling Phase 1 units and Phase 2 residuals within strata with replacement.

### 2) Pre-Fetch vs. Post-Fetch Separation
* **Stage 1: Pre-Fetch CDX Filter (`Rule 1 Only`):** Discards non-HTML binary media extensions (`.pdf`, `.jpg`, `.png`, `.mp4`, `.zip`, `.js`, `.css`) at the CDX index level *before* issuing ARC byte-range HTTP requests. Saves **44.37% of network download bandwidth** (with PDF exclusion contributing 40.12%) with 0 gold false negatives in the audit sample.
* **Stage 2: Post-Fetch Calibrated Extraction ($\tau \approx 1.25$):** Parses HTML, verifies language (English $\ge 0.50$), validates word length ($\ge 100$ words), and applies news scoring. Eliminates **44.12% of non-news text** from local NVMe/SSD storage while maintaining **98.71% out-of-fold word recall**.

### 3) 3-Tier Storage Lifecycle for Corpus Pipeline
* **PostgreSQL:** Transactional source of truth for candidate metadata, sampling weights, processing diagnostics, and gold audit annotations.
* **`.staging/exp/f2/`:** Ephemeral download shards, intermediate text extractions, and scratch audit sheets. Safe to wipe at any time.
* **`artifacts/analysis/f2/`:** Publication deliverables (`feasibility_report.md`, `feasibility_summary.csv`, `offline_calibration_and_prefetch_study.md`, `audit_set_400_annotated.jsonl`).

---

## 3. CLI Command Reference

All corpus operations are executed via `uv run repro f2 corpus <subcommand>`:

```bash
# 1. Database Migrations
uv run repro f2 corpus migrate

# 2. Probability Sampling from CDX Cluster Indexes
uv run repro f2 corpus sample --crawl CC-MAIN-2009-2010 --target-samples 5000
uv run repro f2 corpus sample --crawl CC-MAIN-2012 --target-samples 5000

# 3. Payload Ingestion & Feature Extraction
uv run repro f2 corpus process --limit 10000

# 4. Generate Priority-Ordered Audit Sample Sheet
uv run repro f2 corpus audit-plan --stratum-size 200 --output-file artifacts/analysis/f2/audit_set_400_sheet.jsonl

# 5. Ingest Gold Labels into Database and Parquet Manifest
uv run repro f2 corpus audit-label --audit-file artifacts/analysis/f2/audit_set_400_annotated.jsonl

# 6. Run Two-Phase Estimation, Bootstrap & Deduplication Feasibility
uv run repro f2 corpus analyze

# 7. Run Offline Calibration, Prefilter Ablation & Production Pipeline Recommendations
uv run repro f2 corpus calibrate

# 8. Full Production Corpus Materialization
uv run repro f2 corpus build --crawl CC-MAIN-2012 --target-words 1000000000 --output-dir data/f2/news_1b
```
