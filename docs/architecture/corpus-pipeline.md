# Common Crawl Corpus Pipeline & Auditing Architecture

This document describes the Common Crawl corpus sampling, auditing, calibration, and ingestion subsystem in `studies/f2`. This pipeline serves as the **foundational Phase-0 data infrastructure** for the broader `f2` research campaign (a suite of 8~10 studies comprising 2 corpus-dependent Word2Vec training studies and 5 independent benchmark studies).

---

## 1. Subsystem Architecture & Module Map

```text
studies/f2/
├── pyproject.toml                     # Workspace dependencies (repro-core, duckdb, psycopg, typer)
└── src/f2/
    ├── plugin.py                      # Repro CLI Plugin Entrypoint (discoverable by repro_core)
    ├── cli.py                         # Root Typer CLI dispatcher
    ├── definition.py                  # Study ExecutionDefinitions
    ├── common/                        # Shared network, storage, statistics & analysis standards
    │   ├── network/                   # TokenBucketLimiter, RangeFetcher
    │   ├── storage/                   # TableExporter, CleanTextWriter
    │   ├── stats/                     # BootstrapVarianceEngine, DifferenceEstimator, ClassifierMetrics
    │   └── analysis/                  # BaseAnalysisOrchestrator, theme, declarations
    └── corpus/                        # Common Crawl Extraction Subsystem & Control Plane
        ├── cli.py                     # Typer CLI: migrate, plan, sample, audit, analyze, calibrate, build
        ├── cdx.py                     # CDX binary cluster index reader & block locator (O(log N) lookup)
        ├── discovery.py               # 2-stage Horvitz-Thompson sampler & 8-stratum audit allocator
        ├── fetcher.py                 # HTTP Range fetcher against Common Crawl ARC/WARC archives
        ├── pipeline.py                # Content extraction, language ID, validity filter, news scoring
        ├── storage.py                 # Provenance export & clean text shard writers
        ├── analysis.py                # Two-Phase Stratified Difference Estimator & Bootstrap
        ├── calibration.py             # Offline classifier calibration & pre-fetch filter ablation
            ├── migrations/
            │   ├── 001_initial_schema.sql
            │   ├── 002_add_prefetch_reject_stream.sql
            │   ├── 003_corpus_lifecycle_lineage_and_validation.sql
            │   └── runner.py
            ├── repository.py          # CorpusStateRepository (sampling, auditing, lifecycle DAG lineage & validation)
            └── session.py             # Connection pooling (F2_DATABASE_URL / F2_CORPUS_DATABASE_URL)
```

---

## 2. Core Invariants & Methodological Protocols

### 1) Two-Phase 8-Stratum Difference Estimator
The total true-news word yield $\hat{W}_{\text{true}}$ across Common Crawl crawls $c \in \{\text{2009-2010}, \text{2012}\}$ is estimated without assuming 100% classifier precision or recall:

$$\hat{W}_{\text{true}, c} = \hat{W}_{\text{proxy}, c} + \hat{E}_c = \sum_{i \in s_{1, c}} w_{1, i} y_i^{\text{proxy}} + \sum_{h \in \text{Strata}(c)} \frac{N_{c, h}^{(1)}}{n_{c, h}^{(2)}} \sum_{i \in s_{2, c, h}} w_{1, i} (y_i^{\text{gold}} - y_i^{\text{proxy}})$$

* **Phase 1 Sample ($s_1$):** Large-scale probability sample ($N=50,000$) drawn from CDX cluster index blocks using inclusion probabilities $\pi_{1, i} = \pi_{\text{block}} \times \pi_{\text{record}|\text{block}} \times \pi_{\text{fetch}}$.
* **8 Canonical Design Strata ($S_1 \dots S_8$):** Full factorial partitioning across $(\text{Crawl} \times \text{Prefilter Status [Pass / Reject]} \times \text{Predicted News [1 / 0]})$.
* **Phase 2 Audit Sample ($s_2$):** Stratified subsample ($n=400$) drawn from Phase 1, allocated across design strata with deterministic `priority_order` ranking for sequential waves ($n=200 \to 300 \to 400$).
* **Variance Engine:** $B=1,000$ Two-Stage Cluster Bootstrap (resampling CDX blocks and records within blocks) with reject exploration subsampling multipliers and residual resampling.

### 2) Pre-Fetch vs. Post-Fetch Separation
* **Stage 1: Pre-Fetch CDX Filter (`Rule 1 Only`):** Discards non-HTML binary media extensions (`.pdf`, `.jpg`, `.png`, `.mp4`, `.zip`, `.js`, `.css`) at the CDX index level *before* issuing ARC byte-range HTTP requests. Saves **44.37% of network download bandwidth** (with PDF exclusion contributing 40.12%) with 0 gold false negatives in the audit sample.
* **Stage 2: Post-Fetch Calibrated Extraction ($\tau \approx 1.25$):** Parses HTML, verifies language (English $\ge 0.50$), validates word length ($\ge 100$ words), and applies news scoring. Eliminates **44.12% of non-news text** from local NVMe/SSD storage while maintaining **98.71% out-of-fold word recall**.

### 3) 3-Tier Storage Lifecycle for Corpus Pipeline
* **PostgreSQL:** Transactional source of truth for candidate metadata, sampling weights, processing diagnostics, and gold audit annotations.
* **`.staging/exp/f2/`:** Ephemeral download shards, intermediate text extractions, and scratch audit sheets. Safe to wipe at any time.
* **`artifacts/analysis/f2/corpus/`:** Specialized corpus deliverables (`00_corpus_confirmatory_50k_report.md`, `00_corpus_confirmatory_50k_summary.csv`, `00_corpus_confirmatory_50k_filter_study.md`, `00_corpus_audit_set_50k_400_annotated.jsonl`), resolved via `RuntimePaths.from_environment().analysis_output("f2", "corpus")`.

---

## 3. Generic Corpus Lifecycle, Multi-Hop Lineage & Validation Architecture

Migration `003_corpus_lifecycle_lineage_and_validation.sql` formalizes the end-to-end lifecycle and provenance model:

$$\text{catalog release} \xrightarrow{\text{acquire}} \text{raw artifact} \xrightarrow{\text{multi-stage process}} \text{canonical shard(s)} \xrightarrow{\text{package}} \text{corpus version} \xrightarrow{\text{validate}} \text{validation evidence}$$

```mermaid
flowchart LR
    subgraph Catalog [1. Source Identity]
        CR[catalog.resources] --> CRV[catalog.resource_versions]
    end

    subgraph Acquisition [2. Ingestion]
        CRV -->|catalog_resource_version_id| AR[corpus.acquisition_runs]
        AR -->|produces| ART_RAW[corpus.artifacts\n(raw dump / tar / arc)]
    end

    subgraph Processing [3. Multi-Hop DAG]
        ART_RAW -->|input_role: raw_archive| PR1[corpus.processing_runs\nStage: extract]
        PR1 -->|output_role: extracted_text| ART_EXT[corpus.artifacts\n(extracted docs)]
        ART_EXT -->|input_role: dirty_text| PR2[corpus.processing_runs\nStage: normalize]
        PR2 -->|output_role: clean_text| ART_NORM[corpus.artifacts\n(normalized docs)]
        ART_NORM -->|input_role: uncompressed| PR3[corpus.processing_runs\nStage: shard]
        PR3 -->|output_role: canonical_shard| ART_SHARD[corpus.artifacts\n(sharded gz / zst)]
    end

    subgraph Packaging [4. Corpus Release]
        ART_SHARD -->|artifact_id| CS[corpus.corpus_shards]
        CS --> CV[corpus.corpus_versions\ne.g., news_1b:v1]
    end

    subgraph Validation [5. Provenance & Evidence]
        VP[corpus.validation_profiles\nImmutable Trigger Protected] --> VR[corpus.validation_runs]
        VR -.->|target: run| PR2
        VR -.->|target: artifact| ART_SHARD
        VR -.->|target: version| CRV
        VR --> VC[corpus.validation_checks\nMetrics, Thresholds, Diffs]
    end
```

### 1) Storage Distribution & Single Source of Truth
* **PostgreSQL (`f2`):** SSOT for metadata, content hashes (SHA-256), schema constraints, lineage DAG edges, and validation records.
* **SeaweedFS S3:** Immutable object store for actual archive binaries, intermediate text chunks, and canonical shards (`s3://...`).
* **Catalog Identity SSOT:** All generic sources (WMT News Crawl, LM1B, Gigaword, UMBC, Wikipedia, Common Crawl) are registered exclusively in `catalog.resources` and `catalog.resource_versions`. Corpus lifecycle tables reference `catalog.resource_versions(resource_version_id)` via strict foreign keys without duplicating source definitions.

### 2) Core Entities & Relational Design
* **Acquisition Layer (`corpus.acquisition_runs`):** Captures source snapshots, acquisition method (`crawler`, `dump_download`, `api`, `torrent`, `manual_archive`), parameters, target S3 prefix, status, and error logs.
* **Immutable Artifact Registry (`corpus.artifacts`):** Records all raw, intermediate, and terminal files with immutable SHA-256 digests, size, row counts, byte counts, and token counts.
* **Multi-Hop Processing Lineage (`corpus.processing_runs`, `processing_run_inputs`, `processing_run_outputs`):** Arbitrary-depth DAG support connecting processing runs to input/output artifacts with role semantics (`input_role`, `output_role`) such as `primary_text`, `vocabulary`, `filtered_output`, `reject_stream`.
* **Corpus Versioning & Sharding (`corpus.corpus_shards`, `corpus_version_stats`):** Packages sharded canonical artifacts into a frozen release version registered under `catalog.resource_versions` with aggregate document, token, and byte counters.
* **Validation Provenance (`corpus.validation_profiles`, `validation_runs`, `validation_checks`):**
  * **Target Referential Integrity:** Mutually exclusive non-null foreign keys (`target_processing_run_id`, `target_artifact_id`, `target_resource_version_id`) with CHECK constraint `ck_validation_runs_target` and generated column `target_id`.
  * **Profile Immutability:** Protected by PostgreSQL trigger `prevent_validation_profile_modification` preventing UPDATE of specification, spec_hash, and identity fields on existing revisions.
  * **Detailed Evidence:** Records individual validation rules, pass/fail status, expected/actual metrics, thresholds, and diagnostic JSON payloads.
* **Common Crawl Lineage Bridge (`corpus.pipeline_run_lineage`):** Dedicated identity PK (`lineage_id`) with partial unique indexes (`pipeline_run_id`, `stage`) allowing safe retries and bridging Common Crawl operational pipeline runs to standard lifecycle acquisition/processing runs.

### 3) Provenance Invariants & Recursive Lineage Traversal
* **Strict Provenance Retention:** All historical provenance entities (`resource_versions`, `acquisition_runs`, `processing_runs`, `artifacts`, `corpus_shards`) enforce `ON DELETE RESTRICT`. Accidental deletion of any participant in a lineage chain is rejected at the database level.
* **Multi-Hop Recursive CTE Traversal:**
  * **Reverse Lineage (`get_reverse_lineage`):** Traverses upstream through arbitrarily deep processing hops from any shard or intermediate artifact back to its originating raw artifact and catalog resource version. Includes cycle protection `a_up.artifact_id <> ALL(rd.path)`.
  * **Forward Lineage (`get_forward_lineage`):** Traverses downstream from any raw or intermediate artifact to inspect all derived processing runs, child artifacts, and corpus releases affected.

---

## 4. Word2Vec Non-Common-Crawl Corpus 3-Tier Lifecycle

To supply training inputs for historical Word2Vec reproductions (Mikolov et al., 2013), non-Common-Crawl datasets (WMT News Crawl, LM1B One Billion Word Benchmark, UMBC WebBase 2013, Wikipedia, Gigaword 5) are managed through a strictly decoupled 3-tier representation lifecycle:

$$\text{Raw Release Archive} \xrightarrow{\text{acquire}} \text{Source-Canonical Artifact} \xrightarrow{\text{normalize}} \text{Word2Vec Normalized Artifact} \xrightarrow{\text{validate}} \text{Verified Corpus}$$

```mermaid
flowchart TD
    subgraph Raw [Tier 1: Raw Artifacts]
        RAW_ARCHIVE["raw/<source>/<release>/... (tar.gz / gz / bz2)"]
    end

    subgraph Canonical [Tier 2: Source-Canonical]
        PR_CANON["processing_runs (stage: source-canonical-extraction)"]
        SHARD_CANON["processed/<source>/canonical/shard-XXXXX.txt.zst\n(10M words/shard, zstd -19, un-normalized)"]
    end

    subgraph Normalized [Tier 3: Word2Vec Public Normalized v1]
        PR_NORM["processing_runs (stage: word2vec-public-normalization-v1)"]
        SHARD_NORM["processed/<source>/normalized/shard-XXXXX.txt.zst\n(10M words/shard, zstd -19, Mikolov C normalized)"]
    end

    subgraph Storage [Storage & Lineage SSOT]
        S3[SeaweedFS S3 / Tailscale HTTPS]
        PG[(PostgreSQL f2: catalog.processing_io & corpus.corpus_version_stats)]
    end

    RAW_ARCHIVE -->|iter_tar_records / streaming| PR_CANON
    PR_CANON --> SHARD_CANON
    SHARD_CANON -->|zstd -dc pipe streaming| PR_NORM
    PR_NORM --> SHARD_NORM

    SHARD_CANON -.-> S3
    SHARD_NORM -.-> S3
    PR_CANON -.-> PG
    PR_NORM -.-> PG
```

### 1) Tier 1: Raw Release Ingestion (`raw`)
* Original vendor/upstream archives (`tar.gz`, `gz`, `bz2`) downloaded via `SerialDownloader` with byte-range resume and SHA-256 verification.
* Streamed directly to SeaweedFS S3 under `s3://f2-corpus/raw/<source>/<release>/<filename>`.
* Zero local unpack: archives are never unpacked to disk in full.

### 2) Tier 2: Source-Canonical Extraction (`canonical`)
* **Objective:** Extract pure, visible, author-released text while strictly preserving raw vocabulary, capitalization, casing, and sentence/document boundaries.
* **Extraction Rules by Source:**
  * **WMT (News 2007–2012):** Decompresses yearly `.gz` releases line by line, preserving sentence boundaries (`sentence_per_line`).
  * **LM1B (One Billion Benchmark):** Streams `training-monolingual.tokenized.shuffled/*` members in sequential order (`sentence_per_line`).
  * **UMBC (WebBase 2013):** Sequential in-memory extraction of `webbase_all/*.txt` documents from multi-part `.tar.gz` (`document_paragraph_lines`).
  * **Wikipedia (2012-12-01):** Historical streaming XML parser excluding `#redirect` pages and stripping wiki markup (`<ref>`, `[[image:...]]`, templates).
  * **Gigaword 5 (LDC2011T07):** SGML parsing of `<DOC>` and `<P>` tags with HTML unescaping.
* **Deterministic Sharding (`DeterministicSharder`):**
  * Exact 10,000,000 words per shard.
  * Compressed via `zstd -q -f -T1 -19 --no-progress` into `shard-00000.txt.zst`.
  * Physical & logical SHA-256 digests and source span metadata recorded in `manifest.json`.

### 3) Tier 3: Word2Vec Public Normalization v1 (`normalized`)
* **Objective:** Exact output equivalence to Mikolov's official Word2Vec training script ([`demo-train-big-model-v1.sh`](https://code.google.com/archive/p/word2vec/)) `normalize_text()` under the `C` locale.
* **Concrete Text Transformation Pipeline:**
  1. **Lowercasing:** `text.lower()`
  2. **Smart Quote & Apostrophe Harmonization:** `’`, `′` $\to$ `'`; `“`, `”` $\to$ `"`; `''` $\to$ `' '`
  3. **Punctuation Token Spacing:** Forces isolated tokens for `'`, `"`, `.`, `,`, `(`, `)`, `!`, `?`, `-` by padding with whitespace.
  4. **Symbol Stripping:** Replaces `;`, `:`, `=`, `*`, `|`, `«`, `<br />` with spaces.
  5. **Digit Removal (Numbers to Spaces):** Replaces all ASCII digits `0`~`9` with spaces.
  6. **Tag Exclusion:** Strips and prohibits metadata tags such as `<DOC>` or URLs. Final payload is pure training token stream.
* **4-Metric Corpus Statistics:**
  * `word_count`: Whitespace-separated lexical word count.
  * `newline_count`: Line break count.
  * `train_words_count`: Actual word tokens read by `word2vec.c` (`word_count + newline_count`, since `word2vec.c` counts `\n` as `</s>`).
  * `record_count`: Document or sentence unit count.

### 4) Storage Layout & Remote Worker Access
* **S3 Bucket Layout (`s3://f2-corpus/`):**
  * `raw/<source>/<release>/...`
  * `processed/<source>/canonical/shard-XXXXX.txt.zst`
  * `processed/<source>/normalized/shard-XXXXX.txt.zst`
  * `manifests/<source>/canonical/manifest.json`
  * `manifests/<source>/normalized/manifest.json`
* **Tailscale S3 HTTPS Gateway:**
  * Exposed over Tailnet via Tailscale Serve at `https://esillileu-server.tail4941d3.ts.net:9000`.
  * Remote worker nodes can fetch shards directly via AWS CLI, `boto3`, or `rclone` using AWS SigV4 path-style addressing (`f2_corpus_s3` credentials).

---

## 5. CLI Command Reference

### A. Common Crawl Subsystem Commands
All Common Crawl sampling, auditing, and estimation commands run via `uv run repro f2 corpus <subcommand>`:

```bash
# 1. Database Migrations
uv run repro f2 corpus migrate

# 2. Probability Sampling from CDX Cluster Indexes
uv run repro f2 corpus sample --crawls CC-MAIN-2009-2010,CC-MAIN-2012 --sample-size 50000

# 3. Create 8-Stratum Audit Assignments in Database
uv run repro f2 corpus audit --run-id <RUN_ID> --budget 400

# 4. Export Blinded Audit Review Dossier and JSONL
uv run repro f2 corpus audit-review --run-id <RUN_ID> --blind

# 5. Record Gold Annotations into Database
uv run repro f2 corpus audit-record --run-id <RUN_ID>

# 6. Run Two-Phase 8-Stratum Estimation & Feasibility Verification
uv run repro f2 corpus analyze

# 7. Run Offline Calibration, Prefilter Ablation & Filter Validation Study
uv run repro f2 corpus calibrate

# 8. Full Production Common Crawl Corpus Materialization
uv run repro f2 corpus build --crawl CC-MAIN-2012 --target-words 1000000000 --output-dir data/f2/news_1b
```

### B. Non-Common-Crawl Generic Sources Commands
Word2Vec training corpus acquisition, 2-stage processing, and validation run via `uv run repro f2 corpus sources <subcommand>`:

```bash
# 1. Inspect Inventory & Readiness Status Across All Sources
uv run repro f2 corpus sources status

# 2. Acquire Raw Release Archives (HTTP Download -> S3 Raw)
uv run repro f2 corpus sources acquire --source wmt
uv run repro f2 corpus sources acquire --source lm1b
uv run repro f2 corpus sources acquire --source umbc

# 3. Process Shards (Raw -> Canonical -> Word2Vec Normalized)
uv run repro f2 corpus sources process --source wmt
uv run repro f2 corpus sources process --source lm1b
uv run repro f2 corpus sources process --source umbc

# 4. Validate S3 Shards, Metadata & Statistics Consistency
uv run repro f2 corpus sources validate --source wmt
uv run repro f2 corpus sources validate --source lm1b
uv run repro f2 corpus sources validate --source umbc

# 5. Manual Ingest for Proprietary Datasets (Gigaword 5)
uv run repro f2 corpus sources import-gigaword /path/to/LDC2011T07.tgz
```
