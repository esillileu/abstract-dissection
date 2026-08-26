# F2 Word2Vec (2013) Paper Reproduction & Corpus Feasibility Study

This study package implements the Common Crawl (2009–2012) corpus acquisition, feasibility study, and word-embedding reproduction pipeline for Mikolov et al. (2013, *Distributed Representations of Words and Phrases and their Compositionality*).

---

## 1. Study Overview

* **Primary Objective**: Determine whether Common Crawl snapshots from 2009–2012 contain enough usable English news/article text to build reproducible training corpora up to 1B, 6B, and 33B words.
* **Architecture**:
  * `cdx.py`: Binary search on static Common Crawl `cluster.idx` indices and SURT mapping.
  * `discovery.py`: Two-stage crawl-stratified probability sampling (`CC-MAIN-2009-2010`, `CC-MAIN-2012`) and curated seed catalog.
  * `fetcher.py`: Bounded HTTP Range GET fetcher with Token-Bucket rate limiting (20 Mbps) and concurrency controls.
  * `pipeline.py`: ARC parser (`warcio`), Trafilatura text extraction, explicit rule-based news classifier, language filter, and word counter.
  * `storage.py`: Provenance logging (`provenance.jsonl`), Write-Ahead Log checkpointing, and clean text shard writer.
  * `analysis.py`: DuckDB analytical engine, Two-Stage Horvitz-Thompson estimator, two-phase residual error correction, bootstrap variance replication, and deduplication sensitivity scenarios.

---

## 2. CLI Usage (`repro f2`)

```bash
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
    --output-dir artifacts/analysis/f2/feasibility
```
