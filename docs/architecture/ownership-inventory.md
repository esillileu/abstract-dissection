# Ownership inventory

Baseline: dev 5d3db33. Classification follows implementation semantics.
A: byte transport/parsing; B: object storage; C: source protocols;
D: research selection/representation; E: estimation; F: metadata/lineage; G: study composition/reporting.

| Original component | Class | Final owner | Reason |
| --- | --- | --- | --- |
| F2 common/network | A | repro-io.http | URL/range/bytes mechanism |
| canonical downloader, bandwidth, checksum | A/B | repro-io.http/checksum | Independent transfer mechanism |
| corpus/object_store S3 client | B | repro-io.s3 | Explicit configuration, SigV4 |
| S3 environment and restricted selection | D | F2 corpus adapter | Campaign access policy |
| corpus/cdx and fetcher | C | repro-io.commoncrawl | Common Crawl protocol, no release selection |
| pipeline ARC parser | A | repro-io.archive | Archive parsing independent of filters |
| text extraction, filters, counters | D | F2 corpus | Defines measured text |
| discovery sampling, audit, RNG | D/E | F2 corpus | Research design |
| canonical records, normalization, shards/manifests | D/F | F2 corpus | Reproduction representation |
| common/storage | D/F | F2 corpus/storage | DOC/provenance contract |
| common/stats | E | F2 common | Research calculations and small utilities |
| common/analysis | G | F2 common | Paper targets and reporting |
| common/adapters/checkpoint, paths | G | F2 common | Small composition helpers |
| sources, lifecycle, CLI | D/F/G | F2 corpus | Release, validation and orchestration policies |
| catalog DBML, both SQL/repositories | F | F2 | Research domain schema and lineage |
| DB session/migration runners | F/mechanism | F2 | Small helpers; no new DB package |
| materializer, definitions, suites | G | F2 | Run/resource policies |
| core results/mlflow_store | tracking | repro-mlflow | MLflow result interpretation |
| core analysis/core | mechanism/G | core and DLFS | Pure curves stay; selection/reporting moves |
| core analysis/model_parameters | G | DLFS | Model representation and reporting |
| package tests importing DLFS | G | DLFS tests | Study integration assertions |
| other package components | mechanism | Existing packages | Existing responsibilities preserved |
| deleted local MLflow operations | external operations | infra handoff documentation | Record interface/evidence only; live deployment remains external |
| artifacts | deliverables | artifacts | No runtime state relocation |

Known discrepancies, not methodology fixes: sharding preserves record boundaries
and therefore does not guarantee exactly ten million words per shard. `corpus build`
currently creates a directory and prints guidance, not a production pipeline.
The feasibility estimator lives in corpus/analysis.py, distinct from common/stats.
Legacy architecture checks missed F2, new packages and direct MLflow imports.
DB localhost credentials are removed; dotenv resolution order otherwise stays intact.
No datasets, checkpoints, corpus objects or database records are moved by this refactor.
