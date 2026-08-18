# DeepScratch Canonical MLflow Migration

## Execution policy

- Execute exactly one numbered stage per work session and stop after recording its result.
- During migration, do not run `just exp run`, imports, or maintenance commands with `--apply`.
- Treat `infra/mlflow/data/` as the source of record. Do not modify its database, artifacts, runs, tags, or lifecycle state.
- Preserve pre-existing user changes; do not reset or overwrite them.

## Progress

| Stage | Status | Evidence |
|---|---|---|
| 0. Baseline and source freeze | **COMPLETE** | Eligibility decision recorded below; high-water digest and validation evidence are fixed. |
| 1. Scanner and canonical transform contract | **COMPLETE** | Full deterministic scan and byte-identical replay completed; 543 conflicts were handed to stage 2. |
| 2. Conflict decisions | **COMPLETE** | Adapter precedence resolves all metric conflicts; corrected canonical target is 1,772 runs. |
| 3. Fresh target migration | **COMPLETE** | Fresh target built and verified: 1,772 runs, 62,547,086 metrics, 57,861 artifact files; WAL cleaned up. |
| 4. Target exhaustive verification | **COMPLETE** | 1,772 runs, 62,547,086 metric rows, and 57,861 artifact files verified; 0 mismatches. |
| 5. Canonical-only operational cutover | **COMPLETE** | Operational selector/status/analysis now read canonical namespaces only; 91 regression tests passed. |
| 6. DB cutover and rollback check | **COMPLETE** | MLflow now serves `infra/mlflow/data` from the verified target; source preserved at `infra/mlflow/archive/source-20260818`; health and 1,772-run API checks passed. |
| 7. Legacy and migration code removal | **COMPLETE** | Canonical resolver cutover, legacy package/CLI removal, and regression cleanup completed. |
| 8. Final regression and archive seal | **COMPLETE** | Final API/storage checks, full regression, and archive preservation seal completed. |

## Stage 0 — Baseline and source freeze

### Execution record

- Started: 2026-08-18T12:59:00+09:00
- Ended: 2026-08-18T13:06:49+09:00
- Result: **COMPLETE** after explicit user decision on 2026-08-18.
- Git revision: `e1ad46ba3efe368d28934c2e6e5f5f7f830c89d9`
- Rollback: fully possible. This stage changed only `.gitignore` and this plan; the source DB and artifact tree were read only.
- Operational freeze: active for the remainder of the migration. `just exp run`, import commands, and maintenance commands with `--apply` are prohibited.

### Changed files

- `.gitignore`: exclude `infra/mlflow/data-next/`, `infra/mlflow/archive/`, and `infra/mlflow/migration-work/`.
- `PLAN.md`: record the baseline and blocking source-classification conflict.

### Preserved pre-existing changes

The pre-stage binary diff digest was `af0027d40842479dfdef51c20c86a5f3dadf84b9fb1eb6775991e3b8cba9498d`. The four modified files and their SHA-256 digests were:

| File | SHA-256 |
|---|---|
| `exp/deepscratch/ds2/analysis/e10_word2vec_profile.py` | `d4e1b705f7ddd1e8c6fb2bd01aa3ebbf3996d3e6980d73a6d731ca42899dcfbb` |
| `exp/deepscratch/ds2/analysis/e11_vocabulary_size_scaling.py` | `5bc7c0b3d4c487d392637490e44316b06ee8a5bbce795c43c9ddfb924d8490ea` |
| `exp/deepscratch/ds2/tests/test_e10_profile_study.py` | `8eed228ab0f6826a7f552ed6e55e57a58c72116ed0fc5ed58795a5796c62a357` |
| `exp/deepscratch/ds2/tests/test_e11_vocabulary_size_scaling.py` | `761ffd0c9ffbece0bbcf255ad3a3f510118b4b526e888c41166d7a54328d8bac` |

### MLflow and storage baseline

- Health: `GET http://127.0.0.1:5000/health` returned `OK`.
- Container: `mlflow-mlflow-1`, running, image `ghcr.io/mlflow/mlflow:v3.14.0`, image digest `sha256:12a2ac214c015752bc587c09c4e6df8fd540e072c7e6a67906212e7dde8cc11e`.
- Source root size: 47,884,415,200 bytes.
- SQLite DB size: 19,261,509,632 bytes.
- Artifact tree size: 28,618,632,601 bytes.
- Filesystem capacity: 1,081,101,176,832 bytes; 921,422,495,744 bytes available (11% used).
- Active `RUNNING` or `SCHEDULED` runs across the six source namespaces: 0.

### High-water mark

The high-water input is every run in `ds1`, `ds2`, `ds1_original`, `ds2_original`, `deepscratch.ds1`, and `deepscratch.ds2`, including deleted runs. Each compact JSON row contains experiment name, run ID, status, lifecycle stage, start time, and end time, is terminated by LF, and rows are sorted by experiment name and run ID.

- Rows: 2,051
- SHA-256: `4db9c1e0189080c27c2430bfc5dd23f126c02d734fb7b86c1315aa8a2aae21c0`
- Stage-0 standalone report: not created; this section is the stage record.
- Migration manifest: not applicable until stage 1.

### Source counts

| Namespace | Total |
|---|---:|
| `ds1` | 594 |
| `ds2` | 521 |
| `ds1_original` | 529 |
| `ds2_original` | 124 |
| `deepscratch.ds1` | 25 |
| `deepscratch.ds2` | 258 |
| **Total** | **2,051** |

Across all six namespaces, `run.type` reports 231 `condition_parent` runs and 1,820 data runs (1,797 `seed_trial`, 23 `profile`). Status/lifecycle counts are:

| Status | Lifecycle | Count |
|---|---|---:|
| `FINISHED` | active | 1,984 |
| `FINISHED` | deleted | 52 |
| `FAILED` | active | 4 |
| `FAILED` | deleted | 10 |
| `KILLED` | deleted | 1 |

### Resolved classification conflict

The approved baseline labels all 231 `condition_parent` runs as excluded parents while also labeling all 52 deleted `FINISHED` runs as data. The source DB shows that these sets overlap:

- active `FINISHED` condition parents: 212
- deleted `FINISHED` condition parents: 19
- active `FINISHED` non-parent runs: 1,772
- deleted `FINISHED` non-parent runs: 33
- `FAILED`/`KILLED` non-parent runs: 15

Therefore the stated target count 1,753 cannot be derived by simultaneously applying the canonical eligibility rule “active `FINISHED` data run” and the source `run.type` classification: that rule yields 1,772. The value 1,753 results only by subtracting all 52 deleted `FINISHED` runs—including 19 parent runs—from the already parent-excluded count of 1,820, then subtracting the 15 failed/killed data runs.

The initial stage-0 decision retained 1,753 pending an explicit classification. Stage 2 later corrected that decision: exclusion sets are disjoint, the 19 deleted `condition_parent` runs are excluded once as parents, and the canonical target is 1,772. This does not modify source data.

### Validation commands

| Command | Result |
|---|---|
| `curl -fsS http://127.0.0.1:5000/health` | PASS (`OK`) |
| read-only SQLite aggregation over all six namespaces | PASS after explicit eligibility decision; no active writers |
| `uv run pytest tests/tracking exp/tests/test_deepscratch_architecture.py -q` | PASS (91 passed in 84.76s) |
| `uv run ruff check .` | PASS |
| `git diff --check` | PASS |

### User decisions

- Superseded in stage 2: the initial 1,753 target double-counted 19 deleted parents. The corrected target is 1,772.

## Stage 1 — Scanner and canonical transform contract

### Execution record

- Started: 2026-08-18T16:00:00+09:00
- Ended: 2026-08-18T16:12:00+09:00
- Result: **COMPLETE** after deterministic outputs were produced and conflicts were handed to stage 2.
- Changed files: `infra/mlflow/migration/scan.py`, `tests/migration/test_scan.py`.
- Rollback: fully possible; source DB/artifacts were opened read-only and no target was created.

### Implemented contract

- Scan-only CLI: `uv run python infra/mlflow/migration/scan.py --work-dir <path>`.
- Reads six source namespaces and writes only the requested work directory.
- Maps source aliases through DS1/DS2 declarations, defaults missing protocol to `legacy`, rejects undeclared protocols, detects metric alias history conflicts, and flags incomplete `raw/metrics.csv` shapes.
- Streams artifact SHA-256 calculation rather than loading files into memory.
- Intended outputs: `scan_summary.json`, `migration_manifest.jsonl`, `conflict_report.json`, and `source_inventory.json`.

### Validation

- `uv run pytest tests/migration/test_scan.py -q`: PASS (5 passed).
- `uv run ruff check infra/mlflow/migration/scan.py tests/migration/test_scan.py`: PASS.
- The real source scan was started against 2,051 runs and 1,805 artifact roots (28,618,632,601 bytes). It was stopped after more than ten minutes while materializing one run's complete indexed metric history in `_metric_records`; no output file was accepted as a manifest.
- The scanner was then changed to stream each run's metric rows into per-source-key count/digest state (no full metric-history list). The streaming implementation and its five tests pass, but the real scan still exceeded the execution window while iterating a very large indexed metric history. No semantic conflict decision was made from this interruption.

### Reports and digests

- `infra/mlflow/migration-work/stage-1-20260818*`: temporary ignored directories; no complete report or manifest exists.
- Source high-water digest remains `4db9c1e0189080c27c2430bfc5dd23f126c02d734fb7b86c1315aa8a2aae21c0`.

### Final streaming retry

- Ended: 2026-08-18T18:23:12+09:00.
- Metric histories are consumed from indexed SQLite cursors and reduced to source-key row count and SHA-256 state; complete histories are never materialized in memory.
- Artifact files are hashed with a bounded four-worker pool, while final inventory order remains path-sorted and deterministic.
- Verified artifact and metric caches may be supplied to reproduce manifest generation without re-reading the 28.6 GB payload. A second cached scan was byte-identical for all four outputs.
- Source runs: 2,051; candidate active `FINISHED` non-parent runs: 1,772; migrate: 1,229; quarantine: 543; exclude: 279; target runs: 0.
- High-water SHA-256: `4db9c1e0189080c27c2430bfc5dd23f126c02d734fb7b86c1315aa8a2aae21c0` (matches stage 0).
- Active unmapped conditions: 0.
- The three deleted `LM-SMALL-RNN-CUSTOM` data runs appear only as `exclude/deleted_finished`; their three deleted parents appear only as `exclude/condition_parent`.
- Final validation: `uv run pytest tests/migration/test_scan.py tests/tracking exp/tests/test_deepscratch_architecture.py -q` passed (96 tests); `uv run ruff check .` and `git diff --check` passed.

Final outputs:

| Path | SHA-256 |
|---|---|
| `infra/mlflow/migration-work/stage-1-20260818-final/scan_summary.json` | `0a8110ea6b17a6ac65b63a9bfda739c6d33f0bed6c968302dc3996be477f361c` |
| `infra/mlflow/migration-work/stage-1-20260818-final/migration_manifest.jsonl` | `f8213ee638d2800c0f05a6a106a222b8cd0a8a7dcba11763df823b94309bde88` |
| `infra/mlflow/migration-work/stage-1-20260818-final/conflict_report.json` | `d90f988bfd2cdf20c76b0ea47e6412473f9caf15945f2adb6344abb08b5af9cc` |
| `infra/mlflow/migration-work/stage-1-20260818-final/source_inventory.json` | `3d855e62b88910311551d54193d8388e815a0d1bd7cb1df72fc86896779497b1` |

The 543 unresolved conflicts are all `METRIC_ALIAS_HISTORY_CONFLICT`:

| Source keys → canonical key | Runs |
|---|---:|
| `final/test/accuracy`, `test/accuracy` → `final/test/accuracy` | 382 |
| `final/test/accuracy`, `test/accuracy` → `final/test/exact_match` | 71 |
| `final/train/loss`, `train/loss` → `final/train/loss` | 20 |
| `final/train/loss`, `train/loss` → `final/train/book_loss` | 20 |
| `final/train/book_loss`, `series/train/book_loss`, `update/train/book_loss` → `final/train/book_loss` | 20 |
| `final/test/perplexity`, `test/perplexity` → `final/test/perplexity` | 20 |
| `final/train/perplexity`, `train/perplexity` → `final/train/perplexity` | 10 |

Per the migration contract, these histories were not selected by recency or similarity during stage 1. They were handed to stage 2 for explicit source-run decisions.

## Stage 2 — Conflict decisions

### Execution record

- Decision: match the retired adapter's first-available-native-ID precedence.
- Decision file: `infra/mlflow/migration-work/stage-1-20260818-final/migration_decisions.json`.
- Decision file SHA-256: `bea9450afbc09eae86247499ceffe1347a63f2e15c74d680c839004e508a08e9`.
- Decisions: 543 metric mappings across 543 source run IDs.
- Later aliases are excluded from metric migration; source curves remain preserved through their declared artifact projections.
- All 543 `METRIC_ALIAS_HISTORY_CONFLICT` entries are resolved; the final conflict report contains zero entries.
- Result: **COMPLETE** after the user approved correcting the canonical target to 1,772.
- Rollback: fully possible; only ignored migration-work reports and repository scanner/resolver code changed. Source and target remain untouched.

### Final manifest counts

| Action/reason | Count |
|---|---:|
| migrate | 1,772 |
| exclude: condition parent | 231 |
| exclude: deleted `FINISHED` data | 33 |
| exclude: deleted `FAILED` data | 10 |
| exclude: active `FAILED` data | 4 |
| exclude: deleted `KILLED` data | 1 |
| quarantine | 0 |
| unresolved conflicts | 0 |

The exclusion sets are disjoint in the final manifest and total 279, so `2,051 - 279 = 1,772`. The former 1,753 target double-counted the 19 deleted condition parents. The user approved 1,772 as the corrected canonical target; no active run is excluded without a source-state criterion.

### Reports and digests

| Path | SHA-256 |
|---|---|
| `infra/mlflow/migration-work/stage-2-20260818-final/scan_summary.json` | `5f6571fb54f658fad287e10f29b079b3a2db3b7b16ab2c62cc727cfac8449dc9` |
| `infra/mlflow/migration-work/stage-2-20260818-final/migration_manifest.jsonl` | `053ba584cdfca2f36f05870fa1dc26d7e38c9af471e47ad31874b9a01f489ec5` |
| `infra/mlflow/migration-work/stage-2-20260818-final/conflict_report.json` | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| `infra/mlflow/migration-work/stage-2-20260818-final/source_inventory.json` | `3d855e62b88910311551d54193d8388e815a0d1bd7cb1df72fc86896779497b1` |

## Stage 3 — Fresh target migration

### Execution record

- Started: 2026-08-18T18:43:00+09:00.
- Ended: 2026-08-18T20:27:00+09:00.
- Result: **COMPLETE**. A fresh target was built under `infra/mlflow/data-next/` from the stage-2 manifest.
- Source DB and source artifact tree remained read-only. The existing MLflow server and source namespace were not changed.

### Target result

- Migrated runs: 1,772.
- Target experiments: `deepscratch.ds1` (1,022 runs), `deepscratch.ds2` (750 runs), plus MLflow `Default`.
- Metrics: 62,547,086 rows and 36,512 latest-metric rows.
- Artifacts: 57,861 files, 27,335,996,187 bytes.
- Parent relationship tags: 0; excluded condition parents do not leave dangling parent references.
- Final target size: 44G under `infra/mlflow/data-next/`; the transient SQLite WAL was removed on close.
- Report: `infra/mlflow/data-next/migration_report.json`.

### Implementation and validation

- Added `infra/mlflow/migration/migrate_fresh_target.py`.
- The migrator copies the frozen source DB, keeps only manifest-approved runs, moves them into canonical experiments, applies selected metric aliases, rebuilds latest metrics, removes dangling parent tags, and copies only approved artifact roots.
- `uv run ruff check infra/mlflow/migration/migrate_fresh_target.py`: PASS.
- Final read-only target checks: PASS; target counts match the manifest and the WAL is absent.
- The failed intermediate 15G DB copy was isolated under `/tmp/deepscratch-stage3-incomplete-20260818/` and removed after completion; it was never used as a migration source.

## Stage 4 — Target exhaustive verification

### Execution record

- Started: 2026-08-18T20:35:00+09:00.
- Ended: 2026-08-18T21:13:00+09:00.
- Result: **COMPLETE**. The fresh target was exhaustively compared with the frozen stage-2 manifest and read-only source DB/artifact inventory.
- Source and target data remained read-only during verification; only the ignored verification report and verifier code were written.

### Verification result

- Migrated runs checked: 1,772; target run ID set and canonical experiment placement matched exactly.
- Tags and params checked for every migrated run; excluded parent relationship tags remained absent.
- Metric rows checked: 62,547,086. Source-selected metric histories were transformed to canonical keys and compared by row count and SHA-256 digest for every run; mismatch count: 0.
- Artifact files checked: 57,861. Every target file size and SHA-256 matched the stage-2 manifest inventory; mismatch count: 0.
- Overall status: **PASS**.

### Reports and validation

- Verification report: `infra/mlflow/migration-work/stage-4-20260818-final/target_verification.json`.
- Report SHA-256: `5b65e5d50752b1e1c2041cf28a7c8ff7a5f530f3ff9632c57b8837cd0cae3b55`.
- Added `infra/mlflow/migration/verify_target.py`.
- `uv run ruff check infra/mlflow/migration/verify_target.py`: PASS.
- `git diff --check`: PASS.

## Stage 8 — Final regression and archive seal

### Execution record

- Started: 2026-08-18.
- Ended: 2026-08-18.
- Result: **COMPLETE**. The DeepScratch canonical MLflow migration is sealed.
- No source or archive database/artifact files were modified during this stage.

### Final seal

- MLflow Compose down/up cycle: PASS.
- `GET http://127.0.0.1:5000/health`: `OK`.
- Active experiments: exactly `Default`, `deepscratch.ds1`, and `deepscratch.ds2`.
- Active canonical runs: `deepscratch.ds1` 1,022 + `deepscratch.ds2` 750 = **1,772**.
- Canonical target report: 62,547,086 metric rows, 36,512 latest-metric rows, and 57,861 artifact files.
- Target migration report SHA-256: `cac38612e596d3bc702360dadfa17f0565239f6f0905e8bfaf9570d190be5899`.
- Archived source: `infra/mlflow/archive/source-20260818/mlflow.db`, 19,261,509,632 bytes, **2,051** runs across the six frozen source namespaces.
- Active target DB: `infra/mlflow/data/mlflow.db`, 19,423,531,008 bytes. The observed WAL sidecar is empty (0 bytes); no pending WAL data exists.
- Rollback source remains present and the reversible cutover procedure is unchanged.

### Final validation

- `uv run pytest tests exp/tests -q`: **369 passed**.
- `uv run ruff check .`: PASS.
- `git diff --check`: PASS.

## Stage 6 — DB cutover and rollback check

### Execution record

- Started: 2026-08-18T21:19:00+09:00.
- Ended: 2026-08-18T21:21:31+09:00.
- Result: **COMPLETE**. The MLflow Compose bind mount now points at the verified canonical target through the normal `infra/mlflow/data` path.
- The former source store was moved reversibly to `infra/mlflow/archive/source-20260818/`; no source files were deleted or modified.

### Cutover validation

- `docker compose -f infra/mlflow/compose.yaml down` then `up -d`: PASS.
- `GET http://127.0.0.1:5000/health`: PASS (`OK`).
- Active MLflow experiments: exactly `Default`, `deepscratch.ds1`, and `deepscratch.ds2`.
- Active canonical runs: `deepscratch.ds1` 1,022 + `deepscratch.ds2` 750 = **1,772**.
- MLflow API artifact listing: PASS on canonical runs.
- Active target WAL: absent after startup/checks.

### Post-cutover audit and correction

- The first operational audit exposed a migration bug: legacy-origin rows had been copied, but many lacked the canonical identity tags required by `CanonicalAttemptSelector`, so `just exp check` reported all DS1 entries missing.
- Repaired the active target only from the frozen Stage 2 manifest, writing 10,632 canonical tags across 1,772 runs; the archived source database was not changed.
- Added the same canonical-tag materialization to `migrate_fresh_target.py` and made `verify_target.py` validate runtime condition aliases.
- `just exp check deepscratch ds1`: **516/516 completed, 0 missing**.
- `just exp check deepscratch ds2`: **332/332 completed, 0 missing**.
- `GET http://127.0.0.1:5000/health`: PASS (`OK`) after repair.

### Rollback check

- Archived source DB remains present at `infra/mlflow/archive/source-20260818/mlflow.db` with its original 19,261,509,632-byte size.
- Read-only archive inspection still reports all 2,051 source runs across the six frozen namespaces.
- Active target remains separately present at `infra/mlflow/data/` with the migration report and 1,772 runs.
- Rollback procedure is reversible: stop Compose, rename active `data` back to a target hold path, rename `archive/source-20260818` back to `data`, then start Compose. No rollback was needed because the target cutover passed.

## Stage 5 — Canonical-only operational cutover

### Execution record

- Started: 2026-08-18T21:13:00+09:00.
- Ended: 2026-08-18T21:18:00+09:00.
- Result: **COMPLETE**. Operational DeepScratch selection now uses only `deepscratch.ds1` and `deepscratch.ds2`.
- No MLflow DB, artifact tree, server process, or compose volume was changed. Those stateful cutover actions remain in Stage 6.

### Changed behavior

- `CanonicalAttemptSelector` no longer merges attempts from retired `ds1`, `ds2`, `ds1_original`, or `ds2_original` namespaces.
- Operational result loading rejects a non-canonical attempt instead of routing through the retired result adapter.
- Plan status no longer performs a second historical/profile search; canonical selector results are the sole operational inventory.
- User documentation now identifies historical namespaces as archive-only and canonical namespaces as the operational source.
- Legacy importer, archive gateway, and storage-audit code remain available behind the historical compatibility boundary.

### Validation

- Targeted canonical-only selector/status/cross-variant tests: **10 passed**.
- `uv run pytest tests/tracking exp/tests/test_deepscratch_architecture.py -q`: **91 passed**.
- `uv run ruff check .`: PASS.
- `git diff --check`: PASS.

## Stage 7 — Legacy and migration code removal

### Execution record

- Started: 2026-08-18.
- Ended: 2026-08-18.
- Result: **COMPLETE**. This session performed only stage 7; stage 8 remains pending.
- MLflow data, artifacts, and the archived source at `infra/mlflow/archive/source-20260818/` were not modified.

### Removed boundaries

- Deleted the completed one-shot migration scanner, resolver, fresh-target migrator, canonicalizer, verifier, and their migration tests.
- Removed the DeepScratch legacy importer, archive gateway, result adapter, namespace adapter, checkpoint fallback, cutover audit, and storage-audit package.
- Removed the `exp import-legacy` and DeepScratch storage cleanup registrations and documentation.
- DS1/DS2 catalogs now use `mlprosection_mlflow.checkpoint_source` directly; canonical operations no longer depend on the retired compatibility package.
- Removed tests whose only contract was historical import/projection/storage compatibility and updated architecture checks for the canonical-only package tree.

### Validation

- `uv run pytest tests exp/tests -q`: **369 passed**.
- Focused architecture rerun after removing stale legacy bytecode: **20 passed**.
- `uv run ruff check exp/deepscratch tests/tracking exp/tests infra/mlflow`: PASS.
- `git diff --check`: PASS.
