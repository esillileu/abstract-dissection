# W2V2 phrase corpus preparation

## Reproduction policy

The published [paper, section 4, Equation (6)](https://arxiv.org/pdf/1310.4546)
uses `(count(a,b) - delta) / (count(a) * count(b))`. It describes 2–4
passes with decreasing thresholds. The [original word2phrase C code](https://github.com/tmikolov/word2vec/blob/master/word2phrase.c)
multiplies that expression by `train_words`, using floating-point divisions
and a float score, approximate vocabulary reduction, fixed string buffers and a
combined word/underscore-bigram namespace. It is not a byte-for-byte oracle for F2.

F2 preserves its existing exact policy:
`(count(a,b) - min_count) * token_total / (count(a) * count(b)) > threshold`.
`token_total` is the current pass's lexical token count, excluding newlines.
Arithmetic order and Python double precision are preserved. Counts use separate
byte-token and ordered-byte-pair namespaces, with no pruning or truncation.
Pairs are adjacent within a document (one input line); joins are greedy,
non-overlapping, left to right. Output whitespace is normalized to single spaces
and one newline per document, retaining empty documents and supported non-UTF8 bytes.
Phrase detection consumes no random numbers. Training/RNG and W2V1 are unchanged.

The existing Table 3 configuration remains two passes at scalar threshold 100;
changing it would change historical inputs and run identities. For an explicitly
selected decreasing policy, configure, for example:

```yaml
phrase_detection:
  passes: 4
  thresholds: [200.0, 100.0, 50.0, 25.0]
  min_count: 5
  separator: _
```

`thresholds` must have one finite nonnegative value per pass and takes precedence
over scalar `threshold`. Scalar settings repeat unchanged on every pass. Effective
thresholds determine the cache identity; unused scalar values do not. No claim is
made that the historical scalar policy recovers the paper's unpublished thresholds.

## Token contract

W2V2 phrase corpora use `Corpus(path, full_tokens=True)`: source words and
joined phrases retain their entire byte sequence in vocabulary, checkpoints,
training and lookup artifacts. There is no 99-byte phrase limit. The independent
121-byte collision fixture and a 70 KB token pass training/restore/evaluation.
NUL and the reserved `</s>` token remain invalid phrase inputs; whitespace
(including vertical tab and form feed) and NUL separators are rejected.

The default `Corpus(path)` retains C tokenization, including 99-byte truncation
and historical EOF handling, for the required W2V1 interface and reproduction
results. Full-token mode also emits a final token without a trailing newline.
The phrase policy and vocabulary cache identities include the full-token
contract. Vocabulary caches from different tokenization modes cannot mix.
Existing short, LF-terminated inputs produce bit-identical single-threaded
training in both modes; the historical RNG and experiment IDs are unchanged.

## Disk aggregation and lifecycle

Each pass builds private SQLite WITHOUT ROWID unigram and bigram count tables,
then an indexed eligible-pair table. Input and output stream by document. SQLite's
page cache is capped at 16 MiB, temporary storage uses files, and Python retains
only fixed-size token batches and 64 KiB input chunks. Token parsing and output
writing span chunks without retaining an entire line. Batches flush at 8,192
tokens or 1 MiB of token bytes. Memory is O(fixed page/batch/chunk buffers plus
the largest individual token), independent of total unique pairs and the number
of ordinary tokens in one line. Scratch disk is O(unique
token/pair bytes plus eligible pairs and intermediate corpora). Counts are exact
64-bit SQLite integers; no token count is truncated to 32 bits. The implementation
has no corpus token-budget ceiling, so 1B–33B inputs use the same code path.
Billions of indexed updates/lookups still require substantial disk capacity and
I/O time; full-scale throughput has not been benchmarked or promised.

The executor uses `<RuntimePaths.cache_root>/f2/w2v/phrases/`:

```text
<key>.lock
<key>/phrases.txt
<key>/phrases.txt.json
.<key>-<random>/             # private builder, removed after interruption/restart
```

`key` hashes source SHA256 and the effective phrase policy. Neither seed nor
objective/subsampling condition is included. Every cache access hashes the source
and cached corpus and verifies the self-digested lineage. Configured source hashes
are checked, never trusted. A source change during construction rejects the build.
A per-key POSIX advisory lock serializes builders; process death releases the lock.
Only a completed, fsynced directory is renamed into place, followed by parent
fsync. The next builder removes abandoned work. Invalid corpus or lineage is
removed under the lock and deterministically rebuilt. Valid entries are reused
without rewriting. Cache files are reconstructible, not production run artifacts.
Use a local filesystem supporting POSIX flock and atomic same-filesystem rename.
A failed construction restarts preparation; it does not resume partial counts.

Lineage records source/result SHA256, effective policy and digest, final token and
join counts, and each pass's input/output token count, threshold, document count,
unique unigram/bigram count, joins, and output SHA256. Each run copies this manifest
to its durable result artifacts for MLflow publication. Absolute source paths are
excluded from cache identity and lineage for location-independent reproducibility.

## Table 3 training and evaluation

The existing e02 experiment/atomic IDs and six conditions remain unchanged:
NEG-5, NEG-15 and Huffman HS, each with subsampling 0 or 1e-5. Integration tests
train/evaluate all six with two seeds and assert a single shared phrase entry.
Existing W2V2 fixture tests remain in place. No full corpus processing or training
is part of verification.

Use the strict evaluation policy for a Table 3 result:

```bash
uv run repro f2 evaluate w2v2-table3 --lookup <saved-lookup-directory> \
  --questions <dataset-root>/f2/evaluation/questions-phrases.txt \
  --output <new-report.json>
```

The full resource is pinned to upstream commit
`ced016432bce8d62afbd11a112d353d13e422ea5`:
[questions-phrases.txt](https://raw.githubusercontent.com/tmikolov/word2vec/ced016432bce8d62afbd11a112d353d13e422ea5/questions-phrases.txt).
Its SHA256 is
`55cfc0894d0fe6cf974d106da3e09b5fb2fa6125b6ddc633d06e298dc5aa7901`.
The strict path requires this digest and all 3,218 parsed questions; a missing or
substituted file raises a clear error. Reports include digest, count and pinned
source URL. All questions contribute to totals; OOVs remain reported through
valid/total counts. The generic `w2v2` policy remains available for word evaluation
and explicitly partial fixture checks; those reports are not verified Table 3 data.

During Phase 2 verification the full pinned resource was downloaded to
`RuntimePaths.dataset("f2") / "evaluation/questions-phrases.txt"` and its digest
and 3,218 questions were verified. Dataset bytes are ignored by Git and must be
prepared separately on another machine; there is no automatic fixture fallback.

## Resume identity and evaluation memory

Tracked resume selection checks configuration digest, source corpus SHA256,
seed, suite and atomic variant before selecting or restoring a run. Mismatched
runs remain untouched. New tracked runs include source SHA256 in their tags.
Local execution records configuration/seed/slot and transformed corpus SHA256 in
`execution_identity.json`. Existing result identities are also checked. A missing
or invalid checkpoint never triggers deletion and restart of existing results;
the error is propagated and files are preserved. Raw epoch checkpoints continue
to be validated by the engine's full state descriptor. Historical experiment IDs
and configuration digest computation are unchanged.

Strict phrase evaluation reads the questions once, then verifies and evaluates
the same byte snapshot. Replacing the file during model loading cannot change the
evaluated questions or falsify the reported digest/count.

Evaluation scans vocabulary candidates in 4,096-row blocks. Analogy scoring
normalizes only query rows and the current block in float32 and retains the
original vocabulary tie order, input exclusion and valid-question denominator.
Qualitative neighbors use float64 blocks and a fixed-size top-k heap with the
existing token-byte tie order. Phrase reporting collects at most eight tokens.
Lookup row queries lazily build a private SQLite primary-key index in
`<cache_root>/f2/w2v/lookup-index/`. Building scans token arrays once; subsequent
queries use indexed search, including misses. SQLite uses a 16 MiB page cache,
first-occurrence semantics for duplicate byte tokens, and automatic cleanup when
the lookup object is released. Index disk is O(vocabulary token bytes); index
construction time is O(V log V), and subsequent row queries are O(log V). Analogy
indexing retains only question tokens rather than a Python dictionary for every
vocabulary row.
Candidate temporary memory depends on block size, embedding dimension and query
batch size, rather than full vocabulary size. Embedding files remain memory
mapped; this does not guarantee constant OS page residency or bound the actual
training model/vocabulary memory.

## Source corpus streaming and integrity

The source materializer also reads zstd output in 64 KiB chunks. It preserves
historical source bytes, exact lexical-token budget cuts and blank lines with
bounded RAM. Ambiguous whitespace following the last budgeted token is spooled
to disk until a newline or another token determines whether to retain it.
Tokens spanning chunks are counted once. A missing newline between two source
shards is repaired with one LF so documents cannot concatenate; canonical
LF-terminated inputs are byte-identical to the historical implementation.
The materializer identity is `f2-w2v-materialized-v2` for this boundary policy.

Cache hits verify both input-manifest identity and actual corpus SHA256, including
same-size corruption. Per-binding locks serialize builders. Downloads use private
verified files; construction uses per-key private directories on the cache
filesystem so publication remains an atomic rename even when staging and cache
roots are mounted separately. Output and manifest directories are fsynced.
Abandoned work is cleaned on the next build of the same binding. No production
service or database is required by these fixture tests.

## Independent audit results

Verdict: PASS for the audited fixture and stress-data scope. No unresolved
Critical/High/Medium findings remain from the recorded counterexamples. This is
not a full-corpus throughput, memory-capacity, service, or scientific-score claim.

- The required native API, including DownpourConfig, is present after the user
  synchronized the environment. Full-token mode preserves distinct 121-byte
  tokens as separate vocabulary entries.
- An independent exact-rational reference matched phrase output in 100 cases.
  Greedy overlap, strict threshold ties, LF boundaries, decreasing/scalar pass
  schedules and multi-pass recounting were checked.
- 650 source-prefix cases compared chunked materialization with an independent
  line-based byte oracle. Long tokens, blank lines, CR/LF, EOF, large whitespace
  runs and chunk sizes from 1 byte to 64 KiB were included.
- A 70 KB token passed phrase preparation, vocabulary, training, checkpoint
  restore and partial-fixture evaluation. Historical short LF-terminated inputs
  produced bit-identical input/output embeddings in C and full-token modes.
- Cache tests cover concurrent builders, killed builders, source mutation,
  injected ENOSPC, same-size corpus corruption and swapped source manifests.
  Resume mismatches preserve all prior result bytes and checkpoints.
- Replacing the question file after verification still evaluated the original
  3,218-question snapshot with its pinned checksum. A toy model's OOV coverage
  does not constitute reproduction of the paper's accuracy.
- With 10,000 vocabulary rows, building the private row index took 0.017 s and
  1,000 subsequent queries took 0.008 s; with 100,000 rows, 0.118 s and 0.009 s.
  The query plan uses its primary key, and a regression test makes token-array
  rescanning fail after index construction. These are small local measurements,
  not production throughput estimates.
- Actual zstd-to-materialized-file runs used peak process RSS 37.5, 39.4 and
  39.3 MB for one-line inputs of 0.32, 3.2 and 6.4 MB. This includes the Python
  process baseline; it demonstrates bounded input-buffer behavior for these
  shapes, not a bound on the trained model or an arbitrarily large single token.

Verification commands and final results:

- Root `just check`: 819 passed, 1 skipped, 8 deselected; lint and format clean.
- Package-local `cargo test`: 52 passed, 1 ignored; C tokenizer/RNG/training golden
  tests and full-token tokenizer tests passed.
- Package-local `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  and `just check`: passed. The latter runs the modular C reference/public API
  tests, excluding the immutable original snapshot.
- Package Python binding tests: 9 passed.
- Related F2 pipeline tests: 80 passed; affected W2V1 analysis/evaluation callers
  were updated to the canonical chunk/scoring interfaces and 31 related tests
  passed before the complete root gate.

No 1B/6B/33B corpus processing, large training, production-service writes, commit,
or push was performed. Full-scale disk capacity, throughput and power-loss
behavior remain outside this verification scope.
