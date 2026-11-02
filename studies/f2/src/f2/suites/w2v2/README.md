# W2V2 Table 1: word-only Skip-gram

`config/e01_table1_word_skipgram.yaml` implements seven conditions:
NEG-5, NEG-15 and HS-Huffman, each with no subsampling or threshold `1e-5`.
NCE-5 is a separate objective with five noise draws and no subsampling, as in Table 1.

## Paper facts and reconstruction decisions

Source: [Mikolov et al. (2013), sections 2.1–3, Table 1 and section 4.1](https://arxiv.org/html/1310.4546).
The paper specifies Skip-gram 300d, an internal Google news corpus of 1B words,
`min_count=5`, and an observed vocabulary of 692K. HS uses frequency-based
Huffman codes; NEG and NCE use unigram counts raised to 0.75. Section 4.1's “As before”
context size 5 supports the window choice indirectly; Table 1 does not restate it.
The paper does not specify epochs or the exact training schedule for Table 1.

Reconstruction decisions: 3 epochs, a dynamic radius uniformly drawn from 1..5,
SGD with initial rate 0.025 and the existing engine decay schedule, 20 CPU workers,
LCG streams, and (for NEG) a 100M-entry negative table and the existing sigmoid approximation. NCE uses the sampler and exact logistic computation described below.
No vocabulary size cap is imposed: 692K is a reported outcome, not a target to
force on substitute data. The hash capacity is 30M and can cause the existing
engine's capacity-driven vocabulary reduction on sufficiently large vocabularies.
WMT and UMBC use a 1B lexical-token budget before vocabulary filtering and
subsampling. LM1B is explicitly a reduced 791,844,834-token reconstruction.
These are substitute corpora, not the original private Google dataset.

## Explicit subsampling compatibility

Equation (5) gives discard probability `1 - sqrt(t/f)`, so the effective keep
probability is `min(1, sqrt(t/f))`. The original word2vec C source
(`packages/w2v/reference/z_original_w2v.c`, `TrainModelThread`) instead keeps with
`min(1, sqrt(t/f) + t/f)`. The current Rust worker uses that C expression and
compares its unclamped value with a uniform draw, which implements the cap.
Here `f=count/retained_token_count`; the denominator includes the retained EOS
counts, while EOS is never subsampled. Counts are measured before subsampling,
after the engine's vocabulary filtering. This is also a C compatibility choice.

The new word-only RunSpec requires `subsampling_policy: word2vec_c`, includes it
in the semantic configuration digest, and rejects other policies. It records
this intentional deviation from equation (5); it does not claim to implement a
paper-formula mode. A zero threshold disables subsampling. Existing NEG/HS arithmetic, RNG streams, checkpoint schema, W2V1 configurations and historical phrase-run identities are preserved.

## Execution and artifacts

Plan without accessing services or training:

```bash
uv run repro plan f2 w2v2 -e 01
```

There are seven conditions per corpus, with seeds 1, 7 and 19 (63 planned runs).
Table 1 has separate corpus-specific execution-plan IDs, so its slots cannot
collide with phrase Table 3 or another corpus/seed. `catalog/w2v.json` supplies
matching word-corpus and questions-words resource bindings. Applying that
manifest to an external catalog is an operator action, not performed here.
Normal tracked execution retains the existing large-run approval and service
preflight rules. Local corpus files or verified catalog shard bindings must
already be available; no data acquisition is performed by this change.

Word-only execution calls the shared trainer directly: vocabulary cache,
HS/NEG/NCE, checkpoint restoration, lookup embeddings and MLflow publishing use the
existing infrastructure. No phrase corpus or phrase-lineage artifact is created.
The exported lookup can be evaluated with `evaluate_lookup_artifact` and
questions-words; Table 1 evaluation uses the full vocabulary, with OOV coverage.


## NCE objective and gradients

For input embedding h and output embedding u_w, define s_w = u_w^T h.
NCE classifies a mixture containing one data example and k noise examples:

```
p_model(w | h) = exp(s_w) / Z(h)
z_w = s_w - log Z(h) - log(k q(w))
P(data | w, h) = sigmoid(z_w)
L = softplus(-z_positive) + sum_i softplus(z_noise_i),  i = 1..k
```

Here q is the noise probability, and all k draws are independent of the data
word; repeated noise words and collisions with the positive word are retained.
For a binary label y, dL/ds = sigmoid(z)-y, so
`dL/dh = (sigmoid(z)-y) u_w` and
`dL/du_w = (sigmoid(z)-y) h`. Contributions to the input gradient are summed;
the existing backend applies SGD using the negative of these derivatives.
NCE snapshots each distinct output row once within an objective invocation,
evaluates the positive and all k noise terms without parameter writes, and
sums their score derivatives (in f64) by row before applying one update per row.
Positive/noise collisions retain both labels. SharedModelBackend and
ReplicaBackend therefore use the same single-objective meaning; Downpour still
uses its existing cache and AdaGrad parameter server. Concurrent workers can
change parameters while snapshots are collected: this is step-local consistency,
not a global multithread snapshot. Loss observations sum the binary terms and
the reported mean divides by their count. NCE+CBOW divides the input gradient
by the context count to differentiate the context mean. HS/NEG retain the
original word2vec C sequential updates and CBOW gradient convention.
The loss uses stable f64 softplus/logistic calculations on f32 embedding scores;
NCE does not use the clipped sigmoid lookup table. The positive score gradient
is evaluated as `-sigmoid(-z)` to preserve representable tail gradients.

We fix Z(h)=1 (log Z=0), following
[Mnih and Teh (2012), section 3.1](https://arxiv.org/pdf/1206.6426).
This is an explicit self-normalization assumption, not a guarantee that
`sum_w exp(s_w)=1`, nor an exact full-softmax likelihood objective. No learned
normalizer, output bias, or additional checkpoint tensors are introduced.
Mikolov et al. section 2.2 distinguishes NCE's use of numerical noise
probabilities from NEG; Table 1 does not specify its NCE normalizer policy.
Thus this choice reconstructs an underspecified implementation detail, rather
than claiming bitwise reproduction of the unpublished NCE run.

## Noise distribution and persistence

NCE uses retained lexical counts before subsampling, excluding the internal EOS
row (which is never a positive target). Let `a_w=count(w)^0.75 / sum_v count(v)^0.75`.
The sampler stores positive integer masses `m_w=max(1, floor(2^53 a_w))` and a
cumulative mass array. Its actual distribution is `q(w)=m_w/sum_v m_v`;
this exact mass ratio supplies the NCE correction, rather than ideal a_w or the
NEG table frequencies. Quantization is a reconstruction detail. Integer
rejection sampling removes modulo bias before binary-searching the cumulative
array, assuming uniform RNG words. The existing seeded LCG/Xorshift stream is
used. There is no EOS replacement, positive-word exclusion, or resampling of
collisions. Construction uses O(vocabulary) memory and draws use O(log vocabulary)
time. `negative_table_size` is unused for NCE; `negative_sample_count` must be
positive.

`ObjectiveKind::Nce=2` extends the identity while preserving HS=0 and NEG=1.
The sampler is deterministically rebuilt from checkpoint vocabulary counts;
existing worker RNG state, embeddings, config digest and checkpoint schema
suffice for restart. The corrected NCE semantics add `nce-objective-snapshot-v1`
to both Rust checkpoint config digests and `nce_objective_revision: 1` to the
study semantic configuration (not an engine constructor argument). Pre-correction
NCE checkpoints are rejected with identity mismatch; existing finished results
are not silently reused. Retrain old NCE runs from initialization for corrected
results. This includes Downpour NCE, whose CBOW chain rule and tail arithmetic
also changed. No old/new NCE fallback is provided. HS/NEG digests and checkpoint
schemas are unchanged. Existing lookup files remain readable as historical
embeddings, but do not represent corrected training. PyO3 accepts `objective_kind="nce"`; the shared executor,
checkpoint adapter, lookup export and MLflow artifact paths need no alternate
NCE storage path. Existing historical NEG/HS config digests remain unchanged.

## Verification

`packages/w2v/tests/nce.rs` compares loss with independent density-ratio
calculations, checks real SharedModelBackend and ReplicaBackend/parameter-server updates against finite differences (including CBOW context means),
checks unigram^0.75 probabilities and the k*q correction, retains positive/noise
collisions, verifies seed reproducibility and empirical noise frequencies, and
checks that NEG and NCE differ. Rust resume tests cover CBOW/Skip-gram with both
RNG algorithms. Python tests exercise PyO3 identity, checkpoint restoration and
all seven word-only conditions, including a small NCE-5 corpus with lookup
export and a full-vocabulary analogy evaluation. These are local correctness
checks, not reproduction of the paper's reported accuracy or training times.


The audit regression uses input `[0.3,-0.7]`, five noise draws
`[1,3,3,3,3]`, positive target 3 and learning rate 0.05. The shared loss changes
from `4.934271281392` to `4.922975150870` (independent reference
`4.922975151173`); the input SGD delta changes from
`[-0.014063621,0.018108623]` to `[-0.013508944,0.017465968]`.
At positive logit 40, the score gradient changes from zero to
`-4.248354255292e-18`. Regression tests cover both shared update strategies,
repeated CBOW contexts, and parameter-server AdaGrad accumulator values.

## Section 4.1: 6B and 33B phrase Skip-gram

`e03_section4_phrase_6b.yaml` / `e04_section4_phrase_33b.yaml` select separate
experiments, plans, seed slots and logical model IDs:

- `f2.w2v2.section4.phrase-6b.hs1000.entire-sentence.r2`
- `f2.w2v2.section4.phrase-33b.hs1000.entire-sentence.r3`

These are 1000d f32 Skip-gram models with Huffman HS and entire-sentence context.
The paper's 66%/72% values are reference results, not measurements here. Three
epochs, initial LR 0.025, C subsampling at `1e-5`, two phrase passes at threshold
100/minimum count 5 and the FineWeb substitute remain reconstruction decisions.
The 6B/33B budget counts source lexical tokens before phrase merging, never
post-merge tokens or repeated epochs. Exact counts and any configured SHA256
must match. Catalog plans remain draft until real resources are bound; neither
loading the catalog nor planning creates large data or starts training.

The nominal 33B condition uses the user-approved existing FineWeb corpus of
32,991,052,431 lexical words (8,947,569 words, about 0.0271%, below the paper
scale). `paper_lexical_tokens` records 33B; `lexical_tokens` and the materializer
budget record the exact accepted input count. Preflight still requires that
exact count, and lineage records both counts and the shortfall. This is a
distinct r3 experiment/artifact identity; it does not silently reuse an exact
33B checkpoint. Additional corpus collection is unnecessary for this condition.

### Sentence semantics and reconstruction revision

FineWeb's writer joins normalized document records with LF; linguistic sentence
boundaries are not established by that writer. Before phrase construction,
`terminal-punctuation-or-lf-v1` inserts a boundary after `.`, `!`, `?` with
optional closing quotes/brackets, and treats LF as a hard boundary. This heuristic
can oversplit abbreviations and cannot recover absent boundaries. It never
joins documents. Phrase passes preserve the reconstructed boundaries.

Within a sentence, apply vocabulary filtering, independently apply the existing
C subsampling rule, then train every ordered pair of distinct retained positions
in ascending context-position order. Repeated words at different positions are
valid pairs; EOS is never a target/context. Entire-sentence mode ignores
`window_radius`, consumes no window RNG draws, and retains EOF tails without LF.
There is no 1000-token chunking. More than 65,536 source sentence tokens is an
error before OOV/subsampling. Source tokens are limited to 4096 bytes; two phrase
passes can produce tokens up to 16,387 bytes. No long-sentence truncation,
splitting, pair sampling or context cap is introduced.

Revision 2 assigns each worker a contiguous byte interval aligned forward to LF.
Adjacent intervals share exactly the same boundary; a sentence crossing an
approximate byte cut belongs wholly to the preceding worker. Empty intervals
are valid. Each epoch reads each interval once, with bounded boundary-alignment
and reader-prefetch overhead instead of every worker scanning the full file.
The Skip-gram loop visits the valid context index interval directly.

This changes worker assignment and therefore per-worker subsampling RNG streams
relative to the former modulo-sentence assignment. Configuration/plan/artifact
revision 2 and the Rust checkpoint digest record this change; old entire-sentence
checkpoints are rejected. Fixed/dynamic window traversal order and their identity
digests are preserved. Parallel Hogwild scheduling remains nondeterministic.
Revision 2 also rejects a phrase vocabulary whose exact observed distinct count
would trigger capacity-driven pruning. Increase the hash capacity explicitly;
the large-scale path does not silently change frequencies to fit a hash table.
Existing W2V1/other W2V2 vocabulary-building behavior is unchanged.

### Actual-size preflight

```bash
uv run repro plan f2 w2v2 -e 03,04
uv run repro f2 phrase-preflight studies/f2/src/f2/suites/w2v2/config/e03_section4_phrase_6b.yaml --atomic-run fineweb--phrase-6b
uv run repro f2 phrase-preflight studies/f2/src/f2/suites/w2v2/config/e04_section4_phrase_33b.yaml --atomic-run fineweb--phrase-33b
```

Without prepared statistics, preflight reports `requires_preparation`; missing
source reports `blocked`. It does not fabricate a vocabulary or model-size
estimate. To explicitly prepare an available local source without training:

```bash
uv run repro f2 phrase-preflight studies/f2/src/f2/suites/w2v2/config/e03_section4_phrase_6b.yaml --atomic-run fineweb--phrase-6b --source /absolute/prepared/6b.txt --prepare
```

Preparation can be expensive. The CLI requires `--prepare` for phrase generation;
ordinary inspection never starts it. Training executes the same preparation gate
before model allocation. No acquisition or source repetition occurs.

1. Validate engine settings, source identity/count and limits. Check configured
   initial vocabulary reservation, hash table and sigmoid table before allocation.
   Resolve the current cgroup-v2 path and all visible ancestor limits, including
   namespaced-root mounts; combine their remaining budgets with host available RAM.
2. Check preparation RAM/disk separately. Create or reuse the shared phrase corpus.
   Count exact unigrams in a bounded-cache scratch SQLite database before loading
   a Rust vocabulary. Reject insufficient hash capacity or measured build RAM.
3. Build/reuse the existing verified vocabulary cache. Record its actual word
   count, token bytes, Huffman depth, native allocation sizes and artifact bytes.
   Measure phrase corpus bytes and sentence histograms both before and after OOV
   filtering. Report p50/p95/p99/max, `sum(n*(n-1))`, and the fraction of pairs
   attributable to the longest 1% of sentences. No subsampling discount is assumed.
4. Re-estimate model, checkpoint, observation history, worker buffers, chunk IO and
   additional staging disk from those measurements. Check available resources
   again before training. Cache statistics by source, sentence/phrase/vocabulary
   policies; validate their checksum and underlying artifacts on reuse.

For actual vocabulary size V and dimension D, the two matrices cost `8*V*D`
bytes. There is no longer a universal 168GB matrix or 1.41TB RAM claim based on
hash-table occupancy. At 1000d, one million rows cost 8GB for both matrices.
The report includes actual native vocabulary/hash allocations and extra bounded
vocabulary serialization buffers; matrix snapshot copies are absent. The sigmoid
reservation is `4*sigmoid_table_size`, and oversized initial reservations are
blocked even for tiny input. Observation/history costs depend on the selected
interval and source counts. Preparation estimates remain conservative before
actual data structures exist; they are not measured RSS guarantees.

With checkpoint size C, matrix bytes M and bounded history bytes H, staging
reserves previous+new checkpoints, root history and one verification download:
`2*C + H + max(M/2, largest_vocabulary_file, H, metadata)`. Existing source/phrase/vocabulary cache bytes are
reported but are not charged again as future checkpoint allocations. Aggregate
training reads are phrase bytes times epochs, plus reported alignment/prefetch
costs. Pair/node/coordinate counts are arithmetic bounds, not FLOPs, throughput
or wall-clock predictions. The maximum accepted sentence still has
4,294,901,760 ordered pairs and may dominate training; the histogram makes that
risk visible without changing the research condition.

### Checkpoints and publication

Training writes bounded matrix chunks into atomic local checkpoint generations.
Latest and final pointers share the terminal generation. Large-model lookup reads
that final generation directly; smaller runs publish a standalone lookup.
Preprocessing and vocabulary files belong to staging, not the analysis cache.

Each tracked invocation creates a fresh MLflow run. The executor publishes its
final artifacts before marking the run FINISHED. There is no automatic local or
remote resume, checkpoint-tag registry, custom completion tag or remote generation
retention workflow. Staging is disposable after publication.

See [F2 experiment lifecycle](../../../../../../docs/architecture/f2-experiments.md)
for component ownership, removed paths and cache-only analysis.

### Execution and current operational limits

Default sources resolve through `RuntimePaths.dataset("f2") / "corpus"` as
`fineweb-6b.txt` / `fineweb-33b.txt`, honoring `REPRO_DATA_ROOT`. Use
`--set corpus.path=/absolute/prepared/6b.txt` to select an existing local file.
Verified materialized cache files may be supplied explicitly. FineWeb is the
preferred substitute; the smaller WMT/UMBC/LM1B budgets do not satisfy these
conditions. This checkout has small cached samples, not either full local source.
Remote source existence and production MLflow capacity are not inferred from
local readiness. Fresh preflight runs immediately before model allocation,
even if an earlier preparation report marked the input ready.

After preparing suitable data/resources, an operator can execute:

```bash
uv run repro run f2 w2v2 -e 03 -a fineweb--phrase-6b --seed 1 --set corpus.path=/absolute/prepared/6b.txt --tracking-uri "$F2_MLFLOW_TRACKING_URI" --approve-large-run
```

For 33B use `-e 04 -a fineweb--phrase-33b`. These commands are documentation only;
implementation verification uses small test-owned corpora, fake artifact stores
and a temporary local HTTP server. It performs no large phrase generation,
external corpus download or operational infrastructure change.

## Paper reproduction reporting (read-only)

```bash
direnv exec . uv run repro f2 w2v2-report \
  --word-questions /path/to/questions-words.txt \
  --phrase-questions /path/to/questions-phrases.txt
```

The endpoint resolves from `--tracking-uri` or `F2_MLFLOW_TRACKING_URI`. Only
the CLI downloads existing artifacts before analysis. No training,
preparation, benchmark acquisition, or MLflow publishing occurs. For a complete
missing-evidence inventory without contacting the server:

```bash
direnv exec . uv run repro f2 w2v2-report --offline
```

Outputs resolve through `RuntimePaths.analysis_output("f2", "w2v2")`, normally
`.cache/f2/w2v2/analysis/`: `summary.md`, derived `summary.json`, aggregate
`summary.csv`, seed-level `runs.csv`, `qualitative.csv`, `table1.csv` through
`table6.csv`, and `tablesection4.1.csv`. JSON is a report, not a production run
store. Table 2's validation row contains the same data identity, status and reason
as Markdown and JSON. Paper references remain distinct from measurements.

The [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf),
pages 5–8, maps Table 1 to e01, Tables 3/4 to e02, and Tables 5/6 to the explicit
e04 checkpoint. Section 4.1 compares e03/e04 against 66%/72%. No checkpoint is
selected using test accuracy. All current corpus variants are substitutes for
the original corpus; completion never asserts matching the paper's performance.

### Completion and evidence

`report_generated` means the report files were written. `experiment_complete`
requires all quantitative requirements, both validated benchmarks, and every
required qualitative query for Tables 4–6 across seeds 1, 7 and 19. Table states,
quantitative state and qualitative state are separate. External Table 6 baseline
values are references and cannot satisfy any measurement requirement.

MLflow FINISHED is the only execution-completion authority. The CLI downloads
published lookup, observations and optional provenance into local cache before
calling the analyzer. Standalone lookups do not require another full checkpoint,
execution-identity file or corpus-lineage proof. Missing required artifacts are
reported as unavailable; they are never recovered from staging or source corpora.
FAILED, KILLED and RUNNING runs retain their MLflow status in reports.

Provenance is displayed from published metadata. Raw lexical words, phrase tokens
and cumulative processed tokens remain distinct measurements. The approved 33B r3
raw count is **32,991,052,431**, separate from the paper's **33,000,000,000** target.

### Metrics, time and qualitative protocol

Total, Semantic and Syntactic retain total/valid/correct/excluded question counts,
coverage, accuracy and status. Accuracy is null when valid count is zero and is
excluded from seed statistics. Aggregation is per metric: valid accuracy survives
missing timing or an OOV-only sibling category. Means, sample standard deviations
and two-sided 95% Student-t intervals use verified independent samples only;
n<2 has no standard deviation/interval. These intervals describe seed variation.

Time requires all configured epoch endpoints, cumulative processed-token agreement,
finite monotonic elapsed observations and an inventoried `environment/system.json`
with meaningful machine, processor and platform information. Repeated identical
resume observations are deduplicated. Conflicting observations, resets without
continuity evidence, missing epochs or missing hardware leave time `unverified`.
The report links each epoch time to its stored hardware and records how many raw
observations were verified. It never substitutes the analysis machine's hardware.
Hardware differences across seeds are exposed, and original-paper hardware
comparability remains unverified. Missing time does not discard valid accuracy.

Analogy evaluation keeps the existing compute-accuracy protocol: ASCII uppercase,
normalized input vectors, b−a+c, input-row exclusion, all checkpoint vocabulary,
first-vocabulary-row tie breaking and positive-score requirement. Nearest-neighbor
queries exclude zero-norm inputs and candidates; ranking is cosine descending,
then token bytes and row ascending. OOV, cancellation and too few valid candidates
cannot complete a qualitative query.

Table 5 retains the existing **normalize each input, sum, normalize the sum**
protocol, explicitly recorded as a reconstruction decision rather than a claim
that the paper uniquely specifies normalization. Its five queries use top-4;
Table 4 uses top-1 and Table 6 top-3. These protocols are recorded in JSON and
Markdown. Training artifacts and checkpoints are never modified by reporting.
