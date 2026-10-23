# Feed-forward NNLM reproduction

The implementation is `packages/w2v/src/nnlm/`, independently of the existing
CBOW/Skip-gram model, arithmetic, checkpoint schema and training loops. Corpus,
vocabulary, Huffman paths and initialization RNG primitives are shared.
The evaluated vectors are the shared input/projection embeddings, saved through
the unchanged `f2-w2v-lookup-v1` boundary and unchanged analogy evaluator.

## Paper and reconstruction decisions

Source: [Mikolov et al., arXiv:1301.3781v3](https://arxiv.org/pdf/1301.3781),
sections 2.1/2.3/4.3/4.4, Tables 3/4/6.

| Condition | Paper specifications | Local reconstruction decisions |
|---|---|---|
| Table 3 NNLM | 320M words, 82K lexical vocabulary, D640, H640, previous N=8, Huffman HS, DistBelief | tanh, 3 epochs, 20 local replicas, 4 PS shards |
| Table 4 Our NNLM | Google News 6B, D20/50/100, Huffman HS, DistBelief | history=8 and hidden=640 carried from Table 3, tanh, 3 epochs, 14 local replicas, 4 PS shards |
| Table 6 NNLM | 100d/6B, semantic34.2/syntactic64.5/total50.8, 14 days x 180 CPU cores | Reuse Table 4's canonical 100d durable run IDs and lookup; no new training condition |

The existing catalog specs remain `partially_specified`; absent paper values stay
null. Epoch count for the author's distributed NNLM is treated explicitly as a
reconstruction decision, rather than silently inheriting the single-machine W2V
schedule. All local NNLM configurations explicitly choose AdaGrad gamma0.05,
epsilon1e-6, summed batches of250 targets, queue capacity128, fetch/push once per
batch, and no separate learning-rate decay. `initial_learning_rate=0.05` belongs
to NNLM config identity; AdaGrad's actual update scale is `adagrad_gamma`.

Embeddings initialize uniformly in +/-0.5/D; hidden weights use Xavier uniform;
bias and HS outputs initialize at zero. Tanh is configurable, including sigmoid
as another supported activation. History resets at sentence boundaries and OOV;
a target requires a full history. No subsampling is performed. These choices are
recorded in YAML comments and the existing catalog spec notes, not claimed as
paper-specific settings.

Downpour has sparse stale replica caches and dense hidden snapshots, sharded PS
updates, and per-coordinate AdaGrad for all four parameter groups. Checkpoints
are allowed only after workers join and queued updates drain at an epoch boundary.
They save weights, bias, AdaGrad accumulators, replica boundaries/counters, epoch
and token counters, vocabulary/Huffman state and corpus/config identity.
Training scans consume no RNG; no pending tokenizer/history/batch survives this
boundary. Single-replica resume is bit-identical to an uninterrupted run; asynchronous
multiple-replica scheduling is nondeterministic. This is a local DistBelief
reconstruction, not an original distributed service deployment.

## Planning and analysis

| Table | Canonical YAML | Matrix |
|---|---|---|
| 2 | e01_table2_cbow.yaml | Unchanged,216 runs |
| 3 | e02_table3.yaml | CBOW/Skip-gram/NNLM x WMT/LM1B/UMBC x seeds1/7/19:27 runs |
| 4 | e03_table4.yaml | Existing18 W2V runs plus FineWeb NNLM20/50/100d x three seeds:27 runs |
| 5 | e04_table5.yaml | Unchanged,18 runs |
| 6 | e05_table6.yaml | Unchanged6 W2V training runs; analyzer adds the existing Table4 NNLM100d runs |
| 7 | e06_table7.yaml | Unchanged3 Skip-gram runs; RNNLM remains a separate follow-up |

NNLM Table3 IDs are `<corpus>--nnlm-d640-h640-n8-w320m`, with existing corpus
plan IDs and existing `w2v1-table3-nnlm` spec. The JSON manifest materializes nine
new seed slots without duplicating that spec.
NNLM Table4 IDs are `fineweb--nnlm-d{20,50,100}-w6000m`; the existing
`w2v1-fineweb-reconstruction-r1` registration in `fineweb/register.py` materializes
the nine slots against `w2v1-google-news-nnlm-6b`. Its existing corpus version is
`f2-fineweb-2013-news-normalized-v1`. No alternate corpus policy is introduced.
Registration code is tested. Handoff preparation registered the18 NNLM seed slots
in the production catalog; existing W2V slots were not rewritten.

The existing Table6 YAML uses `w2v1-google-news-{cbow,skipgram}-scale` identities,
whereas FineWeb registration uses the dedicated Table6 spec IDs. The analyzer
selects the existing executor/YAML identities, preserving its runs. This existing
catalog/runtime mismatch is not rewritten by NNLM work.

Use the repository's command-first CLI:

```bash
uv run repro analyze f2 w2v1 --table 3 --corpus lm1b --tracking-uri <uri> \
  --questions <questions-words.txt> --msr-questions <MSR-syntax.txt>
uv run repro analyze f2 w2v1 --table 4 --corpus fineweb --tracking-uri <uri>
uv run repro analyze f2 w2v1 --table 6 --corpus fineweb --tracking-uri <uri>
```

Table3 evaluates both Google's19,544 questions and MSR's8,000 syntactic analogies.
The latter is a public ResponsiblyAI mirror pinned to commit
`715c13ff7cf19de9d42f92d2e8c4de697dd4c638`, size231,634 bytes and SHA-256
`ac90a8f492d19fd892cc0e27c5f63dcd9a520b0044b91acd99e521668cc1ebc5`.
Its section `all` is scored through the overall analogy result, not the Google
syntactic-category classifier. Original Microsoft distribution bytes are not
claimed; this resource is different from MSR Sentence Completion.

Each analyzer requires all three canonical durable seed slots for a condition,
keeps corpus results separate, and reports coverage, run IDs, mean and sample SD.
External Table4 published NNLM/RNNLM rows are references only.
Paper days x CPU cores and observed dense-observation training seconds are
separate fields. The latter can omit an epoch's tail and is not total wall time
or an equivalent historical compute bill.
NNLM checkpoints are included in durable MLflow publication; existing W2V
publication behavior and lookup/evaluation arithmetic are preserved.

## Production preparation and costs

No canonical320M/6B training starts during implementation or verification.
The new training list is18 runs if none already exists:

- Table3: three corpus variants x seeds1/7/19 =9 runs;
- Table4: FineWeb20/50/100d x seeds1/7/19 =9 runs;
- Table6 NNLM:0 new runs; use the Table4 100d runs.

Completing all three tables may also require missing existing W2V runs:
Table3:18, Table4:18, Table6:6. Production MLflow inspection confirmed existing durable W2V runs for Tables3/4/6.
The18 NNLM slots have been materialized, and DB/S3/MLflow read-only preflight
passed with the approved direnv environment. Actual NNLM training remains the
responsibility of the receiving machine; no NNLM run was created during handoff.

At the chosen3 epochs, new NNLM plans require8.64B lexical token visits for
Table3 and162B for Table4, total170.64B. Actual trained target counts are smaller
because of OOV and full-history/sentence-boundary policy.

For vocabulary size V including the sentence token, parameter float count is
`V*D + H*N*D + H + (V-1)*H`. Parameters plus AdaGrad require twice that many
float32 values:

| NNLM | V | Parameters + AdaGrad minimum GiB | Tiny single-replica targets/s | Linear single-replica equivalent days/run at all lexical tokens as targets |
|---|---:|---:|---:|---:|
| Table3 D640 |82,001|0.806|51.65|215.1|
| Table4 D20 |1,000,001|4.918|1,887.10|110.4|
| Table4 D50 |1,000,001|5.143|689.69|302.1|
| Table4 D100 |1,000,001|5.517|370.90|561.7|

The timing probe used a release build on AMD Ryzen7 8845HS/WSL2, a64-word
vocabulary,512 lexical tokens and448 targets, one replica, four PS shards and
batch250. It is a tiny fixture, not a production benchmark. These extrapolations
are conditional arithmetic, not validated wall-time forecasts: multiply by the
actual trained-target fraction and divide by measured scaling, not assumed thread
count. Large vocabularies, PS contention, memory traffic, checkpoint/S3 time and
shared CPU scheduling can change throughput substantially. A price estimate
requires the deployment's billing rate; historical14 days x180 cores is not used
as a local cost estimate. The naive extrapolation shows that this implementation
is expensive on a single workstation; do not infer production throughput from
successful tiny tests.

Memory minima exclude vocabulary/hash tables, replica caches, gradients/queues,
checkpoint state and numpy copies. Snapshot/save can hold multiple parameter
copies, so the5.517GiB figure is not a peak-RSS budget. Measure realistic peak
memory before scheduling a1M-vocabulary job. Checkpoint metadata getters avoid
full snapshots, and parameter arrays are saved sequentially to reduce copies.

The older checked-in corpus policy records5,999,250,706 verified FineWeb lexical
words. A read-only production DB inspection on2026-10-06 instead found3,325
verified shards totaling32,991,052,431 words in the same resource version, so the
6B word-count blocker is resolved. The receiving machine still needs service
preflight and realistic memory/throughput checks. Never repeat the corpus or
silently reduce the exact6B budget.
