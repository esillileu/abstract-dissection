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
