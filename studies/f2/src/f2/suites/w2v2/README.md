# W2V2 Table 1: word-only Skip-gram

`config/e01_table1_word_skipgram.yaml` implements the six NEG/HS conditions:
NEG-5, NEG-15 and HS-Huffman, each with no subsampling or threshold `1e-5`.
NCE-5 is unsupported until Phase 3; it is never mapped to NEG-5.

## Paper facts and reconstruction decisions

Source: [Mikolov et al. (2013), sections 2.1–3, Table 1 and section 4.1](https://arxiv.org/html/1310.4546).
The paper specifies Skip-gram 300d, an internal Google news corpus of 1B words,
`min_count=5`, and an observed vocabulary of 692K. HS uses frequency-based
Huffman codes; NEG uses unigram counts raised to 0.75. Section 4.1's “As before”
context size 5 supports the window choice indirectly; Table 1 does not restate it.
The paper does not specify epochs or the exact training schedule for Table 1.

Reconstruction decisions: 3 epochs, a dynamic radius uniformly drawn from 1..5,
SGD with initial rate 0.025 and the existing engine decay schedule, 20 CPU workers,
LCG streams, 100M-entry negative table, and the existing sigmoid approximation.
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
paper-formula mode. A zero threshold disables subsampling. No engine, RNG,
checkpoint format, W2V1 configuration or historical phrase-run identity changes.

## Execution and artifacts

Plan without accessing services or training:

```bash
uv run repro plan f2 w2v2 -e 01
```

There are six conditions per corpus, with seeds 1, 7 and 19 (54 planned runs).
Table 1 has separate corpus-specific execution-plan IDs, so its slots cannot
collide with phrase Table 3 or another corpus/seed. `catalog/w2v.json` supplies
matching word-corpus and questions-words resource bindings. Applying that
manifest to an external catalog is an operator action, not performed here.
Normal tracked execution retains the existing large-run approval and service
preflight rules. Local corpus files or verified catalog shard bindings must
already be available; no data acquisition is performed by this change.

Word-only execution calls the shared trainer directly: vocabulary cache,
HS/NEG, checkpoint restoration, lookup embeddings and MLflow publishing use the
existing infrastructure. No phrase corpus or phrase-lineage artifact is created.
The exported lookup can be evaluated with `evaluate_lookup_artifact` and
questions-words; Table 1 evaluation uses the full vocabulary, with OOV coverage.
