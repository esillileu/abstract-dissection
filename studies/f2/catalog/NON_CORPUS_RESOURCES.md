# Non-corpus implementation and resource backlog

This list excludes training-corpus acquisition and corpus-size limitations. It
tracks everything else required to turn catalog specifications into complete,
evaluated reproduction runs.

## Evaluation resources

| work | affected specifications | completion condition |
|---|---|---|
| Materialize pinned `questions-words.txt` | W2V1 and W2V2 word evaluations | Download the catalog URI, verify the pinned SHA-256 and pass its immutable local path to the standalone evaluator. |
| Materialize pinned `questions-phrases.txt` | W2V2 phrase evaluations | Download, verify the pinned SHA-256 and pass its immutable local path to the standalone evaluator. |
| Resolve MSR syntactic word-relations data | W2V1 Table 3 | Record exact files, license, version and checksums; add parser and scorer wiring. |
| Resolve MSR Sentence Completion files | `w2v1-msr-sentence-skipgram` | Pin official training/test files and checksums and connect the existing sentence-completion scorer. |
| Pin published-vector baselines | W2V1 NNLM comparisons and W2V2 qualitative tables | Record licenses, immutable artifacts and checksums and implement import adapters. |

Canonical W2V YAML contains training inputs only. Evaluation resources are
resolved independently after a durable model artifact exists. The checked-in
question files remain test fixtures and must not be reported as full paper
evaluation results.

## Engine and training implementations

| work | affected specifications | required implementation |
|---|---|---|
| Feed-forward NNLM | `w2v1-table3-nnlm`, `w2v1-google-news-nnlm-6b` | Model, optimizer/training loop, complete checkpoint state and evaluation adapter; otherwise retain as external baseline only. |
| NCE objective | W2V2 Table 1 NCE row | Implement actual NCE and its checkpoint state. NEG must not be substituted for NCE. |
| DistBelief surrogate | W2V1 Table 6 | Define replica/asynchrony semantics, Adagrad state, missing learning-rate choices and reproducible hardware reporting. |
| Remaining W2V1 suites | Table 3, Table 4/5 and sentence completion | Add validated configs, execution definitions, checkpoint/evaluation scheduling and runnable plans after underspecified policies are fixed. |
| W2V2 word-only suite | `w2v2-word-skipgram-1b-objectives` | Add the word-only objective matrix and bind the full word-analogy evaluation. |

## Research-policy decisions

The papers do not fully specify epochs, learning rates, phrase thresholds,
distributed replica details and some corpus composition. Before a related plan
becomes runnable, each chosen value must be recorded as an explicit
reconstruction parameter. Missing values must not be silently inherited from an
unrelated smoke fixture or represented as exact paper settings.

## Analysis and orchestration

- Aggregate runs by explicit plan revision, planned slot, corpus version and
  MLflow run ID.
- Compare every eligible substitute corpus for the same condition and seed set.
- Report mean, confidence interval, coverage and hardware identity per corpus.
- Add paired/cross-corpus comparisons without pooling distinct corpus domains.
- Preserve reduced-corpus conditions as distinct conditions rather than treating
  them as missing observations for a larger nominal budget.
- Keep external baselines separate from locally trained reconstructions.
