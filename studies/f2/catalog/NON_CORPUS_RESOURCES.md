# Non-corpus implementation and resource backlog

This list excludes training-corpus acquisition and corpus-size limitations. It
tracks everything else required to turn catalog specifications into complete,
evaluated reproduction runs.

## Evaluation resources

| work | affected specifications | completion condition |
|---|---|---|
| Materialize pinned `questions-words.txt` | W2V1 and W2V2 word evaluations | Download the catalog URI, verify the pinned SHA-256 and pass its immutable local path to the standalone evaluator. |
| Materialize pinned `questions-phrases.txt` | W2V2 phrase evaluations | Download, verify the pinned SHA-256 and pass its immutable local path to the standalone evaluator. |
| Materialize MSR syntactic word-relations mirror | W2V1 Table 3 | `table3.ensure_msr_syntactic` verifies the 8,000-question ResponsiblyAI mirror at commit `715c13ff`; catalog records SHA-256, size and origin. Original Microsoft distribution bytes are not claimed. |
| Resolve MSR Sentence Completion files | `w2v1-msr-sentence-skipgram` | Pin official training/test files and checksums and connect the existing sentence-completion scorer. |
| Pin published-vector baselines | W2V1 NNLM comparisons and W2V2 qualitative tables | Record licenses, immutable artifacts and checksums and implement import adapters. |

Canonical W2V YAML contains training inputs only. Evaluation resources are
resolved independently after a durable model artifact exists. The checked-in
question files remain test fixtures and must not be reported as full paper
evaluation results.

## Engine and training implementations

| work | affected specifications | required implementation |
|---|---|---|
| Feed-forward NNLM | `w2v1-table3-nnlm`, `w2v1-google-news-nnlm-6b` | Implemented as an independent Rust subsystem with PS/AdaGrad, epoch-boundary checkpoint/resume and shared lookup evaluation. Table 3/4/6 wiring is complete; see [NNLM reproduction](NNLM_REPRODUCTION.md). |
| NCE objective | W2V2 Table 1 NCE row | Implement actual NCE and its checkpoint state. NEG must not be substituted for NCE. |
| DistBelief surrogate | W2V1 Table 3/4/6 | Local asynchronous PS/AdaGrad exists for W2V and NNLM. Paper days x CPU cores remain separate from observed runtime; multi-replica scheduling is nondeterministic. |
| RNNLM | Table 3/7 reference rows | Separate follow-up; no RNNLM training is added by the NNLM implementation. |
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
