# deepscratch2 MLflow metric contract

This is the canonical metric contract for the current `deepscratch2`
executors.  Legacy names remain available for existing analysis programs.

## All runs

Every run records `final/status/{success,nan_detected,inf_detected,diverged}`
and `final/system/{total_updates,completed_epochs,samples_seen}`.

When `profiling.enabled` is true, every run also retains model parameter and
optimizer-state sizes, run/epoch CPU and GPU memory snapshots, epoch train
duration and throughput, plus the configured detailed
forward/backward/clip/update timing summaries.  The raw per-epoch runtime and
memory records are written alongside the regular metric history artifacts.

The deepscratch2 catalog intentionally sets `num_steps: 0` and
`sample_memory_every_n_steps: 0`: its normal runs synchronize only at run/epoch
boundaries.  Enable either setting explicitly for a short diagnostic run when
per-update detail is required; a positive memory-sampling interval additionally
records sampled memory peaks.

## e01--e02: Word2Vec

Canonical histories are `update/train/{raw_loss,normalized_loss}` and
`epoch/train/{raw_loss,normalized_loss}`.  `raw_loss` is emitted only by the
book negative-sampling trainer; `normalized_loss` is the loss divided by its
prediction-term count.  Finals use `final/train/{raw_loss,normalized_loss}`.
The old `step/train/book_loss` and `step/train/normalized_loss` names remain
aliases.

## e03: language models

Canonical histories are `update/train/ppl`, `epoch/train/ppl`,
`epoch/valid/ppl`, and `epoch/test/ppl`; final values are
`final/train/ppl`, `final/valid/ppl`, and `final/test/ppl`.  The old
`perplexity` spellings remain aliases.

## e04--e05: seq2seq

The executor has train/test, not validation, splits.  It records
`epoch/train/loss`, `epoch/test/exact_match`, and
`epoch/test/token_accuracy`, then the corresponding `final/*` metrics.  An
attention model additionally records `final/attention/entropy`.

## Known gaps

Word2Vec does not yet emit embedding/analogy evaluation.
Language models do not emit cross-entropy history.  Seq2seq does not emit
validation/best-checkpoint/decode metrics or a complete attention summary;
attention models do retain one `analysis/attention_map.npz` artifact.  Runtime
telemetry records timing and memory, but does not yet retain a per-update
gradient-norm history.
