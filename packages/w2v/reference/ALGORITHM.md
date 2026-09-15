# Race-free C oracle

This directory contains the C behavioral oracle for a later Rust port. The
oracle is the modular implementation in this directory. `z_original_w2v.c` is
an immutable upstream snapshot and is excluded from formatting, compilation,
tests, and sanitizers.

## Source layout and public API

The layout follows the algorithm's concepts rather than treating the oracle as
a single opaque library:

- `src/corpus/` owns corpus metadata and tokenization;
- `src/vocab/` owns vocab construction, Huffman coding, and sampling;
- `src/model/` owns embeddings, atomic float operations, and sigmoid lookup;
- `src/training/` owns context traversal, objectives, workers, and orchestration;
- `src/random/` owns deterministic random-number generation.

The matching public headers live under `include/w2v/`. `w2v.h` is only a
convenient umbrella include; callers may include the narrower `corpus.h`,
`vocab.h`, `model.h`, or `trainer.h` directly. Major structures are
visible so that the oracle is easy to inspect and translate. Headers under
`internal/` describe implementation-only collaborators.

Every successful `create` or `build` call transfers one owned pointer to the
caller. Its matching `destroy` call accepts a pointer to that pointer, releases
partial or complete state, and sets it to `NULL`; calling it again is safe.

The caller must keep the corpus, vocab, and model alive while a trainer
that refers to them exists. A trainer owns its copied configuration, sigmoid
table, and optional negative-sampling table. Vocabulary entries own their token
text and Huffman paths. Embedding storage is visible on `Model`, while snapshot
calls remain the convenient race-free way to copy it into ordinary floats.

## Data flow

1. A corpus records an immutable path and byte length.
2. Vocabulary construction tokenizes the corpus, counts tokens, prunes when the
   open-addressed hash table becomes crowded, sorts by descending count and
   implementation-defined tie order, applies `min_count`, then builds Huffman paths.
3. Model construction initializes the input matrix and zeroes the output
   matrix.
4. Trainer construction builds the sigmoid lookup and, for negative sampling,
   the power-0.75 sampling table.
5. Each worker seeks directly to `byte_size / thread_count * worker_id`,
   constructs sentences while applying subsampling, periodically updates its
   local learning rate, chooses a context radius, then runs CBOW or skip-gram
   and the configured output objective. At the end of a sentence fill, an epoch
   ends if EOF was reached or its recognized-token count exceeded
   `retained_token_count / thread_count`.

Workers own their file handle, sentence buffer, hidden vectors, and RNG states.
Corpus metadata, vocab data, and trainer configuration are immutable
during training.

## Ordering and RNG contract

The ordering below is part of the single-thread oracle and must not change:

- Model initialization consumes one model-RNG value per input coordinate in
  row-major order.
- `RNG_LCG` is the default and uses the original 64-bit recurrence
  `state = state * 25214903917 + 11`. Its uniform draw uses the low 16 bits
  divided by 65536. `RNG_XORSHIFT` is an explicit alternative using the
  xorshift64* generator and a 24-bit uniform draw. The selected algorithm is
  used for model initialization and all worker RNG streams.
- Each trained target consumes one window-RNG value before visiting context.
- Context visits proceed from the farthest available left position to the
  nearest left position, then from the nearest right position to the farthest
  available right position.
- Subsampling consumes one subsampling-RNG value only for a retained,
  non-sentence-boundary vocab token when subsampling is enabled.
- Negative sampling consumes values only after the positive sample. A draw
  equal to the positive target is discarded without replacement, so fewer
  output updates may occur than the configured sample count.
- Dot products accumulate coordinates from zero upward. Hidden gradients are
  accumulated before output-row additions. CBOW averages coordinates only
  after all context rows have been accumulated.
- Each worker starts at `initial_learning_rate`. After at least
  `learning_rate_update_interval` additional recognized tokens, it reads the
  relaxed atomic processed-token count and refreshes its local rate. The
  default interval is 10,000; between refreshes the rate is unchanged.
- HS defaults to `HS_OUT_OF_RANGE_SKIP`: a Huffman node whose score is at or
  beyond either sigmoid-table limit receives no update. The explicit
  `HS_OUT_OF_RANGE_USE_BOUNDARY_VALUE` setting instead uses the lookup's zero
  or one boundary value and applies the resulting gradient. This setting does
  not affect negative sampling.

Purpose-specific seed derivation intentionally differs from the upstream
single state, including when LCG is selected. Context traversal, tie handling,
and duplicate negative behavior follow the upstream source. On the same
supported platform and compiler settings, a one-thread run must match the
checked-in golden `float` bit patterns for each selected algorithm.

## Parallel memory model

Embedding coordinates are stored as `_Atomic uint32_t` float bit patterns.
Conversion uses `memcpy`, reads use relaxed atomic loads, and additions use a
relaxed compare-exchange loop. The successful update computes exactly
`current + delta`. `processed_tokens` is also a relaxed atomic.

This removes C data races and guarantees coordinate-level atomicity. It does
not provide row-level snapshots, a global update order, or bitwise reproducible
multi-threaded results. Different legal interleavings may produce different
finite embeddings.

## Differences from the upstream snapshot

These established modular-oracle behaviors are intentional and must not be
changed to resemble `z_original_w2v.c`:

- independent RNG streams are derived from a root seed, worker, and purpose;
- hierarchical softmax and negative sampling are mutually exclusive objectives;
- a negative draw equal to the positive target is rejected without redraw;
- the modular tokenizer and pruning behavior define the accepted-input
  semantics.

The remaining learning-rate difference is intentional: upstream publishes a
worker-local word count to shared state after more than 10,000 words and then
updates a shared `alpha` without synchronization. This oracle counts recognized
tokens atomically as they arrive and refreshes a worker-local rate after at
least the configured interval. The update cadence is similar, while the unsafe
shared `alpha` race is not reproduced.

The upstream snapshot is useful only as provenance. It is not a build target or
the definition of correctness.
