# w2v CPU profiling harness

This directory compares the unmodified upstream C program, the modular C
reference, and the Rust port. It is intentionally absent from the package's
normal Cargo, Python, C test, and sanitizer builds.

Run `just profile-smoke` from `packages/w2v` to exercise all three executables,
all four model/objective combinations, and 1/2 threads once. Run `just profile`
for the 180-sample matrix plus 24 call-stack profiles. The full corpus contains
2,000,000 tokens; increase it only explicitly, for example with
`just profile --scale 2`.

The commands require `cc`, Cargo, `taskset`, and a working Linux `perf` with the
listed hardware/software counters and call-stack access. They never install
dependencies or alter system settings. Generated corpora, binaries, raw
`perf.data`, reports, logs, and summaries are ignored by Git under `data/`,
`build/`, and `results/`.

`manifest.json` records corpus/config hashes, CPU topology and affinity order,
tool versions, WSL detection, Git state, completeness counts, and the semantic
limits of the comparison. `summary.csv` reports median and MAD. The profiling
Rust release uses thin LTO as the retained compiler-only candidate.
Modular C/Rust
stdout also reports the `trainer_train()`/`trainer.train()` interval; these values
are stored in `runner_metrics.csv` and summarized in `training_summary.csv`.
The immutable original runner has no training-only metric. `speedup.csv`
uses whole-process wall time and nominal input tokens/s: upstream vocabulary
construction, table initialization, training, and output cannot be cleanly
separated. The modular runners additionally print actual processed-token counts.

Focused development runs may use `--threads 1,6,12`, `--models cbow`, and
`--objectives negative` without changing the default matrix. Rust policy
comparisons use `--update-strategy cas` or `--update-strategy hogwild`; the
latter is the canonical Rust default after validation on the fixed `scale=0.25`
workload. Stack completeness follows the selected model/objective matrix.

## Upstream oracle snapshot

The following is the recorded upstream `z_original_w2v.c` baseline from
`run-20260919T012028Z` (commit `61fba280f9711aaa3cc30dcad6f700ec86367e91`).
It used the generated corpus with 50,000 lexical words and 500,000 tokens
(`scale=0.25`), dimension 100, window 5, one epoch, and three repetitions per
condition. Values are median whole-process wall seconds; they include
vocabulary construction, table initialization, training, and output setup.

| model | objective | 1 thread | 2 threads | 4 threads | 6 threads | 12 threads |
|---|---:|---:|---:|---:|---:|---:|
| CBOW | HS | 1.287 | 0.825 | 0.529 | 0.453 | 0.380 |
| CBOW | negative | 1.575 | 1.009 | 0.804 | 0.664 | 0.649 |
| Skip-gram | HS | 3.395 | 2.160 | 1.234 | 0.936 | 0.740 |
| Skip-gram | negative | 4.525 | 2.488 | 1.467 | 1.174 | 0.903 |

The corresponding upstream 1-to-12-thread speedups are 3.39×, 2.43×, 4.59×,
and 5.01× in the table's row order. This is a regression oracle for the
unmodified upstream baseline only, not a semantic equivalence target for the
modular C or Rust implementations: their Skip-gram direction, RNG/learning
rate handling, and atomic Hogwild update contract differ. The original binary
also does not report processed-token counts. The raw counters, three-run MAD,
stack reports, corpus hash, and WSL2/CPU topology are preserved in the ignored
result directory `results/run-20260919T012028Z/`.
