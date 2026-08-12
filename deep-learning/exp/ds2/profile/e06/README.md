# DS2 e06 Seq2seq profiler

This profiler measures pure e06 training updates in an isolated directory.
It does not start MLflow, write checkpoints, or modify canonical experiment
results. Evaluation and artifact I/O are excluded from the timing window.

Run it after the shared GPU is available:

```bash
uv run python -m exp.ds2.profile.e06.benchmark
```

For a short smoke profile:

```bash
uv run python -c 'from pathlib import Path; from exp.ds2.profile.e06.benchmark import run; run(warmup=2, iterations=5, repetitions=1, output_dir=Path("exp/ds2/profile/e06/results/smoke"))'
```

The protocol uses the e06 dataset split, batch size 128, `drop_last=true`,
float32, Adam, gradient clipping 5.0, and independent fresh model state per
condition. It records vanilla/peeky and forward/reverse input conditions.

This is an implementation profiler, not yet an original-source comparison.
The resulting update timings can be compared with an equivalent original run
only when both are measured on the same idle GPU and with the same warmup and
repetition protocol.
