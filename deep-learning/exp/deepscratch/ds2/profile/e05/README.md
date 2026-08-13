# DS2 e05 BetterRnnlm profile

This profiler measures the `LM-BETTER-RECIPE` training update without running
the 40-epoch job. Raw JSON, CSV, SQLite, and Nsight files are written below the
ignored `results/` directory. The reviewed summary is tracked in `report.md`.

## Commands

Capture the immutable pre-Phase-1 reference baseline:

```bash
just exp profile ds2 -e 05 \
  --stage baseline \
  --update-warmup 20 \
  --measured-updates 50 \
  --update-repetitions 5
```

The command refuses to overwrite `results/baseline/benchmark.json`. Run the
Phase 1 implementation with the same protocol:

```bash
just exp profile ds2 -e 05 \
  --stage phase1 \
  --update-warmup 20 \
  --measured-updates 50 \
  --update-repetitions 5
uv run python -m exp.ds2.profile.e05.validation
```

Phase 2 and Phase 3 remain independently selectable after production promotion:

```bash
just exp profile ds2 -e 05 --stage phase2
just exp profile ds2 -e 05 --stage phase3
uv run python -m exp.ds2.profile.e05.validation --stage phase3
```

Capture short before/after Nsight traces and summarize them:

```bash
exp/deepscratch/ds2/profile/e05/run_nsys.sh baseline phase1 phase2 phase3
```

The trace was verified with Nsight Systems 2026.4.1. The script retries
transient WSL driver-injection failures up to three times.

## Measurement contract

- TimeLSTM shape: `N=20, T=35, D=H=650`, float32.
- Warmup: 20 operations; measurement: 50 consecutive operations in each of
  five windows. CUDA-event timings are synchronized only at window boundaries.
- Forward restores fixed initial h/c before the event. Backward prepares its
  forward/cache before the event.
- Full update uses the cached PTB train stream, vocabulary 10,000, batch 20,
  BPTT 35, two stateful 650-unit LSTMs, dropout 0.5, tied affine, temporal mean
  softmax, global clipping 0.25, and SGD 20.
- The authoritative wall window includes batch generation, forward, objective,
  backward, clipping, SGD, recurrent-state detach, and the terminal synchronize.
- Epoch estimates use 1,327 steady-state updates. Evaluation, checkpoint, and
  MLflow I/O are explicitly excluded.

The baseline uses `ReferenceTimeLSTM`, a frozen copy of commit `8e29004`.
Phase selectors preserve each implementation independently, so production
promotion cannot silently redefine earlier benchmark stages.
