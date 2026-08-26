# e02 fused negative-sampling runtime

## Scope

- Atomic run: `W2V-PTB-CBOW-FUSED-NS`
- Profile condition: `implemented-cbow-fused-ns`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Dataset: PTB train, 929,579 examples
- Training shape: batch 100, window 5, embedding 100, 5 negatives
- Optimizer: dense Adam, learning rate 0.001
- Timing mode: CUDA-synchronized training

The existing `W2V-PTB-CBOW-NS` and `W2V-PTB-SKIPGRAM-NS` variants retain
their original model, objective, trainer, and Adam optimizer. CBOW and
SkipGram fused paths are separate atomic runs; the timings below are for CBOW.

## Profile estimate

The e02 update protocol measured five independent 50-update throughput windows
after 20 warmup updates:

- steady update: 4.021 ± 0.805 ms
- update-only epoch estimate: 37.42 ± 7.48 s
- update-only 10-epoch estimate: 374.23 ± 74.78 s

This estimate excludes the experiment trainer's periodic recording probes.
The profile result is stored in
`.cache/exp/deepscratch/ds2/e02/implemented/profile/fused-dense-adam/cuda0/update.json`.

## Actual one-epoch training

Seed 1 completed all 9,295 updates without NaN, Inf, or divergence:

- synchronized training time: 59.969 s
- process run wall time: 61.755 s
- command wall time including process startup: 64.13 s
- final reporting loss: 0.371616
- final book loss: 2.229693

The run artifacts are stored in MLflow; no local historical mirror is required.

## Same-as-e02 training-time estimate

The trainer records one timing window every 20 updates. A moving-block
bootstrap resamples five adjacent steady windows at a time (100 updates) for
20,000 replicates, preserving short-range timing correlation.

| Workload | Estimated mean | Estimated standard deviation |
| --- | ---: | ---: |
| steady-state epoch | 59.09 s | 1.18 s |
| e02 10 epochs, one seed | 591.79 s (9m 51.8s) | 3.53 s |
| e02 10 epochs, all 10 seeds, sequential | 5,917.90 s (1h 38m 37.9s) | 11.17 s |

The steady-state epoch bootstrap 95% interval is 56.92–61.52 s. These
standard deviations estimate short-run timing noise on this machine. They do
not include longer-term thermal changes, other system load, checkpoint I/O,
or differences between machines.

## Default profile order

The complete profile runs all original conditions first. Within each
implementation/model group, the objective order is one-hot full softmax,
embedding full softmax, negative sampling, then fused negative sampling where
available. The original and SkipGram groups have no fused condition.
