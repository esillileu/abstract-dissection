# e02 fused negative-sampling numerical validation

## Result

The ordinary `NegativeSampling` path and the fused `FusedNegativeSampling`
path were run in lockstep for five updates for both CBOW and Skip-gram. Each
pair used the same initial `W_in`/`W_out`, batch data, negative candidate IDs,
and Adam configuration (`lr=0.001`).

The CPU validation passed all 10 update comparisons:

| model | max loss abs. error | max gradient combined error | max parameter combined error | max Adam-state combined error |
| --- | ---: | ---: | ---: | ---: |
| CBOW | `0.000e+00` | `2.324e-10` | `9.238e-10` | `5.818e-11` |
| Skip-gram | `0.000e+00` | `1.157e-09` | `2.554e-09` | `2.325e-10` |

After correcting the CUDA RawKernel scalar ABI types, the CUDA validation also
passed all 10 update comparisons:

| model | max loss abs. error | max gradient combined error | max parameter combined error | max Adam-state combined error |
| --- | ---: | ---: | ---: | ---: |
| CBOW | `0.000e+00` | `2.325e-10` | `9.217e-10` | `5.818e-11` |
| Skip-gram | `0.000e+00` | `9.240e-10` | `9.224e-10` | `1.163e-10` |

The configured ceilings were `1e-5` for loss, `1e-4` for gradients, and
`1e-4` for parameters and optimizer state. Every update passed. The complete
per-update JSON artifact and generated Markdown report are produced at
`exp/deepscratch/ds2/profile/e02/results/fused_validation.json` and
`exp/deepscratch/ds2/profile/e02/results/fused_validation.md` (ignored runtime
outputs).

## Protocol

- dtype: float32
- updates: 5 per model
- negative samples: 2 fixed candidates per target
- comparisons: loss, `W_in`/`W_out` gradients, parameters after Adam, Adam `m`/`v`
- CUDA reads are explicitly synchronized before each comparison

Run it with:

```bash
uv run python -m exp.deepscratch.ds2.profile.e02.validation \
  --device cpu
uv run python -m exp.deepscratch.ds2.profile.e02.validation \
  --device cuda:0
```

Each command writes both reports immediately. Use `--output` and `--report` to
choose different paths.

The generated CUDA report completed with overall status `PASS`.

The implementation and regression test are in
[`profile/e02/validation.py`](../profile/e02/validation.py) and
[`tests/test_e02_fused_validation.py`](../tests/test_e02_fused_validation.py).
