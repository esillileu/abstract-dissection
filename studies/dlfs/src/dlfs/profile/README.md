# Canonical profile engine

Performance-profile studies reuse the normal experiment Planner, Runner,
staging, and durable MLflow lifecycle. The profile layer separates four roles:

- `contracts.py`: measurement protocol, generic scaling axis, workload and
  result contracts.
- `engine.py`: backend-synchronized update measurement independent of model,
  dataset, and scaling axis.
- domain study registry: resolves `profiling.study_kind` to one study.
- volume workload adapter: constructs typed workloads for a specific model
  family without leaking legacy condition strings into the executor.

Every executor result is projected to `profile/result.json` using the common
`ProfileStudyResult` schema. Scalar point metrics are also projected to MLflow
under `profile/<axis>/<value>/...`; studies without an axis use
`profile/...` directly.

Profile analysis selects only durable `FINISHED` attempts matching study,
schema, protocol version, device, and timing source. Profile studies are
excluded from the ordinary `--all` expansion and require explicit selection.

To add another axis-scaling study, its study adapter declares an axis rather
than adding a new executor branch:

```yaml
profiling:
  study_kind: axis_scaling
  axis:
    name: batch_size
    values: [32, 64, 128]
  warmup_updates: 20
  measured_updates: 50
  repetitions: 5
```

The promoted DS2 Word2Vec studies use typed workloads under
`ds2/profile/word2vec/`. The former direct e02 profiler is not a second entry
point: e10/PF01 and e11/PF02 run through the registered canonical executor.
