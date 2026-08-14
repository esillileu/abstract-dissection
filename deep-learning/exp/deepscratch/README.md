# DeepScratch commands

The canonical CLI owns the DS1/DS2 volume and implemented/original variant
vocabulary.

The old `exp plan|run|analyze ds1`, `ds2`, `ds1_original`, and `ds2_original`
commands have been removed. Historical MLflow experiment names remain readable
storage namespaces; they are not executable domains. Runtime code, configuration,
analysis, profiling, and vendored source now live below
`exp/deepscratch/ds1` and `exp/deepscratch/ds2`.

The domain root contains only `cli.py`, `definition.py`, `identity.py`, and
the owned `analysis/`, `execution/`, `legacy/`, `original_runtime/`, `ds1/`,
and `ds2/` packages. Result payloads never belong under the source tree.

Human-facing analysis output defaults to one flat directory:

```text
results/exp/deepscratch
```

Use `EXP_RESULTS_ROOT` to replace the `results` root. Only PNG and Markdown
are published there; observations, CSV/JSON data, and manifests stay in cache.

New run state has three distinct owners:

```text
.staging/exp          run records and checkpoints pending MLflow verification
.cache/mlflow_artifact server/run-scoped artifact downloads
.cache/exp            transformed data, observations, and analysis manifests
results/exp           explicitly preserved PNG and Markdown results
```

Override them with `EXP_STAGING_ROOT`, `EXP_CACHE_ROOT`, and
`EXP_RESULTS_ROOT`. A SchemaV1 run remains incomplete
until MLflow has received and digest-verified its result manifest and required
checkpoint payloads. Only then is `result.durable_complete=true` written.

## Check planned run coverage

`check` expands the same catalog, condition, and seed selection as `plan`, then
looks for matching runs in both the new writer namespace and its historical
namespace:

```bash
just exp check deepscratch ds2 -e 05
just exp check deepscratch ds2 -e 05 --seed 1-4
just exp check deepscratch ds2 --all --variant original
```

The summary separates:

- `completed`: a protocol-compatible `FINISHED` attempt exists.
- `running`: there is no completed attempt, but one is `RUNNING` or `SCHEDULED`.
- `failed`: attempts exist, but none completed or remain active.
- `missing`: no canonical attempt exists for the planned condition and seed.

By default only incomplete entries are printed. Use `--show missing` for runs
that were never attempted, `--show all` for the complete matrix, and `--json`
for automation. Imported alternate payloads never satisfy the default plan;
they remain available through explicit run-ID selection.

The command accepts the same `-e`, `-a`, `-x`, `--seed`, `--seed-set`, and
`--set` selection inputs as planning. `-o` is the alias for
`--variant original`.

The same original alias is available for analysis summaries:

```bash
just exp analyze deepscratch ds2 -e 05 -o
```

Cross-variant analysis is explicit and writes a comparison table containing
only declared comparable metrics. Missing native observations are recorded as
`unavailable` rather than synthesized:

```bash
just exp analyze deepscratch ds2 -e 03 --variant all
```

An imported alternate remains excluded by default. Select an exact attempt
when needed:

```bash
just exp analyze deepscratch ds2 -e 03 -o --run-id <mlflow-run-id>
```

## Audit and clean local mirrors

Storage cleanup is always a dry run unless `--apply` is present:

```bash
just exp storage deepscratch audit
just exp storage deepscratch cleanup --verified-mirrors
just exp storage deepscratch cleanup --verified-mirrors --apply
```

Only a locally digest-valid mirror backed by a `FINISHED`, durable MLflow run
is eligible. Running, failed, incomplete, orphaned, and legacy-quarantined data
is never selected. The audit also refuses cutover while a historical writer
namespace contains a running run.
