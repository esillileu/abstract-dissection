# DeepScratch commands

The canonical CLI owns the DS1/DS2 volume and implemented/original variant
vocabulary.

The old `exp plan|run|analyze ds1`, `ds2`, `ds1_original`, and `ds2_original`
commands have been removed. Historical MLflow experiment names remain readable
storage namespaces; they are not executable domains. Runtime code, configuration,
analysis, profiling, and vendored source now live below
`exp/deepscratch/ds1` and `exp/deepscratch/ds2`.

Derived analysis output defaults to:

```text
.artifacts/experiments/deepscratch/<volume>/<experiment>/<variant>/analysis
```

Use `EXP_ARTIFACT_ROOT` to replace the `.artifacts/experiments` root.

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
just exp analyze deepscratch ds2 -e 05 -o -s
```
