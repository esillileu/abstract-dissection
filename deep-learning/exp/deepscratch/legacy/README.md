# Local legacy result inventory

The retired `exp/ds1`, `exp/ds2`, `exp/ds1_original`, and `exp/ds2_original`
directories are no longer Python packages or result roots.

Local data that has not yet been proven recoverable from canonical MLflow is
quarantined under:

```text
.legacy/experiments/ds1/results
.legacy/experiments/ds2/results
.legacy/experiments/ds1_original/results
.legacy/experiments/ds2_original/results
```

This inventory includes historical checkpoint and MLflow artifact mirrors and
must not be deleted until artifact/checkpoint digests have been audited against
MLflow. Derived images in the same quarantine may be regenerated, but keeping
the directory together prevents an unaudited mirror from being mistaken for a
disposable cache.

Original fixed-seed measurements are historical durable data. Their canonical
quarantine locations are:

```text
.legacy/experiments/ds1_original/fixed_seed
.legacy/experiments/ds2_original/fixed_seed
```

Existing source-tree payloads under
`exp/deepscratch/<ds1|ds2>/original/legacy_results/fixed_seed` are supported as
a read-only migration fallback. New tooling never writes there; the payloads
must not be moved or deleted before audit.

## Archive import and recovery

Archives are imported append-only into the exact historical namespace:

```text
ds1          -> ds1
ds1_original -> ds1_original
ds2          -> ds2
ds2_original -> ds2_original
```

Use `exp import-legacy deepscratch <ds1|ds2> --variant <variant> --input
<archive.zip>`. An identical payload is reused, a different payload with the
same run key is retained as `imported-alternate`, and a collision with a
running run is deferred. Imports never copy a historical run into a
`deepscratch.ds1` or `deepscratch.ds2` writer namespace.

The preserved DS2 original regression coordinate is e05 / `BETTER-RNNLM` /
seed 4, run `8b19fdcd874c4c38b6a6480dc865101c`. It remains in
`ds2_original`; recovery validates its artifact and checkpoint inventory in
place instead of cloning it.

To recover derived output on another machine, import the archive, run storage
audit, remove obsolete pre-policy cache and artifact trees if desired, and
rerun `exp analyze`. Do not remove `.legacy` or incomplete staging as part of
that procedure.
