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

Original fixed-seed measurements are durable source data and live with their
variant packages:

```text
exp/deepscratch/ds1/original/legacy_results/fixed_seed
exp/deepscratch/ds2/original/legacy_results/fixed_seed
```

Only their README files are tracked; payloads remain local durable data.
