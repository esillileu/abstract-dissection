# Local MLflow

This Compose project is the single local MLflow server for this repository.
Its official MLflow image is pinned to `v3.14.0`. The SQLite backend and
served artifacts persist in `data/`, which is intentionally excluded from Git.

## Start and stop

From the repository root:

```bash
just mlflow up
just mlflow logs
just mlflow logs -f
```

Open <http://127.0.0.1:5000>. Experiment runners use this address by default;
set `MLFLOW_TRACKING_URI` only when using another server.

Stop the server without removing its data:

```bash
just mlflow down
```

`down -v` is not appropriate here: the experiment state is a bind-mounted host
directory, so remove `infra/mlflow/data/` only when intentionally discarding all
local MLflow history and artifacts.

## Export and import

`just mlflow export`, `export-run`, and `import` are the generic low-level
transfer tools. They remain available for operational backup, recovery, and
non-DeepScratch experiments. In particular, generic import may merge source
experiment tags and supplement missing data on a reused run.

`transfer.py` moves one run or an entire experiment through a portable v2 ZIP
archive. It preserves the complete metric history (value, timestamp, and step),
parameters, tags, run name/status/start/end times, dataset inputs, experiment
tags, and artifacts. Run and experiment IDs and target artifact URIs are newly
generated. Parent-run tags are remapped on import. Exports include referenced
parent runs even when a parent is not `FINISHED`, and importing a parent also
relinks existing children with the same unique `condition.group.key`.
An experiment export includes only runs whose status is `FINISHED`; exporting
one explicitly selected run is allowed regardless of its status.

The v2 manifest inventories `latest` and `best` as `present`, `missing`, or
`not_applicable`, including artifact paths and SHA-256 digests. Export falls
back to the exporting machine's local checkpoint path when MLflow has only the
checkpoint manifest. Import is idempotent by `run.key` for seed trials,
`condition.group.key` for condition parents, and source experiment/run identity
for other runs. Re-import verifies existing values and fills only missing
metrics and artifacts. Conflicting params, metric tuples, or artifact digests
fail without overwriting. Legacy v1 archives remain importable.

The Typer CLI stores archives in `infra/mlflow/exports/` by default. Export one
run:

```bash
just mlflow export-run RUN_ID
```

Export an experiment by name or numeric ID:

```bash
just mlflow export ds2
```

Copy `infra/mlflow/exports/ds2.zip` to the same path on another machine, then
import it. `MLFLOW_TRACKING_URI` selects the source or target server:

```bash
MLFLOW_TRACKING_URI=http://other-mlflow:5000 \
just mlflow import ds2 \
  --reuse-experiment \
  --destination-tag server.name=training-box-a
```

`--output/-o` and `--input/-i` override the default ZIP path.

The Typer command reuses an existing experiment by default. Pass
`--no-reuse-experiment` to require a new experiment and fail on a name
collision. Reuse preserves existing target experiment tag values when source
tags have the same keys; explicit `--destination-tag` values take precedence
over both. Importing the same archive again reuses the prior target runs.

A normal MLflow server provides its own artifact location. When creating a new
experiment directly through a database tracking URI,
`--artifact-location file:///target/artifacts` can set it explicitly. Reused
experiments retain their existing artifact location.

Import automatically adds `transfer.destination.*` tags to the experiment and
every imported run. They describe the machine executing the import command:
hostname, platform, tracking endpoint, CPU model/count, total RAM, and detected
NVIDIA GPU name/memory/driver. Run the command on the destination server when
those tags must describe that server. Repeat `--destination-tag KEY=VALUE` for
deployment-specific identity such as a server or cluster name. Use
`--no-environment-tags` when automatic hardware tags are not wanted.

## Checkpoint maintenance

All maintenance commands are dry-run unless `--apply` is present and write a
JSON report under `infra/mlflow/data/maintenance-reports/`.

```bash
just mlflow checkpoint-backfill ds1 ds2
just mlflow checkpoint-backfill ds1 ds2 --apply
just mlflow dedupe ds2
just mlflow dedupe ds2 --apply --purge-artifacts
just mlflow checkpoint-prune ds1 ds2
just mlflow checkpoint-prune ds1 ds2 --apply
just mlflow relink-parents ds1 ds2
just mlflow relink-parents ds1 ds2 --apply
```

Use the fixed order: canonical selection/backfill, digest verification,
deduplication, then pruning. `checkpoint-prune` directly removes artifacts only
inside the trusted self-hosted root `infra/mlflow/data/artifacts`; use
`--artifact-root` to declare another trusted root. A `file:` artifact URI
outside that root is rejected.

`relink-parents` audits seed trials against the active condition parent selected
by `condition.group.key`. It updates both `mlflow.parentRunId` and the repository
compatibility tag `parent.mlflow_run_id` only when exactly one parent matches.
If every child in a group points to the same soft-deleted matching parent and no
active parent exists, that parent is restored. Missing and ambiguous parents
remain unchanged and are recorded in the report.
