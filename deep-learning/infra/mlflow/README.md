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

`transfer.py` moves one run or an entire experiment through a portable ZIP
archive. It preserves the complete metric history (value, timestamp, and step),
parameters, tags, run name/status/start/end times, dataset inputs, experiment
tags, and artifacts. Run and experiment IDs and target artifact URIs are newly
generated. Parent-run tags are remapped when an entire experiment is imported.
An experiment export includes only runs whose status is `FINISHED`; exporting
one explicitly selected run is allowed regardless of its status.

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
over both. Import does not deduplicate runs, so importing the same archive again
appends another copy.

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
