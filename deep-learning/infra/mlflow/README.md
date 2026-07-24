# Local MLflow

This Compose project is the single local MLflow server for this repository.
Its official MLflow image is pinned to `v3.14.0`. The SQLite backend and
served artifacts persist in `data/`, which is intentionally excluded from Git.

## Start and stop

From the repository root:

```bash
docker compose -f infra/mlflow/compose.yaml up -d
docker compose -f infra/mlflow/compose.yaml ps
```

Open <http://127.0.0.1:5000>. Experiment runners use this address by default;
set `MLFLOW_TRACKING_URI` only when using another server.

Stop the server without removing its data:

```bash
docker compose -f infra/mlflow/compose.yaml down
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

Export one run:

```bash
uv run python infra/mlflow/transfer.py export-run \
  --tracking-uri http://127.0.0.1:5000 \
  --run-id RUN_ID \
  --output /tmp/mlflow-run.zip
```

Export an experiment by name or numeric ID:

```bash
uv run python infra/mlflow/transfer.py export-experiment \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment EXPERIMENT_NAME_OR_ID \
  --output /tmp/mlflow-experiment.zip
```

Import into another store:

```bash
uv run python infra/mlflow/transfer.py import \
  --tracking-uri http://other-mlflow:5000 \
  --input /tmp/mlflow-experiment.zip \
  --destination-tag server.name=training-box-a
```

The target experiment name must not already exist. Use
`--experiment-name NEW_NAME` to rename it during import. A normal MLflow server
provides its own artifact location. When importing directly into a database
tracking URI, `--artifact-location file:///target/artifacts` can set it
explicitly.

Import automatically adds `transfer.destination.*` tags to the experiment and
every imported run. They describe the machine executing the import command:
hostname, platform, tracking endpoint, CPU model/count, total RAM, and detected
NVIDIA GPU name/memory/driver. Run the command on the destination server when
those tags must describe that server. Repeat `--destination-tag KEY=VALUE` for
deployment-specific identity such as a server or cluster name. Use
`--no-environment-tags` when automatic hardware tags are not wanted.
