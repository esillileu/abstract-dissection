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
