# YAML experiment contract

All deepbase1 experiment YAML lives in this directory. Run one configuration with:

```bash
uv run python -m experiments.run_yaml experiments/deepbase1/config/e02_mnist_mlp_optimizer.yaml --atomic-run-id MLP-ADAM-HE --seed 1208965604 --device cpu
```

Start the local MLflow server first:

```bash
docker compose -f infra/mlflow/compose.yaml up -d
```

The SQLite backend and served artifacts are persisted under `infra/mlflow/data/`.
Install the tracking extra once before running experiments:

```bash
uv sync --extra tracking
```

The new catalog records under the MLflow experiment name `deepbase1`.

Run the complete catalog (516 runs) with CPU for MLP/probe experiments and GPU
for e08 CNN experiments. The runner verifies the MLflow server before starting;
if it is unavailable, no training run begins.

```bash
uv run python -m experiments.deepbase1.run_all
```

Preview the matrix or run a subset:

```bash
uv run python -m experiments.deepbase1.run_all --dry-run
uv run python -m experiments.deepbase1.run_all --experiments e02 e08
```

The `e01_…` through `e08_…` files are atomic-run catalogs. Select a declared
condition with `--atomic-run-id`, for example:

```bash
uv run python -m experiments.run_yaml experiments/deepbase1/config/e05_batchnorm_scale.yaml --atomic-run-id BN-ON-08
```

`kind` selects an executor defined in `mlprosection.experiment`.  The remaining
top-level sections are the experiment contract: identity fields, `dataset`,
`loader`, `model`, `initializer`, `optimizer`, `scheduler`, `loss`,
`training`, `evaluation`, `numerics`, `checkpoint`, `profiling`, and `policy`.
They are both the executor input and the schema-v1 MLflow parameter/artifact
projection; no experiment code imports or configures MLflow.

`tracking.enabled: false` keeps the same local schema-v1 artifact tree while
skipping the MLflow upload. During training, progress is forwarded only to the
drop-capable console worker. Once training finishes, metric history and final
metrics are sent to MLflow with bounded `log_batch` requests, followed by the
artifact upload.

Use the fixed values in `seeds.yaml` for repeated trials. For example,
`--seed 1208965604` records that value as the master seed and derives the
remaining named seeds deterministically.

Set `checkpoint.resume` to an `epoch-XXXX` directory from a prior run to resume
at the next epoch. The checkpoint must come from the same resolved configuration
(except for `checkpoint.resume` itself); model, BatchNorm buffers, optimizer,
trainer, Python, NumPy, and backend RNG states are restored together.
