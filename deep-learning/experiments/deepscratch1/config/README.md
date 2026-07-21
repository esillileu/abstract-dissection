# YAML experiment contract

All deepscratch1 experiment YAML lives in this directory. Run one configuration with:

```bash
uv run python -m experiments.run_yaml experiments/deepscratch1/config/e02_mnist_mlp_optimizer.yaml --atomic-run-id MLP-ADAM-HE --seed 1208965604 --device cpu
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

The new catalog records under the MLflow experiment name `deepscratch1`.

Run the complete catalog (516 runs) with CPU for MLP/probe experiments and GPU
for e08 CNN experiments. The CLI verifies the MLflow server before starting;
if it is unavailable, no training run begins.

```bash
just exp deepscratch1 run --all
```

Preview the matrix or select seed-set indexes for a subset:

```bash
just exp deepscratch1 plan --all
just exp deepscratch1 run -e 02 -e 08 --seed 0-4
```

Render MLflow-backed analysis figures for every declared experiment, or one
experiment. Each script declares its own `EXPERIMENT_ID` and the atomic run IDs
it aggregates, so no run IDs need to be repeated in the CLI command.

```bash
just exp deepscratch1 analyze
just exp deepscratch1 analyze -e 02
just exp deepscratch1 analyze --dry-run
```

Use `--tracking-uri URL` when MLflow is not running at the default local URL.

The `e01_…` through `e08_…` files are atomic-run catalogs. Select a declared
condition with `--atomic-run-id`, for example:

```bash
uv run python -m experiments.run_yaml experiments/deepscratch1/config/e05_batchnorm_scale.yaml --atomic-run-id BN-ON-08
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

Checkpoints are stored under `experiments/deepscratch1/results/checkpoints/`.
`checkpoint.save_final` defaults to `true`; its final checkpoint uploads to
MLflow by default (`tracking.upload_checkpoint: true`). Set
`checkpoint.save_on_eval: true` to retain resume checkpoints after each
evaluation. Their MLflow upload remains opt-in via
`tracking.upload_eval_checkpoints: true`.

Use the fixed values in `seeds.yaml` for repeated trials. The common CLI's
`--seed 0-4` selects seed-set indexes; omitting it uses the YAML's declared
`policy.seed_count`.

Set `checkpoint.resume` to an `epoch-XXXX` directory from a prior run to resume
at the next epoch. The checkpoint must come from the same resolved configuration
(except for `checkpoint.resume` itself); model, BatchNorm buffers, optimizer,
trainer, Python, NumPy, and backend RNG states are restored together.
