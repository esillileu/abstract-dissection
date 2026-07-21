# deepscratch2 YAML experiment catalog

The catalog contains e09–e13 PTB and character sequence experiments. Every
configuration records to the MLflow experiment named `deepscratch2`.

Start MLflow before a non-dry run:

```bash
docker compose -f infra/mlflow/compose.yaml up -d
```

The SQLite backend and served artifacts are persisted under `infra/mlflow/data/`.
Install the tracking extra once before running experiments:

```bash
uv sync --extra tracking
```

Final checkpoints upload to MLflow by default. To save a resumable checkpoint
after every evaluation, set `checkpoint.save_on_eval: true`; uploading those
intermediate checkpoints remains opt-in with
`tracking.upload_eval_checkpoints: true`.

Preview the 140-run matrix without starting training:

```bash
just exp deepscratch2 plan --all
```

The default device is `cuda:0`; it is checked before any training starts. Use
CPU only explicitly:

```bash
just exp deepscratch2 run -e 12 -e 13 --seed 0-4 --device cpu
```

Render MLflow-backed analysis figures using the scripts declared in this domain:

```bash
just exp deepscratch2 analyze
just exp deepscratch2 analyze -e 13
just exp deepscratch2 analyze -e 13 --dry-run
```

The selected script owns its `EXPERIMENT_ID` and the atomic run IDs it
aggregates. Add `--tracking-uri URL` to analyze a non-default MLflow server.

Run one declared condition:

```bash
uv run python -m experiments.run_yaml \
  experiments/deepscratch2/config/e11_rnnlm_comparison.yaml \
  --atomic-run-id LM-LSTM-C025 --seed 1208965604 --device cuda:0
```
