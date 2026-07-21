# deepbase2 YAML experiment catalog

The catalog contains e09–e13 PTB and character sequence experiments. Every
configuration records to the MLflow experiment named `deepbase2`.

Start MLflow before a non-dry run:

```bash
docker compose -f infra/mlflow/compose.yaml up -d
```

The SQLite backend and served artifacts are persisted under `infra/mlflow/data/`.
Install the tracking extra once before running experiments:

```bash
uv sync --extra tracking
```

Preview the 140-run matrix without starting training:

```bash
uv run python -m experiments.deepbase2.run_all --dry-run
```

The default device is `cuda:0`; it is checked before any training starts. Use
CPU only explicitly:

```bash
uv run python -m experiments.deepbase2.run_all --experiments e12 e13 --device cpu
```

Run one declared condition:

```bash
uv run python -m experiments.run_yaml \
  experiments/deepbase2/config/e11_rnnlm_comparison.yaml \
  --atomic-run-id LM-LSTM-C025 --seed 1208965604 --device cuda:0
```
