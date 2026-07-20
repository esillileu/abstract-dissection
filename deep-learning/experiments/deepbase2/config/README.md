# deepbase2 YAML experiment catalog

The catalog contains e09–e13 PTB and character sequence experiments. Every
configuration records to the MLflow experiment named `deepbase2`.

Start MLflow before a non-dry run:

```bash
uv sync --extra tracking
uv run mlflow server \
  --backend-store-uri sqlite:///experiments/data/mlflow.db \
  --serve-artifacts \
  --artifacts-destination file://$PWD/experiments/data/artifacts \
  --host 127.0.0.1 --port 5000
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
