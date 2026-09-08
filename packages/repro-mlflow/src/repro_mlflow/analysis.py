def mlflow_client(tracking_uri: str):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Install tracking dependencies with `uv sync --extra tracking`."
        ) from exc
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)
