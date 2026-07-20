from __future__ import annotations

from pathlib import Path
import time

from mlprosection.experiment import ExperimentContext, load_yaml, normalize_config, run_config

from .schema_v1 import SchemaV1Run
from .runtime import build_schema_metrics, write_json


def run_yaml(path: str | Path, *, atomic_run_id: str | None = None, seed: int | None = None, device: str | None = None, resume: str | None = None):
    """Run YAML and project its result to the schema-v1 MLflow record."""
    config = normalize_config(load_yaml(path, atomic_run_id=atomic_run_id))
    if seed is not None:
        config["seed"] = seed
    if device is not None:
        numerics = config["numerics"]
        assert isinstance(numerics, dict)
        numerics["device"] = device
        numerics["backend"] = "cupy" if device.startswith("cuda") else "numpy"
    if resume is not None:
        checkpoint = config["checkpoint"]
        assert isinstance(checkpoint, dict)
        checkpoint["resume"] = resume
    record = SchemaV1Run(config)
    runtime = record.runtime()
    context = ExperimentContext(
        emit_metric=lambda step, metrics: runtime.emit_metric(step=step, metrics=metrics),
        metadata={"checkpoint_root": record.artifact_root / "checkpoints"},
    )
    with runtime:
        started = time.perf_counter()
        result = run_config(
            config,
            context,
        )
        result.metrics.update(build_schema_metrics(
            train_loss=result.metrics.get("final/train/loss"),
            test_loss=result.metrics.get("final/test/loss"),
            train_accuracy=result.metrics.get("final/train/accuracy"),
            test_accuracy=result.metrics.get("final/test/accuracy"),
            profiling_metrics=result.profiling_metrics,
            total_updates=int(result.metrics.get("final/system/total_updates", 0)),
            completed_epochs=int(result.metrics.get("final/system/completed_epochs", 0)),
            samples_seen=int(result.metrics.get("final/system/samples_seen", 0)),
        ))
        result.metrics["runtime/run_wall_total_s"] = time.perf_counter() - started
        history = list(result.history)
        record.write_artifacts(
            model=result.model,
            final_metrics=result.metrics,
            history_rows=history,
            profiling_metrics=result.profiling_metrics,
            reproducibility={key: value for key, value in context.metadata.items() if key != "checkpoint_root"},
        )
        errors = runtime.complete(
            artifact_root=record.artifact_root,
            history_rows=history,
            final_metrics=result.metrics,
        )
    write_json(
        record.artifact_root / "runtime" / "upload_summary.json",
        {"uploaded": not errors and bool(config["tracking"].get("enabled", True)), "errors": errors, "artifact_root": str(record.artifact_root)},
    )
    return result
