from __future__ import annotations

import importlib
import json
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Callable

from mlprosection.experiment import ExperimentContext, run_config
from mlprosection.experiment.progress import ProgressReporter

from .checkpoint_source import resolve_checkpoint_source
from .schema_v1 import SchemaV1Run, write_result_manifest
from .runtime import build_profiling_metric_rows, build_schema_metrics, write_json


@dataclass(frozen=True)
class RunYamlReceipt:
    result: object
    run_id: str | None
    staging_root: Path
    durable_complete: bool


def run_yaml(
    path: str | Path,
    *,
    atomic_run_id: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    resume: str | None = None,
    overrides: dict[str, object] | None = None,
    executor_module: str | None = None,
    spec_module: str | None = None,
    progress_reporter: ProgressReporter | None = None,
    checkpoint_source_resolver: Callable[[dict[str, object]], Path | None] = resolve_checkpoint_source,
):
    """Run a domain YAML and upload the CSV-backed record to MLflow."""
    if spec_module is None:
        raise ValueError("run_yaml requires a domain spec_module")
    if executor_module is None:
        raise ValueError("run_yaml requires a domain executor_module")
    parser = importlib.import_module(spec_module)
    config = parser.parse_run_spec(path, atomic_run_id=atomic_run_id, overrides=overrides).to_executor_config()
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
    checkpoint_source_resolver(config)
    runtime = record.runtime()
    if progress_reporter is not None:
        runtime.sink.console_writer = progress_reporter.write
    evaluation_checkpoints: list[Path] = []

    def record_eval_checkpoint(path: Path) -> None:
        evaluation_checkpoints.append(path)

    context = ExperimentContext(
        emit_metric=lambda step, metrics: runtime.emit_metric(step=step, metrics=metrics),
        metadata={
            "checkpoint_root": record.local_checkpoint_root,
            "artifact_root": record.artifact_root,
            "record_eval_checkpoint": record_eval_checkpoint,
            **({} if progress_reporter is None else {"progress_reporter": progress_reporter}),
        },
    )
    with runtime:
        started = time.perf_counter()
        result = run_config(
            config,
            context,
            executor_module=executor_module,
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
        metric_rows = [*result.metric_rows, *build_profiling_metric_rows(result.profiling_metrics)]
        record.write_artifacts(
            model=result.model,
            final_metrics=result.metrics,
            metric_rows=metric_rows,
            profiling_metrics=result.profiling_metrics,
            reproducibility={
                key: value
                for key, value in context.metadata.items()
                if key not in {"checkpoint_root", "artifact_root", "record_eval_checkpoint", "progress_reporter"}
            },
            evaluation_checkpoints=evaluation_checkpoints,
        )
        checkpoint_paths: dict[str, Path] = {}
        for role in ("latest", "best"):
            pointer = record.local_checkpoint_root / f"{role}.json"
            if not pointer.exists():
                continue
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            candidate = record.local_checkpoint_root / str(payload["path"])
            if candidate.exists():
                checkpoint_paths[role] = candidate
        write_result_manifest(record.artifact_root, checkpoint_paths)
        errors = runtime.complete(
            artifact_root=record.artifact_root,
            metric_rows=metric_rows,
            final_metrics=result.metrics,
            checkpoint_paths=checkpoint_paths,
        )
    write_json(
        record.artifact_root / "runtime" / "upload_summary.json",
        {"uploaded": not errors and bool(config["tracking"].get("enabled", True)), "errors": errors, "artifact_root": str(record.artifact_root)},
    )
    return RunYamlReceipt(
        result=result,
        run_id=runtime.sink.run_id,
        staging_root=record.artifact_root.parent,
        durable_complete=(
            bool(config["tracking"].get("enabled", True))
            and runtime.sink.run_id is not None
            and not errors
        ),
    )
