"""Background async MLflow event consumer and artifact uploader."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .identity import RuntimeOptions
from .metrics import _format_progress, metric_batches
from .parent import get_or_create_condition_parent
from .verification import _verify_uploaded_manifest


def _silence_mlflow_progress_logs() -> None:
    """Keep MLflow's lifecycle INFO messages out of the tqdm render stream."""
    logging.getLogger("mlflow.tracking.fluent").setLevel(logging.WARNING)


class _Sink:
    def __init__(
        self,
        options: RuntimeOptions,
        run_name: str,
        tags: dict[str, str],
        params: dict[str, object],
    ) -> None:
        self.options, self.run_name, self.tags, self.params = (
            options,
            run_name,
            tags,
            params,
        )
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue(options.queue_size)
        self.errors: list[str] = []
        self.mlflow = None
        self.run_id = None
        self.thread = None
        self.console_writer = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def put(self, event: tuple[str, Any], drop: bool = False) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            if not drop:
                self.events.put(event)

    def _consume(self) -> None:
        client = self._start_mlflow()
        while True:
            kind, value = self.events.get()
            try:
                if kind == "stop":
                    if self.mlflow:
                        status = "FAILED" if self.errors else str(value or "FINISHED")
                        try:
                            if status == "FINISHED" and client and self.run_id:
                                try:
                                    _verify_uploaded_manifest(client, self.run_id)
                                except Exception as exc:
                                    self.errors.append(
                                        f"MLflow artifact verification failed: {exc}"
                                    )
                                    status = "FAILED"
                            if client and self.run_id:
                                client.set_tag(
                                    self.run_id,
                                    "trial.status",
                                    "finished" if status == "FINISHED" else "failed",
                                )
                                client.set_tag(
                                    self.run_id,
                                    "result.durable_complete",
                                    "true" if status == "FINISHED" else "false",
                                )
                            self.mlflow.end_run(status=status)
                        except Exception as exc:
                            self.errors.append(f"MLflow finalization failed: {exc}")
                    return
                if kind == "console":
                    if self.console_writer is not None:
                        self.console_writer(str(value))
                    else:
                        print(value, file=sys.stderr)
                elif kind == "metric":
                    step, metrics = value
                    print(_format_progress(step, metrics), file=sys.stderr)
                elif kind == "metrics" and client:
                    for batch in metric_batches(value, self.options.metric_batch_size):
                        client.log_batch(
                            self.run_id,
                            metrics=[
                                self.mlflow.entities.Metric(
                                    key=key,
                                    value=float(metric),
                                    timestamp=int(time.time() * 1000),
                                    step=step,
                                )
                                for step, key, metric in batch
                            ],
                        )
                elif kind == "artifact" and client:
                    root = value
                    for path in root.rglob("*"):
                        relative = path.relative_to(root).as_posix()
                        is_checkpoint_payload = (
                            relative.startswith("checkpoints/")
                            and relative != "checkpoints/checkpoint_manifest.json"
                        )
                        should_upload = not is_checkpoint_payload
                        if path.is_file() and should_upload:
                            relative_parent = path.parent.relative_to(root)
                            artifact_path = (
                                None
                                if relative_parent == Path(".")
                                else relative_parent.as_posix()
                            )
                            client.log_artifact(
                                self.run_id, str(path), artifact_path=artifact_path
                            )
                elif kind == "checkpoint" and client:
                    path, checkpoint_kind, artifact_path = (
                        (*value, None) if len(value) == 2 else value
                    )
                    checkpoint_kind = (
                        "latest" if checkpoint_kind == "final" else checkpoint_kind
                    )
                    should_upload = (
                        checkpoint_kind == "latest" and self.options.upload_checkpoint
                    ) or (
                        checkpoint_kind in {"best", "eval"}
                        and self.options.upload_eval_checkpoints
                    )
                    if should_upload:
                        if path.is_dir():
                            client.log_artifacts(
                                self.run_id,
                                str(path),
                                artifact_path=artifact_path
                                or f"checkpoints/generations/{path.name}",
                            )
                        else:
                            client.log_artifact(
                                self.run_id,
                                str(path),
                                artifact_path=artifact_path or "checkpoints",
                            )
            except Exception as exc:
                self.errors.append(f"MLflow upload failed: {exc}")
            finally:
                self.events.task_done()

    def _start_mlflow(self):
        try:
            import mlflow

            _silence_mlflow_progress_logs()
            mlflow.set_tracking_uri(self.options.tracking_uri)
            experiment = mlflow.set_experiment(self.options.experiment_name)
            client = mlflow.tracking.MlflowClient(
                tracking_uri=self.options.tracking_uri
            )
            parent_run_id = get_or_create_condition_parent(
                client,
                experiment_id=experiment.experiment_id,
                child_tags=self.tags,
            )
            child_tags = {
                **self.tags,
                "result.durable_complete": "false",
                "mlflow.parentRunId": parent_run_id,
                "parent.mlflow_run_id": parent_run_id,
            }
            self.run_id = mlflow.start_run(
                run_name=self.run_name, tags=child_tags
            ).info.run_id
            mlflow.log_params(
                {
                    key: str(value)
                    for key, value in self.params.items()
                    if value is not None
                }
            )
            self.mlflow = mlflow
            return client
        except Exception as exc:
            self.errors.append(f"MLflow startup failed: {exc}")
            return None


__all__ = [
    "_Sink",
    "_silence_mlflow_progress_logs",
]
