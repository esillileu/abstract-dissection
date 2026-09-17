"""ExperimentRun orchestration and callbacks for MLflow tracking."""

from __future__ import annotations

from pathlib import Path

from .identity import RuntimeOptions
from .metrics import _format_progress
from .sink import _Sink


class _Callback:
    def __init__(self, sink: _Sink) -> None:
        self.sink = sink

    def on_batch_end(self, *, step: int) -> None:
        pass

    def on_interval(self, *, metrics: dict[str, float]) -> None:
        self.sink.put(
            (
                "metric",
                (
                    int(metrics.get("iteration", 0)),
                    {f"interval/{k}": v for k, v in metrics.items()},
                ),
            ),
            drop=True,
        )
        self.sink.put(
            (
                "console",
                f"step={int(metrics.get('iteration', 0))} loss={metrics.get('loss', float('nan')):.4f}",
            ),
            drop=True,
        )

    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None:
        self.sink.put(
            ("metric", (epoch, {f"epoch/{k}": v for k, v in metrics.items()}))
        )
        self.sink.put(("console", f"epoch={epoch} {metrics}"), drop=True)


class ExperimentRun:
    def __init__(
        self,
        *,
        options: RuntimeOptions,
        run_name: str,
        tags: dict[str, str],
        params: dict[str, object],
    ) -> None:
        self.options = options
        self.sink = _Sink(options, run_name, tags, params)
        self.trainer_callback = _Callback(self.sink)
        self.finished = False

    def __enter__(self):
        self.sink.start()
        return self

    def emit_metric(
        self, *, step: int, metrics: dict[str, float], kind: str = "step"
    ) -> None:
        """Forward progress to the console only; MLflow metrics upload after training."""
        self.sink.put(("console", _format_progress(step, metrics)), drop=True)

    def emit_checkpoint(
        self,
        path: Path,
        *,
        checkpoint_kind: str,
        artifact_path: str | None = None,
    ) -> None:
        self.sink.put(("checkpoint", (path, checkpoint_kind, artifact_path)))

    def complete(
        self,
        *,
        artifact_root: Path,
        metric_rows: list[tuple[int, str, float]],
        final_metrics: dict[str, float],
        checkpoint_path: Path | None = None,
        checkpoint_paths: dict[str, Path] | None = None,
    ) -> list[str]:
        self.finished = True
        rows = list(metric_rows)
        rows.extend((0, key, value) for key, value in final_metrics.items())
        self.sink.put(("metrics", rows))
        self.sink.put(("artifact", artifact_root))
        roles = dict(checkpoint_paths or {})
        if checkpoint_path is not None:
            roles.setdefault("latest", checkpoint_path)
        uploaded: set[Path] = set()
        for role in ("latest", "best"):
            path = roles.get(role)
            if path is None or not path.exists():
                continue
            resolved = path.resolve()
            if resolved in uploaded:
                continue
            uploaded.add(resolved)
            self.emit_checkpoint(path, checkpoint_kind=role)
        self.sink.put(("stop", "FINISHED"))
        self.sink.thread.join()
        return self.sink.errors

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc and not self.finished:
            self.sink.put(("stop", "FAILED"))
            self.sink.thread.join()
        return False


__all__ = [
    "ExperimentRun",
]
