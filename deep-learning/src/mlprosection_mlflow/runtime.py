"""MLflow sink and artifact helpers, isolated behind the optional tracking extra."""

# ruff: noqa: E701, E702

from __future__ import annotations

import csv
import hashlib
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    schema_version: int; project_name: str; experiment_ids: tuple[str, ...]
    atomic_run_id: str; execution_group_id: str; recipe_id: str; structure_signature: str
    condition_key: str; run_key: str; master_seed: int


@dataclass(frozen=True)
class RuntimeOptions:
    tracking_uri: str; experiment_name: str; mlflow_enabled: bool = True
    upload_checkpoint: bool = False; queue_size: int = 256; metric_batch_size: int = 1000


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def make_condition_key(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def make_run_key(condition: dict[str, Any], seed: dict[str, Any]) -> str:
    return hashlib.sha256((canonical_json(condition) + canonical_json(seed)).encode()).hexdigest()


def flatten_dict(value: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict): output.update(flatten_dict(child, prefix=name))
        else: output[name] = child
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value, encoding="utf-8")


def write_history_csv(path: Path, *, run_key: str, rows: list[tuple[str, int, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file); writer.writerow(["run_key", "step_type", "step", "metric", "value", "timestamp"])
        for row in rows: writer.writerow([run_key, *row, time.time()])


def write_runtime_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step_type", "step", "train_s", "eval_s", "checkpoint_s", "throughput_samples_per_s"]); writer.writeheader(); writer.writerows(rows)


def write_memory_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_s", "cpu_rss_bytes", "gpu_used_bytes", "gpu_reserved_bytes"]); writer.writeheader(); writer.writerows(rows)


def _git(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    if check and result.returncode: raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def current_git_info(entrypoint: str) -> dict[str, Any]:
    diff = _git(["diff"], check=False)
    return {"repository": _git(["rev-parse", "--show-toplevel"]).split("/")[-1], "commit": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"]), "dirty": bool(_git(["status", "--porcelain"])), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "remote": _git(["remote", "get-url", "origin"], check=False), "entrypoint": entrypoint}


def write_git_diff(path: Path) -> None: write_text(path, _git(["diff"], check=False))
def environment_artifacts() -> dict[str, Any]: return {"platform": platform.platform(), "system": platform.system().lower(), "kernel": platform.release(), "python_version": platform.python_version(), "machine": platform.machine(), "processor": platform.processor()}
def pip_freeze() -> str: return subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True).stdout


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def parameter_manifest(model: Any) -> list[dict[str, Any]]:
    output = []
    for name, parameter in model.named_parameters():
        data = parameter.backend.to_numpy(parameter.data)
        output.append({"name": name, "shape": list(data.shape), "dtype": str(data.dtype), "requires_grad": bool(parameter.requires_grad), "numel": int(data.size), "final_mean": float(data.mean()), "final_std": float(data.std()), "final_min": float(data.min()), "final_max": float(data.max()), "final_digest": hashlib.sha256(data.tobytes()).hexdigest()})
    return output


def _set(metrics: dict[str, float], name: str, value: int | float | None) -> None:
    if value is not None: metrics[name] = float(value)


def build_schema_metrics(*, train_loss: float | None, test_loss: float | None, train_accuracy: float | None, test_accuracy: float | None, profiling_metrics: dict[str, int | float], total_updates: int, completed_epochs: int, samples_seen: int) -> dict[str, float]:
    metrics = {"final/status/success": 1.0, "final/status/nan_detected": 0.0, "final/status/inf_detected": 0.0, "final/status/diverged": 0.0, "final/system/total_updates": float(total_updates), "final/system/completed_epochs": float(completed_epochs), "final/system/samples_seen": float(samples_seen)}
    for key, value in (("final/train/loss", train_loss), ("final/test/loss", test_loss), ("final/train/accuracy", train_accuracy), ("final/test/accuracy", test_accuracy)): _set(metrics, key, value)
    _set(metrics, "runtime/train_total_s", None if profiling_metrics.get("runtime.train_total.mean_ms") is None else float(profiling_metrics["runtime.train_total.mean_ms"]) / 1000)
    for source, target in (("memory.run.start.cpu.rss_bytes", "memory/cpu_rss_start_bytes"), ("memory.run.end.cpu.rss_bytes", "memory/cpu_rss_end_bytes"), ("memory.peak_sampled.cpu.rss_bytes", "memory/cpu_rss_peak_sampled_bytes")): _set(metrics, target, profiling_metrics.get(source))
    total_train = metrics.get("runtime/train_total_s")
    for source, target in (("forward", "forward"), ("backward", "backward"), ("optimizer_update", "update"), ("gradient_clip", "gradient_clip"), ("train_step", "train_step")):
        prefix = f"runtime.profile.{source}."; count = profiling_metrics.get(prefix + "count"); mean = profiling_metrics.get(prefix + "mean_ms")
        _set(metrics, f"profile/{target}/count", count)
        if count is not None and mean is not None:
            total = float(count) * float(mean) / 1000; metrics[f"profile/{target}/total_s"] = total
            if total_train: metrics[f"profile/{target}/fraction_of_train_time"] = total / total_train
        for suffix, label in (("mean_ms", "mean_s"), ("p50_ms", "median_s"), ("p95_ms", "p95_s"), ("std_ms", "std_s"), ("min_ms", "min_s"), ("max_ms", "max_s")):
            if (value := profiling_metrics.get(prefix + suffix)) is not None: metrics[f"profile/{target}/{label}"] = float(value) / 1000
    metrics.setdefault("profile/gradient_clip/count", 0.0); metrics.setdefault("profile/gradient_clip/total_s", 0.0)
    return metrics


def build_epoch_metric_rows(*, train_losses: list[float], test_losses: list[float], train_accuracies: list[float], test_accuracies: list[float], profiling_metrics: dict[str, int | float]) -> list[tuple[str, int, str, float]]:
    rows = [("epoch", i, "train/accuracy", float(v)) for i, v in enumerate(train_accuracies)] + [("epoch", i, "test/accuracy", float(v)) for i, v in enumerate(test_accuracies)]
    rows += [("epoch", i, "train/loss", float(v)) for i, v in enumerate(train_losses[-len(train_accuracies):])]
    rows += [("epoch", i, "test/loss", float(v)) for i, v in enumerate(test_losses[-len(test_accuracies):])]
    return rows


def build_runtime_history_rows(profiling_metrics: dict[str, int | float]) -> list[dict[str, Any]]: return []
def build_memory_history_rows(profiling_metrics: dict[str, int | float]) -> list[dict[str, Any]]: return [{"timestamp_s": float(i), "cpu_rss_bytes": profiling_metrics.get(f"memory.{name}.cpu.rss_bytes", ""), "gpu_used_bytes": "", "gpu_reserved_bytes": ""} for i, name in enumerate(("run.start", "train.start", "train.end", "run.end"))]


class _Sink:
    def __init__(self, options: RuntimeOptions, run_name: str, tags: dict[str, str], params: dict[str, object]) -> None:
        self.options, self.run_name, self.tags, self.params = options, run_name, tags, params; self.events: queue.Queue[tuple[str, Any]] = queue.Queue(options.queue_size); self.errors: list[str] = []; self.mlflow = None; self.run_id = None; self.thread = None
    def start(self) -> None:
        self.thread = threading.Thread(target=self._consume, daemon=True); self.thread.start()
    def put(self, event: tuple[str, Any], drop: bool = False) -> None:
        try: self.events.put_nowait(event)
        except queue.Full:
            if not drop: self.events.put(event)
    def _consume(self) -> None:
        client = self._start_mlflow()
        while True:
            kind, value = self.events.get()
            try:
                if kind == "stop":
                    if self.mlflow:
                        try: self.mlflow.end_run(status="FINISHED")
                        except Exception as exc: self.errors.append(f"MLflow finalization failed: {exc}")
                    return
                if kind == "console": print(value, file=sys.stderr)
                elif kind == "metric":
                    step, metrics = value
                    print(_format_progress(step, metrics), file=sys.stderr)
                elif kind == "metrics" and client:
                    for batch in metric_batches(value, self.options.metric_batch_size):
                        client.log_batch(self.run_id, metrics=[
                            self.mlflow.entities.Metric(
                                key=key, value=float(metric), timestamp=int(time.time() * 1000), step=step,
                            )
                            for step, key, metric in batch
                        ])
                elif kind == "artifact" and client:
                    root = value
                    for path in root.rglob("*"):
                        is_checkpoint_payload = path.relative_to(root).as_posix() == "checkpoints/final.npz"
                        if path.is_file() and (self.options.upload_checkpoint or not is_checkpoint_payload):
                            client.log_artifact(self.run_id, str(path), artifact_path=str(path.parent.relative_to(root)))
            except Exception as exc: self.errors.append(f"MLflow upload failed: {exc}")
            finally: self.events.task_done()
    def _start_mlflow(self):
        if not self.options.mlflow_enabled:
            return None
        try:
            import mlflow
            mlflow.set_tracking_uri(self.options.tracking_uri); mlflow.set_experiment(self.options.experiment_name)
            self.run_id = mlflow.start_run(run_name=self.run_name, tags=self.tags).info.run_id
            mlflow.log_params({key: str(value) for key, value in self.params.items() if value is not None})
            self.mlflow = mlflow
            return mlflow.tracking.MlflowClient(tracking_uri=self.options.tracking_uri)
        except Exception as exc:
            self.errors.append(f"MLflow startup failed: {exc}")
            return None


class _Callback:
    def __init__(self, sink: _Sink) -> None: self.sink = sink
    def on_batch_end(self, *, step: int) -> None: pass
    def on_interval(self, *, metrics: dict[str, float]) -> None: self.sink.put(("metric", (int(metrics.get("iteration", 0)), {f"interval/{k}": v for k, v in metrics.items()})), drop=True); self.sink.put(("console", f"step={int(metrics.get('iteration', 0))} loss={metrics.get('loss', float('nan')):.4f}"), drop=True)
    def on_epoch_end(self, *, epoch: int, metrics: dict[str, float]) -> None: self.sink.put(("metric", (epoch, {f"epoch/{k}": v for k, v in metrics.items()}))); self.sink.put(("console", f"epoch={epoch} {metrics}"), drop=True)


class ExperimentRun:
    def __init__(self, *, options: RuntimeOptions, run_name: str, tags: dict[str, str], params: dict[str, object]) -> None: self.options = options; self.sink = _Sink(options, run_name, tags, params); self.trainer_callback = _Callback(self.sink); self.finished = False
    def __enter__(self): self.sink.start(); return self
    def emit_metric(self, *, step: int, metrics: dict[str, float], kind: str = "step") -> None:
        """Forward progress to the console only; MLflow metrics upload after training."""
        self.sink.put(("console", _format_progress(step, metrics)), drop=True)
    def complete(self, *, artifact_root: Path, history_rows: list[tuple[str, int, str, float]], final_metrics: dict[str, float]) -> list[str]:
        self.finished = True
        rows = [(step, f"{step_type}/{metric}", value) for step_type, step, metric, value in history_rows]
        rows.extend((0, key, value) for key, value in final_metrics.items())
        self.sink.put(("metrics", rows)); self.sink.put(("artifact", artifact_root)); self.sink.put(("stop", None)); self.sink.thread.join()
        return self.sink.errors
    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc and not self.finished: self.sink.put(("stop", None)); self.sink.thread.join()
        return False


def metric_batches(rows: list[tuple[int, str, float]], batch_size: int) -> list[list[tuple[int, str, float]]]:
    """Split the post-run MLflow metric payload into bounded API requests."""
    if batch_size < 1:
        raise ValueError("metric_batch_size must be at least 1")
    return [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]


def _format_progress(step: int, metrics: dict[str, float]) -> str:
    values = " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
    return f"step={step} {values}"
