"""Metric projection, epoch history formatting, and batching helpers."""

from __future__ import annotations

import re
from typing import Any


def _set(metrics: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        metrics[name] = float(value)


def build_schema_metrics(
    *,
    train_loss: float | None,
    test_loss: float | None,
    train_accuracy: float | None,
    test_accuracy: float | None,
    profiling_metrics: dict[str, int | float],
    total_updates: int,
    completed_epochs: int,
    samples_seen: int,
) -> dict[str, float]:
    metrics = {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": float(total_updates),
        "final/system/completed_epochs": float(completed_epochs),
        "final/system/samples_seen": float(samples_seen),
    }
    for key, value in (
        ("final/train/loss", train_loss),
        ("final/test/loss", test_loss),
        ("final/train/accuracy", train_accuracy),
        ("final/test/accuracy", test_accuracy),
    ):
        _set(metrics, key, value)
    _set(
        metrics,
        "runtime/train_total_s",
        None
        if profiling_metrics.get("runtime.train_total.mean_ms") is None
        else float(profiling_metrics["runtime.train_total.mean_ms"]) / 1000,
    )
    for source, target in (
        ("memory.run.start.cpu.rss_bytes", "memory/cpu_rss_start_bytes"),
        ("memory.run.end.cpu.rss_bytes", "memory/cpu_rss_end_bytes"),
        ("memory.peak_sampled.cpu.rss_bytes", "memory/cpu_rss_peak_sampled_bytes"),
    ):
        _set(metrics, target, profiling_metrics.get(source))
    total_train = metrics.get("runtime/train_total_s")
    for source, target in (
        ("forward", "forward"),
        ("backward", "backward"),
        ("optimizer_update", "update"),
        ("gradient_clip", "gradient_clip"),
        ("train_step", "train_step"),
    ):
        prefix = f"runtime.profile.{source}."
        count = profiling_metrics.get(prefix + "count")
        mean = profiling_metrics.get(prefix + "mean_ms")
        _set(metrics, f"profile/{target}/count", count)
        if count is not None and mean is not None:
            total = float(count) * float(mean) / 1000
            metrics[f"profile/{target}/total_s"] = total
            if total_train:
                metrics[f"profile/{target}/fraction_of_train_time"] = (
                    total / total_train
                )
        for suffix, label in (
            ("mean_ms", "mean_s"),
            ("p50_ms", "median_s"),
            ("p95_ms", "p95_s"),
            ("std_ms", "std_s"),
            ("min_ms", "min_s"),
            ("max_ms", "max_s"),
        ):
            if (value := profiling_metrics.get(prefix + suffix)) is not None:
                metrics[f"profile/{target}/{label}"] = float(value) / 1000
    metrics.setdefault("profile/gradient_clip/count", 0.0)
    metrics.setdefault("profile/gradient_clip/total_s", 0.0)
    return metrics


def build_epoch_metric_rows(
    *,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
    profiling_metrics: dict[str, int | float],
) -> list[tuple[str, int, str, float]]:
    rows = [
        ("epoch", i, "train/accuracy", float(v)) for i, v in enumerate(train_accuracies)
    ] + [("epoch", i, "test/accuracy", float(v)) for i, v in enumerate(test_accuracies)]
    rows += [
        ("epoch", i, "train/loss", float(v))
        for i, v in enumerate(train_losses[-len(train_accuracies) :])
    ]
    rows += [
        ("epoch", i, "test/loss", float(v))
        for i, v in enumerate(test_losses[-len(test_accuracies) :])
    ]
    return rows


def build_profiling_metric_rows(
    profiling_metrics: dict[str, int | float],
) -> list[tuple[int, str, float]]:
    """Project per-epoch profiler values to direct MLflow metric rows."""
    rows: list[tuple[int, str, float]] = []
    for key, value in profiling_metrics.items():
        duration = re.fullmatch(r"runtime\.epoch\.(\d+)\.(train|eval)_duration_ms", key)
        throughput = re.fullmatch(
            r"throughput\.epoch\.(\d+)\.(train|eval)_samples_per_s", key
        )
        memory = re.fullmatch(
            r"memory\.epoch\.(\d+)\.(train|eval)\.(start|end)\.(.+)", key
        )
        if duration:
            rows.append(
                (
                    int(duration.group(1)) + 1,
                    f"epoch/runtime/{duration.group(2)}_duration_s",
                    float(value) / 1000,
                )
            )
        elif throughput:
            rows.append(
                (
                    int(throughput.group(1)) + 1,
                    f"epoch/throughput/{throughput.group(2)}_samples_per_s",
                    float(value),
                )
            )
        elif memory:
            rows.append(
                (
                    int(memory.group(1)) + 1,
                    f"epoch/memory/{memory.group(2)}_{memory.group(3)}/{memory.group(4).replace('.', '_')}",
                    float(value),
                )
            )
    return rows


def build_runtime_history_rows(
    profiling_metrics: dict[str, int | float],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for step, key, value in build_profiling_metric_rows(profiling_metrics):
        _, metric = key.split("/", 1)
        row = grouped.setdefault(
            step,
            {
                "step_type": "epoch",
                "step": step,
                "train_s": "",
                "eval_s": "",
                "checkpoint_s": "",
                "throughput_samples_per_s": "",
            },
        )
        if metric == "runtime/train_duration_s":
            row["train_s"] = value
        elif metric == "runtime/eval_duration_s":
            row["eval_s"] = value
        elif metric == "throughput/train_samples_per_s":
            row["throughput_samples_per_s"] = value
    return [grouped[step] for step in sorted(grouped)]


def build_memory_history_rows(
    profiling_metrics: dict[str, int | float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(
        sorted(key for key in profiling_metrics if key.endswith(".cpu.rss_bytes"))
    ):
        rows.append(
            {
                "timestamp_s": float(index),
                "cpu_rss_bytes": profiling_metrics[key],
                "gpu_used_bytes": "",
                "gpu_reserved_bytes": "",
            }
        )
    return rows


def metric_batches(
    rows: list[tuple[int, str, float]], batch_size: int
) -> list[list[tuple[int, str, float]]]:
    """Split the post-run MLflow metric payload into bounded API requests."""
    if batch_size < 1:
        raise ValueError("metric_batch_size must be at least 1")
    return [
        rows[index : index + batch_size] for index in range(0, len(rows), batch_size)
    ]


def _format_progress(step: int, metrics: dict[str, float]) -> str:
    values = " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
    return f"step={step} {values}"


__all__ = [
    "build_epoch_metric_rows",
    "build_memory_history_rows",
    "build_profiling_metric_rows",
    "build_runtime_history_rows",
    "build_schema_metrics",
    "metric_batches",
]
