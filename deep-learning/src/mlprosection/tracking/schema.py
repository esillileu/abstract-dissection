from __future__ import annotations

from typing import Any


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
    """Map trainer/profiling metrics to MLflow schema v1 names."""

    metrics: dict[str, float] = {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": float(total_updates),
        "final/system/completed_epochs": float(completed_epochs),
        "final/system/samples_seen": float(samples_seen),
    }

    _set_if_present(metrics, "final/train/loss", train_loss)
    _set_if_present(metrics, "final/test/loss", test_loss)
    _set_if_present(metrics, "final/train/accuracy", train_accuracy)
    _set_if_present(metrics, "final/test/accuracy", test_accuracy)
    _set_if_present(
        metrics,
        "runtime/train_total_s",
        _ms_to_s(profiling_metrics.get("runtime.train_total.mean_ms")),
    )
    _set_if_present(
        metrics,
        "memory/cpu_rss_start_bytes",
        profiling_metrics.get("memory.run.start.cpu.rss_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/cpu_rss_end_bytes",
        profiling_metrics.get("memory.run.end.cpu.rss_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/cpu_rss_peak_sampled_bytes",
        profiling_metrics.get("memory.peak_sampled.cpu.rss_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/gpu_used_start_bytes",
        profiling_metrics.get("memory.run.start.gpu.pool_used_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/gpu_used_end_bytes",
        profiling_metrics.get("memory.run.end.gpu.pool_used_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/gpu_used_peak_sampled_bytes",
        profiling_metrics.get("memory.peak_sampled.gpu.pool_used_bytes"),
    )
    _set_if_present(
        metrics,
        "memory/gpu_reserved_peak_sampled_bytes",
        profiling_metrics.get("memory.peak_sampled.gpu.pool_reserved_bytes"),
    )

    train_total_s = metrics.get("runtime/train_total_s")
    for source, target in {
        "forward": "forward",
        "backward": "backward",
        "optimizer_update": "update",
        "gradient_clip": "gradient_clip",
        "train_step": "train_step",
    }.items():
        _add_phase_metrics(
            metrics,
            profiling_metrics,
            source=source,
            target=target,
            train_total_s=train_total_s,
        )

    if "profile/gradient_clip/count" not in metrics:
        metrics["profile/gradient_clip/count"] = 0.0
        metrics["profile/gradient_clip/total_s"] = 0.0

    return metrics


def build_epoch_metric_rows(
    *,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
    profiling_metrics: dict[str, int | float],
) -> list[tuple[str, int, str, float]]:
    """Create schema v1 epoch metric rows for artifact CSVs."""

    rows: list[tuple[str, int, str, float]] = []
    for epoch, value in enumerate(train_accuracies):
        rows.append(("epoch", epoch, "train/accuracy", float(value)))
    for epoch, value in enumerate(test_accuracies):
        rows.append(("epoch", epoch, "test/accuracy", float(value)))

    for epoch, value in enumerate(_tail_epoch_values(train_losses, len(train_accuracies))):
        rows.append(("epoch", epoch, "train/loss", float(value)))
    for epoch, value in enumerate(_tail_epoch_values(test_losses, len(test_accuracies))):
        rows.append(("epoch", epoch, "test/loss", float(value)))

    for name, value in profiling_metrics.items():
        prefix = "throughput.epoch."
        if name.startswith(prefix) and name.endswith("_samples_per_s"):
            parts = name.split(".")
            epoch = int(parts[2])
            split = parts[3].replace("_samples_per_s", "")
            rows.append(("epoch", epoch, f"runtime/{split}_samples_per_s", float(value)))

    return rows


def build_runtime_history_rows(
    profiling_metrics: dict[str, int | float],
) -> list[dict[str, Any]]:
    """Build epoch runtime rows from trainer profiling metrics."""

    epochs: dict[int, dict[str, Any]] = {}
    for name, value in profiling_metrics.items():
        parts = name.split(".")
        if len(parts) != 5 or parts[:2] != ["runtime", "epoch"]:
            continue
        if not parts[2].isdigit():
            continue

        epoch = int(parts[2])
        field = parts[3]
        row = epochs.setdefault(
            epoch,
            {
                "step_type": "epoch",
                "step": epoch,
                "train_s": "",
                "eval_s": "",
                "checkpoint_s": "",
                "throughput_samples_per_s": "",
            },
        )
        if field == "train_duration_ms":
            row["train_s"] = float(value) / 1_000
        elif field == "eval_duration_ms":
            row["eval_s"] = float(value) / 1_000

    for name, value in profiling_metrics.items():
        prefix = "throughput.epoch."
        if name.startswith(prefix) and name.endswith(".train_samples_per_s"):
            epoch = int(name.split(".")[2])
            row = epochs.setdefault(
                epoch,
                {
                    "step_type": "epoch",
                    "step": epoch,
                    "train_s": "",
                    "eval_s": "",
                    "checkpoint_s": "",
                    "throughput_samples_per_s": "",
                },
            )
            row["throughput_samples_per_s"] = float(value)

    return [epochs[key] for key in sorted(epochs)]


def build_memory_history_rows(
    profiling_metrics: dict[str, int | float],
) -> list[dict[str, Any]]:
    """Build coarse memory history rows from memory snapshot metrics."""

    rows = []
    snapshot_names = ("run.start", "train.start", "train.end", "run.end")
    for index, snapshot in enumerate(snapshot_names):
        rows.append(
            {
                "timestamp_s": float(index),
                "cpu_rss_bytes": profiling_metrics.get(
                    f"memory.{snapshot}.cpu.rss_bytes",
                    "",
                ),
                "gpu_used_bytes": profiling_metrics.get(
                    f"memory.{snapshot}.gpu.pool_used_bytes",
                    "",
                ),
                "gpu_reserved_bytes": profiling_metrics.get(
                    f"memory.{snapshot}.gpu.pool_reserved_bytes",
                    "",
                ),
            }
        )
    return rows


def _add_phase_metrics(
    output: dict[str, float],
    profiling_metrics: dict[str, int | float],
    *,
    source: str,
    target: str,
    train_total_s: float | None,
) -> None:
    prefix = f"runtime.profile.{source}."
    count = profiling_metrics.get(f"{prefix}count")
    mean_ms = profiling_metrics.get(f"{prefix}mean_ms")
    total_s = None
    if count is not None and mean_ms is not None:
        total_s = float(count) * float(mean_ms) / 1_000
        output[f"profile/{target}/total_s"] = total_s

    mapping = {
        "count": "count",
        "mean_ms": "mean_s",
        "p50_ms": "median_s",
        "p95_ms": "p95_s",
        "std_ms": "std_s",
        "min_ms": "min_s",
        "max_ms": "max_s",
    }
    for source_suffix, target_suffix in mapping.items():
        value = profiling_metrics.get(f"{prefix}{source_suffix}")
        if value is None:
            continue
        if source_suffix == "count":
            output[f"profile/{target}/{target_suffix}"] = float(value)
        else:
            output[f"profile/{target}/{target_suffix}"] = float(value) / 1_000

    if total_s is not None and train_total_s:
        output[f"profile/{target}/fraction_of_train_time"] = total_s / train_total_s


def _set_if_present(
    metrics: dict[str, float],
    name: str,
    value: int | float | None,
) -> None:
    if value is not None:
        metrics[name] = float(value)


def _ms_to_s(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000


def _tail_epoch_values(values: list[float], epoch_count: int) -> list[float]:
    if epoch_count == 0:
        return []
    return values[-epoch_count:]
