"""Stable metric names shared by experiment executors and optional sinks."""

from __future__ import annotations


def build_final_metrics(
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
    for name, value in (
        ("final/train/loss", train_loss),
        ("final/test/loss", test_loss),
        ("final/train/accuracy", train_accuracy),
        ("final/test/accuracy", test_accuracy),
    ):
        if value is not None:
            metrics[name] = float(value)
    return metrics


def epoch_history(
    *,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
) -> list[tuple[str, int, str, float]]:
    rows = [
        ("epoch", index, "train/accuracy", float(value))
        for index, value in enumerate(train_accuracies)
    ]
    rows += [
        ("epoch", index, "test/accuracy", float(value))
        for index, value in enumerate(test_accuracies)
    ]
    rows += [
        ("epoch", index, "train/loss", float(value))
        for index, value in enumerate(train_losses[-len(train_accuracies) :])
    ]
    rows += [
        ("epoch", index, "test/loss", float(value))
        for index, value in enumerate(test_losses[-len(test_accuracies) :])
    ]
    return rows


def update_history(
    *, train_logs: list[dict[str, float | int | None]]
) -> list[tuple[str, int, str, float]]:
    """Project interval-aggregated training losses onto the global update axis."""
    rows: list[tuple[str, int, str, float]] = []
    for log in train_logs:
        global_step = log.get("global_step")
        loss = log.get("loss")
        if not isinstance(global_step, int | float) or not isinstance(
            loss, int | float
        ):
            raise ValueError("train interval logs require numeric global_step and loss")
        rows.append(("update", int(global_step), "train/loss", float(loss)))
    return rows


def evaluation_history(
    *, valid_logs: list[dict[str, float | int | None]]
) -> list[tuple[str, int, str, float]]:
    """Project validation intervals onto their own monotonic evaluation axis."""
    rows: list[tuple[str, int, str, float]] = []
    for log in valid_logs:
        eval_step = log.get("eval_step")
        loss = log.get("loss")
        if not isinstance(eval_step, int | float) or not isinstance(loss, int | float):
            raise ValueError(
                "validation interval logs require numeric eval_step and loss"
            )
        rows.append(("eval", int(eval_step), "valid/loss", float(loss)))
    return rows
