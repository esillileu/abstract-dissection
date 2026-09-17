from __future__ import annotations

from pathlib import Path

from deepscratch.core import Tensor
from deepscratch.profiling.backend import create_device_timer

from repro_core.context import ExperimentContext


def _mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _artifact_root(
    config: dict[str, object], context: ExperimentContext | None = None
) -> Path:
    """Return this run's artifact root, never the legacy global results root."""
    if (
        context is not None
        and (root := context.metadata.get("artifact_root")) is not None
    ):
        return Path(str(root))
    raise ValueError("experiment context is missing artifact_root")


def _apply_validation_decay(config: dict[str, object], optimizer) -> None:
    scheduler = _mapping(config, "scheduler")
    if str(scheduler.get("name", "constant")) == "validation_decay":
        optimizer.lr /= float(scheduler.get("factor", 4.0))


def _final(
    *, updates: int, epochs: int, samples: int, **values: float
) -> dict[str, float]:
    return {
        "final/status/success": 1.0,
        "final/status/nan_detected": 0.0,
        "final/status/inf_detected": 0.0,
        "final/status/diverged": 0.0,
        "final/system/total_updates": float(updates),
        "final/system/completed_epochs": float(epochs),
        "final/system/samples_seen": float(samples),
        **{key: float(value) for key, value in values.items()},
    }


def _source_curve_from_objective(config: dict[str, object]):
    """Reduce pre-update source objectives at the book's zero-based cadence.

    The accumulator and completed point stay on the active backend until the
    record sink's bulk flush, so producing a point does not synchronize here.
    """
    recording = _mapping(config, "recording")
    curve = recording.get("source_curve", {})
    if not isinstance(curve, dict):
        return lambda _event: None
    every = int(curve.get("every_updates", 0))
    if every < 1:
        return lambda _event: None
    kind = str(curve.get("kind", "interval_mean_loss"))
    reducer = str(curve.get("reducer", "mean"))
    metric = "perplexity" if kind == "train_perplexity" else "loss"
    unit = "token" if kind == "train_perplexity" else "example"
    total = None
    book_total = None
    count = 0
    unit_count = 0
    update_start = None
    epoch_start = None
    plot_index = 0
    reset_each_epoch = bool(
        _mapping(config, "policy").get("source_curve_reset_each_epoch", False)
    )
    active_epoch = None

    def reduce(event):
        nonlocal \
            total, \
            book_total, \
            count, \
            unit_count, \
            update_start, \
            epoch_start, \
            plot_index, \
            active_epoch
        if (
            reset_each_epoch
            and active_epoch is not None
            and event.epoch != active_epoch
        ):
            total, book_total, count, unit_count = None, None, 0, 0
            update_start, epoch_start = None, None
        active_epoch = event.epoch
        if update_start is None:
            update_start = event.update
            epoch_start = event.epoch
        weight = (
            int(event.unit_count)
            if reducer in {"token_weighted_mean", "exp_token_weighted_mean"}
            else 1
        )
        weighted_objective = event.objective.data * weight
        total = weighted_objective if total is None else total + weighted_objective
        if event.book_objective is not None:
            weighted_book = event.book_objective.data * weight
            book_total = (
                weighted_book if book_total is None else book_total + weighted_book
            )
        count += 1
        unit_count += int(event.unit_count)
        if event.local_iteration % every != 0:
            return None
        denominator = (
            unit_count
            if reducer in {"token_weighted_mean", "exp_token_weighted_mean"}
            else count
        )
        value = Tensor(total / denominator, backend=event.objective.backend)
        if kind == "train_perplexity":
            value = Tensor(
                event.objective.backend.xp.exp(value.data),
                backend=event.objective.backend,
            )
        point = {
            "series_id": kind,
            "plot_index": plot_index,
            "update_start": update_start,
            "update_end": event.update,
            "epoch_start": epoch_start,
            "epoch_end": event.epoch,
            "unit": unit,
            "unit_count": unit_count,
            "metric": metric,
            "reducer": reducer,
            "value": value,
        }
        if book_total is not None:
            point["book_value"] = Tensor(
                book_total / denominator,
                backend=event.book_objective.backend,
            )
        total, book_total, count, unit_count = None, None, 0, 0
        update_start, epoch_start = None, None
        plot_index += 1
        return point

    return reduce


def _device_timer(config: dict[str, object], backend):
    profiling = _mapping(config, "profiling")
    return create_device_timer(
        backend, enabled=bool(profiling.get("device_timing", False))
    )


def _recorded_float(value: object) -> float:
    if hasattr(value, "backend") and hasattr(value, "data"):
        return value.backend.scalar_to_float(value.data)
    return float(value)


def _backend_exp_float(backend, value: object) -> float:
    result = backend.xp.exp(backend.xp.asarray(_recorded_float(value)))
    return backend.scalar_to_float(result)
