from __future__ import annotations

from typing import Any

from repro_core.context import ExperimentContext
from repro_core.context.checkpoint import CheckpointManager
from repro_core.context.event_executor import EventExperimentExecutor

from ..records import DS1Records
from .common import _device_timer


def create_supervised_events(
    *,
    config: dict[str, Any],
    backend: Any,
    context: ExperimentContext,
    records_sink: DS1Records,
    schedule: dict[str, Any],
    request_by_set: dict[str, Any],
    trainer_holder: dict[str, Any],
    checkpoint_manager_holder: dict[str, CheckpointManager],
    checkpoint_config: dict[str, Any],
) -> EventExperimentExecutor:
    seen_epochs: set[int] = set()

    def scheduled_requests(spec: object):
        if not isinstance(spec, dict):
            return ()
        sets = spec.get("sets", ())
        if not isinstance(sets, list | tuple):
            raise ValueError("evaluation schedule sets must be a list")
        return tuple(
            request_by_set[str(name)]
            for name in sets
            if request_by_set.get(str(name)) is not None
        )

    def evaluate_request(request):
        return trainer_holder["trainer"].evaluate(
            *request.source, metrics=request.metrics
        )

    def update_requests(event):
        update_spec = schedule.get("on_update")
        if isinstance(update_spec, dict):
            every = update_spec.get("every")
            start = int(
                update_spec.get(
                    "start",
                    1
                    if update_spec.get("first", False)
                    else int(every or event.update),
                )
            )
            stop = update_spec.get("stop")
            if stop is not None and event.update > int(stop):
                return ()
            should = event.update == start
            should |= (
                every is not None
                and event.update >= start
                and (event.update - start) % int(every) == 0
            )
            if should:
                return scheduled_requests(update_spec)
        first_epoch_spec = schedule.get("on_epoch_first_update")
        if isinstance(first_epoch_spec, dict) and event.epoch not in seen_epochs:
            seen_epochs.add(event.epoch)
            return scheduled_requests(first_epoch_spec)
        return ()

    def after_epoch(event):
        records_sink.flush()
        manager = checkpoint_manager_holder["manager"]
        manager.save_latest()
        periodic = manager.save_periodic_if_due()
        if bool(checkpoint_config.get("save_on_eval", False)) and periodic is not None:
            callback = context.metadata.get("record_eval_checkpoint")
            if callable(callback):
                callback(periodic.path)

    return EventExperimentExecutor(
        records=records_sink,
        evaluate=evaluate_request,
        update_requests=update_requests,
        epoch_requests=lambda _event: scheduled_requests(schedule.get("on_epoch_end")),
        terminal_requests=lambda _event: scheduled_requests(
            schedule.get("on_train_end")
        ),
        after_epoch=after_epoch,
        device_timer=_device_timer(config, backend),
        progress=context.metadata.get("progress_reporter"),
    )
