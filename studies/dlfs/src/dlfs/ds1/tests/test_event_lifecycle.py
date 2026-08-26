from __future__ import annotations

from deepscratch.core import Tensor

from dlfs.ds1.implemented.records import DS1Records
from repro_core.context.event_executor import EvaluationRequest, EventExperimentExecutor
from repro_core.context.events import EpochEvent, EvaluationResult, UpdateEvent


def test_event_executor_uses_one_lifecycle_for_ds1_wide_evaluations() -> None:
    request = EvaluationRequest("mnist-test", "test", object(), ("loss", "accuracy"))
    executor = EventExperimentExecutor(
        records=DS1Records(),
        evaluate=lambda _request: EvaluationResult(
            example_count=10, loss=0.2, accuracy=0.9
        ),
        update_requests=lambda event: (request,) if event.update == 1 else (),
    )
    event = UpdateEvent(1, 1, 2, Tensor(0.3), 0.1)
    executor.run(lambda: executor.on_update(event))

    assert executor.records.updates[0]["update"] == 1
    assert executor.records.evaluations == [
        {
            "axis": "update",
            "axis_step": 1,
            "update": 1,
            "epoch": 1,
            "evaluation_set_id": "mnist-test",
            "split": "test",
            "example_count": 10,
            "loss": 0.2,
            "accuracy": 0.9,
        }
    ]
    assert executor.records.timing_windows[0].closed_by == "probe"
    assert executor.records.timing_windows[0].train_device_time_ns is None
    assert executor.records.timing_windows[0].eval_device_time_ns is None


def test_event_executor_records_device_timing_at_window_boundaries_only() -> None:
    class SpyTimer:
        def __init__(self) -> None:
            self.starts = 0
            self.stops = 0
            self.elapsed_calls = 0

        def start(self):
            self.starts += 1
            return object()

        def stop(self, _token) -> None:
            self.stops += 1

        def elapsed_ns(self, _token) -> int:
            self.elapsed_calls += 1
            return self.elapsed_calls * 1_000

    request = EvaluationRequest("mnist-test", "test", object(), ("loss", "accuracy"))
    timer = SpyTimer()
    executor = EventExperimentExecutor(
        records=DS1Records(),
        evaluate=lambda _request: EvaluationResult(
            example_count=10, loss=0.2, accuracy=0.9
        ),
        update_requests=lambda event: (request,) if event.update == 3 else (),
        device_timer=timer,
    )

    executor.begin()
    executor.on_update(UpdateEvent(1, 1, 2, Tensor(0.5), 0.1))
    executor.on_update(UpdateEvent(2, 1, 2, Tensor(0.4), 0.1))
    assert timer.elapsed_calls == 0
    executor.on_update(UpdateEvent(3, 1, 2, Tensor(0.3), 0.1))

    window = executor.records.timing_windows[0]
    assert window.train_device_time_ns == 1_000
    assert window.eval_device_time_ns == 2_000
    assert timer.starts == 3
    assert timer.stops == 2


def test_event_executor_leaves_eval_timing_empty_without_evaluation_request() -> None:
    class ConstantTimer:
        def start(self):
            return object()

        def stop(self, _token) -> None:
            return None

        def elapsed_ns(self, token) -> int | None:
            return None if token is None else 42

    executor = EventExperimentExecutor(
        records=DS1Records(),
        evaluate=lambda _request: None,
        device_timer=ConstantTimer(),
    )

    executor.begin()
    executor.on_update(UpdateEvent(1, 1, 2, Tensor(0.3), 0.1))
    executor.on_epoch(EpochEvent(epoch=1, start_update=1, end_update=1, sample_count=2))

    window = executor.records.timing_windows[0]
    assert window.train_device_time_ns == 42
    assert window.eval_wall_time_ns is None
    assert window.eval_device_time_ns is None


def test_event_executor_flushes_records_when_train_raises() -> None:
    class Records(DS1Records):
        def __init__(self) -> None:
            super().__init__()
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1

    records = Records()
    executor = EventExperimentExecutor(records=records, evaluate=lambda _request: None)

    try:
        executor.run(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected train exception")

    assert records.flush_count == 1
