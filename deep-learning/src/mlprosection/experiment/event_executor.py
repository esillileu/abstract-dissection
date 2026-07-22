"""Domain-neutral lifecycle for executor-owned trainer observations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, TypeVar

from mlprosection.events import EpochEvent, SourceObjectiveSample, TrainEndEvent, TrainingWindowEvent, UpdateEvent
from mlprosection.experiment.progress import NullProgressReporter, ProgressReporter
from mlprosection.profiling.backend import DeviceTimer, DeviceTimingToken, NullDeviceTimer

@dataclass(frozen=True)
class EvaluationRequest:
    evaluation_set_id: str
    split: str
    source: object
    metrics: tuple[str, ...]


class EventRecordSink(Protocol):
    """Storage projection supplied by an experiment domain."""

    def on_update(self, event: UpdateEvent) -> None: ...
    def on_epoch(self, event: EpochEvent) -> None: ...
    def on_train_end(self, event: TrainEndEvent) -> None: ...
    def add_evaluation(self, *, axis: str, axis_step: int, update: int, epoch: int,
                       evaluation_set_id: str, split: str, result: object) -> None: ...
    def add_timing_window(self, event: TrainingWindowEvent) -> None: ...


RecordSink = TypeVar("RecordSink", bound=EventRecordSink)


class EventExperimentExecutor:
    """Route trainer facts to a domain-provided schedule and record sink.

    This class deliberately has no DS1/DS2 schema, metric name, CSV or artifact
    knowledge.  Domains own those projections through ``records``.
    """

    def __init__(
        self,
        *,
        records: RecordSink,
        evaluate: Callable[[EvaluationRequest], object],
        update_requests: Callable[[UpdateEvent], tuple[EvaluationRequest, ...]] | None = None,
        epoch_requests: Callable[[EpochEvent], tuple[EvaluationRequest, ...]] | None = None,
        terminal_requests: Callable[[TrainEndEvent], tuple[EvaluationRequest, ...]] | None = None,
        source_curve: Callable[[SourceObjectiveSample], dict[str, object] | None] | None = None,
        after_evaluation: Callable[[EvaluationRequest, object, str, int], None] | None = None,
        after_epoch: Callable[[EpochEvent], None] | None = None,
        device_timer: DeviceTimer | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.records = records
        self._evaluate = evaluate
        self._update_requests = update_requests or (lambda _event: ())
        self._epoch_requests = epoch_requests or (lambda _event: ())
        self._terminal_requests = terminal_requests or (lambda _event: ())
        self._source_curve = source_curve or (lambda _event: None)
        self._after_evaluation = after_evaluation or (lambda _request, _result, _axis, _step: None)
        self._after_epoch = after_epoch or (lambda _event: None)
        self._device_timer = device_timer or NullDeviceTimer()
        self._progress = progress or NullProgressReporter()
        self._window_start_update: int | None = None
        self._window_started_ns: int | None = None
        self._window_device_token: DeviceTimingToken | None = None

    def begin(self, *, start_update: int = 1) -> None:
        self._window_start_update = start_update
        self._window_started_ns = time.perf_counter_ns()
        self._window_device_token = self._device_timer.start()

    def run(self, train: Callable[[], None], *, start_update: int = 1) -> RecordSink:
        """Run a prepared Trainer through the common event lifecycle.

        Domain adapters build the model, trainer and train closure; this method
        is intentionally independent of Word2Vec/LM/Seq2seq data shapes.
        """
        self.begin(start_update=start_update)
        try:
            train()
        finally:
            flush = getattr(self.records, "flush", None)
            if callable(flush):
                flush()
        return self.records

    def on_update(self, event: UpdateEvent) -> None:
        self.records.on_update(event)
        self._progress.on_update(event)
        requests = self._update_requests(event)
        if requests:
            self._close_window(end_update=event.update, closed_by="probe", requests=requests, epoch=event.epoch)

    def on_source_objective(self, event: SourceObjectiveSample) -> None:
        receiver = getattr(self.records, "on_source_objective", None)
        if callable(receiver):
            receiver(event)
        point = self._source_curve(event)
        if point is not None:
            receiver = getattr(self.records, "add_source_curve", None)
            if callable(receiver):
                receiver(point)

    def on_epoch(self, event: EpochEvent) -> None:
        self.records.on_epoch(event)
        self._progress.on_epoch(event)
        requests = self._epoch_requests(event)
        self._close_window(
            end_update=event.end_update,
            closed_by="epoch_end",
            requests=requests,
            epoch=event.epoch,
        )
        flush = getattr(self.records, "flush", None)
        if callable(flush):
            flush()
        self._after_epoch(event)

    def on_train_end(self, event: TrainEndEvent) -> None:
        self.records.on_train_end(event)
        self._progress.on_train_end(event)
        requests = self._terminal_requests(event)
        if self._window_start_update is not None and self._window_start_update <= event.update:
            self._close_window(end_update=event.update, closed_by="terminal", requests=requests, epoch=event.epoch)
        else:
            self._record_evaluations(requests, axis="terminal", axis_step=event.update, update=event.update, epoch=event.epoch)

    def _close_window(
        self,
        *,
        end_update: int,
        closed_by: Literal["probe", "epoch_end", "terminal"],
        requests: tuple[EvaluationRequest, ...],
        epoch: int,
    ) -> None:
        if self._window_start_update is None or self._window_started_ns is None:
            self.begin(start_update=end_update + 1)
            return
        if self._window_start_update > end_update:
            self._record_evaluations(requests, axis="epoch" if closed_by == "epoch_end" else "update", axis_step=epoch if closed_by == "epoch_end" else end_update, update=end_update, epoch=epoch)
            return
        train_device_token = self._window_device_token
        self._device_timer.stop(train_device_token)
        train_ns = time.perf_counter_ns() - self._window_started_ns
        evaluate_started = time.perf_counter_ns()
        eval_device_token = self._device_timer.start() if requests else None
        axis = "epoch" if closed_by == "epoch_end" else "update"
        self._record_evaluations(requests, axis=axis, axis_step=epoch if axis == "epoch" else end_update, update=end_update, epoch=epoch)
        if eval_device_token is not None:
            self._device_timer.stop(eval_device_token)
        eval_ns = time.perf_counter_ns() - evaluate_started if requests else None
        train_device_ns = self._device_timer.elapsed_ns(train_device_token)
        eval_device_ns = self._device_timer.elapsed_ns(eval_device_token)
        self.records.add_timing_window(TrainingWindowEvent(
            start_update=self._window_start_update, end_update=end_update,
            update_count=end_update - self._window_start_update + 1,
            closed_by=closed_by, train_wall_time_ns=train_ns,
            eval_wall_time_ns=eval_ns,
            train_device_time_ns=train_device_ns,
            eval_device_time_ns=eval_device_ns,
        ))
        self.begin(start_update=end_update + 1)

    def _record_evaluations(self, requests, *, axis, axis_step, update, epoch) -> None:
        for request in requests:
            result = self._evaluate(request)
            self._after_evaluation(request, result, axis, axis_step)
            self.records.add_evaluation(
                axis=axis, axis_step=axis_step, update=update, epoch=epoch,
                evaluation_set_id=request.evaluation_set_id, split=request.split,
                result=result,
            )
