from __future__ import annotations

from mlprosection import Tensor
from mlprosection.events import EvaluationResult, UpdateEvent
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor
from exp.ds1.records import DS1Records


def test_event_executor_uses_one_lifecycle_for_ds1_wide_evaluations() -> None:
    request = EvaluationRequest("mnist-test", "test", object(), ("loss", "accuracy"))
    executor = EventExperimentExecutor(
        records=DS1Records(),
        evaluate=lambda _request: EvaluationResult(example_count=10, loss=0.2, accuracy=0.9),
        update_requests=lambda event: (request,) if event.update == 1 else (),
    )
    event = UpdateEvent(1, 1, 2, Tensor(0.3), 0.1)
    executor.run(lambda: executor.on_update(event))

    assert executor.records.updates[0]["update"] == 1
    assert executor.records.evaluations == [{
        "axis": "update", "axis_step": 1, "update": 1, "epoch": 1,
        "evaluation_set_id": "mnist-test", "split": "test", "example_count": 10,
        "loss": 0.2, "accuracy": 0.9,
    }]
    assert executor.records.timing_windows[0].closed_by == "probe"
