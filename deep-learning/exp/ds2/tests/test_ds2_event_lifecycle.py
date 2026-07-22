from __future__ import annotations

from mlprosection import Tensor
from mlprosection.events import EpochEvent, EvaluationResult, SourceObjectiveSample, TrainEndEvent, UpdateEvent
from mlprosection.experiment.event_executor import EvaluationRequest, EventExperimentExecutor

from exp.ds2.records import DS2Records


def test_ds2_records_long_metrics_and_source_samples() -> None:
    request = EvaluationRequest("ptb-valid", "valid", object(), ("perplexity",))
    executor = EventExperimentExecutor(
        records=DS2Records(),
        evaluate=lambda _request: EvaluationResult(
            example_count=12, loss=1.2, accuracy=None, unit="token", unit_count=12,
            perplexity=3.32,
        ),
        epoch_requests=lambda _event: (request,),
        source_curve=lambda event: {
            "series_id": "book-ppl", "plot_index": event.local_iteration,
            "update_end": event.update, "value": event.objective,
        },
    )
    executor.begin()
    executor.on_source_objective(SourceObjectiveSample(1, 1, 0, Tensor(1.5), 20))
    executor.on_update(UpdateEvent(1, 1, 2, Tensor(1.4), 0.1))
    executor.on_epoch(EpochEvent(1, 1, 1, 2))
    executor.on_train_end(TrainEndEvent("completed", 1, 1))

    assert executor.records.source_samples[0]["local_iteration"] == 0
    assert executor.records.source_curves[0]["series_id"] == "book-ppl"
    assert {row["metric"] for row in executor.records.evaluations} == {"loss", "perplexity"}
