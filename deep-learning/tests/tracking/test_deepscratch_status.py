from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient

from exp.framework.execution import RunPlan
from exp.deepscratch.identity import Variant, Volume
from exp.deepscratch.execution.status import inspect_plan_status


def _uri(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve() / 'mlflow.db'}"


def _plan(condition: str, seed: int = 1) -> RunPlan:
    return RunPlan(
        domain="ds2",
        experiment_id="e01",
        path=Path("e01.yaml"),
        atomic_run_id=condition,
        seed=seed,
        device="cpu",
    )


def _run(
    client: MlflowClient,
    experiment_id: str,
    condition: str,
    *,
    status: str,
    start_time: int,
    variant: str | None = None,
    disposition: str | None = None,
    protocol_version: str = "legacy",
) -> str:
    tags = {
        "run.type": "seed_trial",
        "experiment.ids": "e01",
        "atomic_run.id": condition,
        "master_seed": "1",
        "protocol.version": protocol_version,
    }
    if variant is not None:
        tags["implementation.variant"] = variant
    if disposition is not None:
        tags["transfer.import.disposition"] = disposition
    run = client.create_run(
        experiment_id,
        start_time=start_time,
        tags=tags,
    )
    if status != "RUNNING":
        client.set_terminated(run.info.run_id, status=status)
    return run.info.run_id


def test_plan_status_combines_new_and_legacy_attempts(tmp_path: Path) -> None:
    client = MlflowClient(_uri(tmp_path))
    new_id = client.create_experiment("deepscratch.ds2")
    legacy_id = client.create_experiment("ds2")
    completed_id = _run(
        client,
        legacy_id,
        "COMPLETE",
        status="FINISHED",
        start_time=10,
    )
    _run(
        client,
        new_id,
        "COMPLETE",
        status="RUNNING",
        start_time=20,
        variant="implemented",
    )
    running_id = _run(
        client,
        new_id,
        "RUNNING",
        status="RUNNING",
        start_time=30,
        variant="implemented",
    )
    failed_id = _run(
        client,
        legacy_id,
        "FAILED",
        status="FAILED",
        start_time=40,
    )
    _run(
        client,
        legacy_id,
        "MISSING",
        status="FINISHED",
        start_time=50,
        disposition="imported-alternate",
    )
    _run(
        client,
        new_id,
        "WRONG-VARIANT",
        status="FINISHED",
        start_time=60,
        variant="original",
    )

    report = inspect_plan_status(
        client,
        [
            _plan("COMPLETE"),
            _plan("RUNNING"),
            _plan("FAILED"),
            _plan("MISSING"),
            _plan("WRONG-VARIANT"),
        ],
        volume=Volume.DS2,
        variant=Variant.IMPLEMENTED,
    )

    assert report.counts == {
        "completed": 1,
        "running": 1,
        "failed": 1,
        "missing": 2,
    }
    by_condition = {entry.condition_id: entry for entry in report.entries}
    assert by_condition["COMPLETE"].run_id == completed_id
    assert by_condition["COMPLETE"].attempt_count == 2
    assert by_condition["RUNNING"].run_id == running_id
    assert by_condition["FAILED"].run_id == failed_id
    assert by_condition["MISSING"].attempt_count == 0


def test_plan_status_matches_new_condition_and_experiment_tags(
    tmp_path: Path,
) -> None:
    client = MlflowClient(_uri(tmp_path))
    experiment_id = client.create_experiment("deepscratch.ds1")
    run = client.create_run(
        experiment_id,
        tags={
            "run.type": "seed_trial",
            "experiment.id": "e01",
            "condition.id": "ADAM",
            "master_seed": "2",
            "implementation.variant": "implemented",
        },
    )
    client.set_terminated(run.info.run_id)

    report = inspect_plan_status(
        client,
        [RunPlan("ds1", "e01", Path("e01.yaml"), "ADAM", 2, "cpu")],
        volume=Volume.DS1,
        variant=Variant.IMPLEMENTED,
    )

    assert report.counts["completed"] == 1
    assert report.entries[0].namespace == "deepscratch.ds1"


def test_plan_status_rejects_attempts_from_another_protocol(
    tmp_path: Path,
) -> None:
    client = MlflowClient(_uri(tmp_path))
    experiment_id = client.create_experiment("ds2")
    _run(
        client,
        experiment_id,
        "RNNLM",
        status="FINISHED",
        start_time=1,
        protocol_version="legacy",
    )

    report = inspect_plan_status(
        client,
        [_plan("RNNLM")],
        volume=Volume.DS2,
        variant=Variant.IMPLEMENTED,
        expected_protocols={("e01", "RNNLM"): "book-source-v1"},
    )

    assert report.counts["missing"] == 1


def test_original_legacy_protocol_maps_to_declared_book_protocol(
    tmp_path: Path,
) -> None:
    client = MlflowClient(_uri(tmp_path))
    experiment_id = client.create_experiment("ds2_original")
    _run(
        client,
        experiment_id,
        "RNNLM",
        status="FINISHED",
        start_time=1,
        protocol_version="legacy",
    )

    report = inspect_plan_status(
        client,
        [_plan("RNNLM")],
        volume=Volume.DS2,
        variant=Variant.ORIGINAL,
        expected_protocols={("e01", "RNNLM"): "book-source-v1"},
    )

    assert report.counts["completed"] == 1
