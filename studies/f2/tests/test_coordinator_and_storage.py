"""Tests for F2 Coordinator and Storage components."""

from unittest.mock import MagicMock

from f2.execution.coordinator import (
    ExecutionDecisionKind,
    F2ExecutionCoordinator,
)
from f2.execution.identity import ConditionIdentity


def test_coordinator_decide_skip_when_finished_exists(tmp_path):
    coordinator = F2ExecutionCoordinator(tracking_uri="http://fake-mlflow")
    coordinator.publication = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    coordinator.publication.find_matching_finished_run.return_value = mock_run

    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow",
        seed=1,
        config_digest="digest123",
        planned_run_slot_id="slot-1",
    )

    decision = coordinator.decide(
        condition,
        experiment_name="f2.w2v1",
        override=False,
    )
    assert decision.kind == ExecutionDecisionKind.SKIP
    assert decision.existing_run_id == "test-run-123"


def test_coordinator_decide_override_ignores_finished(tmp_path):
    coordinator = F2ExecutionCoordinator(tracking_uri="http://fake-mlflow")
    coordinator.publication = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    coordinator.publication.find_matching_finished_run.return_value = mock_run

    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow",
        seed=1,
        config_digest="digest123",
        planned_run_slot_id="slot-1",
    )

    decision = coordinator.decide(
        condition,
        experiment_name="f2.w2v1",
        override=True,
    )
    # With override=True, skip check is bypassed and inspect_run is called
    assert decision.kind == ExecutionDecisionKind.RUN
