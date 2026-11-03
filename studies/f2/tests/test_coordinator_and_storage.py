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


def test_coordinator_rejects_publishing_or_resuming_mismatched_config_digest(tmp_path):
    """P0-1: Mismatched config digest in local result/checkpoint must not be published or resumed."""
    import json

    from repro_core.context.paths import RuntimePaths

    paths = RuntimePaths.from_environment(tmp_path)
    coordinator = F2ExecutionCoordinator(tracking_uri="http://fake-mlflow", paths=paths)
    coordinator.publication = MagicMock()
    coordinator.publication.find_matching_finished_run.return_value = None

    slot_dir = paths.runs("f2", "w2v1", "slot-1")
    slot_dir.mkdir(parents=True, exist_ok=True)

    # Local result with different config_digest
    (slot_dir / "result.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity": {
                    "config_digest": "old_digest",
                    "seed": 1,
                    "planned_run_slot_id": "slot-1",
                },
                "artifacts": {"lookup": "lookup"},
            }
        )
    )
    (slot_dir / "lookup").mkdir(parents=True, exist_ok=True)

    condition_new = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow",
        seed=1,
        config_digest="new_digest_456",
        planned_run_slot_id="slot-1",
    )

    decision = coordinator.decide(
        condition_new,
        experiment_name="f2.w2v1",
        override=False,
    )
    # Must NOT publish the old result; must start a fresh RUN
    assert decision.kind == ExecutionDecisionKind.RUN


def test_publication_preflight_validation_failures(tmp_path):
    """P0-2: PublicationService must validate result completeness, identity, and required artifacts before run creation."""
    import json

    import pytest

    from f2.storage.publication import PublicationService

    service = PublicationService(tracking_uri="http://fake-mlflow")
    service.client = MagicMock()

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow",
        seed=1,
        config_digest="digest_correct",
        planned_run_slot_id="slot-1",
    )
    cfg = {"identity": {"seed": 1}, "atomic_run_id": "cbow"}

    # Case 1: missing result.json
    with pytest.raises(ValueError, match=r"missing result\.json"):
        service.publish_run("exp", condition, run_dir, cfg)

    # Case 2: incomplete result.json (complete=False)
    res_path = run_dir / "result.json"
    res_path.write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="complete=False"):
        service.publish_run("exp", condition, run_dir, cfg)

    # Case 3: mismatched config_digest
    res_path.write_text(
        json.dumps(
            {
                "complete": True,
                "identity": {"config_digest": "different_digest", "seed": 1},
                "artifacts": {"lookup": "lookup"},
            }
        )
    )
    with pytest.raises(ValueError, match="config_digest mismatch"):
        service.publish_run("exp", condition, run_dir, cfg)

    # Case 4: missing designated lookup artifact
    res_path.write_text(
        json.dumps(
            {
                "complete": True,
                "identity": {"config_digest": "digest_correct", "seed": 1},
                "artifacts": {"lookup": "lookup"},
            }
        )
    )
    with pytest.raises(ValueError, match="designated lookup does not exist"):
        service.publish_run("exp", condition, run_dir, cfg)

    # None of these attempts should create an MLflow run
    service.client.create_run.assert_not_called()


def test_attempt_discovery_and_resumption_after_override(tmp_path):
    """P1-3: When an override attempt is interrupted, re-running resumes from that attempt's checkpoint."""
    import json

    from repro_core.context.paths import RuntimePaths

    paths = RuntimePaths.from_environment(tmp_path)
    coordinator = F2ExecutionCoordinator(tracking_uri="http://fake-mlflow", paths=paths)
    coordinator.publication = MagicMock()
    coordinator.publication.find_matching_finished_run.return_value = None

    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow",
        seed=1,
        config_digest="digest123",
        planned_run_slot_id="slot-1",
    )

    slot_dir = paths.runs("f2", "w2v1", "slot-1")
    run_01 = slot_dir / "run_01"
    run_02 = slot_dir / "run_02"
    run_01.mkdir(parents=True, exist_ok=True)
    run_02.mkdir(parents=True, exist_ok=True)

    # Attempt 1 (run_01): completed result
    (run_01 / "result.json").write_text(
        json.dumps(
            {
                "complete": True,
                "identity": {
                    "config_digest": "digest123",
                    "seed": 1,
                    "planned_run_slot_id": "slot-1",
                },
                "artifacts": {"lookup": "lookup"},
            }
        )
    )
    (run_01 / "lookup").mkdir()

    # Attempt 2 (run_02): interrupted at epoch 2 checkpoint
    chk_dir = run_02 / "checkpoints" / "periodic-epoch-0002"
    chk_dir.mkdir(parents=True, exist_ok=True)
    (chk_dir / "manifest.json").write_text(json.dumps({"completed_epochs": 2}))
    (run_02 / "checkpoints" / "latest.json").write_text(
        json.dumps({"path": "periodic-epoch-0002", "generation": 2})
    )
    (run_02 / "config.json").write_text(
        json.dumps(
            {
                "identity": {
                    "config_digest": "digest123",
                    "seed": 1,
                    "planned_run_slot_id": "slot-1",
                }
            }
        )
    )

    # Without override, it should discover attempt 2 and resume it rather than picking attempt 1 or creating run_03
    decision = coordinator.decide(
        condition,
        experiment_name="f2.w2v1",
        override=False,
    )
    assert decision.kind == ExecutionDecisionKind.RESUME
    assert decision.run_dir == run_02
    assert decision.resume_epoch == 2
