import pytest

from f2.execution.identity import ConditionIdentity


def identity(**overrides):
    values = {
        "suite_id": "w2v1",
        "experiment_id": "e01",
        "condition_id": "table2",
        "seed": 1,
        "config_digest": "config-sha256",
        "plan_revision": 1,
        "planned_run_slot_id": "w2v1-r1-slot-1",
    }
    values.update(overrides)
    return ConditionIdentity(**values)


def test_tracked_run_requires_complete_immutable_identity():
    assert identity().tags()["f2.planned_run_slot_id"] == "w2v1-r1-slot-1"
    with pytest.raises(ValueError, match="config_digest"):
        identity(config_digest="")


def test_tracked_run_identity_has_no_synthetic_attempt_dimension():
    tags = identity().tags()
    assert "f2.attempt" not in tags
    assert "f2.predecessor_run_id" not in tags
