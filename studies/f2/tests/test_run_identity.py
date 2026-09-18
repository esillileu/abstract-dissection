import pytest

from f2.run_identity import RunIdentity


def identity(**overrides):
    values = {
        "planned_run_slot_id": "w2v1-r1-slot-1",
        "plan_revision": 1,
        "config_digest": "config-sha256",
        "resource_version_id": "wmt-normalized-v1",
        "resource_manifest_digest": "manifest-sha256",
    }
    values.update(overrides)
    return RunIdentity(**values)


def test_tracked_run_requires_complete_immutable_identity():
    assert identity().tags()["f2.planned_run_slot_id"] == "w2v1-r1-slot-1"
    with pytest.raises(ValueError, match="config_digest"):
        identity(config_digest="")


def test_resume_is_a_new_attempt_with_predecessor_lineage():
    resumed = identity(attempt=2, predecessor_run_id="previous-run")
    assert resumed.tags()["f2.predecessor_run_id"] == "previous-run"
    with pytest.raises(ValueError, match="predecessor"):
        identity(attempt=2)
    with pytest.raises(ValueError, match="first attempt"):
        identity(predecessor_run_id="unexpected")
