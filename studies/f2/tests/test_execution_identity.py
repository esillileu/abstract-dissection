"""Tests for ConditionIdentity and AttemptIdentity in f2.execution.identity."""

import pytest

from f2.execution.identity import (
    AttemptIdentity,
    ConditionIdentity,
    _compute_digest,
)


def test_condition_identity_creation_and_defaults():
    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow-d50",
        seed=42,
        config_digest="abc123def456",
    )
    assert condition.suite_id == "w2v1"
    assert condition.seed == 42
    assert condition.planned_run_slot_id == "w2v1-e01-cbow-d50-s42"
    assert condition.identity_key == "w2v1-e01-cbow-d50-s42:abc123def456"

    tags = condition.tags()
    assert tags["f2.suite_id"] == "w2v1"
    assert tags["f2.seed"] == "42"
    assert tags["f2.config_digest"] == "abc123def456"
    assert tags["f2.plan_revision"] == "1"


def test_condition_identity_validation():
    with pytest.raises(ValueError, match="suite_id"):
        ConditionIdentity(
            suite_id="",
            experiment_id="e01",
            condition_id="cbow",
            seed=1,
            config_digest="abc",
        )
    with pytest.raises(ValueError, match="config_digest"):
        ConditionIdentity(
            suite_id="w2v1",
            experiment_id="e01",
            condition_id="cbow",
            seed=1,
            config_digest="",
        )


def test_attempt_identity_run_directory():
    condition = ConditionIdentity(
        suite_id="w2v1",
        experiment_id="e01",
        condition_id="cbow-d50",
        seed=1,
        config_digest="digest-123",
        planned_run_slot_id="slot-001",
    )
    attempt = AttemptIdentity(
        condition=condition,
        attempt_id="att-1",
        created_at="2026-11-02T19:00:00Z",
    )
    assert attempt.run_directory_relpath() == "w2v1/slot-001/att-1"


def test_compute_digest_deterministic():
    d1 = {"b": 2, "a": [1, 2]}
    d2 = {"a": [1, 2], "b": 2}
    assert _compute_digest(d1) == _compute_digest(d2)
