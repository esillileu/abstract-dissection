from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from experiments.cli import build_plans
from mlprosection.experiment import load_yaml, normalize_config


def test_ds2_gt02_plan_uses_all_atomic_trials_and_seed_registry_index() -> None:
    plans = build_plans(
        domain="ds2",
        experiment_ids=["02"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="1",
        device=None,
        overrides={},
    )

    assert [plan.atomic_run_id for plan in plans] == [
        "W2V-PTB-CBOW-NS",
        "W2V-PTB-SKIPGRAM-NS",
        "W2V-PTB-CBOW-FULL",
        "W2V-PTB-SKIPGRAM-FULL",
    ]
    assert {plan.seed for plan in plans} == {2}
    assert {plan.device for plan in plans} == {"cpu"}


def test_ds2_catalog_matches_declared_training_groups() -> None:
    plans = build_plans(
        domain="ds2",
        experiment_ids=[],
        all_experiments=True,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
    )

    assert {plan.experiment_id for plan in plans} == {f"e{number:02d}" for number in range(1, 8)}
    assert len(plans) == 18


def test_each_ds2_variant_resolves_to_a_runnable_schema_config() -> None:
    plans = build_plans(
        domain="ds2",
        experiment_ids=[],
        all_experiments=True,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
    )

    for plan in plans:
        config = normalize_config(load_yaml(plan.path, atomic_run_id=plan.atomic_run_id))
        assert config["atomic_run_id"] == plan.atomic_run_id
        assert config["execution_group_id"] == f"GT{int(plan.experiment_id[1:]):02d}"
