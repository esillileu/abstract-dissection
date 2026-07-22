from __future__ import annotations

from exp.cli import build_plans


def test_ds1_catalog_plans_all_variants_for_one_seed() -> None:
    plans = build_plans(
        domain="ds1", experiment_ids=[], all_experiments=True,
        seed_set="research_v1", seed_indexes="0", device=None, overrides={},
    )

    assert len(plans) == 65
    assert {plan.seed for plan in plans} == {1}
    assert {plan.device for plan in plans if plan.experiment_id in {"e06", "e07", "e08"}} == {"cuda:0"}
    assert {plan.device for plan in plans if plan.experiment_id in {"e01", "e02", "e03", "e04", "e05", "e09", "e10"}} == {"cpu"}


def test_ds2_catalog_plans_all_variants_for_one_seed() -> None:
    plans = build_plans(
        domain="ds2", experiment_ids=[], all_experiments=True,
        seed_set="research_v1", seed_indexes="0", device=None, overrides={},
    )

    assert len(plans) == 19
    assert {plan.seed for plan in plans} == {1}
    assert {plan.device for plan in plans if plan.experiment_id != "e08"} == {"cuda:0"}
    assert {plan.device for plan in plans if plan.experiment_id == "e08"} == {"cpu"}
