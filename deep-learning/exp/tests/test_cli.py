from __future__ import annotations

from exp.cli import build_plans, parse_experiment_ids


def test_experiment_selection_supports_ranges_and_commas() -> None:
    assert parse_experiment_ids(["01-03", "e05,e07-e08"]) == [
        "e01",
        "e02",
        "e03",
        "e05",
        "e07",
        "e08",
    ]


def test_ds1_range_plans_each_selected_experiment() -> None:
    plans = build_plans(
        domain="ds1",
        experiment_ids=["01-02"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
    )

    assert len(plans) == 7
    assert {plan.experiment_id for plan in plans} == {"e01", "e02"}


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
