from __future__ import annotations

from types import SimpleNamespace

import pytest

from exp.cli import build_plans, main, parse_experiment_ids


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


def test_atomic_run_selection_plans_only_requested_variants() -> None:
    plans = build_plans(
        domain="ds1",
        experiment_ids=["01"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
        atomic_run_ids=["MLP-OPT-SGD,MLP-OPT-ADAM"],
    )

    assert {plan.atomic_run_id for plan in plans} == {"MLP-OPT-SGD", "MLP-OPT-ADAM"}
    assert len(plans) == 2


def test_atomic_run_exclusion_plans_all_other_variants() -> None:
    plans = build_plans(
        domain="ds1",
        experiment_ids=["01"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="0",
        device=None,
        overrides={},
        excluded_atomic_run_ids=["MLP-OPT-SGD", "MLP-OPT-ADAM"],
    )

    assert {plan.atomic_run_id for plan in plans} == {
        "MLP-OPT-MOMENTUM",
        "MLP-OPT-ADAGRAD",
    }


def test_unknown_atomic_run_id_is_rejected_in_selected_experiments() -> None:
    with pytest.raises(
        ValueError,
        match="unknown atomic run ID in selected experiments: MISSING",
    ):
        build_plans(
            domain="ds1",
            experiment_ids=["01"],
            all_experiments=False,
            seed_set="research_v1",
            seed_indexes="0",
            device=None,
            overrides={},
            atomic_run_ids=["MISSING"],
        )


def test_atomic_run_include_and_exclude_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="choose at most one"):
        build_plans(
            domain="ds1",
            experiment_ids=["01"],
            all_experiments=False,
            seed_set="research_v1",
            seed_indexes="0",
            device=None,
            overrides={},
            atomic_run_ids=["MLP-OPT-SGD"],
            excluded_atomic_run_ids=["MLP-OPT-ADAM"],
        )


def test_seed_first_orders_all_selected_atomic_runs_by_seed() -> None:
    plans = build_plans(
        domain="ds1",
        experiment_ids=["01-02"],
        all_experiments=False,
        seed_set="research_v1",
        seed_indexes="1,0",
        device=None,
        overrides={},
        atomic_run_ids=["MLP-OPT-SGD,MLP-INIT-HE"],
        seed_first=True,
    )

    assert [(plan.atomic_run_id, plan.seed) for plan in plans] == [
        ("MLP-OPT-SGD", 2),
        ("MLP-INIT-HE", 2),
        ("MLP-OPT-SGD", 1),
        ("MLP-INIT-HE", 1),
    ]


def test_analyze_forwards_summary_flag(monkeypatch) -> None:
    captured = []
    module = SimpleNamespace(main=lambda argv: captured.extend(argv))
    monkeypatch.setattr("exp.cli.importlib.import_module", lambda _name: module)

    main(["ds2", "analyze", "-e", "01", "--summary"])

    assert captured == ["-e", "01", "--error-style", "band", "--summary"]
