from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from exp.cli import DOMAIN_REGISTRY, app
from exp.commands import select_original_experiments
from exp.domain import RunOptions, RunOrder, RunSelection
from exp.parsing import (
    parse_atomic_run_ids,
    parse_experiment_ids,
    parse_overrides,
    parse_seed_values,
)
from exp.planning import Planner


runner = CliRunner()


def _plans(
    domain: str,
    *,
    experiments: tuple[str, ...] = (),
    all_experiments: bool = False,
    seeds: str | None = "1",
    atomic_runs: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
    order: RunOrder = RunOrder.CATALOG_FIRST,
):
    return Planner(DOMAIN_REGISTRY[domain]).build(
        RunSelection(
            experiment_ids=experiments,
            all_experiments=all_experiments,
            atomic_run_ids=atomic_runs,
            excluded_atomic_run_ids=excluded,
            seed_values=seeds,
        ),
        RunOptions(order=order),
    )


def test_experiment_selection_supports_ranges_commas_and_deduplication() -> None:
    assert parse_experiment_ids(["01-03", "e03,e05,e07-e08"]) == [
        "e01", "e02", "e03", "e05", "e07", "e08"
    ]


def test_seed_atomic_and_override_parsers() -> None:
    assert parse_seed_values("1,3-5,3", available=[1, 2, 3, 4, 5]) == [
        1, 3, 4, 5
    ]
    assert parse_atomic_run_ids(["A,B", "A"]) == ["A", "B"]
    assert parse_overrides(
        ["budget.max_epochs=2", "tracking.enabled=false"]
    ) == {"budget": {"max_epochs": 2}, "tracking": {"enabled": False}}
    with pytest.raises(ValueError, match="not in the selected seed set"):
        parse_seed_values("0", available=[1, 2])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_overrides(["budget.max_epochs"])


def test_catalog_plan_counts_and_default_devices() -> None:
    ds1 = _plans("ds1", all_experiments=True)
    ds2 = _plans("ds2", all_experiments=True)

    assert len(ds1) == 65
    assert len(ds2) == 21
    assert {
        plan.device
        for plan in ds1
        if plan.experiment_id in {"e06", "e07", "e08"}
    } == {"cuda:0"}
    assert {
        plan.device for plan in ds2 if plan.experiment_id != "e08"
    } == {"cuda:0"}
    assert {plan.device for plan in ds2 if plan.experiment_id == "e08"} == {
        "cpu"
    }


def test_atomic_selection_and_seed_first_order() -> None:
    plans = _plans(
        "ds1",
        experiments=("01-02",),
        seeds="2,1",
        atomic_runs=("MLP-OPT-SGD,MLP-INIT-HE",),
        order=RunOrder.SEED_FIRST,
    )
    assert [(plan.atomic_run_id, plan.seed) for plan in plans] == [
        ("MLP-OPT-SGD", 2),
        ("MLP-INIT-HE", 2),
        ("MLP-OPT-SGD", 1),
        ("MLP-INIT-HE", 1),
    ]


def test_atomic_include_exclude_and_unknown_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="choose at most one"):
        _plans(
            "ds1",
            experiments=("01",),
            atomic_runs=("MLP-OPT-SGD",),
            excluded=("MLP-OPT-ADAM",),
        )
    with pytest.raises(ValueError, match="unknown atomic run ID"):
        _plans("ds1", experiments=("01",), atomic_runs=("MISSING",))


@pytest.mark.parametrize("arguments", ([], ["plan"], ["run"], ["analyze"]))
def test_help_exposes_action_and_domain_registry(arguments: list[str]) -> None:
    result = runner.invoke(app, [*arguments, "--help"])
    assert result.exit_code == 0
    if not arguments:
        assert all(action in result.output for action in ("plan", "run", "analyze"))
    else:
        assert all(domain in result.output for domain in DOMAIN_REGISTRY)


def test_new_order_succeeds_and_old_order_is_rejected() -> None:
    current = runner.invoke(app, ["plan", "ds1", "-e", "01", "--seed", "1"])
    old = runner.invoke(app, ["ds1", "plan", "-e", "01", "--seed", "1"])

    assert current.exit_code == 0
    assert "4 planned runs" in current.output
    assert old.exit_code != 0
    assert "No such command" in old.output


def test_run_requires_explicit_experiment_selection() -> None:
    result = runner.invoke(app, ["run", "ds1", "--dry-run"])
    assert result.exit_code != 0
    assert "requires --all or --experiment/-e" in result.output


def test_nonstandard_seed_and_removed_seed_first_are_rejected() -> None:
    assert runner.invoke(
        app, ["plan", "ds1", "-e", "01", "-seed", "1"]
    ).exit_code != 0
    assert runner.invoke(
        app, ["plan", "ds1", "-e", "01", "--seed-first"]
    ).exit_code != 0


def test_analyze_uses_typed_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    module = SimpleNamespace(
        analyze=lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr("exp.commands.importlib.import_module", lambda _name: module)

    result = runner.invoke(
        app, ["analyze", "ds2", "-e", "01", "--summary"]
    )

    assert result.exit_code == 0
    assert captured == {
        "experiments": ["01"],
        "tracking_uri": None,
        "error_style": "band",
        "output_dir": None,
        "seed": None,
        "summary": True,
    }


def test_registered_domain_contracts_are_complete_and_unique() -> None:
    assert len(DOMAIN_REGISTRY) == len(set(DOMAIN_REGISTRY))
    for name, definition in DOMAIN_REGISTRY.items():
        assert definition.name == name
        assert definition.config_root.is_dir()
        assert definition.spec_module
        assert definition.executor_module
        assert definition.analysis_module


def test_ds2_original_default_order_handles_dependencies_and_long_run_last() -> None:
    domain = DOMAIN_REGISTRY["ds2"]

    assert select_original_experiments(domain, []) == [
        "e01", "e03", "e04", "e06", "e07", "e08", "e02"
    ]
    assert select_original_experiments(domain, ["08,01,02"]) == [
        "e08", "e01", "e02"
    ]
