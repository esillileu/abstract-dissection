from __future__ import annotations

import pytest
from typer.testing import CliRunner

from exp.cli import app
from exp.deepscratch.definition import DEFINITION
from exp.deepscratch.identity import Variant, Volume
from exp.framework.execution import RunOptions, RunOrder, RunSelection
from exp.framework.execution.parsing import (
    parse_atomic_run_ids,
    parse_experiment_ids,
    parse_overrides,
    parse_seed_values,
)
from exp.framework.execution.planning import Planner


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
    return Planner(DEFINITION.implementation(Volume(domain), Variant.IMPLEMENTED)).build(
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

    assert len(ds1) == 66
    assert len(ds2) == 35
    assert {
        plan.device
        for plan in ds1
        if plan.experiment_id in {"e06", "e07", "e08", "e12"}
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
        assert "deepscratch" in result.output


def test_new_order_succeeds_and_old_order_is_rejected() -> None:
    current = runner.invoke(
        app, ["plan", "deepscratch", "ds1", "-e", "01", "--seed", "1"]
    )
    old = runner.invoke(app, ["ds1", "plan", "-e", "01", "--seed", "1"])

    assert current.exit_code == 0
    assert "4 planned runs" in current.output
    assert old.exit_code != 0
    assert "No such command" in old.output


def test_run_requires_explicit_experiment_selection() -> None:
    result = runner.invoke(app, ["run", "deepscratch", "ds1", "--dry-run"])
    assert result.exit_code != 0
    assert "requires --all or --experiment/-e" in result.output


def test_nonstandard_seed_and_removed_seed_first_are_rejected() -> None:
    assert runner.invoke(
        app, ["plan", "deepscratch", "ds1", "-e", "01", "-seed", "1"]
    ).exit_code != 0
    assert runner.invoke(
        app, ["plan", "deepscratch", "ds1", "-e", "01", "--seed-first"]
    ).exit_code != 0


def test_analyze_uses_single_normalized_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured["tracking_uri"] = args[0]
        captured.update(kwargs)
        return tmp_path / "observations.csv"

    monkeypatch.setattr(
        "exp.deepscratch.analysis.orchestrator.write_analysis",
        fake_write_analysis,
    )

    result = runner.invoke(
        app, ["analyze", "deepscratch", "ds2", "-e", "01"]
    )

    assert result.exit_code == 0
    assert str(captured.pop("output_dir")).endswith(
        "results/exp/deepscratch"
    )
    assert captured["experiment_ids"] == ["e01"]
    assert captured["variants"] == (Variant.IMPLEMENTED,)
    assert captured["seed"] is None
    assert captured["run_id"] is None
    assert captured["error_style"] == "band"
    assert captured["print_summary"] is False


def test_analyze_accepts_explicit_errorbar_style(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "exp.deepscratch.analysis.orchestrator.write_analysis",
        fake_write_analysis,
    )
    result = runner.invoke(
        app,
        [
            "analyze",
            "deepscratch",
            "ds1",
            "-e",
            "01",
            "--error-style",
            "errorbar",
            "-s",
        ],
    )

    assert result.exit_code == 0
    assert captured["error_style"] == "errorbar"
    assert captured["print_summary"] is True


def test_only_canonical_domain_is_exposed() -> None:
    for action in ("plan", "run", "analyze"):
        help_result = runner.invoke(app, [action, "--help"])
        assert help_result.exit_code == 0
        assert "deepscratch" in help_result.output
        assert "ds1_original" not in help_result.output
