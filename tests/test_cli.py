from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dlfs.identity import Variant, Volume
from dlfs.plugin import DEFINITION
from repro_core.cli import app
from repro_core.execution import RunOptions, RunOrder, RunSelection
from repro_core.execution.parsing import (
    parse_atomic_run_ids,
    parse_experiment_ids,
    parse_overrides,
    parse_seed_values,
)
from repro_core.execution.planning import Planner

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
    return Planner(
        DEFINITION.implementation(Volume(domain), Variant.IMPLEMENTED)
    ).build(
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
        "e01",
        "e02",
        "e03",
        "e05",
        "e07",
        "e08",
    ]


def test_seed_atomic_and_override_parsers() -> None:
    assert parse_seed_values("1,3-5,3", available=[1, 2, 3, 4, 5]) == [1, 3, 4, 5]
    assert parse_atomic_run_ids(["A,B", "A"]) == ["A", "B"]
    assert parse_overrides(
        ["budget.max_epochs=2", "tracking.upload_checkpoint=false"]
    ) == {
        "budget": {"max_epochs": 2},
        "tracking": {"upload_checkpoint": False},
    }
    with pytest.raises(ValueError, match="not in the selected seed set"):
        parse_seed_values("0", available=[1, 2])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_overrides(["budget.max_epochs"])


def test_catalog_plan_counts_and_default_devices() -> None:
    ds1 = _plans("ds1", all_experiments=True)
    ds2 = _plans("ds2", all_experiments=True)

    assert len(ds1) == 71
    assert len(ds2) == 35
    assert {
        plan.device
        for plan in ds1
        if plan.experiment_id in {"e06", "e07", "e08", "e12"}
    } == {"cuda:0"}
    assert {
        plan.device for plan in ds2 if plan.experiment_id not in {"e08", "e12"}
    } == {"cuda:0"}
    assert {plan.device for plan in ds2 if plan.experiment_id == "e08"} == {"cpu"}
    assert {plan.device for plan in ds2 if plan.experiment_id == "e12"} == {"cpu"}


def test_e02_analysis_uses_catalog_canonical_gpu() -> None:
    from dlfs.analysis.orchestrator import _canonical_device

    assert (
        _canonical_device(
            Volume.DS2,
            Variant.IMPLEMENTED,
            study_id="e02",
            condition_ids=("W2V-PTB-SKIPGRAM-NS",),
        )
        == "cuda:0"
    )


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
    assert (
        runner.invoke(
            app, ["plan", "deepscratch", "ds1", "-e", "01", "-seed", "1"]
        ).exit_code
        != 0
    )
    assert (
        runner.invoke(
            app, ["plan", "deepscratch", "ds1", "-e", "01", "--seed-first"]
        ).exit_code
        != 0
    )


def test_analyze_uses_single_normalized_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("F1_MLFLOW_TRACKING_URI", "http://canonical:5001/mlflow-f1/")
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured["tracking_uri"] = args[0]
        captured.update(kwargs)
        return tmp_path / "observations.csv"

    monkeypatch.setattr(
        "dlfs.analysis.orchestrator.write_analysis",
        fake_write_analysis,
    )

    result = runner.invoke(app, ["analyze", "deepscratch", "ds2", "-e", "01"])

    assert result.exit_code == 0
    assert str(captured.pop("output_dir")).endswith("artifacts/analysis/dlfs/ds2")
    assert captured["tracking_uri"] == "http://canonical:5001/mlflow-f1"
    assert captured["experiment_ids"] == ["e01"]
    assert captured["variants"] == (Variant.IMPLEMENTED,)
    assert captured["seed"] is None
    assert captured["run_id"] is None
    assert captured["error_style"] == "band"
    assert captured["print_summary"] is False


def test_analyze_accepts_explicit_errorbar_style(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("F1_MLFLOW_TRACKING_URI", "http://canonical:5001/mlflow-f1")
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "dlfs.analysis.orchestrator.write_analysis",
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


@pytest.mark.parametrize(
    ("refresh_arguments", "expected"),
    ((["--refresh"], "all"), (["--refresh", "analysis"], "analysis")),
)
def test_analyze_supports_raw_and_analysis_refresh_scopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    refresh_arguments: list[str],
    expected: str,
) -> None:
    monkeypatch.setenv("F1_MLFLOW_TRACKING_URI", "http://canonical:5001/mlflow-f1")
    captured = {}

    def fake_write_analysis(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "dlfs.analysis.orchestrator.write_analysis",
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
            *refresh_arguments,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["refresh"] == expected


def test_only_canonical_domain_is_exposed() -> None:
    for action in ("plan", "run", "analyze"):
        help_result = runner.invoke(app, [action, "--help"])
        assert help_result.exit_code == 0
        assert "deepscratch" in help_result.output
        assert "ds1_original" not in help_result.output


def test_dlfs_tracking_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlfs.tracking import resolve_tracking_uri

    monkeypatch.delenv("F1_MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="F1_MLFLOW_TRACKING_URI must be loaded"):
        resolve_tracking_uri()

    monkeypatch.setenv("F1_MLFLOW_TRACKING_URI", "http://10.0.0.1:5001/mlflow-f1")
    assert resolve_tracking_uri() == "http://10.0.0.1:5001/mlflow-f1"
    assert (
        resolve_tracking_uri("http://10.0.0.1:5001/mlflow-f1/")
        == "http://10.0.0.1:5001/mlflow-f1"
    )
    assert resolve_tracking_uri("http://custom:5000/") == "http://custom:5000"


def test_f2_tracking_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from f2.tracking import resolve_tracking_uri

    monkeypatch.delenv("F2_MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValueError, match="F2_MLFLOW_TRACKING_URI"):
        resolve_tracking_uri()
    monkeypatch.setenv("F2_MLFLOW_TRACKING_URI", "https://f2.example/mlflow/")
    assert resolve_tracking_uri() == "https://f2.example/mlflow"
    assert (
        resolve_tracking_uri("https://explicit.example/") == "https://explicit.example"
    )
