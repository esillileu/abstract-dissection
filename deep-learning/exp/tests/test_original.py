from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from exp.cli import main
from exp.original.cache import (
    SCHEMA_VERSION,
    cache_is_valid,
    publish_result,
    save_csv,
    save_npz,
    staging_directory,
)


def _publish(target: Path, *, value: str = "one") -> None:
    staging = staging_directory(target)
    (staging / "raw.txt").write_text(value, encoding="utf-8")
    publish_result(
        staging,
        target,
        identity={
            "seed": 1,
            "backend": "numpy",
            "upstream_commit": "abc",
            "source_hashes": {"source.py": "123"},
            "conditions": {"epochs": 1},
            "config_hash": "config",
        },
    )


def test_cache_requires_marker_manifest_and_matching_artifacts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trial"
    _publish(target)
    assert cache_is_valid(target)

    (target / "raw.txt").write_text("changed", encoding="utf-8")
    assert not cache_is_valid(target)


def test_cache_expected_identity_invalidates_source_or_config_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trial"
    _publish(target)

    assert cache_is_valid(target, {"config_hash": "config"})
    assert not cache_is_valid(target, {"config_hash": "different"})
    assert not cache_is_valid(
        target, {"source_hashes": {"source.py": "different"}}
    )


def test_publish_replaces_an_incomplete_result(tmp_path: Path) -> None:
    target = tmp_path / "trial"
    target.mkdir()
    (target / "partial.csv").write_text("unfinished", encoding="utf-8")

    _publish(target, value="complete")

    assert cache_is_valid(target)
    assert not (target / "partial.csv").exists()
    assert (target / "raw.txt").read_text(encoding="utf-8") == "complete"


def test_original_run_defaults_to_all_registered_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(domain, experiments, *, force, output_dir):
        captured.update(
            domain=domain,
            experiments=experiments,
            force=force,
            output_dir=output_dir,
        )

    monkeypatch.setattr("exp.original.dispatch.run_original", fake_run)
    main(["ds1", "run", "-o"])

    assert captured["domain"] == "ds1"
    assert captured["experiments"] == [
        "e01",
        "e02",
        "e03",
        "e04",
        "e05",
        "e06",
        "e09",
        "e10",
    ]
    assert captured["force"] is False


def test_original_partial_selection_and_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(domain, experiments, *, force, output_dir):
        captured.update(experiments=experiments, force=force)

    monkeypatch.setattr("exp.original.dispatch.run_original", fake_run)
    main(["ds2", "run", "-o", "-e", "02,08", "--force"])

    assert captured == {"experiments": ["e02", "e08"], "force": True}


@pytest.mark.parametrize(
    ("domain", "experiment"),
    (("ds1", "07"), ("ds2", "05")),
)
def test_original_rejects_experiments_without_source_figures(
    domain: str,
    experiment: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main([domain, "run", "-o", "-e", experiment, "--dry-run"])
    assert "no registered original trials" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--seed-set", "research_v1"], "--seed-set"),
        (["--seed", "2"], "fixed seed 1"),
        (["--atomic-run", "ANY"], "atomic-run"),
        (["--set", "budget.max_epochs=1"], "YAML overrides"),
    ],
)
def test_original_rejects_conflicting_options(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["ds1", "run", "-o", *arguments])
    assert message in capsys.readouterr().err


def test_ds1_renderer_uses_only_persisted_fixture(
    tmp_path: Path,
) -> None:
    root = tmp_path / "original"
    trial_ids = tuple(
        f"dlfs1.ch06.optimizer-path.{name}"
        for name in ("sgd", "momentum", "adagrad", "adam")
    )
    for trial_id in trial_ids:
        target = root / "data" / "e09" / trial_id
        staging = staging_directory(target)
        save_csv(
            staging / "trajectory.csv",
            (
                {
                    "update": index,
                    "x": -7 + index,
                    "y": 2 - index,
                    "objective": 1,
                    "gradient_x": 1,
                    "gradient_y": 1,
                }
                for index in range(3)
            ),
        )
        publish_result(
            staging,
            target,
            identity={
                "seed": 1,
                "backend": "numpy",
                "upstream_commit": "fixture",
                "source_hashes": {},
                "conditions": {},
                "config_hash": "fixture",
            },
        )

    before = set(sys.modules)
    from exp.ds1.original.render.api import render

    outputs = render(["e09"], root=root)
    newly_imported = set(sys.modules) - before

    assert [path.name for path in outputs] == [
        "e09_optimizer_compare_naive.png"
    ]
    assert outputs[0].is_file()
    assert not any(".original.run" in name for name in newly_imported)
    assert not any(
        name == "dataset" or name.startswith("dataset.")
        for name in newly_imported
    )


def test_npz_is_host_numpy_and_never_requires_pickle(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    save_npz(path, values=np.arange(4))

    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(archive["values"], np.arange(4))


def test_manifest_records_schema_and_completed_status(tmp_path: Path) -> None:
    target = tmp_path / "trial"
    _publish(target)
    manifest = json.loads(
        (target / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["seed"] == 1
    assert manifest["status"] == "complete"


def test_ds2_backend_policy_applies_cupy_only_to_e02() -> None:
    from exp.ds2.original.run.api import trials_for

    trials = trials_for(
        ["e01", "e02", "e03", "e04", "e06", "e07", "e08"]
    )
    by_experiment = {}
    for experiment, trial in trials:
        by_experiment.setdefault(experiment, set()).add(trial.backend)

    assert by_experiment["e02"] == {"cupy"}
    assert all(
        backends == {"numpy"}
        for experiment, backends in by_experiment.items()
        if experiment != "e02"
    )
