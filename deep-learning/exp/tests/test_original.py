from __future__ import annotations

import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from exp.cli import app
from typer.testing import CliRunner
from exp.original.cache import (
    SCHEMA_VERSION,
    cache_is_valid,
    publish_result,
    save_csv,
    save_npz,
    staging_directory,
)

runner = CliRunner()


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


@pytest.mark.parametrize("command", ("run", "analyze"))
@pytest.mark.parametrize("domain", ("ds1", "ds2"))
def test_legacy_original_flag_points_to_promoted_domain(
    command: str,
    domain: str,
) -> None:
    result = runner.invoke(app, [command, domain, "--original"])

    assert result.exit_code != 0
    assert f"{command} {domain}_original" in result.output


def test_ds2_original_summary_reads_only_original_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from exp.ds2.original.summary import summarize

    root = tmp_path / "original"
    target = (
        root
        / "data"
        / "e01"
        / "dlfs2.ch03.toy-cbow-full-softmax"
    )
    staging = staging_directory(target)
    save_csv(
        staging / "metrics.csv",
        (
            {"plot_index": 0, "loss": 1.25, "eval_interval": 20},
            {"plot_index": 1, "loss": 0.75, "eval_interval": 20},
        ),
    )
    (staging / "timing.json").write_text(
        json.dumps({"training_wall_time_s": 12.34}),
        encoding="utf-8",
    )
    (staging / "parameter_manifest.json").write_text(
        json.dumps({"parameter_count": 1234}),
        encoding="utf-8",
    )
    publish_result(
        staging,
        target,
        identity={
            "seed": 1,
            "backend": "cupy",
            "upstream_commit": "fixture",
            "source_hashes": {},
            "conditions": {},
            "config_hash": "fixture",
        },
    )

    outputs = summarize(["e01"], root=root)
    rows = list(csv.DictReader(outputs[0].open(encoding="utf-8")))

    assert outputs == [root / "image" / "e01_summary.csv"]
    assert rows[0] == {
        "series": "dlfs2.ch03.toy-cbow-full-softmax",
        "metric": "final_loss",
        "seed_runs": "1",
        "unit": "raw",
        "mean": "0.750",
        "standard_deviation": "0.000",
        "minimum": "0.750",
        "maximum": "0.750",
    }
    assert rows[1]["mean"] == "12.3"
    assert rows[2]["mean"] == "1234"
    assert "backend=cupy" in capsys.readouterr().out


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


def test_to_host_converts_a_sequence_of_cupy_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exp.original.cache import to_host

    class FakeCupyArray:
        def __init__(self, value: int) -> None:
            self.value = value

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.ndarray = FakeCupyArray
    fake_cupy.asnumpy = lambda value: np.asarray(value.value)
    fake_cupy.cuda = types.SimpleNamespace(
        get_current_stream=lambda: types.SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    assert np.array_equal(
        to_host([FakeCupyArray(1), FakeCupyArray(2)]),
        np.array([1, 2]),
    )


def test_manifest_records_schema_and_completed_status(tmp_path: Path) -> None:
    target = tmp_path / "trial"
    _publish(target)
    manifest = json.loads(
        (target / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["seed"] == 1
    assert manifest["status"] == "complete"


def test_parameter_count_deduplicates_shared_tensor_references() -> None:
    from exp.original.measurement import count_parameters

    shared = np.zeros((3, 4))
    independent = np.zeros(5)

    assert count_parameters([shared, shared, independent]) == (17, 2, 1)


def test_ds2_original_trials_all_use_cupy() -> None:
    from exp.ds2.original.run.api import trials_for

    trials = trials_for(
        ["e01", "e02", "e03", "e04", "e06", "e07", "e08"]
    )
    by_experiment = {}
    for experiment, trial in trials:
        by_experiment.setdefault(experiment, set()).add(trial.backend)

    assert set(by_experiment) == {"e01", "e02", "e03", "e04", "e06", "e07", "e08"}
    assert all(backends == {"cupy"} for backends in by_experiment.values())


def test_ds2_ch07_compatibility_alias_supports_peeky_import() -> None:
    from exp.ds2.original.run.api import UPSTREAM
    from exp.ds2.original.run.common import (
        install_ch07_compatibility_aliases,
        source_imports,
    )

    previous_alias = sys.modules.get("seq2seq")
    with source_imports(UPSTREAM):
        install_ch07_compatibility_aliases()
        seq2seq = importlib.import_module("ch07.seq2seq")
        peeky_seq2seq = importlib.import_module("ch07.peeky_seq2seq")

        assert peeky_seq2seq.Seq2seq is seq2seq.Seq2seq

    assert sys.modules.get("seq2seq") is previous_alias
