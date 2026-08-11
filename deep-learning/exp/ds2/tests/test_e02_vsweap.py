from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest
from typer.testing import CliRunner

from exp.cli import app
from exp.ds2.profile.e02.vsweap import (
    DEFAULT_CPU_VOCAB_SIZES,
    DEFAULT_VOCAB_SIZES,
    SweepWorkload,
    _crossovers,
    _default_vocab_sizes,
    _synthetic_batches,
    render_individual_sweeps,
    render_sweep,
    run,
)
from mlprosection.core.backend import BackendConfig, make_backend


runner = CliRunner()


def test_vocab_sweep_uses_device_specific_defaults() -> None:
    assert _default_vocab_sizes("cpu") == (
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
    )
    assert _default_vocab_sizes("cpu:0") == DEFAULT_CPU_VOCAB_SIZES
    assert _default_vocab_sizes("cuda:0") == DEFAULT_VOCAB_SIZES


def test_vocab_sweep_measures_implemented_conditions_and_crossovers(tmp_path) -> None:
    run(
        devices=("cpu",),
        vocab_sizes=(16,),
        batch_size=2,
        warmup_updates=0,
        measured_updates=1,
        repetitions=2,
        output_dir=tmp_path,
    )

    payload = json.loads((tmp_path / "cpu" / "vsweap.json").read_text())
    assert payload["schema_version"] == 3
    assert payload["metadata"]["measured_updates"] == 1
    assert payload["metadata"]["repetitions"] == 2
    assert payload["metadata"]["embedding_size"] == 100
    assert payload["metadata"]["timing_source"] == "window"
    assert payload["metadata"]["vocab_order"] == "ascending"
    assert {
        row["condition"] for row in payload["results"]
    } == {
        "implemented-cbow-ns",
        "implemented-cbow-fs",
        "implemented-cbow-fused-ns",
        "implemented-skipgram-ns",
        "implemented-skipgram-fs",
        "implemented-skipgram-fused-ns",
    }
    assert all(row["status"] == "ok" for row in payload["results"])
    assert all(row["timing"]["count"] == 2 for row in payload["results"])
    assert all(row["steady_event_timing"]["count"] == 1 for row in payload["results"])
    assert all(row["ci95_lower_ms"] is not None for row in payload["results"])
    assert set(payload["crossovers"]) == {
        "CBOW",
        "CBOW-Fused",
        "SkipGram",
        "SkipGram-Fused",
    }
    assert (
        payload["crossovers"]["CBOW-Fused"]["sampling_objective"]
        == "FusedNegativeSampling"
    )
    assert (tmp_path / "cpu" / "vsweap.png").is_file()
    assert (tmp_path / "cpu" / "vsweap-cbow.png").is_file()
    assert (tmp_path / "cpu" / "vsweap-skipgram.png").is_file()

    figure = render_sweep(payload)
    assert [axis.get_title() for axis in figure.axes] == [
        "CBOW · CPU",
        "Skip-gram · CPU",
    ]
    assert all(
        axis.get_xlabel() == "Vocabulary size" for axis in figure.axes
    )
    assert all(
        axis.get_ylabel() == "Update time (ms)" for axis in figure.axes
    )
    plt.close(figure)

    payload["metadata"]["timing_source"] = "event"
    event_figure = render_sweep(payload)
    assert all(axis.get_title() for axis in event_figure.axes)
    plt.close(event_figure)

    individual_figures = render_individual_sweeps(payload)
    assert [model for model, _figure in individual_figures] == [
        "CBOW",
        "SkipGram",
    ]
    assert all(
        axis.get_title() == ""
        for _model, figure in individual_figures
        for axis in figure.axes
    )
    for _model, figure in individual_figures:
        plt.close(figure)

    payload["metadata"]["device"] = "cuda:0"
    figure = render_sweep(payload)
    assert [axis.get_title() for axis in figure.axes] == [
        "CBOW · GPU",
        "Skip-gram · GPU",
    ]
    plt.close(figure)


def test_vocab_size_cli_option_requires_explicit_vsweap() -> None:
    result = runner.invoke(
        app,
        ["profile", "ds2", "-e", "02", "--vocab-size", "16"],
    )

    assert result.exit_code != 0
    assert "--vocab-size requires --vsweap" in result.output


def test_vsweap_cli_dispatches_only_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured = {}
    monkeypatch.setattr(
        "exp.ds2.profile.e02.vsweap.run",
        lambda **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "profile",
            "ds2",
            "-e",
            "02",
            "--vsweap",
            "--device",
            "cpu",
            "--vocab-size",
            "16",
            "--measured-updates",
            "7",
            "--update-repetitions",
            "3",
            "--vsweap-timing",
            "event",
            "--reverse-vocab-order",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["devices"] == ("cpu",)
    assert captured["vocab_sizes"] == (16,)
    assert captured["measured_updates"] == 7
    assert captured["repetitions"] == 3
    assert captured["timing_source"] == "event"
    assert captured["reverse_vocab_order"] is True
    assert captured["output_dir"] == tmp_path


def test_vsweap_cli_leaves_default_vocab_sizes_to_each_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    monkeypatch.setattr(
        "exp.ds2.profile.e02.vsweap.run",
        lambda **kwargs: captured.update(kwargs),
    )

    result = runner.invoke(
        app,
        ["profile", "ds2", "-e", "02", "--vsweap", "--device", "cpu"],
    )

    assert result.exit_code == 0
    assert captured["devices"] == ("cpu",)
    assert captured["vocab_sizes"] is None


def test_crossover_requires_two_consecutive_confident_ns_wins() -> None:
    rows = []
    for vocab_size, ns_ms, fs_ms in (
        (10, 5.0, 4.0),
        (20, 3.0, 5.0),
        (30, 2.0, 6.0),
    ):
        for objective, update_ms in (
            ("NegativeSampling", ns_ms),
            ("FullSoftmax", fs_ms),
        ):
            lower = update_ms - 0.1
            upper = update_ms + 0.1
            rows.append(
                {
                    "model": "CBOW",
                    "status": "ok",
                    "vocab_size": vocab_size,
                    "objective": objective,
                    "update_ms": update_ms,
                    "ci95_lower_ms": lower,
                    "ci95_upper_ms": upper,
                }
            )
    result = _crossovers(rows)["CBOW"]

    assert result["first_observed_negative_sampling_win_vocab_size"] == 20
    assert result["first_confirmed_negative_sampling_win_vocab_size"] == 20


def test_fused_crossover_compares_fused_negative_sampling_with_fs() -> None:
    rows = []
    for vocab_size in (10, 20):
        for objective, update_ms in (
            ("FusedNegativeSampling", 2.0),
            ("FullSoftmax", 5.0),
        ):
            rows.append(
                {
                    "model": "CBOW",
                    "status": "ok",
                    "vocab_size": vocab_size,
                    "objective": objective,
                    "update_ms": update_ms,
                    "ci95_lower_ms": update_ms - 0.1,
                    "ci95_upper_ms": update_ms + 0.1,
                }
            )

    result = _crossovers(rows)["CBOW-Fused"]

    assert result["sampling_objective"] == "FusedNegativeSampling"
    assert result["first_confirmed_negative_sampling_win_vocab_size"] == 10
    assert result["comparisons"][0]["full_softmax_ms"] == 5.0


def test_fused_skipgram_crossover_compares_with_skipgram_fs() -> None:
    rows = []
    for vocab_size in (10, 20):
        for objective, update_ms in (
            ("FusedNegativeSampling", 3.0),
            ("FullSoftmax", 6.0),
        ):
            rows.append(
                {
                    "model": "SkipGram",
                    "status": "ok",
                    "vocab_size": vocab_size,
                    "objective": objective,
                    "update_ms": update_ms,
                    "ci95_lower_ms": update_ms - 0.1,
                    "ci95_upper_ms": update_ms + 0.1,
                }
            )

    result = _crossovers(rows)["SkipGram-Fused"]

    assert result["sampling_objective"] == "FusedNegativeSampling"
    assert result["first_confirmed_negative_sampling_win_vocab_size"] == 10


def test_sweep_reuses_negative_samples_for_post_update_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=1))
    contexts, targets = _synthetic_batches(16, batch_size=2, update_count=1)
    workload = SweepWorkload(
        "implemented-cbow-ns",
        vocab_size=16,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    prepare_calls = 0
    original_prepare = workload.objective.prepare

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(workload.objective, "prepare", counted_prepare)
    workload.update(0, 2)

    assert prepare_calls == 1


def test_fused_sweep_reuses_negative_samples_for_post_update_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = make_backend(BackendConfig(device="cpu", dtype="float32", seed=1))
    contexts, targets = _synthetic_batches(16, batch_size=2, update_count=1)
    workload = SweepWorkload(
        "implemented-cbow-fused-ns",
        vocab_size=16,
        contexts=contexts,
        targets=targets,
        backend=backend,
    )
    prepare_calls = 0
    original_prepare = workload.objective.prepare

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(workload.objective, "prepare", counted_prepare)
    workload.update(0, 2)

    assert prepare_calls == 1
