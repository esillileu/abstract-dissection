from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pytest

from exp.framework.analysis.core import (
    AnalysisClient,
    aggregate,
    cached_analysis_console_output,
    cached_analysis_outputs,
    completed_seed_runs,
    parse_experiment_selection,
    plot_curve,
    write_analysis_cache,
)
from exp.deepscratch.ds2.analysis import render as ds2_render
from exp.framework.plotting.theme import (
    ACCENT_COLORS,
    BACKGROUND,
    CORE_HIGHLIGHT,
    FONT_FAMILY,
    INK,
    MPL_WINTER_FATAL,
    MUTED,
    SECONDARY_DATA,
    SURFACE,
    remove_figure_title,
)


def _run(run_id: str, *, atomic: str, seed: int, start_time: int):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=start_time),
        data=SimpleNamespace(
            tags={
                "run.type": "seed_trial",
                "execution_group.id": "GT01",
                "atomic_run.id": atomic,
            },
            params={"seed/master": str(seed)},
        ),
    )


class FakeClient:
    def __init__(self, runs=()):
        self.runs = list(runs)

    def get_experiment_by_name(self, name):
        return None if name == "missing" else SimpleNamespace(experiment_id="1")

    def search_runs(self, **kwargs):
        return self.runs

    def download_artifacts(self, run_id, artifact_path):
        raise FileNotFoundError((run_id, artifact_path))


def test_aggregate_uses_only_steps_common_to_every_seed():
    curve = aggregate([{0.0: 1.0, 1.0: 3.0}, {1.0: 5.0, 2.0: 9.0}])

    np.testing.assert_array_equal(curve.steps, [1.0])
    np.testing.assert_array_equal(curve.mean, [4.0])
    np.testing.assert_array_equal(curve.minimum, [3.0])
    np.testing.assert_array_equal(curve.maximum, [5.0])
    np.testing.assert_allclose(curve.standard_deviation, [np.sqrt(2.0)])
    assert curve.run_count == 2


def test_completed_seed_runs_keeps_latest_attempt_and_all_seeds():
    runs = [
        _run("new-seed-1", atomic="A", seed=1, start_time=30),
        _run("seed-2", atomic="A", seed=2, start_time=20),
        _run("old-seed-1", atomic="A", seed=1, start_time=10),
    ]

    grouped = completed_seed_runs(
        FakeClient(runs), experiment_name="ds1", group_id="GT01", atomic_run_ids=["A", "B"]
    )

    assert [run.run_id for run in grouped["A"]] == ["new-seed-1", "seed-2"]
    assert grouped["B"] == []


def test_completed_seed_runs_can_select_one_master_seed():
    client = AnalysisClient(
        FakeClient(
            [
                _run("seed-1", atomic="A", seed=1, start_time=20),
                _run("seed-2", atomic="A", seed=2, start_time=10),
            ]
        ),
        seed=2,
    )

    grouped = completed_seed_runs(
        client, experiment_name="ds1", group_id="GT01", atomic_run_ids=["A"]
    )

    assert [run.run_id for run in grouped["A"]] == ["seed-2"]


def test_analysis_cache_reuses_outputs_for_identical_run_ids(tmp_path):
    client = AnalysisClient(
        FakeClient([_run("run-1", atomic="A", seed=1, start_time=20)])
    )
    completed_seed_runs(
        client,
        experiment_name="ds1",
        group_id="GT01",
        atomic_run_ids=["A"],
    )
    output = tmp_path / "e01_band.png"
    summary = output.with_suffix(".csv")
    output.write_bytes(b"figure")
    summary.write_text("summary\n", encoding="utf-8")
    write_analysis_cache(client, output, [output, summary])

    assert cached_analysis_outputs(client, output) == [output, summary]


def test_analysis_cache_replays_saved_console_output(tmp_path):
    client = AnalysisClient(
        FakeClient([_run("run-1", atomic="A", seed=1, start_time=20)])
    )
    completed_seed_runs(
        client,
        experiment_name="ds1",
        group_id="GT01",
        atomic_run_ids=["A"],
    )
    output = tmp_path / "e01_summary.csv"
    output.write_text("summary\n", encoding="utf-8")
    write_analysis_cache(
        client,
        output,
        [output],
        console_output="accuracy (%): 99.00 ± 0.10 (n=10)\n",
    )

    assert cached_analysis_outputs(client, output) == [output]
    assert cached_analysis_console_output(output) == (
        "accuracy (%): 99.00 ± 0.10 (n=10)\n"
    )


def test_analysis_cache_misses_when_selected_run_id_changes(tmp_path):
    source = FakeClient([_run("run-1", atomic="A", seed=1, start_time=20)])
    client = AnalysisClient(source)
    completed_seed_runs(
        client,
        experiment_name="ds1",
        group_id="GT01",
        atomic_run_ids=["A"],
    )
    output = tmp_path / "e01_summary.csv"
    output.write_text("summary\n", encoding="utf-8")
    write_analysis_cache(client, output, [output])
    source.runs = [_run("run-2", atomic="A", seed=1, start_time=30)]

    assert cached_analysis_outputs(client, output) is None


def test_ds2_render_study_saves_figure_and_summary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "ds2_e01_imp.png"
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 2.0])
    curve = aggregate([{0.0: 1.0, 1.0: 2.0}])
    monkeypatch.setitem(
        ds2_render.RENDERERS,
        "e01",
        lambda *_args: (figure, {"CBOW": curve}),
    )

    paths = ds2_render.render_study(object(), "e01", output)

    assert paths == [output, tmp_path / "ds2_e01_imp_curves.csv"]
    assert all(path.is_file() for path in paths)


def test_band_uses_one_sample_standard_deviation(monkeypatch):
    curve = aggregate(
        [
            {0.0: 0.0, 1.0: 0.0},
            {0.0: 2.0, 1.0: 4.0},
            {0.0: 10.0, 1.0: 8.0},
        ]
    )
    figure, axis = plt.subplots()
    captured = {}
    fill_between = axis.fill_between

    def capture_fill_between(steps, lower, upper, **kwargs):
        captured["lower"] = lower
        captured["upper"] = upper
        return fill_between(steps, lower, upper, **kwargs)

    monkeypatch.setattr(axis, "fill_between", capture_fill_between)

    plot_curve(axis, curve, label="series", error_style="band")

    np.testing.assert_allclose(
        captured["lower"],
        curve.mean - curve.standard_deviation,
    )
    np.testing.assert_allclose(
        captured["upper"],
        curve.mean + curve.standard_deviation,
    )
    assert not np.array_equal(captured["lower"], curve.minimum)
    assert not np.array_equal(captured["upper"], curve.maximum)
    plt.close(figure)


def test_both_uncertainty_plot_styles_are_supported():
    curve = aggregate(
        [
            {0.0: 1.0, 1.0: 2.0},
            {0.0: 4.0, 1.0: 5.0},
            {0.0: 2.0, 1.0: 3.0},
        ]
    )
    for style in ("band", "errorbar"):
        figure, axis = plt.subplots()
        assert plot_curve(axis, curve, label="series", error_style=style) is not None
        plt.close(figure)


def test_plot_theme_uses_repository_palette():
    assert rcParams["figure.facecolor"] == BACKGROUND
    assert rcParams["axes.facecolor"] == SURFACE
    assert rcParams["text.color"] == INK
    assert rcParams["axes.edgecolor"] == MUTED
    assert rcParams["axes.prop_cycle"].by_key()["color"] == list(ACCENT_COLORS)
    assert ACCENT_COLORS == MPL_WINTER_FATAL
    assert SECONDARY_DATA == "#879096"
    assert CORE_HIGHLIGHT == "#A3163B"
    assert rcParams["font.family"] == ["serif"]
    assert rcParams["font.serif"][0] == FONT_FAMILY
    assert rcParams["mathtext.fontset"] == "stix"


def test_remove_figure_title_preserves_subplot_titles():
    figure, axis = plt.subplots()
    figure.suptitle("figure title")
    axis.set_title("left title", loc="left")
    axis.set_title("center title")
    axis.set_title("right title", loc="right")

    remove_figure_title(figure)

    assert figure._suptitle is None
    assert axis.get_title(loc="left") == "left title"
    assert axis.get_title() == "center title"
    assert axis.get_title(loc="right") == "right title"
    plt.close(figure)


def test_renderer_registry_saves_empty_figure(tmp_path, monkeypatch):
    output = tmp_path / "empty.png"
    figure, axis = plt.subplots()
    monkeypatch.setitem(
        ds2_render.RENDERERS,
        "e01",
        lambda *_args: (figure, {}),
    )
    paths = ds2_render.render_study(object(), "e01", output)

    assert output in paths
    assert output.is_file()
    assert all(size > 0 for size in plt.imread(output).shape[:2])
    assert figure._suptitle is None
    assert axis.get_title() == ""
    assert output.with_name("empty_curves.csv").is_file()


def test_experiment_selection_supports_ranges_and_reports_extensions():
    selected, skipped = parse_experiment_selection(
        ["01,03-05", "e08"], ["e01", "e02", "e03", "e04", "e06", "e08"]
    )

    assert selected == ["e01", "e03", "e04", "e08"]
    assert skipped == ["e05"]
