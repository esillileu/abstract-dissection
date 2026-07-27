from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from exp.analyze import (
    AnalysisClient,
    Curve,
    RunRef,
    aggregate,
    completed_seed_runs,
    parse_experiment_selection,
    plot_curve,
)
from exp.ds1.analyze.final_metrics import (
    final_test_accuracy_curve,
    summaries_for_runs,
    training_time_curve,
)
from exp.ds2.analyze.e01_toy_word2vec import render
from exp.ds2.analyze.render import _save_result
from exp.plot_theme import (
    ACCENT_COLORS,
    BACKGROUND,
    CORE_HIGHLIGHT,
    FONT_FAMILY,
    INK,
    MPL_WINTER_FATAL,
    MUTED,
    SECONDARY_DATA,
    SURFACE,
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


def test_both_minmax_plot_styles_are_supported():
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([2.0, 3.0]),
        minimum=np.asarray([1.0, 2.0]),
        maximum=np.asarray([4.0, 5.0]),
        run_count=3,
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


def test_missing_experiment_still_renders_empty_figure(tmp_path):
    client = FakeClient()
    client.get_experiment_by_name = lambda _name: None
    output = tmp_path / "empty.png"

    paths = _save_result(render(client, "band", output), output)

    assert output in paths
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()


def test_experiment_selection_supports_ranges_and_reports_extensions():
    selected, skipped = parse_experiment_selection(
        ["01,03-05", "e08"], ["e01", "e02", "e03", "e04", "e06", "e08"]
    )

    assert selected == ["e01", "e03", "e04", "e08"]
    assert skipped == ["e05"]


def test_final_metrics_aggregate_full_test_accuracy_and_training_wall_time(tmp_path):
    run_refs = []
    for index, (accuracy, timing_ns) in enumerate(
        ((0.991, (1_000_000_000, 2_000_000_000)), (0.993, (4_000_000_000,)))
    ):
        root = tmp_path / f"seed-{index}"
        root.mkdir()
        (root / "evaluations.csv").write_text(
            "axis,axis_step,update,epoch,evaluation_set_id,split,example_count,loss,accuracy\n"
            f"terminal,20,12000,20,mnist-test-full,test,10000,,{accuracy}\n",
            encoding="utf-8",
        )
        (root / "timing_windows.csv").write_text(
            "start_update,end_update,update_count,closed_by,train_wall_time_ns\n"
            + "".join(
                f"1,2,1,epoch,{value}\n"
                for value in timing_ns
            ),
            encoding="utf-8",
        )
        run_refs.append(
            RunRef(
                run_id=f"run-{index}",
                atomic_run_id="CNN-DEEP-BOOK",
                seed=str(index),
                start_time=index,
                local_artifact_root=root,
            )
        )

    accuracy_curve = final_test_accuracy_curve(FakeClient(), run_refs)
    time_curve = training_time_curve(FakeClient(), run_refs)
    summaries = summaries_for_runs(
        FakeClient(),
        {"CNN-DEEP-BOOK": run_refs},
    )

    np.testing.assert_allclose(
        [accuracy_curve.mean[0], accuracy_curve.minimum[0], accuracy_curve.maximum[0]],
        [0.992, 0.991, 0.993],
    )
    np.testing.assert_allclose(
        [time_curve.mean[0], time_curve.minimum[0], time_curve.maximum[0]],
        [3.5, 3.0, 4.0],
    )
    assert accuracy_curve.run_count == time_curve.run_count == 2
    np.testing.assert_allclose(
        summaries["CNN-DEEP-BOOK/final_test_accuracy"].standard_deviation,
        np.sqrt(2) / 1000,
    )
    np.testing.assert_allclose(
        summaries["CNN-DEEP-BOOK/training_time_s"].standard_deviation,
        np.sqrt(0.5),
    )
