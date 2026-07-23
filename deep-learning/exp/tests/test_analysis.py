from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import (
    AnalysisClient,
    Curve,
    aggregate,
    completed_seed_runs,
    parse_experiment_selection,
    plot_curve,
)
from exp.ds2.analyze.e01_toy_word2vec import render
from exp.ds2.analyze.render import _save_result


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
