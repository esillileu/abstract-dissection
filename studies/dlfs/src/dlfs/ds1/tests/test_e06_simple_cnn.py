from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dlfs.ds1.analysis import e06_simple_cnn
from repro_core.analysis.core import Curve


def _curve(mean, minimum, maximum):
    return Curve(
        steps=np.arange(len(mean), dtype=float),
        mean=np.asarray(mean, dtype=float),
        minimum=np.asarray(minimum, dtype=float),
        maximum=np.asarray(maximum, dtype=float),
        run_count=10,
    )


def test_e06_reproduces_simple_convnet_plot_with_ten_seed_minmax(monkeypatch):
    train = _curve([0.2, 0.8], [0.1, 0.7], [0.3, 0.9])
    test = _curve([0.15, 0.75], [0.1, 0.65], [0.2, 0.85])

    monkeypatch.setattr(
        e06_simple_cnn,
        "runs",
        lambda _client, group, atomic_ids: {
            atomic_ids[0]: ["seed"] if group == "GT06" else []
        },
    )
    monkeypatch.setattr(
        e06_simple_cnn,
        "accuracy_percent_curve",
        lambda _client, _runs, *, split, axis, x_value: (
            train if split == "train" else test
        ),
    )

    figure, curves = e06_simple_cnn.render(object(), "band", None)
    axis = figure.axes[0]

    assert figure.get_size_inches().tolist() == [6.4, 4.8]
    assert figure._analysis_match_original_canvas is True
    assert axis.get_title() == ""
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "accuracy (%)"
    np.testing.assert_allclose(axis.get_ylim(), (0.0, 100.0))
    assert [line.get_marker() for line in axis.lines] == ["o", "s"]
    assert [line.get_label() for line in axis.lines] == [
        "train (n=10)",
        "test (n=10)",
    ]
    assert len(axis.collections) == 2
    assert curves == {"train": train, "test": test}
    plt.close(figure)
