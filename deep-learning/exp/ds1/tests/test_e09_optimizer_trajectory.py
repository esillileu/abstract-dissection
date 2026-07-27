from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exp.ds1.analyze import e09_optimizer_trajectory


def test_optimizer_trajectories_are_stacked_in_book_order(monkeypatch) -> None:
    monkeypatch.setattr(
        e09_optimizer_trajectory,
        "runs",
        lambda _client, _group, atomic_ids: {atomic: [] for atomic in atomic_ids},
    )
    monkeypatch.setattr(
        e09_optimizer_trajectory,
        "histories_from_artifact",
        lambda _client, _runs, *, y, **_kwargs: [
            {0.0: -7.0, 1.0: 0.25} if y == "x" else {0.0: 2.0, 1.0: -0.5}
        ],
    )

    figure, _curves = e09_optimizer_trajectory.render(None, "band", None)

    plot_axes = figure.axes[:4]
    assert [axis.get_title() for axis in plot_axes] == [
        "SGD",
        "Momentum",
        "AdaGrad",
        "Adam",
    ]
    assert all(
        upper.get_position().y0 > lower.get_position().y0
        for upper, lower in zip(plot_axes[:-1], plot_axes[1:], strict=True)
    )
    assert all(
        "(0.25, -0.50)" in [text.get_text() for text in axis.texts]
        for axis in plot_axes
    )
    assert all(
        max(line.get_zorder() for line in axis.lines) == 20
        for axis in plot_axes
    )
    plt.close(figure)


def test_objective_contours_use_the_theme_purple_to_green_gradient() -> None:
    figure, axis = plt.subplots()
    grid_x, grid_y = np.meshgrid(
        np.linspace(-10, 10, 21),
        np.linspace(-2.5, 2.5, 21),
    )

    contours = e09_optimizer_trajectory._objective_contours(axis, grid_x, grid_y)

    np.testing.assert_array_equal(contours.levels, np.arange(8, dtype=float))
    assert contours.cmap(0.0) == e09_optimizer_trajectory.OBJECTIVE_CMAP(0.0)
    assert contours.cmap(1.0) == e09_optimizer_trajectory.OBJECTIVE_CMAP(1.0)
    assert contours.cmap(0.0) != contours.cmap(1.0)
    plt.close(figure)
