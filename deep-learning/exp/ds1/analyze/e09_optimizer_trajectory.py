"""DS1 GO01: reproduce four optimizer trajectories on the objective contour."""

import matplotlib.pyplot as plt
import numpy as np

from exp.analyze import aggregate, histories_from_artifact, mark_empty

from .common import runs


def render(client, error_style, output):
    del output
    definitions = [
        ("TOY-SGD", "SGD"),
        ("TOY-MOMENTUM", "Momentum"),
        ("TOY-ADAGRAD", "AdaGrad"),
        ("TOY-ADAM", "Adam"),
    ]
    grouped = runs(client, "GO01", [item[0] for item in definitions])
    figure, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    curves = {}
    grid_x, grid_y = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-10, 10, 0.01))
    objective = grid_x**2 / 20 + grid_y**2
    objective[objective > 7] = 0
    for axis, (atomic, title) in zip(axes.flat, definitions, strict=True):
        x_curve = aggregate(
            histories_from_artifact(
                client, grouped[atomic], artifact_path="observations/trajectory.csv", x="update", y="x"
            )
        )
        y_curve = aggregate(
            histories_from_artifact(
                client, grouped[atomic], artifact_path="observations/trajectory.csv", x="update", y="y"
            )
        )
        curves[f"{atomic}/x"] = x_curve
        curves[f"{atomic}/y"] = y_curve
        if len(x_curve.steps) and np.array_equal(x_curve.steps, y_curve.steps):
            axis.plot(x_curve.mean, y_curve.mean, "o-", color="red", ms=2, label=f"mean (n={x_curve.run_count})")
            if error_style == "band":
                axis.fill_between(x_curve.mean, y_curve.minimum, y_curve.maximum, color="red", alpha=0.2)
            else:
                axis.errorbar(
                    x_curve.mean,
                    y_curve.mean,
                    xerr=np.vstack((x_curve.mean - x_curve.minimum, x_curve.maximum - x_curve.mean)),
                    yerr=np.vstack((y_curve.mean - y_curve.minimum, y_curve.maximum - y_curve.mean)),
                    fmt="none",
                    ecolor="red",
                    elinewidth=0.7,
                    capsize=1,
                )
        axis.contour(grid_x, grid_y, objective)
        axis.plot(0, 0, "+")
        axis.set(xlim=(-10, 10), ylim=(-10, 10), title=title, xlabel="x", ylabel="y")
        if not len(x_curve.steps):
            mark_empty(axis)
    return figure, curves
