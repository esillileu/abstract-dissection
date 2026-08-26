"""Render the original four-panel optimizer trajectories."""

from pathlib import Path

from repro_core.plotting.theme import ACCENT_COLORS, CORE_HIGHLIGHT

from .common import load_csv, np, plt, save, trial

TRIAL_IDS = tuple(
    f"dlfs1.ch06.optimizer-path.{name}"
    for name in ("sgd", "momentum", "adagrad", "adam")
)


def render(root: Path, image_dir: Path) -> list[Path]:
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    grid_x, grid_y = np.meshgrid(x, y)
    z = grid_x**2 / 20.0 + grid_y**2
    z[z > 7] = 0
    for index, (trial_id, title) in enumerate(
        zip(TRIAL_IDS, ("SGD", "Momentum", "AdaGrad", "Adam"), strict=True),
        start=1,
    ):
        rows = load_csv(trial(root, "e09", trial_id) / "trajectory.csv")
        plt.subplot(2, 2, index)
        plt.plot(
            [float(row["x"]) for row in rows],
            [float(row["y"]) for row in rows],
            "o-",
            color=CORE_HIGHLIGHT,
        )
        plt.contour(grid_x, grid_y, z, colors=ACCENT_COLORS[4])
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, "+")
        plt.title(f"Original {title}")
        plt.xlabel("x")
        plt.ylabel("y")
    path = image_dir / "e09_optimizer_compare_naive.png"
    save(path)
    return [path]
