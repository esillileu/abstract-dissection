"""CSV/NPZ and plotting helpers; intentionally no runner/source imports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp.original.cache import load_csv, load_npz
from exp.plot_theme import apply_plot_theme


apply_plot_theme()


def trial(root: Path, experiment: str, trial_id: str) -> Path:
    return root / "data" / experiment / trial_id


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


__all__ = ["load_csv", "load_npz", "np", "plt", "save", "trial"]
