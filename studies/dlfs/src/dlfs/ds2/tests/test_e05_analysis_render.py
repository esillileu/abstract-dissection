from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dlfs.ds2.analysis import e05_lm_recipes
from repro_core.analysis.core import Curve


def test_e05_lm_recipes_rendering(monkeypatch, tmp_path) -> None:
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([100.0, 80.0]),
        minimum=np.asarray([95.0, 75.0]),
        maximum=np.asarray([105.0, 85.0]),
        run_count=2,
        standard_deviation=np.asarray([2.0, 2.0]),
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {f"{atomic}/valid": curve for atomic in atomic_ids},
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_train_curves",
        lambda _client, atomic_ids: {f"{atomic}/train": curve for atomic in atomic_ids},
    )

    # Main graph
    figure, curves = e05_lm_recipes.render(
        object(), "band", tmp_path / "ds2_e05_imp.png"
    )
    assert set(curves) == {"LM-BETTER-RECIPE/valid", "LM-BETTER-RECIPE/train"}
    assert len(figure.axes) == 1
    plt.close(figure)

    # Additional graphs
    out1 = e05_lm_recipes.render_additional_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )
    assert out1.name == "ds2_e05_all_rnnlm.png"

    out2 = e05_lm_recipes.render_additional_better_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )
    assert out2.name == "ds2_e05_better_rnnlm_dropout.png"
