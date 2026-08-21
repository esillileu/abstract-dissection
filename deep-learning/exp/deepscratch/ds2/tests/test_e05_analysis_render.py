from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import Curve
from exp.deepscratch.identity import Variant
from exp.deepscratch.ds2.analysis import e05_lm_recipes


def test_e05_main_graph_is_better_train_and_validation(monkeypatch, tmp_path) -> None:
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
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_train_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/train": curve for atomic in atomic_ids
        },
    )

    figure, curves = e05_lm_recipes.render(
        object(), "band", tmp_path / "ds2_e05_imp.png"
    )

    assert set(curves) == {"LM-BETTER-RECIPE/valid", "LM-BETTER-RECIPE/train"}
    assert len(figure.axes) == 1
    assert figure.get_size_inches().tolist() == [6.4, 4.8]
    assert figure._analysis_match_original_canvas is True
    assert len(figure.axes[0].lines) == 2
    assert figure.axes[0].get_ylim() == (0, 250)
    plt.close(figure)


def test_e05_original_main_graph_is_validation_only(monkeypatch, tmp_path) -> None:
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
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_train_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/train": curve for atomic in atomic_ids
        },
    )
    data = type("OriginalInput", (), {"variant": Variant.ORIGINAL})()

    figure, curves = e05_lm_recipes.render(
        data, "band", tmp_path / "ds2_e05_org.png"
    )

    assert set(curves) == {"LM-BETTER-RECIPE/valid"}
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].get_ylim() == (0, 250)
    plt.close(figure)


def test_e05_additional_graph_contains_all_recipes(monkeypatch, tmp_path) -> None:
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([100.0, 80.0]),
        minimum=np.asarray([95.0, 75.0]),
        maximum=np.asarray([105.0, 85.0]),
        run_count=2,
        standard_deviation=np.asarray([2.0, 2.0]),
    )
    figures = []
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "save_figure",
        lambda figure, path: figures.append(figure) or path,
    )

    output = e05_lm_recipes.render_additional_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )

    assert output.name == "ds2_e05_all_rnnlm.png"
    assert len(figures) == 1
    assert len(figures[0].axes) == 2
    assert figures[0].get_size_inches().tolist() == [9.0, 6.0]
    upper, lower = figures[0].axes
    assert len(upper.lines) == 1
    assert len(lower.lines) == 4
    assert upper.get_yscale() == "symlog"
    assert lower.get_yscale() == "linear"
    legend_labels = [text.get_text() for text in upper.get_legend().get_texts()]
    assert [label.split(" (n=", 1)[0] for label in legend_labels] == [
        "Vanilla RNNLM",
        "LSTM RNNLM",
        "Tied RNNLM",
        "Better RNNLM",
        "Better RNNLM (no dropout)",
    ]
    assert lower.get_xlabel() == "epochs"
    plt.close(figures[0])


def test_e05_better_graph_is_single_axis_without_wave_break(monkeypatch, tmp_path) -> None:
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([100.0, 80.0]),
        minimum=np.asarray([95.0, 75.0]),
        maximum=np.asarray([105.0, 85.0]),
        run_count=2,
        standard_deviation=np.asarray([2.0, 2.0]),
    )
    figures = []
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "_train_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/train": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "save_figure",
        lambda figure, path: figures.append(figure) or path,
    )

    output = e05_lm_recipes.render_additional_better_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )

    assert output.name == "ds2_e05_better_rnnlm_dropout.png"
    assert len(figures) == 1
    assert len(figures[0].axes) == 1
    axis = figures[0].axes[0]
    assert axis.get_ylim() == (0, 500)
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert [label.split(" (n=", 1)[0] for label in labels] == [
        "Better RNNLM",
        "Better RNNLM (train)",
        "Better RNNLM (no dropout)",
        "Better RNNLM (no dropout) (train)",
    ]
    assert [line.get_linestyle() for line in axis.lines] == ["-", ":", "-", ":"]
    assert axis.lines[0].get_color() == axis.lines[1].get_color()
    assert axis.lines[2].get_color() == axis.lines[3].get_color()
    plt.close(figures[0])


def test_e05_better_validation_graph_is_standalone(monkeypatch, tmp_path) -> None:
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([100.0, 80.0]),
        minimum=np.asarray([95.0, 75.0]),
        maximum=np.asarray([105.0, 85.0]),
        run_count=2,
        standard_deviation=np.asarray([2.0, 2.0]),
    )
    figures = []
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "save_figure",
        lambda figure, path: figures.append(figure) or path,
    )

    output = e05_lm_recipes.render_additional_better_validation_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )

    assert output.name == "ds2_e05_better_rnnlm_validation.png"
    assert len(figures) == 1
    assert figures[0].get_size_inches().tolist() == [6.4, 4.8]
    axis = figures[0].axes[0]
    assert len(figures[0].axes) == 1
    assert len(axis.lines) == 1
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert [label.split(" (n=", 1)[0] for label in labels] == ["Better RNNLM"]
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "perplexity"
    plt.close(figures[0])


def test_e05_lstm_graph_uses_validation_only(monkeypatch, tmp_path) -> None:
    curve = Curve(
        steps=np.asarray([0.0, 1.0]),
        mean=np.asarray([100.0, 80.0]),
        minimum=np.asarray([95.0, 75.0]),
        maximum=np.asarray([105.0, 85.0]),
        run_count=2,
        standard_deviation=np.asarray([2.0, 2.0]),
    )
    figures = []
    monkeypatch.setattr(
        e05_lm_recipes,
        "_valid_curves",
        lambda _client, atomic_ids: {
            f"{atomic}/valid": curve for atomic in atomic_ids
        },
    )
    monkeypatch.setattr(
        e05_lm_recipes,
        "save_figure",
        lambda figure, path: figures.append(figure) or path,
    )

    output = e05_lm_recipes.render_additional_lstm_graph(
        object(), "band", tmp_path / "ds2_e05.png"
    )

    assert output.name == "ds2_e05_lstm_vs_tied_rnnlm.png"
    axis = figures[0].axes[0]
    assert len(figures[0].axes) == 1
    assert len(axis.lines) == 2
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert [label.split(" (n=", 1)[0] for label in labels] == [
        "RNNLM",
        "RNNLM(weight tying)",
    ]
    assert all(line.get_linestyle() == "-" for line in axis.lines)
    plt.close(figures[0])
