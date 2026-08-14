from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import Curve
from exp.framework.plotting.theme import ACCENT_COLORS
from exp.deepscratch.ds2.analysis import e06_addition_seq2seq, e09_addition_seq2seq_150


def test_e09_analysis_visualization_uses_gt09_and_150_epoch_axis(
    monkeypatch,
) -> None:
    calls = []

    def fake_runs(_client, group_id, atomic_run_ids):
        calls.append((group_id, tuple(atomic_run_ids)))
        return {atomic_run_id: [object()] for atomic_run_id in atomic_run_ids}

    def fake_source_curve(_client, _run_refs, metric):
        assert metric == "exact_match_accuracy"
        return Curve(
            steps=np.asarray([0, 74, 149], dtype=float),
            mean=np.asarray([0.1, 0.6, 0.9], dtype=float),
            minimum=np.asarray([0.05, 0.55, 0.85], dtype=float),
            maximum=np.asarray([0.15, 0.65, 0.95], dtype=float),
            run_count=2,
            standard_deviation=np.asarray([0.02, 0.03, 0.04], dtype=float),
        )

    monkeypatch.setattr(e06_addition_seq2seq, "runs", fake_runs)
    monkeypatch.setattr(e06_addition_seq2seq, "source_curve", fake_source_curve)

    figure, curves = e09_addition_seq2seq_150.render(
        object(),
        "band",
        None,
    )
    axis = figure.axes[0]

    assert calls == [
        (
            "GT09",
            (
                "SEQA-VAN-FWD", "SEQA-VAN-REV", "SEQA-PEEKY-FWD",
                "SEQA-PEEKY-REV", "SEQA-ATTN-FWD", "SEQA-ATTN-REV",
                "SEQA-ATTN-PEEKY-FWD", "SEQA-ATTN-PEEKY-REV",
            ),
        )
    ]
    assert tuple(curves) == calls[0][1]
    assert len(axis.lines) == 8
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "accuracy"
    assert axis.get_ylim() == (0.0, 1.0)
    assert axis.get_xlim() == (0.0, 149.0)
    legend = axis.get_legend()
    assert legend is not None
    assert legend._loc == 4  # matplotlib's "lower right"
    assert figure.get_size_inches().tolist() == [10.0, 5.0]

    plt.close(figure)


def test_e09_additional_visualization_writes_six_standalone_graphs(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_runs(_client, group_id, atomic_run_ids):
        assert group_id == "GT09"
        return {atomic_run_id: [object()] for atomic_run_id in atomic_run_ids}

    def fake_source_curve(_client, _run_refs, metric):
        assert metric == "exact_match_accuracy"
        return Curve(
            steps=np.asarray([0, 74, 149], dtype=float),
            mean=np.asarray([0.1, 0.6, 0.9], dtype=float),
            minimum=np.asarray([0.05, 0.55, 0.85], dtype=float),
            maximum=np.asarray([0.15, 0.65, 0.95], dtype=float),
            run_count=2,
            standard_deviation=np.asarray([0.02, 0.03, 0.04], dtype=float),
        )

    monkeypatch.setattr(e06_addition_seq2seq, "runs", fake_runs)
    monkeypatch.setattr(e06_addition_seq2seq, "source_curve", fake_source_curve)

    outputs = e09_addition_seq2seq_150.render_additional_graphs(
        object(),
        "band",
        tmp_path / "e09_addition_seq2seq_150.png",
    )

    assert [path.name for path in outputs] == [
        "seq2seq_vanilla_forward_vs_reverse.png",
        "seq2seq_peeky_forward_vs_reverse.png",
        "seq2seq_attention_forward_vs_reverse.png",
        "seq2seq_attention_peeky_forward_vs_reverse.png",
        "seq2seq_peeky_reverse_vs_attention_peeky_reverse.png",
        "seq2seq_vanilla_reverse_vs_attention_reverse.png",
    ]
    assert all(path.is_file() for path in outputs)
    assert all(path.stat().st_size > 0 for path in outputs)


def test_e09_additional_graphs_reuse_full_graph_colors_and_standard_size(
    monkeypatch,
    tmp_path,
) -> None:
    figures = []

    def fake_runs(_client, _group_id, atomic_run_ids):
        return {atomic_run_id: [object()] for atomic_run_id in atomic_run_ids}

    def fake_source_curve(_client, _run_refs, _metric):
        return Curve(
            steps=np.asarray([0, 149], dtype=float),
            mean=np.asarray([0.1, 0.9], dtype=float),
            minimum=np.asarray([0.05, 0.85], dtype=float),
            maximum=np.asarray([0.15, 0.95], dtype=float),
            run_count=2,
            standard_deviation=np.asarray([0.02, 0.04], dtype=float),
        )

    monkeypatch.setattr(e06_addition_seq2seq, "runs", fake_runs)
    monkeypatch.setattr(e06_addition_seq2seq, "source_curve", fake_source_curve)
    monkeypatch.setattr(
        e09_addition_seq2seq_150,
        "save_figure",
        lambda figure, path: (figures.append(figure), path)[1],
    )

    outputs = e09_addition_seq2seq_150.render_additional_graphs(
        object(), "band", tmp_path / "e09.png"
    )

    assert len(outputs) == len(figures) == 6
    assert all(figure.get_size_inches().tolist() == [6.4, 4.8] for figure in figures)
    assert all(figure.axes[0].get_title() == "" for figure in figures)
    first_lines = figures[0].axes[0].lines
    assert [line.get_color() for line in first_lines] == [ACCENT_COLORS[0], ACCENT_COLORS[1]]
    for figure in figures:
        plt.close(figure)
