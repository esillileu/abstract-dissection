from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from exp.framework.analysis.core import Curve
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
            ),
        )
    ]
    assert tuple(curves) == calls[0][1]
    assert len(axis.lines) == 6
    assert axis.get_xlabel() == "epochs"
    assert axis.get_ylabel() == "accuracy"
    assert axis.get_ylim() == (0.0, 1.0)
    assert axis.get_xlim() == (0.0, 149.0)
    assert axis.get_legend() is not None

    plt.close(figure)
