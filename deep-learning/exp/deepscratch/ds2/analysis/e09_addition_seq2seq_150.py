"""DS2 GT09: addition Seq2seq accuracy graphs with a 150-epoch budget."""

from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import mark_empty, plot_curve, save_figure
from exp.framework.plotting.theme import ACCENT_COLORS

from . import e06_addition_seq2seq as _e06
from .e06_addition_seq2seq import render as _render_e06_style


DEFINITIONS = (
    ("SEQA-VAN-FWD", "vanilla / forward", "o"),
    ("SEQA-VAN-REV", "vanilla / reverse", "s"),
    ("SEQA-PEEKY-FWD", "peeky / forward", "^"),
    ("SEQA-PEEKY-REV", "peeky / reverse", "D"),
    ("SEQA-ATTN-FWD", "attention / forward", "v"),
    ("SEQA-ATTN-REV", "attention / reverse", "P"),
    ("SEQA-ATTN-PEEKY-FWD", "attention + peeky / forward", "<"),
    ("SEQA-ATTN-PEEKY-REV", "attention + peeky / reverse", ">"),
)

ADDITIONAL_GRAPHS = (
    (
        "ds2_e09_01_van_fwd_vs_van_rev.png",
        "Vanilla Seq2seq: Forward vs. Reverse",
        ("SEQA-VAN-FWD", "SEQA-VAN-REV"),
        "upper left",
    ),
    (
        "ds2_e09_01_pky_fwd_vs_pky_rev.png",
        "Peeky Seq2seq: Forward vs. Reverse",
        ("SEQA-PEEKY-FWD", "SEQA-PEEKY-REV"),
        "upper left",
    ),
    (
        "ds2_e09_01_atn_fwd_vs_atn_rev.png",
        "Attention Seq2seq: Forward vs. Reverse",
        ("SEQA-ATTN-FWD", "SEQA-ATTN-REV"),
        "upper left",
    ),
    (
        "ds2_e09_01_atn_pky_fwd_vs_atn_pky_rev.png",
        "Attention + Peeky Seq2seq: Forward vs. Reverse",
        ("SEQA-ATTN-PEEKY-FWD", "SEQA-ATTN-PEEKY-REV"),
        "upper left",
    ),
    (
        "ds2_e09_02_pky_rev_vs_atn_pky_rev.png",
        "Does Attention Help on Top of Peeky + Reverse?",
        ("SEQA-PEEKY-REV", "SEQA-ATTN-PEEKY-REV"),
        "lower right",
    ),
    (
        "ds2_e09_02_van_rev_vs_atn_rev.png",
        "Vanilla + Reverse vs. Attention + Reverse",
        ("SEQA-VAN-REV", "SEQA-ATTN-REV"),
        "lower right",
    ),
)


def render(client, error_style, output):
    del output
    return _render_e06_style(
        client,
        error_style,
        group_id="GT09",
        max_epoch_index=149,
        definitions=DEFINITIONS,
        figsize=(10, 5),
        legend_loc="lower right",
    )


def render_additional_graphs(client, error_style, output) -> list[Path]:
    """Render the six standalone GT09 comparisons beside the main figure."""
    grouped = _e06.runs(client, "GT09", [item[0] for item in DEFINITIONS])
    curves = {
        atomic: _e06.source_curve(client, grouped[atomic], "exact_match_accuracy")
        for atomic, _label, _marker in DEFINITIONS
    }
    labels = {atomic: label for atomic, label, _marker in DEFINITIONS}
    markers = {atomic: marker for atomic, _label, marker in DEFINITIONS}
    colors = {
        atomic: ACCENT_COLORS[index]
        for index, (atomic, _label, _marker) in enumerate(DEFINITIONS)
    }
    output_dir = Path(output).parent
    outputs = []
    for filename, _title, atomic_ids, legend_loc in ADDITIONAL_GRAPHS:
        figure, axis = plt.subplots()
        for atomic in atomic_ids:
            plot_curve(
                axis,
                curves[atomic],
                label=labels[atomic],
                marker=markers[atomic],
                error_style=error_style,
                error_every=5,
                color=colors[atomic],
            )
        axis.set(
            xlabel="epochs",
            ylabel="accuracy",
            xlim=(0, 149),
            ylim=(0, 1),
        )
        mark_empty(axis)
        if axis.has_data():
            axis.legend(loc=legend_loc)
        path = output_dir / filename
        save_figure(figure, path)
        plt.close(figure)
        outputs.append(path)
    return outputs
