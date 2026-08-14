"""DS2 GT09: addition Seq2seq accuracy graph with 150-epoch budget."""

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
