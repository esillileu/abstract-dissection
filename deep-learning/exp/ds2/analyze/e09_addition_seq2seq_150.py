"""DS2 GT09: addition Seq2seq accuracy graph with 150-epoch budget."""

from .e06_addition_seq2seq import render as _render_e06_style


def render(client, error_style, output):
    del output
    return _render_e06_style(client, error_style, group_id="GT09")
