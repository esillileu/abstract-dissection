"""DS2 book-layout renderer registry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from exp.framework.analysis.core import save_figure, write_summary

from . import (
    e01_toy_word2vec,
    e02_ptb_word2vec,
    e03_small_rnnlm,
    e04_lstm_rnnlm,
    e05_lm_recipes,
    e06_addition_seq2seq,
    e07_date_seq2seq,
    e08_attention,
    e09_addition_seq2seq_150,
    e10_word2vec_profile,
    e11_vocabulary_size_scaling,
    e12_count_based_embeddings,
)


RENDERERS = {
    "e01": e01_toy_word2vec.render,
    "e02": e02_ptb_word2vec.render,
    "e03": e03_small_rnnlm.render,
    "e04": e04_lstm_rnnlm.render,
    "e05": e05_lm_recipes.render,
    "e06": e06_addition_seq2seq.render,
    "e07": e07_date_seq2seq.render,
    "e08": e08_attention.render,
    "e09": e09_addition_seq2seq_150.render,
    "e10": e10_word2vec_profile.render,
    "e11": e11_vocabulary_size_scaling.render,
    "e12": e12_count_based_embeddings.render,
}
ADDITIONAL_RENDERERS = {
    "e05": lambda data, error_style, output: [
        e05_lm_recipes.render_additional_graph(data, error_style, output),
        e05_lm_recipes.render_additional_better_graph(data, error_style, output),
        e05_lm_recipes.render_additional_lstm_graph(data, error_style, output),
    ],
    "e09": e09_addition_seq2seq_150.render_additional_graphs,
}
STUDY_SOURCES = {}
MARKDOWN_APPENDERS = {
    "e02": e02_ptb_word2vec.append_markdown_report,
    "e12": e12_count_based_embeddings.append_markdown_report,
}


def render_study(
    data,
    study_id: str,
    output: Path,
    *,
    error_style: str = "band",
) -> list[Path]:
    """Render the book's study-specific composition with project error bars."""
    result = RENDERERS[study_id](data, error_style, output)
    if isinstance(result, list):
        return result
    figure, curves = result
    save_figure(figure, output)
    outputs = [output]
    additional_renderer = ADDITIONAL_RENDERERS.get(study_id)
    if additional_renderer is not None:
        outputs.extend(additional_renderer(data, error_style, output))
    summary = output.with_name(f"{output.stem}_curves.csv")
    write_summary(summary, curves)
    plt.close(figure)
    return [*outputs, summary]


__all__ = ["ADDITIONAL_RENDERERS", "RENDERERS", "STUDY_SOURCES", "render_study"]
