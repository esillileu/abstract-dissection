from __future__ import annotations

import sys
from pathlib import Path

from deepscratch.datasets import load_ptb as _default_load_ptb

from ..common import runs
from .constants import ATOMIC_RUN_IDS
from .curves import _render_ns_curves
from .evaluation import (
    RunEvaluation,
    _checkpoint_weights_path,
    _word_vectors,
    evaluate_vectors,
)
from .reporting import _output_paths, _text, _write_csv


def _get_load_ptb():
    pkg = sys.modules.get("dlfs.ds2.analysis.e02_ptb_word2vec")
    if pkg is not None and hasattr(pkg, "load_ptb"):
        return pkg.load_ptb
    return _default_load_ptb


def render(client, error_style, output: Path) -> list[Path]:
    grouped = runs(client, "GT02", list(ATOMIC_RUN_IDS))
    ptb = _get_load_ptb()()
    word_to_id = ptb["word_to_id"]
    id_to_word = ptb["id_to_word"]
    evaluations: list[RunEvaluation] = []
    missing: dict[str, int] = {}
    for series in ATOMIC_RUN_IDS:
        for run in grouped[series]:
            checkpoint = _checkpoint_weights_path(client, run)
            if checkpoint is None:
                missing[series] = missing.get(series, 0) + 1
                continue
            try:
                vectors = _word_vectors(checkpoint)
                evaluations.append(
                    evaluate_vectors(
                        series,
                        run.seed,
                        run.run_id,
                        vectors,
                        word_to_id,
                        id_to_word,
                    )
                )
            except (OSError, ValueError):
                missing[series] = missing.get(series, 0) + 1
    variant = getattr(client, "variant", None)
    variant_name = None if variant is None else variant.value
    text = _text(evaluations, missing, variant=variant_name)
    text_path, csv_path = _output_paths(output)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    _write_csv(csv_path, evaluations)
    return [*_render_ns_curves(client, error_style, output), text_path, csv_path]
