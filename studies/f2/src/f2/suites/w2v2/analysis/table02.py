"""Table 2 qualitative phrase-relationship examples evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import write_table_summary
from f2.suites.common.w2v.lookup import load_lookup_artifact
from repro_core.context import RuntimePaths

from .. import paper


def load_phrase_lookup(runs: list[Any], corpus_source: str) -> tuple[str | None, Any]:
    """Select a loadable phrase model for the requested corpus."""
    for r in runs:
        tags = r.get("tags") or {}
        if tags.get("experiment_spec.id") not in {
            "w2v2-phrase-skipgram-1b-objectives",
            "w2v2-section4-phrase-6b",
            "w2v2-section4-phrase-33b",
        }:
            continue
        variant = tags.get("implementation.variant", "")
        if (tags.get("corpus.id") or variant.split("--", 1)[0]) != corpus_source:
            continue
        art_root = Path(r.get("artifact_root", ""))
        try:
            result = json.loads((art_root / "result.json").read_text())
            lookup = (art_root / result["artifacts"]["lookup"]).resolve()
            if not lookup.is_relative_to(art_root.resolve()):
                continue
            return r.get("run_id"), load_lookup_artifact(lookup)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def analyze_table2(
    runs: list[Any],
    *,
    output_dir: Path,
    corpus_source: str = "news-1b",
    paths: RuntimePaths | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths or RuntimePaths.from_environment()

    selected_run_id, lookup = load_phrase_lookup(runs, corpus_source)
    if lookup is None:
        return unavailable_report(
            output_dir, "No loadable phrase-trained model for Table 2."
        )

    headers = ["Relationship", "a", "b", "c", "Published Target (d)", "Predicted"]
    rows = []
    from f2.suites.common.w2v.geometry import _nearest_query, _normalized_embeddings

    for row in paper.TABLE2:
        category, a_s, b_s, c_s, d_s = row
        a_b, b_b, c_b = a_s.encode("utf-8"), b_s.encode("utf-8"), c_s.encode("utf-8")
        ra, rb, rc = lookup.row(a_b), lookup.row(b_b), lookup.row(c_b)
        pred_token = "<OOV>"
        if ra is not None and rb is not None and rc is not None:
            va = _normalized_embeddings(lookup.embeddings[[ra]])[0]
            vb = _normalized_embeddings(lookup.embeddings[[rb]])[0]
            vc = _normalized_embeddings(lookup.embeddings[[rc]])[0]
            q_vec = vb - va + vc
            norm = float(np.linalg.norm(q_vec))
            if norm > 0.0:
                neighbors = _nearest_query(lookup, q_vec / norm, {ra, rb, rc}, 1)
                if neighbors:
                    pred_token = neighbors[0].token.decode("utf-8", "replace")
        rows.append([category, a_s, b_s, c_s, d_s, pred_token])

    write_table_summary(
        output_dir,
        title="W2V2 Table 2: Examples of Qualitative Phrase Analogies",
        paper_reference=paper.SOURCE,
        headers=headers,
        rows=rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "paper": paper.SOURCE,
                "table": "2",
                "mlflow_run_id": selected_run_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def analyze_table2_sources(
    cache_root: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> list[Path]:
    paths = paths or RuntimePaths.from_environment()
    runs_file = cache_root / "runs.json"
    if not runs_file.is_file():
        return []
    runs = json.loads(runs_file.read_text(encoding="utf-8"))
    sources = [corpus_source] if corpus_source else ["wmt", "lm1b", "umbc"]
    return [
        analyze_table2(
            runs,
            output_dir=paths.analysis_results_root / f"f2/w2v2/table02/{source}",
            corpus_source=source,
            paths=paths,
        )
        for source in sources
    ]


__all__ = ["analyze_table2", "analyze_table2_sources", "load_phrase_lookup"]
