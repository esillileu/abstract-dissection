"""Table 2 qualitative phrase-relationship examples evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import write_table_summary
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from repro_core.context import RuntimePaths

from .. import paper


def analyze_table2(
    runs: list[Any],
    *,
    output_dir: Path,
    corpus_source: str = "news-1b",
    paths: RuntimePaths | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths or RuntimePaths.from_environment()

    # Find phrase-capable runs (table3 or section4)
    matching_runs = [
        r
        for r in runs
        if (
            "e02" in (r.get("artifact_root") or "")
            or "table3" in (r.get("artifact_root") or "")
            or "section4" in (r.get("artifact_root") or "")
            or "e03" in (r.get("artifact_root") or "")
            or "e04" in (r.get("artifact_root") or "")
        )
    ]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            "No phrase-trained models available for Table 2 relationship queries.",
        )

    # Use first available run with valid lookup
    lookup = None
    selected_run_id = None
    for r in matching_runs:
        art_root = Path(r.get("artifact_root", ""))
        lookup_dir = art_root / "lookup"
        if not lookup_dir.is_dir():
            lookup_dir = art_root
        try:
            lookup = load_lookup_artifact(lookup_dir)
            selected_run_id = r.get("run_id")
            break
        except Exception:
            continue

    if lookup is None:
        return unavailable_report(
            output_dir,
            "Could not load lookup artifact from any phrase run.",
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
    c_source = corpus_source or "news-1b"
    out = paths.analysis_results_root / f"f2/w2v2/table02/{c_source}"
    return [analyze_table2(runs, output_dir=out, corpus_source=c_source, paths=paths)]


__all__ = ["analyze_table2", "analyze_table2_sources"]
