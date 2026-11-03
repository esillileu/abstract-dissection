"""Table 4 examples of qualitative nearest neighbors with additive composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro_core.context import RuntimePaths

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import write_table_summary
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import nearest_tokens
from .. import paper


def analyze_table4(
    runs: list[Any],
    *,
    output_dir: Path,
    corpus_source: str = "news-1b",
    paths: RuntimePaths | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = paths or RuntimePaths.from_environment()

    matching_runs = [
        r for r in runs
        if ("e02" in (r.get("artifact_root") or "") or "table3" in (r.get("artifact_root") or "")
            or "section4" in (r.get("artifact_root") or "") or "e03" in (r.get("artifact_root") or "")
            or "e04" in (r.get("artifact_root") or ""))
    ]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            paper_reference=paper.SOURCE,
            table="4",
            reasons=["No phrase-trained models available for Table 4 queries."],
        )

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
            paper_reference=paper.SOURCE,
            table="4",
            reasons=["Could not load lookup artifact from any phrase run."],
        )

    headers = ["Query Word/Phrase", "Published Targets", "Top Nearest Neighbors"]
    rows = []
    results_map = {}
    for query, published_targets in paper.TABLE4.items():
        neighbors = nearest_tokens(lookup, query.encode("utf-8"), limit=5)
        neighbor_tokens = [n.token.decode("utf-8", "replace") for n in neighbors]
        rows.append([
            query,
            ", ".join(published_targets),
            ", ".join(neighbor_tokens) if neighbor_tokens else "<OOV>",
        ])
        results_map[query] = {
            "targets": list(published_targets),
            "neighbors": neighbor_tokens,
        }

    write_table_summary(
        output_dir,
        title="W2V2 Table 4: Nearest Neighbors for Query Words and Phrases",
        paper_reference=paper.SOURCE,
        headers=headers,
        rows=rows,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "paper": paper.SOURCE,
                "table": "4",
                "mlflow_run_id": selected_run_id,
                "queries": results_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def analyze_table4_sources(
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
    out = paths.analysis_results("f2", "w2v2", "table04", c_source)
    return [analyze_table4(runs, output_dir=out, corpus_source=c_source, paths=paths)]


__all__ = ["analyze_table4", "analyze_table4_sources"]
