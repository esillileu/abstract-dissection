"""Table 4 examples of qualitative nearest neighbors with additive composition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import write_table_summary
from f2.suites.common.w2v.evaluation import nearest_tokens
from repro_core.context import RuntimePaths

from .. import paper
from .table02 import load_phrase_lookup


def analyze_table4(
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
            output_dir, "No loadable phrase-trained model for Table 4."
        )

    headers = ["Query Word/Phrase", "Published Targets", "Top Nearest Neighbors"]
    rows = []
    results_map = {}
    for query, published_targets in paper.TABLE4.items():
        neighbors = nearest_tokens(lookup, query.encode("utf-8"), limit=5)
        neighbor_tokens = [n.token.decode("utf-8", "replace") for n in neighbors]
        rows.append(
            [
                query,
                ", ".join(published_targets),
                ", ".join(neighbor_tokens) if neighbor_tokens else "<OOV>",
            ]
        )
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
    sources = [corpus_source] if corpus_source else ["wmt", "lm1b", "umbc"]
    return [
        analyze_table4(
            runs,
            output_dir=paths.analysis_results_root / f"f2/w2v2/table04/{source}",
            corpus_source=source,
            paths=paths,
        )
        for source in sources
    ]


__all__ = ["analyze_table4", "analyze_table4_sources"]
