"""Table 2 qualitative phrase-relationship examples evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro_core.context import RuntimePaths

from f2.storage.analysis_cache import unavailable_report
from f2.suites.common.w2v.analysis import write_table_summary
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
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
        r for r in runs
        if ("e02" in (r.get("artifact_root") or "") or "table3" in (r.get("artifact_root") or "")
            or "section4" in (r.get("artifact_root") or "") or "e03" in (r.get("artifact_root") or "")
            or "e04" in (r.get("artifact_root") or ""))
    ]
    if not matching_runs:
        return unavailable_report(
            output_dir,
            paper_reference=paper.SOURCE,
            table="2",
            reasons=["No phrase-trained models available for Table 2 relationship queries."],
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
            paper_reference=paper.SOURCE,
            table="2",
            reasons=["Could not load lookup artifact from any phrase run."],
        )

    from f2.suites.common.w2v.evaluation import evaluate_analogies, AnalogyQuestion

    questions = [
        AnalogyQuestion(
            category=row[0],
            a=row[1].encode("utf-8"),
            b=row[2].encode("utf-8"),
            c=row[3].encode("utf-8"),
            d=row[4].encode("utf-8"),
        )
        for row in paper.TABLE2
    ]
    eval_res = evaluate_analogies(lookup, questions)

    headers = ["Relationship", "a", "b", "c", "Published Target (d)", "Predicted"]
    rows = []
    for q, pred in zip(questions, eval_res.predictions):
        rows.append([
            q.category,
            q.a.decode("utf-8", "replace"),
            q.b.decode("utf-8", "replace"),
            q.c.decode("utf-8", "replace"),
            q.d.decode("utf-8", "replace"),
            pred.decode("utf-8", "replace") if pred else "<OOV>",
        ])

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
                "total_accuracy_percent": eval_res.total_accuracy_percent,
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
    out = paths.analysis_results("f2", "w2v2", "table02", c_source)
    return [analyze_table2(runs, output_dir=out, corpus_source=c_source, paths=paths)]


__all__ = ["analyze_table2", "analyze_table2_sources"]
