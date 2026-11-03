"""Render and write summary markdown, json, and csv reports."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .. import paper
from .cache import EVALUATION_PROTOCOL, SEEDS, status_from_items


def write_analysis_deliverables(
    output: Path,
    *,
    result_source: str,
    active_tables: tuple[str, ...],
    target_table: str | None,
    target_corpus: str | None,
    records: list[dict[str, Any]],
    datasets: dict[str, Any],
    inventory: list,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    qualitative: list[dict[str, Any]],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    table_status = {}
    for number in active_tables:
        if number == "2":
            states = [datasets["phrase"][0]["status"]]
        elif number in {"4", "5", "6"}:
            states = [q["status"] for q in qualitative if q["table"] == number]
        else:
            states = [a["status"] for a in aggregates if a["table"] == number]
        table_status[number] = status_from_items(states)

    complete = all(status == "complete" for status in table_status.values())
    analysis_signature = {
        "run_ids": sorted({r.get("run_id") for r in records if r.get("run_id")}),
        "table_filter": target_table,
        "corpus_filter": target_corpus,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "eval_datasets": {
            k: {
                "status": v[0].get("status"),
                "sha256": v[0].get("sha256") or v[0].get("questions_sha256"),
                "count": v[0].get("count") or v[0].get("total_questions"),
            }
            for k, v in datasets.items()
        },
    }
    payload = {
        "result_source": result_source,
        "source": paper.SOURCE,
        "analysis_signature": analysis_signature,
        "status": "complete" if complete else "incomplete",
        "report_generated": True,
        "experiment_complete": complete,
        "table_status": table_status,
        "quantitative_status": status_from_items(a["status"] for a in aggregates),
        "qualitative_status": status_from_items(q["status"] for q in qualitative),
        "identity_diagnostics_status": "complete",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "seeds": SEEDS,
        "datasets": {k: v[0] for k, v in datasets.items()},
        "runs": rows,
        "aggregates": aggregates,
        "qualitative": qualitative,
        "diagnostics": [],
        "paper": {
            "table2": paper.TABLE2,
            "table4": paper.TABLE4,
            "table5": paper.TABLE5,
            "table6": paper.TABLE6,
            "table6_queries": paper.TABLE6_QUERIES,
        },
        "external_baselines": {
            "status": "unsupported",
            "reason": "reference-only; external models are not retrained",
        },
    }

    table_exports = []
    for number in active_tables:
        data = [
            {"kind": "measurement", **r}
            for r in (rows if number in {"1", "3", "section4.1"} else qualitative)
            if r["table"] == number
        ]
        if number == "2":
            data.append(
                {"kind": "evaluation_validation", **payload["datasets"]["phrase"]}
            )
            data.extend(
                {"kind": "paper_reference", "category": r[0], "example": r[1:]}
                for r in paper.TABLE2
            )
        elif number in {"4", "5", "6"}:
            data.extend(
                {
                    "kind": "paper_reference",
                    "query_or_model": k,
                    "reference": v,
                    "reference_columns": paper.TABLE6_QUERIES
                    if number == "6"
                    else None,
                }
                for k, v in getattr(paper, f"TABLE{number}").items()
            )
        table_exports.append((f"table{number}", data))

    for name, data in [
        ("summary", aggregates),
        ("runs", rows),
        ("qualitative", qualitative),
        *table_exports,
    ]:
        fields = sorted({k for r in data for k in r}) or ["status"]
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (dict, list, tuple))
                    else v
                    for k, v in r.items()
                }
                for r in data
            )

    def _metric_str(item_runs, key):
        values = [r[key] for r in item_runs if r.get(key) is not None]
        if not values:
            return "—"
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.2f} ± {sd:.2f}"

    lines = [
        "| Corpus | Condition | total | semantic | syntactic | training_times(s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    seen_conditions = set()
    for _, corpus, condition, _, _ in inventory:
        cond_key = (corpus, condition)
        if cond_key in seen_conditions:
            continue
        seen_conditions.add(cond_key)
        cond_runs = [
            r for r in rows if r["corpus"] == corpus and r["condition"] == condition
        ]
        tot = _metric_str(cond_runs, "total")
        sem = _metric_str(cond_runs, "semantic")
        syn = _metric_str(cond_runs, "syntactic")
        sec = _metric_str(cond_runs, "training_seconds")
        lines.append(f"| {corpus} | {condition} | {tot} | {sem} | {syn} | {sec} |")

    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return output


__all__ = ["write_analysis_deliverables"]
