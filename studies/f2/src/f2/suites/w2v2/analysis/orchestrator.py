"""Orchestrate evaluation and produce W2V2 reproduction reports."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from repro_core.context import RuntimePaths

from .. import analysis as analysis_mod
from .. import paper
from .cache import (
    SEEDS,
    W2V2EvaluationCache,
    status_from_items,
    summarize,
    targets,
)
from .evaluator import (
    evaluate_single_seed,
    filter_inventory,
    index_records_by_key,
    paper_reference,
    questions_sha256,
    resolve_question_files,
)
from .qualitative import evaluate_qualitative_query, queries_for
from .reporting import write_analysis_deliverables


def generate_report(
    *,
    records=(),
    paths=None,
    word_questions=None,
    phrase_questions=None,
    result_source="supplied records",
    table_filter: str | int | None = None,
    corpus_filter: str | None = None,
    use_evaluation_cache: bool = False,
) -> Path:
    """Evaluate only independently verified stored runs; report writing is not run completion."""
    paths = paths or RuntimePaths.from_environment()
    records = list(records)
    eval_cache = (
        W2V2EvaluationCache(
            paths.analysis_cache("f2", "w2v2") / "eval_cache"
            if hasattr(paths, "analysis_cache")
            else paths.cache_root / "f2/w2v2/eval_cache"
        )
        if use_evaluation_cache
        else None
    )

    w_q, p_q = resolve_question_files(paths, records, word_questions, phrase_questions)
    datasets = {
        "word": analysis_mod._dataset(w_q, False),
        "phrase": analysis_mod._dataset(p_q, True),
    }
    full_inventory = list(targets())
    target_table = str(table_filter).lower() if table_filter is not None else None
    if target_table in {"section4", "4.1"}:
        target_table = "section4.1"
    target_corpus = str(corpus_filter).lower() if corpus_filter is not None else None

    inventory = filter_inventory(full_inventory, target_table, target_corpus)
    records_by_key, conflicts = index_records_by_key(records, full_inventory)

    rows, qualitative = [], []
    total_eval_runs = len(inventory) * len(SEEDS)
    pbar = tqdm(total=total_eval_runs, desc="Evaluating W2V2 runs", unit="run")

    for table, corpus, condition, variant, raw in inventory:
        from ..spec import parse_run_spec

        config = {
            **parse_run_spec(raw["_path"], atomic_run_id=variant).to_executor_config(),
            "_path": raw["_path"],
        }
        dataset, questions = datasets["word" if table == "1" else "phrase"]
        q_sha = questions_sha256(dataset, questions)
        reference = paper_reference(table, condition)

        for seed in SEEDS:
            pbar.set_postfix_str(f"{corpus} {condition} (s{seed})")
            row, lookup, cached_eval = evaluate_single_seed(
                table,
                corpus,
                condition,
                variant,
                seed,
                reference,
                config,
                dataset,
                questions,
                q_sha,
                records_by_key,
                conflicts,
                eval_cache,
            )
            rows.append(row)

            new_qual_map = _eval_qualitative(
                table, corpus, condition, seed, row, lookup, cached_eval, qualitative
            )

            if (
                eval_cache is not None
                and lookup is not None
                and questions
                and cached_eval is None
                and (table, variant, seed) in records_by_key
            ):
                rec = records_by_key[(table, variant, seed)]
                eval_cache.store(
                    rec["run_id"],
                    table=table,
                    condition=condition,
                    questions_sha256=q_sha,
                    lookup_digest=row.get("provenance", {}).get("engine_config_digest")
                    or row.get("provenance", {}).get("vocabulary_digest"),
                    result={"metrics": row["metrics"], "qualitative": new_qual_map},
                )
            pbar.update(1)
    pbar.close()

    qualitative.sort(
        key=lambda q: (q["table"], q["corpus"], q["condition"], q["seed"], q["query"])
    )
    aggregates = _compute_aggregates(inventory, rows)

    all_table_numbers = ("1", "2", "3", "4", "5", "6", "section4.1")
    active_tables = (
        (target_table,) if target_table in all_table_numbers else all_table_numbers
    )

    output = (
        paths.analysis_output("f2", "w2v2")
        if hasattr(paths, "analysis_output")
        else paths.cache_root / "f2/w2v2/analysis"
    )
    if target_corpus is not None:
        output = output / target_corpus
    if target_table is not None:
        tbl_dir = (
            f"table{target_table}" if target_table != "section4.1" else "section4_1"
        )
        output = output / tbl_dir

    return write_analysis_deliverables(
        output,
        result_source=result_source,
        active_tables=active_tables,
        target_table=target_table,
        target_corpus=target_corpus,
        records=records,
        datasets=datasets,
        inventory=inventory,
        rows=rows,
        aggregates=aggregates,
        qualitative=qualitative,
    )


def _eval_qualitative(
    table, corpus, condition, seed, row, lookup, cached_eval, qualitative
):
    qual_cache_map = (
        cached_eval.get("qualitative", {}) if cached_eval is not None else {}
    )
    new_qual_map = {}
    for number, query, tokens, limit in queries_for(table, condition):
        q_key = f"{number}:{query}"
        output = {
            "table": number,
            "corpus": corpus,
            "condition": condition,
            "seed": seed,
            "run_id": row.get("run_id", ""),
            "checkpoint": row.get("checkpoint", ""),
            "checkpoint_reused": False,
            "query": query,
            "corpus_class": "substitute",
            "composition_protocol": "reconstruction decision",
            "candidate_scope": "full checkpoint vocabulary",
            "query_exclusion": "exclude input token rows",
            "tie_breaking": "cosine descending",
            "status": row["status"] if lookup is None else "no_data",
            "reason": row["reason"],
            "neighbors": [],
        }
        if lookup is not None:
            if q_key in qual_cache_map:
                output.update(qual_cache_map[q_key])
                new_qual_map[q_key] = qual_cache_map[q_key]
            else:
                res = evaluate_qualitative_query(lookup, number, query, tokens, limit)
                output.update(res)
                new_qual_map[q_key] = res
        qualitative.append(output)
    return new_qual_map


def _compute_aggregates(inventory, rows):
    aggregates = []
    for table, corpus, condition, _, _ in inventory:
        group = [
            r
            for r in rows
            if (r["table"], r["corpus"], r["condition"]) == (table, corpus, condition)
        ]
        metrics_to_compute = (
            ("total", "semantic", "syntactic", "training_minutes", "training_seconds")
            if table == "1"
            else ("total", "training_minutes", "training_seconds")
        )
        for metric in metrics_to_compute:
            valid = [r for r in group if r.get(metric) is not None]
            t1_refs = (
                dict(
                    zip(
                        ("training_minutes", "syntactic", "semantic", "total"),
                        paper.TABLE1[condition],
                        strict=True,
                    )
                )
                if table == "1"
                else {}
            )
            if metric == "training_seconds":
                reference = (
                    t1_refs.get("training_minutes", 0) * 60.0 if table == "1" else None
                )
            elif table == "1":
                reference = t1_refs.get(metric)
            else:
                reference = group[0]["paper_reference"] if metric == "total" else None

            summary = summarize([r[metric] for r in valid])
            hardware = (
                [r["timing"]["hardware"] for r in valid if r["timing"].get("hardware")]
                if metric == "training_minutes"
                else []
            )
            aggregates.append(
                {
                    "table": table,
                    "corpus": corpus,
                    "corpus_class": "substitute",
                    "condition": condition,
                    "metric": metric,
                    "paper_reference": reference,
                    "status": status_from_items(
                        "complete"
                        if r.get(metric) is not None
                        else r["status"]
                        if r["identity_status"] != "complete"
                        else r["timing"]["status"]
                        if metric == "training_minutes"
                        else r["metrics"].get(metric, {}).get("status", "no_data")
                        for r in group
                    ),
                    "missing_seeds": [
                        s for s in SEEDS if s not in [r["seed"] for r in valid]
                    ],
                    "hardware_environments": hardware,
                    "hardware_comparable_across_seeds": len(
                        {json.dumps(h, sort_keys=True) for h in hardware}
                    )
                    <= 1,
                    "paper_hardware_comparability": "unverified; original hardware not recorded here"
                    if metric == "training_minutes"
                    else None,
                    **summary,
                    "difference_from_paper": (
                        summary["mean"] - reference
                        if summary["mean"] is not None and reference is not None
                        else None
                    ),
                }
            )
    return aggregates


__all__ = ["generate_report"]
