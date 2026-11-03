"""Run indexer, question resolver, and per-seed evaluator for W2V2."""

from __future__ import annotations

import hashlib
from pathlib import Path

from f2.suites.common.w2v.evaluation import evaluate_analogies

from .. import paper
from .loader import (
    EvidenceMissing,
    extract_timing,
    load_run_artifacts,
    metric_dict,
    sanitize_relative_path,
)


def resolve_question_files(paths, records, word_questions, phrase_questions):
    w_q = word_questions
    if w_q is None:
        for candidate in (
            paths.inputs("f2", "benchmarks") / "questions-words.txt",
            paths.dataset("benchmarks") / "questions-words.txt",
            paths.dataset("questions-words") / "questions-words.txt",
        ):
            if candidate.is_file():
                w_q = candidate
                break
        if w_q is None:
            for r in records:
                art_root = r.get("artifact_root")
                if (
                    art_root
                    and (Path(art_root) / "evaluation/questions-words.txt").is_file()
                ):
                    w_q = Path(art_root) / "evaluation/questions-words.txt"
                    break

    p_q = phrase_questions
    if p_q is None:
        for candidate in (
            paths.inputs("f2", "benchmarks") / "questions-phrases.txt",
            paths.dataset("benchmarks") / "questions-phrases.txt",
            paths.dataset("questions-phrases") / "questions-phrases.txt",
        ):
            if candidate.is_file():
                p_q = candidate
                break
        if p_q is None:
            for r in records:
                art_root = r.get("artifact_root")
                if (
                    art_root
                    and (Path(art_root) / "evaluation/questions-phrases.txt").is_file()
                ):
                    p_q = Path(art_root) / "evaluation/questions-phrases.txt"
                    break
    return w_q, p_q


def filter_inventory(full_inventory, target_table, target_corpus):
    if target_table is None and target_corpus is None:
        return full_inventory
    needed = (
        {"3"}
        if target_table == "4"
        else {"section4.1"}
        if target_table in {"5", "6"}
        else {target_table}
        if target_table is not None
        else None
    )
    return [
        item
        for item in full_inventory
        if (needed is None or item[0] in needed)
        and (target_corpus is None or item[1] == target_corpus)
    ]


def index_records_by_key(records, full_inventory):
    records_by_key = {}
    run_owners = {}
    known = {(tbl, var) for tbl, _, _, var, _ in full_inventory}
    spec_tables = {
        "w2v2-word-skipgram-1b-objectives": "1",
        "w2v2-phrase-skipgram-1b-objectives": "3",
        "w2v2-section4-phrase-6b": "section4.1",
        "w2v2-section4-phrase-33b": "section4.1",
    }
    for record in records:
        tags = record.get("tags", {})
        table = spec_tables.get(tags.get("experiment_spec.id"))
        variant = tags.get("implementation.variant")
        seed_str = tags.get("seed", "")
        if (table, variant) in known and seed_str:
            try:
                seed = int(seed_str)
                key = (table, variant, seed)
                records_by_key[key] = record
                if record.get("run_id"):
                    run_owners.setdefault(record["run_id"], set()).add(key)
            except ValueError:
                pass
        elif variant and seed_str:
            try:
                seed = int(seed_str)
                for tbl, _, _, v, _ in full_inventory:
                    if v == variant:
                        key = (tbl, variant, seed)
                        records_by_key.setdefault(key, record)
                        if record.get("run_id"):
                            run_owners.setdefault(record["run_id"], set()).add(key)
            except ValueError:
                pass
    conflicts = {k for grp in run_owners.values() if len(grp) > 1 for k in grp}
    return records_by_key, conflicts


def questions_sha256(dataset, questions):
    sha = dataset.get("questions_sha256")
    if not sha and questions:
        return hashlib.sha256(
            repr(
                tuple((q.category, q.a, q.b, q.c, q.expected) for q in questions)
            ).encode("utf-8")
        ).hexdigest()
    return sha


def paper_reference(table, condition):
    if table == "1":
        return paper.TABLE1[condition]
    if table == "3":
        return paper.TABLE3[condition]
    return paper.SECTION4[condition]


def evaluate_single_seed(
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
):
    row = {
        "table": table,
        "corpus": corpus,
        "corpus_class": "substitute",
        "condition": condition,
        "seed": seed,
        "paper_reference": reference,
        "status": "no_data",
        "identity_status": "no_data",
        "reason": "missing run/artifacts",
        "checkpoint_reused": False,
        "metrics": {},
        "timing": {"status": "no_data", "minutes": None, "seconds": None},
        "training_minutes": None,
        "training_seconds": None,
    }
    record = records_by_key.get((table, variant, seed))
    lookup = None
    cached_eval = None
    if record is None:
        return row, lookup, cached_eval

    row.update(
        run_id=record["run_id"],
        mlflow_status=record.get("status", ""),
        identity_status="unverified",
    )
    try:
        if (table, variant, seed) in conflicts:
            raise ValueError("run_id cannot supply multiple seeds")
        state = record.get("status")
        if state != "FINISHED":
            row.update(
                status={
                    "FAILED": "failed",
                    "KILLED": "killed",
                    "RUNNING": "running",
                }.get(state, "unverified"),
                reason=f"MLflow {state}",
            )
            return row, lookup, cached_eval

        from f2.suites.common.w2v.artifacts import load_lookup_artifact

        lookup_or_path, result, root, file_inventory, provenance = load_run_artifacts(
            record,
            config,
            variant,
            seed,
            load_arrays=eval_cache is None,
        )
        checkpoint = root / sanitize_relative_path(result["artifacts"]["lookup"])
        lookup_digest = provenance.get("engine_config_digest") or provenance.get(
            "vocabulary_digest"
        )
        if eval_cache is not None and questions:
            cached_eval = eval_cache.load(
                record["run_id"],
                table=table,
                condition=condition,
                questions_sha256=q_sha,
                lookup_digest=lookup_digest,
            )
        lookup = (
            lookup_or_path
            if not isinstance(lookup_or_path, Path)
            else (
                load_lookup_artifact(lookup_or_path)
                if cached_eval is None
                else lookup_or_path
            )
        )
        row.update(
            status="complete",
            identity_status="complete",
            reason="",
            checkpoint=str(checkpoint),
            checkpoint_reused=True,
            provenance=provenance,
        )
        row["timing"] = extract_timing(root, file_inventory, result, record)
        row["training_minutes"] = row["timing"]["minutes"]
        row["training_seconds"] = row["timing"].get("seconds")
        if row["training_seconds"] is None and row["training_minutes"] is not None:
            row["training_seconds"] = row["training_minutes"] * 60.0

        if questions:
            if cached_eval is not None and "metrics" in cached_eval:
                row["metrics"] = cached_eval["metrics"]
            else:
                if isinstance(lookup, Path):
                    lookup = load_lookup_artifact(lookup)
                score = evaluate_analogies(lookup, questions)
                row["metrics"] = {
                    "total": metric_dict(score.overall),
                    "semantic": metric_dict(score.semantic),
                    "syntactic": metric_dict(score.syntactic),
                }
            required = (
                ("total", "semantic", "syntactic") if table == "1" else ("total",)
            )
            if any(row["metrics"][m]["status"] != "complete" for m in required):
                row.update(
                    status="incomplete",
                    reason="evaluation category has no valid questions",
                )
            for metric in row["metrics"]:
                row[metric] = row["metrics"][metric]["accuracy_percent"]
        else:
            row.update(
                status="incomplete",
                reason=f"evaluation data: {dataset['status']}",
            )
    except EvidenceMissing as exc:
        lookup = None
        row.update(status="unverified", identity_status="unverified", reason=str(exc))
    except (KeyError, FileNotFoundError) as exc:
        lookup = None
        row.update(
            status="unverified",
            identity_status="unverified",
            reason=f"missing evidence: {exc}",
        )
    except (ValueError, TypeError, OSError) as exc:
        lookup = None
        row.update(status="failed", identity_status="failed", reason=str(exc))
    return row, lookup, cached_eval


__all__ = [
    "evaluate_single_seed",
    "filter_inventory",
    "index_records_by_key",
    "paper_reference",
    "questions_sha256",
    "resolve_question_files",
]
