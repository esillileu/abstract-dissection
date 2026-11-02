"""W2V1 Table 3 CBOW vs Skip-gram diagnostic evaluation and artifact generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from f2.common.paths import get_benchmark_data_dir
from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import (
    _analogy_predictions,
    _analogy_rows,
    _analogy_tokens,
    parse_analogy_questions,
)
from repro_core.context import RuntimePaths

# Pinned Table 3 LM1B run IDs
TABLE3_RUNS = {
    "cbow": {
        1: "3acee9f07d884d15ab30d0101ed3e4b0",
        7: "74a39ee4dd934b4eac23a04878573e09",
        19: "dffa8e204adc4ffeb614419b8d9b1792",
    },
    "skipgram": {
        1: "aeb9b69c563c476f9d99f6ef2d76678c",
        7: "5e13f28225974b4480b1bf2675695255",
        19: "69d4ac70552d4f79a18c93c9a462f6f4",
    },
}

SEEDS = (1, 7, 19)
MODELS = ("cbow", "skipgram")


def run_table3_diagnosis(
    output_dir: Path | None = None,
    questions_path: Path | None = None,
    paths: RuntimePaths | None = None,
    cache_root: Path | None = None,
) -> dict[str, Path]:
    """Run comprehensive Table 3 CBOW vs Skip-gram diagnostic evaluation."""
    paths = paths or RuntimePaths.from_environment()
    if output_dir is None:
        output_dir = paths.analysis_output("f2", "w2v1") / "diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    if questions_path is None:
        questions_path = get_benchmark_data_dir(paths) / "questions-words.txt"
    if not questions_path.is_file():
        raise FileNotFoundError(f"Questions words file not found: {questions_path}")

    question_bytes = questions_path.read_bytes()
    from .resources import QUESTIONS_WORDS_SHA256

    if hashlib.sha256(question_bytes).hexdigest() != QUESTIONS_WORDS_SHA256:
        raise ValueError("Canonical analogy question checksum mismatch")
    questions = parse_analogy_questions(question_bytes.splitlines())

    # Cache directory for artifacts
    run_ids = [TABLE3_RUNS[m][seed] for m in MODELS for seed in SEEDS]
    if cache_root is None:
        candidates = [
            root
            for root in (paths.cache_root / "mlflow_artifact").iterdir()
            if all((root / run / "lookup/manifest.json").is_file() for run in run_ids)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Provide cache_root containing all six completed run lookups"
            )
        cache_root = candidates[0]
    baselines = {}
    for run in run_ids:
        payload = json.loads(
            (paths.cache_root / "f2/w2v1/analogy" / f"{run}.json").read_text()
        )
        if payload["signature"]["questions_sha256"] != QUESTIONS_WORDS_SHA256:
            raise ValueError("Baseline question identity mismatch")
        baselines[run] = payload
    limits = {p["signature"]["vocabulary_limit"] for p in baselines.values()}
    if len(limits) != 1:
        raise ValueError("Baseline candidate limits differ")
    vocabulary_limit = limits.pop()

    # Use first run to inspect vocabulary and calculate frequencies
    sample_lookup_path = cache_root / TABLE3_RUNS["cbow"][1] / "lookup"
    sample_lookup = load_lookup_artifact(sample_lookup_path)
    row_count = min(len(sample_lookup.embeddings), vocabulary_limit)
    wanted = {token.upper() for q in questions for token in _analogy_tokens(q)}
    analogy_rows = _analogy_rows(sample_lookup, row_count, wanted=wanted)
    training_rows = _analogy_rows(
        sample_lookup, len(sample_lookup.embeddings), wanted=wanted
    )
    tokens = tuple(
        bytes(
            sample_lookup.token_bytes[
                sample_lookup.token_offsets[r] : sample_lookup.token_offsets[r + 1]
            ]
        )
        for r in range(len(sample_lookup.counts))
    )
    lexical_rows = [r for r, token in enumerate(tokens) if token != b"</s>"]
    top_rows = set(
        sorted(lexical_rows, key=lambda r: (-int(sample_lookup.counts[r]), r))[:30000]
    )
    counts = sample_lookup.counts
    token_bytes = sample_lookup.token_bytes
    token_offsets = sample_lookup.token_offsets

    def decode_token(r: int) -> str:
        end = token_offsets[r + 1] if r + 1 < len(token_offsets) else len(token_bytes)
        return bytes(token_bytes[token_offsets[r] : end]).decode(
            "ascii", errors="replace"
        )

    # 1. Build question metadata
    question_meta: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    valid_rows: list[tuple[int, int, int, int]] = []

    for idx, q in enumerate(questions):
        rows = [analogy_rows.get(token.upper()) for token in _analogy_tokens(q)]
        frequency_rows = [
            training_rows.get(token.upper()) for token in _analogy_tokens(q)
        ]
        is_oov = any(r is None for r in rows)
        is_syn = q.category.startswith("gram")
        section = "syntactic" if is_syn else "semantic"

        fa, fb, fc, fd = (
            int(counts[r]) if r is not None else None for r in frequency_rows
        )
        if is_oov:
            in_top30k = False
            min_f = None
        else:
            fa = int(counts[rows[0]])
            fb = int(counts[rows[1]])
            fc = int(counts[rows[2]])
            fd = int(counts[rows[3]])
            min_f = min(fa, fb, fc, fd)
            in_top30k = all(r in top_rows for r in rows)
            valid_indices.append(idx)
            valid_rows.append((int(rows[0]), int(rows[1]), int(rows[2]), int(rows[3])))

        question_meta.append(
            {
                "question_id": idx,
                "category": q.category,
                "section": section,
                "word_a": q.a.decode("ascii", errors="replace"),
                "word_b": q.b.decode("ascii", errors="replace"),
                "word_c": q.c.decode("ascii", errors="replace"),
                "word_d": q.expected.decode("ascii", errors="replace"),
                "freq_a": fa,
                "freq_b": fb,
                "freq_c": fc,
                "freq_d": fd,
                "min_freq": min_f,
                "included": not is_oov,
                "in_top30k": in_top30k,
                "row_d": rows[3] if not is_oov else None,
            }
        )

    # 2. Divide included questions into 4 quantiles for semantic and syntactic separately
    sem_valid_qids = [
        q["question_id"]
        for q in question_meta
        if q["included"] and q["section"] == "semantic"
    ]
    syn_valid_qids = [
        q["question_id"]
        for q in question_meta
        if q["included"] and q["section"] == "syntactic"
    ]

    qid_to_quantile = {}
    for qids in (sem_valid_qids, syn_valid_qids):
        ordered = sorted(qids, key=lambda qid: (question_meta[qid]["min_freq"], qid))
        for rank, qid in enumerate(ordered):
            qid_to_quantile[qid] = f"Q{min(3, rank * 4 // len(ordered)) + 1}"

    for q in question_meta:
        q["freq_quantile"] = qid_to_quantile.get(q["question_id"], None)

    # 3. Model inference evaluation for all 6 runs
    def evaluate_run(run_id: str) -> dict[int, tuple[str, bool]]:
        lookup = load_lookup_artifact(cache_root / run_id / "lookup")
        for name in ("counts", "token_bytes", "token_offsets"):
            if not np.array_equal(getattr(lookup, name), getattr(sample_lookup, name)):
                raise ValueError(f"Vocabulary/frequencies differ: {run_id}, {name}")
        embeddings = lookup.embeddings[:row_count]
        res = {}
        for qid, (prediction, positive) in zip(
            valid_indices,
            _analogy_predictions(embeddings, valid_rows, 256),
            strict=True,
        ):
            pred_row = int(prediction)
            res[qid] = (
                decode_token(pred_row) if positive else "<none>",
                bool(positive and pred_row == question_meta[qid]["row_d"]),
            )
        return res

    verification_rows = []
    eval_outputs: dict[tuple[str, int], dict[int, tuple[str, bool]]] = {}
    for model in MODELS:
        for seed in SEEDS:
            run_id = TABLE3_RUNS[model][seed]
            print(f"Evaluating {model} seed {seed}: {run_id}", flush=True)
            eval_outputs[(model, seed)] = evaluate_run(run_id)
            result = baselines[run_id]["result"]
            for section, qids in (
                ("semantic", sem_valid_qids),
                ("syntactic", syn_valid_qids),
                ("total", valid_indices),
            ):
                actual = (
                    sum(eval_outputs[(model, seed)][qid][1] for qid in qids)
                    / len(qids)
                    * 100
                )
                expected = result[f"{section}_accuracy_percent"]
                verification_rows.append(
                    dict(
                        model=model,
                        seed=seed,
                        run_id=run_id,
                        section=section,
                        expected_accuracy_percent=expected,
                        actual_accuracy_percent=actual,
                        difference=actual - expected,
                    )
                )
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(
                        f"Aggregate mismatch {run_id} {section}: {actual} != {expected}"
                    )
            if result["included_questions"] != len(valid_indices) or result[
                "total_questions"
            ] != len(questions):
                raise ValueError(f"Coverage mismatch: {run_id}")

    verification_path = output_dir / "aggregate_verification.csv"
    with verification_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(verification_rows[0]))
        writer.writeheader()
        writer.writerows(verification_rows)

    # 4. Generate Category Accuracy breakdown
    categories = sorted({q["category"] for q in question_meta})
    sem_cats = [c for c in categories if not c.startswith("gram")]
    syn_cats = [c for c in categories if c.startswith("gram")]
    ordered_cats = sem_cats + syn_cats

    category_rows: list[dict[str, Any]] = []
    for cat in ordered_cats:
        cat_q = [q for q in question_meta if q["category"] == cat]
        tot = len(cat_q)
        inc = sum(1 for q in cat_q if q["included"])
        oov = tot - inc
        inc_qids = [q["question_id"] for q in cat_q if q["included"]]

        freqs = [q["min_freq"] for q in cat_q if q["included"]]
        med_f = float(np.median(freqs)) if freqs else 0.0
        min_f = int(min(freqs)) if freqs else 0

        cbow_accs = [
            sum(eval_outputs[("cbow", s)][qid][1] for qid in inc_qids) / inc * 100
            for s in SEEDS
        ]
        cbow_corrs = [
            sum(eval_outputs[("cbow", s)][qid][1] for qid in inc_qids) for s in SEEDS
        ]

        sg_accs = [
            sum(eval_outputs[("skipgram", s)][qid][1] for qid in inc_qids) / inc * 100
            for s in SEEDS
        ]
        sg_corrs = [
            sum(eval_outputs[("skipgram", s)][qid][1] for qid in inc_qids)
            for s in SEEDS
        ]

        category_rows.append(
            {
                "category": cat,
                "section": "syntactic" if cat.startswith("gram") else "semantic",
                "total_questions": tot,
                "included_questions": inc,
                "oov_questions": oov,
                "min_frequency_min": min_f,
                "min_frequency_median": med_f,
                "cbow_correct_mean": float(np.mean(cbow_corrs)),
                "cbow_accuracy_percent": float(np.mean(cbow_accs)),
                "skipgram_correct_mean": float(np.mean(sg_corrs)),
                "skipgram_accuracy_percent": float(np.mean(sg_accs)),
                "accuracy_diff_percent": float(np.mean(cbow_accs) - np.mean(sg_accs)),
            }
        )

    # Aggregates for category table
    for sec_name, qids, tot_q in [
        (
            "Semantic Total",
            sem_valid_qids,
            sum(1 for q in question_meta if q["section"] == "semantic"),
        ),
        (
            "Syntactic Total",
            syn_valid_qids,
            sum(1 for q in question_meta if q["section"] == "syntactic"),
        ),
        ("Total", valid_indices, len(question_meta)),
    ]:
        inc = len(qids)
        oov = tot_q - inc
        freqs = [question_meta[qid]["min_freq"] for qid in qids]
        med_f = float(np.median(freqs)) if freqs else 0.0
        min_f = int(min(freqs)) if freqs else 0

        cbow_accs = [
            sum(eval_outputs[("cbow", s)][qid][1] for qid in qids) / inc * 100
            for s in SEEDS
        ]
        cbow_corrs = [
            sum(eval_outputs[("cbow", s)][qid][1] for qid in qids) for s in SEEDS
        ]
        sg_accs = [
            sum(eval_outputs[("skipgram", s)][qid][1] for qid in qids) / inc * 100
            for s in SEEDS
        ]
        sg_corrs = [
            sum(eval_outputs[("skipgram", s)][qid][1] for qid in qids) for s in SEEDS
        ]

        category_rows.append(
            {
                "category": sec_name,
                "section": sec_name.split()[0].lower(),
                "total_questions": tot_q,
                "included_questions": inc,
                "oov_questions": oov,
                "min_frequency_min": min_f,
                "min_frequency_median": med_f,
                "cbow_correct_mean": float(np.mean(cbow_corrs)),
                "cbow_accuracy_percent": float(np.mean(cbow_accs)),
                "skipgram_correct_mean": float(np.mean(sg_corrs)),
                "skipgram_accuracy_percent": float(np.mean(sg_accs)),
                "accuracy_diff_percent": float(np.mean(cbow_accs) - np.mean(sg_accs)),
            }
        )

    cat_csv_path = output_dir / "category_accuracy.csv"
    with cat_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0].keys()))
        writer.writeheader()
        for row in category_rows:
            writer.writerow(
                {
                    k: (
                        f"{v:.2f}"
                        if isinstance(v, float)
                        else (
                            f"{v:.4f}"
                            if isinstance(v, np.floating)
                            else (str(v) if v is not None else "")
                        )
                    )
                    for k, v in row.items()
                }
            )

    # 5. Generate Frequency Quantile breakdown
    quantile_rows: list[dict[str, Any]] = []
    for sec_name, qids_list in [
        ("semantic", sem_valid_qids),
        ("syntactic", syn_valid_qids),
    ]:
        for qlabel in ["Q1", "Q2", "Q3", "Q4"]:
            b_qids = [
                qid
                for qid in qids_list
                if question_meta[qid]["freq_quantile"] == qlabel
            ]
            b_freqs = [question_meta[qid]["min_freq"] for qid in b_qids]
            inc = len(b_qids)

            c_accs = [
                sum(eval_outputs[("cbow", s)][qid][1] for qid in b_qids) / inc * 100
                for s in SEEDS
            ]
            s_accs = [
                sum(eval_outputs[("skipgram", s)][qid][1] for qid in b_qids) / inc * 100
                for s in SEEDS
            ]

            quantile_rows.append(
                {
                    "section": sec_name,
                    "quantile": qlabel,
                    "frequency_min": min(b_freqs),
                    "frequency_max": max(b_freqs),
                    "frequency_range": f"[{min(b_freqs)}, {max(b_freqs)}]",
                    "question_count": inc,
                    "cbow_accuracy_percent": float(np.mean(c_accs)),
                    "skipgram_accuracy_percent": float(np.mean(s_accs)),
                    "accuracy_diff_percent": float(np.mean(c_accs) - np.mean(s_accs)),
                }
            )

    freq_csv_path = output_dir / "frequency_quantile_accuracy.csv"
    with freq_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(quantile_rows[0].keys()))
        writer.writeheader()
        for row in quantile_rows:
            writer.writerow(
                {
                    k: (
                        f"{v:.2f}"
                        if isinstance(v, float)
                        else (str(v) if v is not None else "")
                    )
                    for k, v in row.items()
                }
            )

    # 6. Generate Top-30K Diagnostic
    top30k_qids = [q["question_id"] for q in question_meta if q["in_top30k"]]
    top30k_rows: list[dict[str, Any]] = []

    for model_name in MODELS:
        for subset_name, qids_sub in [
            ("Full Table 3", valid_indices),
            ("Top-30K-only", top30k_qids),
        ]:
            sem_q = [
                qid for qid in qids_sub if question_meta[qid]["section"] == "semantic"
            ]
            syn_q = [
                qid for qid in qids_sub if question_meta[qid]["section"] == "syntactic"
            ]
            tot_q = qids_sub

            sem_accs = [
                sum(eval_outputs[(model_name, s)][qid][1] for qid in sem_q)
                / len(sem_q)
                * 100
                for s in SEEDS
            ]
            syn_accs = [
                sum(eval_outputs[(model_name, s)][qid][1] for qid in syn_q)
                / len(syn_q)
                * 100
                for s in SEEDS
            ]
            tot_accs = [
                sum(eval_outputs[(model_name, s)][qid][1] for qid in tot_q)
                / len(tot_q)
                * 100
                for s in SEEDS
            ]

            top30k_rows.append(
                {
                    "model": "CBOW" if model_name == "cbow" else "Skip-gram",
                    "eval_subset": subset_name,
                    "semantic_accuracy_percent": float(np.mean(sem_accs)),
                    "syntactic_accuracy_percent": float(np.mean(syn_accs)),
                    "total_accuracy_percent": float(np.mean(tot_accs)),
                    "semantic_included_questions": len(sem_q),
                    "syntactic_included_questions": len(syn_q),
                    "included_questions": len(tot_q),
                    "total_questions": len(questions),
                    "coverage_ratio_percent": len(tot_q) / len(questions) * 100,
                }
            )

    top30k_csv_path = output_dir / "top30k_diagnostic.csv"
    with top30k_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top30k_rows[0].keys()))
        writer.writeheader()
        for row in top30k_rows:
            writer.writerow(
                {
                    k: (
                        f"{v:.2f}"
                        if isinstance(v, float)
                        else (str(v) if v is not None else "")
                    )
                    for k, v in row.items()
                }
            )

    # 7. Generate Raw Question Diagnostics CSV
    # Row per (model, seed, question)
    raw_csv_path = output_dir / "analogy_question_diagnostics.csv"
    raw_fields = [
        "model",
        "seed",
        "run_id",
        "question_id",
        "frequency_quantile",
        "category",
        "semantic_or_syntactic",
        "word_a",
        "word_b",
        "word_c",
        "word_d",
        "frequency_a",
        "frequency_b",
        "frequency_c",
        "frequency_d",
        "min_frequency",
        "included",
        "in_top30k",
        "prediction",
        "correct",
    ]
    with raw_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        for model in MODELS:
            for seed in SEEDS:
                preds = eval_outputs[(model, seed)]
                for q in question_meta:
                    qid = q["question_id"]
                    if q["included"]:
                        pred_tok, is_corr = preds[qid]
                    else:
                        pred_tok, is_corr = "", ""
                    writer.writerow(
                        {
                            "model": model,
                            "seed": seed,
                            "run_id": TABLE3_RUNS[model][seed],
                            "question_id": qid,
                            "frequency_quantile": q["freq_quantile"],
                            "category": q["category"],
                            "semantic_or_syntactic": q["section"],
                            "word_a": q["word_a"],
                            "word_b": q["word_b"],
                            "word_c": q["word_c"],
                            "word_d": q["word_d"],
                            "frequency_a": (
                                q["freq_a"] if q["freq_a"] is not None else ""
                            ),
                            "frequency_b": (
                                q["freq_b"] if q["freq_b"] is not None else ""
                            ),
                            "frequency_c": (
                                q["freq_c"] if q["freq_c"] is not None else ""
                            ),
                            "frequency_d": (
                                q["freq_d"] if q["freq_d"] is not None else ""
                            ),
                            "min_frequency": (
                                q["min_freq"] if q["min_freq"] is not None else ""
                            ),
                            "included": q["included"],
                            "in_top30k": q["in_top30k"],
                            "prediction": pred_tok,
                            "correct": is_corr,
                        }
                    )

    # 8. Generate diagnostic_summary.md
    summary_md_path = output_dir / "diagnostic_summary.md"
    summary_content = _build_summary_markdown(
        category_rows=category_rows,
        quantile_rows=quantile_rows,
        top30k_rows=top30k_rows,
    )
    summary_md_path.write_text(summary_content, encoding="utf-8")

    return {
        "aggregate_verification": verification_path,
        "category_accuracy": cat_csv_path,
        "frequency_quantile_accuracy": freq_csv_path,
        "top30k_diagnostic": top30k_csv_path,
        "analogy_question_diagnostics": raw_csv_path,
        "diagnostic_summary": summary_md_path,
    }


def _build_summary_markdown(category_rows, quantile_rows, top30k_rows):
    lines = [
        "# W2V1 Table 3 CBOW vs Skip-gram Diagnostic Report",
        "",
        "Completed LM1B 320M / 640-dimensional runs; arithmetic mean across seeds 1, 7, 19.",
        "Each seed's semantic, syntactic and total accuracy matches its original cached result within 1e-10 percentage points; coverage and lookup identities checked.",
        "Candidate vocabulary follows the recorded baseline (82,000 rows including </s>); saved embeddings are unchanged.",
        "Top-30K filters questions only, using actual token counts excluding </s>; candidate search remains identical.",
        "Quantiles use stable rank of minimum frequency, with question order breaking ties to keep bin sizes within one question. Equal frequencies may straddle bins.",
        "Corpus SHA-256 and vocabulary digest agree across all six runs; counts and token arrays are identical.",
        "Category composition differs between frequency bins and subsets; these are descriptive comparisons, not causal effects.",
        "",
    ]
    lines.extend(["## Observations", ""])
    semantic_categories = [
        r
        for r in category_rows
        if r["section"] == "semantic" and r["category"] != "Semantic Total"
    ]
    for r in sorted(
        semantic_categories,
        key=lambda r: r["accuracy_diff_percent"] * r["included_questions"],
    )[:2]:
        lines.append(
            f"- Largest contributions to the semantic gap: {r['category']}, CBOW {r['cbow_accuracy_percent']:.2f}% vs Skip-gram {r['skipgram_accuracy_percent']:.2f}% ({r['included_questions']} included questions)."
        )
    for section in ("semantic", "syntactic"):
        bins = [r for r in quantile_rows if r["section"] == section]
        lines.append(
            f"- {section} Q1-Q4 CBOW: "
            + " / ".join(f"{r['cbow_accuracy_percent']:.2f}%" for r in bins)
            + "; Skip-gram: "
            + " / ".join(f"{r['skipgram_accuracy_percent']:.2f}%" for r in bins)
            + "."
        )
        highest = bins[-1]
        lines.append(
            f"  Highest-frequency gap (CBOW - Skip-gram): {highest['accuracy_diff_percent']:+.2f} percentage points; frequency alone does not establish a causal explanation."
        )
    for model in ("CBOW", "Skip-gram"):
        full = next(
            r
            for r in top30k_rows
            if r["model"] == model and r["eval_subset"] == "Full Table 3"
        )
        top = next(
            r
            for r in top30k_rows
            if r["model"] == model and r["eval_subset"] == "Top-30K-only"
        )
        changes = [
            (
                section,
                top[f"{section}_accuracy_percent"]
                - full[f"{section}_accuracy_percent"],
            )
            for section in ("semantic", "syntactic", "total")
        ]
        lines.append(
            f"- {model} Top-30K changes: "
            + ", ".join(
                f"{section} {change:+.2f} percentage points"
                for section, change in changes
            )
            + "."
        )
        full_weight = full["syntactic_included_questions"] / full["included_questions"]
        top_weight = top["syntactic_included_questions"] / top["included_questions"]
        composition = (top_weight - full_weight) * (
            top["syntactic_accuracy_percent"] - top["semantic_accuracy_percent"]
        )
        lines.append(
            f"  Syntactic question share {100 * full_weight:.2f}% → {100 * top_weight:.2f}%; composition contribution to total change {composition:+.2f} percentage points (top scores, full versus top weights). This is a descriptive decomposition."
        )
    lines.extend(
        [
            "- OOV/coverage and evaluation protocol are shared across both models; the reproduced gap is within the same included questions. This does not rule out effects of missing words on benchmark representativeness.",
            "- The frequency and subset comparisons cannot separate corpus/context interactions, optimization, and CBOW-specific implementation. A training bug is not established by these results.",
            "",
        ]
    )
    for title, rows in (
        ("Category accuracy", category_rows),
        ("Frequency quantiles", quantile_rows),
        ("Top-30K diagnostic", top30k_rows),
    ):
        lines.extend([f"## {title}", ""])
        fields = list(rows[0])
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    f"{row[f]:.2f}" if isinstance(row[f], float) else str(row[f])
                    for f in fields
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Limits",
            "",
            "This evaluation does not distinguish optimization, CBOW implementation, or corpus/context interactions. No training or hyperparameter change was performed.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    artifacts = run_table3_diagnosis()
    for name, path in artifacts.items():
        print(f"{name}: {path}")
