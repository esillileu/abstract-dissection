"""Table 6 combines existing W2V 6B runs with Table 4's same NNLM 100d runs."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import LocalResults, unavailable_report
from f2.suites.common.w2v.analysis import (
    W2VTableEvaluationCache,
    format_accuracy_and_time,
    format_mean_sd,
    observed_training_seconds,
    write_run_records_csv,
    write_table_summary,
)
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from f2.suites.common.w2v.lookup import load_lookup_artifact
from repro_core.context import RuntimePaths

from .table04 import complete_conditions as table4_conditions

_REFERENCE = {
    "cbow": (1000, 57.3, 68.9, 63.7, 2, 140),
    "skipgram": (1000, 66.1, 65.1, 65.6, 2.5, 125),
    "nnlm": (100, 34.2, 64.5, 50.8, 14, 180),
}
_SEEDS = {1, 7, 19}
_EVALUATION_PROTOCOL = "table6-v1"


@dataclass(frozen=True)
class Table6RunResult:
    architecture: str
    embedding_dimension: int
    seed: int
    mlflow_run_id: str
    source_table: int
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    included_questions: int
    total_questions: int
    observed_training_seconds: float
    paper_training_days: float
    paper_cpu_cores: int


def complete_conditions(runs: list[Any]) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        for architecture in ("cbow", "skipgram"):
            condition = f"{architecture}-d1000-w6000m"
            if (
                tags.get("implementation.variant") != f"fineweb--{condition}"
                or tags.get("experiment_spec.id")
                != f"w2v1-google-news-{architecture}-scale"
            ):
                continue
            try:
                seed = int(tags.get("seed", tags.get("f2.seed", "")))
            except ValueError:
                continue
            if seed not in _SEEDS:
                continue
            grouped.setdefault(architecture, {}).setdefault(seed, run)
    selected = {arch: seeds for arch, seeds in grouped.items() if set(seeds) == _SEEDS}
    nnlm = table4_conditions(runs, "fineweb").get("nnlm-d100")
    if nnlm is not None:
        selected["nnlm"] = nnlm
    return selected


def analyze_table6_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    if corpus_source not in (None, "fineweb"):
        raise ValueError(
            "Table 6 uses the existing FineWeb 6B surrogate; --corpus must be fineweb"
        )
    paths = paths or RuntimePaths.from_environment()

    if not Path(questions_path).is_file():
        out = unavailable_report(
            paths.analysis_results_root / "f2/w2v1/table06",
            "cached evaluation questions missing",
        )
        return (out,)

    q_bytes = Path(questions_path).read_bytes()
    q_digest = hashlib.sha256(q_bytes).hexdigest()
    questions = parse_analogy_questions(q_bytes.splitlines())

    output = paths.analysis_results_root / "f2/w2v1/table06/fineweb"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected = complete_conditions(runs)
    if not selected:
        return (unavailable_report(output, "no complete condition sets found"),)

    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / "f2/w2v1/analysis/table6",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table6RunResult,
        questions_sha256=q_digest,
    )

    records: list[Table6RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for arch, reference in _REFERENCE.items():
        seed_runs = selected.get(arch)
        if not seed_runs:
            continue
        dim, _, _, _, days, cores = reference
        for seed in sorted(_SEEDS):
            run = seed_runs[seed]
            cached = evaluation_cache.load(run.info.run_id)
            if cached is not None:
                records.append(cached)
                cache_hits += 1
                continue
            try:
                cache_misses += 1
                lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                res = evaluate_analogies(
                    lookup, questions, vocabulary_limit=None, batch_size=256
                )
                seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table6RunResult(
                    architecture=arch,
                    embedding_dimension=dim,
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    source_table=4 if arch == "nnlm" else 6,
                    semantic_accuracy_percent=100.0 * res.semantic.require_score(),
                    syntactic_accuracy_percent=100.0 * res.syntactic.require_score(),
                    total_accuracy_percent=100.0 * res.overall.require_score(),
                    included_questions=res.overall.valid_count,
                    total_questions=res.overall.total_count,
                    observed_training_seconds=seconds,
                    paper_training_days=days,
                    paper_cpu_cores=cores,
                )
                evaluation_cache.store(run.info.run_id, record)
                records.append(record)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {
                        "architecture": arch,
                        "seed": seed,
                        "run_id": run.info.run_id,
                        "reason": str(exc),
                    }
                )

    print(
        f"W2V1 Table 6 cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    write_run_records_csv(output / "runs.csv", records)

    rows: list[list[str]] = []
    for arch, reference in _REFERENCE.items():
        dim, sem_ref, syn_ref, tot_ref, days, cores = reference
        items = [r for r in records if r.architecture == arch]
        if not items:
            continue
        tot_str, time_str = format_accuracy_and_time(items, "total_accuracy_percent")
        sem_str = format_mean_sd([x.semantic_accuracy_percent for x in items])
        syn_str = format_mean_sd([x.syntactic_accuracy_percent for x in items])
        paper_ref = f"{sem_ref} / {syn_ref} / {tot_ref}"
        paper_cost = f"{days}d x {cores} cores"
        rows.append(
            [
                arch,
                str(dim),
                str(len(items)),
                tot_str,
                sem_str,
                syn_str,
                time_str,
                paper_ref,
                paper_cost,
            ]
        )

    write_table_summary(
        output / "summary.md",
        title="W2V1 Table 6 — FineWeb surrogate (Scaled Models Comparison)",
        headers=[
            "Model",
            "Dim",
            "Runs",
            "Total (%)",
            "Semantic (%)",
            "Syntactic (%)",
            "Training Time",
            "Paper Acc (Sem/Syn/Tot)",
            "Paper Historical Cost",
        ],
        rows=rows,
    )
    return (output,)


__all__ = ["Table6RunResult", "analyze_table6_sources", "complete_conditions"]
