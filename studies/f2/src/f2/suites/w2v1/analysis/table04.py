"""Full-vocabulary Table 4 analogy analysis over durable canonical runs."""

from __future__ import annotations

import hashlib
import json
import re
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
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .table02 import CORPUS_SOURCES

_CONDITION = re.compile(r"(wmt|lm1b|umbc)--(cbow|skipgram)-d300-w783m")
_NNLM_CONDITION = re.compile(r"fineweb--nnlm-d(20|50|100)-w6000m")
TABLE4_CORPUS_SOURCES = (*CORPUS_SOURCES, "fineweb")
_SEEDS = {1, 7, 19}
_REFERENCE = {
    "cbow": (15.5, 53.1, 36.1),
    "skipgram": (50.0, 55.9, 53.3),
    "nnlm-d20": (12.9, 26.4, 20.3),
    "nnlm-d50": (27.9, 55.8, 43.2),
    "nnlm-d100": (34.2, 64.5, 50.8),
}
_EXTERNAL_REFERENCE = (
    ("Collobert-Weston NNLM", 50, "660M", 9.3, 12.3, 11.0),
    ("Turian NNLM", 50, "37M", 1.4, 2.6, 2.1),
    ("Turian NNLM", 200, "37M", 1.4, 2.2, 1.8),
    ("Mnih NNLM", 50, "37M", 1.8, 9.1, 5.8),
    ("Mnih NNLM", 100, "37M", 3.3, 13.2, 8.8),
    ("Mikolov RNNLM", 80, "320M", 4.9, 18.4, 12.7),
    ("Mikolov RNNLM", 640, "320M", 8.6, 36.5, 24.6),
    ("Huang NNLM", 50, "990M", 13.3, 11.6, 12.3),
)


@dataclass(frozen=True)
class Table4RunResult:
    architecture: str
    corpus: str
    seed: int
    mlflow_run_id: str
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    included_questions: int
    total_questions: int
    observed_training_seconds: float


_EVALUATION_PROTOCOL = "table4-v1"


def complete_conditions(runs: list[Any], corpus: str) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        slot = tags.get("f2.planned_run_slot_id", "")
        if corpus in CORPUS_SOURCES:
            match = _CONDITION.search(tags.get("implementation.variant", ""))
            if match is None or match.group(1) != corpus:
                continue
            arch = match.group(2)
        elif corpus == "fineweb":
            match = _NNLM_CONDITION.search(tags.get("implementation.variant", ""))
            if match is None:
                continue
            arch = f"nnlm-d{match.group(1)}"
        else:
            continue

        try:
            seed = int(tags.get("seed", tags.get("f2.seed", "")))
        except ValueError:
            continue
        if seed in _SEEDS:
            grouped.setdefault(arch, {})[seed] = run
    return {arch: s for arch, s in grouped.items() if set(s) == _SEEDS}


def analyze_table4_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    sources = TABLE4_CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    paths = paths or RuntimePaths.from_environment()

    if not Path(questions_path).is_file():
        out = unavailable_report(
            paths.analysis_results_root / "f2/w2v1/table04",
            "cached evaluation questions missing",
        )
        return (out,)

    q_bytes = Path(questions_path).read_bytes()
    q_digest = hashlib.sha256(q_bytes).hexdigest()
    questions = parse_analogy_questions(q_bytes.splitlines())

    return tuple(
        _analyze_single_corpus(cache_root, c, questions, q_digest, paths)
        for c in sources
    )


def _analyze_single_corpus(
    cache_root: Path,
    corpus: str,
    questions: Any,
    q_digest: str,
    paths: RuntimePaths,
) -> Path:
    output = paths.analysis_results_root / f"f2/w2v1/table04/{corpus}"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected = complete_conditions(runs, corpus)
    if not selected:
        return unavailable_report(output, "no complete condition sets found")

    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / f"f2/w2v1/analysis/table4/{corpus}",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table4RunResult,
        questions_sha256=q_digest,
    )

    records: list[Table4RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for arch, seed_runs in sorted(selected.items()):
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
                eval_res = evaluate_analogies(
                    lookup, questions, vocabulary_limit=None, batch_size=256
                )
                seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table4RunResult(
                    architecture=arch,
                    corpus=corpus,
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    semantic_accuracy_percent=100.0 * eval_res.semantic.require_score(),
                    syntactic_accuracy_percent=100.0
                    * eval_res.syntactic.require_score(),
                    total_accuracy_percent=100.0 * eval_res.overall.require_score(),
                    included_questions=eval_res.overall.valid_count,
                    total_questions=eval_res.overall.total_count,
                    observed_training_seconds=seconds,
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
        f"W2V1 Table 4 ({corpus}) cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    write_run_records_csv(output / "runs.csv", records)

    rows: list[list[str]] = []
    for arch in sorted(selected.keys()):
        items = [r for r in records if r.architecture == arch]
        if not items:
            continue
        tot_str, time_str = format_accuracy_and_time(items, "total_accuracy_percent")
        sem_str = format_mean_sd([x.semantic_accuracy_percent for x in items])
        syn_str = format_mean_sd([x.syntactic_accuracy_percent for x in items])
        paper_ref = " / ".join(str(v) for v in _REFERENCE.get(arch, ()))
        rows.append(
            [arch, str(len(items)), tot_str, sem_str, syn_str, time_str, paper_ref]
        )

    write_table_summary(
        output / "summary.md",
        title=f"W2V1 Table 4 — {corpus.upper()} (Word Analogy Accuracies & Training Time)",
        headers=[
            "Architecture",
            "Runs",
            "Total (%)",
            "Semantic (%)",
            "Syntactic (%)",
            "Training Time",
            "Paper Ref (Sem/Syn/Tot)",
        ],
        rows=rows,
    )
    return output


__all__ = [
    "TABLE4_CORPUS_SOURCES",
    "Table4RunResult",
    "analyze_table4_sources",
    "complete_conditions",
]
