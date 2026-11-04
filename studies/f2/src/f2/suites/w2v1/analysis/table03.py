"""Full-vocabulary Table 3 comparison on both word-relationship benchmarks."""

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
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .table02 import CORPUS_SOURCES

_SEEDS = {1, 7, 19}
_PLANS = {
    "wmt": "w2v1-reconstruction-r2",
    "lm1b": "w2v1-lm1b-reconstruction-r1",
    "umbc": "w2v1-umbc-reconstruction-r1",
}
_CONDITIONS = {
    "cbow": "cbow-d640-w320m",
    "skipgram": "skipgram-d640-w320m",
    "nnlm": "nnlm-d640-h640-n8-w320m",
}
_REFERENCE = {
    "rnnlm": (9, 36, 35),
    "nnlm": (23, 53, 47),
    "cbow": (24, 64, 61),
    "skipgram": (55, 59, 56),
}


@dataclass(frozen=True)
class Table3RunResult:
    architecture: str
    corpus: str
    seed: int
    mlflow_run_id: str
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    msr_accuracy_percent: float
    semantic_included: int
    semantic_total: int
    syntactic_included: int
    syntactic_total: int
    msr_included: int
    msr_total: int
    observed_training_seconds: float


_EVALUATION_PROTOCOL = "table3-v1"


def complete_conditions(runs: list[Any], corpus: str) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        variant = tags.get("implementation.variant") or ""
        if not variant.startswith(f"{corpus}--"):
            continue
        try:
            seed = int(tags.get("seed") or tags.get("f2.seed", ""))
        except ValueError:
            continue
        if seed not in _SEEDS:
            continue
        cond_slug = variant.split("--", 1)[1]
        for architecture, slug in _CONDITIONS.items():
            if cond_slug == slug:
                grouped.setdefault(architecture, {})[seed] = run
    return {arch: s for arch, s in grouped.items() if set(s) == _SEEDS}


def analyze_table3_sources(
    cache_root: Path,
    questions_path: Path | None = None,
    *,
    google_path: Path | None = None,
    msr_path: Path | None = None,
    msr_questions_path: Path | None = None,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    paths = paths or RuntimePaths.from_environment()

    g_path = google_path or questions_path
    m_path = msr_path or msr_questions_path
    if m_path is None:
        benchmarks = paths.inputs("f2", "benchmarks")
        default_msr = benchmarks / "MSR-syntax.txt"
        if default_msr.is_file():
            m_path = default_msr
        else:
            m_path = cache_root / "evaluation/MSR-syntax.txt"

    for path, name in ((g_path, "Google"), (m_path, "MSR")):
        if path is None or not Path(path).is_file():
            out = unavailable_report(
                paths.analysis_results_root / "f2/w2v1/table03",
                f"cached {name} evaluation questions missing",
            )
            return (out,)

    google_bytes = Path(g_path).read_bytes()
    google_digest = hashlib.sha256(google_bytes).hexdigest()
    google_questions = parse_analogy_questions(google_bytes.splitlines())

    msr_bytes = Path(m_path).read_bytes()
    msr_digest = hashlib.sha256(msr_bytes).hexdigest()
    msr_questions = parse_analogy_questions(msr_bytes.splitlines())

    outputs = []
    for corpus in sources:
        outputs.append(
            _analyze_single_corpus(
                cache_root,
                corpus,
                google_questions,
                google_digest,
                msr_questions,
                msr_digest,
                paths,
            )
        )
    return tuple(outputs)


def _analyze_single_corpus(
    cache_root: Path,
    corpus: str,
    google_questions: Any,
    google_digest: str,
    msr_questions: Any,
    msr_digest: str,
    paths: RuntimePaths,
) -> Path:
    output = paths.analysis_results_root / f"f2/w2v1/table03/{corpus}"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected = complete_conditions(runs, corpus)
    if not selected:
        return unavailable_report(output, "no complete condition sets found")

    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / f"f2/w2v1/analysis/table3/{corpus}",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table3RunResult,
        questions_sha256=google_digest,
        extra_signature={"msr_sha256": msr_digest},
    )

    records: list[Table3RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for architecture, seed_runs in sorted(selected.items()):
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
                g_res = evaluate_analogies(
                    lookup, google_questions, vocabulary_limit=None, batch_size=256
                )
                m_res = evaluate_analogies(
                    lookup, msr_questions, vocabulary_limit=None, batch_size=256
                )
                seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table3RunResult(
                    architecture=architecture,
                    corpus=corpus,
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    semantic_accuracy_percent=100.0 * g_res.semantic.require_score(),
                    syntactic_accuracy_percent=100.0 * g_res.syntactic.require_score(),
                    total_accuracy_percent=100.0 * g_res.overall.require_score(),
                    msr_accuracy_percent=100.0 * m_res.overall.require_score(),
                    semantic_included=g_res.semantic.valid_count,
                    semantic_total=g_res.semantic.total_count,
                    syntactic_included=g_res.syntactic.valid_count,
                    syntactic_total=g_res.syntactic.total_count,
                    msr_included=m_res.overall.valid_count,
                    msr_total=m_res.overall.total_count,
                    observed_training_seconds=seconds,
                )
                evaluation_cache.store(run.info.run_id, record)
                records.append(record)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "run_id": run.info.run_id,
                        "reason": str(exc),
                    }
                )

    print(
        f"W2V1 Table 3 ({corpus}) cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    write_run_records_csv(output / "runs.csv", records)

    rows: list[list[str]] = []
    for arch in ("cbow", "skipgram", "nnlm"):
        items = [r for r in records if r.architecture == arch]
        if not items:
            continue
        sem_str = format_mean_sd([x.semantic_accuracy_percent for x in items])
        syn_str = format_mean_sd([x.syntactic_accuracy_percent for x in items])
        msr_str = format_mean_sd([x.msr_accuracy_percent for x in items])
        _, time_str = format_accuracy_and_time(items)
        paper_ref = " / ".join(str(v) for v in _REFERENCE.get(arch, ()))
        rows.append(
            [arch, str(len(items)), sem_str, syn_str, msr_str, time_str, paper_ref]
        )

    write_table_summary(
        output / "summary.md",
        title=f"W2V1 Table 3 — {corpus.upper()} (Relationship Accuracies & Training Time)",
        headers=[
            "Architecture",
            "Runs",
            "Semantic (%)",
            "Syntactic (%)",
            "MSR (%)",
            "Training Time",
            "Paper Ref (Sem/Syn/MSR)",
        ],
        rows=rows,
    )
    return output


__all__ = ["Table3RunResult", "analyze_table3_sources", "complete_conditions"]
