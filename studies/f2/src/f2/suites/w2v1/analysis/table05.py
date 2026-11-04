"""Full-vocabulary Table 5 epoch/data/dimension trade-off analysis over durable canonical runs."""

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

ConditionKey = tuple[
    str, int, int, int
]  # (architecture, dimension, words_millions, epochs)

TABLE5_SPEC: tuple[tuple[ConditionKey, float], ...] = (
    (("skipgram", 300, 783, 3), 65.6),
    (("skipgram", 300, 783, 1), 59.2),
    (("skipgram", 600, 783, 1), 60.9),
    (("skipgram", 600, 1600, 1), 65.8),
    (("cbow", 300, 783, 3), 63.7),
    (("cbow", 300, 783, 1), 58.7),
    (("cbow", 600, 783, 1), 61.6),
    (("cbow", 600, 1600, 1), 67.3),
)

_SEEDS = {1, 7, 19}
_EVALUATION_PROTOCOL = "table5-v1"


@dataclass(frozen=True)
class Table5RunResult:
    architecture: str
    dimension: int
    training_words_millions: int
    epochs: int
    corpus: str
    seed: int
    mlflow_run_id: str
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    total_accuracy_percent: float
    included_questions: int
    total_questions: int
    observed_training_seconds: float


def collect_table5_runs(
    runs: list[Any], corpus: str
) -> dict[ConditionKey, dict[int, Any]]:
    grouped: dict[ConditionKey, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        variant = tags.get("implementation.variant", "")
        if not variant.startswith(f"{corpus}--"):
            continue
        try:
            seed = int(tags.get("seed", tags.get("f2.seed", "")))
        except ValueError:
            continue
        if seed not in _SEEDS:
            continue
        # variant: <corpus>--<arch>-d<dim>-w<words>m[-ep1]; unsuffixed means 3 epochs
        parts = variant.split("--", 1)[1].split("-")
        if (
            len(parts) < 3
            or parts[0] not in ("cbow", "skipgram")
            or not (parts[1].startswith("d") and parts[2].startswith("w"))
        ):
            continue
        if parts[3:] not in ([], ["ep1"]) or not parts[2].endswith("m"):
            continue
        epochs = 1 if parts[3:] else 3
        try:
            key = (parts[0], int(parts[1][1:]), int(parts[2][1:-1]), epochs)
        except ValueError:
            continue
        if key not in {condition for condition, _ in TABLE5_SPEC}:
            continue
        grouped.setdefault(key, {})[seed] = run
    return {k: s for k, s in grouped.items() if set(s) == _SEEDS}


def analyze_table5_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    paths = paths or RuntimePaths.from_environment()

    if not Path(questions_path).is_file():
        out = unavailable_report(
            paths.analysis_results_root / "f2/w2v1/table05",
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
    output = paths.analysis_results_root / f"f2/w2v1/table05/{corpus}"
    output.mkdir(parents=True, exist_ok=True)
    cache = LocalResults(cache_root)
    runs = cache.runs()
    selected = collect_table5_runs(runs, corpus)
    if not selected:
        return unavailable_report(output, "no complete condition sets found")

    evaluation_cache = W2VTableEvaluationCache(
        paths.analysis_cache_root / f"f2/w2v1/analysis/table5/{corpus}",
        protocol=_EVALUATION_PROTOCOL,
        result_cls=Table5RunResult,
        questions_sha256=q_digest,
    )

    records: list[Table5RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = cache_misses = 0

    for (arch, dim, words, epochs), seed_runs in sorted(selected.items()):
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
                record = Table5RunResult(
                    architecture=arch,
                    dimension=dim,
                    training_words_millions=words,
                    epochs=epochs,
                    corpus=corpus,
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    semantic_accuracy_percent=100.0 * res.semantic.require_score(),
                    syntactic_accuracy_percent=100.0 * res.syntactic.require_score(),
                    total_accuracy_percent=100.0 * res.overall.require_score(),
                    included_questions=res.overall.valid_count,
                    total_questions=res.overall.total_count,
                    observed_training_seconds=seconds,
                )
                evaluation_cache.store(run.info.run_id, record)
                records.append(record)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {
                        "condition": f"{arch}-d{dim}-w{words}m-ep{epochs}",
                        "seed": seed,
                        "run_id": run.info.run_id,
                        "reason": str(exc),
                    }
                )

    print(
        f"W2V1 Table 5 ({corpus}) cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    write_run_records_csv(output / "runs.csv", records)

    spec_dict = dict(TABLE5_SPEC)
    rows: list[list[str]] = []
    for cond, orig_acc in TABLE5_SPEC:
        arch, dim, words, epochs = cond
        items = [
            r
            for r in records
            if (r.architecture, r.dimension, r.training_words_millions, r.epochs)
            == cond
        ]
        if not items:
            continue
        tot_str, time_str = format_accuracy_and_time(items, "total_accuracy_percent")
        sem_str = format_mean_sd([x.semantic_accuracy_percent for x in items])
        syn_str = format_mean_sd([x.syntactic_accuracy_percent for x in items])
        rows.append(
            [
                arch,
                str(dim),
                f"{words}M",
                str(epochs),
                str(len(items)),
                tot_str,
                sem_str,
                syn_str,
                time_str,
                f"{orig_acc:.1f}",
            ]
        )

    write_table_summary(
        output / "summary.md",
        title=f"W2V1 Table 5 — {corpus.upper()} (Epoch/Data/Dimension Trade-off)",
        headers=[
            "Model",
            "Dim",
            "Words",
            "Epochs",
            "Runs",
            "Total (%)",
            "Semantic (%)",
            "Syntactic (%)",
            "Training Time",
            "Paper Target (%)",
        ],
        rows=rows,
    )
    return output


__all__ = [
    "TABLE5_SPEC",
    "ConditionKey",
    "Table5RunResult",
    "analyze_table5_sources",
    "collect_table5_runs",
]
