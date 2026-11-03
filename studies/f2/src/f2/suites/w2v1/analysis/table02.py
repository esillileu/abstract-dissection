"""Table 2 word-analogy evaluation over durable W2V1 MLflow artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import LocalResults, unavailable_report
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .table02_report import write_run_results, write_summary

_CONDITION = re.compile(r"(wmt|lm1b|umbc)--d(50|100|300|600)-w(24|49|98|196|391|783)m")
CORPUS_SOURCES = ("wmt", "lm1b", "umbc")
_DIMENSIONS = (50, 100, 300, 600)
_TRAINING_WORDS = (24, 49, 98, 196, 391, 783)
_SEEDS = (1, 7, 19)
_VOCABULARY_LIMIT = 30_000
_EVALUATION_PROTOCOL = "compute-accuracy-v1"


@dataclass(frozen=True)
class Table2RunResult:
    training_words_millions: int
    vector_dimension: int
    seed: int
    mlflow_run_id: str
    total_accuracy_percent: float
    semantic_accuracy_percent: float
    syntactic_accuracy_percent: float
    observed_training_seconds: float
    included_questions: int
    total_questions: int


class Table2EvaluationCache:
    """Signature-checked per-run cache for reconstructible analogy results."""

    def __init__(self, root: Path, *, questions_sha256: str) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
            "vocabulary_limit": _VOCABULARY_LIMIT,
        }

    def load(self, run_id: str) -> Table2RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table2RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table2RunResult) -> None:
        path = self._path(result.mlflow_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "result": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _path(self, run_id: str) -> Path:
        if re.fullmatch(r"[0-9a-zA-Z_-]+", run_id) is None:
            raise ValueError(f"invalid MLflow run ID: {run_id}")
        return self.root / f"{run_id}.json"


def analyze_table2(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str = "wmt",
    paths: RuntimePaths | None = None,
) -> Path:
    """Evaluate Table 2 conditions directly from MLflow artifacts."""
    paths = paths or RuntimePaths.from_environment()
    output = (paths.cache_root / "f2/w2v1/analysis") / corpus_source / "table2"
    if not Path(questions_path).is_file():
        return unavailable_report(output, "cached evaluation questions missing")
    if corpus_source not in CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")
    output.mkdir(parents=True, exist_ok=True)
    question_bytes = Path(questions_path).read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if not questions:
        raise ValueError("questions-words.txt must contain at least one question")
    questions_sha256 = hashlib.sha256(question_bytes).hexdigest()
    results = LocalResults(cache_root)
    evaluation_cache = Table2EvaluationCache(
        paths.cache_root / "f2/w2v1/table2",
        questions_sha256=questions_sha256,
    )
    records: list[Table2RunResult] = []
    unavailable: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for condition, runs in sorted(
        _complete_conditions(results.runs(), corpus_source).items()
    ):
        for seed in sorted(runs):
            run = runs[seed]
            cached = evaluation_cache.load(run.info.run_id)
            if (
                cached is not None
                and cached.seed == seed
                and cached.training_words_millions == condition[0]
                and cached.vector_dimension == condition[1]
            ):
                records.append(cached)
                cache_hits += 1
                continue
            try:
                cache_misses += 1
                lookup = load_lookup_artifact(
                    results.get(run.info.run_id, "lookup"),
                )
                evaluation = evaluate_analogies(
                    lookup,
                    questions,
                    vocabulary_limit=_VOCABULARY_LIMIT,
                    batch_size=256,
                )
                seconds = observed_training_seconds(
                    results.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table2RunResult(
                    training_words_millions=condition[0],
                    vector_dimension=condition[1],
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    total_accuracy_percent=100.0 * evaluation.overall.require_score(),
                    semantic_accuracy_percent=100.0
                    * evaluation.semantic.require_score(),
                    syntactic_accuracy_percent=100.0
                    * evaluation.syntactic.require_score(),
                    observed_training_seconds=seconds,
                    included_questions=evaluation.overall.valid_count,
                    total_questions=evaluation.overall.total_count,
                )
                evaluation_cache.store(record)
                records.append(record)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {
                        "condition": f"d{condition[1]}-w{condition[0]}m",
                        "seed": seed,
                        "run_id": run.info.run_id,
                        "status": "no_data",
                        "reason": str(exc),
                    }
                )
    print(
        f"W2V1 Table 2 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    (output / "unavailable.json").write_text(
        json.dumps(unavailable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_run_results(output / "table2-runs.csv", records)
    write_summary(
        output / "summary.md",
        records,
        corpus_source=corpus_source,
        questions_sha256=questions_sha256,
    )
    (output / "results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "evaluation_protocol": _EVALUATION_PROTOCOL,
                    "vocabulary_limit": _VOCABULARY_LIMIT,
                    "questions_sha256": questions_sha256,
                    "analogy": "b - a + c; cosine; exclude a, b, c",
                },
                "runs": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def analyze_table2_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    return tuple(
        analyze_table2(
            cache_root,
            questions_path,
            corpus_source=source,
            paths=paths,
        )
        for source in sources
    )


def _complete_conditions(
    runs: list[Any], corpus_source: str
) -> dict[tuple[int, int], dict[int, Any]]:
    grouped: dict[tuple[int, int], dict[int, Any]] = {}
    for run in runs:
        experiment = run.data.tags.get("repro.experiment_spec_id")
        if experiment != "w2v1-table2-cbow":
            continue
        slot = run.data.tags.get("f2.planned_run_slot_id", "")
        match = _CONDITION.search(slot)
        if match is None or match.group(1) != corpus_source:
            continue
        try:
            seed = int(run.data.tags.get("f2.seed", ""))
        except ValueError:
            continue
        if seed not in _SEEDS:
            continue
        key = (int(match.group(3)), int(match.group(2)))
        seeds = grouped.setdefault(key, {})
        if seed in seeds:
            continue
        seeds[seed] = run
    return {key: seeds for key, seeds in grouped.items() if set(seeds) == set(_SEEDS)}


def observed_training_seconds(path: Path) -> float:
    """Sum the final observed wall time from each training epoch."""
    by_epoch: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            epoch = int(row["epoch"])
            elapsed = float(row["elapsed_seconds"])
            by_epoch[epoch] = max(by_epoch.get(epoch, 0.0), elapsed)
    if not by_epoch:
        raise ValueError(f"training observations are empty: {path}")
    return sum(by_epoch.values())


__all__ = [
    "CORPUS_SOURCES",
    "Table2EvaluationCache",
    "Table2RunResult",
    "analyze_table2",
    "analyze_table2_sources",
    "observed_training_seconds",
]
