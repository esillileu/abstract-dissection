"""Full-vocabulary Table 5 epoch/data/dimension trade-off analysis over durable canonical runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from f2.storage.analysis_cache import LocalResults, unavailable_report
from f2.suites.common.w2v.artifacts import load_lookup_artifact
from f2.suites.common.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .table02 import CORPUS_SOURCES, observed_training_seconds
from .table05_report import write_table5_summary
from .table05_types import TABLE5_SPEC, ConditionKey, Table5RunResult

_SEEDS = {1, 7, 19}
_T4_SLOT_PATTERN = re.compile(
    r"w2v1-table4-([a-z0-9]+)-r1-(cbow|skipgram)-d300-w783m-s(\d+)"
)
_T5_SLOT_PATTERN = re.compile(
    r"w2v1-table5-([a-z0-9]+)-r1-(cbow|skipgram)-d(300|600)-w(783|1600)m-ep1-s(\d+)"
)
_EVALUATION_PROTOCOL = "table5-v1"


class Table5EvaluationCache:
    """Signature-checked per-run cache for reconstructible Table 5 analogy results."""

    def __init__(self, root: Path, *, questions_sha256: str) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
        }

    def load(self, run_id: str) -> Table5RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table5RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table5RunResult) -> None:
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


def collect_table5_runs(
    runs: list[Any], corpus: str
) -> dict[ConditionKey, dict[int, Any]]:
    """Group durable runs by (architecture, dimension, words, epochs) -> seed -> run."""
    grouped: dict[ConditionKey, dict[int, Any]] = {}

    for run in runs:
        slot = run.data.tags.get("f2.planned_run_slot_id", "")
        match_t4 = _T4_SLOT_PATTERN.fullmatch(slot)
        if match_t4 and match_t4.group(1) == corpus:
            arch = match_t4.group(2)
            seed = int(match_t4.group(3))
            if seed in _SEEDS:
                key: ConditionKey = (arch, 300, 783, 3)
                grouped.setdefault(key, {})[seed] = run
            continue

        match_t5 = _T5_SLOT_PATTERN.fullmatch(slot)
        if match_t5 and match_t5.group(1) == corpus:
            arch = match_t5.group(2)
            dim = int(match_t5.group(3))
            words = int(match_t5.group(4))
            seed = int(match_t5.group(5))
            if seed in _SEEDS:
                key = (arch, dim, words, 1)
                grouped.setdefault(key, {})[seed] = run

    return grouped


def analyze_table5_sources(
    cache_root: Path,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    """Evaluate Table 5 conditions directly from MLflow artifacts."""
    paths = paths or RuntimePaths.from_environment()
    if not Path(questions_path).is_file():
        output = unavailable_report(
            paths.cache_root / "f2/w2v1/analysis/table5",
            "cached evaluation questions missing",
        )
        return (output,)
    if corpus_source is not None and corpus_source not in CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")

    questions_path = Path(questions_path)
    question_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if not questions:
        raise ValueError("questions-words.txt must contain at least one question")
    question_digest = hashlib.sha256(question_bytes).hexdigest()

    artifact_cache = LocalResults(cache_root)
    runs = artifact_cache.runs()

    evaluation_cache = Table5EvaluationCache(
        paths.cache_root / "f2/w2v1/table5", questions_sha256=question_digest
    )

    sources = ("wmt",) if corpus_source is None else (corpus_source,)
    outputs = []

    for corpus in sources:
        output = (paths.cache_root / "f2/w2v1/analysis") / corpus / "table5"
        output.mkdir(parents=True, exist_ok=True)
        records = []
        unavailable = []
        cache_hits = 0
        cache_misses = 0
        conditions_map = collect_table5_runs(runs, corpus)

        for spec in TABLE5_SPEC:
            arch, dim, words, epochs, _, _, _, _ = spec
            key: ConditionKey = (arch, dim, words, epochs)
            condition_runs = conditions_map.get(key, {})
            for seed in sorted(condition_runs):
                run = condition_runs[seed]
                cached = evaluation_cache.load(run.info.run_id)
                if (
                    cached is not None
                    and cached.seed == seed
                    and cached.architecture == arch
                    and cached.dimension == dim
                    and cached.training_words_millions == words
                    and cached.epochs == epochs
                    and cached.corpus == corpus
                ):
                    records.append(cached)
                    cache_hits += 1
                    continue
                try:
                    cache_misses += 1
                    lookup = load_lookup_artifact(
                        artifact_cache.get(run.info.run_id, "lookup")
                    )
                    result = evaluate_analogies(
                        lookup, questions, vocabulary_limit=None, batch_size=256
                    )
                    seconds = observed_training_seconds(
                        artifact_cache.get(run.info.run_id, "metrics/observations.csv")
                    )
                    record = Table5RunResult(
                        architecture=arch,
                        dimension=dim,
                        training_words_millions=words,
                        epochs=epochs,
                        corpus=corpus,
                        seed=seed,
                        mlflow_run_id=run.info.run_id,
                        semantic_accuracy_percent=100 * result.semantic.require_score(),
                        syntactic_accuracy_percent=100
                        * result.syntactic.require_score(),
                        total_accuracy_percent=100 * result.overall.require_score(),
                        included_questions=result.overall.valid_count,
                        total_questions=result.overall.total_count,
                        observed_training_seconds=seconds,
                    )
                    evaluation_cache.store(record)
                    records.append(record)

                except (FileNotFoundError, KeyError, ValueError) as exc:
                    unavailable.append(
                        {
                            "run_id": run.info.run_id,
                            "status": "no_data",
                            "reason": str(exc),
                        }
                    )
        (output / "unavailable.json").write_text(
            json.dumps(unavailable, indent=2) + "\n"
        )
        print(
            f"W2V1 {corpus} Table 5 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
            file=sys.stderr,
        )
        with (output / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=tuple(Table5RunResult.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)

        write_table5_summary(output / "summary.md", corpus, records, question_digest)

        (output / "results.json").write_text(
            json.dumps(
                {
                    "protocol": {
                        "questions_sha256": question_digest,
                        "vocabulary_limit": None,
                        "analogy": "b - a + c; cosine; exclude a, b, c",
                    },
                    "reference_table5": [
                        {
                            "model": s[0],
                            "dim": s[1],
                            "words_millions": s[2],
                            "epochs": s[3],
                            "semantic": s[4],
                            "syntactic": s[5],
                            "total": s[6],
                            "original_time": s[7],
                        }
                        for s in TABLE5_SPEC
                    ],
                    "runs": [asdict(record) for record in records],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(output)

    return tuple(outputs)


__all__ = [
    "TABLE5_SPEC",
    "Table5EvaluationCache",
    "Table5RunResult",
    "analyze_table5_sources",
    "collect_table5_runs",
]
