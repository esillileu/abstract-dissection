"""Table 6 combines existing W2V 6B runs with Table 4's same NNLM 100d runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.cache import LocalResults, unavailable_report
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths

from .analysis import observed_training_seconds
from .table4 import (
    Table4EvaluationCache,
)
from .table4 import (
    complete_conditions as table4_conditions,
)

_REFERENCE = {
    "cbow": (1000, 57.3, 68.9, 63.7, 2, 140),
    "skipgram": (1000, 66.1, 65.1, 65.6, 2.5, 125),
    "nnlm": (100, 34.2, 64.5, 50.8, 14, 180),
}
_SEEDS = {1, 7, 19}


def complete_conditions(runs: list[Any]) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        for architecture in ("cbow", "skipgram"):
            condition = f"{architecture}-d1000-w6000m"
            # These are the existing e05 Table 6 execution identities, preserved verbatim.
            if (
                tags.get("implementation.variant") != f"fineweb--{condition}"
                or tags.get("experiment_spec.id")
                != f"w2v1-google-news-{architecture}-scale"
            ):
                continue
            seed = int(tags.get("seed", "0"))
            if (
                seed not in _SEEDS
                or tags.get("f2.planned_run_slot_id")
                != f"w2v1-fineweb-reconstruction-r1-{condition}-s{seed}"
            ):
                continue
            grouped.setdefault(architecture, {}).setdefault(seed, run)
    selected = {arch: seeds for arch, seeds in grouped.items() if set(seeds) == _SEEDS}
    nnlm = table4_conditions(runs, "fineweb").get("nnlm-d100")
    if nnlm is not None:
        selected["nnlm"] = nnlm
    return selected


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


_EVALUATION_PROTOCOL = "table6-v1"


class Table6EvaluationCache:
    """Signature-checked per-run cache for reconstructible Table 6 analogy results."""

    def __init__(self, root: Path, *, questions_sha256: str) -> None:
        self.root = root
        self.signature = {
            "schema_version": 2,
            "evaluation_protocol": _EVALUATION_PROTOCOL,
            "questions_sha256": questions_sha256,
        }

    def load(self, run_id: str) -> Table6RunResult | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signature") != self.signature:
                return None
            result = Table6RunResult(**payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return result if result.mlflow_run_id == run_id else None

    def store(self, result: Table6RunResult) -> None:
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
        output = unavailable_report(
            paths.cache_root / "f2/w2v1/analysis/table6",
            "cached evaluation questions missing",
        )
        return (output,)
    question_bytes = Path(questions_path).read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if not questions:
        raise ValueError("questions-words.txt must contain at least one question")
    digest = hashlib.sha256(question_bytes).hexdigest()
    cache = LocalResults(cache_root)
    runs = cache.runs()

    output = (paths.cache_root / "f2/w2v1/analysis") / "fineweb" / "table6"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_cache = Table6EvaluationCache(
        paths.cache_root / "f2/w2v1/table6", questions_sha256=digest
    )
    table4_cache = Table4EvaluationCache(
        paths.cache_root / "f2/w2v1/table4", questions_sha256=digest
    )
    records = []
    unavailable = []
    cache_hits = 0
    cache_misses = 0
    for architecture, seeds in sorted(complete_conditions(runs).items()):
        dimension, _, _, _, days, cores = _REFERENCE[architecture]
        for seed, run in sorted(seeds.items()):
            cached = evaluation_cache.load(run.info.run_id)
            if (
                cached is not None
                and cached.seed == seed
                and cached.architecture == architecture
            ):
                records.append(cached)
                cache_hits += 1
                continue
            if architecture == "nnlm":
                t4_cached = table4_cache.load(run.info.run_id)
                if (
                    t4_cached is not None
                    and t4_cached.seed == seed
                    and t4_cached.architecture == "nnlm-d100"
                ):
                    record = Table6RunResult(
                        architecture,
                        dimension,
                        seed,
                        run.info.run_id,
                        4,
                        t4_cached.semantic_accuracy_percent,
                        t4_cached.syntactic_accuracy_percent,
                        t4_cached.total_accuracy_percent,
                        t4_cached.included_questions,
                        t4_cached.total_questions,
                        t4_cached.observed_training_seconds,
                        days,
                        cores,
                    )
                    evaluation_cache.store(record)
                    records.append(record)
                    cache_hits += 1
                    continue
            try:
                cache_misses += 1
                lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                result = evaluate_analogies(
                    lookup,
                    questions,
                    vocabulary_limit=None,
                    batch_size=256,
                )
                seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table6RunResult(
                    architecture,
                    dimension,
                    seed,
                    run.info.run_id,
                    4 if architecture == "nnlm" else 6,
                    100 * result.semantic.require_score(),
                    100 * result.syntactic.require_score(),
                    100 * result.overall.require_score(),
                    result.overall.valid_count,
                    result.overall.total_count,
                    seconds,
                    days,
                    cores,
                )
                evaluation_cache.store(record)
                records.append(record)

            except (FileNotFoundError, KeyError, ValueError) as exc:
                unavailable.append(
                    {"run_id": run.info.run_id, "status": "no_data", "reason": str(exc)}
                )
    (output / "unavailable.json").write_text(json.dumps(unavailable, indent=2) + "\n")
    print(
        f"W2V1 Table 6 analogy cache: {cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )
    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=Table6RunResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    lines = [
        "# W2V1 Table 6 — FineWeb surrogate",
        "",
        "Only complete three-seed durable canonical conditions are shown.",
        "",
        "| Architecture | D | Semantic (%) | Syntactic (%) | Total (%) | Included / total per seed | Observed training seconds | Paper accuracy: semantic / syntactic / total (%) | Paper days x CPU cores | Source table / MLflow run IDs |",
        "|---|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for architecture, reference in _REFERENCE.items():
        values = [value for value in records if value.architecture == architecture]
        dimension, sem, synt, total, days, cores = reference
        if not values:
            lines.append(
                f"| {architecture} | {dimension} | unavailable | unavailable | unavailable | — | — | {sem} / {synt} / {total} | {days} x {cores} | awaiting canonical runs |"
            )
            continue
        coverage = ", ".join(
            f"s{x.seed}: {x.included_questions}/{x.total_questions}" for x in values
        )
        ids = ", ".join(f"Table {x.source_table}: {x.mlflow_run_id}" for x in values)
        lines.append(
            f"| {architecture} | {dimension} | {_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | {_summary(values, 'total_accuracy_percent')} | {coverage} | {_summary(values, 'observed_training_seconds')} | {sem} / {synt} / {total} | {days} x {cores} | {ids} |"
        )
    lines.extend(
        [
            "",
            "Scores are mean ± sample SD; the unchanged full-vocabulary compute-accuracy evaluator is used, with OOV coverage reported.",
            "NNLM reuses the exact canonical Table 4 w2v1-google-news-nnlm-6b 100d durable run IDs and shared projection embeddings. There is no separate Table 6 NNLM training condition.",
            "Paper days x CPU cores are historical reference costs, not measurements of the local environment. Observed seconds sum final dense elapsed observations per epoch and may omit tails; no conversion or cost equivalence is assumed.",
            "FineWeb corpus substitution and NNLM unspecified history/hidden/activation/schedule reconstruction decisions remain explicit in YAML and catalog notes.",
            f"questions-words.txt SHA-256: `{digest}`.",
            "Paper reference: https://arxiv.org/pdf/1301.3781 (Table 6).",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (output,)


def _summary(values: list[Table6RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"
