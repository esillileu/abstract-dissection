"""Full-vocabulary Table 4 analogy analysis over durable canonical runs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths
from repro_mlflow.artifact_cache import MlflowArtifactCache, artifact_download_progress

from .analysis import CORPUS_SOURCES, ensure_questions_words, observed_training_seconds

_CONDITION = re.compile(r"(wmt|lm1b|umbc)--(cbow|skipgram)-d300-w783m")
_SEEDS = {1, 7, 19}
_REFERENCE = {"cbow": (15.5, 53.1, 36.1), "skipgram": (50.0, 55.9, 53.3)}


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


def complete_conditions(runs: list[Any], corpus: str) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        match = _CONDITION.fullmatch(run.data.tags.get("implementation.variant", ""))
        if match is None or match.group(1) != corpus:
            continue
        architecture = match.group(2)
        if (
            run.data.tags.get("experiment_spec.id")
            != f"w2v1-google-news-{architecture}-scale"
        ):
            continue
        seed = int(run.data.tags.get("seed", "0"))
        slot = f"w2v1-table4-{corpus}-r1-{architecture}-d300-w783m-s{seed}"
        if seed not in _SEEDS or run.data.tags.get("f2.planned_run_slot_id") != slot:
            continue
        grouped.setdefault(architecture, {}).setdefault(seed, run)
    return {key: value for key, value in grouped.items() if set(value) == _SEEDS}


def analyze_table4_sources(
    tracking_uri: str,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    from mlflow import MlflowClient

    paths = paths or RuntimePaths.from_environment()
    if corpus_source is not None and corpus_source not in CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")
    questions_path = ensure_questions_words(questions_path)
    question_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if len(questions) != 19_544:
        raise ValueError(
            f"canonical questions-words.txt must contain 19,544 questions; found {len(questions)}"
        )
    question_digest = hashlib.sha256(question_bytes).hexdigest()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("f2.w2v1")
    if experiment is None:
        raise ValueError("MLflow experiment f2.w2v1 does not exist")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.`result.durable_complete` = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )
    artifact_cache = MlflowArtifactCache(
        client, tracking_uri, root=paths.cache_root / "mlflow_artifact"
    )
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    outputs = []
    with artifact_download_progress():
        for corpus in sources:
            records = []
            for architecture, condition_runs in sorted(
                complete_conditions(runs, corpus).items()
            ):
                for seed, run in sorted(condition_runs.items()):
                    lookup = load_lookup_artifact(
                        artifact_cache.get(run.info.run_id, "lookup")
                    )
                    result = evaluate_analogies(
                        lookup, questions, vocabulary_limit=None, batch_size=16
                    )
                    seconds = observed_training_seconds(
                        artifact_cache.get(run.info.run_id, "metrics/observations.csv")
                    )
                    records.append(
                        Table4RunResult(
                            architecture,
                            corpus,
                            seed,
                            run.info.run_id,
                            100 * result.semantic.score,
                            100 * result.syntactic.score,
                            100 * result.overall.score,
                            result.overall.valid_count,
                            result.overall.total_count,
                            seconds,
                        )
                    )
            output = paths.analysis_output("f2", "w2v1") / corpus / "table4"
            output.mkdir(parents=True, exist_ok=True)
            with (output / "runs.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=Table4RunResult.__dataclass_fields__
                )
                writer.writeheader()
                writer.writerows(asdict(record) for record in records)
            _write_summary(output / "summary.md", corpus, records, question_digest)
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "protocol": {
                            "questions_sha256": question_digest,
                            "vocabulary_limit": None,
                            "analogy": "b - a + c; cosine; exclude a, b, c",
                        },
                        "reference": _REFERENCE,
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


def _write_summary(
    path: Path, corpus: str, records: list[Table4RunResult], digest: str
) -> None:
    lines = [
        f"# W2V1 Table 4 — {corpus.upper()}",
        "",
        "Only conditions with all three durable canonical runs are shown.",
        "",
        "| Architecture | Corpus | Runs | Semantic mean ± sample SD (%) | Syntactic mean ± sample SD (%) | Total mean ± sample SD (%) | Questions included / total | Observed training time mean ± sample SD (s) | MLflow run IDs | Paper reference: semantic / syntactic / total (%) |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for architecture in ("cbow", "skipgram"):
        values = [record for record in records if record.architecture == architecture]
        if not values:
            continue
        coverage = ", ".join(
            f"{x.included_questions}/{x.total_questions}" for x in values
        )
        reference = " / ".join(f"{number:.1f}" for number in _REFERENCE[architecture])
        lines.append(
            f"| {architecture} | {corpus} | {len(values)} | "
            f"{_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | "
            f"{_summary(values, 'total_accuracy_percent')} | {coverage} | "
            f"{_summary(values, 'observed_training_seconds')} | "
            f"{', '.join(x.mlflow_run_id for x in values)} | {reference} |"
        )
    lines.extend(
        [
            "",
            "The 1M lexical vocabulary cap is a reconstruction decision; Table 4 does not state it explicitly.",
            "Evaluation uses the full saved lookup vocabulary. OOV questions are excluded.",
            "Scores preserve the existing compute-accuracy casing, cosine normalization, positive-score, and source-exclusion behavior.",
            f"questions-words.txt SHA-256: `{digest}`.",
            "Observed time sums final dense elapsed observations per epoch and may omit each epoch's tail.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(values: list[Table4RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"
