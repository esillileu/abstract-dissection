"""Table 2 word-analogy evaluation over durable W2V1 MLflow artifacts."""

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

_CONDITION = re.compile(r"wmt--d(50|100|300|600)-w(24|49|98|196|391|783)m")
_DIMENSIONS = (50, 100, 300, 600)
_TRAINING_WORDS = (24, 49, 98, 196, 391, 783)
_SEEDS = (1, 7, 19)
_VOCABULARY_LIMIT = 30_000


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


def analyze_table2(
    tracking_uri: str,
    questions_path: Path,
    *,
    paths: RuntimePaths | None = None,
) -> Path:
    """Evaluate only conditions having all three durable canonical seed runs."""
    from mlflow import MlflowClient

    paths = paths or RuntimePaths.from_environment()
    questions_path = Path(questions_path)
    if not questions_path.is_file():
        raise ValueError(f"word analogy questions do not exist: {questions_path}")
    question_bytes = questions_path.read_bytes()
    questions = parse_analogy_questions(question_bytes.splitlines())
    if len(questions) != 19_544:
        raise ValueError(
            "canonical questions-words.txt must contain 19,544 questions; "
            f"found {len(questions)}"
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("f2.w2v1")
    if experiment is None:
        raise ValueError("MLflow experiment f2.w2v1 does not exist")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            "tags.`result.durable_complete` = 'true' AND "
            "tags.`experiment_spec.id` = 'w2v1-table2-cbow'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=10_000,
    )
    selected = _complete_conditions(runs)
    cache = MlflowArtifactCache(
        client, tracking_uri, root=paths.cache_root / "mlflow_artifact"
    )
    records: list[Table2RunResult] = []
    with artifact_download_progress():
        for (words, dimension), condition_runs in sorted(selected.items()):
            for seed, run in sorted(condition_runs.items()):
                lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                result = evaluate_analogies(
                    lookup,
                    questions,
                    vocabulary_limit=_VOCABULARY_LIMIT,
                )
                training_seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                records.append(
                    Table2RunResult(
                        training_words_millions=words,
                        vector_dimension=dimension,
                        seed=seed,
                        mlflow_run_id=run.info.run_id,
                        total_accuracy_percent=100.0 * result.overall.score,
                        semantic_accuracy_percent=100.0 * result.semantic.score,
                        syntactic_accuracy_percent=100.0 * result.syntactic.score,
                        observed_training_seconds=training_seconds,
                        included_questions=result.overall.valid_count,
                        total_questions=result.overall.total_count,
                    )
                )

    output = paths.analysis_output("f2", "w2v1")
    output.mkdir(parents=True, exist_ok=True)
    _write_run_results(output / "table2-runs.csv", records)
    _write_summary(
        output / "summary.md",
        records,
        questions_sha256=hashlib.sha256(question_bytes).hexdigest(),
    )
    (output / "table2-results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "questions_path": str(questions_path),
                    "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
                    "vocabulary_limit": _VOCABULARY_LIMIT,
                    "analogy": "b - a + c; cosine; exclude a, b, c",
                    "source_protocol": "tmikolov/word2vec compute-accuracy.c",
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


def _complete_conditions(runs: list[Any]) -> dict[tuple[int, int], dict[int, Any]]:
    grouped: dict[tuple[int, int], dict[int, Any]] = {}
    for run in runs:
        variant = run.data.tags.get("implementation.variant", "")
        match = _CONDITION.fullmatch(variant)
        if match is None:
            continue
        seed = int(run.data.tags.get("seed", "0"))
        if seed not in _SEEDS:
            continue
        key = (int(match.group(2)), int(match.group(1)))
        seeds = grouped.setdefault(key, {})
        if seed in seeds:
            raise ValueError(f"multiple durable runs found for {variant}, seed {seed}")
        seeds[seed] = run
    return {key: seeds for key, seeds in grouped.items() if set(seeds) == set(_SEEDS)}


def _write_run_results(path: Path, records: list[Table2RunResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=Table2RunResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


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


def _write_summary(
    path: Path, records: list[Table2RunResult], *, questions_sha256: str
) -> None:
    grouped: dict[tuple[int, int], list[Table2RunResult]] = {}
    for record in records:
        grouped.setdefault(
            (record.training_words_millions, record.vector_dimension), []
        ).append(record)
    lines = [
        "# W2V1 Table 2 word analogy accuracy",
        "",
        "Cells are the mean total accuracy (%) across seeds 1, 7, and 19. "
        "Only conditions with all three durable completed runs are shown.",
        "",
        "| Training words | 50 | 100 | 300 | 600 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for words in _TRAINING_WORDS:
        cells = []
        for dimension in _DIMENSIONS:
            values = grouped.get((words, dimension))
            cells.append(
                ""
                if values is None
                else f"{statistics.fmean(x.total_accuracy_percent for x in values):.2f}"
            )
        lines.append(f"| {words}M | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Condition details",
            "",
            "| Training words | Dimension | Runs | Total accuracy mean ± sample SD (%) | Observed training time mean ± sample SD | Questions included / total |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, values in sorted(grouped.items()):
        scores = [value.total_accuracy_percent for value in values]
        training_times = [value.observed_training_seconds for value in values]
        coverage = sorted({(x.included_questions, x.total_questions) for x in values})
        coverage_text = ", ".join(f"{valid}/{total}" for valid, total in coverage)
        lines.append(
            f"| {key[0]}M | {key[1]} | {len(values)} | "
            f"{statistics.fmean(scores):.2f} ± {statistics.stdev(scores):.2f} | "
            f"{_duration(statistics.fmean(training_times))} ± "
            f"{_duration(statistics.stdev(training_times))} | "
            f"{coverage_text} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Evaluation vocabulary: the first {_VOCABULARY_LIMIT:,} model vocabulary rows.",
            "- Questions with any OOV term are excluded; accuracy uses the remaining questions.",
            "- Candidate score: cosine with `b - a + c`; input terms `a`, `b`, and `c` are excluded.",
            "- Semantic is the first five source sections; syntactic is the remaining nine sections.",
            f"- `questions-words.txt` SHA-256: `{questions_sha256}`.",
            "- Per-run semantic, syntactic, total, observed training time, coverage, and MLflow IDs are in `table2-runs.csv`.",
            "- Per-run observed training time is the sum of the last dense `elapsed_seconds` observation in each epoch. It excludes setup and artifact publication, but may omit the short tail after the final observation interval.",
            "- Difference from the C executable: batched NumPy matrix multiplication replaces its scalar loop; token casing, float32 vector normalization, candidate order, positive-score initialization, filtering, and aggregation are preserved.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600.0)
    minutes, remaining_seconds = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {remaining_seconds:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {remaining_seconds:04.1f}s"
    return f"{remaining_seconds:.1f}s"


__all__ = ["Table2RunResult", "analyze_table2", "observed_training_seconds"]
