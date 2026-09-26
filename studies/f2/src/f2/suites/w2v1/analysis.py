"""Table 2 word-analogy evaluation over durable W2V1 MLflow artifacts."""

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
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths
from repro_mlflow.artifact_cache import MlflowArtifactCache, artifact_download_progress

_CONDITION = re.compile(r"(wmt|lm1b|umbc)--d(50|100|300|600)-w(24|49|98|196|391|783)m")
CORPUS_SOURCES = ("wmt", "lm1b", "umbc")
_DIMENSIONS = (50, 100, 300, 600)
_TRAINING_WORDS = (24, 49, 98, 196, 391, 783)
_SEEDS = (1, 7, 19)
_VOCABULARY_LIMIT = 30_000
_EVALUATION_PROTOCOL = "compute-accuracy-v1"
QUESTIONS_WORDS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/20c129af10659f7c50e86e3be406df663beff438/questions-words.txt"
QUESTIONS_WORDS_SHA256 = (
    "8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36"
)


def ensure_questions_words(target_path: Path) -> Path:
    target = Path(target_path)
    if target.is_file():
        return target
    if target.name != "questions-words.txt":
        raise ValueError(f"word analogy questions do not exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    try:
        with urllib.request.urlopen(QUESTIONS_WORDS_URL) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != QUESTIONS_WORDS_SHA256:
            raise ValueError(
                f"downloaded questions-words.txt checksum mismatch: {digest} != {QUESTIONS_WORDS_SHA256}"
            )
        target.write_bytes(data)
        return target
    except Exception as err:
        raise ValueError(f"word analogy questions do not exist: {target}") from err


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
            "schema_version": 1,
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

    def store(
        self, result: Table2RunResult, *, lookup_identity: dict[str, object]
    ) -> None:
        path = self._path(result.mlflow_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "signature": self.signature,
                    "lookup_identity": lookup_identity,
                    "result": asdict(result),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _path(self, run_id: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise ValueError(f"invalid MLflow run ID: {run_id}")
        return self.root / f"{run_id}.json"


def analyze_table2(
    tracking_uri: str,
    questions_path: Path,
    *,
    corpus_source: str = "wmt",
    paths: RuntimePaths | None = None,
) -> Path:
    """Evaluate only conditions having all three durable canonical seed runs."""
    from mlflow import MlflowClient

    paths = paths or RuntimePaths.from_environment()
    if corpus_source not in CORPUS_SOURCES:
        raise ValueError(
            f"unsupported W2V1 corpus source {corpus_source!r}; "
            f"expected one of {', '.join(CORPUS_SOURCES)}"
        )
    questions_path = ensure_questions_words(questions_path)
    question_bytes = questions_path.read_bytes()
    questions_sha256 = hashlib.sha256(question_bytes).hexdigest()
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
    selected = _complete_conditions(runs, corpus_source=corpus_source)
    cache = MlflowArtifactCache(
        client, tracking_uri, root=paths.cache_root / "mlflow_artifact"
    )
    evaluation_cache = Table2EvaluationCache(
        paths.cache_root / "f2/w2v1/analogy", questions_sha256=questions_sha256
    )
    records: list[Table2RunResult] = []
    cache_hits = 0
    cache_misses = 0
    with artifact_download_progress():
        for (words, dimension), condition_runs in sorted(selected.items()):
            for seed, run in sorted(condition_runs.items()):
                cached = evaluation_cache.load(run.info.run_id)
                if cached is not None and (
                    cached.training_words_millions,
                    cached.vector_dimension,
                    cached.seed,
                ) == (words, dimension, seed):
                    records.append(cached)
                    cache_hits += 1
                    continue
                cache_misses += 1
                lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                result = evaluate_analogies(
                    lookup,
                    questions,
                    vocabulary_limit=_VOCABULARY_LIMIT,
                )
                training_seconds = observed_training_seconds(
                    cache.get(run.info.run_id, "metrics/observations.csv")
                )
                record = Table2RunResult(
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
                evaluation_cache.store(record, lookup_identity=lookup.manifest)
                records.append(record)

    print(
        f"W2V1 {corpus_source} analogy cache: "
        f"{cache_hits} hit(s), {cache_misses} miss(es)",
        file=sys.stderr,
    )

    output = paths.analysis_output("f2", "w2v1") / corpus_source
    output.mkdir(parents=True, exist_ok=True)
    _write_run_results(output / "table2-runs.csv", records)
    _write_summary(
        output / "summary.md",
        records,
        corpus_source=corpus_source,
        questions_sha256=questions_sha256,
    )
    (output / "table2-results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "corpus_source": corpus_source,
                    "questions_path": str(questions_path),
                    "questions_sha256": questions_sha256,
                    "vocabulary_limit": _VOCABULARY_LIMIT,
                    "evaluation_protocol": _EVALUATION_PROTOCOL,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
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


def analyze_table2_sources(
    tracking_uri: str,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    """Analyze one requested corpus or render each canonical corpus independently."""
    paths = paths or RuntimePaths.from_environment()
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    outputs = tuple(
        analyze_table2(
            tracking_uri,
            questions_path,
            corpus_source=source,
            paths=paths,
        )
        for source in sources
    )
    if corpus_source is None:
        root = paths.analysis_output("f2", "w2v1")
        root.mkdir(parents=True, exist_ok=True)
        lines = ["# W2V1 Table 2 analyses", ""]
        for source, output in zip(sources, outputs, strict=True):
            lines.append(f"- [{source.upper()}](./{output.name}/summary.md)")
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def _complete_conditions(
    runs: list[Any], *, corpus_source: str
) -> dict[tuple[int, int], dict[int, Any]]:
    grouped: dict[tuple[int, int], dict[int, Any]] = {}
    for run in runs:
        variant = run.data.tags.get("implementation.variant", "")
        match = _CONDITION.fullmatch(variant)
        if match is None or match.group(1) != corpus_source:
            continue
        seed = int(run.data.tags.get("seed", "0"))
        if seed not in _SEEDS:
            continue
        key = (int(match.group(3)), int(match.group(2)))
        seeds = grouped.setdefault(key, {})
        if seed in seeds:
            continue
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
    path: Path,
    records: list[Table2RunResult],
    *,
    corpus_source: str,
    questions_sha256: str,
) -> None:
    grouped: dict[tuple[int, int], list[Table2RunResult]] = {}
    for record in records:
        grouped.setdefault(
            (record.training_words_millions, record.vector_dimension), []
        ).append(record)
    lines = [
        f"# W2V1 Table 2 word analogy accuracy — {corpus_source.upper()}",
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


__all__ = [
    "CORPUS_SOURCES",
    "Table2EvaluationCache",
    "Table2RunResult",
    "analyze_table2",
    "analyze_table2_sources",
    "observed_training_seconds",
]
