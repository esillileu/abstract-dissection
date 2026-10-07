"""Full-vocabulary Table 3 comparison on both word-relationship benchmarks."""

from __future__ import annotations

import csv
import hashlib
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from repro_io.checksum import sha256_file
from repro_io.http import SerialDownloader

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import evaluate_analogies, parse_analogy_questions
from repro_core.context import RuntimePaths
from repro_mlflow.artifact_cache import MlflowArtifactCache, artifact_download_progress

from .analysis import CORPUS_SOURCES, ensure_questions_words, observed_training_seconds

MSR_SYNTACTIC_URL = "https://raw.githubusercontent.com/ResponsiblyAI/responsibly/715c13ff7cf19de9d42f92d2e8c4de697dd4c638/responsibly/we/data/benchmark/MSR-syntax.txt"
MSR_SYNTACTIC_SHA256 = (
    "ac90a8f492d19fd892cc0e27c5f63dcd9a520b0044b91acd99e521668cc1ebc5"
)
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


def ensure_msr_syntactic(target: Path) -> Path:
    """Verify a pinned MSR mirror; this is not the Sentence Completion dataset."""
    target = Path(target)
    if not target.is_file():
        if target.name != "MSR-syntax.txt":
            raise ValueError(f"MSR syntactic analogy questions do not exist: {target}")
        SerialDownloader().download(
            MSR_SYNTACTIC_URL,
            target,
            expected_sha256=MSR_SYNTACTIC_SHA256,
            expected_length=231634,
        )
    if sha256_file(target) != MSR_SYNTACTIC_SHA256:
        raise ValueError(f"MSR syntactic analogy checksum mismatch: {target}")
    questions = parse_analogy_questions(target.read_bytes().splitlines())
    if len(questions) != 8000 or {question.category for question in questions} != {
        "all"
    }:
        raise ValueError(
            "MSR syntactic mirror must contain 8,000 questions in its all section"
        )
    return target


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


def complete_conditions(runs: list[Any], corpus: str) -> dict[str, dict[int, Any]]:
    grouped: dict[str, dict[int, Any]] = {}
    for run in runs:
        tags = run.data.tags
        for architecture, condition in _CONDITIONS.items():
            if tags.get("implementation.variant") != f"{corpus}--{condition}":
                continue
            if (
                tags.get("experiment_spec.id") != f"w2v1-table3-{architecture}"
                or tags.get("result.durable_complete") != "true"
            ):
                continue
            seed = int(tags.get("seed", "0"))
            if (
                seed not in _SEEDS
                or tags.get("f2.planned_run_slot_id")
                != f"{_PLANS[corpus]}-{condition}-s{seed}"
            ):
                continue
            grouped.setdefault(architecture, {}).setdefault(seed, run)
    return {
        architecture: seeds
        for architecture, seeds in grouped.items()
        if set(seeds) == _SEEDS
    }


def analyze_table3_sources(
    tracking_uri: str,
    questions_path: Path,
    *,
    msr_questions_path: Path | None = None,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    from mlflow import MlflowClient

    paths = paths or RuntimePaths.from_environment()
    if corpus_source is not None and corpus_source not in CORPUS_SOURCES:
        raise ValueError(f"unsupported W2V1 corpus source: {corpus_source}")
    google_bytes = ensure_questions_words(questions_path).read_bytes()
    questions = parse_analogy_questions(google_bytes.splitlines())
    if len(questions) != 19_544:
        raise ValueError(
            f"canonical questions-words.txt must contain 19,544 questions; found {len(questions)}"
        )
    msr_path = ensure_msr_syntactic(
        msr_questions_path or paths.dataset("f2") / "benchmarks" / "MSR-syntax.txt"
    )
    msr_bytes = msr_path.read_bytes()
    msr = parse_analogy_questions(msr_bytes.splitlines())
    google_digest = hashlib.sha256(google_bytes).hexdigest()
    msr_digest = hashlib.sha256(msr_bytes).hexdigest()
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
    cache = MlflowArtifactCache(
        client, tracking_uri, root=paths.cache_root / "mlflow_artifact"
    )
    outputs = []
    sources = CORPUS_SOURCES if corpus_source is None else (corpus_source,)
    with artifact_download_progress():
        for corpus in sources:
            records = []
            for architecture, seeds in sorted(
                complete_conditions(runs, corpus).items()
            ):
                for seed, run in sorted(seeds.items()):
                    lookup = load_lookup_artifact(cache.get(run.info.run_id, "lookup"))
                    word_result = evaluate_analogies(
                        lookup, questions, vocabulary_limit=None, batch_size=16
                    )
                    msr_result = evaluate_analogies(
                        lookup, msr, vocabulary_limit=None, batch_size=16
                    ).overall
                    seconds = observed_training_seconds(
                        cache.get(run.info.run_id, "metrics/observations.csv")
                    )
                    records.append(
                        Table3RunResult(
                            architecture,
                            corpus,
                            seed,
                            run.info.run_id,
                            100 * word_result.semantic.score,
                            100 * word_result.syntactic.score,
                            100 * word_result.overall.score,
                            100 * msr_result.score,
                            word_result.semantic.valid_count,
                            word_result.semantic.total_count,
                            word_result.syntactic.valid_count,
                            word_result.syntactic.total_count,
                            msr_result.valid_count,
                            msr_result.total_count,
                            seconds,
                        )
                    )
            output = paths.analysis_output("f2", "w2v1") / corpus / "table3"
            output.mkdir(parents=True, exist_ok=True)
            with (output / "runs.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=Table3RunResult.__dataclass_fields__
                )
                writer.writeheader()
                writer.writerows(asdict(record) for record in records)
            _write_summary(
                output / "summary.md", corpus, records, google_digest, msr_digest
            )
            outputs.append(output)
    return tuple(outputs)


def _write_summary(
    path: Path,
    corpus: str,
    records: list[Table3RunResult],
    google_digest: str,
    msr_digest: str,
) -> None:
    lines = [
        f"# W2V1 Table 3 — {corpus.upper()}",
        "",
        "Only conditions with all three durable canonical seed runs are shown. Scores are mean ± sample SD (%).",
        "",
        "| Architecture | Runs | Semantic | Syntactic | MSR syntactic | Included / total: semantic; syntactic; MSR (per seed) | Observed training seconds | MLflow IDs | Paper reference: semantic / syntactic / MSR (%) |",
        "|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for architecture in _CONDITIONS:
        values = [record for record in records if record.architecture == architecture]
        if not values:
            continue
        reference = " / ".join(str(value) for value in _REFERENCE[architecture])
        coverage = ", ".join(
            f"s{x.seed}: {x.semantic_included}/{x.semantic_total}; {x.syntactic_included}/{x.syntactic_total}; {x.msr_included}/{x.msr_total}"
            for x in values
        )

        lines.append(
            f"| {architecture} | {len(values)} | {_summary(values, 'semantic_accuracy_percent')} | {_summary(values, 'syntactic_accuracy_percent')} | {_summary(values, 'msr_accuracy_percent')} | {coverage} | {_summary(values, 'observed_training_seconds')} | {', '.join(x.mlflow_run_id for x in values)} | {reference} |"
        )
    lines.extend(
        [
            "",
            "RNNLM remains a paper reference only (semantic / syntactic / MSR: 9 / 36 / 35); it is a separate follow-up.",
            "Both benchmarks use the full saved input-embedding vocabulary and the unchanged compute-accuracy protocol: float32 normalization, ASCII casing, b - a + c, positive cosine, exclude a/b/c, exclude OOV and report coverage.",
            "NNLM uses the shared input/projection embedding, not hidden or HS output weights.",
            "NNLM tanh, 3 epochs and concrete PS/AdaGrad settings are reconstruction decisions, recorded in e02_table3.yaml and existing catalog notes. Corpus substitutions are explicit.",
            "The MSR syntactic benchmark is a commit-pinned public mirror of 8,000 word analogies, distinct from Sentence Completion. Original Microsoft distribution bytes are not claimed.",
            f"questions-words.txt SHA-256: `{google_digest}`.",
            f"MSR-syntax.txt SHA-256: `{msr_digest}`.",
            f"MSR mirror: {MSR_SYNTACTIC_URL}",
            "Paper reference: https://arxiv.org/pdf/1301.3781 (Table 3).",
            "Observed training time sums the final dense elapsed observation per epoch and may omit each epoch's tail. It is separate from paper reference accuracies.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(values: list[Table3RunResult], field: str) -> str:
    numbers = [float(getattr(value, field)) for value in values]
    return f"{statistics.fmean(numbers):.2f} ± {statistics.stdev(numbers):.2f}"
