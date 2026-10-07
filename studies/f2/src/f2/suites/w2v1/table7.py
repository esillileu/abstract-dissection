"""Microsoft Research Sentence Completion Challenge Table 7 reproduction and analysis."""

from __future__ import annotations

import csv
import glob
import hashlib
import re
import statistics
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from f2.suites.w2v.artifacts import load_lookup_artifact
from f2.suites.w2v.evaluation import (
    evaluate_msr_sentence_completion,
    parse_msr_sentence_completion_questions,
)
from repro_core.context import RuntimePaths
from repro_mlflow.artifact_cache import MlflowArtifactCache, artifact_download_progress

from .analysis import observed_training_seconds

# Condition specification for Table 7: Skip-gram 640d on 50M words Holmes corpus
# (model, dim, words_millions, epochs, orig_accuracy)
TABLE7_SPEC = ("skipgram", 640, 50, 1, 48.0)

_SEEDS = {1, 7, 19}

_T7_SLOT_PATTERN = re.compile(r"w2v1-table7-holmes-r1-skipgram-d640-w50m-s(\d+)")

# Canonical benchmark mirror URLs and SHA-256 checksums
MSR_QUESTIONS_URL = (
    "https://raw.githubusercontent.com/XingxingZhang/td-treelstm/"
    "5f11c4f4e22362a476b9900f87e65f238b891dcb/msr_scripts/Holmes.lm_format.questions.txt"
)
MSR_ANSWERS_URL = (
    "https://raw.githubusercontent.com/XingxingZhang/td-treelstm/"
    "5f11c4f4e22362a476b9900f87e65f238b891dcb/msr_scripts/Holmes.lm_format.answers.txt"
)
MSR_CSV_URL = (
    "https://raw.githubusercontent.com/pochih/Sentence-Completion/"
    "346ff7f4106a452dc92d6e1f35fe1f0da854c800/data/testing_data.csv"
)

MSR_QUESTIONS_SHA256 = (
    "659b752b219e64b748f54333d0019a2878645ae430c8fb9d3254c2e561d220dc"
)
MSR_ANSWERS_SHA256 = "c9dec1e8c9168cb67260dad36ebd2041d12776c0f36cbfc295d9b8f08a699fad"
MSR_CSV_SHA256 = "f3a76019be32c5714aec7fef9763f87d7587d23421e38156c8bb16f4e40bfe0e"

HOLMES_REPO_URL = "https://github.com/pochih/Sentence-Completion.git"
HOLMES_REPO_COMMIT = "346ff7f4106a452dc92d6e1f35fe1f0da854c800"
HOLMES_CORPUS_SHA256 = (
    "2d37eb27d9de873da4f6419740230e23cb9f75b91cc67b456d1e1a3fc34b680f"
)
HOLMES_RAW_BOOKS_COUNT = 522
HOLMES_RAW_BYTES = 236110188


def _download_verified(url: str, expected_sha256: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == expected_sha256:
            return target

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"downloaded file from {url} checksum mismatch: {digest} != {expected_sha256}"
        )
    target.write_bytes(data)
    return target


def ensure_msr_benchmark_data(
    benchmark_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Ensure the canonical 1,040 questions and gold answers exist with verified checksums."""
    paths = RuntimePaths.from_environment()
    base_dir = (
        Path(benchmark_dir)
        if benchmark_dir
        else paths.dataset("f2") / "benchmarks" / "msr_sentence_completion"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    q_path = base_dir / "Holmes.lm_format.questions.txt"
    a_path = base_dir / "Holmes.lm_format.answers.txt"

    _download_verified(MSR_QUESTIONS_URL, MSR_QUESTIONS_SHA256, q_path)
    _download_verified(MSR_ANSWERS_URL, MSR_ANSWERS_SHA256, a_path)

    return q_path, a_path


def ensure_holmes_corpus(corpus_path: Path | None = None) -> Path:
    """Ensure the verified 522-book Holmes training corpus exists at corpus_path."""
    paths = RuntimePaths.from_environment()
    target = (
        Path(corpus_path)
        if corpus_path
        else paths.dataset("f2") / "corpus" / "holmes.txt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        h = hashlib.sha256()
        with target.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        if h.hexdigest() == HOLMES_CORPUS_SHA256:
            return target

    # Check cache directory for repo clone
    cache_dir = paths.cache_root / "f2" / "pochih_repo"
    data_dir = cache_dir / "data" / "Holmes_Training_Data"
    if not data_dir.is_dir():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                HOLMES_REPO_URL,
                str(cache_dir),
            ],
            check=True,
            capture_output=True,
        )

    book_files = sorted(glob.glob(str(data_dir / "*.TXT")))
    if len(book_files) != HOLMES_RAW_BOOKS_COUNT:
        raise ValueError(
            f"expected {HOLMES_RAW_BOOKS_COUNT} Gutenberg book files; found {len(book_files)}"
        )

    h = hashlib.sha256()
    with target.open("wb") as out:
        for fpath in book_files:
            with open(fpath, "rb") as inp:
                chunk = inp.read()
                out.write(chunk)
                h.update(chunk)

    digest = h.hexdigest()
    if digest != HOLMES_CORPUS_SHA256:
        raise ValueError(
            f"concatenated Holmes corpus checksum mismatch: {digest} != {HOLMES_CORPUS_SHA256}"
        )

    return target


@dataclass(frozen=True)
class Table7RunResult:
    architecture: str
    dimension: int
    training_words_millions: int
    epochs: int
    corpus: str
    seed: int
    mlflow_run_id: str
    overall_accuracy_percent: float
    dev_accuracy_percent: float
    test_accuracy_percent: float
    total_questions: int
    observed_training_seconds: float


def collect_table7_runs(runs: list[Any]) -> dict[int, Any]:
    """Group durable Table 7 runs by seed -> run."""
    grouped: dict[int, Any] = {}
    for run in runs:
        slot = run.data.tags.get("f2.planned_run_slot_id", "")
        match = _T7_SLOT_PATTERN.fullmatch(slot)
        if match:
            seed = int(match.group(1))
            if seed in _SEEDS:
                grouped[seed] = run
    return grouped


def analyze_table7_sources(
    tracking_uri: str,
    questions_path: Path | None = None,
    *,
    paths: RuntimePaths | None = None,
) -> list[Path]:
    """Aggregate durable Table 7 runs, evaluate sentence completion, and write reports."""
    paths = paths or RuntimePaths.from_environment()
    from mlflow import MlflowClient

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
    q_file, a_file = ensure_msr_benchmark_data(
        questions_path.parent
        if questions_path and questions_path.is_file()
        else questions_path
    )

    questions = parse_msr_sentence_completion_questions(
        q_file.read_bytes().splitlines(),
        a_file.read_bytes().splitlines(),
    )

    grouped_runs = collect_table7_runs(runs)

    run_results: list[Table7RunResult] = []
    with artifact_download_progress():
        for seed, run in sorted(grouped_runs.items()):
            lookup_dir = artifact_cache.get(run.info.run_id, "lookup")
            lookup = load_lookup_artifact(lookup_dir)
            evaluation = evaluate_msr_sentence_completion(lookup, questions)
            metrics_file = artifact_cache.get(
                run.info.run_id, "metrics/observations.csv"
            )
            seconds = observed_training_seconds(metrics_file)

            run_results.append(
                Table7RunResult(
                    architecture="skipgram",
                    dimension=640,
                    training_words_millions=50,
                    epochs=1,
                    corpus="holmes",
                    seed=seed,
                    mlflow_run_id=run.info.run_id,
                    overall_accuracy_percent=round(evaluation.overall.score * 100.0, 2),
                    dev_accuracy_percent=round(evaluation.dev.score * 100.0, 2),
                    test_accuracy_percent=round(evaluation.test.score * 100.0, 2),
                    total_questions=len(questions),
                    observed_training_seconds=seconds,
                )
            )

    out_dir = paths.analysis_output("f2", "table7")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_md = out_dir / "summary.md"
    summary_csv = out_dir / "summary.csv"

    _write_table7_markdown(summary_md, run_results)
    _write_table7_csv(summary_csv, run_results)

    return [summary_md, summary_csv]


def _write_table7_markdown(path: Path, results: list[Table7RunResult]) -> None:
    lines = [
        "# Table 7: Microsoft Research Sentence Completion Challenge Reproduction",
        "",
        "Evaluation of 640-dimensional Skip-gram on 1,040 questions from the MSR Sentence Completion Challenge.",
        "",
        "| Architecture | Dimension | Training Words | Epochs | Corpus | Paper Acc [%] | Repro Mean [%] | Repro Std [%] | Seeds | Questions |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if results:
        overall_scores = [r.overall_accuracy_percent for r in results]
        mean_acc = statistics.mean(overall_scores)
        std_acc = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        seed_str = ",".join(str(r.seed) for r in results)
        lines.append(
            f"| Skip-gram | 640 | 50M | 1 | Holmes | 48.0 | {mean_acc:.2f} | {std_acc:.2f} | {seed_str} | {results[0].total_questions} |"
        )
        lines.extend(
            [
                "",
                "## Seed-level Results",
                "",
                "| Seed | Overall Acc [%] | Dev Acc (520) [%] | Test Acc (520) [%] | Training Time [s] | MLflow Run ID |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in results:
            lines.append(
                f"| {r.seed} | {r.overall_accuracy_percent:.2f} | {r.dev_accuracy_percent:.2f} | {r.test_accuracy_percent:.2f} | {r.observed_training_seconds:.1f} | `{r.mlflow_run_id[:8]}` |"
            )
    else:
        lines.append(
            "| Skip-gram | 640 | 50M | 1 | Holmes | 48.0 | Pending | Pending | - | 1,040 |"
        )

    lines.extend(
        [
            "",
            "## Provenance & Data Specifications",
            "",
            f"- Training Corpus: Project Gutenberg Holmes collection (522 books, {HOLMES_RAW_BYTES:,} bytes, concatenated SHA-256 `{HOLMES_CORPUS_SHA256}`).",
            f"- Evaluation Questions: 1,040 SAT-style sentence completion questions (SHA-256 `{MSR_QUESTIONS_SHA256}`).",
            f"- Evaluation Gold Answers: 1,040 gold completion sentences (SHA-256 `{MSR_ANSWERS_SHA256}`).",
            "- Sentence Scoring: Candidate missing word at input, sum of predictions for all surrounding context words (cosine similarity on L2-normalized embeddings).",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table7_csv(path: Path, results: list[Table7RunResult]) -> None:
    fieldnames = [
        "architecture",
        "dimension",
        "training_words_millions",
        "epochs",
        "corpus",
        "seed",
        "mlflow_run_id",
        "overall_accuracy_percent",
        "dev_accuracy_percent",
        "test_accuracy_percent",
        "total_questions",
        "observed_training_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


__all__ = [
    "HOLMES_CORPUS_SHA256",
    "HOLMES_RAW_BOOKS_COUNT",
    "HOLMES_RAW_BYTES",
    "MSR_ANSWERS_SHA256",
    "MSR_QUESTIONS_SHA256",
    "TABLE7_SPEC",
    "Table7RunResult",
    "analyze_table7_sources",
    "collect_table7_runs",
    "ensure_holmes_corpus",
    "ensure_msr_benchmark_data",
]
