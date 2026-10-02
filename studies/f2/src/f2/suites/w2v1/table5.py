"""Full-vocabulary Table 5 epoch/data/dimension trade-off analysis over durable canonical runs."""

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

# Condition specifications for Table 5
# (model, dim, words_millions, epochs, orig_semantic, orig_syntactic, orig_total, orig_time)
TABLE5_SPEC = [
    ("cbow", 300, 783, 3, 15.5, 53.1, 36.1, "1 day"),
    ("skipgram", 300, 783, 3, 50.0, 55.9, 53.3, "3 days"),
    ("cbow", 300, 783, 1, 13.8, 49.9, 33.6, "0.3 day"),
    ("cbow", 300, 1600, 1, 16.1, 52.6, 36.1, "0.6 day"),
    ("cbow", 600, 783, 1, 15.4, 53.3, 36.2, "0.7 day"),
    ("skipgram", 300, 783, 1, 45.6, 52.2, 49.2, "1 day"),
    ("skipgram", 300, 1600, 1, 52.2, 55.1, 53.8, "2 days"),
    ("skipgram", 600, 783, 1, 56.7, 54.5, 55.5, "2.5 days"),
]

_SEEDS = {1, 7, 19}

_T4_SLOT_PATTERN = re.compile(
    r"w2v1-table4-([a-z0-9]+)-r1-(cbow|skipgram)-d300-w783m-s(\d+)"
)
_T5_SLOT_PATTERN = re.compile(
    r"w2v1-table5-([a-z0-9]+)-r1-(cbow|skipgram)-d(300|600)-w(783|1600)m-ep1-s(\d+)"
)


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


ConditionKey = tuple[str, int, int, int]  # (arch, dim, words, epochs)


def collect_table5_runs(
    runs: list[Any], corpus: str
) -> dict[ConditionKey, dict[int, Any]]:
    """Group durable runs by (architecture, dimension, words, epochs) -> seed -> run."""
    grouped: dict[ConditionKey, dict[int, Any]] = {}

    for run in runs:
        slot = run.data.tags.get("f2.planned_run_slot_id", "")
        # Check Table 4 reusable 3-epoch runs
        match_t4 = _T4_SLOT_PATTERN.fullmatch(slot)
        if match_t4 and match_t4.group(1) == corpus:
            arch = match_t4.group(2)
            seed = int(match_t4.group(3))
            if seed in _SEEDS:
                key: ConditionKey = (arch, 300, 783, 3)
                grouped.setdefault(key, {})[seed] = run
            continue

        # Check Table 5 1-epoch runs
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
    tracking_uri: str,
    questions_path: Path,
    *,
    corpus_source: str | None = None,
    paths: RuntimePaths | None = None,
) -> tuple[Path, ...]:
    """Evaluate Table 5 conditions directly from MLflow artifacts."""
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

    sources = ("wmt",) if corpus_source is None else (corpus_source,)
    outputs = []

    with artifact_download_progress():
        for corpus in sources:
            records = []
            conditions_map = collect_table5_runs(runs, corpus)

            for spec in TABLE5_SPEC:
                arch, dim, words, epochs, _, _, _, _ = spec
                key: ConditionKey = (arch, dim, words, epochs)
                condition_runs = conditions_map.get(key, {})
                for seed in sorted(condition_runs):
                    run = condition_runs[seed]
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
                        Table5RunResult(
                            architecture=arch,
                            dimension=dim,
                            training_words_millions=words,
                            epochs=epochs,
                            corpus=corpus,
                            seed=seed,
                            mlflow_run_id=run.info.run_id,
                            semantic_accuracy_percent=100 * result.semantic.score,
                            syntactic_accuracy_percent=100 * result.syntactic.score,
                            total_accuracy_percent=100 * result.overall.score,
                            included_questions=result.overall.valid_count,
                            total_questions=result.overall.total_count,
                            observed_training_seconds=seconds,
                        )
                    )

            output = paths.analysis_output("f2", "w2v1") / corpus / "table5"
            output.mkdir(parents=True, exist_ok=True)

            # 1. runs.csv
            with (output / "runs.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=tuple(Table5RunResult.__dataclass_fields__)
                )
                writer.writeheader()
                writer.writerows(asdict(record) for record in records)

            # 2. summary.md
            _write_summary(output / "summary.md", corpus, records, question_digest)

            # 3. results.json
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


def _fmt_stats(records: list[Table5RunResult], field: str) -> str:
    if not records:
        return "—"
    nums = [float(getattr(r, field)) for r in records]
    if len(nums) == 1:
        return f"{nums[0]:.2f}"
    return f"{statistics.fmean(nums):.2f} ± {statistics.stdev(nums):.2f}"


def _get_mean(records: list[Table5RunResult], field: str) -> float | None:
    if not records:
        return None
    nums = [float(getattr(r, field)) for r in records]
    return float(statistics.fmean(nums))


def _write_summary(
    path: Path, corpus: str, records: list[Table5RunResult], digest: str
) -> None:
    lines = [
        f"# W2V1 Table 5 — {corpus.upper()}",
        "",
        "## Table 5 Reproduction Comparison",
        "",
        "| Model | Dim | Words | Epochs | Original Semantic | Repro Semantic (%) | Original Syntactic | Repro Syntactic (%) | Original Total | Repro Total (%) | Original Time | Repro Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    grouped: dict[ConditionKey, list[Table5RunResult]] = {}
    for r in records:
        key = (r.architecture, r.dimension, r.training_words_millions, r.epochs)
        grouped.setdefault(key, []).append(r)

    for (
        arch,
        dim,
        words,
        epochs,
        orig_sem,
        orig_syn,
        orig_tot,
        orig_time,
    ) in TABLE5_SPEC:
        cond_records = grouped.get((arch, dim, words, epochs), [])
        repro_sem = _fmt_stats(cond_records, "semantic_accuracy_percent")
        repro_syn = _fmt_stats(cond_records, "syntactic_accuracy_percent")
        repro_tot = _fmt_stats(cond_records, "total_accuracy_percent")
        repro_sec = _fmt_stats(cond_records, "observed_training_seconds")
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        words_label = f"{words}M" if words < 1000 else f"{words / 1000:.1f}B"

        lines.append(
            f"| {model_name} | {dim} | {words_label} | {epochs} | "
            f"{orig_sem:.1f} | {repro_sem} | {orig_syn:.1f} | {repro_syn} | "
            f"{orig_tot:.1f} | {repro_tot} | {orig_time} | {repro_sec} |"
        )

    # Core Comparisons A, B, C
    lines.extend(
        [
            "",
            "---",
            "",
            "## Core Comparisons",
            "",
            "### A. Repeated Training (783M x 3 epochs) vs More Data (1.6B x 1 epoch)",
            "",
            "| Model | Dim | Condition 1 | Condition 2 | Δ Semantic | Δ Syntactic | Δ Total | Runtime Ratio (1.6B 1ep / 783M 3ep) |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_3ep = grouped.get((arch, 300, 783, 3), [])
        r_1ep = grouped.get((arch, 300, 1600, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_3ep and r_1ep:
            d_sem = _get_mean(r_1ep, "semantic_accuracy_percent") - _get_mean(
                r_3ep, "semantic_accuracy_percent"
            )
            d_syn = _get_mean(r_1ep, "syntactic_accuracy_percent") - _get_mean(
                r_3ep, "syntactic_accuracy_percent"
            )
            d_tot = _get_mean(r_1ep, "total_accuracy_percent") - _get_mean(
                r_3ep, "total_accuracy_percent"
            )
            ratio = _get_mean(r_1ep, "observed_training_seconds") / max(
                1e-6, _get_mean(r_3ep, "observed_training_seconds")
            )
            lines.append(
                f"| {model_name} | 300 | 783M x 3ep | 1.6B x 1ep | {d_sem:+.2f}% | {d_syn:+.2f}% | {d_tot:+.2f}% | {ratio:.2f}x |"
            )
        else:
            lines.append(
                f"| {model_name} | 300 | 783M x 3ep | 1.6B x 1ep | — | — | — | — |"
            )

    lines.extend(
        [
            "",
            "### B. Epoch Effect (783M x 3 epochs vs 783M x 1 epoch)",
            "",
            "| Model | Dim | Words | 3 Epochs Total (%) | 1 Epoch Total (%) | Δ Total (3ep - 1ep) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_3ep = grouped.get((arch, 300, 783, 3), [])
        r_1ep = grouped.get((arch, 300, 783, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_3ep and r_1ep:
            m_3ep = _get_mean(r_3ep, "total_accuracy_percent")
            m_1ep = _get_mean(r_1ep, "total_accuracy_percent")
            lines.append(
                f"| {model_name} | 300 | 783M | {m_3ep:.2f}% | {m_1ep:.2f}% | {m_3ep - m_1ep:+.2f}% |"
            )
        else:
            lines.append(f"| {model_name} | 300 | 783M | — | — | — |")

    lines.extend(
        [
            "",
            "### C. Vector Dimension Effect (300d vs 600d at 783M x 1 epoch)",
            "",
            "| Model | Words | Epochs | 300d Total (%) | 600d Total (%) | Δ Total (600d - 300d) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for arch in ("cbow", "skipgram"):
        r_300d = grouped.get((arch, 300, 783, 1), [])
        r_600d = grouped.get((arch, 600, 783, 1), [])
        model_name = "CBOW" if arch == "cbow" else "Skip-gram"
        if r_300d and r_600d:
            m_300d = _get_mean(r_300d, "total_accuracy_percent")
            m_600d = _get_mean(r_600d, "total_accuracy_percent")
            lines.append(
                f"| {model_name} | 783M | 1 | {m_300d:.2f}% | {m_600d:.2f}% | {m_600d - m_300d:+.2f}% |"
            )
        else:
            lines.append(f"| {model_name} | 783M | 1 | — | — | — |")

    lines.extend(
        [
            "",
            "---",
            "",
            "### Notes",
            "- 1M lexical vocabulary cap is a reconstruction decision matching Table 4.",
            "- Evaluation uses the full saved lookup vocabulary. OOV questions are excluded.",
            f"- questions-words.txt SHA-256: `{digest}`.",
            "- Observed time sums final dense elapsed observations per epoch.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
