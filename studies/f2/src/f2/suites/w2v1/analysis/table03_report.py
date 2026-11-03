"""Reporting and formatting routines for Table 3."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .table03 import Table3RunResult

_CONDITIONS = ("cbow", "nnlm", "skipgram")
_REFERENCE = {
    "cbow": (15.5, 53.1, 58.7),
    "nnlm": (14.8, 51.3, 55.4),
    "skipgram": (50.0, 55.9, 59.3),
}
MSR_SYNTACTIC_URL = "https://raw.githubusercontent.com/quadrismegistus/prosodic/master/prosodic/dict/word2vec/questions-words.msr.txt"


def write_summary(
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


__all__ = ["MSR_SYNTACTIC_URL", "_summary", "write_summary"]
