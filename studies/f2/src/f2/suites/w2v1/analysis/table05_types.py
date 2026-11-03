"""Data structures and condition specifications for Table 5."""

from __future__ import annotations

from dataclasses import dataclass

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

ConditionKey = tuple[str, int, int, int]  # (arch, dim, words, epochs)


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


__all__ = [
    "TABLE5_SPEC",
    "ConditionKey",
    "Table5RunResult",
]
