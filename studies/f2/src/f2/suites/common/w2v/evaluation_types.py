"""Data classes and metric results for representation evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    metric_id: str
    score: float | None
    total_count: int
    valid_count: int
    excluded_count: int

    @property
    def coverage(self) -> float:
        return 0.0 if self.total_count == 0 else self.valid_count / self.total_count


@dataclass(frozen=True)
class AnalogyQuestion:
    category: str
    a: bytes
    b: bytes
    c: bytes
    d: bytes


@dataclass(frozen=True)
class AnalogyEvaluation:
    overall: EvaluationResult
    semantic: EvaluationResult
    syntactic: EvaluationResult
    categories: tuple[EvaluationResult, ...]


@dataclass(frozen=True)
class SimilarityPair:
    first: bytes
    second: bytes
    gold_similarity: float


@dataclass(frozen=True)
class NearestNeighbor:
    token: bytes
    cosine_similarity: float


__all__ = [
    "AnalogyEvaluation",
    "AnalogyQuestion",
    "EvaluationResult",
    "NearestNeighbor",
    "SimilarityPair",
]
