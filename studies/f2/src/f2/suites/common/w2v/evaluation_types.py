"""Data classes and metric results for representation evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """A score together with the denominator and vocabulary coverage that produced it."""

    metric_id: str
    score: float | None
    total_count: int
    valid_count: int
    excluded_count: int

    @property
    def coverage(self) -> float | None:
        return self.valid_count / self.total_count if self.total_count else None

    def require_score(self) -> float:
        if self.score is None:
            raise ValueError(f"undefined evaluation: {self.metric_id}")
        return self.score


@dataclass(frozen=True)
class AnalogyQuestion:
    category: str
    a: bytes
    b: bytes
    c: bytes
    expected: bytes

    @property
    def d(self) -> bytes:
        return self.expected


@dataclass(frozen=True)
class AnalogyEvaluation:
    overall: EvaluationResult
    semantic: EvaluationResult
    syntactic: EvaluationResult
    categories: tuple[EvaluationResult, ...]


@dataclass(frozen=True)
class SimilarityPair:
    left: bytes
    right: bytes
    human_score: float

    @property
    def first(self) -> bytes:
        return self.left

    @property
    def second(self) -> bytes:
        return self.right

    @property
    def gold_similarity(self) -> float:
        return self.human_score


@dataclass(frozen=True)
class SentenceCompletionQuestion:
    context: tuple[bytes, ...]
    candidates: tuple[bytes, ...]
    expected: bytes


@dataclass(frozen=True)
class NearestNeighbor:
    token: bytes
    score: float

    @property
    def cosine_similarity(self) -> float:
        return self.score


__all__ = [
    "AnalogyEvaluation",
    "AnalogyQuestion",
    "EvaluationResult",
    "NearestNeighbor",
    "SentenceCompletionQuestion",
    "SimilarityPair",
]
