"""Shared metric and condition declaration contracts for F2 experimental suites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDeclaration:
    metric_id: str
    unit: str
    split: str
    axis: str
    native_ids: tuple[str, ...]
    value_scale: float = 1.0


@dataclass(frozen=True)
class ConditionDeclaration:
    canonical_id: str
    aliases: tuple[str, ...]
    metrics: tuple[MetricDeclaration, ...]


@dataclass(frozen=True)
class StudyDeclaration:
    study_id: str
    conditions: tuple[ConditionDeclaration, ...]


__all__ = [
    "ConditionDeclaration",
    "MetricDeclaration",
    "StudyDeclaration",
]
