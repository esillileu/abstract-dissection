"""Shared declaration types for suite-owned result schemas."""

from __future__ import annotations

from dataclasses import dataclass

from ..identity import Variant


@dataclass(frozen=True)
class MetricDeclaration:
    metric_id: str
    unit: str
    split: str
    axis: str
    implemented_native_ids: tuple[str, ...]
    original_native_ids: tuple[str, ...]
    protocols: tuple[str, ...] = ("book-source-v1",)
    value_scale: float = 1.0

    @property
    def canonical_native_id(self) -> str:
        """Return the single metric key persisted by the canonical writer."""
        return self.implemented_native_ids[0]

    def native_ids(self, variant: Variant) -> tuple[str, ...]:
        """Return canonical storage keys, independent of implementation variant."""
        del variant
        return (self.canonical_native_id,)


@dataclass(frozen=True)
class ConditionDeclaration:
    canonical_id: str
    implemented_aliases: tuple[str, ...]
    original_aliases: tuple[str, ...]
    metrics: tuple[MetricDeclaration, ...]

    def aliases(self, variant: Variant) -> tuple[str, ...]:
        return (
            self.implemented_aliases
            if variant is Variant.IMPLEMENTED
            else self.original_aliases
        )


@dataclass(frozen=True)
class StudyDeclaration:
    study_id: str
    conditions: tuple[ConditionDeclaration, ...]
