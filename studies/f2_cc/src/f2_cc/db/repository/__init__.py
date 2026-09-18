"""F2-CC operational repository for sampling, processing, audit, and export."""

from __future__ import annotations

from .auditing import AuditingRepositoryMixin
from .base import BaseRepository
from .provenance import ProvenanceRepositoryMixin
from .runs import RunsRepositoryMixin
from .sampling import SamplingRepositoryMixin


class CorpusStateRepository(
    RunsRepositoryMixin,
    SamplingRepositoryMixin,
    AuditingRepositoryMixin,
    ProvenanceRepositoryMixin,
    BaseRepository,
):
    """PostgreSQL-backed operational state repository for the F2 corpus pipeline."""


__all__ = [
    "AuditingRepositoryMixin",
    "BaseRepository",
    "CorpusStateRepository",
    "ProvenanceRepositoryMixin",
    "RunsRepositoryMixin",
    "SamplingRepositoryMixin",
]
