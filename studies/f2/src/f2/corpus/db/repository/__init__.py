"""Corpus state repository implementing atomic transitions and queries for PostgreSQL."""

from __future__ import annotations

from .acquisition import AcquisitionRepositoryMixin
from .artifacts import ArtifactsRepositoryMixin
from .base import BaseCorpusRepository
from .in_memory import InMemoryCorpusRepository
from .lineage import LineageRepositoryMixin
from .processing import ProcessingRepositoryMixin
from .protocol import CorpusStateRepositoryProtocol
from .validation import ValidationRepositoryMixin


class CorpusStateRepository(
    AcquisitionRepositoryMixin,
    ArtifactsRepositoryMixin,
    ProcessingRepositoryMixin,
    LineageRepositoryMixin,
    ValidationRepositoryMixin,
    BaseCorpusRepository,
):
    """PostgreSQL-backed operational state repository for the F2 corpus pipeline."""


__all__ = [
    "AcquisitionRepositoryMixin",
    "ArtifactsRepositoryMixin",
    "BaseCorpusRepository",
    "CorpusStateRepository",
    "CorpusStateRepositoryProtocol",
    "InMemoryCorpusRepository",
    "LineageRepositoryMixin",
    "ProcessingRepositoryMixin",
    "ValidationRepositoryMixin",
]
