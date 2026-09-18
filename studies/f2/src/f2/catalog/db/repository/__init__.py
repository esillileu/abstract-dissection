"""Repository layer providing transactional operations and view queries for F2 Catalog DB."""

from __future__ import annotations

from .base import BaseCatalogRepository
from .papers import PapersRepositoryMixin
from .plans import PlansRepositoryMixin
from .requirements import RequirementsRepositoryMixin
from .resources import ResourcesRepositoryMixin
from .views import ViewsRepositoryMixin


class CatalogRepository(
    PapersRepositoryMixin,
    ResourcesRepositoryMixin,
    RequirementsRepositoryMixin,
    PlansRepositoryMixin,
    ViewsRepositoryMixin,
    BaseCatalogRepository,
):
    """PostgreSQL-backed repository for reproduction catalog entities and execution plans."""


__all__ = [
    "BaseCatalogRepository",
    "CatalogRepository",
    "PapersRepositoryMixin",
    "PlansRepositoryMixin",
    "RequirementsRepositoryMixin",
    "ResourcesRepositoryMixin",
    "ViewsRepositoryMixin",
]
