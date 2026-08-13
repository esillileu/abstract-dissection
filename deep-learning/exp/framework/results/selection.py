"""Domain-neutral selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


def attempt_priority(
    *,
    canonical_namespace: bool,
    durable_complete: bool | None,
    disposition: str | None,
    start_time: int,
) -> tuple[int, int]:
    """Shared precedence for canonical, native legacy, and imported runs."""
    if canonical_namespace and durable_complete is not False:
        rank = 0
    elif not canonical_namespace and disposition is None:
        rank = 1
    elif disposition == "imported":
        rank = 2
    else:
        rank = 3
    return rank, -start_time


def select_first(
    candidates: Iterable[T],
    *,
    priority: Callable[[T], tuple[object, ...]],
) -> T | None:
    """Select the lowest-priority candidate, or ``None`` when empty."""
    values = tuple(candidates)
    return None if not values else min(values, key=priority)
