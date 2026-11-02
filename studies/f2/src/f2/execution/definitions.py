"""Execution definitions and suite registry for F2.

Implements ref.md section 1, 2 & 3.4:
- Centralizes F2 suite registration and definitions
- Eliminates divergence between definition.py and protocol.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class F2SuiteDefinition:
    """Canonical registry entry for an F2 reproduction suite."""

    suite_id: str
    paper_title: str
    config_dir: Path
    executor_module: str
    description: str = ""


_SUITES: dict[str, F2SuiteDefinition] = {}


def register_suite(suite: F2SuiteDefinition) -> None:
    _SUITES[suite.suite_id] = suite


def get_suite(suite_id: str) -> F2SuiteDefinition | None:
    return _SUITES.get(suite_id)


def list_suites() -> list[F2SuiteDefinition]:
    return list(_SUITES.values())


__all__ = [
    "F2SuiteDefinition",
    "get_suite",
    "list_suites",
    "register_suite",
]
