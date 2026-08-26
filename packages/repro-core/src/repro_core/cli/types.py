"""Small helpers for domain-owned Typer callbacks."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Annotated, ParamSpec, TypeVar

import typer

P = ParamSpec("P")
T = TypeVar("T")

Experiments = Annotated[
    list[str] | None,
    typer.Option("-e", "--experiment", help="Experiment IDs or ranges."),
]
AtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-a", "--atomic-run", help="Include atomic run IDs."),
]
ExcludedAtomicRuns = Annotated[
    list[str] | None,
    typer.Option("-x", "--exclude-atomic-run", help="Exclude atomic run IDs."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", metavar="KEY=VALUE", help="Override YAML values."),
]


def cli_errors(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from None

    return wrapped
