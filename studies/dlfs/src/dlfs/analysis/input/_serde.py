"""Serialization and caching fingerprint helpers for analysis input."""

from __future__ import annotations

import hashlib
import marshal
from collections.abc import Callable, Mapping


def seed_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def encode_history(history: Mapping[float, float]) -> list[list[float]]:
    return [[float(step), float(value)] for step, value in history.items()]


def decode_history(payload: object) -> dict[float, float]:
    if not isinstance(payload, list):
        return {}
    return {float(item[0]): float(item[1]) for item in payload}


def callable_fingerprint(function: Callable | None) -> str | None:
    if function is None:
        return None
    code = getattr(function, "__code__", None)
    if code is None:
        return repr(function)
    closure = tuple(
        repr(cell.cell_contents)
        for cell in (getattr(function, "__closure__", None) or ())
    )
    payload = marshal.dumps(code) + repr(closure).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
