"""Phrase detection policy specification and token validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TOKEN_CHUNK_BYTES = 64 * 1024
_WHITESPACE = re.compile(rb"([ \t\r\n\v\f]+)")


def _validate_token(token: bytes) -> None:
    if b"\0" in token or token == b"</s>":
        raise ValueError(
            "phrase token violates engine contract (no NUL or reserved </s>)"
        )


def _tokens(path: Path, *, max_token_bytes: int | None = None):
    """Yield byte tokens and LF boundaries with bounded chunk/token storage."""
    pending = bytearray()
    unfinished_line = False
    with path.open("rb") as stream:
        while chunk := stream.read(TOKEN_CHUNK_BYTES):
            unfinished_line = (
                bool(chunk.rsplit(b"\n", 1)[-1]) if b"\n" in chunk else True
            )
            for index, part in enumerate(_WHITESPACE.split(chunk)):
                if index % 2 == 0:
                    pending.extend(part)
                    if max_token_bytes is not None and len(pending) > max_token_bytes:
                        raise ValueError(
                            "sentence preparation token exceeds 4096 bytes"
                        )
                else:
                    if pending:
                        _validate_token(pending)
                        yield bytes(pending)
                        pending.clear()
                    for _ in range(part.count(b"\n")):
                        yield None
        if pending:
            _validate_token(pending)
            yield bytes(pending)
        if unfinished_line:
            yield None


@dataclass(frozen=True)
class PhrasePolicy:
    passes: int
    threshold: float
    min_count: int
    separator: bytes = b"_"
    thresholds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.passes < 1
            or not math.isfinite(self.threshold)
            or self.threshold < 0
            or self.min_count < 1
        ):
            raise ValueError("invalid phrase detection policy")
        if self.thresholds is not None and (
            len(self.thresholds) != self.passes
            or any(not math.isfinite(t) or t < 0 for t in self.thresholds)
        ):
            raise ValueError(
                "thresholds must contain one finite nonnegative value per pass"
            )
        if (
            not self.separator
            or b"\0" in self.separator
            or any(byte in b" \t\r\n\v\f" for byte in self.separator)
        ):
            raise ValueError("phrase separator must be non-empty non-whitespace bytes")

    @classmethod
    def from_config(cls, values: Mapping[str, object]) -> PhrasePolicy:
        thresholds = values.get("thresholds")
        if thresholds is not None and not isinstance(thresholds, (list, tuple)):
            raise ValueError("phrase thresholds must be a list")
        return cls(
            passes=int(values["passes"]),
            threshold=float(values.get("threshold", 100.0)),
            min_count=int(values["min_count"]),
            separator=str(values.get("separator", "_")).encode("ascii"),
            thresholds=None
            if thresholds is None
            else tuple(float(t) for t in thresholds),
        )

    def record(self) -> dict[str, object]:
        return {
            "score_normalization": "token_total",
            "token_contract": "full_byte_tokens_no_nul_no_eos",
            "thresholds": list(self.thresholds or (self.threshold,) * self.passes),
            "passes": self.passes,
            "min_count": self.min_count,
            "separator_hex": self.separator.hex(),
        }


__all__ = [
    "TOKEN_CHUNK_BYTES",
    "PhrasePolicy",
    "_tokens",
    "_validate_token",
]
