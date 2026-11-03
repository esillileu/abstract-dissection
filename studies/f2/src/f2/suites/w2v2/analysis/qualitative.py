"""Qualitative query definitions and nearest-neighbor evaluation for Tables 4, 5, 6."""

from __future__ import annotations

from pathlib import Path

from .. import analysis as analysis_mod
from .. import paper


def queries_for(table: str, condition: str):
    if table == "3" and condition in {"neg15-subsampling", "hs-subsampling"}:
        return [("4", q, (q,), 1) for q in paper.TABLE4]
    if condition == "phrase-33b":
        return [("5", q, tuple(q.split()), 4) for q in paper.TABLE5] + [
            ("6", q, (q,), 3) for q in paper.TABLE6_QUERIES
        ]
    return []


def evaluate_qualitative_query(
    lookup_or_path,
    number: str,
    query: str,
    tokens: tuple[str, ...],
    limit: int,
) -> dict:
    try:
        if isinstance(lookup_or_path, Path):
            from f2.suites.common.w2v.artifacts import load_lookup_artifact

            lookup = load_lookup_artifact(lookup_or_path)
        else:
            lookup = lookup_or_path
        encoded = [t.encode() for t in tokens]
        neighbors = (
            analysis_mod.additive_composition(lookup, encoded, limit=limit)
            if number == "5"
            else analysis_mod.nearest_tokens(lookup, encoded[0], limit=limit)
        )
        return {
            "checkpoint_reused": True,
            "status": "complete" if len(neighbors) == limit else "no_data",
            "reason": ""
            if len(neighbors) == limit
            else "OOV, zero norm or insufficient valid candidates",
            "neighbors": [
                {
                    "token": n.token.decode("utf-8", errors="replace"),
                    "cosine": n.score,
                }
                for n in neighbors
            ],
        }
    except ValueError as exc:
        return {"status": "failed", "reason": str(exc)}


__all__ = [
    "evaluate_qualitative_query",
    "queries_for",
]
