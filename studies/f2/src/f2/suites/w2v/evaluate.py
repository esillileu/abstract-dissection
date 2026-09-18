"""Standalone evaluation of durable Word2Vec lookup artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import LookupEmbeddings, load_lookup_artifact
from .evaluation import (
    additive_composition,
    evaluate_analogies,
    nearest_tokens,
    parse_analogy_questions,
    pca_projection,
)

_IDENTITY_KEYS = (
    "schema_version",
    "config_digest",
    "vocabulary_digest",
    "corpus_digest",
    "resource_version",
    "completed_epochs",
    "processed_tokens",
)


def evaluate_lookup_artifact(
    suite: str,
    lookup_path: Path,
    questions_path: Path,
    *,
    phrase_separator: bytes = b"_",
    evaluation_resource_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable lookup artifact without a training session."""
    if suite not in {"w2v1", "w2v2"}:
        raise ValueError("evaluation suite must be w2v1 or w2v2")
    lookup = load_lookup_artifact(lookup_path)
    if not questions_path.is_file():
        raise ValueError(f"analogy questions do not exist: {questions_path}")
    try:
        separator = phrase_separator.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("phrase separator must be ASCII") from exc
    questions = questions_path.read_bytes()
    analogy = evaluate_analogies(
        lookup, parse_analogy_questions(questions.splitlines())
    )
    evaluation_identity = {
        "questions_sha256": hashlib.sha256(questions).hexdigest(),
        "phrase_separator": separator,
    }
    if evaluation_resource_version is not None:
        evaluation_identity["resource_version"] = evaluation_resource_version
    payload: dict[str, Any] = {
        "suite": suite,
        "lookup_identity": {key: lookup.manifest[key] for key in _IDENTITY_KEYS},
        "evaluation_identity": evaluation_identity,
        "analogy": {
            "overall": asdict(analogy.overall),
            "semantic": asdict(analogy.semantic),
            "syntactic": asdict(analogy.syntactic),
            "categories": [asdict(category) for category in analogy.categories],
        },
    }
    if suite == "w2v2":
        payload["phrase"] = _phrase_report(lookup, phrase_separator)
    return payload


def write_evaluation_report(payload: dict[str, Any], output_path: Path) -> Path:
    """Create a separate evaluation report without overwriting any artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"evaluation report already exists: {output_path}") from exc
    return output_path


def _phrase_report(lookup: LookupEmbeddings, separator: bytes) -> dict[str, object]:
    phrase_tokens = [
        token
        for row in range(lookup.counts.size)
        if separator and separator in (token := _token_at(lookup, row))
    ][:8]
    neighbors = {
        token.hex(): [
            asdict(item) | {"token": item.token.hex()}
            for item in nearest_tokens(lookup, token, limit=5)
        ]
        for token in phrase_tokens
    }
    composition_tokens = phrase_tokens[:2]
    return {
        "nearest_entities": neighbors,
        "additive_composition": {
            "tokens": [token.hex() for token in composition_tokens],
            "neighbors": [
                asdict(item) | {"token": item.token.hex()}
                for item in additive_composition(lookup, composition_tokens, limit=5)
            ],
        },
        "pca": {
            token.hex(): point
            for token, point in pca_projection(lookup, phrase_tokens).items()
        },
        "token_encoding": "hex",
    }


def _token_at(lookup: LookupEmbeddings, row: int) -> bytes:
    return bytes(
        lookup.token_bytes[lookup.token_offsets[row] : lookup.token_offsets[row + 1]]
    )


__all__ = ["evaluate_lookup_artifact", "write_evaluation_report"]
