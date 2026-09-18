from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dlfs.analysis.input import artifact_file as _default_artifact_file

from .constants import ANALOGY_QUERIES, SIMILARITY_QUERIES, TOP_K


def _get_artifact_file():
    """Allow monkeypatching via e02_ptb_word2vec.artifact_file."""
    pkg = sys.modules.get("dlfs.ds2.analysis.e02_ptb_word2vec")
    if pkg is not None and hasattr(pkg, "artifact_file"):
        return pkg.artifact_file
    return _default_artifact_file


@dataclass(frozen=True)
class RankedCandidate:
    word: str
    score: float


@dataclass(frozen=True)
class SimilarityResult:
    query: str
    candidates: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class AnalogyResult:
    a: str
    b: str
    c: str
    expected: str
    expected_rank: int | None
    candidates: tuple[RankedCandidate, ...]

    @property
    def query(self) -> str:
        return f"{self.a}:{self.b} = {self.c}:?"

    @property
    def hit_at_5(self) -> bool:
        return self.expected_rank is not None and self.expected_rank <= TOP_K


@dataclass(frozen=True)
class RunEvaluation:
    series: str
    seed: str
    run_id: str
    similarities: tuple[SimilarityResult, ...]
    analogies: tuple[AnalogyResult, ...]


def _checkpoint_weights_path(client, run) -> Path | None:
    artifact_file = _get_artifact_file()
    manifest_path = artifact_file(
        client,
        run,
        "checkpoints/checkpoint_manifest.json",
    )
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final = manifest.get("final")
        if not isinstance(final, dict) or not final.get("path"):
            return None
        final_path = Path(str(final["path"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    candidates = []
    if final_path.is_dir():
        candidates.append(final_path / "model_parameters.npz")
    elif final_path.suffix == ".npz":
        candidates.append(final_path)
    if run.local_artifact_root is not None:
        candidates.append(
            run.local_artifact_root
            / "checkpoints"
            / final_path.name
            / "model_parameters.npz"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    remote_paths = (
        ("checkpoints/final.npz",)
        if final_path.suffix == ".npz"
        else (
            f"checkpoints/generations/{final_path.name}/model_parameters.npz",
            f"checkpoints/{final_path.name}/model_parameters.npz",
        )
    )
    return next(
        (
            downloaded
            for remote_path in remote_paths
            if (downloaded := artifact_file(client, run, remote_path)) is not None
        ),
        None,
    )


def _word_vectors(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if "W_in" not in arrays:
            raise ValueError("Word2Vec checkpoint does not contain W_in")
        vectors = np.asarray(arrays["W_in"], dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"W_in must be a matrix, got {vectors.shape}")
    return vectors


def _ordered_indices(scores: np.ndarray, excluded: set[int]) -> np.ndarray:
    filtered = np.asarray(scores, dtype=float).copy()
    if excluded:
        filtered[np.fromiter(excluded, dtype=np.int64)] = -np.inf
    filtered[~np.isfinite(filtered)] = -np.inf
    return np.argsort(-filtered, kind="stable")


def _nearest_words(
    query: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    normalized_vectors: np.ndarray,
    *,
    top: int = TOP_K,
) -> SimilarityResult:
    query_id = word_to_id[query]
    scores = normalized_vectors @ normalized_vectors[query_id]
    ordered = _ordered_indices(scores, {query_id})
    candidates = tuple(
        RankedCandidate(id_to_word[int(index)], float(scores[index]))
        for index in ordered[:top]
    )
    return SimilarityResult(query, candidates)


def _analogy(
    a: str,
    b: str,
    c: str,
    expected: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    vectors: np.ndarray,
    *,
    top: int = TOP_K,
) -> AnalogyResult:
    source_ids = {word_to_id[word] for word in (a, b, c)}
    query = vectors[word_to_id[b]] - vectors[word_to_id[a]] + vectors[word_to_id[c]]
    norm = float(np.linalg.norm(query))
    scores = vectors @ (query / norm) if norm else np.full(len(vectors), np.nan)
    ordered = _ordered_indices(scores, source_ids)
    candidates = tuple(
        RankedCandidate(id_to_word[int(index)], float(scores[index]))
        for index in ordered[:top]
    )
    expected_id = word_to_id.get(expected)
    expected_rank = None
    if expected_id is not None and expected_id not in source_ids:
        matches = np.flatnonzero(ordered == expected_id)
        if len(matches):
            expected_rank = int(matches[0]) + 1
    return AnalogyResult(a, b, c, expected, expected_rank, candidates)


def evaluate_vectors(
    series: str,
    seed: str,
    run_id: str,
    vectors: np.ndarray,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
) -> RunEvaluation:
    if len(vectors) != len(word_to_id):
        raise ValueError(
            f"vocabulary/checkpoint mismatch: {len(word_to_id)} != {len(vectors)}"
        )
    required = set(SIMILARITY_QUERIES)
    required.update(word for query in ANALOGY_QUERIES for word in query)
    missing = sorted(required - set(word_to_id))
    if missing:
        raise ValueError(f"PTB vocabulary is missing book queries: {missing}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors),
        where=norms != 0,
    )
    similarities = tuple(
        _nearest_words(query, word_to_id, id_to_word, normalized)
        for query in SIMILARITY_QUERIES
    )
    analogies = tuple(
        _analogy(*query, word_to_id, id_to_word, vectors) for query in ANALOGY_QUERIES
    )
    return RunEvaluation(series, seed, run_id, similarities, analogies)
