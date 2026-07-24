"""DS2 GT02: inspect final PTB Word2Vec embeddings with the book queries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from exp.analyze import artifact_file
from mlprosection.datasets import load_ptb

from .common import runs


ATOMIC_RUN_IDS = (
    "W2V-PTB-CBOW-NS",
    "W2V-PTB-SKIPGRAM-NS",
    "W2V-PTB-CBOW-FULL",
    "W2V-PTB-SKIPGRAM-FULL",
)
SIMILARITY_QUERIES = ("you", "year", "car", "toyota")
ANALOGY_QUERIES = (
    ("king", "man", "queen", "woman"),
    ("take", "took", "go", "went"),
    ("car", "cars", "child", "children"),
    ("good", "better", "bad", "worse"),
)
TOP_K = 5
CSV_FIELDS = (
    "series",
    "seed",
    "run_id",
    "task",
    "query",
    "expected",
    "expected_rank",
    "hit_at_5",
    "candidate_rank",
    "candidate",
    "score",
)


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

    remote_path = (
        "checkpoints/final.npz"
        if final_path.suffix == ".npz"
        else f"checkpoints/{final_path.name}/model_parameters.npz"
    )
    return artifact_file(client, run, remote_path)


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
        _analogy(*query, word_to_id, id_to_word, vectors)
        for query in ANALOGY_QUERIES
    )
    return RunEvaluation(series, seed, run_id, similarities, analogies)


def _candidate_text(candidates: tuple[RankedCandidate, ...]) -> str:
    return ", ".join(
        f"{candidate.word} ({candidate.score:.6f})"
        for candidate in candidates
    )


def _text(evaluations: list[RunEvaluation], missing: dict[str, int]) -> str:
    lines = ["e02 PTB Word2Vec embedding evaluation (book queries; top 5)"]
    for series in ATOMIC_RUN_IDS:
        selected = [item for item in evaluations if item.series == series]
        if not selected:
            lines.append(f"[{series}] no completed runs with readable final checkpoints")
            continue
        for evaluation in selected:
            lines.append(
                f"[{series}] seed={evaluation.seed}, run_id={evaluation.run_id}"
            )
            for result in evaluation.similarities:
                lines.append(
                    f"similarity {result.query}: {_candidate_text(result.candidates)}"
                )
            for result in evaluation.analogies:
                expected_rank = (
                    "n/a" if result.expected_rank is None else str(result.expected_rank)
                )
                lines.append(
                    f"analogy {result.query} expected={result.expected}, "
                    f"rank={expected_rank}, hit@5={'yes' if result.hit_at_5 else 'no'}: "
                    f"{_candidate_text(result.candidates)}"
                )
        if missing.get(series):
            lines.append(
                f"[{series}] skipped unreadable checkpoints: {missing[series]}"
            )
    return "\n".join(lines) + "\n"


def _csv_rows(evaluations: list[RunEvaluation]):
    for evaluation in evaluations:
        base = {
            "series": evaluation.series,
            "seed": evaluation.seed,
            "run_id": evaluation.run_id,
        }
        for result in evaluation.similarities:
            for rank, candidate in enumerate(result.candidates, start=1):
                yield {
                    **base,
                    "task": "similarity",
                    "query": result.query,
                    "expected": "",
                    "expected_rank": "",
                    "hit_at_5": "",
                    "candidate_rank": rank,
                    "candidate": candidate.word,
                    "score": f"{candidate.score:.9g}",
                }
        for result in evaluation.analogies:
            for rank, candidate in enumerate(result.candidates, start=1):
                yield {
                    **base,
                    "task": "analogy",
                    "query": result.query,
                    "expected": result.expected,
                    "expected_rank": (
                        "" if result.expected_rank is None else result.expected_rank
                    ),
                    "hit_at_5": "true" if result.hit_at_5 else "false",
                    "candidate_rank": rank,
                    "candidate": candidate.word,
                    "score": f"{candidate.score:.9g}",
                }


def _output_paths(output: Path) -> tuple[Path, Path]:
    stem = output.stem
    for error_style in ("band", "errorbar"):
        marker = f"_{error_style}"
        if marker in stem:
            stem = stem.replace(marker, "_word_vectors", 1)
            break
    else:
        stem = f"{stem}_word_vectors"
    return output.with_name(stem).with_suffix(".txt"), output.with_name(stem).with_suffix(".csv")


def _write_csv(path: Path, evaluations: list[RunEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(evaluations))


def render(client, error_style, output):
    del error_style
    grouped = runs(client, "GT02", list(ATOMIC_RUN_IDS))
    ptb = load_ptb()
    word_to_id = ptb["word_to_id"]
    id_to_word = ptb["id_to_word"]
    evaluations: list[RunEvaluation] = []
    missing: dict[str, int] = {}
    for series in ATOMIC_RUN_IDS:
        for run in grouped[series]:
            checkpoint = _checkpoint_weights_path(client, run)
            if checkpoint is None:
                missing[series] = missing.get(series, 0) + 1
                continue
            try:
                vectors = _word_vectors(checkpoint)
                evaluations.append(
                    evaluate_vectors(
                        series,
                        run.seed,
                        run.run_id,
                        vectors,
                        word_to_id,
                        id_to_word,
                    )
                )
            except (OSError, ValueError):
                missing[series] = missing.get(series, 0) + 1
    text = _text(evaluations, missing)
    text_path, csv_path = _output_paths(output)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    _write_csv(csv_path, evaluations)
    print(text, end="")
    return [text_path, csv_path]
