from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from exp.analyze import RunRef
from exp.ds2.analyze import e02_ptb_word2vec as analysis


def test_nearest_words_uses_cosine_similarity_and_excludes_query() -> None:
    words = ["query", "near", "opposite"]
    word_to_id = {word: index for index, word in enumerate(words)}
    id_to_word = dict(enumerate(words))
    vectors = np.asarray([[1.0, 0.0], [2.0, 0.1], [-1.0, 0.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    result = analysis._nearest_words(
        "query",
        word_to_id,
        id_to_word,
        vectors,
        top=2,
    )

    assert [candidate.word for candidate in result.candidates] == [
        "near",
        "opposite",
    ]


def test_analogy_reports_expected_rank_and_top_five_hit() -> None:
    words = ["king", "man", "queen", "woman", "other"]
    word_to_id = {word: index for index, word in enumerate(words)}
    id_to_word = dict(enumerate(words))
    vectors = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [2.0, 0.0],
            [-1.0, -1.0],
        ]
    )

    result = analysis._analogy(
        "king",
        "man",
        "queen",
        "woman",
        word_to_id,
        id_to_word,
        vectors,
    )

    assert result.expected_rank == 1
    assert result.hit_at_5 is True
    assert result.candidates[0].word == "woman"


def test_render_writes_text_and_csv_without_a_graph(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    required_words = sorted(
        set(analysis.SIMILARITY_QUERIES).union(
            word for query in analysis.ANALOGY_QUERIES for word in query
        )
    )
    word_to_id = {word: index for index, word in enumerate(required_words)}
    id_to_word = dict(enumerate(required_words))
    vectors = np.random.default_rng(7).normal(size=(len(required_words), 4))

    artifact_root = tmp_path / "artifacts"
    checkpoint = tmp_path / "checkpoint"
    (artifact_root / "checkpoints").mkdir(parents=True)
    checkpoint.mkdir()
    np.savez(checkpoint / "model_parameters.npz", W_in=vectors)
    (artifact_root / "checkpoints" / "checkpoint_manifest.json").write_text(
        json.dumps({"final": {"path": str(checkpoint)}}),
        encoding="utf-8",
    )
    run = RunRef(
        "run-1",
        analysis.ATOMIC_RUN_IDS[0],
        "1",
        1,
        artifact_root,
    )
    grouped = {series: [] for series in analysis.ATOMIC_RUN_IDS}
    grouped[analysis.ATOMIC_RUN_IDS[0]] = [run]
    monkeypatch.setattr(analysis, "runs", lambda *_args, **_kwargs: grouped)
    monkeypatch.setattr(
        analysis,
        "load_ptb",
        lambda: {"word_to_id": word_to_id, "id_to_word": id_to_word},
    )

    outputs = analysis.render(
        object(),
        "band",
        tmp_path / "e02_band.png",
    )

    assert outputs == [
        tmp_path / "e02_word_vectors.txt",
        tmp_path / "e02_word_vectors.csv",
    ]
    assert not (tmp_path / "e02_band.png").exists()
    text = outputs[0].read_text(encoding="utf-8")
    assert "similarity you:" in text
    assert "expected=woman" in text
    assert text in capsys.readouterr().out
    with outputs[1].open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40
    assert {row["task"] for row in rows} == {"similarity", "analogy"}
