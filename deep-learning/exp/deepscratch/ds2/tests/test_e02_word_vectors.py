from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from exp.framework.analysis.core import RunRef
from exp.framework.analysis.core import aggregate
from exp.deepscratch.ds2.analysis import e02_ptb_word2vec as analysis


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


def test_render_writes_ns_graphs_text_and_csv(
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

    class AnalysisInput:
        def runs(self, condition_ids):
            return {condition: grouped[condition] for condition in condition_ids}

        def artifact_file(self, _run, artifact_path):
            candidate = artifact_root / artifact_path
            return candidate if candidate.is_file() else None
    monkeypatch.setattr(
        analysis,
        "load_ptb",
        lambda: {"word_to_id": word_to_id, "id_to_word": id_to_word},
    )
    monkeypatch.setattr(
        analysis,
        "source_curve",
        lambda *_args: aggregate([{0.0: 2.0, 1.0: 1.0}]),
    )

    outputs = analysis.render(
        AnalysisInput(),
        "band",
        tmp_path / "e02_band.png",
    )

    assert outputs == [
        tmp_path / "e02_band_ns_cbow.png",
        tmp_path / "e02_band_ns_skipgram.png",
        tmp_path / "e02_band_ns_combined.png",
        tmp_path / "e02_band_ns_curves.csv",
        tmp_path / "e02_word_vectors.txt",
        tmp_path / "e02_word_vectors.csv",
    ]
    assert all(path.is_file() for path in outputs)
    text = outputs[4].read_text(encoding="utf-8")
    assert "similarity you:" in text
    assert "expected=woman" in text
    assert capsys.readouterr().out == ""
    with outputs[5].open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40
    assert {row["task"] for row in rows} == {"similarity", "analogy"}


def test_append_markdown_report_places_word_vectors_after_summary(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "e02.md"
    summary.write_text("# Summary\n\n| table |\n| --- |\n| value |\n", encoding="utf-8")
    report = tmp_path / "e02.txt"
    report.write_text(
        "[W2V-PTB-CBOW-NS] seed=1, run_id=run-1\n"
        "similarity you: we (0.9), i (0.8)\n"
        "analogy king:man = queen:? expected=woman, rank=1, hit@5=yes: woman (0.9), other (0.8)\n",
        encoding="utf-8",
    )

    analysis.append_markdown_report(summary, report)

    text = summary.read_text(encoding="utf-8")
    assert text.index("| value |") < text.index("## Word2Vec embedding evaluation")
    assert "| question | 1위 | 2위 | 3위 | 4위 | 5위 |" in text
    assert "| you | we | i |  |  |  |" in text
    assert "| king:man = queen:? (expected=woman, rank=1, hit@5=yes) | woman | other |  |  |  |" in text
