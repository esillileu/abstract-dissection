"""Tests for W2V1 Table 7 Microsoft Research Sentence Completion reproduction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from f2.suites.w2v.evaluation import (
    SentenceCompletionQuestion,
    evaluate_msr_sentence_completion,
    parse_msr_sentence_completion_questions,
)
from f2.suites.w2v1.spec import parse_run_spec
from f2.suites.w2v1.table7 import (
    HOLMES_CORPUS_SHA256,
    HOLMES_RAW_BOOKS_COUNT,
    MSR_ANSWERS_SHA256,
    MSR_QUESTIONS_SHA256,
    TABLE7_SPEC,
    Table7RunResult,
    _write_table7_csv,
    _write_table7_markdown,
    ensure_msr_benchmark_data,
)


class DictLookup:
    """Minimal VectorLookup test double."""

    def __init__(self, vectors: dict[bytes, tuple[float, ...]]) -> None:
        self._tokens = list(vectors.keys())
        self._rows = {token: i for i, token in enumerate(self._tokens)}
        self.embeddings = np.asarray(
            [vectors[tok] for tok in self._tokens], dtype=np.float32
        )

    def row(self, token: bytes) -> int | None:
        return self._rows.get(token)

    def vector(self, token: bytes) -> np.ndarray | None:
        row = self.row(token)
        return None if row is None else self.embeddings[row]


def test_table7_dataset_parser_and_validation() -> None:
    """Verify that all 1,040 questions and gold answers parse cleanly and are bijective."""
    q_file, a_file = ensure_msr_benchmark_data()
    questions = parse_msr_sentence_completion_questions(
        q_file.read_bytes().splitlines(),
        a_file.read_bytes().splitlines(),
    )

    assert len(questions) == 1040
    # Dev: 520, Test: 520
    assert len(questions[:520]) == 520
    assert len(questions[520:]) == 520

    gold_choice_indices: list[int] = []
    for idx, q in enumerate(questions):
        assert len(q.candidates) == 5, f"question {idx} must have 5 candidates"
        assert q.expected in q.candidates, (
            f"question {idx} expected must be in candidates"
        )
        assert len(q.context) >= 1, f"question {idx} context must not be empty"
        choice_idx = q.candidates.index(q.expected)
        gold_choice_indices.append(choice_idx)

    # Choice distribution should cover all 5 slots (0..4) and be reasonably balanced
    for opt in range(5):
        count = gold_choice_indices.count(opt)
        assert count > 150, (
            f"option {opt} count {count} is too low for balanced choices"
        )


def test_msr_sentence_completion_deterministic_toy_case() -> None:
    """Verify Skip-gram sentence completion on a known vector space."""
    lookup = DictLookup(
        {
            b"sun": (1.0, 0.0),
            b"summer": (0.95, 0.05),
            b"hot": (0.9, 0.1),
            b"cold": (-0.9, 0.1),
            b"ice": (-1.0, 0.0),
            b"winter": (-0.95, 0.05),
        }
    )

    questions = [
        SentenceCompletionQuestion(
            context=(b"sun", b"summer"),
            candidates=(b"hot", b"cold"),
            expected=b"hot",
        ),
        SentenceCompletionQuestion(
            context=(b"ice", b"winter"),
            candidates=(b"hot", b"cold"),
            expected=b"cold",
        ),
    ]

    # Cosine scoring
    evaluation = evaluate_msr_sentence_completion(lookup, questions, scoring="cosine")
    assert evaluation.overall.score == 1.0
    assert evaluation.overall.total_count == 2
    assert evaluation.overall.valid_count == 2
    assert evaluation.predictions == (b"hot", b"cold")


def test_msr_sentence_completion_oov_and_tie_breaking() -> None:
    """Verify OOV context tokens are skipped gracefully and ties break deterministically."""
    lookup = DictLookup(
        {
            b"hot": (1.0, 0.0),
            b"cold": (-1.0, 0.0),
        }
    )

    questions = [
        # Missing context token (rare word) should not drop the question
        SentenceCompletionQuestion(
            context=(b"oov_word_1", b"oov_word_2"),
            candidates=(b"hot", b"cold"),
            expected=b"hot",
        ),
    ]

    evaluation = evaluate_msr_sentence_completion(lookup, questions, scoring="cosine")
    assert evaluation.overall.total_count == 1
    # Ties break to first candidate
    assert evaluation.predictions == (b"hot",)
    assert evaluation.overall.score == 1.0


def test_table7_config_and_run_spec_resolution() -> None:
    """Verify e06_table7.yaml parses as Skip-gram 640d on 50M words Holmes corpus."""
    config_path = (
        Path(__file__).parents[1]
        / "src"
        / "f2"
        / "suites"
        / "w2v1"
        / "config"
        / "e06_table7.yaml"
    )
    assert config_path.is_file()

    spec = parse_run_spec(config_path, atomic_run_id="holmes--skipgram-d640-w50m")
    assert spec.identity["study"] == "table7"
    assert spec.identity["experiment_spec_id"] == "w2v1-msr-sentence-skipgram"
    assert spec.training["model_kind"] == "skip_gram"
    assert spec.training["embedding_dimension"] == 640
    assert spec.training["window_radius"] == 10
    assert spec.training["context_policy"] == "dynamic"
    assert spec.training["epochs"] == 1
    assert spec.training["objective_kind"] == "hierarchical_softmax"
    assert spec.corpus["lexical_token_budget"] == 50000000

    spec_s1 = spec.with_seed(1)
    assert (
        spec_s1.identity["planned_run_slot_id"]
        == "w2v1-table7-holmes-r1-skipgram-d640-w50m-s1"
    )

    spec_s7 = spec.with_seed(7)
    assert (
        spec_s7.identity["planned_run_slot_id"]
        == "w2v1-table7-holmes-r1-skipgram-d640-w50m-s7"
    )


def test_table7_constants_and_provenance() -> None:
    """Verify Table 7 specification and SHA-256 constants."""
    assert TABLE7_SPEC == ("skipgram", 640, 50, 1, 48.0)
    assert HOLMES_RAW_BOOKS_COUNT == 522
    assert (
        HOLMES_CORPUS_SHA256
        == "2d37eb27d9de873da4f6419740230e23cb9f75b91cc67b456d1e1a3fc34b680f"
    )
    assert (
        MSR_QUESTIONS_SHA256
        == "659b752b219e64b748f54333d0019a2878645ae430c8fb9d3254c2e561d220dc"
    )
    assert (
        MSR_ANSWERS_SHA256
        == "c9dec1e8c9168cb67260dad36ebd2041d12776c0f36cbfc295d9b8f08a699fad"
    )


def test_table7_report_rendering(tmp_path: Path) -> None:
    """Verify markdown and CSV report rendering with paper 48.0% baseline."""
    mock_results = [
        Table7RunResult(
            architecture="skipgram",
            dimension=640,
            training_words_millions=50,
            epochs=1,
            corpus="holmes",
            seed=1,
            mlflow_run_id="abcdef123456",
            overall_accuracy_percent=48.2,
            dev_accuracy_percent=48.5,
            test_accuracy_percent=47.9,
            total_questions=1040,
            observed_training_seconds=120.5,
        ),
        Table7RunResult(
            architecture="skipgram",
            dimension=640,
            training_words_millions=50,
            epochs=1,
            corpus="holmes",
            seed=7,
            mlflow_run_id="fedcba654321",
            overall_accuracy_percent=47.8,
            dev_accuracy_percent=48.1,
            test_accuracy_percent=47.5,
            total_questions=1040,
            observed_training_seconds=118.0,
        ),
    ]

    md_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"

    _write_table7_markdown(md_path, mock_results)
    _write_table7_csv(csv_path, mock_results)

    content = md_path.read_text(encoding="utf-8")
    assert "48.0" in content
    assert "48.00" in content or "48.0" in content
    assert "Skip-gram" in content
    assert "Holmes" in content

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["architecture"] == "skipgram"
    assert rows[0]["dimension"] == "640"
    assert float(rows[0]["overall_accuracy_percent"]) == 48.2


def test_evaluate_lookup_artifact_w2v1_table7(tmp_path: Path) -> None:
    """Verify standalone evaluate_lookup_artifact on w2v1-table7 policy."""
    from w2v import (
        Corpus,
        Model,
        TrainingConfig,
        TrainingSession,
        Vocabulary,
        VocabularyConfig,
    )

    from f2.suites.w2v.artifacts import save_lookup_artifact
    from f2.suites.w2v.evaluate import evaluate_lookup_artifact

    corpus_file = tmp_path / "toy_corpus.txt"
    corpus_file.write_bytes(
        b"the quick brown fox jumps over the lazy dog\n"
        b"swear write migrate climb contribute alone in london\n"
        b"i have seen it on him and could swear to it\n"
    )
    corpus = Corpus(corpus_file)
    vocab = Vocabulary.build(corpus, VocabularyConfig(min_count=1, hash_capacity=100))
    cfg = TrainingConfig(
        model_kind="skip_gram",
        objective_kind="hierarchical_softmax",
        embedding_dimension=8,
        window_radius=2,
        epochs=1,
        thread_count=1,
        subsampling_threshold=0.0,
        negative_sample_count=0,
    )
    model = Model.create(vocab, cfg)
    session = TrainingSession(corpus, vocab, model, cfg)
    session.train_epoch()

    lookup_dir = tmp_path / "lookup"
    save_lookup_artifact(session.export_state(), lookup_dir)

    q_file, a_file = ensure_msr_benchmark_data()
    # Evaluate with benchmark directory
    report = evaluate_lookup_artifact("w2v1-table7", lookup_dir, q_file.parent)
    assert report["suite"] == "w2v1-table7"
    assert "sentence_completion" in report
    overall = report["sentence_completion"]["overall"]
    assert overall["total_count"] == 1040
    assert overall["valid_count"] == 1040
    assert 0.0 <= overall["score"] <= 1.0
