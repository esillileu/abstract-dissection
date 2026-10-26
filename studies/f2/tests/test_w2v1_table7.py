"""Tests for W2V1 Table 7 Microsoft Research Sentence Completion reproduction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

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
    Table7EvaluationCache,
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


def test_materialize_corpus_for_table7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _materialize_corpus resolves direct file path without querying database shards."""
    from f2.suites.w2v.tracked import _materialize_corpus
    from repro_core.context import RuntimePaths

    fake_corpus = tmp_path / "holmes.txt"
    fake_corpus.write_text("sample content")

    config = {
        "identity": {
            "planned_run_slot_id": "w2v1-table7-holmes-r1-skipgram-d640-w50m-s1",
            "execution_plan_id": "w2v1-table7-holmes-r1",
            "config_digest": "dummy",
        },
        "corpus": {
            "path": str(fake_corpus),
            "lexical_token_budget": 50_000_000,
        },
    }
    paths = RuntimePaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        references_root=tmp_path / "references",
        studies_root=tmp_path / "studies",
    )
    monkeypatch.setattr(
        "f2.suites.w2v1.table7.ensure_holmes_corpus",
        lambda p: fake_corpus,
    )
    _materialize_corpus(config, paths)
    assert config["corpus"]["path"] == str(fake_corpus)
    assert "sha256" in config["corpus"]
    assert config["corpus"]["lexical_tokens"] == 50_000_000


def test_hs_manual_calculation_parity_and_bit_direction() -> None:
    """Verify manual calculation vs implementation for HS scoring and bit direction.

    Rust training convention:
      score = dot(v_c, u_j)
      if bit == 0: P = sigmoid(score)
      if bit == 1: P = 1 - sigmoid(score) = sigmoid(-score)
      Formula: log P = log sigmoid((1 - 2*b_j) * dot(v_c, u_j))
    """
    import math

    # 3-word vocabulary: w0, w1, w2
    # Huffman tree: 3 leaves, 2 internal nodes (0, 1)
    # w0 (token 0): path [1], bit [0]
    # w1 (token 1): path [1, 0], bit [1, 0]
    # w2 (token 2): path [1, 0], bit [1, 1]
    offsets = np.array([0, 1, 3, 5], dtype=np.uint64)
    paths = np.array([1, 1, 0, 1, 0], dtype=np.uint64)
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)

    lookup = DictLookup(
        {
            b"cand_a": (1.0, 0.5),
            b"cand_b": (-0.5, 1.0),
            b"ctx_w1": (0.2, -0.3),
        }
    )

    # 2 internal node embeddings (node 0 and node 1)
    # shape: (2, 2)
    output_embeddings = np.array(
        [
            [0.4, -0.2],  # node 0
            [-0.3, 0.6],  # node 1
        ],
        dtype=np.float32,
    )

    # For cand_a: v = [1.0, 0.5]
    # Context word is ctx_w1 (token 1): path = [node 1, node 0], bits = [1, 0]
    # Step 1: node 1, bit 1:
    #   dot(v, u1) = 1.0*(-0.3) + 0.5*0.6 = 0.0
    #   bit=1 -> sign = -1 -> score = -0.0 -> P1 = sigma(0.0) = 0.5, log P1 = log(0.5) = -0.693147
    # Step 2: node 0, bit 0:
    #   dot(v, u0) = 1.0*0.4 + 0.5*(-0.2) = 0.3
    #   bit=0 -> sign = +1 -> score = +0.3 -> P2 = 1/(1 + exp(-0.3)) = 0.5744425, log P2 = -0.5543552
    # Total log P(w1 | cand_a) = log(0.5) + log(0.5744425) = -1.247502
    # Total P(w1 | cand_a) = 0.5 * 0.5744425 = 0.287221

    v_a = np.array([1.0, 0.5], dtype=np.float64)
    u1 = output_embeddings[1].astype(np.float64)
    u0 = output_embeddings[0].astype(np.float64)

    dot_a_1 = float(np.dot(v_a, u1))  # 0.0
    dot_a_0 = float(np.dot(v_a, u0))  # 0.3
    log_p_a_step1 = -math.log(1.0 + math.exp(dot_a_1))  # bit 1
    log_p_a_step2 = -math.log(1.0 + math.exp(-dot_a_0))  # bit 0
    expected_log_p_a = log_p_a_step1 + log_p_a_step2
    expected_p_a = math.exp(expected_log_p_a)

    # For cand_b: v = [-0.5, 1.0]
    v_b = np.array([-0.5, 1.0], dtype=np.float64)
    dot_b_1 = float(np.dot(v_b, u1))  # 0.75
    dot_b_0 = float(np.dot(v_b, u0))  # -0.4
    log_p_b_step1 = -math.log(1.0 + math.exp(dot_b_1))  # bit 1 -> sigma(-0.75)
    log_p_b_step2 = -math.log(1.0 + math.exp(-dot_b_0))  # bit 0 -> sigma(-0.4)
    expected_log_p_b = log_p_b_step1 + log_p_b_step2
    expected_p_b = math.exp(expected_log_p_b)

    question = SentenceCompletionQuestion(
        context=(b"ctx_w1",),
        candidates=(b"cand_a", b"cand_b"),
        expected=b"cand_a",
    )

    # Evaluate HS log probability
    eval_log = evaluate_msr_sentence_completion(
        lookup,
        [question],
        scoring="hs_log_prob",
        output_embeddings=output_embeddings,
        huffman_offsets=offsets,
        huffman_paths=paths,
        huffman_bits=bits,
    )
    # expected_log_p_a vs expected_log_p_b
    assert expected_log_p_a > expected_log_p_b
    assert eval_log.predictions == (b"cand_a",)
    assert eval_log.overall.score == 1.0

    # Evaluate HS probability
    eval_prob = evaluate_msr_sentence_completion(
        lookup,
        [question],
        scoring="hs_prob",
        output_embeddings=output_embeddings,
        huffman_offsets=offsets,
        huffman_paths=paths,
        huffman_bits=bits,
    )
    assert eval_prob.predictions == (b"cand_a",)
    assert eval_prob.overall.score == 1.0


def test_hs_numerical_stability_extreme_values() -> None:
    """Verify log-sigmoid and probability calculations remain finite with extreme values."""
    lookup = DictLookup(
        {
            b"word_pos": (1000.0, 500.0),
            b"word_neg": (-1000.0, -500.0),
            b"word_zero": (0.0, 0.0),
            b"ctx": (1.0, 1.0),
        }
    )
    output_embeddings = np.array([[1000.0, 1000.0]], dtype=np.float32)
    offsets = np.array([0, 0, 0, 0, 1], dtype=np.uint64)
    paths = np.array([0], dtype=np.uint64)
    bits = np.array([0], dtype=np.uint8)

    question = SentenceCompletionQuestion(
        context=(b"ctx",),
        candidates=(b"word_pos", b"word_neg", b"word_zero"),
        expected=b"word_pos",
    )

    eval_log = evaluate_msr_sentence_completion(
        lookup,
        [question],
        scoring="hs_log_prob",
        output_embeddings=output_embeddings,
        huffman_offsets=offsets,
        huffman_paths=paths,
        huffman_bits=bits,
    )
    assert eval_log.overall.score == 1.0
    assert eval_log.predictions == (b"word_pos",)

    eval_prob = evaluate_msr_sentence_completion(
        lookup,
        [question],
        scoring="hs_prob",
        output_embeddings=output_embeddings,
        huffman_offsets=offsets,
        huffman_paths=paths,
        huffman_bits=bits,
    )
    assert eval_prob.overall.score == 1.0
    assert eval_prob.predictions == (b"word_pos",)


def test_oov_statistics_and_tie_breaking() -> None:
    """Verify comprehensive OOV metrics and tie tracking across sentence completion."""
    lookup = DictLookup(
        {
            b"cand1": (1.0, 0.0),
            b"cand2": (0.0, 1.0),
            b"ctx_known": (0.5, 0.5),
        }
    )

    questions = [
        # Q1: Normal question
        SentenceCompletionQuestion(
            context=(b"ctx_known",),
            candidates=(b"cand1", b"cand2"),
            expected=b"cand1",
        ),
        # Q2: 1 candidate OOV, 1 context OOV
        SentenceCompletionQuestion(
            context=(b"ctx_known", b"ctx_oov1"),
            candidates=(b"cand1", b"cand_oov"),
            expected=b"cand1",
        ),
        # Q3: All context words OOV (no valid context)
        SentenceCompletionQuestion(
            context=(b"ctx_oov1", b"ctx_oov2"),
            candidates=(b"cand1", b"cand2"),
            expected=b"cand1",
        ),
        # Q4: All candidates OOV
        SentenceCompletionQuestion(
            context=(b"ctx_known",),
            candidates=(b"oov_a", b"oov_b"),
            expected=b"oov_a",
        ),
    ]

    eval_res = evaluate_msr_sentence_completion(lookup, questions, scoring="cosine")
    stats = eval_res.oov_stats
    assert stats is not None
    assert stats.total_questions == 4
    assert stats.candidate_oov_questions == 2  # Q2 and Q4
    assert stats.candidate_all_oov_questions == 1  # Q4
    assert stats.context_oov_questions == 2  # Q2 and Q3
    assert stats.total_context_tokens == 6  # 1 + 2 + 2 + 1
    assert stats.oov_context_tokens == 3  # 0 + 1 + 2 + 0
    assert abs(stats.context_oov_token_ratio - 0.5) < 1e-6
    assert stats.no_valid_context_questions == 1  # Q3
    # Q3 has no valid context -> all scores 0.0 (tie). Q4 has all -inf (invalid/tie).
    assert stats.tie_or_invalid_score_questions >= 2


def test_table7_cache_protocol_separation(tmp_path: Path) -> None:
    """Verify cache protocols are segregated so cosine and HS results do not collide."""
    cache_cosine = Table7EvaluationCache(
        tmp_path / "cache_cosine",
        questions_sha256=MSR_QUESTIONS_SHA256,
        answers_sha256=MSR_ANSWERS_SHA256,
        scoring="cosine",
    )
    cache_hs = Table7EvaluationCache(
        tmp_path / "cache_hs",
        questions_sha256=MSR_QUESTIONS_SHA256,
        answers_sha256=MSR_ANSWERS_SHA256,
        scoring="hs_log_prob",
    )

    assert cache_cosine.protocol == "table7-v1"
    assert cache_hs.protocol == "table7-hs_log_prob-v1"
    assert cache_cosine.signature != cache_hs.signature

    res = Table7RunResult(
        architecture="skipgram",
        dimension=640,
        training_words_millions=50,
        epochs=1,
        corpus="holmes",
        seed=1,
        mlflow_run_id="run12345",
        overall_accuracy_percent=37.7,
        dev_accuracy_percent=38.0,
        test_accuracy_percent=37.4,
        total_questions=1040,
        observed_training_seconds=300.0,
        scoring="cosine",
    )
    cache_cosine.store(res, lookup_identity={"source": "test"})

    # Loading with hs cache on the same directory should fail signature check
    cache_hs_same_dir = Table7EvaluationCache(
        tmp_path / "cache_cosine",
        questions_sha256=MSR_QUESTIONS_SHA256,
        answers_sha256=MSR_ANSWERS_SHA256,
        scoring="hs_log_prob",
    )
    assert cache_hs_same_dir.load("run12345") is None

    # CSV seeding with cosine summary should not seed HS cache
    csv_path = tmp_path / "summary.csv"
    _write_table7_csv(csv_path, [res])
    seeded = cache_hs.seed_from_csv(csv_path)
    assert seeded == 0


def test_hs_evaluation_artifact_toy_session(tmp_path: Path) -> None:
    """Verify HSEvaluationArtifact loading and evaluation with an engine training session."""
    from w2v import (
        Corpus,
        Model,
        TrainingConfig,
        TrainingSession,
        Vocabulary,
        VocabularyConfig,
    )

    from f2.suites.w2v.artifacts import (
        load_hs_evaluation_artifact,
        save_checkpoint,
    )

    corpus_file = tmp_path / "toy_corpus.txt"
    corpus_file.write_bytes(
        b"the quick brown fox jumps over the lazy dog\n"
        b"sherlock holmes was an observant detective in london\n"
        b"the case was difficult but holmes had solved it\n"
    )
    corpus = Corpus(corpus_file)
    vocab = Vocabulary.build(corpus, VocabularyConfig(min_count=1, hash_capacity=100))
    cfg = TrainingConfig(
        model_kind="skip_gram",
        objective_kind="hierarchical_softmax",
        embedding_dimension=16,
        window_radius=2,
        epochs=1,
        thread_count=1,
        subsampling_threshold=0.0,
        negative_sample_count=0,
    )
    model = Model.create(vocab, cfg)
    session = TrainingSession(corpus, vocab, model, cfg)
    session.train_epoch()

    ckpt_dir = tmp_path / "checkpoint_full"
    save_checkpoint(session.export_state(), ckpt_dir, payload="full")

    # Load via HSEvaluationArtifact
    hs_artifact = load_hs_evaluation_artifact(ckpt_dir)
    assert hs_artifact.embeddings.shape == (vocab.size, 16)
    assert hs_artifact.output_embeddings.shape[1] == 16
    assert hs_artifact.huffman_offsets.shape == (vocab.size + 1,)

    questions = [
        SentenceCompletionQuestion(
            context=(b"sherlock", b"detective"),
            candidates=(b"holmes", b"dog", b"fox"),
            expected=b"holmes",
        )
    ]

    eval_cos = evaluate_msr_sentence_completion(
        hs_artifact, questions, scoring="cosine"
    )
    assert eval_cos.overall.total_count == 1

    eval_log = evaluate_msr_sentence_completion(
        hs_artifact, questions, scoring="hs_log_prob"
    )
    assert eval_log.overall.total_count == 1
    assert eval_log.predictions[0] in [b"holmes", b"dog", b"fox"]

    eval_prob = evaluate_msr_sentence_completion(
        hs_artifact, questions, scoring="hs_prob"
    )
    assert eval_prob.overall.total_count == 1
    assert eval_prob.predictions[0] in [b"holmes", b"dog", b"fox"]
