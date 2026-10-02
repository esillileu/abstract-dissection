"""Unit and integration tests for FineWeb 2013 source adapter and pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pytest

from f2.corpus.fineweb.adapter import (
    FINEWEB_DATASET,
    FINEWEB_PINNED_REVISION,
    FineWebSourceAdapter,
)
from f2.corpus.fineweb.classifier import FineWebNewsClassifier
from f2.corpus.fineweb.pipeline import FineWebPipeline
from f2.corpus.fineweb.smoke import run_fineweb_smoke
from f2.corpus.fineweb.writer import FineWebShardWriter


def _create_mock_table() -> pa.Table:
    """Create an in-memory PyArrow table simulating FineWeb parquet records."""
    data = [
        {
            "id": "<urn:uuid:00000000-0000-0000-0000-000000000001>",
            "text": (
                "WASHINGTON — Jan 15, 2013 — By John Doe and Jane Smith. "
                "The Federal Reserve announced on Tuesday that it would maintain interest rates. "
                '"We are seeing steady economic progress," the chairman said today. '
                "Analysts reported that inflation remains subdued across all sectors."
            ),
            "dump": "CC-MAIN-2013-20",
            "url": "https://www.reuters.com/article/fed-rates-2013",
            "date": "2013-05-18T05:00:00Z",
            "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/warc.gz",
            "language": "en",
            "language_score": 0.99,
            "token_count": 60,
        },
        {
            "id": "<urn:uuid:00000000-0000-0000-0000-000000000002>",
            "text": (
                "Welcome to the forum! Joined: 2011. Posts: 42. "
                "Quote reply: Buy now! Add to cart with free shipping! "
                "Product details: price: $99.99 in stock now."
            ),
            "dump": "CC-MAIN-2013-20",
            "url": "https://shop.example.com/item",
            "date": "2013-05-18T06:00:00Z",
            "file_path": None,
            "language": "en",
            "language_score": 0.95,
            "token_count": 35,
        },
        {
            "id": "<urn:uuid:00000000-0000-0000-0000-000000000003>",
            "text": "Too short text to qualify as an article.",
            "dump": "CC-MAIN-2013-20",
            "url": "https://example.com/short",
            "date": "2013-05-18T07:00:00Z",
            "file_path": None,
            "language": "en",
            "language_score": 0.90,
            "token_count": 8,
        },
    ]
    return pa.Table.from_pylist(data)


def test_adapter_metadata_and_url_resolution() -> None:
    adapter = FineWebSourceAdapter()
    assert adapter.dataset == FINEWEB_DATASET
    assert adapter.revision == FINEWEB_PINNED_REVISION

    files_20 = adapter.list_parquet_files("CC-MAIN-2013-20")
    assert len(files_20) == 205
    assert files_20[0] == "data/CC-MAIN-2013-20/000_00000.parquet"
    assert files_20[-1] == "data/CC-MAIN-2013-20/004_00004.parquet"

    url = adapter.resolve_url(files_20[0])
    assert url.startswith(
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/"
    )
    assert FINEWEB_PINNED_REVISION in url


def test_adapter_iter_arrow_table() -> None:
    adapter = FineWebSourceAdapter()
    table = _create_mock_table()
    docs = list(
        adapter.iter_arrow_table(table, parquet_path="data/test/000_00000.parquet")
    )

    assert len(docs) == 3
    assert docs[0].fineweb_id == "<urn:uuid:00000000-0000-0000-0000-000000000001>"
    assert docs[0].dump == "CC-MAIN-2013-20"
    assert "000_00000:0000000" in docs[0].canonical_id
    assert docs[0].language == "en"


def test_news_classifier_cues() -> None:
    table = _create_mock_table()
    news_text = table.column("text")[0].as_py()
    score, is_news, details = FineWebNewsClassifier.evaluate(news_text)
    assert is_news
    assert score >= 1.5
    assert details["has_dateline"]
    assert details["has_byline"]

    spam_text = table.column("text")[1].as_py()
    s_score, s_is_news, _ = FineWebNewsClassifier.evaluate(spam_text)
    assert not s_is_news
    assert s_score < 0.0


def test_pipeline_and_writer_end_to_end(tmp_path: Path) -> None:
    table = _create_mock_table()
    adapter = FineWebSourceAdapter()
    docs = adapter.iter_arrow_table(table)
    pipeline = FineWebPipeline(min_words=10)
    writer = FineWebShardWriter(tmp_path, target_words=100)

    for news_rec, prov_rec in pipeline.process_stream(docs):
        writer.append(news_rec, prov_rec)
    shards = writer.close()

    assert pipeline.stats.documents_seen == 3
    assert pipeline.stats.documents_accepted == 1
    assert len(shards) == 1

    # Check shard exists and is valid zstd
    shard_file = tmp_path / shards[0].path
    assert shard_file.is_file()
    out = subprocess.check_output(["zstd", "-dc", "-q", str(shard_file)], text=True)
    assert "announced on tuesday" in out
    # Mikolov digits stripped to space
    assert "2013" not in out

    # Check manifest
    manifest_data = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == 2
    assert manifest_data["summary"]["total_records"] == 1
    assert manifest_data["corpus_identity"]["dataset"] == FINEWEB_DATASET

    # Check provenance
    prov_lines = writer.provenance_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(prov_lines) == 1
    prov = json.loads(prov_lines[0])
    assert prov["fineweb_id"] == "<urn:uuid:00000000-0000-0000-0000-000000000001>"


def test_deterministic_reproduction(tmp_path: Path) -> None:
    table = _create_mock_table()
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    res1 = run_fineweb_smoke(dir1, arrow_table=table, min_words=10)
    res2 = run_fineweb_smoke(dir2, arrow_table=table, min_words=10)

    assert res1["documents_accepted"] == res2["documents_accepted"]
    assert res1["total_accepted_words"] == res2["total_accepted_words"]

    # Manifest and shard bytes must match byte-for-byte
    m1 = (dir1 / "manifest.json").read_bytes()
    m2 = (dir2 / "manifest.json").read_bytes()
    assert m1 == m2

    p1 = (dir1 / "provenance.jsonl").read_bytes()
    p2 = (dir2 / "provenance.jsonl").read_bytes()
    assert p1 == p2

    s1 = (dir1 / res1["shards"][0].path).read_bytes()
    s2 = (dir2 / res2["shards"][0].path).read_bytes()
    assert s1 == s2


@pytest.mark.network
def test_remote_parquet_smoke(tmp_path: Path) -> None:
    res = run_fineweb_smoke(
        tmp_path / "remote",
        sample_size=10,
        min_words=50,
        remote=True,
    )
    assert res["documents_seen"] == 10
    assert (tmp_path / "remote" / "manifest.json").is_file()


def test_production_spec_invariants() -> None:
    from f2.corpus.fineweb.spec import PRODUCTION_SPEC

    assert PRODUCTION_SPEC.source.dataset == "HuggingFaceFW/fineweb"
    assert PRODUCTION_SPEC.source.primary_dump == "CC-MAIN-2013-20"
    assert PRODUCTION_SPEC.selection.target_33b_words == 33_000_000_000
    assert PRODUCTION_SPEC.selection.target_6b_words == 6_000_000_000
    assert PRODUCTION_SPEC.spec_hash is not None
    assert len(PRODUCTION_SPEC.spec_hash) == 64


def test_materializer_checkpoint_lifecycle(tmp_path: Path) -> None:
    from f2.corpus.fineweb.materializer import (
        FineWebProductionMaterializer,
    )

    mat = FineWebProductionMaterializer(tmp_path, target_words=100_000)
    cp = mat.load_checkpoint()
    assert cp.cumulative_words == 0
    assert cp.last_completed_file_index == -1

    cp.cumulative_words = 50_000
    cp.last_completed_file_index = 2
    mat.save_checkpoint(cp)

    reloaded = mat.load_checkpoint()
    assert reloaded.cumulative_words == 50_000
    assert reloaded.last_completed_file_index == 2


def test_w2v_trainer_consumes_fineweb_corpus(tmp_path: Path) -> None:
    from w2v import (
        Corpus,
        Model,
        TrainingConfig,
        TrainingSession,
        Vocabulary,
        VocabularyConfig,
    )

    table = _create_mock_table()
    res = run_fineweb_smoke(tmp_path / "smoke", arrow_table=table, min_words=10)
    shard_path = tmp_path / "smoke" / res["shards"][0].path

    # Decompress shard to raw text
    raw_text_path = tmp_path / "plain_corpus.txt"
    subprocess.run(
        ["zstd", "-dc", "-q", "-f", str(shard_path), "-o", str(raw_text_path)],
        check=True,
    )

    corpus = Corpus(raw_text_path)
    assert corpus.byte_size > 0
    vocab = Vocabulary.build(
        corpus,
        VocabularyConfig(initial_capacity=50, hash_capacity=500, min_count=1),
    )
    assert vocab.size > 0

    config = TrainingConfig(
        model_kind="cbow",
        embedding_dimension=8,
        window_radius=2,
        epochs=1,
        thread_count=1,
        subsampling_threshold=0.0,
        negative_sample_count=2,
        negative_table_size=100,
        sigmoid_table_size=100,
        observation_interval=0,
    )
    model = Model.create(vocab, config)
    session = TrainingSession(corpus, vocab, model, config)
    report = session.train_epoch()
    assert report.epoch == 1
    assert report.processed_tokens > 0


def test_load_and_validate_checkpoint(tmp_path: Path) -> None:
    from f2.corpus.fineweb.register import load_and_validate_checkpoint

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard0 = shards_dir / "shard-00000.txt.zst"
    shard0.write_bytes(b"shard0-content-12345")

    checkpoint_data = {
        "completed_shards": [
            {
                "index": 0,
                "compressed_bytes": len(b"shard0-content-12345"),
                "physical_sha256": "dummy-hash",
                "word_count": 100,
                "record_count": 10,
            }
        ]
    }
    checkpoint_file = tmp_path / "checkpoint.json"
    checkpoint_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

    shards = load_and_validate_checkpoint(checkpoint_file, shards_dir, max_shards=1)
    assert len(shards) == 1
    assert shards[0]["index"] == 0
    assert shards[0]["word_count"] == 100

    shards_all = load_and_validate_checkpoint(
        checkpoint_file, shards_dir, max_shards=None
    )
    assert len(shards_all) == 1
