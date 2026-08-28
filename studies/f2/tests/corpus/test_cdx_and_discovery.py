"""Tests for CDX cluster.idx indexing, SURT mapping, two-stage sampling, and sequential audit sampling."""

from __future__ import annotations

from f2.corpus.cdx import (
    CDXIndexReader,
    CDXRecord,
    domain_to_surt_prefix,
    url_to_surt,
)
from f2.corpus.discovery import (
    CandidateRecord,
    SequentialAuditSampler,
    TwoStageProbabilitySampler,
    is_news_path_heuristic,
)


def test_surt_transformations():
    assert domain_to_surt_prefix("reuters.com") == "com,reuters)"
    assert domain_to_surt_prefix("www.bbc.co.uk") == "uk,co,bbc)"
    assert (
        url_to_surt("http://www.nytimes.com/2012/03/15/world.html")
        == "com,nytimes)/2012/03/15/world.html"
    )


def test_cdx_index_reader(sample_cluster_idx_text: str):
    reader = CDXIndexReader.from_text(sample_cluster_idx_text)
    assert reader.total_blocks() == 5

    # Binary search lookup
    idx = reader.find_block_index_for_surt("com,reuters)")
    assert idx == 3
    assert reader.entries[idx].surt_key == "com,reuters)/"

    blocks = reader.find_blocks_for_prefix("com,nytimes)")
    assert len(blocks) >= 1
    assert blocks[0].surt_key == "com,nytimes)/"


def test_cdx_record_parsing():
    line = 'com,reuters)/article 20120315 {"status":"200","url":"http://www.reuters.com/article","filename":"seg/1.arc.gz","offset":"1000","length":"500","mime":"text/html","digest":"ABC"}'
    rec = CDXRecord.from_cdx_line(line)
    assert rec is not None
    assert rec.url == "http://www.reuters.com/article"
    assert rec.status == "200"
    assert rec.offset == 1000
    assert rec.length == 500


def test_two_stage_probability_sampler_inclusion_probabilities(
    sample_cluster_idx_text: str,
):
    reader = CDXIndexReader.from_text(sample_cluster_idx_text)
    sampler = TwoStageProbabilitySampler(
        crawl_id="CC-MAIN-2012", index_reader=reader, seed=42
    )

    # Stage 1: Select 2 blocks out of 5 (p_k = 2 / 5 = 0.4)
    blocks = sampler.plan_stage1_blocks(num_blocks=2)
    assert len(blocks) == 2

    # Mock block with 10 records
    mock_records = [
        CDXRecord(
            url=f"http://test.com/{i}",
            timestamp="20120101",
            status="200",
            mime="text/html",
            digest=f"D{i}",
            filename="seg.arc.gz",
            offset=i * 100,
            length=100,
        )
        for i in range(10)
    ]

    # Stage 2: Sample 4 records out of 10 in block 0 (pi_within = 4 / 10 = 0.4)
    sampled = sampler.sample_block_records(
        blocks[0], mock_records, num_records_per_block=4
    )
    assert len(sampled) == 4

    # Finalize probabilities: pi_ki = (2/5) * (4/10) = 0.16 -> weight = 1/0.16 = 6.25
    finalized = sampler.finalize_inclusion_probabilities(
        sampled,
        num_selected_blocks=2,
        total_crawl_blocks=5,
    )
    for cand in finalized:
        assert abs(cand.inclusion_probability - 0.16) < 1e-6
        assert abs(cand.design_weight - 6.25) < 1e-4


def test_sequential_audit_sampler_permutation():
    sampler = SequentialAuditSampler(seed=1337)
    mock_cands = [
        CandidateRecord(
            crawl_id="CC-MAIN-2012",
            url=f"http://test.com/{i}",
            timestamp="20120101",
            filename="a.arc.gz",
            offset=i,
            length=10,
            digest=f"D{i}",
            source_type="prob",
            stratum="unbiased",
            inclusion_probability=0.1,
            design_weight=10.0,
            block_index=0,
            record_index_in_block=i,
            block_total_records=10,
        )
        for i in range(10)
    ]
    predicted_classes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    schedule = sampler.generate_audit_schedule(mock_cands, predicted_classes)
    assert len(schedule) == 10

    # Test Wave 1 selection: 2 per stratum
    wave1 = sampler.select_audit_wave(schedule, num_per_stratum={1: 2, 0: 2})
    assert len(wave1) == 4
    for w in wave1:
        assert w["audit_inclusion_prob_cond"] == 0.4  # 2 out of 5


def test_news_path_heuristics():
    assert (
        is_news_path_heuristic("http://reuters.com/article/2012/03/15/us-economy.html")
        is True
    )
    assert (
        is_news_path_heuristic(
            "http://nytimes.com/2012/03/15/business/market-rally.html"
        )
        is True
    )
    assert is_news_path_heuristic("http://example.com/about-us.html") is False
    assert (
        is_news_path_heuristic("http://store.example.com/products/view.php?id=123")
        is False
    )
