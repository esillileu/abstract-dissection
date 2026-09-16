"""Integration tests for corpus lifecycle, multi-hop lineage DAG, validation provenance, and RESTRICT deletion policies."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
import pytest

from f2.catalog.db.migrations.runner import run_catalog_migrations
from f2.corpus.db.migrations.runner import run_migrations
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection


@pytest.fixture
def db_conn(f2_test_database):
    url = f2_test_database
    with get_connection(url) as conn:
        run_catalog_migrations(conn)
        run_migrations(conn)
        yield conn


@pytest.fixture
def repo(db_conn: psycopg.Connection[Any]):
    with db_conn.transaction(force_rollback=True):
        yield CorpusStateRepository(db_conn)


def test_migrations_and_idempotency(repo: CorpusStateRepository):
    """Verify running migrations again is an idempotent no-op."""
    applied = run_migrations(repo.conn)
    assert applied == []


def test_end_to_end_multihop_lineage_and_traversal(repo: CorpusStateRepository):
    """Test full multi-hop lineage:
    catalog raw release -> acq -> raw -> extract -> norm -> filter -> shard -> canonical.
    Verify reverse and forward recursive CTE traversals.
    """
    conn = repo.conn
    uid = uuid.uuid4().hex[:8]
    raw_res_id = f"wmt_news_raw_{uid}"
    raw_ver_id = f"wmt_news_2012_{uid}"
    canonical_res_id = f"f2_news_corpus_{uid}"
    canonical_ver_id = f"f2_news_1b_v1_{uid}"

    # 1. Setup upstream catalog source & releases
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO catalog.resources (resource_id, kind, name, access_status, acquisition_status, readiness_status)
            VALUES (%s, 'dataset', 'WMT News Raw Archives', 'public', 'acquired', 'ready');
            """,
            (raw_res_id,),
        )
        cur.execute(
            """
            INSERT INTO catalog.resource_sources (resource_id, source_type, url, license, is_preferred)
            VALUES (%s, 'web', 'http://statmt.org/wmt12', 'Public Crawl', TRUE);
            """,
            (raw_res_id,),
        )
        cur.execute(
            """
            INSERT INTO catalog.resource_versions (resource_version_id, resource_id, version_label)
            VALUES (%s, %s, '2012_v1');
            """,
            (raw_ver_id, raw_res_id),
        )
        cur.execute(
            """
            INSERT INTO catalog.resources (resource_id, kind, name, access_status, acquisition_status, readiness_status)
            VALUES (%s, 'dataset', 'F2 Word2Vec Clean News Corpus', 'public', 'acquired', 'ready');
            """,
            (canonical_res_id,),
        )
        cur.execute(
            """
            INSERT INTO catalog.resource_versions (resource_version_id, resource_id, version_label)
            VALUES (%s, %s, '1.0.0');
            """,
            (canonical_ver_id, canonical_res_id),
        )

    # 2. Acquisition run producing raw archive artifact
    acq_run_id = f"acq_{uid}"
    raw_art_id = f"art_raw_{uid}"
    repo.create_acquisition_run(
        run_id=acq_run_id,
        resource_version_id=raw_ver_id,
        acquisition_method="http_archive_download",
        code_version="git_sha_acq_01",
        config_hash="cfg_hash_acq_01",
        config={"target_year": 2012},
    )
    repo.finish_acquisition_run(
        acq_run_id, status="completed", log_s3_uri=f"s3://logs/{acq_run_id}.log"
    )

    repo.register_artifact(
        artifact_id=raw_art_id,
        stage="raw",
        s3_uri=f"s3://corpus/raw/{raw_art_id}.warc.gz",
        sha256="sha256_raw_content",
        byte_size=10_000_000,
        format="warc.gz",
        record_count=5000,
        resource_version_id=raw_ver_id,
        acquisition_run_id=acq_run_id,
        integrity_status="verified",
    )

    # 3. Hop 1: Extract run (raw -> extracted text)
    proc_extract_id = f"proc_extract_{uid}"
    art_extract_id = f"art_extracted_{uid}"
    repo.create_processing_run(
        run_id=proc_extract_id,
        recipe_name="warc_text_extractor",
        recipe_version="1.0.0",
        code_version="git_sha_extract",
        config_hash="cfg_hash_extract",
    )
    repo.register_artifact(
        artifact_id=art_extract_id,
        stage="extracted",
        s3_uri=f"s3://corpus/extracted/{art_extract_id}.jsonl.gz",
        sha256="sha256_extract_content",
        byte_size=8_000_000,
        format="jsonl.gz",
        record_count=4800,
    )
    repo.record_processing_io(
        run_id=proc_extract_id,
        inputs=[(raw_art_id, "primary")],
        outputs=[(art_extract_id, "clean_extract")],
    )
    repo.finish_processing_run(proc_extract_id, status="completed")

    # 4. Hop 2: Normalize run (extracted -> normalized text)
    proc_norm_id = f"proc_norm_{uid}"
    art_norm_id = f"art_norm_{uid}"
    repo.create_processing_run(
        run_id=proc_norm_id,
        recipe_name="text_normalizer",
        recipe_version="1.1.0",
        code_version="git_sha_norm",
        config_hash="cfg_hash_norm",
    )
    repo.register_artifact(
        artifact_id=art_norm_id,
        stage="normalized",
        s3_uri=f"s3://corpus/normalized/{art_norm_id}.jsonl.gz",
        sha256="sha256_norm_content",
        byte_size=7_500_000,
        format="jsonl.gz",
        record_count=4800,
    )
    repo.record_processing_io(
        run_id=proc_norm_id,
        inputs=[(art_extract_id, "primary")],
        outputs=[(art_norm_id, "normalized")],
    )
    repo.finish_processing_run(proc_norm_id, status="completed")

    # 5. Hop 3: Filter & Dedup run (normalized -> filtered)
    proc_filter_id = f"proc_filter_{uid}"
    art_filter_id = f"art_filtered_{uid}"
    repo.create_processing_run(
        run_id=proc_filter_id,
        recipe_name="news_filter_dedup",
        recipe_version="2.0.0",
        code_version="git_sha_filter",
        config_hash="cfg_hash_filter",
    )
    repo.register_artifact(
        artifact_id=art_filter_id,
        stage="filtered",
        s3_uri=f"s3://corpus/filtered/{art_filter_id}.jsonl.gz",
        sha256="sha256_filter_content",
        byte_size=5_000_000,
        format="jsonl.gz",
        record_count=3200,
    )
    repo.record_processing_io(
        run_id=proc_filter_id,
        inputs=[{"artifact_id": art_norm_id, "role": "primary"}],
        outputs=[{"artifact_id": art_filter_id, "role": "clean_filtered"}],
    )
    repo.finish_processing_run(proc_filter_id, status="completed")

    # 6. Hop 4: Sharding run (filtered -> canonical shard)
    proc_shard_id = f"proc_shard_{uid}"
    art_shard_id = f"art_shard_01_{uid}"
    repo.create_processing_run(
        run_id=proc_shard_id,
        recipe_name="word2vec_sharder",
        recipe_version="1.0.0",
        code_version="git_sha_sharder",
        config_hash="cfg_hash_sharder",
    )
    repo.register_artifact(
        artifact_id=art_shard_id,
        stage="canonical_shard",
        s3_uri=f"s3://corpus/canonical/{canonical_ver_id}/shard_00001.txt",
        sha256="sha256_shard_content",
        byte_size=4_800_000,
        format="txt",
        record_count=3200,
        resource_version_id=canonical_ver_id,
    )
    repo.record_processing_io(
        run_id=proc_shard_id,
        inputs=[art_filter_id],
        outputs=[(art_shard_id, "canonical_shard")],
    )
    repo.finish_processing_run(proc_shard_id, status="completed")

    # 7. Register Shard and Stats in published corpus version
    repo.register_corpus_shard(
        resource_version_id=canonical_ver_id,
        shard_index=1,
        artifact_id=art_shard_id,
        word_count=800_000,
        doc_count=3200,
        byte_size=4_800_000,
    )
    repo.upsert_corpus_version_stats(
        resource_version_id=canonical_ver_id,
        total_words=800_000,
        total_tokens=850_000,
        total_documents=3200,
        total_sentences=45_000,
        total_bytes=4_800_000,
        total_shards=1,
        source_composition={"wmt_2012": 1.0},
        year_distribution={"2012": 1.0},
    )

    # -------------------------------------------------------------------------
    # Verification: Reverse Lineage Traversal
    # -------------------------------------------------------------------------
    rev_lineage = repo.get_reverse_lineage(canonical_ver_id, shard_index=1)
    assert len(rev_lineage) == 5, (
        f"Expected 5 hops in DAG traversal, got {len(rev_lineage)}"
    )

    # Order: depth 4 (raw archive) -> depth 0 (canonical shard)
    stages = [row["current_stage"] for row in rev_lineage]
    assert stages[0] == "raw"
    assert stages[1] == "extracted"
    assert stages[2] == "normalized"
    assert stages[3] == "filtered"
    assert stages[4] == "canonical_shard"

    raw_hop = rev_lineage[0]
    assert raw_hop["current_artifact_id"] == raw_art_id
    assert raw_hop["origin_source_name"] == "WMT News Raw Archives"
    assert raw_hop["origin_release_label"] == "2012_v1"
    assert raw_hop["origin_license"] == "Public Crawl"
    assert raw_hop["origin_acquisition_run"] == acq_run_id
    assert raw_hop["acquisition_method"] == "http_archive_download"

    # -------------------------------------------------------------------------
    # Verification: Forward Lineage Traversal
    # -------------------------------------------------------------------------
    fwd_lineage = repo.get_forward_lineage(raw_ver_id)
    assert len(fwd_lineage) >= 4
    terminal = [
        row
        for row in fwd_lineage
        if row["canonical_corpus_version"] == canonical_ver_id
    ]
    assert len(terminal) == 1
    assert terminal[0]["shard_index"] == 1
    assert terminal[0]["shard_words"] == 800_000


def test_provenance_restrict_deletion_policy(repo: CorpusStateRepository):
    """Verify that deleting records involved in lineage raises RESTRICT foreign key violations."""
    conn = repo.conn
    uid = uuid.uuid4().hex[:8]
    res_id = f"res_{uid}"
    ver_id = f"ver_{uid}"
    acq_id = f"acq_{uid}"
    art_id = f"art_{uid}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO catalog.resources (resource_id, kind, name, access_status, acquisition_status, readiness_status)
            VALUES (%s, 'dataset', 'Res', 'public', 'acquired', 'ready');
            """,
            (res_id,),
        )
        cur.execute(
            "INSERT INTO catalog.resource_versions (resource_version_id, resource_id, version_label) VALUES (%s, %s, 'v1');",
            (ver_id, res_id),
        )

    repo.create_acquisition_run(
        run_id=acq_id,
        resource_version_id=ver_id,
        acquisition_method="manual",
        code_version="commit_1",
        config_hash="hash_1",
    )
    repo.register_artifact(
        artifact_id=art_id,
        stage="raw",
        s3_uri=f"s3://test/{art_id}",
        sha256="sha",
        byte_size=100,
        format="txt",
        resource_version_id=ver_id,
        acquisition_run_id=acq_id,
    )

    # Attempting to delete resource_version while referenced by artifact must fail with RESTRICT
    with pytest.raises(psycopg.errors.RestrictViolation):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM catalog.resource_versions WHERE resource_version_id = %s;",
                    (ver_id,),
                )

    # Attempting to delete acquisition_run while referenced by artifact must fail with RESTRICT
    with pytest.raises(psycopg.errors.RestrictViolation):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM acquisition_runs WHERE run_id = %s;", (acq_id,)
                )


def test_validation_referential_integrity_and_checks(repo: CorpusStateRepository):
    """Test validation_runs mutually exclusive FKs, check constraints, and generated target_id."""
    conn = repo.conn
    uid = uuid.uuid4().hex[:8]
    paper_id = f"paper_{uid}"
    prof_key = f"word2vec_eval_{uid}"
    prof_id = f"{prof_key}_r1"

    # Setup target paper in catalog
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO catalog.papers (paper_id, title) VALUES (%s, 'Efficient Estimation');",
            (paper_id,),
        )

    # Create immutable validation profile
    repo.create_validation_profile(
        profile_id=prof_id,
        profile_key=prof_key,
        revision=1,
        name="Word2Vec News 1B Quality Profile",
        spec_hash="hash_profile_spec",
        specification={"max_rejection_rate": 0.45, "min_words": 100_000},
        target_paper_id=paper_id,
    )

    # 1. Validate processing run
    proc_id = f"proc_val_{uid}"
    repo.create_processing_run(
        run_id=proc_id,
        recipe_name="extractor",
        recipe_version="1.0",
        code_version="sha",
        config_hash="hash",
    )
    val_run_1 = f"val_proc_{uid}"
    repo.record_validation_run(
        validation_run_id=val_run_1,
        profile_id=prof_id,
        target_type="processing_run",
        target_id=proc_id,
        validator_code_version="vcode_sha",
        validator_config_hash="vcfg_hash",
        overall_verdict="PASS",
        checks=[
            {
                "check_name": "rejection_rate",
                "category": "transformation",
                "status": "PASS",
                "expected_condition": "<= 0.45",
                "observed_value": "0.38",
            }
        ],
    )

    run_data = repo.get_validation_run(val_run_1)
    assert run_data is not None
    assert run_data["target_id"] == proc_id
    assert run_data["target_processing_run_id"] == proc_id
    assert run_data["target_artifact_id"] is None
    assert run_data["target_resource_version_id"] is None
    assert len(run_data["checks"]) == 1
    assert run_data["checks"][0]["status"] == "PASS"

    # 2. Reject nonexistent target ID (FK violation) wrapped in sub-transaction
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            repo.record_validation_run(
                validation_run_id=f"val_invalid_{uid}",
                profile_id=prof_id,
                target_type="processing_run",
                target_id="nonexistent_processing_run_id",
                validator_code_version="vcode",
                validator_config_hash="vcfg",
                overall_verdict="FAIL",
                checks=[],
            )

    # 3. Reject target_type and target column mismatch (CHECK violation) wrapped in sub-transaction
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO validation_runs (
                        validation_run_id, profile_id, target_type,
                        target_processing_run_id, validator_code_version,
                        validator_config_hash, overall_verdict
                    ) VALUES (%s, %s, 'artifact', %s, 'vcode', 'vcfg', 'PASS');
                    """,
                    (f"val_mismatch_{uid}", prof_id, proc_id),
                )


def test_validation_profile_immutability(repo: CorpusStateRepository):
    """Test that validation_profiles triggers prohibit updating spec, target_paper_id, or revision."""
    conn = repo.conn
    uid = uuid.uuid4().hex[:8]
    paper_1 = f"paper_1_{uid}"
    paper_2 = f"paper_2_{uid}"
    prof_key = f"profile_immut_{uid}"
    prof_id = f"{prof_key}_r1"

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO catalog.papers (paper_id, title) VALUES (%s, 'P1'), (%s, 'P2');",
            (paper_1, paper_2),
        )

    repo.create_validation_profile(
        profile_id=prof_id,
        profile_key=prof_key,
        revision=1,
        name="Eval Profile",
        spec_hash="hash_v1",
        specification={"strict": True},
        target_paper_id=paper_1,
    )

    # Updating specification must fail via trigger
    with pytest.raises(
        psycopg.Error, match="Validation profile specification is immutable"
    ):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE validation_profiles SET specification = '{\"strict\": false}'::jsonb WHERE profile_id = %s;",
                    (prof_id,),
                )

    # Updating target_paper_id must fail via trigger
    with pytest.raises(
        psycopg.Error, match="Validation profile specification is immutable"
    ):
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE validation_profiles SET target_paper_id = %s WHERE profile_id = %s;",
                    (paper_2, prof_id),
                )

    # Deprecating profile (updating is_active, notes) must succeed
    repo.deprecate_validation_profile(prof_id, notes="Superseded by r2")
    prof = repo.get_validation_profile(prof_id)
    assert prof is not None
    assert prof["is_active"] is False
    assert prof["superseded_at"] is not None

    # Creating revision 2 of the same key must succeed
    prof_id_r2 = f"{prof_key}_r2"
    repo.create_validation_profile(
        profile_id=prof_id_r2,
        profile_key=prof_key,
        revision=2,
        name="Eval Profile Rev 2",
        spec_hash="hash_v2",
        specification={"strict": False},
        target_paper_id=paper_2,
    )
    prof2 = repo.get_validation_profile(prof_id_r2)
    assert prof2 is not None
    assert prof2["revision"] == 2


def test_pipeline_run_lineage_retry_and_duplicate_protection(
    repo: CorpusStateRepository,
):
    """Test Common Crawl bridge relation: retry linkage succeeds, duplicate link fails."""
    conn = repo.conn
    uid = uuid.uuid4().hex[:8]
    pipe_run_id = f"pipe_run_{uid}"
    acq_1 = f"acq_1_{uid}"
    acq_2 = f"acq_2_{uid}"
    res_id = f"res_{uid}"
    ver_id = f"ver_{uid}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO catalog.resources (resource_id, kind, name, access_status, acquisition_status, readiness_status)
            VALUES (%s, 'dataset', 'Res', 'public', 'acquired', 'ready');
            """,
            (res_id,),
        )
        cur.execute(
            "INSERT INTO catalog.resource_versions (resource_version_id, resource_id, version_label) VALUES (%s, %s, 'v1');",
            (ver_id, res_id),
        )
        cur.execute(
            """
            INSERT INTO pipeline_runs (run_id, run_type, crawl_ids, sample_size, seed, bandwidth_mbps, concurrency, output_dir)
            VALUES (%s, 'cc_eval', ARRAY['CC-MAIN-2012'], 100, 42, 10.0, 1, '/tmp');
            """,
            (pipe_run_id,),
        )

    repo.create_acquisition_run(acq_1, ver_id, "cdx_range_fetch", "sha", "hash")
    repo.create_acquisition_run(acq_2, ver_id, "cdx_range_fetch", "sha", "hash")

    # Link attempt 1
    lid1 = repo.link_pipeline_run_lineage(
        pipeline_run_id=pipe_run_id,
        stage_role="arc_range_fetch",
        acquisition_run_id=acq_1,
        execution_attempt=1,
    )
    assert lid1 > 0

    # Link attempt 2 (retry)
    lid2 = repo.link_pipeline_run_lineage(
        pipeline_run_id=pipe_run_id,
        stage_role="arc_range_fetch",
        acquisition_run_id=acq_2,
        execution_attempt=2,
    )
    assert lid2 > lid1

    lineage = repo.get_pipeline_run_lineage(pipe_run_id)
    assert len(lineage) == 2
    assert lineage[0]["execution_attempt"] == 1
    assert lineage[1]["execution_attempt"] == 2

    # Duplicate link of attempt 1 must raise UniqueViolation (wrapped in sub-transaction)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction():
            repo.link_pipeline_run_lineage(
                pipeline_run_id=pipe_run_id,
                stage_role="arc_range_fetch",
                acquisition_run_id=acq_1,
                execution_attempt=1,
            )
