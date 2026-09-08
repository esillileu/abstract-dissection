"""Tests for acquisition lifecycle: resume, failure injection, and idempotency."""

from __future__ import annotations

import io
import tarfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from repro_io.checksum import sha256_file
from repro_io.http.download import SerialDownloader

from f2.catalog.db.migrations.runner import run_catalog_migrations
from f2.corpus.db.migrations.runner import run_migrations
from f2.corpus.db.repository import CorpusStateRepository
from f2.corpus.db.session import get_connection
from f2.corpus.lifecycle import (
    acquire_source,
    catalog_sources,
    install_validation_profiles,
)
from f2.corpus.sources import SOURCE_BY_KEY, CorpusSource, SourceFile, stable_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_umbc_tar(path: Path) -> None:
    """Create a minimal UMBC-shaped .tgz archive (no sha256 check required for umbc)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for member_path, content in [
            ("webbase_all/a.txt", b"The quick brown fox.\nJumped over the lazy dog.\n"),
            ("webbase_all_tagged/a.txt", b"word_NN another_NN\n"),
        ]:
            data = content
            info = tarfile.TarInfo(name=member_path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    path.write_bytes(buf.getvalue())


def _umbc_source(tmp_path: Path) -> tuple[Path, CorpusSource]:
    """Create a unique UMBC test archive + CorpusSource in *tmp_path*.

    A UUID suffix is embedded in both the release label and the archive
    filename so that the derived S3 key is always unique across test runs and
    never collides with artifacts already registered in the shared test DB.
    Uses 'umbc' key so the lifecycle.py sha256 guard does not apply.
    """
    uid = uuid.uuid4().hex[:12]
    archive = tmp_path / f"umbc_webbase_corpus_{uid}.tgz"
    _make_umbc_tar(archive)
    release = f"test-{uid}"
    source = CorpusSource(
        key="umbc",
        release=release,
        name=f"Test UMBC {release}",
        homepage="http://localhost/",
        license="CC BY 3.0",
        access="public",
        files=(
            SourceFile(
                name=archive.name,
                url=archive.as_uri(),
                sha256=None,  # umbc does not require a hash override
            ),
        ),
        raw_resource_version_id=f"f2-umbc-test-{uid}-raw",
        canonical_resource_version_id=f"f2-umbc-test-{uid}-canonical-v1",
        normalized_resource_version_id=f"f2-umbc-test-{uid}-normalized-v1",
    )
    return archive, source


def _register_test_source(conn, source: CorpusSource) -> None:
    with conn.cursor() as cur:
        resource_id = f"f2-{source.key}-test-{source.release}"
        cur.execute(
            """
            INSERT INTO catalog.resources
            (resource_id, kind, name, description, access_status, acquisition_status, readiness_status)
            VALUES (%s, 'dataset', %s, 'Test resource', 'public', 'available', 'ready')
            ON CONFLICT (resource_id) DO NOTHING;
            """,
            (resource_id, source.name),
        )
        cur.execute(
            """
            INSERT INTO catalog.resource_versions (resource_version_id, resource_id, version_label)
            VALUES (%s, %s, %s)
            ON CONFLICT (resource_version_id) DO NOTHING;
            """,
            (source.raw_resource_version_id, resource_id, source.release),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Unit tests (no DB required)
# ---------------------------------------------------------------------------


def test_serial_downloader_checksum_mismatch_raises(tmp_path: Path) -> None:
    """SerialDownloader must raise ValueError when SHA-256 does not match expected."""
    server_content = b"real content"
    expected_wrong = "a" * 64  # wrong digest

    with patch("urllib.request.urlopen") as mock_open:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.side_effect = [server_content, b""]
        mock_open.return_value = mock_resp

        dest = tmp_path / "dl" / "file.tar.gz"
        downloader = SerialDownloader(retries=0)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            downloader.download(
                "http://example.com/file.tar.gz",
                dest,
                expected_sha256=expected_wrong,
            )
        # Destination must NOT be finalized on checksum failure (staging preserved)
        assert not dest.exists(), "destination should not be finalized on mismatch"
        # The .part file should remain for retry continuity
        assert (tmp_path / "dl" / "file.tar.gz.part").exists()


def test_serial_downloader_range_header_sent_when_partial_exists(
    tmp_path: Path,
) -> None:
    """Downloader must include a Range header when a .part file already exists."""
    first_chunk = b"Hello "

    dest = tmp_path / "file.tar.gz"
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(first_chunk)

    captured_headers: list[dict] = []

    def capturing_urlopen(req, **kwargs):
        captured_headers.append(dict(req.headers))
        # Return a response that ends immediately (simulates empty server response)
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.status = 206
        resp.read.side_effect = [b""]  # no data → triggers length check
        return resp

    with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
        downloader = SerialDownloader(retries=0)
        try:
            downloader.download(
                "http://example.com/file.tar.gz", dest, expected_sha256=None
            )
        except (ValueError, OSError):
            pass  # expected - empty response won't match any sha

    assert len(captured_headers) == 1
    # urllib capitalizes first letter of each word in header names
    range_val = captured_headers[0].get("Range") or captured_headers[0].get("range")
    assert range_val == f"bytes={len(first_chunk)}-"


# ---------------------------------------------------------------------------
# Integration tests (require F2_DATABASE_URL)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_conn(f2_test_database):
    url = f2_test_database
    with get_connection(url) as conn:
        run_catalog_migrations(conn)
        run_migrations(conn)
        yield conn


def test_acquire_source_is_idempotent_on_completed_run(db_conn, tmp_path: Path) -> None:
    """acquire_source returns early without re-downloading when run is already completed."""
    repo = CorpusStateRepository(db_conn)
    catalog_sources(db_conn)
    install_validation_profiles(repo)

    archive, source = _umbc_source(tmp_path)
    _register_test_source(db_conn, source)
    real_digest = sha256_file(archive)

    mock_store = MagicMock()
    mock_store.uri.side_effect = lambda key: f"s3://test-bucket/{key}"
    mock_store.put_file.return_value = None
    mock_store.sha256.return_value = real_digest  # remote SHA matches

    staging = tmp_path / "staging"

    # First acquisition
    result1 = acquire_source(source, staging, mock_store, repo)
    assert result1 != ["already-completed"]
    assert mock_store.put_file.call_count == 1

    # Second acquisition on same run ID must short-circuit without re-uploading
    result2 = acquire_source(source, staging, mock_store, repo)
    assert result2 == ["already-completed"]
    assert mock_store.put_file.call_count == 1  # unchanged


def test_acquire_source_s3_checksum_mismatch_leaves_status_failed(
    db_conn, tmp_path: Path
) -> None:
    """If the remote SHA-256 diverges from local, acquisition run must end as 'failed'."""
    repo = CorpusStateRepository(db_conn)
    catalog_sources(db_conn)

    archive, source = _umbc_source(tmp_path)
    _register_test_source(db_conn, source)
    wrong_remote_digest = "b" * 64  # remote returns a different hash

    mock_store = MagicMock()
    mock_store.uri.side_effect = lambda key: f"s3://test-bucket/{key}"
    mock_store.put_file.return_value = None
    mock_store.sha256.return_value = wrong_remote_digest  # mismatch

    staging = tmp_path / "staging"

    with pytest.raises(OSError, match="post-upload SHA-256 mismatch"):
        acquire_source(source, staging, mock_store, repo)

    # Verify the acquisition run is recorded as 'failed', not 'completed'
    run_id = "acq-" + stable_id(source.key, source.release, source.config_hash)
    run_record = repo.get_acquisition_run(run_id)
    assert run_record is not None
    assert run_record["status"] == "failed"
    assert run_record["error_message"] is not None


def test_acquire_source_db_commit_failure_leaves_status_failed(
    db_conn, tmp_path: Path
) -> None:
    """If the DB artifact registration fails, the acquisition run is recorded as 'failed'."""
    repo = CorpusStateRepository(db_conn)
    catalog_sources(db_conn)

    archive, source = _umbc_source(tmp_path)
    _register_test_source(db_conn, source)
    real_digest = sha256_file(archive)

    mock_store = MagicMock()
    mock_store.uri.side_effect = lambda key: f"s3://test-bucket/{key}"
    mock_store.put_file.return_value = None
    mock_store.sha256.return_value = real_digest

    staging = tmp_path / "staging"

    # Patch register_artifact to simulate a DB failure mid-transaction
    def failing_register(*args, **kwargs) -> None:
        raise RuntimeError("simulated DB failure during artifact registration")

    with patch.object(repo, "register_artifact", side_effect=failing_register):
        with pytest.raises(RuntimeError, match="simulated DB failure"):
            acquire_source(source, staging, mock_store, repo)

    # Acquisition run must be recorded as 'failed', not 'completed'
    run_id = "acq-" + stable_id(source.key, source.release, source.config_hash)
    run_record = repo.get_acquisition_run(run_id)
    assert run_record is not None
    assert run_record["status"] == "failed"


def test_blocked_source_raises_permission_error_without_creating_run(
    db_conn, tmp_path: Path
) -> None:
    """acquire_source must raise PermissionError immediately without creating an acquisition run."""
    repo = CorpusStateRepository(db_conn)
    catalog_sources(db_conn)

    gigaword = SOURCE_BY_KEY["gigaword"]
    mock_store = MagicMock()

    with pytest.raises(PermissionError, match="authorized LDC access required"):
        acquire_source(gigaword, tmp_path / "staging", mock_store, repo)

    # No acquisition run should be registered for a blocked source
    run_id = "acq-" + stable_id(gigaword.key, gigaword.release, gigaword.config_hash)
    assert repo.get_acquisition_run(run_id) is None
