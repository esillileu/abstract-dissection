"""Pytest configuration and shared fixtures for F2 study tests."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_arc_bytes() -> bytes:
    """Constructs a valid compressed ARC record containing standard news HTML."""
    html_content = (
        "<html><head><title>Global Markets Rally as Economic Data Beats Estimates</title></head>"
        "<body>"
        "<article>"
        "<h1>Global Markets Rally as Economic Data Beats Estimates</h1>"
        "<p class='byline'>By John Smith, Reuters</p>"
        "<p class='date'>March 15, 2012</p>"
        "<p>NEW YORK — Global equity markets rallied sharply on Thursday following stronger-than-expected manufacturing data from major economies. "
        "The benchmark indices rose across Europe and North America as investor sentiment improved dramatically.</p>"
        '<p>"We are seeing robust demand across multiple sectors, which signals resilience in the underlying economy," said Jane Doe, chief market strategist at Global Investments. '
        "She added that corporate earnings have consistently outperformed analyst forecasts over the past two quarters.</p>"
        "<p>Central bank officials announced that interest rate policies would remain accommodative to support sustained long-term growth. "
        "According to the latest quarterly report, inflation pressures remained well within target thresholds, giving policymakers flexibility.</p>"
        "<p>Trading volumes were significantly higher than seasonal averages, with technology and financial shares leading the broad market advance. "
        "Analysts reported that sovereign bond yields stabilized following the economic release, reflecting reduced market volatility and growing optimism among institutional investors worldwide.</p>"
        "</article>"
        "</body></html>"
    )
    http_payload = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(html_content)}\r\n\r\n"
        f"{html_content}"
    ).encode()

    arc_header = f"http://www.reuters.com/article/2012/03/15/us-markets-idUSBRE82E0NV 192.168.1.1 20120315120000 text/html {len(http_payload)}\n".encode()
    raw_arc = arc_header + http_payload
    return gzip.compress(raw_arc)


@pytest.fixture
def sample_cluster_idx_text() -> str:
    return (
        "com,apple)/ 20120101000000\tcdx-00000.gz\t0\t200000\t1\n"
        "com,bbc)/ 20120101000000\tcdx-00000.gz\t200000\t200000\t2\n"
        "com,nytimes)/ 20120101000000\tcdx-00001.gz\t0\t200000\t3\n"
        "com,reuters)/ 20120101000000\tcdx-00001.gz\t200000\t200000\t4\n"
        "com,yahoo)/ 20120101000000\tcdx-00002.gz\t0\t200000\t5\n"
    )


@pytest.fixture(scope="session")
def f2_test_database_url():
    """Only an explicit test URL or a disposable PostgreSQL 18 instance."""
    import os
    import shutil
    import subprocess
    import time
    import uuid

    import psycopg

    from f2.catalog.db.migrations.runner import run_catalog_migrations
    from f2.corpus.db.migrations.runner import run_migrations

    url = os.environ.get("F2_TEST_DATABASE_URL")
    container = None
    engine = None
    try:
        if not url:
            engine = shutil.which("podman") or shutil.which("docker")
            if engine is None:
                pytest.fail(
                    "F2 DB verification requires F2_TEST_DATABASE_URL or Podman/Docker"
                )
            container = "f2-test-" + uuid.uuid4().hex
            result = subprocess.run(
                [
                    engine,
                    "run",
                    "--detach",
                    "--rm",
                    "--name",
                    container,
                    "-e",
                    "POSTGRES_PASSWORD=f2-test-only",
                    "-e",
                    "POSTGRES_DB=f2_test",
                    "-p",
                    "127.0.0.1::5432",
                    "docker.io/library/postgres:18",
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                pytest.fail("Cannot start disposable F2 PostgreSQL: " + result.stderr)
            port = (
                subprocess.check_output(
                    [engine, "port", container, "5432/tcp"], text=True
                )
                .strip()
                .rsplit(":", 1)[1]
            )
            url = f"postgresql://postgres:f2-test-only@127.0.0.1:{port}/f2_test"
        deadline = time.monotonic() + 40
        while True:
            try:
                with psycopg.connect(url, connect_timeout=2) as conn:
                    run_catalog_migrations(conn)
                    run_migrations(conn)
                break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    pytest.fail("Dedicated F2 test database did not become ready")
                time.sleep(0.25)
        yield url
    finally:
        if container and engine:
            subprocess.run(
                [engine, "rm", "--force", container], capture_output=True, timeout=30
            )


@pytest.fixture
def f2_test_database(f2_test_database_url, monkeypatch):
    # CLI opens its own connections. Patch the resolvers as well as the environment
    # so no test can fall through to a developer dotenv file.
    from f2.catalog.db.session import CatalogDatabaseConfig
    from f2.corpus.db.session import DatabaseConfig

    monkeypatch.setenv("F2_DATABASE_URL", f2_test_database_url)
    monkeypatch.setattr(
        CatalogDatabaseConfig,
        "from_environment",
        classmethod(lambda cls: cls(f2_test_database_url)),
    )
    monkeypatch.setattr(
        DatabaseConfig,
        "from_environment",
        classmethod(lambda cls: cls(f2_test_database_url)),
    )
    return f2_test_database_url
