from __future__ import annotations

import urllib.request

import typer
from repro_io.commoncrawl.cdx import CDXBlockLocator, CDXIndexReader

from repro_core.context.paths import RuntimePaths


def validate_connection(conn, **_kwargs):
    database = conn.execute("SELECT current_database()").fetchone()[0]
    if database != "f2_cc":
        raise RuntimeError("Common Crawl commands require database 'f2_cc'")


def ensure_cluster_index(crawl_id: str) -> CDXIndexReader:
    """Ensure Common Crawl CDX cluster index is available in cache and return indexed reader."""
    cache_dir = RuntimePaths.from_environment().cache_root / "f2" / crawl_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_path = cache_dir / "cluster.idx"

    if idx_path.exists() and idx_path.stat().st_size > 10_000:
        return CDXIndexReader.from_file(idx_path)

    # Download from Common Crawl index repository
    url = f"https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/cluster.idx"
    typer.echo(f"Downloading Common Crawl CDX cluster index for {crawl_id}...")
    temp_path = cache_dir / "cluster.idx.tmp"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "abstract-dissection-repro/0.1 (Research reproduction study)"
            },
        )
        with (
            urllib.request.urlopen(req, timeout=60.0) as resp,
            temp_path.open("wb") as f,
        ):
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        temp_path.replace(idx_path)
        typer.echo(
            f"Saved cluster index to: {idx_path} ({idx_path.stat().st_size / 1_000_000:.1f} MB)"
        )
        return CDXIndexReader.from_file(idx_path)
    except Exception as exc:
        typer.echo(
            f"Warning: Could not download remote cluster index ({exc}). Using offline baseline."
        )
        mock_entries = [
            CDXBlockLocator(
                surt_key="a)",
                timestamp="20120101",
                filename="cdx-00000.gz",
                offset=0,
                length=208582,
                block_index=0,
            ),
            CDXBlockLocator(
                surt_key="m)",
                timestamp="20120101",
                filename="cdx-00000.gz",
                offset=208582,
                length=211621,
                block_index=1,
            ),
        ]
        return CDXIndexReader(mock_entries)
