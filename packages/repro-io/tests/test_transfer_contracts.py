"""Deterministic wire/byte contracts for extracted transfer mechanisms."""

import gzip
import hashlib
import io
import urllib.error
from unittest.mock import MagicMock

import pytest
from repro_io.archive import ARCParser
from repro_io.commoncrawl.cdx import CDXIndexReader, domain_to_surt_prefix, url_to_surt
from repro_io.http import RangeFetcher, SerialDownloader
from repro_io.s3 import S3Config, S3ObjectStore


def response(payload, status=200):
    stream = io.BytesIO(payload)
    stream.status = status
    return stream


@pytest.mark.parametrize("status,expected", [(206, b"abcdef"), (200, b"def")])
def test_resume_and_server_ignoring_range(tmp_path, monkeypatch, status, expected):
    destination = tmp_path / "payload"
    destination.with_suffix(".part").write_bytes(b"abc")
    requests = []

    def open_url(request, **kwargs):
        requests.append(request)
        return response(b"def", status)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    SerialDownloader(retries=0, user_agent="fixture-agent").download(
        "https://example.test/object",
        destination,
        expected_length=len(expected),
        expected_sha256=hashlib.sha256(expected).hexdigest(),
    )
    assert destination.read_bytes() == expected
    assert requests[0].get_header("Range") == "bytes=3-"
    assert requests[0].get_header("User-agent") == "fixture-agent"
    assert not destination.with_suffix(".part").exists()


def test_downloader_retry_keeps_partial(tmp_path, monkeypatch):
    destination = tmp_path / "payload"
    requests = []

    def open_url(request, **kwargs):
        requests.append(request)
        return response(
            b"abc" if len(requests) == 1 else b"def", 200 if len(requests) == 1 else 206
        )

    sleeps = []
    monkeypatch.setattr("urllib.request.urlopen", open_url)
    monkeypatch.setattr("repro_io.http.download.time.sleep", sleeps.append)
    SerialDownloader(retries=1, backoff_seconds=2).download(
        "https://example.test/object", destination, expected_length=6
    )
    assert destination.read_bytes() == b"abcdef"
    assert sleeps == [2]
    assert requests[1].get_header("Range") == "bytes=3-"


@pytest.mark.parametrize("code,attempts", [(503, 3), (404, 1)])
def test_range_retry_and_error_contract(monkeypatch, code, attempts):
    calls = []

    def open_url(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, code, "fixture", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    monkeypatch.setattr("repro_io.http.fetcher.time.sleep", lambda _: None)
    result = RangeFetcher("https://example.test", max_retries=3).fetch_range(
        "object", 2, 4
    )
    assert len(calls) == attempts
    assert calls[0].get_header("Range") == "bytes=2-5"
    assert result.status_code == 500 and result.data == b""
    assert result.downloaded_bytes == 0
    assert result.error_message == f"HTTP {code}: fixture"


def test_range_bytes_and_injected_headers(monkeypatch):
    def open_url(request, **kwargs):
        assert request.full_url == "https://example.test/base/object"
        assert request.get_header("User-agent") == "fixture-agent"
        assert request.get_header("Accept-encoding") == "identity"
        return response(b"\x00\xff", 206)

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    result = RangeFetcher(
        "https://example.test/base/", user_agent="fixture-agent"
    ).fetch_range("/object", 0, 2)
    assert (result.status_code, result.data, result.downloaded_bytes) == (
        206,
        b"\x00\xff",
        2,
    )


def test_s3_signature_and_object_bytes(tmp_path, monkeypatch):
    import datetime

    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2020, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(datetime, "datetime", FixedDateTime)
    store = S3ObjectStore(
        S3Config(
            "https://objects.example.test",
            "s3://bucket/prefix",
            "example-access",
            "example-secret",
        )
    )
    digest = hashlib.sha256(b"\x00payload\xff").hexdigest()
    headers = store._headers("PUT", "bucket", "/prefix/a%20b", digest)
    assert headers["x-amz-date"] == "20200102T030405Z"
    assert (
        headers["Authorization"]
        == "AWS4-HMAC-SHA256 Credential=example-access/20200102/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=7ef9fb989e67e6689d171246963a39c2afcb56f0cedc70a73fbb10e5c9e65a78"
    )
    source = tmp_path / "input"
    source.write_bytes(b"\x00payload\xff")

    def put(url, *, data, headers, timeout):
        assert url == "https://objects.example.test/bucket/prefix/a%20b"
        assert data.read() == source.read_bytes()
        assert headers["x-amz-content-sha256"] == digest
        return MagicMock()

    monkeypatch.setattr("repro_io.s3.requests.put", put)
    store.put_file(source, store.uri("a b"))

    def get(url, **kwargs):
        result = MagicMock()
        result.iter_content.return_value = [b"\x00pay", b"load\xff"]
        return result

    monkeypatch.setattr("repro_io.s3.requests.get", get)
    assert store.sha256(store.uri("a b")) == digest
    target = store.get_file(store.uri("a b"), tmp_path / "download")
    assert target.read_bytes() == source.read_bytes()


def test_arc_fallback_and_cdx_surt(monkeypatch):
    monkeypatch.setattr(
        "repro_io.archive.ArchiveIterator",
        MagicMock(side_effect=ValueError("fallback")),
    )
    payload = b"http://example.test/a 0.0.0.0 20120101000000 text/html 8\nHTTP/1.1 200 OK\r\n\r\nhi\xff"
    record = ARCParser.parse_arc_bytes(gzip.compress(payload))
    assert (record.url, record.http_status, record.content_type, record.html_body) == (
        "http://example.test/a",
        200,
        "text/html",
        "hi\ufffd",
    )
    assert ARCParser.parse_arc_bytes(b"invalid") is None
    assert domain_to_surt_prefix("www.example.com") == "com,example)"
    assert url_to_surt("https://example.com/a") == "com,example)/a"
    records = CDXIndexReader.parse_block_records(
        gzip.compress(
            b'com,example)/a 20120101 {"url":"https://example.com/a","offset":"12","length":"34","filename":"x.gz","status":"200"}\ninvalid\n'
        )
    )
    assert len(records) == 1
    assert (records[0].offset, records[0].length) == (12, 34)
