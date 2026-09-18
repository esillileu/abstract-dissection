"""AWS Signature-V4 client for S3-compatible object stores."""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

from repro_io.checksum import sha256_file


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    root_uri: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"


@dataclass(frozen=True)
class S3ObjectMetadata:
    uri: str
    byte_size: int
    etag: str | None = None


class S3ObjectStore:
    def __init__(self, config: S3Config) -> None:
        self.config = config

    def uri(self, key: str) -> str:
        return f"{self.config.root_uri.rstrip('/')}/{key.lstrip('/')}"

    def _target(self, uri: str) -> tuple[str, str, str]:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("object URI must use s3://bucket/key")
        path = "/" + urllib.parse.quote(parsed.path.lstrip("/"), safe="/-_.~")
        endpoint = self.config.endpoint.rstrip("/")
        return parsed.netloc, path, f"{endpoint}/{parsed.netloc}{path}"

    def _headers(
        self,
        method: str,
        bucket: str,
        path: str,
        payload_hash: str,
        *,
        canonical_query: str = "",
    ) -> dict[str, str]:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        host = urllib.parse.urlparse(self.config.endpoint).netloc
        canonical_headers = (
            f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{stamp}\n"
        )
        signed = "host;x-amz-content-sha256;x-amz-date"
        canonical = f"{method}\n/{bucket}{path}\n{canonical_query}\n{canonical_headers}\n{signed}\n{payload_hash}"
        scope = f"{date}/{self.config.region}/s3/aws4_request"
        string_to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"

        def sign(key: bytes, value: str) -> bytes:
            return hmac.new(key, value.encode(), hashlib.sha256).digest()

        key = sign(("AWS4" + self.config.secret_key).encode(), date)
        key = sign(key, self.config.region)
        key = sign(key, "s3")
        key = sign(key, "aws4_request")
        signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        authorization = f"AWS4-HMAC-SHA256 Credential={self.config.access_key}/{scope}, SignedHeaders={signed}, Signature={signature}"
        return {
            "Host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": stamp,
            "Authorization": authorization,
        }

    def put_file(self, local_path: Path, uri: str) -> None:
        bucket, path, url = self._target(uri)
        payload_hash = sha256_file(local_path)
        with local_path.open("rb") as payload:
            response = requests.put(
                url,
                data=payload,
                headers=self._headers("PUT", bucket, path, payload_hash),
                timeout=300,
            )
        response.raise_for_status()

    def sha256(self, uri: str) -> str:
        bucket, path, url = self._target(uri)
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = requests.get(
            url,
            headers=self._headers("GET", bucket, path, empty_hash),
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        digest = hashlib.sha256()
        for chunk in response.iter_content(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()

    def get_file(self, uri: str, local_path: Path) -> Path:
        bucket, path, url = self._target(uri)
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = requests.get(
            url,
            headers=self._headers("GET", bucket, path, empty_hash),
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        partial = local_path.with_name(local_path.name + ".part")
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
        partial.replace(local_path)
        return local_path

    def head(self, uri: str) -> S3ObjectMetadata:
        """Return object size and ETag without downloading object bytes."""
        bucket, path, url = self._target(uri)
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = requests.head(
            url,
            headers=self._headers("HEAD", bucket, path, empty_hash),
            timeout=30,
        )
        response.raise_for_status()
        return S3ObjectMetadata(
            uri=uri,
            byte_size=int(response.headers["Content-Length"]),
            etag=response.headers.get("ETag"),
        )

    def list(self, prefix_uri: str) -> list[S3ObjectMetadata]:
        """List every object below an S3 URI prefix using ListObjectsV2."""
        bucket, path, _ = self._target(prefix_uri)
        prefix = path.lstrip("/")
        endpoint = self.config.endpoint.rstrip("/")
        result: list[S3ObjectMetadata] = []
        continuation: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if continuation:
                query["continuation-token"] = continuation
            canonical_query = urllib.parse.urlencode(
                sorted(query.items()), quote_via=urllib.parse.quote
            )
            url = f"{endpoint}/{bucket}/?{canonical_query}"
            empty_hash = hashlib.sha256(b"").hexdigest()
            response = requests.get(
                url,
                headers=self._headers(
                    "GET", bucket, "/", empty_hash, canonical_query=canonical_query
                ),
                timeout=30,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespace = root.tag.removesuffix("ListBucketResult")
            for item in root.findall(f"{namespace}Contents"):
                key = item.findtext(f"{namespace}Key")
                size = item.findtext(f"{namespace}Size")
                if key is not None and size is not None:
                    result.append(
                        S3ObjectMetadata(
                            f"s3://{bucket}/{key}",
                            int(size),
                            item.findtext(f"{namespace}ETag"),
                        )
                    )
            continuation = root.findtext(f"{namespace}NextContinuationToken")
            if not continuation:
                return result

    def probe(self) -> None:
        bucket, path, url = self._target(self.config.root_uri.rstrip("/") + "/")
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = requests.head(
            url, headers=self._headers("HEAD", bucket, path, empty_hash), timeout=15
        )
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()


__all__ = ["S3Config", "S3ObjectMetadata", "S3ObjectStore"]
