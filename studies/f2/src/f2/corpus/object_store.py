"""Small AWS Signature-V4 S3 client used for SeaweedFS corpus objects."""

from __future__ import annotations

import hashlib
import hmac
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    root_uri: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"

    @classmethod
    def from_environment(cls, *, restricted: bool = False) -> S3Config:
        prefix = "F2_GIGAWORD_S3" if restricted else "F2_CORPUS_S3"
        values = {
            "endpoint": os.getenv(f"{prefix}_ENDPOINT", ""),
            "root_uri": os.getenv(f"{prefix}_ROOT", ""),
            "access_key": os.getenv(f"{prefix}_ACCESS_KEY", ""),
            "secret_key": os.getenv(f"{prefix}_SECRET_KEY", ""),
        }
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError(f"missing {prefix} configuration: {', '.join(missing)}")
        if not values["root_uri"].startswith("s3://"):
            raise ValueError(f"{prefix}_ROOT must be an s3:// URI")
        return cls(**values, region=os.getenv(f"{prefix}_REGION", "us-east-1"))


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
        self, method: str, bucket: str, path: str, payload_hash: str
    ) -> dict[str, str]:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        stamp, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        host = urllib.parse.urlparse(self.config.endpoint).netloc
        canonical_headers = (
            f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{stamp}\n"
        )
        signed = "host;x-amz-content-sha256;x-amz-date"
        canonical = f"{method}\n/{bucket}{path}\n\n{canonical_headers}\n{signed}\n{payload_hash}"
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
        payload_hash = _sha256(local_path)
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

    def probe(self) -> None:
        bucket, path, url = self._target(self.config.root_uri.rstrip("/") + "/")
        empty_hash = hashlib.sha256(b"").hexdigest()
        response = requests.head(
            url, headers=self._headers("HEAD", bucket, path, empty_hash), timeout=15
        )
        if response.status_code not in {200, 204, 404}:
            response.raise_for_status()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["S3Config", "S3ObjectStore"]
