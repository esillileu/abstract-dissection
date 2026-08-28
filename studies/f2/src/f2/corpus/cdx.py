"""Common Crawl CDX cluster index reading, SURT mapping, and record parsing."""

from __future__ import annotations

import bisect
import gzip
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CDXBlockLocator:
    surt_key: str
    timestamp: str
    filename: str
    offset: int
    length: int
    block_index: int


@dataclass(frozen=True)
class CDXRecord:
    url: str
    timestamp: str
    status: str
    mime: str
    digest: str
    filename: str
    offset: int
    length: int

    @classmethod
    def from_cdx_line(cls, line: str) -> CDXRecord | None:
        """Parse a single line from a CDX block JSON format."""
        parts = line.strip().split(" ", 2)
        if len(parts) < 3:
            return None
        _surt, ts, raw_json = parts[0], parts[1], parts[2]
        try:
            payload = json.loads(raw_json)
        except Exception:
            return None

        return cls(
            url=payload.get("url", ""),
            timestamp=ts,
            status=payload.get("status", ""),
            mime=payload.get("mime", ""),
            digest=payload.get("digest", ""),
            filename=payload.get("filename", ""),
            offset=int(payload.get("offset", 0)),
            length=int(payload.get("length", 0)),
        )


def domain_to_surt_prefix(domain: str) -> str:
    """Convert domain to SURT prefix. Example: 'reuters.com' -> 'com,reuters)'."""
    clean = domain.strip().lower()
    if clean.startswith("www."):
        clean = clean[4:]
    parts = clean.split(".")
    reversed_parts = reversed([p for p in parts if p])
    return ",".join(reversed_parts) + ")"


def url_to_surt(url: str) -> str:
    """Convert a full URL to SURT form."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.split(":")[0].lower()
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    reversed_host = ",".join(reversed([p for p in parts if p]))
    return f"{reversed_host}){path}"


class CDXIndexReader:
    """Reads Common Crawl CDX cluster.idx files and parses block records."""

    def __init__(self, cluster_idx_entries: list[CDXBlockLocator]) -> None:
        self.entries = cluster_idx_entries
        self._surt_keys = [e.surt_key for e in self.entries]

    @classmethod
    def from_text(cls, text: str) -> CDXIndexReader:
        """Parse cluster.idx content into an indexed reader."""
        entries: list[CDXBlockLocator] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            key_ts, filename, offset, length, block_idx = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
            )
            key_parts = key_ts.rsplit(" ", 1)
            surt_key = key_parts[0]
            ts = key_parts[1] if len(key_parts) > 1 else ""
            entries.append(
                CDXBlockLocator(
                    surt_key=surt_key,
                    timestamp=ts,
                    filename=filename,
                    offset=int(offset),
                    length=int(length),
                    block_index=int(block_idx),
                )
            )
        return cls(entries)

    @classmethod
    def from_file(cls, path: Path) -> CDXIndexReader:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return cls.from_text(text)

    def total_blocks(self) -> int:
        return len(self.entries)

    def find_block_index_for_surt(self, surt: str) -> int:
        """Locate the CDX block index that contains or immediately precedes a SURT key."""
        target = surt if surt.endswith("/") else surt + "/"
        idx = bisect.bisect_right(self._surt_keys, target)
        return max(0, idx - 1)

    def find_blocks_for_prefix(self, surt_prefix: str) -> list[CDXBlockLocator]:
        """Return all block locators that may contain records matching the SURT prefix."""
        start_idx = self.find_block_index_for_surt(surt_prefix)
        matched: list[CDXBlockLocator] = []
        for i in range(start_idx, len(self.entries)):
            entry = self.entries[i]
            matched.append(entry)
            # If the next block's start key no longer matches prefix and is strictly greater
            if (
                not entry.surt_key.startswith(surt_prefix)
                and entry.surt_key > surt_prefix
            ):
                break
        return matched

    @staticmethod
    def parse_block_lines(block_bytes: bytes) -> list[str]:
        """Decompress a GZIP CDX block slice and return individual text lines."""
        decompressed = gzip.decompress(block_bytes)
        return [
            line
            for line in decompressed.decode("utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

    @classmethod
    def parse_block_records(cls, block_bytes: bytes) -> list[CDXRecord]:
        """Decompress block and return parsed CDXRecord instances."""
        lines = cls.parse_block_lines(block_bytes)
        records: list[CDXRecord] = []
        for line in lines:
            rec = CDXRecord.from_cdx_line(line)
            if rec is not None:
                records.append(rec)
        return records


__all__ = [
    "CDXBlockLocator",
    "CDXIndexReader",
    "CDXRecord",
    "domain_to_surt_prefix",
    "url_to_surt",
]
