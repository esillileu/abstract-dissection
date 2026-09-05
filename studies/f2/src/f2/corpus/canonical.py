"""Deterministic acquisition and record-preserving corpus transformations."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import html
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class SerialDownloader:
    """HTTP downloader with Range resume, bounded retry, and integrity checks."""

    def __init__(
        self, retries: int = 4, backoff_seconds: float = 1.0, timeout: float = 60.0
    ) -> None:
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout

    def download(
        self,
        url: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
        expected_length: int | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        for attempt in range(self.retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "abstract-dissection-f2/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=self.timeout
                ) as response:
                    status = getattr(response, "status", 200)
                    if offset and status != 206:
                        partial.unlink()
                        offset = 0
                    mode = "ab" if offset else "wb"
                    with partial.open(mode) as output:
                        while chunk := response.read(CHUNK_SIZE):
                            output.write(chunk)
                size = partial.stat().st_size
                if expected_length is not None and size != expected_length:
                    raise ValueError(
                        f"length mismatch: expected {expected_length}, observed {size}"
                    )
                observed = sha256_file(partial)
                if expected_sha256 is not None and observed != expected_sha256.lower():
                    raise ValueError(
                        f"SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
                    )
                partial.replace(destination)
                return destination
            except (OSError, urllib.error.URLError, ValueError):
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return not path.is_absolute() and ".." not in path.parts and member.isfile()


def iter_tar_records(path: Path, source: str) -> Iterator[tuple[str, bytes]]:
    """Yield only source-approved regular files, without extracting the archive."""
    with tarfile.open(path, "r:*") as archive:
        members = sorted(archive.getmembers(), key=lambda member: member.name)
        for member in members:
            if not _safe_member(member):
                if (
                    PurePosixPath(member.name).is_absolute()
                    or ".." in PurePosixPath(member.name).parts
                ):
                    raise ValueError(f"unsafe archive member: {member.name}")
                continue
            parts = PurePosixPath(member.name).parts
            accepted = (
                source == "lm1b" and "training-monolingual.tokenized.shuffled" in parts
            )
            accepted = accepted or (
                source == "umbc"
                and len(parts) >= 2
                and parts[-2] == "webbase_all"
                and parts[-1].endswith(".txt")
            )
            if not accepted:
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                yield member.name, extracted.read()


_NORMALIZE_REPLACEMENTS = (
    ("\u2019", "'"),
    ("\u2032", "'"),
    ("''", " "),
    ("'", " ' "),
    ("“", '"'),
    ("”", '"'),
    ('"', ' " '),
    (".", " . "),
    ("<br />", " "),
    (", ", " , "),
    ("(", " ( "),
    (")", " ) "),
    ("!", " ! "),
    ("?", " ? "),
    (";", " "),
    (":", " "),
    ("-", " - "),
    ("=", " "),
    ("*", " "),
    ("|", " "),
    ("«", " "),
)


def normalize_text(text: str) -> str:
    """Port of demo-train-big-model-v1.sh normalize_text under the C locale."""
    text = text.lower()
    for old, new in _NORMALIZE_REPLACEMENTS:
        text = text.replace(old, new)
    return text.translate(str.maketrans({str(number): " " for number in range(10)}))


def iter_text_lines(payload: bytes, compression: str | None = None) -> Iterator[str]:
    if compression == "gzip":
        payload = gzip.decompress(payload)
    yield from payload.decode("utf-8", errors="replace").splitlines()


def extract_gigaword_documents(text: str) -> Iterator[str]:
    for match in re.finditer(
        r"<DOC\b[^>]*>(.*?)</DOC>", text, flags=re.IGNORECASE | re.DOTALL
    ):
        body = match.group(1)
        paragraphs = re.findall(
            r"<P\b[^>]*>(.*?)</P>", body, flags=re.IGNORECASE | re.DOTALL
        )
        if paragraphs:
            clean = "\n".join(re.sub(r"<[^>]+>", " ", item) for item in paragraphs)
            clean = html.unescape(clean).strip()
            if clean:
                yield clean


def extract_wikipedia_records(xml_text: str) -> Iterator[str]:
    """Historical extractor semantics, scoped to page text and redirect exclusion."""
    for page in re.findall(
        r"<page\b.*?</page>", xml_text, flags=re.IGNORECASE | re.DOTALL
    ):
        if re.search(r"<redirect\b|#redirect", page, flags=re.IGNORECASE):
            continue
        match = re.search(
            r"<text\b[^>]*>(.*?)</text>", page, flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            continue
        value = _clean_wikipedia_text(match.group(1))
        if value:
            yield value


def _clean_wikipedia_text(value: str) -> str | None:
    value = html.unescape(value)
    value = re.sub(r"<ref[^<]*</ref>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]*>", "", value)
    value = re.sub(r"\[https?://[^\] ]*", "[", value)
    value = re.sub(r"\|(?:thumb|left|right|\d+px)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\[\[image:[^\[\]]*\|", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\[\[category:([^|\]]*)[^]]*\]\]", r"[[\1]]", value, flags=re.IGNORECASE
    )
    value = re.sub(r"\[\[[a-z-]*:[^\]]*\]\]", "", value)
    value = re.sub(r"\[\[[^|\]]*\|", "[[", value)
    value = re.sub(r"{{[^}]*}}|{[^}]*}|\[|\]", "", value)
    value = re.sub(r"&[^;]*;", " ", value)
    value = normalize_text(value)
    return value.strip() if len(value.split()) > 1 else None


@dataclass(frozen=True)
class ShardInfo:
    index: int
    path: str
    physical_sha256: str
    logical_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int
    word_count: int
    record_count: int
    source_span: dict[str, object]


class DeterministicSharder:
    def __init__(self, output_dir: Path, target_words: int = 10_000_000) -> None:
        self.output_dir = output_dir
        self.target_words = target_words

    def write(
        self, records: Iterable[tuple[str, str]], *, source: str
    ) -> list[ShardInfo]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shards: list[ShardInfo] = []
        batch: list[tuple[str, str]] = []
        words = 0

        def flush() -> None:
            nonlocal batch, words
            if not batch:
                return
            index = len(shards)
            raw = b"".join(
                text.rstrip("\n").encode("utf-8") + b"\n" for _, text in batch
            )
            raw_path = self.output_dir / f"shard-{index:05d}.txt"
            compressed_path = raw_path.with_suffix(".txt.zst")
            raw_path.write_bytes(raw)
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            subprocess.run(
                [
                    "zstd",
                    "-q",
                    "-f",
                    "-T1",
                    "-19",
                    "--no-progress",
                    raw_path.as_posix(),
                    "-o",
                    compressed_path.as_posix(),
                ],
                check=True,
                env=env,
            )
            raw_path.unlink()
            shards.append(
                ShardInfo(
                    index,
                    compressed_path.name,
                    sha256_file(compressed_path),
                    hashlib.sha256(raw).hexdigest(),
                    compressed_path.stat().st_size,
                    len(raw),
                    words,
                    len(batch),
                    {"source": source, "first": batch[0][0], "last": batch[-1][0]},
                )
            )
            batch, words = [], 0

        for record_id, text in records:
            count = len(text.split())
            if batch and words + count > self.target_words:
                flush()
            batch.append((record_id, text))
            words += count
            if count > self.target_words:
                flush()
        flush()
        manifest = {
            "schema_version": 1,
            "source": source,
            "target_words": self.target_words,
            "shards": [asdict(shard) for shard in shards],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return shards


def open_source_records(source: str, paths: list[Path]) -> Iterator[tuple[str, str]]:
    for path in sorted(paths, key=lambda item: item.name):
        if source in {"lm1b", "umbc"}:
            for name, payload in iter_tar_records(path, source):
                for number, line in enumerate(iter_text_lines(payload)):
                    yield f"{name}:{number}", line
        elif source == "wmt":
            for number, line in enumerate(iter_text_lines(path.read_bytes(), "gzip")):
                yield f"{path.name}:{number}", line
        elif source == "gigaword":
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
                for number, document in enumerate(
                    extract_gigaword_documents(stream.read())
                ):
                    yield f"{path.name}:{number}", document
        elif source == "wikipedia":
            with bz2.open(path, "rb") as stream:
                number = 0
                for _, element in ET.iterparse(stream, events=("end",)):
                    if element.tag.rsplit("}", 1)[-1] != "page":
                        continue
                    redirect = next(
                        (
                            child
                            for child in element
                            if child.tag.rsplit("}", 1)[-1] == "redirect"
                        ),
                        None,
                    )
                    text_node = next(
                        (
                            node
                            for node in element.iter()
                            if node.tag.rsplit("}", 1)[-1] == "text"
                        ),
                        None,
                    )
                    raw = text_node.text if text_node is not None else None
                    if (
                        redirect is None
                        and raw
                        and not re.match(r"\s*#redirect", raw, flags=re.IGNORECASE)
                    ):
                        document = _clean_wikipedia_text(raw)
                        if document:
                            yield f"{path.name}:{number}", document
                            number += 1
                    element.clear()


__all__ = [
    "DeterministicSharder",
    "SerialDownloader",
    "extract_gigaword_documents",
    "extract_wikipedia_records",
    "iter_tar_records",
    "normalize_text",
    "open_source_records",
    "sha256_file",
]
