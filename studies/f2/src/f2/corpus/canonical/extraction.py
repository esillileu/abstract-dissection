"""Raw source archive extraction and record streaming."""

from __future__ import annotations

import bz2
import gzip
import html
import re
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path, PurePosixPath


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
    lines = [" ".join(line.split()) for line in value.splitlines()]
    value = "\n".join(line for line in lines if line)
    return value.strip() if len(value.split()) > 1 else None


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
    "extract_gigaword_documents",
    "extract_wikipedia_records",
    "iter_tar_records",
    "iter_text_lines",
    "open_source_records",
]
