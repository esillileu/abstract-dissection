from __future__ import annotations

import gzip
import io
from dataclasses import dataclass

from warcio.archiveiterator import ArchiveIterator


@dataclass(frozen=True)
class ExtractedARCRecord:
    url: str
    http_status: int
    content_type: str
    html_body: str


class ARCParser:
    """Extracts HTTP headers and HTML payload from compressed ARC byte slices."""

    @staticmethod
    def parse_arc_bytes(compressed_bytes: bytes) -> ExtractedARCRecord | None:
        try:
            decompressed = gzip.decompress(compressed_bytes)
        except Exception:
            return None

        try:
            stream = io.BytesIO(decompressed)
            for record in ArchiveIterator(stream):
                if record.rec_type in {"response", "arc"}:
                    url = (
                        record.rec_headers.get_header("WARC-Target-URI")
                        or record.rec_headers.get_header("ARC-Target-URI")
                        or ""
                    )
                    http_status = int(
                        record.http_headers.get_statuscode()
                        if record.http_headers
                        else 200
                    )
                    content_type = (
                        record.http_headers.get_header("Content-Type")
                        if record.http_headers
                        else "text/html"
                    )
                    payload = record.content_stream().read()
                    html_text = payload.decode("utf-8", errors="replace")
                    return ExtractedARCRecord(
                        url=url,
                        http_status=http_status,
                        content_type=content_type,
                        html_body=html_text,
                    )
        except Exception:
            pass

        # Fallback for plain ARC decompression if ArchiveIterator fails
        return ARCParser._fallback_parse_raw(decompressed)

    @staticmethod
    def _fallback_parse_raw(decompressed: bytes) -> ExtractedARCRecord | None:
        try:
            # ARC header line: URL IP DATE MIME LENGTH
            header_end = decompressed.find(b"\n")
            if header_end == -1:
                return None
            header_line = (
                decompressed[:header_end].decode("utf-8", errors="ignore").strip()
            )
            parts = header_line.split(" ")
            url = parts[0] if parts else ""
            mime = parts[3] if len(parts) > 3 else "text/html"

            body_bytes = decompressed[header_end + 1 :]
            # Check for HTTP header separator \r\n\r\n or \n\n
            sep_idx = body_bytes.find(b"\r\n\r\n")
            if sep_idx != -1:
                html_bytes = body_bytes[sep_idx + 4 :]
            else:
                sep_idx2 = body_bytes.find(b"\n\n")
                html_bytes = (
                    body_bytes[sep_idx2 + 2 :] if sep_idx2 != -1 else body_bytes
                )

            return ExtractedARCRecord(
                url=url,
                http_status=200,
                content_type=mime,
                html_body=html_bytes.decode("utf-8", errors="replace"),
            )
        except Exception:
            return None
