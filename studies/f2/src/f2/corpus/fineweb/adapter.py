"""Hugging Face FineWeb 2013 parquet streaming and source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .models import FineWebDocument

FINEWEB_DATASET: str = "HuggingFaceFW/fineweb"
FINEWEB_PINNED_REVISION: str = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB_DEFAULT_DUMP: str = "CC-MAIN-2013-20"
FINEWEB_FALLBACK_DUMP: str = "CC-MAIN-2013-48"
FINEWEB_TOTAL_PARQUET_FILES: dict[str, int] = {
    "CC-MAIN-2013-20": 205,
    "CC-MAIN-2013-48": 212,
}
_DUMP_FILES_PATH: Path = Path(__file__).parent / "dump_files.json"


def _load_dump_files() -> dict[str, list[str]]:
    if _DUMP_FILES_PATH.is_file():
        try:
            with _DUMP_FILES_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


class FineWebSourceAdapter:
    """Streams and maps FineWeb parquet files with deterministic document identity."""

    def __init__(
        self,
        dataset: str = FINEWEB_DATASET,
        revision: str = FINEWEB_PINNED_REVISION,
    ) -> None:
        self.dataset = dataset
        self.revision = revision

    def parquet_filename(self, index: int) -> str:
        """Deterministic parquet filename for a part index."""
        return f"{index // 50:03d}_{index % 50:05d}.parquet"

    def list_parquet_files(self, dump: str = FINEWEB_DEFAULT_DUMP) -> list[str]:
        """Ordered list of relative parquet paths for a given dump."""
        cached = _load_dump_files().get(dump)
        if cached:
            return list(cached)
        count = FINEWEB_TOTAL_PARQUET_FILES.get(dump, 205)
        return [f"data/{dump}/{self.parquet_filename(i)}" for i in range(count)]

    def resolve_url(self, relative_path: str) -> str:
        """Resolve a direct HTTP URL for a pinned FineWeb repository file."""
        return (
            f"https://huggingface.co/datasets/{self.dataset}/resolve/"
            f"{self.revision}/{relative_path.lstrip('/')}"
        )

    def iter_local_parquet(
        self,
        path: Path | str,
        *,
        dump: str = FINEWEB_DEFAULT_DUMP,
        limit: int | None = None,
        batch_size: int = 1000,
    ) -> Iterator[FineWebDocument]:
        """Read documents sequentially from a local parquet file using PyArrow."""
        path_str = str(path)
        pfile = pq.ParquetFile(path_str)
        seen = 0
        global_row_index = 0

        for batch in pfile.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            for row in rows:
                doc = self._row_to_document(
                    row,
                    dump=row.get("dump") or dump,
                    parquet_path=path_str,
                    row_index=global_row_index,
                )
                yield doc
                seen += 1
                global_row_index += 1
                if limit is not None and seen >= limit:
                    return

    def iter_remote_parquet(
        self,
        url: str,
        *,
        dump: str = FINEWEB_DEFAULT_DUMP,
        limit: int | None = None,
        chunk_size: int = 200,
    ) -> Iterator[FineWebDocument]:
        """Stream documents from a remote parquet URL using DuckDB httpfs range requests."""
        import duckdb

        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            query = """
                SELECT id, text, dump, url, date, file_path, language, language_score, token_count
                FROM read_parquet(?)
            """
            cursor = con.execute(query, [url])
            seen = 0
            global_row_index = 0
            while True:
                fetch_n = chunk_size if limit is None else min(chunk_size, limit - seen)
                if fetch_n <= 0:
                    break
                batch = cursor.fetchmany(fetch_n)
                if not batch:
                    break
                for row_tuple in batch:
                    row_dict = {
                        "id": row_tuple[0],
                        "text": row_tuple[1],
                        "dump": row_tuple[2] or dump,
                        "url": row_tuple[3] or "",
                        "date": row_tuple[4] or "",
                        "file_path": row_tuple[5],
                        "language": row_tuple[6] or "en",
                        "language_score": float(row_tuple[7] or 1.0),
                        "token_count": int(row_tuple[8] or 0),
                    }
                    doc = self._row_to_document(
                        row_dict,
                        dump=dump,
                        parquet_path=url,
                        row_index=global_row_index,
                    )
                    yield doc
                    seen += 1
                    global_row_index += 1
                    if limit is not None and seen >= limit:
                        return
        finally:
            con.close()

    def iter_arrow_table(
        self,
        table: pa.Table,
        *,
        dump: str = FINEWEB_DEFAULT_DUMP,
        parquet_path: str = "mock.parquet",
        limit: int | None = None,
    ) -> Iterator[FineWebDocument]:
        """Stream documents from an in-memory PyArrow table (hermetic testing)."""
        seen = 0
        global_row_index = 0
        for batch in table.to_batches():
            rows = batch.to_pylist()
            for row in rows:
                doc = self._row_to_document(
                    row,
                    dump=dump,
                    parquet_path=parquet_path,
                    row_index=global_row_index,
                )
                yield doc
                seen += 1
                global_row_index += 1
                if limit is not None and seen >= limit:
                    return

    def _row_to_document(
        self,
        row: dict[str, Any],
        dump: str,
        parquet_path: str,
        row_index: int,
    ) -> FineWebDocument:
        return FineWebDocument(
            fineweb_id=str(row.get("id", "")),
            text=str(row.get("text", "")),
            dump=str(row.get("dump") or dump),
            url=str(row.get("url", "")),
            date=str(row.get("date", "")),
            file_path=row.get("file_path"),
            language=str(row.get("language", "en")),
            language_score=float(row.get("language_score", 1.0)),
            token_count=int(row.get("token_count", 0)),
            parquet_path=parquet_path,
            row_index=row_index,
        )


__all__ = [
    "FINEWEB_DATASET",
    "FINEWEB_DEFAULT_DUMP",
    "FINEWEB_FALLBACK_DUMP",
    "FINEWEB_PINNED_REVISION",
    "FINEWEB_TOTAL_PARQUET_FILES",
    "FineWebSourceAdapter",
]
