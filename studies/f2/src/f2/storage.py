"""Corpus persistence, clean text shard materialization, and provenance dataset export."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .db.repository import CorpusStateRepository


class CleanTextWriter:
    """Materializes clean, sentence/paragraph-intact text into sharded files."""

    def __init__(self, output_dir: Path, max_words_per_shard: int = 10_000_000) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_words_per_shard = max_words_per_shard
        self.current_shard_idx = 0
        self.current_shard_words = 0
        self._current_file = None
        self._current_shard_path: Path | None = None

    def _get_file(self):
        if (
            self._current_file is None
            or self.current_shard_words >= self.max_words_per_shard
        ):
            if self._current_file is not None:
                self._current_file.close()
            shard_name = f"shard_{self.current_shard_idx:05d}.txt"
            self._current_shard_path = self.output_dir / shard_name
            self._current_file = self._current_shard_path.open("a", encoding="utf-8")
            self.current_shard_idx += 1
            self.current_shard_words = 0
        return self._current_file

    def write_document(
        self, text: str, word_count: int, doc_url: str
    ) -> tuple[str, str]:
        """Writes document text and returns (clean_text_sha256, shard_relative_path)."""
        f = self._get_file()
        header = f'<DOC url="{doc_url}" words="{word_count}">\n'
        footer = "\n</DOC>\n\n"
        f.write(header + text.strip() + footer)
        self.current_shard_words += word_count

        sha256_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        shard_path_str = (
            self._current_shard_path.as_posix() if self._current_shard_path else ""
        )
        return sha256_hash, shard_path_str

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None


class ProvenanceExporter:
    """Exports operational database state to long-term immutable Parquet and streaming JSONL."""

    def __init__(self, repo: CorpusStateRepository) -> None:
        self.repo = repo

    def export(self, run_id: str, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "provenance.parquet"
        jsonl_path = output_dir / "provenance.jsonl"

        self.repo.export_provenance_to_parquet(run_id, parquet_path)
        self.repo.export_provenance_to_jsonl(run_id, jsonl_path)

        return {"parquet": parquet_path, "jsonl": jsonl_path}


__all__ = [
    "CleanTextWriter",
    "ProvenanceExporter",
]
