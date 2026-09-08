"""Export operational database or DuckDB tabular records to Parquet and JSONL formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ExportableRepository(Protocol):
    def export_provenance_to_parquet(self, run_id: str, output_path: Path) -> None: ...
    def export_provenance_to_jsonl(self, run_id: str, output_path: Path) -> None: ...


class ProvenanceExporter:
    """Exports operational database state to long-term immutable Parquet and streaming JSONL."""

    def __init__(self, repo: Any) -> None:
        self.repo = repo

    def export(self, run_id: str, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "provenance.parquet"
        jsonl_path = output_dir / "provenance.jsonl"

        self.repo.export_provenance_to_parquet(run_id, parquet_path)
        self.repo.export_provenance_to_jsonl(run_id, jsonl_path)

        return {"parquet": parquet_path, "jsonl": jsonl_path}


__all__ = [
    "ExportableRepository",
    "ProvenanceExporter",
]
