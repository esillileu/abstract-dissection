"""Prepared analysis cache store for renderer-facing artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class PreparedAnalysisStore:
    """Materialize and replay the renderer-facing analysis inputs."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path, *, refresh: bool = False) -> None:
        self.root = root
        self.index_path = root / "prepared_analysis.json"
        self._entries: dict[str, object] = {}
        self._dirty = False
        if not refresh:
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == self.SCHEMA_VERSION:
                    self._entries = dict(payload["entries"])
            except (KeyError, OSError, TypeError, json.JSONDecodeError):
                pass

    def key(self, operation: str, payload: object) -> str:
        encoded = json.dumps(
            {"operation": operation, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"{operation}:{hashlib.sha256(encoded).hexdigest()}"

    def get(self, key: str) -> object | None:
        return self._entries.get(key)

    def contains(self, key: str) -> bool:
        return key in self._entries

    def put(self, key: str, value: object) -> None:
        self._entries[key] = value
        self._dirty = True

    def materialize_file(self, key: str, source: Path) -> Path:
        digest = key.rsplit(":", 1)[-1]
        target = self.root / "files" / digest / source.name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.put(key, {"path": str(target.relative_to(self.root))})
        return target

    def cached_file(self, key: str) -> Path | None:
        entry = self.get(key)
        if not isinstance(entry, dict) or "path" not in entry:
            return None
        path = self.root / str(entry["path"])
        return path if path.exists() else None

    def commit(self) -> None:
        if not self._dirty:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "entries": self._entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.index_path)
        self._dirty = False
