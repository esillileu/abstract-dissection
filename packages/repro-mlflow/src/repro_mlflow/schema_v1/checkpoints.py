"""Checkpoint manifest generation and synchronization for schema-v1."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _select_final_checkpoint(
    root: Path,
    *,
    save_final: bool,
) -> tuple[Path | None, str | None]:
    pointer = root / "latest.json"
    if not save_final or not pointer.exists():
        return None, None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("only checkpoint schema version 2 is supported")
    path = root / str(payload["path"])
    return path, str(payload["sha256"])


def _checkpoint_role_manifest(root: Path, role: str) -> dict[str, object] | None:
    pointer = root / f"{role}.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    path = root / str(payload["path"])
    return {
        "path": str(path.resolve()),
        "epoch": int(payload["epoch"]),
        "update": int(payload["update"]),
        "digest": str(payload["sha256"]),
    }


def _normalize_checkpoints_csv(
    path: Path,
    roles: dict[str, dict[str, object] | None],
) -> None:
    """Keep the raw checkpoint index aligned with the retained semantic roles."""
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    retained: list[dict[str, str]] = []
    for role, accepted_kinds in (
        ("latest", {"latest", "final"}),
        ("best", {"best", "selected"}),
    ):
        manifest = roles.get(role)
        if manifest is None:
            continue
        candidates = [row for row in rows if row.get("kind") in accepted_kinds]
        if candidates:
            row = dict(candidates[-1])
        else:
            row = {key: "" for key in fieldnames}
        row["kind"] = "selected" if role == "best" else "latest"
        row["path"] = str(manifest["path"])
        if "sha256" in fieldnames:
            row["sha256"] = str(manifest["digest"])
        retained.append(row)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)


def _periodic_checkpoint_manifests(root: Path) -> list[dict[str, object]]:
    generations = root / "generations"
    if not generations.exists():
        return []
    output = []
    for path in sorted(generations.glob("periodic-*")):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        output.append(
            {
                "path": str(path.resolve()),
                "epoch": int(manifest["epoch"]),
                "update": int(manifest["global_step"]),
            }
        )
    return output
