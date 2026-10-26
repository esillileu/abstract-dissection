"""Progress and artifact discovery for local experiment execution roots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class RunLocation:
    """Discovered state of an execution path."""

    status: str  # "final_found" | "checkpoint_found" | "not_found"
    root: Path
    final_result_path: Path | None = None
    checkpoint_path: Path | None = None
    epoch: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.status == "final_found"

    @property
    def is_checkpoint(self) -> bool:
        return self.status == "checkpoint_found"

    @property
    def not_found(self) -> bool:
        return self.status == "not_found"


class RunLocator(Protocol):
    """Protocol for discovering experiment progress in local storage."""

    def locate(self, run_root: Path) -> RunLocation: ...


class PathRunLocator:
    """Default filesystem run locator inspecting defined execution paths.

    Discovers existing final artifacts and checkpoints using established
    checkpoint and result serialization contracts.
    Does not access MLflow, create state databases, or determine execution policies.
    """

    def locate(self, run_root: Path) -> RunLocation:
        root = Path(run_root)
        if not root.is_dir():
            return RunLocation(status="not_found", root=root)

        # 1. Check for completed final result
        result_json = root / "result.json"
        if result_json.is_file():
            try:
                data = json.loads(result_json.read_text(encoding="utf-8"))
                if data.get("complete") is True:
                    artifacts = data.get("artifacts")
                    if isinstance(artifacts, dict) and artifacts:
                        all_exist = all(
                            (root / str(p)).exists()
                            for p in artifacts.values()
                            if isinstance(p, (str, Path))
                        )
                        if all_exist:
                            return RunLocation(
                                status="final_found",
                                root=root,
                                final_result_path=result_json,
                                metadata=data,
                            )
            except (json.JSONDecodeError, OSError):
                pass

        # 2. Check for intermediate checkpoint
        checkpoint_dir = root / "checkpoints"
        latest_pointer = checkpoint_dir / "latest.json"
        if latest_pointer.is_file():
            try:
                pointer_data = json.loads(latest_pointer.read_text(encoding="utf-8"))
                rel_path = pointer_data.get("path")
                if rel_path:
                    gen_path = checkpoint_dir / rel_path
                    if gen_path.is_dir() and (gen_path / "manifest.json").is_file():
                        manifest = json.loads(
                            (gen_path / "manifest.json").read_text(encoding="utf-8")
                        )
                        epoch = int(
                            manifest.get(
                                "completed_epochs",
                                pointer_data.get("generation", 0),
                            )
                        )
                        if epoch == 0 and "epoch-" in gen_path.name:
                            try:
                                epoch = int(gen_path.name.split("epoch-")[-1])
                            except ValueError:
                                pass
                        return RunLocation(
                            status="checkpoint_found",
                            root=root,
                            checkpoint_path=gen_path,
                            epoch=epoch,
                            metadata=manifest,
                        )
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass

        # Also check for direct generation directories if latest.json is missing but generations exist
        gen_dir = checkpoint_dir / "generations"
        if gen_dir.is_dir():
            candidates = sorted(
                [
                    d
                    for d in gen_dir.iterdir()
                    if d.is_dir()
                    and (d / "manifest.json").is_file()
                    and d.name != "final"
                ],
                key=lambda p: p.name,
            )
            if candidates:
                latest_gen = candidates[-1]
                try:
                    manifest = json.loads(
                        (latest_gen / "manifest.json").read_text(encoding="utf-8")
                    )
                    epoch = int(manifest.get("completed_epochs", 0))
                    if epoch == 0 and "epoch-" in latest_gen.name:
                        epoch = int(latest_gen.name.split("epoch-")[-1])
                except (json.JSONDecodeError, OSError, ValueError):
                    epoch = 0
                return RunLocation(
                    status="checkpoint_found",
                    root=root,
                    checkpoint_path=latest_gen,
                    epoch=epoch,
                )

        return RunLocation(status="not_found", root=root)


__all__ = ["PathRunLocator", "RunLocation", "RunLocator"]
